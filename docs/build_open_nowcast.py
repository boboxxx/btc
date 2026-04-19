#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import time
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from pandas.tseries.offsets import BDay
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_nq_quant_model import (
    HEADERS,
    apply_wave_fib_overlay,
    archive_nq_contracts,
    build_archived_nq_main,
    build_roll_adjusted_nq_main,
    build_signal_sessions,
    estimate_cn_etf_beta,
    fetch_sina_cn_intraday,
    fetch_yahoo_chart,
    fetch_yahoo_daily,
    make_price_features,
    merge_asof_features,
    next_business_day,
    pick_feature_columns,
)


@dataclass
class NowcastArtifacts:
    name: str
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    pipeline: Pipeline
    feature_cols: list[str]
    decision_threshold: float


CN_TIMEZONE = ZoneInfo("Asia/Shanghai")
US_TIMEZONE = ZoneInfo("America/New_York")
STATE_STORE_COLUMNS = [
    "market",
    "symbol",
    "session_date",
    "model_state",
    "updated_at",
    "feature_bar_time",
    "signal_effective_price",
    "probability_long",
    "action",
    "market_phase",
]


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue


def fetch_cn_quote(symbol: str) -> dict[str, float | str | pd.Timestamp]:
    market_prefix = "0" if symbol.endswith(".SZ") else "1"
    code = symbol.split(".")[0]
    url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={market_prefix}.{code}&fields=f43,f44,f45,f46,f57,f58,f60,f169,f170"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        data = response.json()["data"]
        return {
            "symbol": symbol,
            "name": data["f58"],
            "last": data["f43"] / 1000.0,
            "high": data["f44"] / 1000.0,
            "low": data["f45"] / 1000.0,
            "open": data["f46"] / 1000.0,
            "prev_close": data["f60"] / 1000.0,
            "change": data["f169"] / 1000.0,
            "pct": data["f170"] / 100.0,
            "snapshot_time": pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None),
        }
    except Exception:  # noqa: BLE001
        bars = fetch_sina_cn_intraday(symbol, interval_minutes=5, datalen=500)
        bars = bars.dropna(subset=["open", "close"]).sort_values("datetime").reset_index(drop=True)
        latest = bars.iloc[-1]
        same_day = bars[bars["date"].eq(latest["date"])]
        previous = bars[bars["date"] < latest["date"]]
        prev_close = float(previous.iloc[-1]["close"]) if not previous.empty else float(latest["close"])
        change = float(latest["close"] - prev_close)
        pct = change / prev_close if prev_close else 0.0
        return {
            "symbol": symbol,
            "name": symbol,
            "last": float(latest["close"]),
            "high": float(same_day["high"].max()),
            "low": float(same_day["low"].min()),
            "open": float(same_day.iloc[0]["open"]),
            "prev_close": prev_close,
            "change": change,
            "pct": pct,
            "snapshot_time": pd.Timestamp(latest["datetime"]),
        }


def _coerce_local_now(now: pd.Timestamp | None, tz: ZoneInfo) -> pd.Timestamp:
    if now is None:
        return pd.Timestamp.now(tz=tz).tz_localize(None)
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        return ts.tz_localize(tz).tz_localize(None)
    return ts.tz_convert(tz).tz_localize(None)


def resolve_cn_session(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, str]:
    local_now = _coerce_local_now(now, CN_TIMEZONE)
    current_time = local_now.time()
    if local_now.weekday() >= 5:
        return next_business_day(local_now), "weekend"
    if current_time < time(9, 30):
        return local_now.normalize(), "pre_open"
    if current_time <= time(11, 30):
        return local_now.normalize(), "live"
    if current_time < time(13, 0):
        return local_now.normalize(), "midday_break"
    if current_time <= time(15, 0):
        return local_now.normalize(), "live"
    return next_business_day(local_now), "post_close"


def resolve_us_session(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, str]:
    local_now = _coerce_local_now(now, US_TIMEZONE)
    current_time = local_now.time()
    if local_now.weekday() >= 5:
        return next_business_day(local_now), "weekend"
    if current_time < time(9, 30):
        return local_now.normalize(), "pre_open"
    if current_time <= time(16, 0):
        return local_now.normalize(), "live"
    return next_business_day(local_now), "post_close"


def first_valid_price(*values: object, default: float | None = None) -> float:
    for value in values:
        if value is None or pd.isna(value):
            continue
        number = float(value)
        if number > 0:
            return number
    if default is not None:
        return float(default)
    raise ValueError("no valid price available")


def action_from_states(current_state: int, recommended_state: int) -> str:
    if current_state == 0 and recommended_state == 1:
        return "买入"
    if current_state == 1 and recommended_state == 0:
        return "卖出"
    if recommended_state == 1:
        return "持有"
    return "空仓"


def load_state_store(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STATE_STORE_COLUMNS)
    state = pd.read_csv(path, parse_dates=["session_date", "updated_at", "feature_bar_time"])
    missing = [col for col in STATE_STORE_COLUMNS if col not in state.columns]
    for col in missing:
        state[col] = np.nan
    return state[STATE_STORE_COLUMNS].copy()


def resolve_reference_state(
    state_store: pd.DataFrame,
    market: str,
    symbol: str,
    session_date: pd.Timestamp,
    fallback_state: int,
) -> tuple[int, str]:
    if state_store.empty:
        return int(fallback_state), "historical_session"
    mask = (
        state_store["market"].eq(market)
        & state_store["symbol"].eq(symbol)
        & pd.to_datetime(state_store["session_date"]).eq(pd.Timestamp(session_date))
    )
    prior = state_store.loc[mask].sort_values("updated_at")
    if prior.empty:
        return int(fallback_state), "historical_session"
    return int(prior.iloc[-1]["model_state"]), "state_store"


def persist_state_store(path: Path, live_df: pd.DataFrame) -> pd.DataFrame:
    state_store = load_state_store(path)
    update_rows = live_df[
        [
            "market",
            "symbol",
            "session_date",
            "recommended_state",
            "signal_effective_time",
            "feature_bar_time",
            "signal_effective_price",
            "probability_long",
            "action",
            "market_phase",
        ]
    ].rename(columns={"recommended_state": "model_state", "signal_effective_time": "updated_at"})
    combined = update_rows.copy() if state_store.empty else pd.concat([state_store, update_rows], ignore_index=True)
    combined = combined.sort_values("updated_at").drop_duplicates(["market", "symbol", "session_date"], keep="last")
    combined.to_csv(path, index=False)
    return combined.sort_values(["market", "session_date"]).reset_index(drop=True)


def merge_previous_daily_features(base: pd.DataFrame, features: pd.DataFrame, feature_date_col: str = "date") -> pd.DataFrame:
    return pd.merge_asof(
        base.sort_values("session_date"),
        features.sort_values(feature_date_col),
        left_on="session_date",
        right_on=feature_date_col,
        direction="backward",
        allow_exact_matches=False,
    )


def build_cn_open_dataset(
    events_path: Path,
    cn_symbol: str,
    cn_daily: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_features: pd.DataFrame,
    cn_sessions: pd.DataFrame,
) -> pd.DataFrame:
    events = pd.read_excel(events_path, sheet_name="events")
    events = events[events["market"] == "场内"].copy()
    events["datetime"] = pd.to_datetime(events["datetime"])
    events["session_date"] = events["datetime"].dt.normalize()
    events["nq_signal"] = pd.to_numeric(
        events["content"].astype(str).str.extract(r"纳斯达克指数\(([-+]?\d+(?:\.\d+)?)\)", expand=False), errors="coerce"
    )
    open_labels = events.sort_values("datetime").groupby("session_date").first().reset_index()
    open_labels = open_labels[(open_labels["session_date"] >= pd.Timestamp("2026-01-28")) & open_labels["nq_signal"].notna()].copy()
    open_labels = open_labels.rename(columns={"state": "target"})

    bars = fetch_sina_cn_intraday(cn_symbol, interval_minutes=5, datalen=5000)
    open_bars = bars[bars["datetime"].dt.strftime("%H:%M") == "09:35"].copy()
    open_bars = open_bars.rename(columns={"date": "session_date", "open": "cn_open_5m", "close": "cn_close_5m"})

    day_last_nq = events.dropna(subset=["nq_signal"]).sort_values("datetime").groupby("session_date")["nq_signal"].last().shift(1).reset_index()
    day_last_nq = day_last_nq.rename(columns={"nq_signal": "prev_nq_signal"})
    prev_state = cn_sessions[["session_date", "state"]].copy().rename(columns={"state": "prev_state"})

    beta = estimate_cn_etf_beta(cn_daily, nq_features[["date", "nq_close"]].rename(columns={"nq_close": "adjclose"}))
    prev_close = cn_daily[["date", "adjclose"]].rename(columns={"date": "session_date", "adjclose": "prev_cn_close"})
    prev_close["session_date"] = pd.to_datetime(prev_close["session_date"]) + BDay(1)

    base = open_labels[["session_date", "target", "signal", "nq_signal"]].merge(open_bars[["session_date", "cn_open_5m", "cn_close_5m"]], on="session_date", how="inner")
    base = base.merge(day_last_nq, on="session_date", how="left").merge(prev_state, on="session_date", how="left").merge(prev_close, on="session_date", how="left")
    base["nq_overnight_ret"] = base["nq_signal"] / base["prev_nq_signal"] - 1.0
    base["cn_open_gap"] = base["cn_open_5m"] / base["prev_cn_close"] - 1.0
    base["fair_value"] = base["prev_cn_close"] * (1.0 + beta * base["nq_overnight_ret"])
    base["premium_rate"] = base["cn_open_5m"] / base["fair_value"] - 1.0
    base["intraday_open_move"] = base["cn_close_5m"] / base["cn_open_5m"] - 1.0

    base = merge_previous_daily_features(base, cn_features)
    base = merge_previous_daily_features(base, qqq_features.drop(columns=["date"]).assign(date=qqq_features["date"]), feature_date_col="date")
    base = merge_previous_daily_features(base, nq_features.drop(columns=["date"]).assign(date=nq_features["date"]), feature_date_col="date")
    return base.dropna(subset=["target"]).sort_values("session_date").reset_index(drop=True)


def build_us_open_dataset(
    events_path: Path,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    us_sessions: pd.DataFrame,
) -> pd.DataFrame:
    events = pd.read_excel(events_path, sheet_name="events")
    events = events[events["market"] == "美股"].copy()
    events["datetime"] = pd.to_datetime(events["datetime"])
    events["session_date"] = (events["datetime"] - pd.to_timedelta((events["datetime"].dt.hour < 12).astype(int), unit="D")).dt.normalize()
    open_labels = events.sort_values("datetime").groupby("session_date").first().reset_index()
    open_labels = open_labels.rename(columns={"state": "target"})

    qqq_pre, _ = fetch_yahoo_chart("QQQ", interval="5m", date_range="60d", include_prepost=True, include_adjusted=False)
    qqq_pre["session_date"] = qqq_pre["datetime"].dt.normalize()
    qqq_pre = qqq_pre[qqq_pre["datetime"].dt.strftime("%H:%M") < "09:30"].groupby("session_date").tail(1)
    qqq_pre = qqq_pre.rename(columns={"close": "qqq_pre_close"})

    nq_pre, _ = fetch_yahoo_chart("NQ=F", interval="5m", date_range="60d", include_prepost=True, include_adjusted=False)
    nq_pre["session_date"] = nq_pre["datetime"].dt.normalize()
    nq_pre = nq_pre[nq_pre["datetime"].dt.strftime("%H:%M") < "09:30"].groupby("session_date").tail(1)
    nq_pre = nq_pre.rename(columns={"close": "nq_pre_close"})

    prev_state = us_sessions[["session_date", "state"]].copy().rename(columns={"state": "prev_state"})
    base = open_labels[["session_date", "target", "signal"]].merge(qqq_pre[["session_date", "qqq_pre_close"]], on="session_date", how="inner")
    base = base.merge(nq_pre[["session_date", "nq_pre_close"]], on="session_date", how="inner")
    base = base.merge(prev_state, on="session_date", how="left")

    base = merge_previous_daily_features(base, qqq_features)
    base = merge_previous_daily_features(base, nq_features)
    base["qqq_pre_gap"] = base["qqq_pre_close"] / base["qqq_close"] - 1.0
    base["nq_pre_gap"] = base["nq_pre_close"] / base["nq_close"] - 1.0
    base["pre_gap_spread"] = base["qqq_pre_gap"] - base["nq_pre_gap"]
    return base.dropna(subset=["target"]).sort_values("session_date").reset_index(drop=True)


def select_decision_threshold(preds: pd.DataFrame, market: str, default: float = 0.5) -> float:
    if preds.empty:
        return default
    actual = preds["actual_target"].astype(int)
    if actual.nunique() < 2:
        return default

    prefer_precision = market == "场内开盘"
    threshold_grid = np.arange(0.45, 0.801, 0.025) if prefer_precision else np.arange(0.35, 0.801, 0.025)
    best_threshold = default
    best_score = -np.inf
    best_precision = -np.inf

    for threshold in threshold_grid:
        pred = (preds["probability"] >= threshold).astype(int)
        precision = float(precision_score(actual, pred, zero_division=0))
        f1 = float(f1_score(actual, pred, zero_division=0))
        score = (0.7 * f1 + 0.3 * precision) if prefer_precision else f1
        if score > best_score or (np.isclose(score, best_score) and precision > best_precision) or (
            np.isclose(score, best_score) and np.isclose(precision, best_precision) and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_score = score
            best_precision = precision
    return best_threshold


def walk_forward_classifier(dataset: pd.DataFrame, market: str) -> NowcastArtifacts:
    dataset = dataset.copy().reset_index(drop=True)
    feature_cols = pick_feature_columns(dataset)
    min_train = max(20, int(len(dataset) * 0.45))
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    rows = []
    for i in range(min_train, len(dataset)):
        train = dataset.iloc[:i]
        y_train = train["target"].astype(int)
        if y_train.nunique() < 2:
            continue
        pipeline.fit(train[feature_cols], y_train)
        row = dataset.iloc[[i]]
        raw_prob = float(pipeline.predict_proba(row[feature_cols])[0, 1])
        prob, overlay_reason = apply_wave_fib_overlay(raw_prob, row, market)
        rows.append(
            {
                "session_date": row["session_date"].iloc[0],
                "actual_target": int(row["target"].iloc[0]),
                "raw_probability": raw_prob,
                "probability": prob,
                "wave_overlay": overlay_reason,
            }
        )
    preds = pd.DataFrame(rows)
    decision_threshold = select_decision_threshold(preds, market)
    metrics = {
        "market": market,
        "sample_count": len(dataset),
        "oos_count": len(preds),
        "decision_threshold": decision_threshold,
    }
    if not preds.empty:
        preds["predicted_state"] = (preds["probability"] >= decision_threshold).astype(int)
        actual = preds["actual_target"].astype(int)
        pred = preds["predicted_state"].astype(int)
        metrics.update(
            {
                "accuracy": float(accuracy_score(actual, pred)),
                "precision": float(precision_score(actual, pred, zero_division=0)),
                "recall": float(recall_score(actual, pred, zero_division=0)),
                "f1": float(f1_score(actual, pred, zero_division=0)),
                "auc": float(roc_auc_score(actual, preds["probability"])) if actual.nunique() > 1 else np.nan,
            }
        )
    pipeline.fit(dataset[feature_cols], dataset["target"].astype(int))
    return NowcastArtifacts(
        name=market,
        dataset=dataset,
        predictions=preds,
        metrics=pd.DataFrame([metrics]),
        pipeline=pipeline,
        feature_cols=feature_cols,
        decision_threshold=decision_threshold,
    )


def current_cn_nowcast_row(
    events_path: Path,
    cn_symbol: str,
    cn_daily: pd.DataFrame,
    cn_features: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_sessions: pd.DataFrame,
) -> pd.DataFrame:
    quote = fetch_cn_quote(cn_symbol)
    bars = fetch_sina_cn_intraday(cn_symbol, interval_minutes=5, datalen=5000)
    bars = bars.dropna(subset=["open", "close"]).sort_values("datetime").reset_index(drop=True)
    latest_bar = bars.iloc[-1]
    session_date, market_phase = resolve_cn_session(pd.Timestamp(quote["snapshot_time"]))

    events = pd.read_excel(events_path, sheet_name="events")
    events = events[events["market"] == "场内"].copy()
    events["datetime"] = pd.to_datetime(events["datetime"])
    events["nq_signal"] = pd.to_numeric(
        events["content"].astype(str).str.extract(r"纳斯达克指数\(([-+]?\d+(?:\.\d+)?)\)", expand=False), errors="coerce"
    )
    last_nq = events.dropna(subset=["nq_signal"]).sort_values("datetime").groupby(events["datetime"].dt.normalize())["nq_signal"].last().iloc[-1]

    nq_live, _ = fetch_yahoo_chart("NQ=F", interval="5m", date_range="5d", include_prepost=True, include_adjusted=False)
    nq_live = nq_live.dropna(subset=["close"]).sort_values("datetime")
    current_nq = float(nq_live["close"].iloc[-1])

    beta = estimate_cn_etf_beta(cn_daily, nq_features[["date", "nq_close"]].rename(columns={"nq_close": "adjclose"}))
    latest_bar_open = first_valid_price(latest_bar["open"], quote.get("open"), quote["prev_close"])
    latest_bar_close = first_valid_price(quote["last"], latest_bar["close"], latest_bar_open)
    feature_bar_time = pd.Timestamp(latest_bar["datetime"])
    signal_effective_time = pd.Timestamp(quote["snapshot_time"])
    row = pd.DataFrame(
        [
            {
                "session_date": session_date,
                "target": np.nan,
                "signal": "live",
                "nq_signal": current_nq,
                "cn_open_5m": latest_bar_open,
                "cn_close_5m": latest_bar_close,
                "prev_nq_signal": last_nq,
                "prev_state": int(cn_sessions.sort_values("session_date")["state"].iloc[-1]),
                "prev_cn_close": first_valid_price(quote["prev_close"], cn_daily.sort_values("date")["adjclose"].iloc[-1]),
                "feature_bar_time": feature_bar_time,
                "market_phase": market_phase,
                "signal_effective_time": signal_effective_time,
                "signal_effective_price": latest_bar_close,
            }
        ]
    )
    row["nq_overnight_ret"] = row["nq_signal"] / row["prev_nq_signal"] - 1.0
    row["cn_open_gap"] = row["cn_open_5m"] / row["prev_cn_close"] - 1.0
    row["fair_value"] = row["prev_cn_close"] * (1.0 + beta * row["nq_overnight_ret"])
    row["premium_rate"] = row["cn_open_5m"] / row["fair_value"] - 1.0
    row["intraday_open_move"] = 0.0
    row = merge_previous_daily_features(row, cn_features)
    row = merge_previous_daily_features(row, qqq_features, feature_date_col="date")
    row = merge_previous_daily_features(row, nq_features, feature_date_col="date")
    row["snapshot_time"] = row["signal_effective_time"]
    row["snapshot_price"] = row["signal_effective_price"]
    return row


def current_us_nowcast_row(
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    us_sessions: pd.DataFrame,
) -> pd.DataFrame:
    qqq_live, _ = fetch_yahoo_chart("QQQ", interval="5m", date_range="5d", include_prepost=True, include_adjusted=False)
    qqq_live = qqq_live.dropna(subset=["close"]).sort_values("datetime")
    nq_live, _ = fetch_yahoo_chart("NQ=F", interval="5m", date_range="5d", include_prepost=True, include_adjusted=False)
    nq_live = nq_live.dropna(subset=["close"]).sort_values("datetime")
    latest_qqq = qqq_live.iloc[-1]
    latest_nq = nq_live.iloc[-1]
    feature_bar_time = pd.Timestamp(latest_qqq["datetime"])
    session_date, market_phase = resolve_us_session()
    row = pd.DataFrame(
        [
            {
                "session_date": session_date,
                "target": np.nan,
                "signal": "live",
                "qqq_pre_close": float(latest_qqq["close"]),
                "nq_pre_close": float(latest_nq["close"]),
                "prev_state": int(us_sessions.sort_values("session_date")["state"].iloc[-1]),
                "feature_bar_time": feature_bar_time,
                "market_phase": market_phase,
                "signal_effective_time": feature_bar_time,
                "signal_effective_price": float(latest_qqq["close"]),
            }
        ]
    )
    row = merge_previous_daily_features(row, qqq_features)
    row = merge_previous_daily_features(row, nq_features)
    row["qqq_pre_gap"] = row["qqq_pre_close"] / row["qqq_close"] - 1.0
    row["nq_pre_gap"] = row["nq_pre_close"] / row["nq_close"] - 1.0
    row["pre_gap_spread"] = row["qqq_pre_gap"] - row["nq_pre_gap"]
    row["snapshot_time"] = row["signal_effective_time"]
    row["snapshot_price"] = row["signal_effective_price"]
    return row


def infer_live(
    art: NowcastArtifacts,
    row: pd.DataFrame,
    market: str,
    symbol: str,
    state_store: pd.DataFrame,
) -> dict[str, object]:
    raw_prob = float(art.pipeline.predict_proba(row[art.feature_cols])[0, 1])
    prob, overlay_reason = apply_wave_fib_overlay(raw_prob, row, market)
    session_date = pd.Timestamp(row["session_date"].iloc[0])
    fallback_state = int(row["prev_state"].iloc[0])
    current_state, current_state_source = resolve_reference_state(state_store, market, symbol, session_date, fallback_state)
    recommended_state = int(prob >= art.decision_threshold)
    action = action_from_states(current_state, recommended_state)
    return {
        "market": market,
        "session_date": session_date,
        "market_phase": row["market_phase"].iloc[0],
        "feature_bar_time": pd.Timestamp(row["feature_bar_time"].iloc[0]),
        "snapshot_time": pd.Timestamp(row["snapshot_time"].iloc[0]),
        "snapshot_price": float(row["snapshot_price"].iloc[0]),
        "signal_effective_time": pd.Timestamp(row["signal_effective_time"].iloc[0]),
        "signal_effective_price": float(row["signal_effective_price"].iloc[0]),
        "current_state": current_state,
        "current_state_source": current_state_source,
        "raw_probability": raw_prob,
        "probability_long": prob,
        "decision_threshold": art.decision_threshold,
        "recommended_state": recommended_state,
        "action": action,
        "wave_overlay": overlay_reason,
        "symbol": symbol,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build open-nowcast models for 159941 and QQQ.")
    parser.add_argument("--events", default="wsquant_market_timeline.xlsx")
    parser.add_argument("--market", default="all", choices=["all", "cn", "us"])
    parser.add_argument("--cn-symbol", default="159941.SZ")
    parser.add_argument("--output", default="open_nowcast_results.xlsx")
    parser.add_argument("--live-output", default="open_nowcast_live.csv")
    parser.add_argument("--state-output", default="open_nowcast_state.csv")
    parser.add_argument("--nq-archive-dir", default="nq_contract_archive")
    parser.add_argument("--strict-nq-archive", action="store_true")
    parser.add_argument("--range", default="2y")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    events_path = Path(args.events).resolve()
    output_path = Path(args.output).resolve()
    live_output_path = Path(args.live_output).resolve()
    state_output_path = Path(args.state_output)
    if not state_output_path.is_absolute():
        state_output_path = (output_path.parent / state_output_path).resolve()
    else:
        state_output_path = state_output_path.resolve()
    archive_dir = Path(args.nq_archive_dir).resolve()
    need_cn = args.market in {"all", "cn"}
    need_us = args.market in {"all", "us"}
    state_store = load_state_store(state_output_path)

    us_sessions, cn_sessions = build_signal_sessions(events_path)
    qqq_daily = fetch_yahoo_daily("QQQ", args.range)
    cn_daily = fetch_yahoo_daily(args.cn_symbol, args.range) if need_cn else None
    archive_years = range(pd.Timestamp.today().year - 1, pd.Timestamp.today().year + 2)
    archive_nq_status = archive_nq_contracts(archive_dir, archive_years, date_range=args.range)
    archived_nq_daily, _, archive_complete = build_archived_nq_main(archive_dir, qqq_daily["date"].min(), qqq_daily["date"].max())
    if archive_complete:
        nq_daily = archived_nq_daily
        nq_source = "single-month archive"
    else:
        if args.strict_nq_archive:
            raise RuntimeError("NQ archive incomplete for open-nowcast model.")
        nq_daily, _ = build_roll_adjusted_nq_main(args.range)
        nq_source = "roll-adjusted NQ=F fallback"

    qqq_features = make_price_features(qqq_daily, "qqq")
    nq_features = make_price_features(nq_daily, "nq")
    cn_features = make_price_features(cn_daily, "cnetf") if need_cn and cn_daily is not None else None

    metrics_frames = []
    live_rows = []
    cn_dataset = us_dataset = None
    cn_art = us_art = None

    if need_cn and cn_daily is not None and cn_features is not None:
        cn_dataset = build_cn_open_dataset(events_path, args.cn_symbol, cn_daily, qqq_features, nq_features, cn_features, cn_sessions)
        cn_art = walk_forward_classifier(cn_dataset, "场内开盘")
        cn_live = current_cn_nowcast_row(events_path, args.cn_symbol, cn_daily, cn_features, qqq_features, nq_features, cn_sessions)
        live_rows.append(infer_live(cn_art, cn_live, "场内开盘", args.cn_symbol, state_store))
        metrics_frames.append(cn_art.metrics)

    if need_us:
        us_dataset = build_us_open_dataset(events_path, qqq_features, nq_features, us_sessions)
        us_art = walk_forward_classifier(us_dataset, "美股开盘")
        us_live = current_us_nowcast_row(qqq_features, nq_features, us_sessions)
        live_rows.append(infer_live(us_art, us_live, "美股开盘", "QQQ", state_store))
        metrics_frames.append(us_art.metrics)

    live_df = pd.DataFrame(live_rows)
    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    updated_state_store = persist_state_store(state_output_path, live_df) if not live_df.empty else state_store

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        if cn_art is not None and cn_dataset is not None:
            cn_art.predictions.to_excel(writer, sheet_name="cn_open_predictions", index=False)
            cn_dataset.to_excel(writer, sheet_name="cn_open_dataset", index=False)
        if us_art is not None and us_dataset is not None:
            us_art.predictions.to_excel(writer, sheet_name="us_open_predictions", index=False)
            us_dataset.to_excel(writer, sheet_name="us_open_dataset", index=False)
        live_df.to_excel(writer, sheet_name="live_nowcast", index=False)
        updated_state_store.to_excel(writer, sheet_name="signal_state_store", index=False)
        archive_nq_status.to_excel(writer, sheet_name="nq_archive_status", index=False)
        notes = pd.DataFrame(
            [
                ("market_mode", args.market),
                ("cn_live_granularity", "CN live nowcast uses the latest 5-minute bar plus current quote and keeps the session on the same trade date until the market close."),
                ("us_live_granularity", "US live nowcast uses the latest 5-minute QQQ/NQ bar and keeps the session on the same US trade date until the regular-session close."),
                ("decision_rule", "Nowcast entry/exit uses a market-specific decision threshold calibrated from out-of-sample predictions instead of a fixed 0.5 cutoff."),
                ("nq_source", nq_source),
                ("cn_live_source", "Eastmoney quote + Yahoo NQ live bar"),
                ("us_live_source", "Yahoo QQQ pre/post + Yahoo NQ live bar"),
                ("state_store", str(state_output_path)),
                ("action_rule", "0->1=买入, 1->0=卖出, 1->1=持有, 0->0=空仓."),
            ],
            columns=["item", "value"],
        )
        notes.to_excel(writer, sheet_name="notes", index=False)

    live_df.to_csv(live_output_path, index=False)
    print(f"Wrote workbook: {output_path}")
    print(f"Wrote live CSV: {live_output_path}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nLive nowcast:")
    print(live_df.to_string(index=False))


if __name__ == "__main__":
    main()
