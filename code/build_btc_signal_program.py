#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_5m_signal_program import (
    STATE_COLUMNS,
    _now_local,
    action_from_states,
    append_signal_history,
    latest_state_from_store,
    latest_store_row,
    load_state_store,
    price_format,
    save_state_store,
    signal_result_text,
    terminal_rows,
)
from build_nq_quant_model import (
    fetch_yahoo_chart,
    make_price_features,
    merge_asof_features,
    summarize_live_drivers,
)

BTC_ENTRY_THRESHOLD = 0.64
BTC_EXIT_THRESHOLD = 0.36
BTC_CONFIRM_THRESHOLD = 0.58
BTC_CONFIRM_WINDOW = 3
BTC_CONFIRM_REQUIRED = 2
BTC_MIN_EXPECTED_EDGE = 0.0015
BTC_STALE_NQ_ENTRY_THRESHOLD = 0.68
BTC_STALE_NQ_MIN_EXPECTED_EDGE = 0.0020
BTC_STOP_LOSS = 0.008
BTC_PROFIT_THRESHOLD = 0.0015
BTC_LABEL_HORIZON = 6
BTC_COOLDOWN_BARS = 2
BTC_MODEL_REFIT_STRIDE = 6
EDGE_LOOKBACK_ROWS = 240
EDGE_BANDWIDTH = 0.10
EDGE_MIN_NEIGHBORS = 24
NQ_STALE_MINUTES = 180.0
COINBASE_REST_URL = "https://api.exchange.coinbase.com"
COINBASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}
COINBASE_MAX_CANDLES = 300
FIVE_MINUTES = pd.Timedelta(minutes=5)
ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class BTCArtifacts:
    market: str
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    pipeline: Pipeline
    feature_cols: list[str]


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BTC-USD 5m live long/flat strategy using NQ as context.")
    parser.add_argument("--output", default=str(ROOT_DIR / "data/signals/btc/btc_signal_results.xlsx"))
    parser.add_argument("--live-output", default=str(ROOT_DIR / "data/signals/btc/btc_live_signals.csv"))
    parser.add_argument("--state-output", default=str(ROOT_DIR / "data/signals/btc/btc_signal_state.csv"))
    parser.add_argument("--history-output", default=str(ROOT_DIR / "data/signals/btc/btc_signal_history.csv"))
    parser.add_argument("--range", default="60d")
    parser.add_argument("--intraday-bars", type=int, default=5000)
    parser.add_argument("--print-mode", default="event", choices=["event", "all", "none"])
    parser.add_argument("--symbol", default="BTC-USD")
    return parser.parse_args(argv)


def _safe_feature_value(row: pd.Series, key: str) -> float:
    if key not in row or pd.isna(row[key]):
        return np.nan
    return float(row[key])


def fetch_coinbase_product_candles(symbol: str, bars: int, granularity_seconds: int = 300) -> pd.DataFrame:
    endpoint = f"{COINBASE_REST_URL}/products/{symbol}/candles"
    end = pd.Timestamp.utcnow().floor("5min")
    frames: list[pd.DataFrame] = []
    remaining = int(bars)

    while remaining > 0:
        chunk_size = min(COINBASE_MAX_CANDLES, remaining)
        start = end - pd.Timedelta(seconds=granularity_seconds * (chunk_size - 1))
        response = requests.get(
            endpoint,
            params={
                "granularity": granularity_seconds,
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
            },
            headers=COINBASE_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break

        chunk = pd.DataFrame(payload, columns=["timestamp", "low", "high", "open", "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["datetime"] = pd.to_datetime(chunk["timestamp"], unit="s", utc=True).dt.tz_localize(None)
        chunk["date"] = chunk["datetime"]
        chunk["symbol"] = symbol
        chunk = chunk[["datetime", "date", "open", "high", "low", "close", "volume", "symbol"]]
        frames.append(chunk)

        remaining -= len(chunk)
        if len(chunk) < chunk_size:
            break
        end = chunk["datetime"].min() - FIVE_MINUTES

    if not frames:
        raise RuntimeError(f"failed to fetch {symbol} candles from Coinbase")

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return frame


def fetch_btc_intraday(symbol: str, bars: int) -> pd.DataFrame:
    frame = fetch_coinbase_product_candles(symbol, bars=bars, granularity_seconds=300)
    frame = frame.dropna(subset=["datetime", "open", "close"]).sort_values("datetime").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["datetime"])
    frame["adjclose"] = frame["close"].astype(float)
    frame["trade_date"] = frame["datetime"].dt.normalize()
    frame["tod_min"] = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    frame["tod_sin"] = np.sin(2.0 * np.pi * frame["tod_min"] / 1440.0)
    frame["tod_cos"] = np.cos(2.0 * np.pi * frame["tod_min"] / 1440.0)
    frame["bar_index"] = frame.groupby("trade_date").cumcount()
    frame["session_high_so_far"] = frame.groupby("trade_date")["high"].cummax()
    frame["session_low_so_far"] = frame.groupby("trade_date")["low"].cummin()
    frame["session_range_so_far"] = frame["session_high_so_far"] - frame["session_low_so_far"]
    frame["open_to_now_ret"] = frame["close"] / frame.groupby("trade_date")["open"].transform("first") - 1.0
    frame["range_pos_so_far"] = (
        (frame["close"] - frame["session_low_so_far"]) / frame["session_range_so_far"].replace(0.0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    return frame


def fetch_nq_intraday(date_range: str) -> pd.DataFrame:
    frame, _ = fetch_yahoo_chart("NQ=F", interval="5m", date_range=date_range, include_prepost=True, include_adjusted=False)
    frame = frame[["datetime", "date", "open", "high", "low", "close", "volume", "symbol"]]
    frame = frame.dropna(subset=["datetime", "open", "close"]).sort_values("datetime").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["datetime"])
    frame["adjclose"] = frame["close"].astype(float)
    return frame


def latest_nqmain_rt_value(nq_raw: pd.DataFrame) -> float:
    return float(nq_raw.sort_values("datetime").iloc[-1]["close"])


def make_intraday_feature_frame(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    feature_input = raw.copy()
    feature_input["date"] = feature_input["datetime"]
    feature_input["adjclose"] = feature_input["close"]
    features = make_price_features(feature_input, prefix).rename(columns={"date": "bar_time"})
    context_cols = [
        "datetime",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tod_min",
        "tod_sin",
        "tod_cos",
        "bar_index",
        "session_high_so_far",
        "session_low_so_far",
        "session_range_so_far",
        "open_to_now_ret",
        "range_pos_so_far",
    ]
    base = raw[context_cols].merge(features, left_on="datetime", right_on="bar_time", how="left")
    base["date"] = base["datetime"]
    return base


def make_btc_dataset(raw: pd.DataFrame, nq_features: pd.DataFrame) -> pd.DataFrame:
    base = make_intraday_feature_frame(raw, "btc")
    nq_context = nq_features.rename(columns={"date": "nq_bar_time"})
    base = merge_asof_features(base, nq_context, "nq_bar_time", drop_right_key=False)
    base["nq_minutes_stale"] = (
        (pd.to_datetime(base["datetime"]) - pd.to_datetime(base["nq_bar_time"])).dt.total_seconds() / 60.0
    )
    base["nq_is_fresh"] = (base["nq_minutes_stale"] <= NQ_STALE_MINUTES).astype(float)
    base = base.sort_values("datetime").reset_index(drop=True)

    close = base["btc_close"].astype(float)
    future_low = pd.concat([base["low"].shift(-i) for i in range(1, BTC_LABEL_HORIZON + 1)], axis=1).min(axis=1)
    future_high = pd.concat([base["high"].shift(-i) for i in range(1, BTC_LABEL_HORIZON + 1)], axis=1).max(axis=1)
    trade_ret_1 = close.shift(-1) / close - 1.0
    trade_ret_h = close.shift(-BTC_LABEL_HORIZON) / close - 1.0
    fwd_low_dd_h = future_low / close - 1.0
    edge = 0.55 * trade_ret_1 + 0.45 * trade_ret_h - 1.20 * fwd_low_dd_h.clip(upper=0).abs()

    base = base.assign(
        trade_ret_1=trade_ret_1,
        trade_ret_h=trade_ret_h,
        fwd_low_dd_h=fwd_low_dd_h,
        fwd_high_up_h=future_high / close - 1.0,
        edge_proxy_label=edge,
        target=((edge > BTC_PROFIT_THRESHOLD) & (fwd_low_dd_h > -BTC_STOP_LOSS)).astype(float),
        current_state=0,
        target_time=base["datetime"].shift(-1),
    )
    return base.dropna(subset=["target", "trade_ret_1"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "datetime",
        "date",
        "nq_bar_time",
        "target_time",
        "target",
        "trade_ret_1",
        "trade_ret_h",
        "fwd_low_dd_h",
        "fwd_high_up_h",
        "edge_proxy_label",
        "current_state",
        "trade_date",
        "symbol",
    }
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if "fwd_ret" in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def estimate_expected_edge(prediction_history: pd.DataFrame, probability: float) -> float:
    if prediction_history.empty or "edge_proxy_realized" not in prediction_history.columns:
        return np.nan
    history = prediction_history.dropna(subset=["probability_long", "edge_proxy_realized"]).copy()
    if history.empty:
        return np.nan
    history = history.tail(EDGE_LOOKBACK_ROWS)
    distance = (history["probability_long"].astype(float) - float(probability)).abs()
    local = history.loc[distance <= EDGE_BANDWIDTH]
    if len(local) < min(EDGE_MIN_NEIGHBORS, len(history)):
        nearest_index = distance.nsmallest(min(EDGE_MIN_NEIGHBORS, len(history))).index
        local = history.loc[nearest_index]
    if local.empty:
        return np.nan
    return float(local["edge_proxy_realized"].mean())


def btc_entry_confirmation_stats(recent_probabilities: list[float]) -> tuple[float, int]:
    if not recent_probabilities:
        return np.nan, 0
    sample = recent_probabilities[-BTC_CONFIRM_WINDOW:]
    smooth_probability = float(np.mean(sample))
    confirm_count = int(sum(prob >= BTC_CONFIRM_THRESHOLD for prob in sample))
    return smooth_probability, confirm_count


def apply_btc_overlay(probability: float, row: pd.Series | pd.DataFrame) -> tuple[float, str]:
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    adjusted = float(probability)
    btc_ret_13 = _safe_feature_value(row, "btc_ret_13")
    btc_fib_8 = _safe_feature_value(row, "btc_fib_pos_8")
    btc_ew = _safe_feature_value(row, "btc_ew_thrust_5")
    nq_ret_13 = _safe_feature_value(row, "nq_ret_13")
    nq_fib_8 = _safe_feature_value(row, "nq_fib_pos_8")
    nq_ew = _safe_feature_value(row, "nq_ew_thrust_5")
    nq_stale = _safe_feature_value(row, "nq_minutes_stale")

    if pd.notna(nq_stale) and nq_stale > NQ_STALE_MINUTES:
        if all(pd.notna(x) for x in (btc_ret_13, btc_fib_8, btc_ew)):
            if btc_ret_13 < 0 and btc_fib_8 > 0.80 and btc_ew < 0:
                return min(adjusted, 0.22), "btc_exhaustion_no_nq"
            if btc_ret_13 < 0 and btc_fib_8 < 0.18 and btc_ew > -0.15:
                return max(adjusted, 0.56), "btc_oversold_rebound_no_nq"
        return adjusted, "btc_nq_stale"

    if all(pd.notna(x) for x in (btc_ret_13, btc_fib_8, btc_ew, nq_ret_13, nq_fib_8, nq_ew)):
        if btc_ret_13 < 0 and nq_ret_13 < 0:
            if btc_fib_8 > 0.82 and btc_ew < 0 and nq_ew < 0:
                return min(adjusted, 0.18), "btc_distribution_cap"
            if btc_fib_8 <= 0.382 and nq_fib_8 <= 0.618 and btc_ew < 0:
                return min(adjusted, 0.30), "btc_bear_rebound_cap"
            if btc_fib_8 < 0.20 and nq_fib_8 < 0.382 and btc_ew > -0.15 and nq_ew > -0.25:
                return max(adjusted, 0.60), "btc_joint_rebound"
        if btc_ret_13 > 0 and btc_fib_8 < 0.25 and btc_ew > 0 and nq_ew > -0.10:
            return max(adjusted, 0.58), "btc_trend_pullback"

    return adjusted, "none"


def entry_signal_allowed(
    probability: float,
    recent_probabilities: list[float],
    expected_edge: float,
    nq_minutes_stale: float,
) -> tuple[bool, float, int]:
    smooth_probability, confirm_count = btc_entry_confirmation_stats(recent_probabilities)
    entry_allowed = (
        probability >= BTC_ENTRY_THRESHOLD
        and smooth_probability >= BTC_CONFIRM_THRESHOLD
        and confirm_count >= BTC_CONFIRM_REQUIRED
    )
    if pd.notna(expected_edge):
        entry_allowed = entry_allowed and expected_edge >= BTC_MIN_EXPECTED_EDGE
    if pd.notna(nq_minutes_stale) and nq_minutes_stale > NQ_STALE_MINUTES:
        entry_allowed = (
            entry_allowed
            and probability >= BTC_STALE_NQ_ENTRY_THRESHOLD
            and smooth_probability >= 0.60
        )
        if pd.notna(expected_edge):
            entry_allowed = entry_allowed and expected_edge >= BTC_STALE_NQ_MIN_EXPECTED_EDGE
    return entry_allowed, smooth_probability, confirm_count


def walk_forward_btc(dataset: pd.DataFrame) -> BTCArtifacts:
    dataset = dataset.copy().reset_index(drop=True)
    cols = feature_columns(dataset)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    min_train = max(480, int(len(dataset) * 0.25))
    rows: list[dict[str, object]] = []
    current_position = 0
    cooldown = 0
    probability_history: list[float] = []
    model_ready = False

    for i in range(min_train, len(dataset)):
        should_refit = (not model_ready) or ((i - min_train) % BTC_MODEL_REFIT_STRIDE == 0)
        if should_refit:
            train = dataset.iloc[:i].dropna(subset=cols + ["target"])
            if train.empty or train["target"].nunique() < 2:
                continue
            pipeline.fit(train[cols], train["target"].astype(int))
            model_ready = True

        if not model_ready:
            continue
        row = dataset.iloc[[i]]
        raw_probability = float(pipeline.predict_proba(row[cols])[0, 1])
        probability, overlay_reason = apply_btc_overlay(raw_probability, row.iloc[0])
        recent_probabilities = (probability_history + [probability])[-BTC_CONFIRM_WINDOW:]
        history_for_edge = pd.DataFrame(rows)
        expected_edge = estimate_expected_edge(history_for_edge, probability)
        entry_allowed, smooth_probability, confirm_count = entry_signal_allowed(
            probability,
            recent_probabilities,
            expected_edge,
            float(row["nq_minutes_stale"].iloc[0]) if pd.notna(row["nq_minutes_stale"].iloc[0]) else np.nan,
        )

        next_position = current_position
        if cooldown > 0:
            next_position = 0
            cooldown -= 1
        elif current_position == 0 and entry_allowed:
            next_position = 1
        elif current_position == 1 and probability <= BTC_EXIT_THRESHOLD:
            next_position = 0

        next_ret = float(row["trade_ret_1"].iloc[0])
        trade_ret_h = float(row["trade_ret_h"].iloc[0]) if pd.notna(row["trade_ret_h"].iloc[0]) else 0.0
        low_dd = float(row["fwd_low_dd_h"].iloc[0]) if pd.notna(row["fwd_low_dd_h"].iloc[0]) else np.nan
        edge_proxy_realized = 0.55 * next_ret + 0.45 * trade_ret_h - 1.20 * abs(min(low_dd, 0.0))
        realized_ret = 0.0
        stop_triggered = False
        position_for_return = next_position
        if next_position == 1:
            realized_ret = next_ret
            if pd.notna(low_dd) and low_dd < -BTC_STOP_LOSS:
                realized_ret = -BTC_STOP_LOSS
                stop_triggered = True
                next_position = 0
                cooldown = BTC_COOLDOWN_BARS

        rows.append(
            {
                "feature_bar_time": row["datetime"].iloc[0],
                "trade_date": row["trade_date"].iloc[0],
                "current_state": current_position,
                "raw_probability": raw_probability,
                "probability_long": probability,
                "smoothed_probability": smooth_probability,
                "entry_confirm_count": confirm_count,
                "position": next_position,
                "prior_position": current_position,
                "wave_overlay": overlay_reason,
                "trade_ret_1": next_ret,
                "trade_ret_h": trade_ret_h,
                "edge_proxy_realized": edge_proxy_realized,
                "expected_edge": expected_edge,
                "strategy_ret": position_for_return * realized_ret,
                "stop_triggered": int(stop_triggered),
            }
        )
        probability_history.append(probability)
        current_position = next_position

    preds = pd.DataFrame(rows)
    if preds.empty:
        metrics = pd.DataFrame([{"market": "比特币5m", "metric": "rows", "value": 0.0}])
    else:
        returns = preds["strategy_ret"].fillna(0.0)
        equity = (1.0 + returns).cumprod()
        dd = equity / equity.cummax() - 1.0
        metrics = pd.DataFrame(
            [
                {"market": "比特币5m", "metric": "rows", "value": float(len(preds))},
                {"market": "比特币5m", "metric": "cum_return", "value": float(equity.iloc[-1] - 1.0)},
                {"market": "比特币5m", "metric": "avg_return", "value": float(returns.mean())},
                {"market": "比特币5m", "metric": "win_rate", "value": float((returns > 0).mean())},
                {"market": "比特币5m", "metric": "max_drawdown", "value": float(dd.min())},
                {"market": "比特币5m", "metric": "avg_position", "value": float(preds["position"].mean())},
            ]
        )

    pipeline.fit(dataset[cols], dataset["target"].astype(int))
    return BTCArtifacts(
        market="比特币5m",
        dataset=dataset,
        predictions=preds,
        metrics=metrics,
        pipeline=pipeline,
        feature_cols=cols,
    )


def classify_btc_market_phase(feature_bar_time: pd.Timestamp) -> str:
    age_minutes = (_now_local() - pd.Timestamp(feature_bar_time)).total_seconds() / 60.0
    if age_minutes <= 10:
        return "live"
    if age_minutes <= 60:
        return "delayed"
    return "stale"


def build_signal_summary(
    symbol: str,
    nqmain_rt_value: float,
    signal_value: float,
    recommended_state: int,
    feature_bar_time: pd.Timestamp,
    action: str,
    position_since: pd.Timestamp | None = None,
    position_since_exact: int | None = None,
) -> str:
    time_text = pd.Timestamp(feature_bar_time).strftime("%Y-%m-%d %H:%M:%S")
    summary = (
        f"DATA_TIME({time_text}) "
        f"NQMAIN_RT({price_format('NQ=F', nqmain_rt_value)}) "
        f"{symbol}({price_format(symbol, signal_value)}) -> {signal_result_text(recommended_state)}"
    )
    if action == "买入":
        return f"{summary} ENTRY_TIME({time_text})"
    if action == "持有" and pd.notna(position_since):
        since_text = pd.Timestamp(position_since).strftime("%Y-%m-%d %H:%M:%S")
        if int(position_since_exact or 0) == 1:
            return f"{summary} HOLD_SINCE({since_text})"
        return f"{summary} HELD_AT_LEAST_SINCE({since_text})"
    if action == "卖出":
        return f"{summary} EXIT_TIME({time_text})"
    return summary


def live_row_from_frame(
    art: BTCArtifacts,
    live_frame: pd.DataFrame,
    symbol: str,
    prior_row: pd.Series | None,
    nqmain_rt_value: float,
) -> dict[str, object]:
    valid_live = live_frame.dropna(subset=art.feature_cols).sort_values("datetime").copy()
    latest = valid_live.tail(1)
    feature_bar_time = pd.Timestamp(latest["datetime"].iloc[0])
    trade_date = pd.Timestamp(latest["trade_date"].iloc[0]).normalize()
    current_state = latest_state_from_store(
        pd.DataFrame([prior_row]) if prior_row is not None else pd.DataFrame(columns=STATE_COLUMNS),
        "比特币",
        symbol,
        0,
    )
    recent_rows = valid_live.tail(BTC_CONFIRM_WINDOW).copy()
    recent_raw_probabilities = art.pipeline.predict_proba(recent_rows[art.feature_cols])[:, 1]
    recent_probabilities: list[float] = []
    overlay_reason = "none"
    for idx, (_, recent_row) in enumerate(recent_rows.iterrows()):
        adjusted_probability, recent_overlay_reason = apply_btc_overlay(float(recent_raw_probabilities[idx]), recent_row)
        recent_probabilities.append(adjusted_probability)
        if idx == len(recent_rows) - 1:
            overlay_reason = recent_overlay_reason
    raw_probability = float(recent_raw_probabilities[-1])
    probability = float(recent_probabilities[-1])
    expected_edge = estimate_expected_edge(art.predictions, probability)
    entry_allowed, smooth_probability, confirm_count = entry_signal_allowed(
        probability,
        recent_probabilities,
        expected_edge,
        float(latest["nq_minutes_stale"].iloc[0]) if pd.notna(latest["nq_minutes_stale"].iloc[0]) else np.nan,
    )
    recommended_state = current_state
    if current_state == 0 and entry_allowed:
        recommended_state = 1
    elif current_state == 1 and probability <= BTC_EXIT_THRESHOLD:
        recommended_state = 0
    action = action_from_states(current_state, recommended_state)
    market_phase = classify_btc_market_phase(feature_bar_time)

    prior_feature_bar_time = pd.NaT if prior_row is None else prior_row.get("feature_bar_time", pd.NaT)
    prior_position_since = pd.NaT if prior_row is None else prior_row.get("position_since", pd.NaT)
    prior_position_since_exact = 0 if prior_row is None or pd.isna(prior_row.get("position_since_exact", pd.NA)) else int(prior_row["position_since_exact"])
    prior_last_action_time = pd.NaT if prior_row is None else prior_row.get("last_action_time", pd.NaT)

    position_since = pd.NaT
    position_since_exact = 0
    last_action_time = prior_last_action_time
    if action == "买入":
        position_since = feature_bar_time
        position_since_exact = 1
        last_action_time = feature_bar_time
    elif action == "持有":
        if pd.notna(prior_position_since):
            position_since = pd.Timestamp(prior_position_since)
            position_since_exact = prior_position_since_exact
        elif current_state == 1 and pd.notna(prior_feature_bar_time):
            position_since = pd.Timestamp(prior_feature_bar_time)
            position_since_exact = 0
        else:
            position_since = feature_bar_time
            position_since_exact = 1
    elif action == "卖出":
        last_action_time = feature_bar_time

    return {
        "market": "比特币",
        "symbol": symbol,
        "feature_bar_time": feature_bar_time,
        "updated_at": _now_local(),
        "current_state": int(current_state),
        "raw_probability": raw_probability,
        "probability_long": probability,
        "smoothed_probability": smooth_probability,
        "entry_confirm_count": confirm_count,
        "expected_edge": expected_edge,
        "recommended_state": int(recommended_state),
        "action": action,
        "market_phase": market_phase,
        "wave_overlay": overlay_reason,
        "nqmain_rt_value": float(nqmain_rt_value),
        "signal_price": float(latest["close"].iloc[0]),
        "trade_date": trade_date,
        "position_since": position_since,
        "position_since_exact": int(position_since_exact),
        "last_action_time": last_action_time,
        "signal_summary": build_signal_summary(
            symbol,
            float(nqmain_rt_value),
            float(latest["close"].iloc[0]),
            int(recommended_state),
            feature_bar_time,
            action,
            position_since if pd.notna(position_since) else None,
            position_since_exact,
        ),
        "top_drivers": summarize_live_drivers(art.pipeline, latest, art.feature_cols),
    }


def main(argv: list[str] | None = None) -> None:
    configure_console_output()
    args = parse_args(argv)
    output_path = Path(args.output).resolve()
    live_output_path = Path(args.live_output).resolve()
    state_output_path = Path(args.state_output).resolve()
    history_output_path = Path(args.history_output).resolve()
    for path in [output_path, live_output_path, state_output_path, history_output_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    run_at = _now_local()
    state_store = load_state_store(state_output_path)

    print("Fetching NQ 5m history...", flush=True)
    nq_raw = fetch_nq_intraday(args.range)
    nq_features = make_price_features(nq_raw.copy(), "nq")
    nqmain_rt_value = latest_nqmain_rt_value(nq_raw)
    print(f"Loaded NQ 5m bars: {len(nq_raw)}", flush=True)

    print(f"Fetching {args.symbol} 5m history from Coinbase Exchange...", flush=True)
    btc_raw = fetch_btc_intraday(args.symbol, args.intraday_bars)
    print("Building BTC 5m dataset...", flush=True)
    btc_dataset = make_btc_dataset(btc_raw, nq_features)
    print("Training BTC 5m model...", flush=True)
    btc_art = walk_forward_btc(btc_dataset)

    btc_live_features = make_intraday_feature_frame(btc_raw, "btc")
    nq_context = nq_features.rename(columns={"date": "nq_bar_time"})
    btc_live_features = merge_asof_features(btc_live_features, nq_context, "nq_bar_time", drop_right_key=False)
    btc_live_features["nq_minutes_stale"] = (
        (pd.to_datetime(btc_live_features["datetime"]) - pd.to_datetime(btc_live_features["nq_bar_time"])).dt.total_seconds() / 60.0
    )
    btc_live_features["nq_is_fresh"] = (btc_live_features["nq_minutes_stale"] <= NQ_STALE_MINUTES).astype(float)

    prior_row = latest_store_row(state_store, "比特币", args.symbol)
    live_df = pd.DataFrame([live_row_from_frame(btc_art, btc_live_features, args.symbol, prior_row, nqmain_rt_value)])
    live_df["run_at"] = run_at

    updated_state = save_state_store(state_output_path, live_df)
    history_df = append_signal_history(history_output_path, live_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        btc_art.metrics.to_excel(writer, sheet_name="btc_metrics", index=False)
        btc_art.predictions.to_excel(writer, sheet_name="btc_predictions", index=False)
        btc_art.dataset.to_excel(writer, sheet_name="btc_dataset", index=False)
        live_df.to_excel(writer, sheet_name="live_signals", index=False)
        history_df.tail(5000).to_excel(writer, sheet_name="signal_history", index=False)
        updated_state.to_excel(writer, sheet_name="signal_state_store", index=False)
        notes = pd.DataFrame(
            [
                ("signal_granularity", "5m"),
                ("signal_type", "BTC long/flat 24/7"),
                ("entry_threshold", f"{BTC_ENTRY_THRESHOLD:.2f}"),
                ("exit_threshold", f"{BTC_EXIT_THRESHOLD:.2f}"),
                ("confirm_threshold", f"{BTC_CONFIRM_THRESHOLD:.2f}"),
                ("confirm_window", str(BTC_CONFIRM_WINDOW)),
                ("confirm_required", str(BTC_CONFIRM_REQUIRED)),
                ("label_horizon_bars", str(BTC_LABEL_HORIZON)),
                ("stop_loss", f"{BTC_STOP_LOSS:.4f}"),
                ("min_expected_edge", f"{BTC_MIN_EXPECTED_EDGE:.4f}"),
                ("stale_nq_minutes", f"{NQ_STALE_MINUTES:.0f}"),
                ("stale_nq_entry_threshold", f"{BTC_STALE_NQ_ENTRY_THRESHOLD:.2f}"),
                ("stale_nq_min_expected_edge", f"{BTC_STALE_NQ_MIN_EXPECTED_EDGE:.4f}"),
                ("context_driver", "Coinbase BTC-USD 5m as tradable; NQ=F 5m as macro context"),
                ("btc_data_source", "Coinbase Exchange public candles REST"),
            ],
            columns=["item", "value"],
        )
        notes.to_excel(writer, sheet_name="notes", index=False)

    live_df.to_csv(live_output_path, index=False)

    print(f"Wrote history CSV: {history_output_path}", flush=True)
    print(f"Wrote workbook: {output_path}", flush=True)
    print(f"Wrote live CSV: {live_output_path}", flush=True)
    print(f"Wrote state CSV: {state_output_path}", flush=True)

    rows_to_print = terminal_rows(live_df, state_store, args.print_mode)
    print("\nLive BTC 5m signal:", flush=True)
    if rows_to_print.empty:
        print("No opening or state-change event to print.", flush=True)
    else:
        for _, row in rows_to_print.iterrows():
            print(f"[btc] {row['signal_summary']}", flush=True)


if __name__ == "__main__":
    main()
