#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_nq_quant_model import (
    NQ_PATTERN,
    apply_wave_fib_overlay,
    archive_nq_contracts,
    build_archived_nq_main,
    build_roll_adjusted_nq_main,
    fetch_yahoo_daily,
    make_price_features,
    merge_asof_features,
    next_business_day,
    summarize_live_drivers,
)
from signal_ledger import (
    append_live_signals,
    latest_recommended_state,
    market_feedback,
    reconcile_signal_ledger,
)


STRATEGY_NAME = "own_quant_daily_v1"


@dataclass
class StrategyArtifacts:
    market: str
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    pipeline: Pipeline
    feature_cols: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a proprietary quant program using market data plus extracted signal context.")
    parser.add_argument("--events", default="wsquant_market_timeline.xlsx")
    parser.add_argument("--output", default="own_quant_program_results.xlsx")
    parser.add_argument("--live-output", default="own_quant_live_signals.csv")
    parser.add_argument("--ledger-output", default="own_quant_signal_ledger.csv")
    parser.add_argument("--cn-symbol", default="159941.SZ")
    parser.add_argument("--range", default="2y")
    parser.add_argument("--nq-archive-dir", default="nq_contract_archive")
    parser.add_argument("--strict-nq-archive", action="store_true")
    return parser.parse_args()


def _session_date_from_events(events: pd.DataFrame) -> pd.Series:
    us_mask = events["market"].eq("美股")
    session_date = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    session_date.loc[us_mask] = (
        events.loc[us_mask, "datetime"]
        - pd.to_timedelta((events.loc[us_mask, "datetime"].dt.hour < 12).astype(int), unit="D")
    ).dt.normalize()
    session_date.loc[~us_mask] = events.loc[~us_mask, "datetime"].dt.normalize()
    return pd.to_datetime(session_date)


def build_event_context(events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_excel(events_path, sheet_name="events")
    events["datetime"] = pd.to_datetime(events["datetime"])
    events["session_date"] = _session_date_from_events(events)
    events["nq_signal"] = pd.to_numeric(events["content"].astype(str).str.extract(NQ_PATTERN, expand=False), errors="coerce")
    events = events.sort_values(["market", "session_date", "datetime"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for (market, session_date), group in events.groupby(["market", "session_date"], sort=True):
        group = group.sort_values("datetime").reset_index(drop=True)
        nq_group = group.dropna(subset=["nq_signal"])
        nq_open = float(nq_group["nq_signal"].iloc[0]) if not nq_group.empty else np.nan
        nq_close = float(nq_group["nq_signal"].iloc[-1]) if not nq_group.empty else np.nan
        nq_high = float(nq_group["nq_signal"].max()) if not nq_group.empty else np.nan
        nq_low = float(nq_group["nq_signal"].min()) if not nq_group.empty else np.nan
        rows.append(
            {
                "market": market,
                "session_date": pd.Timestamp(session_date),
                "session_state": int(group["state"].iloc[-1]),
                "session_state_start": int(group["state"].iloc[0]),
                "session_event_count": int(len(group)),
                "session_change_count": int(group["state"].ne(group["state"].shift()).sum() - 1),
                "session_signal_buy_ratio": float((group["state"].astype(int) == 1).mean()),
                "nq_signal_open": nq_open,
                "nq_signal_close": nq_close,
                "nq_signal_ret": nq_close / nq_open - 1.0 if pd.notna(nq_open) and nq_open else np.nan,
                "nq_signal_range": (nq_high - nq_low) / nq_open if pd.notna(nq_open) and nq_open else np.nan,
            }
        )

    context = pd.DataFrame(rows).sort_values(["market", "session_date"]).reset_index(drop=True)
    context["context_state_mean_3"] = context.groupby("market")["session_state"].transform(lambda s: s.rolling(3).mean())
    context["context_state_mean_5"] = context.groupby("market")["session_state"].transform(lambda s: s.rolling(5).mean())
    context["context_change_mean_5"] = context.groupby("market")["session_change_count"].transform(lambda s: s.rolling(5).mean())
    context["context_nq_signal_ret_3"] = context.groupby("market")["nq_signal_ret"].transform(lambda s: s.rolling(3).mean())
    context["context_nq_signal_ret_5"] = context.groupby("market")["nq_signal_ret"].transform(lambda s: s.rolling(5).mean())
    context["context_nq_range_5"] = context.groupby("market")["nq_signal_range"].transform(lambda s: s.rolling(5).mean())
    us_context = context[context["market"].eq("美股")].drop(columns=["market"]).reset_index(drop=True)
    cn_context = context[context["market"].eq("场内")].drop(columns=["market"]).reset_index(drop=True)
    us_context = us_context.rename(columns={col: f"us_{col}" for col in us_context.columns if col != "session_date"})
    cn_context = cn_context.rename(columns={col: f"cn_{col}" for col in cn_context.columns if col != "session_date"})
    return us_context, cn_context


def make_trade_labels(raw: pd.DataFrame, prefix: str, style: str) -> pd.DataFrame:
    out = raw.copy().sort_values("date").reset_index(drop=True)
    close = out["adjclose"].astype(float)
    future_low_3 = pd.concat([out["low"].shift(-i) for i in range(1, 4)], axis=1).min(axis=1)
    future_high_3 = pd.concat([out["high"].shift(-i) for i in range(1, 4)], axis=1).max(axis=1)
    out["trade_ret"] = close.shift(-1) / close - 1.0
    out["trade_open_ret"] = out["open"].shift(-1) / close - 1.0
    out["fwd_ret_3"] = close.shift(-3) / close - 1.0
    out["fwd_low_dd_3"] = future_low_3 / close - 1.0
    out["fwd_high_up_3"] = future_high_3 / close - 1.0
    if style == "us":
        edge = 0.45 * out["trade_ret"] + 0.55 * out["fwd_ret_3"] - 1.25 * out["fwd_low_dd_3"].clip(upper=0).abs()
        target = ((edge > 0.0025) & (out["fwd_low_dd_3"] > -0.028)) | (
            (out["trade_open_ret"] > 0.0035) & (out["fwd_low_dd_3"] > -0.02)
        )
    else:
        edge = 0.5 * out["trade_ret"] + 0.5 * out["fwd_ret_3"] - 1.10 * out["fwd_low_dd_3"].clip(upper=0).abs()
        target = ((edge > 0.0012) & (out["fwd_low_dd_3"] > -0.022)) | (
            (out["fwd_high_up_3"] > 0.008) & (out["fwd_low_dd_3"] > -0.015)
        )
    out["target"] = target.astype(float)
    return out[["date", "trade_ret", "trade_open_ret", "fwd_ret_3", "fwd_low_dd_3", "fwd_high_up_3", "target"]]


def strategy_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "session_date",
        "cn_session_date",
        "us_session_date",
        "cn_feature_date",
        "us_feature_date",
        "us_available_date",
        "target",
        "trade_ret",
        "trade_open_ret",
        "fwd_ret_3",
        "fwd_low_dd_3",
        "fwd_high_up_3",
        "current_state",
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


def forward_fill_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    context_cols = [col for col in out.columns if col.startswith("us_") or col.startswith("cn_") or col.startswith("context_")]
    if context_cols:
        out[context_cols] = out[context_cols].ffill()
    return out


def build_us_dataset(
    qqq_daily: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_features: pd.DataFrame,
    us_context: pd.DataFrame,
    cn_context: pd.DataFrame,
) -> pd.DataFrame:
    labels = make_trade_labels(qqq_daily, "qqq", "us")
    base = qqq_features.merge(nq_features, on="date", how="left")
    base = base.merge(labels, on="date", how="left")
    base = merge_asof_features(base, cn_features.rename(columns={"date": "cn_feature_date"}), "cn_feature_date")
    base = base.merge(us_context, left_on="date", right_on="session_date", how="left")
    base = base.merge(
        cn_context.rename(columns={"session_date": "cn_session_date"}),
        left_on="date",
        right_on="cn_session_date",
        how="left",
    )
    base["nq_minus_qqq_ret_5"] = base["nq_ret_5"] - base["qqq_ret_5"]
    base["cn_minus_qqq_ret_5"] = base["cnetf_ret_5"] - base["qqq_ret_5"]
    base["context_state_spread"] = base["us_context_state_mean_5"] - base["cn_context_state_mean_5"]
    base["context_nq_ret_spread"] = base["us_context_nq_signal_ret_5"] - base["cn_context_nq_signal_ret_5"]
    base["current_state"] = np.where(base["qqq_fib_pos_13"] >= 0.55, 1, 0)
    return base.dropna(subset=["target", "trade_ret"]).sort_values("date").reset_index(drop=True)


def build_cn_dataset(
    cn_daily: pd.DataFrame,
    cn_features: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_context: pd.DataFrame,
    us_context: pd.DataFrame,
) -> pd.DataFrame:
    labels = make_trade_labels(cn_daily, "cnetf", "cn")
    us_pack = qqq_features.merge(nq_features, on="date", how="left")
    us_pack = us_pack.rename(columns={"date": "us_feature_date"})
    us_pack["us_available_date"] = us_pack["us_feature_date"] + pd.Timedelta(days=1)
    us_context_shifted = us_context.rename(columns={"session_date": "us_session_date"})
    us_context_shifted["us_available_date"] = us_context_shifted["us_session_date"] + pd.Timedelta(days=1)

    base = cn_features.merge(labels, on="date", how="left")
    base = merge_asof_features(base, us_pack, "us_available_date")
    base = merge_asof_features(base, us_context_shifted, "us_available_date")
    base = base.merge(cn_context, left_on="date", right_on="session_date", how="left")
    base["nq_minus_cnetf_ret_5"] = base["nq_ret_5"] - base["cnetf_ret_5"]
    base["qqq_minus_cnetf_ret_5"] = base["qqq_ret_5"] - base["cnetf_ret_5"]
    base["context_state_spread"] = base["cn_context_state_mean_5"] - base["us_context_state_mean_5"]
    base["context_nq_ret_spread"] = base["cn_context_nq_signal_ret_5"] - base["us_context_nq_signal_ret_5"]
    base["current_state"] = np.where(base["cnetf_fib_pos_13"] >= 0.58, 1, 0)
    return base.dropna(subset=["target", "trade_ret"]).sort_values("date").reset_index(drop=True)


def build_us_live_frame(
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_features: pd.DataFrame,
    us_context: pd.DataFrame,
    cn_context: pd.DataFrame,
) -> pd.DataFrame:
    base = qqq_features.merge(nq_features, on="date", how="left")
    base = merge_asof_features(base, cn_features.rename(columns={"date": "cn_feature_date"}), "cn_feature_date")
    base = base.merge(us_context, left_on="date", right_on="session_date", how="left")
    base = base.merge(cn_context.rename(columns={"session_date": "cn_session_date"}), left_on="date", right_on="cn_session_date", how="left")
    base["nq_minus_qqq_ret_5"] = base["nq_ret_5"] - base["qqq_ret_5"]
    base["cn_minus_qqq_ret_5"] = base["cnetf_ret_5"] - base["qqq_ret_5"]
    base["context_state_spread"] = base["us_context_state_mean_5"] - base["cn_context_state_mean_5"]
    base["context_nq_ret_spread"] = base["us_context_nq_signal_ret_5"] - base["cn_context_nq_signal_ret_5"]
    return forward_fill_context(base)


def build_cn_live_frame(
    cn_features: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_context: pd.DataFrame,
    us_context: pd.DataFrame,
) -> pd.DataFrame:
    us_pack = qqq_features.merge(nq_features, on="date", how="left")
    us_pack = us_pack.rename(columns={"date": "us_feature_date"})
    us_pack["us_available_date"] = us_pack["us_feature_date"] + pd.Timedelta(days=1)
    us_context_shifted = us_context.rename(columns={"session_date": "us_session_date"})
    us_context_shifted["us_available_date"] = us_context_shifted["us_session_date"] + pd.Timedelta(days=1)
    base = cn_features.copy()
    base = merge_asof_features(base, us_pack, "us_available_date")
    base = merge_asof_features(base, us_context_shifted, "us_available_date")
    base = base.merge(cn_context, left_on="date", right_on="session_date", how="left")
    base["nq_minus_cnetf_ret_5"] = base["nq_ret_5"] - base["cnetf_ret_5"]
    base["qqq_minus_cnetf_ret_5"] = base["qqq_ret_5"] - base["cnetf_ret_5"]
    base["context_state_spread"] = base["cn_context_state_mean_5"] - base["us_context_state_mean_5"]
    base["context_nq_ret_spread"] = base["cn_context_nq_signal_ret_5"] - base["us_context_nq_signal_ret_5"]
    return forward_fill_context(base)


def apply_proprietary_overlay(probability: float, row: pd.Series | pd.DataFrame, market: str) -> tuple[float, str]:
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    adjusted, reason = apply_wave_fib_overlay(probability, row, market)
    if market == "美股":
        qqq_fib_13 = float(row.get("qqq_fib_pos_13", np.nan))
        qqq_ew = float(row.get("qqq_ew_thrust_5", np.nan))
        qqq_ret_20 = float(row.get("qqq_ret_20", np.nan))
        nq_ret_20 = float(row.get("nq_ret_20", np.nan))
        nq_ew = float(row.get("nq_ew_thrust_5", np.nan))
        event_ret = float(row.get("us_context_nq_signal_ret_5", np.nan))
        if all(pd.notna(x) for x in (qqq_fib_13, qqq_ew, qqq_ret_20, nq_ret_20)):
            if qqq_fib_13 > 0.88 and qqq_ret_20 < 0 and nq_ret_20 < 0:
                return min(adjusted, 0.18), "us_countertrend_exhaustion"
        if all(pd.notna(x) for x in (qqq_fib_13, qqq_ew, nq_ew, event_ret)):
            if qqq_fib_13 < 0.14 and qqq_ew > -0.32 and nq_ew > -0.45 and event_ret > -0.002:
                return max(adjusted, 0.58), "us_deep_oversold_reversal"
            if qqq_fib_13 > 0.62 and qqq_ew < 0.10:
                return min(adjusted, 0.30), "us_late_swing_exit"
        return adjusted, reason

    cnetf_fib_13 = float(row.get("cnetf_fib_pos_13", np.nan))
    cnetf_ew = float(row.get("cnetf_ew_thrust_5", np.nan))
    event_ret = float(row.get("cn_context_nq_signal_ret_5", np.nan))
    if all(pd.notna(x) for x in (cnetf_fib_13, cnetf_ew, event_ret)):
        if cnetf_fib_13 < 0.22 and cnetf_ew > -0.45 and event_ret > -0.003:
            return max(adjusted, 0.60), "cn_deep_oversold_reversal"
        if cnetf_fib_13 > 0.55 and cnetf_ew < 0.05:
            return min(adjusted, 0.33), "cn_swing_distribution"
    return adjusted, reason


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame([{"metric": "rows", "value": 0.0}])
    returns = predictions["strategy_ret"].fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    win_rate = float((returns > 0).mean())
    out = [
        {"metric": "rows", "value": float(len(predictions))},
        {"metric": "cum_return", "value": float(equity.iloc[-1] - 1.0)},
        {"metric": "avg_daily_return", "value": float(returns.mean())},
        {"metric": "win_rate", "value": win_rate},
        {"metric": "max_drawdown", "value": float(drawdown.min())},
        {"metric": "avg_position", "value": float(predictions["position"].mean())},
        {"metric": "avg_probability", "value": float(predictions["probability"].mean())},
    ]
    if returns.std(ddof=0) > 0:
        out.append({"metric": "sharpe", "value": float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))})
    return pd.DataFrame(out)


def walk_forward_strategy(dataset: pd.DataFrame, market: str, stop_loss: float, entry_threshold: float, exit_threshold: float) -> StrategyArtifacts:
    dataset = dataset.copy().reset_index(drop=True)
    feature_cols = strategy_feature_columns(dataset)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    min_train = max(60, int(len(dataset) * 0.45))
    rows: list[dict[str, object]] = []
    current_position = 0
    cooldown = 0

    for i in range(min_train, len(dataset)):
        train = dataset.iloc[:i].dropna(subset=feature_cols + ["target"]).copy()
        if train.empty or train["target"].nunique() < 2:
            continue
        pipeline.fit(train[feature_cols], train["target"].astype(int))
        row = dataset.iloc[[i]]
        raw_probability = float(pipeline.predict_proba(row[feature_cols])[0, 1])
        probability, overlay_reason = apply_proprietary_overlay(raw_probability, row, market)

        next_position = current_position
        if cooldown > 0:
            next_position = 0
            cooldown -= 1
        elif current_position == 0 and probability >= entry_threshold:
            next_position = 1
        elif current_position == 1 and probability <= exit_threshold:
            next_position = 0

        trade_ret = float(row["trade_ret"].iloc[0])
        low_ret = float(row["fwd_low_dd_3"].iloc[0]) if pd.notna(row["fwd_low_dd_3"].iloc[0]) else np.nan
        realized_ret = 0.0
        stop_triggered = False
        position_for_return = next_position
        if next_position == 1:
            realized_ret = trade_ret
            if pd.notna(low_ret) and low_ret < -stop_loss:
                realized_ret = -stop_loss
                stop_triggered = True
                next_position = 0
                cooldown = 1

        rows.append(
            {
                "feature_date": row["date"].iloc[0],
                "probability": probability,
                "raw_probability": raw_probability,
                "wave_overlay": overlay_reason,
                "position": next_position,
                "prior_position": current_position,
                "trade_ret": trade_ret,
                "strategy_ret": position_for_return * realized_ret,
                "stop_triggered": int(stop_triggered),
            }
        )
        current_position = next_position

    preds = pd.DataFrame(rows)
    metrics = evaluate_predictions(preds)
    if not dataset.dropna(subset=feature_cols + ["target"]).empty:
        pipeline.fit(dataset.dropna(subset=feature_cols + ["target"])[feature_cols], dataset.dropna(subset=feature_cols + ["target"])["target"].astype(int))
    return StrategyArtifacts(
        market=market,
        dataset=dataset,
        predictions=preds,
        metrics=metrics,
        pipeline=pipeline,
        feature_cols=feature_cols,
    )


def build_live_row(dataset: pd.DataFrame) -> pd.DataFrame:
    return dataset.dropna().sort_values("date").tail(1)


def state_action(current_state: int, recommended_state: int) -> str:
    if current_state == 0 and recommended_state == 1:
        return "买入"
    if current_state == 1 and recommended_state == 0:
        return "卖出"
    if current_state == 1 and recommended_state == 1:
        return "持有"
    return "空仓"


def make_live_signal(
    art: StrategyArtifacts,
    live_frame: pd.DataFrame,
    market: str,
    symbol: str,
    price_col: str,
    current_state: int,
    entry_threshold: float,
    feedback: dict[str, int],
) -> dict[str, object]:
    latest = live_frame.dropna(subset=art.feature_cols).sort_values("date").tail(1)
    raw_probability = float(art.pipeline.predict_proba(latest[art.feature_cols])[0, 1])
    probability, overlay_reason = apply_proprietary_overlay(raw_probability, latest, market)
    position = int(probability >= entry_threshold)
    if feedback.get("cooldown_suggested", 0) > 0:
        position = 0
        overlay_reason = f"{overlay_reason}+risk_cooldown" if overlay_reason != "none" else "risk_cooldown"
    return {
        "market": market,
        "feature_date": pd.Timestamp(latest["date"].iloc[0]),
        "next_session_date": next_business_day(latest["date"].iloc[0]),
        "current_state": int(current_state),
        "raw_probability": raw_probability,
        "probability_long": probability,
        "recommended_state": position,
        "action": state_action(current_state, position),
        "recent_wrong_streak": int(feedback.get("recent_wrong_streak", 0)),
        "cooldown_suggested": int(feedback.get("cooldown_suggested", 0)),
        "wave_overlay": overlay_reason,
        "symbol": symbol,
        "signal_price": float(latest[price_col].iloc[0]),
        "top_drivers": summarize_live_drivers(art.pipeline, latest, art.feature_cols),
    }


def main() -> None:
    args = parse_args()
    events_path = Path(args.events).resolve()
    output_path = Path(args.output).resolve()
    live_output_path = Path(args.live_output).resolve()
    ledger_output_path = Path(args.ledger_output).resolve()
    archive_dir = Path(args.nq_archive_dir).resolve()

    us_context, cn_context = build_event_context(events_path)
    qqq_daily = fetch_yahoo_daily("QQQ", args.range)
    cn_daily = fetch_yahoo_daily(args.cn_symbol, args.range)
    archive_nq_status = archive_nq_contracts(
        archive_dir,
        range(pd.Timestamp.today().year - 1, pd.Timestamp.today().year + 2),
        date_range=args.range,
    )
    archived_nq_daily, _, archive_complete = build_archived_nq_main(
        archive_dir,
        qqq_daily["date"].min(),
        qqq_daily["date"].max(),
    )
    if archive_complete:
        nq_daily = archived_nq_daily
        nq_source = "single-month archive"
    else:
        if args.strict_nq_archive:
            raise RuntimeError("NQ archive incomplete for own quant program.")
        nq_daily, _ = build_roll_adjusted_nq_main(args.range)
        nq_source = "roll-adjusted NQ=F fallback"

    qqq_features = make_price_features(qqq_daily, "qqq")
    cn_features = make_price_features(cn_daily, "cnetf")
    nq_features = make_price_features(nq_daily, "nq")

    us_dataset = build_us_dataset(qqq_daily, qqq_features, nq_features, cn_features, us_context, cn_context)
    cn_dataset = build_cn_dataset(cn_daily, cn_features, qqq_features, nq_features, cn_context, us_context)
    us_live_frame = build_us_live_frame(qqq_features, nq_features, cn_features, us_context, cn_context)
    cn_live_frame = build_cn_live_frame(cn_features, qqq_features, nq_features, cn_context, us_context)
    us_feature_date = pd.Timestamp(us_live_frame.dropna(subset=["qqq_close"]).sort_values("date").iloc[-1]["date"]).normalize()
    cn_feature_date = pd.Timestamp(cn_live_frame.dropna(subset=["cnetf_close"]).sort_values("date").iloc[-1]["date"]).normalize()

    us_entry_threshold = 0.60
    cn_entry_threshold = 0.58
    us_art = walk_forward_strategy(us_dataset, "美股", stop_loss=0.018, entry_threshold=us_entry_threshold, exit_threshold=0.40)
    cn_art = walk_forward_strategy(cn_dataset, "场内", stop_loss=0.012, entry_threshold=cn_entry_threshold, exit_threshold=0.40)

    ledger = reconcile_signal_ledger(
        ledger_output_path,
        {"QQQ": qqq_daily, args.cn_symbol: cn_daily},
        STRATEGY_NAME,
    )
    us_current_state = latest_recommended_state(
        ledger,
        STRATEGY_NAME,
        "美股",
        int(us_art.predictions["position"].iloc[-1]) if not us_art.predictions.empty else 0,
        as_of_date=us_feature_date,
    )
    cn_current_state = latest_recommended_state(
        ledger,
        STRATEGY_NAME,
        "场内",
        int(cn_art.predictions["position"].iloc[-1]) if not cn_art.predictions.empty else 0,
        as_of_date=cn_feature_date,
    )
    us_feedback = market_feedback(ledger, STRATEGY_NAME, "美股")
    cn_feedback = market_feedback(ledger, STRATEGY_NAME, "场内")

    live_rows = [
        make_live_signal(
            us_art,
            us_live_frame,
            "美股",
            "QQQ",
            "qqq_close",
            us_current_state,
            us_entry_threshold,
            us_feedback,
        ),
        make_live_signal(
            cn_art,
            cn_live_frame,
            "场内",
            args.cn_symbol,
            "cnetf_close",
            cn_current_state,
            cn_entry_threshold,
            cn_feedback,
        ),
    ]
    live_df = pd.DataFrame(live_rows)
    live_df.to_csv(live_output_path, index=False)
    ledger = append_live_signals(ledger_output_path, live_df, STRATEGY_NAME)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        us_art.dataset.to_excel(writer, sheet_name="us_dataset", index=False)
        cn_art.dataset.to_excel(writer, sheet_name="cn_dataset", index=False)
        us_art.predictions.to_excel(writer, sheet_name="us_predictions", index=False)
        cn_art.predictions.to_excel(writer, sheet_name="cn_predictions", index=False)
        us_art.metrics.to_excel(writer, sheet_name="metrics", index=False)
        cn_art.metrics.to_excel(writer, sheet_name="metrics", index=False, startrow=len(us_art.metrics) + 3)
        us_context.to_excel(writer, sheet_name="us_context", index=False)
        cn_context.to_excel(writer, sheet_name="cn_context", index=False)
        archive_nq_status.to_excel(writer, sheet_name="nq_archive_status", index=False)
        live_df.to_excel(writer, sheet_name="live_signals", index=False)
        ledger.to_excel(writer, sheet_name="signal_ledger", index=False)
        notes = pd.DataFrame(
            [
                ("program_style", "Hybrid proprietary quant: event-context + wave/fibonacci + risk-adjusted price targets."),
                ("data_usage", "Uses extracted signal sessions as context features, not as direct target labels."),
                ("risk_control", "Long/flat only, hysteresis thresholds, stop-loss, wave/fibonacci overlay, and ledger-based cooldown after wrong streaks."),
                ("signal_reuse_rule", "Prior predictions are stored for execution state and ex-post feedback, but are not fed back as model features."),
                ("signal_ledger", str(ledger_output_path)),
                ("nq_source", nq_source),
            ],
            columns=["item", "value"],
        )
        notes.to_excel(writer, sheet_name="notes", index=False)

    print(f"Wrote workbook: {output_path}")
    print(f"Wrote live CSV: {live_output_path}")
    print(f"Wrote ledger CSV: {ledger_output_path}")
    print("\nLive signals:")
    print(live_df.to_string(index=False))


if __name__ == "__main__":
    main()
