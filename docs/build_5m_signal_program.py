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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_nq_quant_model import (
    apply_wave_fib_overlay,
    fetch_sina_cn_intraday,
    fetch_yahoo_chart,
    make_price_features,
    merge_asof_features,
    summarize_live_drivers,
)

US_TZ = ZoneInfo("America/New_York")
CN_TZ = ZoneInfo("Asia/Shanghai")

US_ENTRY_THRESHOLD = 0.68
US_EXIT_THRESHOLD = 0.32
US_ENTRY_CONFIRM_THRESHOLD = 0.62
US_ENTRY_CONFIRM_WINDOW = 3
US_ENTRY_CONFIRM_REQUIRED = 2
CN_ENTRY_THRESHOLD = 0.55
CN_EXIT_THRESHOLD = 0.40
CN_ENTRY_CONFIRM_THRESHOLD = 0.52
CN_ENTRY_CONFIRM_WINDOW = 3
CN_ENTRY_CONFIRM_REQUIRED = 2
CN_OPEN_ENTRY_THRESHOLD = 0.66
CN_OPEN_CONFIRM_THRESHOLD = 0.58
CN_OPEN_CONFIRM_REQUIRED = 3
CN_OPEN_MIN_BAR_INDEX = 4
CN_MIN_EXPECTED_EDGE = 0.0012
CN_OPEN_MIN_EXPECTED_EDGE = 0.0016
EDGE_LOOKBACK_ROWS = 180
EDGE_BANDWIDTH = 0.08
EDGE_MIN_NEIGHBORS = 18
CN_REBOUND_MIN_RAW_PROBABILITY = 0.25
CN_REBOUND_FLOOR = 0.56


@dataclass
class FiveMinArtifacts:
    market: str
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    pipeline: Pipeline
    feature_cols: list[str]


STATE_COLUMNS = [
    "market",
    "symbol",
    "updated_at",
    "feature_bar_time",
    "current_state",
    "recommended_state",
    "action",
    "raw_probability",
    "probability_long",
    "wave_overlay",
    "signal_price",
    "market_phase",
    "position_since",
    "position_since_exact",
    "last_action_time",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5m signal-level models for QQQ and 159941.SZ.")
    parser.add_argument("--output", default="five_minute_signal_results.xlsx")
    parser.add_argument("--live-output", default="five_minute_live_signals.csv")
    parser.add_argument("--state-output", default="five_minute_signal_state.csv")
    parser.add_argument("--history-output", default="five_minute_signal_history.csv")
    parser.add_argument("--cn-symbol", default="159941.SZ")
    parser.add_argument("--range", default="20d")
    parser.add_argument("--intraday-bars", type=int, default=5000)
    parser.add_argument("--market", default="all", choices=["all", "cn", "us"])
    parser.add_argument("--print-mode", default="event", choices=["event", "all", "none"])
    return parser.parse_args()


def _now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)


def load_state_store(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=STATE_COLUMNS)
    state = pd.read_csv(path)
    for col in ["updated_at", "feature_bar_time", "position_since", "last_action_time"]:
        if col in state.columns:
            state[col] = pd.to_datetime(state[col], errors="coerce")
    if "market" in state.columns and "feature_bar_time" in state.columns and "updated_at" in state.columns:
        run_cutoff = state["updated_at"].dt.floor("5min")
        invalid_future_cn = (
            state["market"].eq("场内")
            & state["feature_bar_time"].notna()
            & run_cutoff.notna()
            & (state["feature_bar_time"] > run_cutoff)
        )
        state = state.loc[~invalid_future_cn].copy()
    for col in STATE_COLUMNS:
        if col not in state.columns:
            state[col] = pd.NA
    return state[STATE_COLUMNS].copy()


def save_state_store(path: Path, state: pd.DataFrame) -> pd.DataFrame:
    out = state.copy()
    for col in STATE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[STATE_COLUMNS].sort_values(["market", "symbol", "updated_at"]).reset_index(drop=True)
    out = out.drop_duplicates(["market", "symbol"], keep="last")
    out.to_csv(path, index=False)
    return out


def filter_completed_cn_bars(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    current_time = _now_local() if now is None else pd.Timestamp(now)
    completed_bar_time = current_time.floor("5min")
    dt = pd.to_datetime(frame["datetime"])
    return frame.loc[dt <= completed_bar_time].copy()


def _coerce_history_datetimes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["updated_at", "feature_bar_time", "position_since", "last_action_time", "trade_date", "run_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _normalize_signal_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    history = _coerce_history_datetimes(frame)
    if "run_at" not in history.columns:
        history["run_at"] = pd.NaT
    if "updated_at" in history.columns:
        missing_run_at = history["run_at"].isna()
        history.loc[missing_run_at, "run_at"] = history.loc[missing_run_at, "updated_at"]
    if "market" in history.columns and "feature_bar_time" in history.columns:
        run_cutoff = history["run_at"].dt.floor("5min")
        invalid_future_cn = (
            history["market"].eq("场内")
            & history["feature_bar_time"].notna()
            & run_cutoff.notna()
            & (history["feature_bar_time"] > run_cutoff)
        )
        history = history.loc[~invalid_future_cn].copy()
    dedup_keys = [
        col
        for col in ["market", "symbol", "run_at", "updated_at", "feature_bar_time", "recommended_state", "action"]
        if col in history.columns
    ]
    if dedup_keys:
        history = history.drop_duplicates(subset=dedup_keys, keep="last").copy()
    history = history.sort_values(["run_at", "updated_at", "market", "symbol"]).reset_index(drop=True)
    unique_runs = pd.Series(history["run_at"].dropna().sort_values().unique())
    run_map = {ts: idx + 1 for idx, ts in enumerate(unique_runs)}
    history["run_seq"] = history["run_at"].map(run_map)

    def _mark_changes(group: pd.DataFrame) -> pd.DataFrame:
        out = group.sort_values(["run_at", "updated_at"]).copy()
        prev_state = pd.to_numeric(out["recommended_state"], errors="coerce").shift(1)
        curr_state = pd.to_numeric(out["recommended_state"], errors="coerce")
        action = out["action"].astype("string")
        out["signal_changed"] = (
            action.isin(["买入", "卖出"])
            | (prev_state.notna() & curr_state.notna() & (prev_state != curr_state))
        ).astype(int)
        return out

    history = history.groupby(["market", "symbol"], group_keys=False, dropna=False).apply(_mark_changes)
    return history.reset_index(drop=True)


def append_signal_history(path: Path, live_df: pd.DataFrame) -> pd.DataFrame:
    if live_df.empty:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    current = live_df.copy()
    if path.exists():
        existing = pd.read_csv(path)
        for col in set(existing.columns).union(current.columns):
            if col not in existing.columns:
                existing[col] = pd.NA
            if col not in current.columns:
                current[col] = pd.NA
        existing = existing.astype("object")
        current = current.astype("object")
        history = pd.concat([existing, current], ignore_index=True, sort=False)
    else:
        history = current
    history = _normalize_signal_history(history)
    history_to_write = history.copy()
    for col in ["updated_at", "feature_bar_time", "position_since", "last_action_time", "trade_date", "run_at"]:
        if col in history_to_write.columns:
            history_to_write[col] = history_to_write[col].astype("string")
    history_to_write.to_csv(path, index=False)
    return history_to_write


def action_from_states(current_state: int, recommended_state: int) -> str:
    if current_state == 0 and recommended_state == 1:
        return "买入"
    if current_state == 1 and recommended_state == 0:
        return "卖出"
    if current_state == 1 and recommended_state == 1:
        return "持有"
    return "空仓"


def signal_result_text(recommended_state: int) -> str:
    return "100%" if int(recommended_state) == 1 else "0%"


def price_format(symbol: str, value: float) -> str:
    precision = 4 if symbol.endswith(".SZ") or symbol.endswith(".SS") else 2
    return f"{float(value):.{precision}f}"


def recent_session_probabilities(
    probabilities: list[float],
    trade_dates: list[pd.Timestamp],
    current_trade_date: pd.Timestamp,
    window: int,
) -> list[float]:
    current_day = pd.Timestamp(current_trade_date).normalize()
    paired = [
        prob
        for prob, trade_date in zip(probabilities, trade_dates)
        if pd.Timestamp(trade_date).normalize() == current_day
    ]
    return paired[-window:]


def entry_confirmation_stats(
    market: str,
    recent_probabilities: list[float],
) -> tuple[float, int]:
    if not recent_probabilities:
        return np.nan, 0
    if market == "美股":
        smooth_probability = float(np.mean(recent_probabilities[-US_ENTRY_CONFIRM_WINDOW:]))
        confirm_count = int(sum(prob >= US_ENTRY_CONFIRM_THRESHOLD for prob in recent_probabilities[-US_ENTRY_CONFIRM_WINDOW:]))
        return smooth_probability, confirm_count
    if market == "场内":
        smooth_probability = float(np.mean(recent_probabilities[-CN_ENTRY_CONFIRM_WINDOW:]))
        confirm_count = int(sum(prob >= CN_ENTRY_CONFIRM_THRESHOLD for prob in recent_probabilities[-CN_ENTRY_CONFIRM_WINDOW:]))
        return smooth_probability, confirm_count
    smooth_probability = float(recent_probabilities[-1])
    confirm_count = int(recent_probabilities[-1] >= 0.0)
    return smooth_probability, confirm_count


def is_cn_opening_window(feature_bar_time: pd.Timestamp | None) -> bool:
    if feature_bar_time is None or pd.isna(feature_bar_time):
        return False
    current_time = pd.Timestamp(feature_bar_time).time()
    return time(9, 35) <= current_time <= time(10, 0)


def estimate_expected_edge(
    prediction_history: pd.DataFrame,
    probability: float,
) -> float:
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


def entry_signal_allowed(
    market: str,
    probability: float,
    recent_probabilities: list[float],
    entry_threshold: float,
    *,
    feature_bar_time: pd.Timestamp | None = None,
    bar_index: int | None = None,
    expected_edge: float = np.nan,
) -> tuple[bool, float, int]:
    smooth_probability, confirm_count = entry_confirmation_stats(market, recent_probabilities)
    if market == "美股":
        entry_allowed = (
            probability >= entry_threshold
            and smooth_probability >= US_ENTRY_CONFIRM_THRESHOLD
            and confirm_count >= US_ENTRY_CONFIRM_REQUIRED
        )
        return entry_allowed, smooth_probability, confirm_count
    if market == "场内":
        entry_allowed = (
            probability >= entry_threshold
            and smooth_probability >= CN_ENTRY_CONFIRM_THRESHOLD
            and confirm_count >= CN_ENTRY_CONFIRM_REQUIRED
        )
        if pd.notna(expected_edge):
            entry_allowed = entry_allowed and expected_edge >= CN_MIN_EXPECTED_EDGE
        if is_cn_opening_window(feature_bar_time):
            entry_allowed = (
                entry_allowed
                and (bar_index is not None and int(bar_index) >= CN_OPEN_MIN_BAR_INDEX)
                and probability >= CN_OPEN_ENTRY_THRESHOLD
                and smooth_probability >= CN_OPEN_CONFIRM_THRESHOLD
                and confirm_count >= CN_OPEN_CONFIRM_REQUIRED
            )
            if pd.notna(expected_edge):
                entry_allowed = entry_allowed and expected_edge >= CN_OPEN_MIN_EXPECTED_EDGE
        return entry_allowed, smooth_probability, confirm_count
    entry_allowed = probability >= entry_threshold
    return entry_allowed, smooth_probability, confirm_count


def is_cn_midday_break(now: pd.Timestamp | None = None) -> bool:
    local_now = _now_local() if now is None else pd.Timestamp(now)
    current_time = local_now.time()
    return time(11, 30) <= current_time < time(13, 0)


def build_signal_summary(
    market: str,
    symbol: str,
    nqmain_rt_value: float,
    signal_value: float,
    recommended_state: int,
    feature_bar_time: pd.Timestamp,
    action: str,
    position_since: pd.Timestamp | None = None,
    position_since_exact: int | None = None,
) -> str:
    label = "QQQ" if market == "美股" else symbol
    time_text = pd.Timestamp(feature_bar_time).strftime("%Y-%m-%d %H:%M:%S")
    summary = (
        f"DATA_TIME({time_text}) "
        f"NQMAIN_RT({price_format('NQ=F', nqmain_rt_value)}) "
        f"{label}({price_format(symbol, signal_value)}) -> {signal_result_text(recommended_state)}"
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


def filter_us_regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(frame["datetime"])
    tod_min = dt.dt.hour * 60 + dt.dt.minute
    regular_open = 9 * 60 + 30
    regular_close = 16 * 60
    return frame.loc[tod_min.between(regular_open, regular_close)].copy()


def fetch_intraday(symbol: str, market: str, bars: int, date_range: str) -> pd.DataFrame:
    if market == "场内":
        frame = fetch_sina_cn_intraday(symbol, interval_minutes=5, datalen=bars)
        frame = filter_completed_cn_bars(frame)
    else:
        frame, _ = fetch_yahoo_chart(symbol, interval="5m", date_range=date_range, include_prepost=True, include_adjusted=False)
        frame = frame[["datetime", "date", "open", "high", "low", "close", "volume", "symbol"]]
        frame = filter_us_regular_session(frame)
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
    frame["range_pos_so_far"] = (frame["close"] - frame["session_low_so_far"]) / frame["session_range_so_far"].replace(0, np.nan)
    frame["range_pos_so_far"] = frame["range_pos_so_far"].clip(lower=0.0, upper=1.0)
    return frame


def fetch_nq_intraday(date_range: str, bars: int) -> pd.DataFrame:
    frame, _ = fetch_yahoo_chart("NQ=F", interval="5m", date_range=date_range, include_prepost=True, include_adjusted=False)
    frame = frame[["datetime", "date", "open", "high", "low", "close", "volume", "symbol"]]
    frame = frame.dropna(subset=["datetime", "open", "close"]).sort_values("datetime").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["datetime"])
    frame["adjclose"] = frame["close"].astype(float)
    return frame


def convert_intraday_timezone(frame: pd.DataFrame, source_tz: ZoneInfo, target_tz: ZoneInfo) -> pd.DataFrame:
    converted = frame.copy()
    converted["datetime"] = (
        pd.to_datetime(converted["datetime"])
        .dt.tz_localize(source_tz)
        .dt.tz_convert(target_tz)
        .dt.tz_localize(None)
    )
    converted["date"] = pd.to_datetime(converted["datetime"])
    return converted.sort_values("datetime").reset_index(drop=True)


def latest_nqmain_rt_value(nq_raw: pd.DataFrame) -> float:
    frame = nq_raw.copy().sort_values("datetime").reset_index(drop=True)
    return float(frame.iloc[-1]["close"])


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


def make_intraday_dataset(
    raw: pd.DataFrame,
    nq_features: pd.DataFrame,
    prefix: str,
    market: str,
    label_horizon: int,
    stop_loss: float,
    profit_threshold: float,
) -> pd.DataFrame:
    base = make_intraday_feature_frame(raw, prefix)
    base = merge_asof_features(base, nq_features, "date")
    base = base.sort_values("datetime").reset_index(drop=True)

    close = base[f"{prefix}_close"].astype(float)
    future_low = pd.concat([base["low"].shift(-i) for i in range(1, label_horizon + 1)], axis=1).min(axis=1)
    future_high = pd.concat([base["high"].shift(-i) for i in range(1, label_horizon + 1)], axis=1).max(axis=1)
    trade_ret_1 = close.shift(-1) / close - 1.0
    trade_ret_h = close.shift(-label_horizon) / close - 1.0
    edge = 0.6 * trade_ret_1 + 0.4 * trade_ret_h - 1.15 * future_low.sub(close).clip(upper=0).abs().div(close)
    fwd_low_dd_h = future_low / close - 1.0
    base = base.assign(
        trade_ret_1=trade_ret_1,
        trade_ret_h=trade_ret_h,
        fwd_low_dd_h=fwd_low_dd_h,
        fwd_high_up_h=future_high / close - 1.0,
        target=((edge > profit_threshold) & (fwd_low_dd_h > -stop_loss)).astype(float),
        current_state=0,
        target_time=base["datetime"].shift(-1),
    )
    return base.dropna(subset=["target", "trade_ret_1"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "datetime",
        "date",
        "target_time",
        "target",
        "trade_ret_1",
        "trade_ret_h",
        "fwd_low_dd_h",
        "fwd_high_up_h",
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


def apply_5m_overlay(probability: float, row: pd.Series, market: str) -> tuple[float, str]:
    adjusted, reason = apply_wave_fib_overlay(probability, row, market)
    if market == "美股":
        qqq_ew = float(row.get("qqq_ew_thrust_5", np.nan))
        nq_ew = float(row.get("nq_ew_thrust_5", np.nan))
        qqq_fib = float(row.get("qqq_fib_pos_13", np.nan))
        if all(pd.notna(x) for x in (qqq_ew, nq_ew, qqq_fib)):
            if qqq_fib > 0.85 and qqq_ew < 0 and nq_ew < 0:
                return min(adjusted, 0.18), "us_intraday_exhaustion"
            if qqq_fib < 0.18 and qqq_ew > -0.2 and nq_ew > -0.25:
                return max(adjusted, 0.58), "us_intraday_rebound"
        return adjusted, reason

    cnetf_ew = float(row.get("cnetf_ew_thrust_5", np.nan))
    nq_ew = float(row.get("nq_ew_thrust_5", np.nan))
    cnetf_fib = float(row.get("cnetf_fib_pos_13", np.nan))
    if all(pd.notna(x) for x in (cnetf_ew, nq_ew, cnetf_fib)):
        if cnetf_fib > 0.78 and cnetf_ew < 0 and nq_ew < 0:
            return min(adjusted, 0.22), "cn_intraday_distribution"
        if cnetf_fib < 0.20 and cnetf_ew > -0.25 and nq_ew > -0.30:
            if probability < CN_REBOUND_MIN_RAW_PROBABILITY:
                return adjusted, "cn_oversold_rebound_watch"
            return max(adjusted, CN_REBOUND_FLOOR), "cn_intraday_rebound"
    return adjusted, reason


def walk_forward_5m(dataset: pd.DataFrame, market: str, entry_threshold: float, exit_threshold: float, stop_loss: float) -> FiveMinArtifacts:
    dataset = dataset.copy().reset_index(drop=True)
    cols = feature_columns(dataset)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    min_train = max(80, int(len(dataset) * 0.4))
    eval_stride = 10
    rows: list[dict[str, object]] = []
    current_position = 0
    cooldown = 0
    probability_history: list[float] = []
    probability_trade_dates: list[pd.Timestamp] = []

    for i in range(min_train, len(dataset), eval_stride):
        train = dataset.iloc[:i].dropna(subset=cols + ["target"])
        if train.empty or train["target"].nunique() < 2:
            continue
        pipeline.fit(train[cols], train["target"].astype(int))
        row = dataset.iloc[[i]]
        trade_date = pd.Timestamp(row["trade_date"].iloc[0]).normalize()
        raw_probability = float(pipeline.predict_proba(row[cols])[0, 1])
        probability, overlay_reason = apply_5m_overlay(raw_probability, row.iloc[0], market)
        history_for_edge = pd.DataFrame(rows)
        expected_edge = estimate_expected_edge(history_for_edge, probability)
        recent_probabilities = recent_session_probabilities(
            probability_history + [probability],
            probability_trade_dates + [trade_date],
            trade_date,
            max(US_ENTRY_CONFIRM_WINDOW, CN_ENTRY_CONFIRM_WINDOW),
        )
        entry_allowed, smooth_probability, confirm_count = entry_signal_allowed(
            "美股" if market.startswith("美股") else "场内",
            probability,
            recent_probabilities,
            entry_threshold,
            feature_bar_time=pd.Timestamp(row["datetime"].iloc[0]),
            bar_index=int(row["bar_index"].iloc[0]),
            expected_edge=expected_edge,
        )

        next_position = current_position
        if cooldown > 0:
            next_position = 0
            cooldown -= 1
        elif current_position == 0 and entry_allowed:
            next_position = 1
        elif current_position == 1 and probability <= exit_threshold:
            next_position = 0

        next_ret = float(row["trade_ret_1"].iloc[0])
        trade_ret_h = float(row["trade_ret_h"].iloc[0]) if pd.notna(row["trade_ret_h"].iloc[0]) else 0.0
        low_dd = float(row["fwd_low_dd_h"].iloc[0]) if pd.notna(row["fwd_low_dd_h"].iloc[0]) else np.nan
        edge_proxy_realized = 0.6 * next_ret + 0.4 * trade_ret_h - 1.15 * abs(min(low_dd, 0.0))
        realized_ret = 0.0
        stop_triggered = False
        position_for_return = next_position
        if next_position == 1:
            realized_ret = next_ret
            if pd.notna(low_dd) and low_dd < -stop_loss:
                realized_ret = -stop_loss
                stop_triggered = True
                next_position = 0
                cooldown = 1

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
        probability_trade_dates.append(trade_date)
        current_position = next_position

    preds = pd.DataFrame(rows)
    if preds.empty:
        metrics = pd.DataFrame([{"market": market, "metric": "rows", "value": 0.0}])
    else:
        returns = preds["strategy_ret"].fillna(0.0)
        equity = (1.0 + returns).cumprod()
        dd = equity / equity.cummax() - 1.0
        metrics = pd.DataFrame(
            [
                {"market": market, "metric": "rows", "value": float(len(preds))},
                {"market": market, "metric": "cum_return", "value": float(equity.iloc[-1] - 1.0)},
                {"market": market, "metric": "avg_return", "value": float(returns.mean())},
                {"market": market, "metric": "win_rate", "value": float((returns > 0).mean())},
                {"market": market, "metric": "max_drawdown", "value": float(dd.min())},
                {"market": market, "metric": "avg_position", "value": float(preds["position"].mean())},
            ]
        )

    pipeline.fit(dataset[cols], dataset["target"].astype(int))
    return FiveMinArtifacts(market=market, dataset=dataset, predictions=preds, metrics=metrics, pipeline=pipeline, feature_cols=cols)


def latest_state_from_store(state_store: pd.DataFrame, market: str, symbol: str, default: int) -> int:
    if state_store.empty:
        return default
    rows = state_store[(state_store["market"].eq(market)) & (state_store["symbol"].eq(symbol))].copy()
    if rows.empty:
        return default
    rows = rows.sort_values("updated_at")
    value = rows["recommended_state"].dropna()
    if value.empty:
        return default
    return int(value.iloc[-1])


def latest_store_row(state_store: pd.DataFrame, market: str, symbol: str) -> pd.Series | None:
    if state_store.empty:
        return None
    rows = state_store[(state_store["market"].eq(market)) & (state_store["symbol"].eq(symbol))].copy()
    if rows.empty:
        return None
    return rows.sort_values("updated_at").iloc[-1]


def classify_market_phase(latest: pd.DataFrame, market: str) -> str:
    bar_index = int(latest["bar_index"].iloc[0]) if "bar_index" in latest.columns else -1
    tod_min = int(latest["tod_min"].iloc[0]) if "tod_min" in latest.columns else -1
    close_min = 15 * 60 if market == "场内" else 16 * 60
    if bar_index == 0:
        return "open"
    if tod_min >= close_min:
        return "close"
    return "intraday"


def row_emit_reason(row: pd.Series, prior: pd.Series | None) -> str:
    if row["market_phase"] == "open":
        if prior is None:
            return "session_open"
        prior_feature = prior.get("feature_bar_time", pd.NaT)
        prior_trade_date = prior.get("trade_date", pd.NaT)
        if pd.isna(prior_feature) or pd.Timestamp(prior_feature) != pd.Timestamp(row["feature_bar_time"]):
            if pd.isna(prior_trade_date) or pd.Timestamp(prior_trade_date) != pd.Timestamp(row["trade_date"]):
                return "session_open"

    prior_state = None
    if prior is not None and pd.notna(prior.get("recommended_state", pd.NA)):
        prior_state = int(prior["recommended_state"])
    current_state = int(row["recommended_state"])
    if prior_state is None:
        if row["action"] in {"买入", "卖出"}:
            return "bootstrap_change"
        return ""
    if current_state != prior_state:
        return "state_change"
    return ""


def terminal_rows(live_df: pd.DataFrame, state_store: pd.DataFrame, print_mode: str) -> pd.DataFrame:
    if live_df.empty or print_mode == "none":
        return live_df.iloc[0:0].copy()
    if print_mode == "all":
        out = live_df.copy()
        out["emit_reason"] = "all"
        return out

    rows: list[dict[str, object]] = []
    for _, row in live_df.iterrows():
        prior = latest_store_row(state_store, row["market"], row["symbol"])
        emit_reason = row_emit_reason(row, prior)
        if emit_reason:
            row_dict = row.to_dict()
            row_dict["emit_reason"] = emit_reason
            rows.append(row_dict)
    return pd.DataFrame(rows)


def live_row_from_frame(
    art: FiveMinArtifacts,
    live_frame: pd.DataFrame,
    market: str,
    symbol: str,
    prior_row: pd.Series | None,
    entry_threshold: float,
    exit_threshold: float,
    nqmain_rt_value: float,
) -> dict[str, object]:
    valid_live = live_frame.dropna(subset=art.feature_cols).sort_values("datetime").copy()
    latest = valid_live.tail(1)
    feature_bar_time = pd.Timestamp(latest["datetime"].iloc[0])
    trade_date = pd.Timestamp(latest["trade_date"].iloc[0]).normalize()
    current_state = latest_state_from_store(
        pd.DataFrame([prior_row]) if prior_row is not None else pd.DataFrame(columns=STATE_COLUMNS),
        market,
        symbol,
        0,
    )
    recent_rows = valid_live[valid_live["trade_date"].dt.normalize().eq(trade_date)].tail(US_ENTRY_CONFIRM_WINDOW).copy()
    recent_raw_probabilities = art.pipeline.predict_proba(recent_rows[art.feature_cols])[:, 1]
    recent_probabilities: list[float] = []
    overlay_reason = "none"
    for idx, (_, recent_row) in enumerate(recent_rows.iterrows()):
        adjusted_probability, recent_overlay_reason = apply_5m_overlay(
            float(recent_raw_probabilities[idx]),
            recent_row,
            market,
        )
        recent_probabilities.append(adjusted_probability)
        if idx == len(recent_rows) - 1:
            overlay_reason = recent_overlay_reason
    raw_probability = float(recent_raw_probabilities[-1])
    probability = float(recent_probabilities[-1])
    expected_edge = estimate_expected_edge(art.predictions, probability)
    entry_allowed, smooth_probability, confirm_count = entry_signal_allowed(
        market,
        probability,
        recent_probabilities,
        entry_threshold,
        feature_bar_time=feature_bar_time,
        bar_index=int(latest["bar_index"].iloc[0]),
        expected_edge=expected_edge,
    )
    recommended_state = current_state
    market_phase = classify_market_phase(latest, market)
    if market == "场内" and is_cn_midday_break():
        market_phase = "midday_break"
    if market_phase != "midday_break":
        if current_state == 0 and entry_allowed:
            recommended_state = 1
        elif current_state == 1 and probability <= exit_threshold:
            recommended_state = 0
    action = action_from_states(current_state, recommended_state)
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
        "market": market,
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
            market,
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


def main() -> None:
    configure_console_output()
    args = parse_args()
    output_path = Path(args.output).resolve()
    live_output_path = Path(args.live_output).resolve()
    state_output_path = Path(args.state_output).resolve()
    history_output_path = Path(args.history_output).resolve()
    run_at = _now_local()

    need_cn = args.market in {"all", "cn"}
    need_us = args.market in {"all", "us"}
    state_store = load_state_store(state_output_path)

    print("Fetching NQ 5m history...", flush=True)
    nq_raw = fetch_nq_intraday(args.range, args.intraday_bars)
    nq_raw["date"] = nq_raw["datetime"]
    nq_raw["adjclose"] = nq_raw["close"]
    nq_features_us = make_price_features(nq_raw, "nq")
    nq_raw_cn = convert_intraday_timezone(nq_raw, US_TZ, CN_TZ)
    nq_raw_cn["adjclose"] = nq_raw_cn["close"]
    nq_features_cn = make_price_features(nq_raw_cn, "nq")
    nqmain_rt_value = latest_nqmain_rt_value(nq_raw)
    print(f"Loaded NQ 5m bars: {len(nq_raw)}", flush=True)

    live_rows = []
    metrics_frames = []
    cn_art = us_art = None
    cn_dataset = us_dataset = None

    if need_us:
        print("Fetching QQQ 5m history...", flush=True)
        qqq_raw = fetch_intraday("QQQ", "美股", args.intraday_bars, args.range)
        us_features = make_intraday_feature_frame(qqq_raw, "qqq")
        print("Building US 5m dataset...", flush=True)
        us_dataset = make_intraday_dataset(qqq_raw, nq_features_us, "qqq", "美股", label_horizon=3, stop_loss=0.004, profit_threshold=0.0005)
        print("Training US 5m model...", flush=True)
        us_art = walk_forward_5m(
            us_dataset,
            "美股5m",
            entry_threshold=US_ENTRY_THRESHOLD,
            exit_threshold=US_EXIT_THRESHOLD,
            stop_loss=0.004,
        )
        us_live_features = us_features.copy()
        us_live_features = merge_asof_features(us_live_features, nq_features_us, "date")
        prior_row = latest_store_row(state_store, "美股", "QQQ")
        live_rows.append(
            live_row_from_frame(
                us_art,
                us_live_features,
                "美股",
                "QQQ",
                prior_row,
                US_ENTRY_THRESHOLD,
                US_EXIT_THRESHOLD,
                nqmain_rt_value,
            )
        )
        metrics_frames.append(us_art.metrics)
        print(f"Built US 5m dataset: {len(us_dataset)}", flush=True)

    if need_cn:
        print(f"Fetching {args.cn_symbol} 5m history...", flush=True)
        cn_raw = fetch_intraday(args.cn_symbol, "场内", args.intraday_bars, args.range)
        cn_features = make_intraday_feature_frame(cn_raw, "cnetf")
        print("Building CN 5m dataset...", flush=True)
        cn_dataset = make_intraday_dataset(cn_raw, nq_features_cn, "cnetf", "场内", label_horizon=3, stop_loss=0.006, profit_threshold=0.0008)
        print("Training CN 5m model...", flush=True)
        cn_art = walk_forward_5m(
            cn_dataset,
            "场内5m",
            entry_threshold=CN_ENTRY_THRESHOLD,
            exit_threshold=CN_EXIT_THRESHOLD,
            stop_loss=0.006,
        )
        cn_live_features = cn_features.copy()
        cn_live_features = merge_asof_features(cn_live_features, nq_features_cn, "date")
        prior_row = latest_store_row(state_store, "场内", args.cn_symbol)
        live_rows.append(
            live_row_from_frame(
                cn_art,
                cn_live_features,
                "场内",
                args.cn_symbol,
                prior_row,
                CN_ENTRY_THRESHOLD,
                CN_EXIT_THRESHOLD,
                nqmain_rt_value,
            )
        )
        metrics_frames.append(cn_art.metrics)
        print(f"Built CN 5m dataset: {len(cn_dataset)}", flush=True)

    live_df = pd.DataFrame(live_rows)
    if not live_df.empty:
        live_df["run_at"] = run_at
    updated_state = save_state_store(state_output_path, live_df) if not live_df.empty else state_store
    history_df = append_signal_history(history_output_path, live_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if us_art is not None and us_dataset is not None:
            us_art.metrics.to_excel(writer, sheet_name="us_metrics", index=False)
            us_art.predictions.to_excel(writer, sheet_name="us_predictions", index=False)
            us_art.dataset.to_excel(writer, sheet_name="us_dataset", index=False)
        if cn_art is not None and cn_dataset is not None:
            cn_art.metrics.to_excel(writer, sheet_name="cn_metrics", index=False)
            cn_art.predictions.to_excel(writer, sheet_name="cn_predictions", index=False)
            cn_art.dataset.to_excel(writer, sheet_name="cn_dataset", index=False)
        live_df.to_excel(writer, sheet_name="live_signals", index=False)
        history_df.tail(5000).to_excel(writer, sheet_name="signal_history", index=False)
        updated_state.to_excel(writer, sheet_name="signal_state_store", index=False)
        notes = pd.DataFrame(
            [
                ("signal_granularity", "5m"),
                ("signal_type", "bar-level long/flat nowcast"),
                ("us_entry_threshold", f"{US_ENTRY_THRESHOLD:.2f}"),
                ("us_exit_threshold", f"{US_EXIT_THRESHOLD:.2f}"),
                ("us_entry_confirm_threshold", f"{US_ENTRY_CONFIRM_THRESHOLD:.2f}"),
                ("us_entry_confirm_window", str(US_ENTRY_CONFIRM_WINDOW)),
                ("us_entry_confirm_required", str(US_ENTRY_CONFIRM_REQUIRED)),
                ("cn_entry_threshold", f"{CN_ENTRY_THRESHOLD:.2f}"),
                ("cn_exit_threshold", f"{CN_EXIT_THRESHOLD:.2f}"),
                ("cn_entry_confirm_threshold", f"{CN_ENTRY_CONFIRM_THRESHOLD:.2f}"),
                ("cn_entry_confirm_window", str(CN_ENTRY_CONFIRM_WINDOW)),
                ("cn_entry_confirm_required", str(CN_ENTRY_CONFIRM_REQUIRED)),
                ("cn_open_entry_threshold", f"{CN_OPEN_ENTRY_THRESHOLD:.2f}"),
                ("cn_open_confirm_threshold", f"{CN_OPEN_CONFIRM_THRESHOLD:.2f}"),
                ("cn_open_confirm_required", str(CN_OPEN_CONFIRM_REQUIRED)),
                ("cn_open_min_bar_index", str(CN_OPEN_MIN_BAR_INDEX)),
                ("cn_min_expected_edge", f"{CN_MIN_EXPECTED_EDGE:.4f}"),
                ("cn_open_min_expected_edge", f"{CN_OPEN_MIN_EXPECTED_EDGE:.4f}"),
                ("cn_rebound_min_raw_probability", f"{CN_REBOUND_MIN_RAW_PROBABILITY:.2f}"),
                ("cn_rebound_floor", f"{CN_REBOUND_FLOOR:.2f}"),
                ("us_stop_loss", "0.004"),
                ("cn_stop_loss", "0.006"),
                ("cn_midday_break_rule", "No new CN buy/sell signals are generated from 11:30 to 13:00 Asia/Shanghai; the model freezes the current state during the lunch break."),
                ("cn_opening_filter_rule", "CN entries from 09:35 to 10:00 require stronger thresholds, 3-of-3 confirmation, and a minimum bar index to avoid first-wave opening noise."),
                ("cn_expected_edge_rule", "CN entries are blocked when the probability-conditioned expected edge estimate does not exceed the round-trip cost floor."),
                ("signal_memory", "state store persists the latest recommended_state per market"),
            ],
            columns=["item", "value"],
        )
        notes.to_excel(writer, sheet_name="notes", index=False)

    live_df.to_csv(live_output_path, index=False)
    print(f"Wrote history CSV: {history_output_path}")
    print(f"Wrote workbook: {output_path}")
    print(f"Wrote live CSV: {live_output_path}")
    print(f"Wrote state CSV: {state_output_path}")
    print_df = terminal_rows(live_df, state_store, args.print_mode)
    print("\nLive 5m signals:")
    if print_df.empty:
        print("No opening or state-change event to print.")
    else:
        for _, row in print_df.iterrows():
            print(f"[{row['emit_reason']}] {row['signal_summary']}")


if __name__ == "__main__":
    main()
