#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pandas.tseries.offsets import BDay
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


YAHOO_ENDPOINTS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
NQ_PATTERN = re.compile(r"纳斯达克指数\(([-+]?\d+(?:\.\d+)?)\)")
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}
FIB_WINDOWS = (5, 8, 13, 21)
FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue


@dataclass
class ModelArtifacts:
    name: str
    dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    pipeline: Pipeline
    feature_cols: list[str]


@dataclass
class IntradayArtifacts:
    bars: pd.DataFrame
    signals: pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.DataFrame


@dataclass
class LiveSignal:
    market: str
    feature_date: pd.Timestamp
    next_session_date: pd.Timestamp
    last_signal_date: pd.Timestamp
    current_state: int
    raw_probability: float
    probability_long: float
    recommended_state: int
    tradable_symbol: str
    tradable_close: float
    wave_overlay: str
    top_drivers: str


def fetch_yahoo_chart(
    symbol: str,
    interval: str,
    date_range: str,
    include_prepost: bool = False,
    include_adjusted: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = {
        "interval": interval,
        "range": date_range,
        "events": "history",
        "includePrePost": str(include_prepost).lower(),
    }
    if include_adjusted:
        params["includeAdjustedClose"] = "true"
    last_error = None
    for endpoint in YAHOO_ENDPOINTS:
        url = endpoint.format(symbol=symbol)
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            payload = response.json()
            result = payload["chart"]["result"][0]
            meta = result["meta"]
            quote = result["indicators"]["quote"][0]
            frame = pd.DataFrame(
                {
                    "timestamp": result["timestamp"],
                    "open": quote["open"],
                    "high": quote["high"],
                    "low": quote["low"],
                    "close": quote["close"],
                    "volume": quote["volume"],
                }
            )
            adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
            if adjclose is not None:
                frame["adjclose"] = adjclose
            datetime_local = (
                pd.to_datetime(frame["timestamp"], unit="s", utc=True)
                .dt.tz_convert(meta["exchangeTimezoneName"])
                .dt.tz_localize(None)
            )
            frame["datetime"] = datetime_local
            frame["date"] = frame["datetime"].dt.normalize()
            if "adjclose" not in frame.columns:
                frame["adjclose"] = frame["close"]
            frame["adjclose"] = frame["adjclose"].fillna(frame["close"])
            frame = frame.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            frame["symbol"] = symbol
            return frame, meta
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"failed to fetch {symbol}: {last_error}") from last_error


def fetch_yahoo_daily(symbol: str, date_range: str = "2y") -> pd.DataFrame:
    frame, _ = fetch_yahoo_chart(symbol, interval="1d", date_range=date_range, include_adjusted=True)
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return frame[["date", "open", "high", "low", "close", "adjclose", "volume", "symbol"]]


def fetch_yahoo_intraday(symbol: str, interval: str = "5m", date_range: str = "60d") -> pd.DataFrame:
    frame, _ = fetch_yahoo_chart(symbol, interval=interval, date_range=date_range, include_adjusted=False)
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return frame[["datetime", "date", "open", "high", "low", "close", "volume", "symbol"]]


def sina_symbol(symbol: str) -> str:
    clean = symbol.upper()
    if clean.endswith(".SZ"):
        return f"sz{clean[:-3]}"
    if clean.endswith(".SS"):
        return f"sh{clean[:-3]}"
    raise ValueError(f"unsupported CN symbol for Sina: {symbol}")


def fetch_sina_cn_intraday(symbol: str, interval_minutes: int = 5, datalen: int = 5000) -> pd.DataFrame:
    url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    response = requests.get(
        url,
        params={"symbol": sina_symbol(symbol), "scale": interval_minutes, "ma": "no", "datalen": datalen},
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["datetime"] = pd.to_datetime(frame["day"])
    frame["date"] = frame["datetime"].dt.normalize()
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["datetime", "open", "close"]).sort_values("datetime").reset_index(drop=True)
    return frame[["datetime", "date", "open", "high", "low", "close", "volume", "symbol"]]


def third_friday(year: int, month: int) -> pd.Timestamp:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    first_friday = month_start + pd.offsets.Week(weekday=4)
    return first_friday + pd.Timedelta(days=14)


def official_equity_roll_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    roll_dates: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        for month in (3, 6, 9, 12):
            expiry = third_friday(year, month)
            roll = expiry - pd.Timedelta(days=4)
            if start <= roll <= end:
                roll_dates.append(roll.normalize())
    return sorted(roll_dates)


def quarterly_symbol(year: int, month: int) -> str:
    return f"NQ{MONTH_CODE[month]}{str(year)[-2:]}.CME"


def quarterly_contract_schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    expanded_start = pd.Timestamp(start.year - 1, 1, 1)
    expanded_end = pd.Timestamp(end.year + 1, 12, 31)
    quarters = []
    for year in range(expanded_start.year, expanded_end.year + 1):
        for month in (3, 6, 9, 12):
            expiry = third_friday(year, month).normalize()
            roll = (expiry - pd.Timedelta(days=4)).normalize()
            quarters.append({"year": year, "month": month, "expiry": expiry, "roll": roll})
    quarters = pd.DataFrame(quarters).sort_values("expiry").reset_index(drop=True)
    for i, row in quarters.iterrows():
        lead_start = quarters.iloc[i - 1]["roll"] if i > 0 else expanded_start
        lead_end = row["roll"] - pd.Timedelta(days=1)
        rows.append(
            {
                "symbol": quarterly_symbol(int(row["year"]), int(row["month"])),
                "expiry": row["expiry"],
                "roll_date": row["roll"],
                "lead_start": pd.Timestamp(lead_start).normalize(),
                "lead_end": pd.Timestamp(lead_end).normalize(),
            }
        )
    schedule = pd.DataFrame(rows)
    return schedule[(schedule["lead_end"] >= start) & (schedule["lead_start"] <= end)].reset_index(drop=True)


def try_fetch_yahoo_daily(symbol: str, date_range: str = "2y") -> pd.DataFrame | None:
    try:
        return fetch_yahoo_daily(symbol, date_range=date_range)
    except Exception:  # noqa: BLE001
        return None


def archive_nq_contracts(archive_dir: Path, years: Iterable[int], date_range: str = "2y") -> pd.DataFrame:
    archive_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in sorted(set(years)):
        for month in (3, 6, 9, 12):
            symbol = quarterly_symbol(year, month)
            path = archive_dir / f"{symbol}.csv"
            frame = try_fetch_yahoo_daily(symbol, date_range=date_range)
            if frame is None or frame.empty:
                rows.append({"symbol": symbol, "status": "missing", "path": str(path)})
                continue
            frame.to_csv(path, index=False)
            rows.append(
                {
                    "symbol": symbol,
                    "status": "archived",
                    "path": str(path),
                    "first_date": frame["date"].min(),
                    "last_date": frame["date"].max(),
                    "rows": len(frame),
                }
            )
    return pd.DataFrame(rows)


def load_archived_nq_contracts(archive_dir: Path) -> dict[str, pd.DataFrame]:
    contracts: dict[str, pd.DataFrame] = {}
    for path in sorted(archive_dir.glob("NQ*.CME.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        if frame.empty:
            continue
        contracts[path.stem] = frame.sort_values("date").reset_index(drop=True)
    return contracts


def build_archived_nq_main(
    archive_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    contracts = load_archived_nq_contracts(archive_dir)
    schedule = quarterly_contract_schedule(start, end)
    segments = []
    roll_rows = []
    cumulative_adjustment = 0.0
    previous_contract_frame = None
    previous_symbol = None

    for row in schedule.itertuples(index=False):
        symbol = row.symbol
        contract = contracts.get(symbol)
        if contract is None:
            return pd.DataFrame(), schedule, False
        seg_start = max(pd.Timestamp(row.lead_start), start)
        seg_end = min(pd.Timestamp(row.lead_end), end)
        segment = contract[(contract["date"] >= seg_start) & (contract["date"] <= seg_end)].copy()
        if segment.empty:
            return pd.DataFrame(), schedule, False
        if previous_contract_frame is not None and previous_symbol is not None:
            new_roll_date = pd.Timestamp(row.lead_start)
            new_close = contract.loc[contract["date"] >= new_roll_date, "close"]
            old_close = previous_contract_frame.loc[previous_contract_frame["date"] >= new_roll_date, "close"]
            if new_close.empty:
                return pd.DataFrame(), schedule, False
            if old_close.empty:
                old_close = previous_contract_frame.loc[previous_contract_frame["date"] < new_roll_date, "close"]
            if old_close.empty:
                return pd.DataFrame(), schedule, False
            gap = float(new_close.iloc[0] - old_close.iloc[-1])
            cumulative_adjustment += gap
            roll_rows.append(
                {
                    "roll_date": new_roll_date,
                    "from_symbol": previous_symbol,
                    "to_symbol": symbol,
                    "gap_points": gap,
                    "cumulative_adjustment": cumulative_adjustment,
                }
            )
        segment["contract_symbol"] = symbol
        segment["roll_adjustment"] = cumulative_adjustment
        for col in ["open", "high", "low", "close", "adjclose"]:
            segment[col] = segment[col] + cumulative_adjustment
        segments.append(segment)
        previous_contract_frame = contract
        previous_symbol = symbol

    if not segments:
        return pd.DataFrame(), schedule, False
    out = pd.concat(segments, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["symbol"] = "NQ_MAIN_ARCHIVE"
    roll_info = pd.DataFrame(roll_rows)
    return out[["date", "open", "high", "low", "close", "adjclose", "volume", "symbol", "roll_adjustment", "contract_symbol"]], roll_info, True


def build_roll_adjusted_nq_main(date_range: str = "2y") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_yahoo_daily("NQ=F", date_range).copy()
    raw = raw.sort_values("date").reset_index(drop=True)
    raw["raw_close"] = raw["adjclose"]
    raw["raw_open"] = raw["open"]
    raw["raw_high"] = raw["high"]
    raw["raw_low"] = raw["low"]

    roll_dates = official_equity_roll_dates(raw["date"].min(), raw["date"].max())
    adjustment = pd.Series(0.0, index=raw.index)
    roll_rows = []
    cumulative_gap = 0.0
    for roll_date in reversed(roll_dates):
        current_idx = raw.index[raw["date"] >= roll_date]
        prev_idx = raw.index[raw["date"] < roll_date]
        if len(current_idx) == 0 or len(prev_idx) == 0:
            continue
        current_i = int(current_idx[0])
        prev_i = int(prev_idx[-1])
        gap = float(raw.at[current_i, "raw_close"] - raw.at[prev_i, "raw_close"])
        cumulative_gap += gap
        adjustment.loc[:prev_i] += gap
        roll_rows.append(
            {
                "roll_date": roll_date,
                "prev_trade_date": raw.at[prev_i, "date"],
                "gap_points": gap,
                "cumulative_adjustment": cumulative_gap,
            }
        )

    raw["roll_adjustment"] = adjustment
    for col, source in (
        ("open", "raw_open"),
        ("high", "raw_high"),
        ("low", "raw_low"),
        ("close", "raw_close"),
        ("adjclose", "raw_close"),
    ):
        raw[col] = raw[source] + raw["roll_adjustment"]
    raw["symbol"] = "NQ_MAIN_PROXY"
    roll_info = pd.DataFrame(roll_rows).sort_values("roll_date").reset_index(drop=True)
    return raw[["date", "open", "high", "low", "close", "adjclose", "volume", "symbol", "roll_adjustment", "raw_close"]], roll_info


def make_price_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    px = out["adjclose"].astype(float)
    ret1 = px.pct_change()
    out[f"{prefix}_close"] = px
    out[f"{prefix}_ret_1"] = ret1
    for lookback in (2, 3, 5, 8, 10, 13, 20, 21):
        out[f"{prefix}_ret_{lookback}"] = px.pct_change(lookback)
    for window in (3, 5, 8, 10, 13, 20, 21):
        sma = px.rolling(window).mean()
        out[f"{prefix}_sma_ratio_{window}"] = px / sma - 1.0
    for window in (5, 8, 10, 13, 20, 21):
        out[f"{prefix}_vol_{window}"] = ret1.rolling(window).std()
        out[f"{prefix}_volume_rel_{window}"] = out["volume"] / out["volume"].rolling(window).mean() - 1.0
    out[f"{prefix}_range_1"] = (out["high"] - out["low"]) / out["close"]
    for window in FIB_WINDOWS:
        rolling_high = out["high"].rolling(window).max()
        rolling_low = out["low"].rolling(window).min()
        swing_range = (rolling_high - rolling_low).replace(0, np.nan)
        fib_pos = (px - rolling_low) / swing_range
        out[f"{prefix}_fib_pos_{window}"] = fib_pos.clip(lower=0.0, upper=1.0)
        out[f"{prefix}_swing_range_ratio_{window}"] = swing_range / px
        out[f"{prefix}_dist_to_swing_high_{window}"] = px / rolling_high - 1.0
        out[f"{prefix}_dist_to_swing_low_{window}"] = px / rolling_low - 1.0
        for level in FIB_LEVELS:
            level_tag = str(level).replace(".", "")
            fib_level = rolling_low + level * swing_range
            out[f"{prefix}_fib_dist_{level_tag}_{window}"] = px / fib_level - 1.0

    up_streaks: list[int] = []
    down_streaks: list[int] = []
    up_seq = 0
    down_seq = 0
    for value in ret1.fillna(0.0):
        if value > 0:
            up_seq += 1
            down_seq = 0
        elif value < 0:
            down_seq += 1
            up_seq = 0
        else:
            up_seq = 0
            down_seq = 0
        up_streaks.append(up_seq)
        down_streaks.append(down_seq)
    out[f"{prefix}_up_seq"] = up_streaks
    out[f"{prefix}_down_seq"] = down_streaks

    high_13 = out["high"].rolling(13).max()
    low_13 = out["low"].rolling(13).min()
    range_13 = (high_13 - low_13).replace(0.0, np.nan)
    high_idx_13 = out["high"].rolling(13).apply(np.argmax, raw=True)
    low_idx_13 = out["low"].rolling(13).apply(np.argmin, raw=True)
    up_leg_13 = low_idx_13 < high_idx_13
    retrace_13 = pd.Series(
        np.where(up_leg_13, (high_13 - px) / range_13, (px - low_13) / range_13),
        index=out.index,
    )
    leg_ratio_5_8 = (px - px.shift(5)).abs() / (px.shift(5) - px.shift(13)).abs().replace(0.0, np.nan)
    path_5 = ret1.abs().rolling(5).sum().replace(0.0, np.nan)
    out[f"{prefix}_fib_signed_retrace_13"] = np.where(up_leg_13, retrace_13, -retrace_13)
    out[f"{prefix}_fib_retrace_fit_13"] = pd.concat(
        [(retrace_13 - 0.382).abs(), (retrace_13 - 0.5).abs(), (retrace_13 - 0.618).abs()],
        axis=1,
    ).min(axis=1)
    out[f"{prefix}_fib_leg_fit_5_8"] = pd.concat(
        [(leg_ratio_5_8 - 0.618).abs(), (leg_ratio_5_8 - 1.0).abs(), (leg_ratio_5_8 - 1.618).abs()],
        axis=1,
    ).min(axis=1)
    out[f"{prefix}_ew_thrust_5"] = ret1.rolling(5).sum() / path_5
    out[f"{prefix}_fwd_ret_1"] = ret1.shift(-1)
    feature_cols = [col for col in out.columns if col == "date" or col.startswith(f"{prefix}_")]
    return out[feature_cols]


def _safe_feature_value(row: pd.Series, key: str) -> float:
    if key not in row or pd.isna(row[key]):
        return np.nan
    return float(row[key])


def apply_wave_fib_overlay(probability: float, row: pd.Series | pd.DataFrame, market: str) -> tuple[float, str]:
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    adjusted = float(probability)
    market = str(market)

    if market.startswith("美股"):
        qqq_ret_13 = _safe_feature_value(row, "qqq_ret_13")
        nq_ret_13 = _safe_feature_value(row, "nq_ret_13")
        qqq_fib_8 = _safe_feature_value(row, "qqq_fib_pos_8")
        nq_fib_8 = _safe_feature_value(row, "nq_fib_pos_8")
        qqq_sma_21 = _safe_feature_value(row, "qqq_sma_ratio_21")
        if all(pd.notna(x) for x in (qqq_ret_13, nq_ret_13, qqq_fib_8, nq_fib_8, qqq_sma_21)) and qqq_ret_13 < 0 and nq_ret_13 < 0:
            if qqq_fib_8 < 0.236 and nq_fib_8 < 0.382:
                return min(adjusted, 0.12), "us_bear_breakdown"
            if qqq_fib_8 <= 0.382 and nq_fib_8 <= 0.618 and qqq_sma_21 < 0:
                return min(adjusted, 0.20), "us_bear_wave4_low"
            if qqq_fib_8 <= 0.618 and nq_fib_8 <= 0.618 and qqq_sma_21 < 0:
                return min(adjusted, 0.35), "us_bear_retracement_cap"
        return adjusted, "none"

    if market.startswith("场内"):
        cnetf_ret_13 = _safe_feature_value(row, "cnetf_ret_13")
        nq_ret_13 = _safe_feature_value(row, "nq_ret_13")
        cnetf_fib_8 = _safe_feature_value(row, "cnetf_fib_pos_8")
        nq_fib_8 = _safe_feature_value(row, "nq_fib_pos_8")
        cnetf_down_seq = _safe_feature_value(row, "cnetf_down_seq")
        if all(pd.notna(x) for x in (cnetf_ret_13, nq_ret_13, cnetf_fib_8, nq_fib_8, cnetf_down_seq)) and cnetf_ret_13 < 0 and nq_ret_13 < 0:
            if cnetf_fib_8 < 0.146 and cnetf_down_seq >= 2:
                return max(adjusted, 0.62), "cn_oversold_rebound_floor"
            if 0.382 <= cnetf_fib_8 <= 0.618 and nq_fib_8 <= 0.618:
                return min(adjusted, 0.35), "cn_corrective_rebound_exit"
            if cnetf_fib_8 <= 0.382 and nq_fib_8 <= 0.382:
                return min(adjusted, 0.25), "cn_downtrend_follow"
        return adjusted, "none"

    return adjusted, "none"


def build_signal_sessions(events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_excel(events_path, sheet_name="events")
    events["datetime"] = pd.to_datetime(events["datetime"])
    events = events.sort_values(["market", "datetime", "seq_newest"]).reset_index(drop=True)
    us_mask = events["market"].eq("美股")
    events.loc[us_mask, "session_date"] = (
        events.loc[us_mask, "datetime"]
        - pd.to_timedelta((events.loc[us_mask, "datetime"].dt.hour < 12).astype(int), unit="D")
    ).dt.normalize()
    events.loc[~us_mask, "session_date"] = events.loc[~us_mask, "datetime"].dt.normalize()
    events["session_date"] = pd.to_datetime(events["session_date"])
    events["state"] = events["state"].astype(int)

    rows = []
    for (market, session_date), group in events.groupby(["market", "session_date"], sort=True):
        group = group.sort_values("datetime").reset_index(drop=True)
        state_changes = int(group["state"].ne(group["state"].shift()).sum() - 1)
        rows.append(
            {
                "market": market,
                "session_date": session_date,
                "first_time": group["datetime"].dt.strftime("%H:%M").iloc[0],
                "last_time": group["datetime"].dt.strftime("%H:%M").iloc[-1],
                "first_signal": group["signal"].iloc[0],
                "last_signal": group["signal"].iloc[-1],
                "state": int(group["state"].iloc[-1]),
                "change_count": state_changes,
                "event_count": len(group),
                "last_event_text": group["event_text"].iloc[-1],
            }
        )

    sessions = pd.DataFrame(rows).sort_values(["market", "session_date"]).reset_index(drop=True)
    sessions["session_date"] = pd.to_datetime(sessions["session_date"])
    us_sessions = sessions[sessions["market"] == "美股"].copy()
    cn_sessions = sessions[sessions["market"] == "场内"].copy()
    return us_sessions, cn_sessions


def merge_asof_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_date_col: str,
    drop_right_key: bool = True,
) -> pd.DataFrame:
    left_frame = left.copy()
    right_frame = right.copy()
    left_frame["date"] = pd.to_datetime(left_frame["date"], errors="coerce").astype("datetime64[ns]")
    right_frame[right_date_col] = pd.to_datetime(right_frame[right_date_col], errors="coerce").astype("datetime64[ns]")
    merged = pd.merge_asof(
        left_frame.sort_values("date"),
        right_frame.sort_values(right_date_col),
        left_on="date",
        right_on=right_date_col,
        direction="backward",
    )
    if drop_right_key and right_date_col != "date":
        merged = merged.drop(columns=[right_date_col])
    return merged


def build_us_dataset(
    us_sessions: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_features: pd.DataFrame,
) -> pd.DataFrame:
    base = qqq_features.copy()
    base = base.merge(nq_features, on="date", how="left")
    base = merge_asof_features(base, cn_features.rename(columns={"date": "cn_feature_date"}), "cn_feature_date")
    base["nq_minus_qqq_ret_1"] = base["nq_ret_1"] - base["qqq_ret_1"]
    base["cn_minus_qqq_ret_1"] = base["cnetf_ret_1"] - base["qqq_ret_1"]
    base = base.merge(
        us_sessions[["session_date", "state", "change_count", "event_count", "last_signal"]],
        left_on="date",
        right_on="session_date",
        how="inner",
    )
    base = base.sort_values("date").reset_index(drop=True)
    base["target"] = base["state"].shift(-1)
    base["target_date"] = base["date"].shift(-1)
    base["trade_ret"] = base["qqq_fwd_ret_1"]
    base["current_state"] = base["state"]
    return base.dropna(subset=["target", "trade_ret"]).reset_index(drop=True)


def build_cn_dataset(
    cn_sessions: pd.DataFrame,
    cn_features: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
) -> pd.DataFrame:
    us_for_cn = qqq_features.merge(nq_features, on="date", how="left").copy()
    us_for_cn = us_for_cn.rename(columns={"date": "us_session_date"})
    us_for_cn["us_available_date"] = us_for_cn["us_session_date"] + pd.Timedelta(days=1)
    base = cn_features.copy()
    base = merge_asof_features(base, us_for_cn, "us_available_date")
    base["nq_minus_cnetf_ret_1"] = base["nq_ret_1"] - base["cnetf_ret_1"]
    base["qqq_minus_cnetf_ret_1"] = base["qqq_ret_1"] - base["cnetf_ret_1"]
    base = base.merge(
        cn_sessions[["session_date", "state", "change_count", "event_count", "last_signal"]],
        left_on="date",
        right_on="session_date",
        how="inner",
    )
    base = base.sort_values("date").reset_index(drop=True)
    base["target"] = base["state"].shift(-1)
    base["target_date"] = base["date"].shift(-1)
    base["trade_ret"] = base["cnetf_fwd_ret_1"]
    base["current_state"] = base["state"]
    return base.dropna(subset=["target", "trade_ret"]).reset_index(drop=True)


def build_us_live_frame(
    us_sessions: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
    cn_features: pd.DataFrame,
) -> pd.DataFrame:
    base = qqq_features.copy()
    base = base.merge(nq_features, on="date", how="left")
    base = merge_asof_features(base, cn_features.rename(columns={"date": "cn_feature_date"}), "cn_feature_date")
    base["nq_minus_qqq_ret_1"] = base["nq_ret_1"] - base["qqq_ret_1"]
    base["cn_minus_qqq_ret_1"] = base["cnetf_ret_1"] - base["qqq_ret_1"]
    session_state = us_sessions[["session_date", "state", "change_count", "event_count", "last_signal"]].rename(
        columns={"session_date": "state_date"}
    )
    base = merge_asof_features(base, session_state, "state_date", drop_right_key=False)
    return base.sort_values("date").reset_index(drop=True)


def build_cn_live_frame(
    cn_sessions: pd.DataFrame,
    cn_features: pd.DataFrame,
    qqq_features: pd.DataFrame,
    nq_features: pd.DataFrame,
) -> pd.DataFrame:
    us_for_cn = qqq_features.merge(nq_features, on="date", how="left").copy()
    us_for_cn = us_for_cn.rename(columns={"date": "us_session_date"})
    us_for_cn["us_available_date"] = us_for_cn["us_session_date"] + pd.Timedelta(days=1)
    base = cn_features.copy()
    base = merge_asof_features(base, us_for_cn, "us_available_date")
    base["nq_minus_cnetf_ret_1"] = base["nq_ret_1"] - base["cnetf_ret_1"]
    base["qqq_minus_cnetf_ret_1"] = base["qqq_ret_1"] - base["cnetf_ret_1"]
    session_state = cn_sessions[["session_date", "state", "change_count", "event_count", "last_signal"]].rename(
        columns={"session_date": "state_date"}
    )
    base = merge_asof_features(base, session_state, "state_date", drop_right_key=False)
    return base.sort_values("date").reset_index(drop=True)


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "session_date",
        "target_date",
        "target",
        "trade_ret",
        "state",
        "current_state",
        "change_count",
        "event_count",
        "last_signal",
    }
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if "fwd_ret" in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def evaluate_return_series(returns: pd.Series) -> tuple[float, float, float]:
    returns = returns.fillna(0.0)
    if returns.empty:
        return 0.0, 0.0, 0.0
    cumulative = float((1.0 + returns).prod() - 1.0)
    sharpe = 0.0
    if returns.std(ddof=0) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(252))
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    return cumulative, sharpe, max_dd


def build_metrics(predictions: pd.DataFrame, strategy_name: str, pred_col: str) -> dict[str, float | str]:
    actual = predictions["actual_target"].astype(int)
    pred = predictions[pred_col].astype(int)
    ret = predictions[f"{pred_col}_ret"]
    cumulative, sharpe, max_dd = evaluate_return_series(ret)
    metrics = {
        "strategy": strategy_name,
        "accuracy": float(accuracy_score(actual, pred)),
        "precision": float(precision_score(actual, pred, zero_division=0)),
        "recall": float(recall_score(actual, pred, zero_division=0)),
        "f1": float(f1_score(actual, pred, zero_division=0)),
        "cum_return": cumulative,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_position": float(pred.mean()),
    }
    if predictions["probability"].nunique() > 1 and actual.nunique() > 1:
        metrics["auc"] = float(roc_auc_score(actual, predictions["probability"]))
    else:
        metrics["auc"] = np.nan
    return metrics


def next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(ts).normalize() + BDay(1)).normalize()


def summarize_live_drivers(pipeline: Pipeline, row: pd.DataFrame, feature_cols: list[str], top_n: int = 5) -> str:
    imputed = pipeline.named_steps["imputer"].transform(row[feature_cols])
    scaled = pipeline.named_steps["scaler"].transform(imputed)
    coefs = pipeline.named_steps["model"].coef_[0]
    contrib = scaled[0] * coefs
    driver = pd.DataFrame({"feature": feature_cols, "contribution": contrib})
    driver["abs_contribution"] = driver["contribution"].abs()
    driver = driver.sort_values("abs_contribution", ascending=False).head(top_n)
    return " | ".join(f"{feat}:{val:+.3f}" for feat, val in zip(driver["feature"], driver["contribution"]))


def build_live_signal(
    market: str,
    live_frame: pd.DataFrame,
    art: ModelArtifacts,
    tradable_symbol: str,
    tradable_price_col: str,
) -> LiveSignal:
    latest = live_frame.dropna(subset=art.feature_cols).iloc[[-1]].copy()
    raw_probability = float(art.pipeline.predict_proba(latest[art.feature_cols])[0, 1])
    probability, overlay_reason = apply_wave_fib_overlay(raw_probability, latest, market)
    recommended_state = int(probability >= 0.5)
    last_signal_date = latest["state_date"].iloc[0]
    return LiveSignal(
        market=market,
        feature_date=pd.Timestamp(latest["date"].iloc[0]),
        next_session_date=next_business_day(latest["date"].iloc[0]),
        last_signal_date=pd.Timestamp(last_signal_date),
        current_state=int(latest["state"].iloc[0]),
        raw_probability=raw_probability,
        probability_long=probability,
        recommended_state=recommended_state,
        tradable_symbol=tradable_symbol,
        tradable_close=float(latest[tradable_price_col].iloc[0]),
        wave_overlay=overlay_reason,
        top_drivers=summarize_live_drivers(art.pipeline, latest, art.feature_cols),
    )


def extract_cn_nq_signal_frame(events_path: Path, start_date: str) -> pd.DataFrame:
    events = pd.read_excel(events_path, sheet_name="events")
    events = events[events["market"] == "场内"].copy()
    events["datetime"] = pd.to_datetime(events["datetime"])
    events = events[events["datetime"] >= pd.Timestamp(start_date)].copy()
    events["nq_signal"] = pd.to_numeric(events["content"].astype(str).str.extract(NQ_PATTERN, expand=False), errors="coerce")
    events = events.dropna(subset=["nq_signal"]).sort_values("datetime").reset_index(drop=True)
    return events[["datetime", "nq_signal", "signal", "state", "content"]]


def estimate_cn_etf_beta(cn_daily: pd.DataFrame, nq_daily: pd.DataFrame) -> float:
    merged = cn_daily[["date", "adjclose"]].rename(columns={"adjclose": "cn_close"}).merge(
        nq_daily[["date", "adjclose"]].rename(columns={"adjclose": "nq_close"}),
        on="date",
        how="inner",
    )
    merged["cn_ret"] = merged["cn_close"].pct_change()
    merged["nq_ret"] = merged["nq_close"].pct_change()
    merged = merged.dropna(subset=["cn_ret", "nq_ret"])
    if merged.empty or merged["nq_ret"].var(ddof=0) == 0:
        return 1.0
    beta = float(np.cov(merged["cn_ret"], merged["nq_ret"], ddof=0)[0, 1] / merged["nq_ret"].var(ddof=0))
    return max(0.1, min(beta, 3.0))


def build_cn_intraday_context(
    bars: pd.DataFrame,
    nq_signal_frame: pd.DataFrame,
    cn_daily: pd.DataFrame,
    nq_daily: pd.DataFrame,
    limit_pct: float,
) -> pd.DataFrame:
    context = bars.copy().sort_values("datetime").reset_index(drop=True)
    context = pd.merge_asof(
        context,
        nq_signal_frame[["datetime", "nq_signal"]].sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    context["trade_date"] = context["date"].dt.normalize()
    context["prev_close"] = context.groupby("trade_date")["close"].transform("first")
    prev_day_close = context.groupby("trade_date")["close"].last().shift(1)
    prev_day_close.index = pd.to_datetime(prev_day_close.index)
    context = context.merge(prev_day_close.rename("prev_session_close"), left_on="trade_date", right_index=True, how="left")
    context["prev_session_close"] = context["prev_session_close"].fillna(context["prev_close"])

    daily_nq = nq_signal_frame.assign(trade_date=nq_signal_frame["datetime"].dt.normalize()).groupby("trade_date")["nq_signal"].last()
    prior_nq = daily_nq.shift(1)
    prior_nq.index = pd.to_datetime(prior_nq.index)
    context = context.merge(prior_nq.rename("prev_session_nq"), left_on="trade_date", right_index=True, how="left")
    context["prev_session_nq"] = context["prev_session_nq"].fillna(context["nq_signal"])

    beta = estimate_cn_etf_beta(cn_daily, nq_daily)
    nq_move = context["nq_signal"] / context["prev_session_nq"] - 1.0
    context["fair_value"] = context["prev_session_close"] * (1.0 + beta * nq_move)
    context["premium_rate"] = context["open"] / context["fair_value"] - 1.0
    context["limit_up"] = context["prev_session_close"] * (1.0 + limit_pct)
    context["limit_down"] = context["prev_session_close"] * (1.0 - limit_pct)
    tol = 1e-6
    context["buy_blocked_limit"] = context["low"] >= context["limit_up"] - tol
    context["sell_blocked_limit"] = context["high"] <= context["limit_down"] + tol
    context["beta_estimate"] = beta
    return context


def build_cn_intraday_backtest(
    events_path: Path,
    symbol: str,
    cn_daily: pd.DataFrame,
    nq_daily: pd.DataFrame,
    start_date: str = "2026-01-28",
    interval: str = "5m",
    date_range: str = "60d",
    source: str = "sina",
    commission_bps: float = 2.0,
    slippage_bps: float = 3.0,
    premium_limit_bps: float = 120.0,
    discount_limit_bps: float = 120.0,
    limit_pct: float = 0.10,
) -> IntradayArtifacts:
    transitions = pd.read_excel(events_path, sheet_name="transitions")
    transitions["datetime"] = pd.to_datetime(transitions["datetime"])
    transitions = transitions[(transitions["market"] == "场内") & transitions["datetime"].ge(pd.Timestamp(start_date))].copy()
    transitions = transitions.sort_values("datetime").reset_index(drop=True)
    if source == "sina":
        bars = fetch_sina_cn_intraday(symbol, interval_minutes=int(interval.rstrip("m")), datalen=5000)
    else:
        bars = fetch_yahoo_intraday(symbol, interval=interval, date_range=date_range)
    bars = bars.dropna(subset=["open", "close"]).sort_values("datetime").reset_index(drop=True)
    min_dt = bars["datetime"].min()
    transitions = transitions[transitions["datetime"] >= min_dt].copy()
    transitions["signal_time"] = transitions["datetime"]
    nq_signal_frame = extract_cn_nq_signal_frame(events_path, start_date)
    bars = build_cn_intraday_context(bars, nq_signal_frame, cn_daily, nq_daily, limit_pct=limit_pct)

    fills = []
    trades = []
    position = 0
    entry_fill = None

    for idx, signal_row in transitions.reset_index(drop=True).iterrows():
        signal_time = pd.Timestamp(signal_row["signal_time"])
        next_signal_time = (
            pd.Timestamp(transitions.iloc[idx + 1]["signal_time"]) if idx + 1 < len(transitions) else bars["datetime"].max() + pd.Timedelta(minutes=5)
        )
        side = "buy" if int(signal_row["state"]) == 1 else "sell"
        eligible = bars[(bars["datetime"] > signal_time) & (bars["datetime"] <= next_signal_time)].copy()
        if eligible.empty:
            fills.append({**signal_row.to_dict(), "fill_time": pd.NaT, "fill_open": np.nan, "blocked_reason": "no_bar"})
            continue

        chosen = None
        blocked_reason = "signal_replaced"
        for bar in eligible.itertuples(index=False):
            premium_rate = float(bar.premium_rate) if pd.notna(bar.premium_rate) else np.nan
            if side == "buy" and bool(bar.buy_blocked_limit):
                blocked_reason = "limit_up"
                continue
            if side == "sell" and bool(bar.sell_blocked_limit):
                blocked_reason = "limit_down"
                continue
            if side == "buy" and pd.notna(premium_rate) and premium_rate > premium_limit_bps / 10000.0:
                blocked_reason = "premium_too_high"
                continue
            if side == "sell" and pd.notna(premium_rate) and premium_rate < -discount_limit_bps / 10000.0:
                blocked_reason = "discount_too_high"
                continue
            chosen = bar
            blocked_reason = ""
            break

        if chosen is None:
            fills.append({**signal_row.to_dict(), "fill_time": pd.NaT, "fill_open": np.nan, "blocked_reason": blocked_reason})
            continue

        raw_open = float(chosen.open)
        fill_open = raw_open * (1.0 + slippage_bps / 10000.0) if side == "buy" else raw_open * (1.0 - slippage_bps / 10000.0)
        fill_row = {
            **signal_row.to_dict(),
            "fill_time": pd.Timestamp(chosen.datetime),
            "fill_open": fill_open,
            "raw_fill_open": raw_open,
            "premium_rate": float(chosen.premium_rate) if pd.notna(chosen.premium_rate) else np.nan,
            "fair_value": float(chosen.fair_value) if pd.notna(chosen.fair_value) else np.nan,
            "blocked_reason": "",
            "side": side,
        }
        fills.append(fill_row)

        if position == 0 and side == "buy":
            position = 1
            entry_fill = fill_row
        elif position == 1 and side == "sell" and entry_fill is not None:
            gross_ret = fill_open / float(entry_fill["fill_open"]) - 1.0
            net_ret = ((1.0 - commission_bps / 10000.0) * fill_open) / ((1.0 + commission_bps / 10000.0) * float(entry_fill["fill_open"])) - 1.0
            trades.append(
                {
                    "symbol": symbol,
                    "entry_signal_time": entry_fill["signal_time"],
                    "entry_time": entry_fill["fill_time"],
                    "entry_price": entry_fill["fill_open"],
                    "entry_raw_price": entry_fill["raw_fill_open"],
                    "entry_premium_rate": entry_fill["premium_rate"],
                    "exit_signal_time": fill_row["signal_time"],
                    "exit_time": fill_row["fill_time"],
                    "exit_price": fill_open,
                    "exit_raw_price": raw_open,
                    "exit_premium_rate": fill_row["premium_rate"],
                    "holding_minutes": int((pd.Timestamp(fill_row["fill_time"]) - pd.Timestamp(entry_fill["fill_time"])).total_seconds() // 60),
                    "gross_return": gross_ret,
                    "return": net_ret,
                    "commission_bps": commission_bps,
                    "slippage_bps": slippage_bps,
                }
            )
            position = 0
            entry_fill = None

    if position == 1 and entry_fill is not None:
        last_bar = bars.iloc[-1]
        raw_close = float(last_bar["close"])
        exit_price = raw_close * (1.0 - slippage_bps / 10000.0)
        gross_ret = exit_price / float(entry_fill["fill_open"]) - 1.0
        net_ret = ((1.0 - commission_bps / 10000.0) * exit_price) / ((1.0 + commission_bps / 10000.0) * float(entry_fill["fill_open"])) - 1.0
        trades.append(
            {
                "symbol": symbol,
                "entry_signal_time": entry_fill["signal_time"],
                "entry_time": entry_fill["fill_time"],
                "entry_price": entry_fill["fill_open"],
                "entry_raw_price": entry_fill["raw_fill_open"],
                "entry_premium_rate": entry_fill["premium_rate"],
                "exit_signal_time": pd.NaT,
                "exit_time": pd.Timestamp(last_bar["datetime"]),
                "exit_price": exit_price,
                "exit_raw_price": raw_close,
                "exit_premium_rate": float(last_bar["premium_rate"]) if pd.notna(last_bar["premium_rate"]) else np.nan,
                "holding_minutes": int((pd.Timestamp(last_bar["datetime"]) - pd.Timestamp(entry_fill["fill_time"])).total_seconds() // 60),
                "gross_return": gross_ret,
                "return": net_ret,
                "commission_bps": commission_bps,
                "slippage_bps": slippage_bps,
            }
        )

    fills_df = pd.DataFrame(fills)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        metrics_df = pd.DataFrame([{"metric": "trade_count", "value": 0.0}])
    else:
        trades_df["equity"] = (1.0 + trades_df["return"]).cumprod()
        equity = trades_df["equity"]
        dd = equity / equity.cummax() - 1.0
        metrics_df = pd.DataFrame(
            [
                {"metric": "trade_count", "value": float(len(trades_df))},
                {"metric": "win_rate", "value": float((trades_df["return"] > 0).mean())},
                {"metric": "avg_trade_return", "value": float(trades_df["return"].mean())},
                {"metric": "median_trade_return", "value": float(trades_df["return"].median())},
                {"metric": "gross_cum_return", "value": float((1.0 + trades_df["gross_return"]).prod() - 1.0)},
                {"metric": "cum_return", "value": float(equity.iloc[-1] - 1.0)},
                {"metric": "max_drawdown", "value": float(dd.min())},
                {"metric": "avg_holding_minutes", "value": float(trades_df["holding_minutes"].mean())},
                {"metric": "avg_entry_premium_bps", "value": float(trades_df["entry_premium_rate"].mean() * 10000.0)},
                {"metric": "avg_exit_premium_bps", "value": float(trades_df["exit_premium_rate"].mean() * 10000.0)},
                {"metric": "blocked_limit_count", "value": float(fills_df["blocked_reason"].isin(["limit_up", "limit_down"]).sum())},
                {"metric": "blocked_premium_count", "value": float(fills_df["blocked_reason"].isin(["premium_too_high", "discount_too_high"]).sum())},
            ]
        )

    return IntradayArtifacts(
        bars=bars,
        signals=fills_df,
        trades=trades_df,
        metrics=metrics_df,
    )


def walk_forward_logistic(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    tradable_prefix: str,
    name: str,
) -> ModelArtifacts:
    dataset = dataset.copy().reset_index(drop=True)
    min_train = max(40, int(len(dataset) * 0.4))
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
        current = dataset.iloc[[i]]
        raw_probability = float(pipeline.predict_proba(current[feature_cols])[0, 1])
        probability, overlay_reason = apply_wave_fib_overlay(raw_probability, current, name)
        pred = int(probability >= 0.5)
        trend_pred = int(
            (current.filter(regex="^nq_sma_ratio_20$").iloc[0, 0] > 0)
            and (current.filter(regex="^nq_ret_5$").iloc[0, 0] > 0)
        )
        row = {
            "feature_date": current["date"].iloc[0],
            "target_date": current["target_date"].iloc[0],
            "actual_target": int(current["target"].iloc[0]),
            "current_state": int(current["current_state"].iloc[0]),
            "raw_probability": raw_probability,
            "probability": probability,
            "model_pred": pred,
            "wave_overlay": overlay_reason,
            "persist_pred": int(current["current_state"].iloc[0]),
            "trend_pred": trend_pred,
            "buyhold_pred": 1,
            "trade_ret": float(current["trade_ret"].iloc[0]),
            f"{tradable_prefix}_close": float(current[f"{tradable_prefix}_close"].iloc[0]),
        }
        rows.append(row)

    predictions = pd.DataFrame(rows)
    for pred_col in ("model_pred", "persist_pred", "trend_pred", "buyhold_pred"):
        predictions[f"{pred_col}_ret"] = predictions[pred_col] * predictions["trade_ret"]
        predictions[f"{pred_col}_equity"] = (1.0 + predictions[f"{pred_col}_ret"]).cumprod()

    predictions["teacher_ret"] = predictions["actual_target"] * predictions["trade_ret"]
    predictions["teacher_equity"] = (1.0 + predictions["teacher_ret"]).cumprod()

    metrics_rows = [
        build_metrics(predictions, "model", "model_pred"),
        build_metrics(predictions, "persist", "persist_pred"),
        build_metrics(predictions, "trend", "trend_pred"),
        build_metrics(predictions, "buyhold", "buyhold_pred"),
    ]
    teacher_cum, teacher_sharpe, teacher_max_dd = evaluate_return_series(predictions["teacher_ret"])
    metrics_rows.append(
        {
            "strategy": "teacher",
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "auc": 1.0,
            "cum_return": teacher_cum,
            "sharpe": teacher_sharpe,
            "max_drawdown": teacher_max_dd,
            "avg_position": float(predictions["actual_target"].mean()),
        }
    )
    metrics = pd.DataFrame(metrics_rows)
    metrics.insert(0, "market", name)

    pipeline.fit(dataset[feature_cols], dataset["target"].astype(int))
    coefs = pipeline.named_steps["model"].coef_[0]
    coefficients = pd.DataFrame({"feature": feature_cols, "coef": coefs})
    coefficients["abs_coef"] = coefficients["coef"].abs()
    coefficients = coefficients.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coefficients.insert(0, "market", name)
    return ModelArtifacts(
        name=name,
        dataset=dataset,
        predictions=predictions,
        metrics=metrics,
        coefficients=coefficients,
        pipeline=pipeline,
        feature_cols=feature_cols,
    )


def plot_backtest(us_art: ModelArtifacts, cn_art: ModelArtifacts, cn_intraday: IntradayArtifacts, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)
    for ax, art, title in zip(axes[:2], [us_art, cn_art], ["US QQQ Daily", "CN Nasdaq ETF Daily"]):
        frame = art.predictions.copy()
        x = pd.to_datetime(frame["target_date"])
        ax.plot(x, frame["teacher_equity"], label="Teacher Signal", linewidth=2.2, color="#1f77b4")
        ax.plot(x, frame["model_pred_equity"], label="NQ Model", linewidth=2.0, color="#d62728")
        ax.plot(x, frame["persist_pred_equity"], label="Persist Baseline", linewidth=1.5, color="#2ca02c")
        ax.plot(x, frame["trend_pred_equity"], label="Trend Baseline", linewidth=1.5, color="#9467bd")
        ax.plot(x, frame["buyhold_pred_equity"], label="Buy & Hold", linewidth=1.5, color="#7f7f7f")
        ax.set_title(title)
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", ncol=3)
    intraday_ax = axes[2]
    if not cn_intraday.trades.empty:
        trade_x = pd.to_datetime(cn_intraday.trades["exit_time"])
        intraday_ax.plot(trade_x, cn_intraday.trades["equity"], label="CN 5m Intraday", linewidth=2.0, color="#ff7f0e")
    intraday_ax.set_title("CN Nasdaq ETF 5m Intraday")
    intraday_ax.set_ylabel("Equity")
    intraday_ax.grid(alpha=0.25)
    intraday_ax.legend(loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("NQ-driven Quant Backtest")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_excel(
    output_path: Path,
    us_art: ModelArtifacts,
    cn_art: ModelArtifacts,
    us_sessions: pd.DataFrame,
    cn_sessions: pd.DataFrame,
    cn_symbol: str,
    nq_roll_info: pd.DataFrame,
    nq_archive_status: pd.DataFrame,
    nq_source_note: str,
    cn_intraday: IntradayArtifacts,
    live_signals: list[LiveSignal],
    intraday_source: str,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.concat([us_art.metrics, cn_art.metrics], ignore_index=True).to_excel(writer, sheet_name="metrics", index=False)
        us_art.coefficients.head(20).to_excel(writer, sheet_name="top_features", index=False)
        cn_art.coefficients.head(20).to_excel(writer, sheet_name="top_features", index=False, startrow=24)
        us_art.predictions.to_excel(writer, sheet_name="us_predictions", index=False)
        cn_art.predictions.to_excel(writer, sheet_name="cn_predictions", index=False)
        us_art.dataset.to_excel(writer, sheet_name="us_dataset", index=False)
        cn_art.dataset.to_excel(writer, sheet_name="cn_dataset", index=False)
        us_sessions.to_excel(writer, sheet_name="us_sessions", index=False)
        cn_sessions.to_excel(writer, sheet_name="cn_sessions", index=False)
        nq_roll_info.to_excel(writer, sheet_name="nq_roll_info", index=False)
        nq_archive_status.to_excel(writer, sheet_name="nq_archive_status", index=False)
        cn_intraday.metrics.to_excel(writer, sheet_name="cn_intraday_metrics", index=False)
        cn_intraday.trades.to_excel(writer, sheet_name="cn_intraday_trades", index=False)
        cn_intraday.signals.to_excel(writer, sheet_name="cn_intraday_signals", index=False)
        pd.DataFrame([signal.__dict__ for signal in live_signals]).to_excel(writer, sheet_name="live_signals", index=False)
        notes = pd.DataFrame(
            [
                ("nq_source", nq_source_note),
                ("us_signal_session", "US events before 12:00 Asia/Shanghai are shifted back one day to the prior US session."),
                ("cn_proxy", f"{cn_symbol} is used as the onshore Nasdaq ETF proxy."),
                ("cn_backtest_limit", "CN daily backtest understates intraday edges; 5m intraday trades are exported separately."),
                ("cn_intraday_source", f"CN 5m bars are fetched from {intraday_source}."),
                ("nq_roll_rule", "CME equity index roll date is the Monday prior to the third Friday of the expiry month."),
                ("cn_trade_cost", "CN intraday backtest applies commission, slippage, price limit blocking, and premium/discount filtering."),
                ("label_definition", "Target is next-session final state, predicted with information available at the prior session close."),
            ],
            columns=["item", "value"],
        )
        notes.to_excel(writer, sheet_name="notes", index=False)


def print_summary(
    us_art: ModelArtifacts,
    cn_art: ModelArtifacts,
    cn_intraday: IntradayArtifacts,
    live_signals: list[LiveSignal],
    output_path: Path,
    plot_path: Path,
    live_output_path: Path,
) -> None:
    metrics = pd.concat([us_art.metrics, cn_art.metrics], ignore_index=True)
    model_rows = metrics[metrics["strategy"].eq("model")].copy()
    teacher_rows = metrics[metrics["strategy"].eq("teacher")].copy()
    print(f"Wrote workbook: {output_path}")
    print(f"Wrote chart: {plot_path}")
    print(f"Wrote live CSV: {live_output_path}")
    print("\nModel metrics:")
    print(model_rows[["market", "accuracy", "f1", "cum_return", "sharpe", "max_drawdown"]].to_string(index=False))
    print("\nTeacher metrics:")
    print(teacher_rows[["market", "cum_return", "sharpe", "max_drawdown"]].to_string(index=False))
    if not cn_intraday.metrics.empty:
        print("\nCN intraday metrics:")
        print(cn_intraday.metrics.to_string(index=False))
    print("\nLive signals:")
    live_df = pd.DataFrame([signal.__dict__ for signal in live_signals])
    print(
        live_df[
            ["market", "feature_date", "next_session_date", "current_state", "probability_long", "recommended_state", "tradable_symbol", "tradable_close"]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build an NQ-driven quant model from extracted QQQ / CN ETF signals.")
    parser.add_argument("--events", default=str(root_dir / "data/raw/wsquant_market_timeline.xlsx"), help="Path to the signal timeline workbook.")
    parser.add_argument("--output", default=str(root_dir / "data/legacy/nq_quant_model_results.xlsx"), help="Excel output path.")
    parser.add_argument("--plot", default=str(root_dir / "data/legacy/nq_quant_model_backtest.png"), help="Backtest plot path.")
    parser.add_argument("--live-output", default=str(root_dir / "data/legacy/nq_live_signals.csv"), help="CSV output path for next-session live signals.")
    parser.add_argument("--cn-symbol", default="159941.SZ", help="Onshore Nasdaq ETF proxy ticker.")
    parser.add_argument("--range", default="2y", help="Yahoo chart range for daily bars.")
    parser.add_argument("--intraday-range", default="60d", help="Yahoo chart range for CN intraday bars.")
    parser.add_argument("--intraday-source", default="sina", choices=["sina", "yahoo"], help="Intraday source for CN ETF bars.")
    parser.add_argument("--nq-archive-dir", default=str(root_dir / "data/contracts/nq_contract_archive"), help="Directory for archived NQ single-month contracts.")
    parser.add_argument("--strict-nq-archive", action="store_true", help="Require full NQ single-month archive coverage; otherwise fail.")
    parser.add_argument("--cn-commission-bps", type=float, default=2.0, help="CN ETF commission in bps per side.")
    parser.add_argument("--cn-slippage-bps", type=float, default=3.0, help="CN ETF slippage in bps per side.")
    parser.add_argument("--cn-premium-limit-bps", type=float, default=120.0, help="Max allowed buy premium / sell discount in bps.")
    parser.add_argument("--cn-limit-pct", type=float, default=0.10, help="CN ETF daily price limit ratio.")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    events_path = Path(args.events).resolve()
    output_path = Path(args.output).resolve()
    plot_path = Path(args.plot).resolve()
    live_output_path = Path(args.live_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    live_output_path.parent.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.nq_archive_dir).resolve()

    us_sessions, cn_sessions = build_signal_sessions(events_path)

    qqq_daily = fetch_yahoo_daily("QQQ", args.range)
    cn_daily = fetch_yahoo_daily(args.cn_symbol, args.range)
    archive_years = range(pd.Timestamp.today().year - 1, pd.Timestamp.today().year + 2)
    nq_archive_status = archive_nq_contracts(archive_dir, archive_years, date_range=args.range)
    archived_nq_daily, archived_roll_info, archive_complete = build_archived_nq_main(
        archive_dir,
        qqq_daily["date"].min(),
        qqq_daily["date"].max(),
    )
    if archive_complete:
        nq_daily = archived_nq_daily
        nq_roll_info = archived_roll_info
        nq_source_note = "True archived single-month NQ splice."
    else:
        if args.strict_nq_archive:
            raise RuntimeError("NQ single-month archive is incomplete for the requested date range.")
        nq_daily, nq_roll_info = build_roll_adjusted_nq_main(args.range)
        nq_source_note = "Archive incomplete; fallback to roll-adjusted NQ=F proxy for uncovered history."

    qqq = make_price_features(qqq_daily, "qqq")
    nq = make_price_features(nq_daily, "nq")
    cnetf = make_price_features(cn_daily, "cnetf")

    us_dataset = build_us_dataset(us_sessions, qqq, nq, cnetf)
    cn_dataset = build_cn_dataset(cn_sessions, cnetf, qqq, nq)

    us_features = pick_feature_columns(us_dataset)
    cn_features = pick_feature_columns(cn_dataset)

    us_art = walk_forward_logistic(us_dataset, us_features, "qqq", "美股")
    cn_art = walk_forward_logistic(cn_dataset, cn_features, "cnetf", "场内")

    us_live_frame = build_us_live_frame(us_sessions, qqq, nq, cnetf)
    cn_live_frame = build_cn_live_frame(cn_sessions, cnetf, qqq, nq)
    live_signals = [
        build_live_signal("美股", us_live_frame, us_art, "QQQ", "qqq_close"),
        build_live_signal("场内", cn_live_frame, cn_art, args.cn_symbol, "cnetf_close"),
    ]

    cn_intraday = build_cn_intraday_backtest(
        events_path,
        symbol=args.cn_symbol,
        cn_daily=cn_daily,
        nq_daily=nq_daily,
        start_date="2026-01-28",
        interval="5m",
        date_range=args.intraday_range,
        source=args.intraday_source,
        commission_bps=args.cn_commission_bps,
        slippage_bps=args.cn_slippage_bps,
        premium_limit_bps=args.cn_premium_limit_bps,
        discount_limit_bps=args.cn_premium_limit_bps,
        limit_pct=args.cn_limit_pct,
    )

    write_excel(
        output_path,
        us_art,
        cn_art,
        us_sessions,
        cn_sessions,
        args.cn_symbol,
        nq_roll_info,
        nq_archive_status,
        nq_source_note,
        cn_intraday,
        live_signals,
        args.intraday_source,
    )
    pd.DataFrame([signal.__dict__ for signal in live_signals]).to_csv(live_output_path, index=False)
    plot_backtest(us_art, cn_art, cn_intraday, plot_path)
    print_summary(us_art, cn_art, cn_intraday, live_signals, output_path, plot_path, live_output_path)


if __name__ == "__main__":
    main()
