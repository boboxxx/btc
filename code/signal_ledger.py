from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LEDGER_COLUMNS = [
    "generated_at",
    "strategy_name",
    "market",
    "feature_date",
    "next_session_date",
    "current_state",
    "raw_probability",
    "probability_long",
    "recommended_state",
    "action",
    "recent_wrong_streak",
    "cooldown_suggested",
    "symbol",
    "signal_price",
    "wave_overlay",
    "top_drivers",
    "status",
    "feature_close",
    "next_close",
    "realized_return",
    "signal_correct",
]


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_signal_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_ledger()
    ledger = pd.read_csv(path)
    if ledger.empty:
        return empty_ledger()
    for col in ("generated_at", "feature_date", "next_session_date"):
        if col in ledger.columns:
            ledger[col] = pd.to_datetime(ledger[col], errors="coerce")
            if getattr(ledger[col].dt, "tz", None) is not None:
                ledger[col] = ledger[col].dt.tz_localize(None)
    return ledger


def save_signal_ledger(path: Path, ledger: pd.DataFrame) -> None:
    out = ledger.copy()
    for col in LEDGER_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[LEDGER_COLUMNS].sort_values(["market", "next_session_date", "generated_at"]).reset_index(drop=True)
    out.to_csv(path, index=False)


def append_live_signals(path: Path, live_df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    ledger = load_signal_ledger(path)
    rows = live_df.copy()
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    rows["generated_at"] = now
    rows["strategy_name"] = strategy_name
    rows["status"] = "pending"
    rows["feature_close"] = np.nan
    rows["next_close"] = np.nan
    rows["realized_return"] = np.nan
    rows["signal_correct"] = np.nan
    for col in ("generated_at", "feature_date", "next_session_date"):
        rows[col] = pd.to_datetime(rows[col], errors="coerce")
    for col in LEDGER_COLUMNS:
        if col not in rows.columns:
            rows[col] = np.nan
    if ledger.empty:
        combined = rows[LEDGER_COLUMNS].copy()
    else:
        records = ledger[LEDGER_COLUMNS].to_dict("records") + rows[LEDGER_COLUMNS].to_dict("records")
        combined = pd.DataFrame.from_records(records, columns=LEDGER_COLUMNS)
    combined = combined.sort_values(["market", "next_session_date", "generated_at"])
    combined = combined.drop_duplicates(subset=["strategy_name", "market", "next_session_date"], keep="last")
    save_signal_ledger(path, combined)
    return combined


def reconcile_signal_ledger(
    path: Path,
    price_history: dict[str, pd.DataFrame],
    strategy_name: str,
) -> pd.DataFrame:
    ledger = load_signal_ledger(path)
    if ledger.empty:
        return ledger
    ledger = ledger.copy()
    for symbol, history in price_history.items():
        frame = history.copy().sort_values("date")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.drop_duplicates(subset=["date"], keep="last")
        close_map = frame.set_index("date")["adjclose"].astype(float).to_dict()
        symbol_mask = ledger["strategy_name"].eq(strategy_name) & ledger["symbol"].eq(symbol)
        for idx in ledger.index[symbol_mask]:
            session_date = pd.Timestamp(ledger.at[idx, "next_session_date"]).normalize()
            feature_date = pd.Timestamp(ledger.at[idx, "feature_date"]).normalize()
            if pd.isna(session_date) or pd.isna(feature_date):
                continue
            if feature_date not in close_map or session_date not in close_map:
                continue
            feature_close = float(close_map[feature_date])
            next_close = float(close_map[session_date])
            realized_return = next_close / feature_close - 1.0
            recommended_state = int(ledger.at[idx, "recommended_state"])
            signal_correct = int((recommended_state == 1 and realized_return > 0) or (recommended_state == 0 and realized_return <= 0))
            ledger.at[idx, "feature_close"] = feature_close
            ledger.at[idx, "next_close"] = next_close
            ledger.at[idx, "realized_return"] = realized_return
            ledger.at[idx, "signal_correct"] = signal_correct
            ledger.at[idx, "status"] = "resolved"
    save_signal_ledger(path, ledger)
    return ledger


def latest_recommended_state(
    ledger: pd.DataFrame,
    strategy_name: str,
    market: str,
    default: int,
    as_of_date: pd.Timestamp | None = None,
) -> int:
    if ledger.empty:
        return default
    rows = ledger[(ledger["strategy_name"].eq(strategy_name)) & (ledger["market"].eq(market))].copy()
    if rows.empty:
        return default
    rows["next_session_date"] = pd.to_datetime(rows["next_session_date"], errors="coerce")
    if as_of_date is not None:
        rows = rows[rows["next_session_date"].le(pd.Timestamp(as_of_date).normalize())]
        if rows.empty:
            return default
    rows = rows.sort_values(["next_session_date", "generated_at"])
    value = rows["recommended_state"].dropna()
    if value.empty:
        return default
    return int(value.iloc[-1])


def market_feedback(ledger: pd.DataFrame, strategy_name: str, market: str) -> dict[str, int]:
    rows = ledger[
        ledger["strategy_name"].eq(strategy_name)
        & ledger["market"].eq(market)
        & ledger["status"].eq("resolved")
        & ledger["signal_correct"].notna()
    ].copy()
    if rows.empty:
        return {"recent_wrong_streak": 0, "cooldown_suggested": 0}
    rows = rows.sort_values(["next_session_date", "generated_at"], ascending=False)
    wrong_streak = 0
    for value in rows["signal_correct"]:
        if int(value) == 0:
            wrong_streak += 1
        else:
            break
    cooldown = 1 if wrong_streak >= 2 else 0
    return {"recent_wrong_streak": wrong_streak, "cooldown_suggested": cooldown}
