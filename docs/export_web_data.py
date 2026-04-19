#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export local signal outputs into a GitHub Pages JSON payload.")
    parser.add_argument("--five-minute-live", default="five_minute_live_signals.csv")
    parser.add_argument("--five-minute-history", default="five_minute_signal_history.csv")
    parser.add_argument("--btc-live", default="btc_live_signals.csv")
    parser.add_argument("--btc-history", default="btc_signal_history.csv")
    parser.add_argument("--open-nowcast-live", default="open_nowcast_live.csv")
    parser.add_argument("--output", default="docs/data/dashboard.json")
    parser.add_argument("--recent-limit", type=int, default=20)
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_datetime_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def iso_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


def scalar_or_none(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return iso_or_none(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def market_key(name: str) -> str:
    mapping = {
        "美股": "us",
        "场内": "cn",
        "比特币": "btc",
        "美股开盘": "us_open",
        "场内开盘": "cn_open",
    }
    return mapping.get(name, name)


def intraday_payload(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    parsed = parse_datetime_columns(
        frame,
        ["feature_bar_time", "updated_at", "trade_date", "position_since", "last_action_time", "run_at"],
    ).sort_values(["market", "symbol"])
    payload: dict[str, dict[str, object]] = {}
    for row in parsed.to_dict(orient="records"):
        key = market_key(str(row["market"]))
        payload[key] = {
            "market": scalar_or_none(row.get("market")),
            "symbol": scalar_or_none(row.get("symbol")),
            "feature_bar_time": iso_or_none(row.get("feature_bar_time")),
            "updated_at": iso_or_none(row.get("updated_at")),
            "trade_date": iso_or_none(row.get("trade_date")),
            "current_state": scalar_or_none(row.get("current_state")),
            "recommended_state": scalar_or_none(row.get("recommended_state")),
            "action": scalar_or_none(row.get("action")),
            "market_phase": scalar_or_none(row.get("market_phase")),
            "wave_overlay": scalar_or_none(row.get("wave_overlay")),
            "raw_probability": scalar_or_none(row.get("raw_probability")),
            "probability_long": scalar_or_none(row.get("probability_long")),
            "smoothed_probability": scalar_or_none(row.get("smoothed_probability")),
            "entry_confirm_count": scalar_or_none(row.get("entry_confirm_count")),
            "nqmain_rt_value": scalar_or_none(row.get("nqmain_rt_value")),
            "signal_price": scalar_or_none(row.get("signal_price")),
            "position_since": iso_or_none(row.get("position_since")),
            "last_action_time": iso_or_none(row.get("last_action_time")),
            "signal_summary": scalar_or_none(row.get("signal_summary")),
            "top_drivers": scalar_or_none(row.get("top_drivers")),
            "run_at": iso_or_none(row.get("run_at")),
        }
    return payload


def open_nowcast_payload(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    parsed = parse_datetime_columns(
        frame,
        ["session_date", "feature_bar_time", "snapshot_time", "signal_effective_time"],
    ).sort_values(["market", "symbol"])
    payload: dict[str, dict[str, object]] = {}
    for row in parsed.to_dict(orient="records"):
        key = market_key(str(row["market"]))
        payload[key] = {
            "market": scalar_or_none(row.get("market")),
            "symbol": scalar_or_none(row.get("symbol")),
            "session_date": iso_or_none(row.get("session_date")),
            "market_phase": scalar_or_none(row.get("market_phase")),
            "feature_bar_time": iso_or_none(row.get("feature_bar_time")),
            "snapshot_time": iso_or_none(row.get("snapshot_time")),
            "snapshot_price": scalar_or_none(row.get("snapshot_price")),
            "signal_effective_time": iso_or_none(row.get("signal_effective_time")),
            "signal_effective_price": scalar_or_none(row.get("signal_effective_price")),
            "current_state": scalar_or_none(row.get("current_state")),
            "raw_probability": scalar_or_none(row.get("raw_probability")),
            "probability_long": scalar_or_none(row.get("probability_long")),
            "decision_threshold": scalar_or_none(row.get("decision_threshold")),
            "recommended_state": scalar_or_none(row.get("recommended_state")),
            "action": scalar_or_none(row.get("action")),
            "wave_overlay": scalar_or_none(row.get("wave_overlay")),
            "current_state_source": scalar_or_none(row.get("current_state_source")),
        }
    return payload


def recent_changes_payload(frame: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    if frame.empty:
        return []
    parsed = parse_datetime_columns(
        frame,
        ["feature_bar_time", "updated_at", "trade_date", "position_since", "last_action_time", "run_at"],
    )
    if "signal_changed" in parsed.columns:
        parsed["signal_changed"] = pd.to_numeric(parsed["signal_changed"], errors="coerce").fillna(0)
        changes = parsed.loc[parsed["signal_changed"].eq(1)].copy()
    else:
        changes = parsed.copy()
    changes = changes.sort_values(["run_at", "updated_at"], ascending=[False, False]).head(limit)
    records: list[dict[str, object]] = []
    for row in changes.to_dict(orient="records"):
        records.append(
            {
                "market": scalar_or_none(row.get("market")),
                "symbol": scalar_or_none(row.get("symbol")),
                "feature_bar_time": iso_or_none(row.get("feature_bar_time")),
                "updated_at": iso_or_none(row.get("updated_at")),
                "trade_date": iso_or_none(row.get("trade_date")),
                "action": scalar_or_none(row.get("action")),
                "recommended_state": scalar_or_none(row.get("recommended_state")),
                "signal_price": scalar_or_none(row.get("signal_price")),
                "nqmain_rt_value": scalar_or_none(row.get("nqmain_rt_value")),
                "signal_summary": scalar_or_none(row.get("signal_summary")),
                "top_drivers": scalar_or_none(row.get("top_drivers")),
                "run_at": iso_or_none(row.get("run_at")),
                "run_seq": scalar_or_none(row.get("run_seq")),
            }
        )
    return records


def recent_history_payload(frame: pd.DataFrame, limit: int) -> list[dict[str, object]]:
    if frame.empty:
        return []
    parsed = parse_datetime_columns(frame, ["feature_bar_time", "updated_at", "run_at"])
    recent = parsed.sort_values(["run_at", "updated_at"], ascending=[False, False]).head(limit)
    records: list[dict[str, object]] = []
    for row in recent.to_dict(orient="records"):
        records.append(
            {
                "market": scalar_or_none(row.get("market")),
                "symbol": scalar_or_none(row.get("symbol")),
                "feature_bar_time": iso_or_none(row.get("feature_bar_time")),
                "updated_at": iso_or_none(row.get("updated_at")),
                "action": scalar_or_none(row.get("action")),
                "recommended_state": scalar_or_none(row.get("recommended_state")),
                "signal_price": scalar_or_none(row.get("signal_price")),
                "run_at": iso_or_none(row.get("run_at")),
                "run_seq": scalar_or_none(row.get("run_seq")),
            }
        )
    return records


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    five_minute_live = read_csv_if_exists(Path(args.five_minute_live))
    five_minute_history = read_csv_if_exists(Path(args.five_minute_history))
    btc_live = read_csv_if_exists(Path(args.btc_live))
    btc_history = read_csv_if_exists(Path(args.btc_history))
    open_nowcast_live = read_csv_if_exists(Path(args.open_nowcast_live))
    combined_live = pd.concat([five_minute_live, btc_live], ignore_index=True, sort=False)
    combined_history = pd.concat([five_minute_history, btc_history], ignore_index=True, sort=False)

    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    return {
        "generated_at": iso_or_none(generated_at),
        "intraday": intraday_payload(combined_live),
        "open_nowcast": open_nowcast_payload(open_nowcast_live),
        "recent_changes": recent_changes_payload(combined_history, args.recent_limit),
        "recent_history": recent_history_payload(combined_history, args.recent_limit),
        "sources": {
            "five_minute_live_exists": not five_minute_live.empty,
            "btc_live_exists": not btc_live.empty,
            "five_minute_history_exists": not five_minute_history.empty,
            "btc_history_exists": not btc_history.empty,
            "open_nowcast_live_exists": not open_nowcast_live.empty,
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote web JSON: {output_path.resolve()}")


if __name__ == "__main__":
    main()
