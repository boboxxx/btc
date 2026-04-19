#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from build_5m_signal_program import (
    CN_EXIT_THRESHOLD,
    US_EXIT_THRESHOLD,
    is_cn_bearish_overlay,
    is_us_bearish_overlay,
    should_hold_cn_trend,
    should_hold_us_trend,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AuditConfig:
    workbook: str
    trade_book: str
    prediction_sheet: str
    dataset_sheet: str
    price_column: str
    market: str
    symbol: str
    exit_threshold: float


CONFIGS = {
    "us": AuditConfig(
        workbook=str(ROOT_DIR / "data/signals/us/us_qqq_signal_results.xlsx"),
        trade_book=str(ROOT_DIR / "data/reports/2026-04/us_2026-04_trade_book.xlsx"),
        prediction_sheet="us_predictions",
        dataset_sheet="us_dataset",
        price_column="qqq_close",
        market="美股",
        symbol="QQQ",
        exit_threshold=US_EXIT_THRESHOLD,
    ),
    "cn": AuditConfig(
        workbook=str(ROOT_DIR / "data/signals/cn/cn_nasdaq_signal_results.xlsx"),
        trade_book=str(ROOT_DIR / "data/reports/2026-04/cn_2026-04_trade_book.xlsx"),
        prediction_sheet="cn_predictions",
        dataset_sheet="cn_dataset",
        price_column="cnetf_close",
        market="场内",
        symbol="159941.SZ",
        exit_threshold=CN_EXIT_THRESHOLD,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit trade exits and flag likely early sells.")
    parser.add_argument("--market", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--month", default="2026-04")
    parser.add_argument("--lookahead-bars", type=int, default=12)
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--trade-book", default=None)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data/reports"))
    return parser.parse_args()


def classify_exit(
    market: str,
    exit_overlay: str,
    stop_triggered: int,
    trend_hold_active: bool,
    exit_probability: float,
    exit_threshold: float,
    missed_upside_lookahead: float,
    gross_return: float,
) -> str:
    if int(stop_triggered) == 1:
        return "stop_loss"
    if market == "美股" and is_us_bearish_overlay(str(exit_overlay)):
        return "bearish_overlay"
    if market == "场内" and is_cn_bearish_overlay(str(exit_overlay)):
        return "bearish_overlay"
    if trend_hold_active and missed_upside_lookahead > 0.005:
        return "early_sell_trend_intact"
    if gross_return > 0 and missed_upside_lookahead > 0.003:
        return "premature_profit_take"
    if gross_return < 0 and missed_upside_lookahead > 0.003:
        return "premature_loss_cut"
    if exit_probability <= exit_threshold:
        return "probability_breakdown"
    return "other"


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.market]
    workbook = Path(args.workbook or config.workbook).resolve()
    trade_book = Path(args.trade_book or config.trade_book).resolve()
    out_dir = Path(args.output_dir).resolve() / args.month
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_excel(workbook, sheet_name=config.prediction_sheet)
    dataset = pd.read_excel(workbook, sheet_name=config.dataset_sheet)
    trades = pd.read_excel(trade_book, sheet_name="trades")

    preds["feature_bar_time"] = pd.to_datetime(preds["feature_bar_time"], errors="coerce")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    trades = trades.loc[trades["status"].eq("closed")].copy()

    dataset_indexed = dataset.set_index("datetime").sort_index()
    preds_indexed = preds.set_index("feature_bar_time").sort_index()
    audit_rows: list[dict[str, object]] = []

    for trade in trades.itertuples(index=False):
        if pd.isna(trade.exit_time):
            continue
        if trade.exit_time not in preds_indexed.index or trade.exit_time not in dataset_indexed.index:
            continue

        exit_pred = preds_indexed.loc[trade.exit_time]
        if isinstance(exit_pred, pd.DataFrame):
            exit_pred = exit_pred.iloc[-1]
        exit_row = dataset_indexed.loc[trade.exit_time]
        if isinstance(exit_row, pd.DataFrame):
            exit_row = exit_row.iloc[-1]

        future_prices = dataset_indexed.loc[dataset_indexed.index > trade.exit_time, config.price_column].head(args.lookahead_bars)
        future_max_price = float(future_prices.max()) if not future_prices.empty else np.nan
        future_last_price = float(future_prices.iloc[-1]) if not future_prices.empty else np.nan
        exit_price = float(trade.exit_price)
        missed_upside = future_max_price / exit_price - 1.0 if pd.notna(future_max_price) else np.nan
        future_path_return = future_last_price / exit_price - 1.0 if pd.notna(future_last_price) else np.nan

        if config.market == "美股":
            trend_hold_active = should_hold_us_trend(
                exit_row,
                float(exit_pred["probability_long"]),
                float(exit_pred["smoothed_probability"]),
                str(exit_pred["wave_overlay"]),
            )
        else:
            trend_hold_active = should_hold_cn_trend(
                exit_row,
                float(exit_pred["probability_long"]),
                float(exit_pred["smoothed_probability"]),
                str(exit_pred["wave_overlay"]),
            )

        exit_label = classify_exit(
            config.market,
            str(trade.exit_overlay),
            int(trade.stop_triggered_on_exit),
            bool(trend_hold_active),
            float(trade.exit_probability),
            config.exit_threshold,
            float(missed_upside) if pd.notna(missed_upside) else 0.0,
            float(trade.gross_return),
        )
        audit_rows.append(
            {
                "market": config.market,
                "symbol": config.symbol,
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "hold_bars": trade.hold_bars,
                "gross_return": trade.gross_return,
                "exit_probability": trade.exit_probability,
                "exit_expected_edge": trade.exit_expected_edge,
                "exit_overlay": trade.exit_overlay,
                "stop_triggered_on_exit": trade.stop_triggered_on_exit,
                "trend_hold_active_at_exit": int(trend_hold_active),
                "missed_upside_lookahead": missed_upside,
                "future_path_return_lookahead": future_path_return,
                "exit_classification": exit_label,
            }
        )

    audit = pd.DataFrame(audit_rows)
    summary = (
        audit.groupby("exit_classification", dropna=False)
        .agg(
            trades=("exit_classification", "size"),
            avg_gross_return=("gross_return", "mean"),
            avg_missed_upside=("missed_upside_lookahead", "mean"),
            trend_hold_active_count=("trend_hold_active_at_exit", "sum"),
        )
        .reset_index()
        .sort_values(["trades", "avg_missed_upside"], ascending=[False, False])
    )

    csv_path = out_dir / f"{args.market}_{args.month}_exit_audit.csv"
    xlsx_path = out_dir / f"{args.market}_{args.month}_exit_audit.xlsx"
    audit.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        audit.to_excel(writer, sheet_name="audit", index=False)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote workbook: {xlsx_path}")


if __name__ == "__main__":
    main()
