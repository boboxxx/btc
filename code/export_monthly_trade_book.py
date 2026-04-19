#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MarketConfig:
    workbook: str
    prediction_sheet: str
    dataset_sheet: str
    price_column: str
    symbol: str
    market: str


MARKETS = {
    "us": MarketConfig(
        workbook=str(ROOT_DIR / "data/signals/us/us_qqq_signal_results.xlsx"),
        prediction_sheet="us_predictions",
        dataset_sheet="us_dataset",
        price_column="qqq_close",
        symbol="QQQ",
        market="美股",
    ),
    "cn": MarketConfig(
        workbook=str(ROOT_DIR / "data/signals/cn/cn_nasdaq_signal_results.xlsx"),
        prediction_sheet="cn_predictions",
        dataset_sheet="cn_dataset",
        price_column="cnetf_close",
        symbol="159941.SZ",
        market="场内",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a monthly trade book from signal backtest workbooks.")
    parser.add_argument("--market", required=True, choices=sorted(MARKETS))
    parser.add_argument("--month", default=pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m"))
    parser.add_argument("--workbook", default=None)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data/reports"))
    return parser.parse_args()


def load_market_frames(config: MarketConfig, workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_excel(workbook_path, sheet_name=config.prediction_sheet)
    dataset = pd.read_excel(workbook_path, sheet_name=config.dataset_sheet)
    predictions["feature_bar_time"] = pd.to_datetime(predictions["feature_bar_time"], errors="coerce")
    dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce")
    return predictions, dataset


def monthly_actions(predictions: pd.DataFrame, dataset: pd.DataFrame, config: MarketConfig, month: str) -> pd.DataFrame:
    month_start = pd.Timestamp(f"{month}-01")
    month_end = month_start + pd.offsets.MonthBegin(1)
    price_map = dataset.set_index("datetime")[config.price_column].astype(float)

    monthly = predictions.loc[
        predictions["feature_bar_time"].ge(month_start) & predictions["feature_bar_time"].lt(month_end)
    ].copy()
    monthly = monthly.sort_values("feature_bar_time").reset_index(drop=True)
    monthly["action"] = ""
    monthly.loc[(monthly["prior_position"] == 0) & (monthly["position"] == 1), "action"] = "buy"
    monthly.loc[(monthly["prior_position"] == 1) & (monthly["position"] == 0), "action"] = "sell"
    monthly = monthly.loc[monthly["action"] != ""].copy()
    monthly["signal_price"] = monthly["feature_bar_time"].map(price_map)
    monthly["market"] = config.market
    monthly["symbol"] = config.symbol
    return monthly


def build_trade_book(
    actions: pd.DataFrame,
    predictions: pd.DataFrame,
    config: MarketConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_rows = actions[
        [
            "market",
            "symbol",
            "feature_bar_time",
            "action",
            "signal_price",
            "probability_long",
            "expected_edge",
            "wave_overlay",
            "strategy_ret",
            "stop_triggered",
        ]
    ].copy()

    trades: list[dict[str, object]] = []
    open_trade: dict[str, object] | None = None

    for row in actions.itertuples(index=False):
        if row.action == "buy":
            if open_trade is None:
                open_trade = {
                    "market": config.market,
                    "symbol": config.symbol,
                    "entry_time": row.feature_bar_time,
                    "entry_price": row.signal_price,
                    "entry_probability": row.probability_long,
                    "entry_expected_edge": row.expected_edge,
                    "entry_overlay": row.wave_overlay,
                }
            continue

        if row.action == "sell" and open_trade is not None:
            entry_price = float(open_trade["entry_price"]) if pd.notna(open_trade["entry_price"]) else float("nan")
            exit_price = float(row.signal_price) if pd.notna(row.signal_price) else float("nan")
            hold_window = predictions.loc[
                predictions["feature_bar_time"].between(open_trade["entry_time"], row.feature_bar_time, inclusive="both")
            ]
            hold_bars = int(hold_window.shape[0])
            gross_return = exit_price / entry_price - 1.0 if pd.notna(entry_price) and pd.notna(exit_price) else float("nan")
            trades.append(
                {
                    **open_trade,
                    "exit_time": row.feature_bar_time,
                    "exit_price": exit_price,
                    "exit_probability": row.probability_long,
                    "exit_expected_edge": row.expected_edge,
                    "exit_overlay": row.wave_overlay,
                    "hold_bars": hold_bars,
                    "gross_return": gross_return,
                    "stop_triggered_on_exit": int(row.stop_triggered),
                    "status": "closed",
                }
            )
            open_trade = None

    if open_trade is not None:
        trades.append(
            {
                **open_trade,
                "exit_time": pd.NaT,
                "exit_price": pd.NA,
                "exit_probability": pd.NA,
                "exit_expected_edge": pd.NA,
                "exit_overlay": pd.NA,
                "hold_bars": pd.NA,
                "gross_return": pd.NA,
                "stop_triggered_on_exit": pd.NA,
                "status": "open",
            }
        )

    trades_df = pd.DataFrame(trades)
    return action_rows, trades_df


def trade_summary(trades: pd.DataFrame, month: str, config: MarketConfig) -> pd.DataFrame:
    closed = trades.loc[trades["status"].eq("closed")].copy()
    if closed.empty:
        return pd.DataFrame(
            [
                {
                    "market": config.market,
                    "symbol": config.symbol,
                    "month": month,
                    "closed_trades": 0,
                    "win_rate": 0.0,
                    "avg_gross_return": 0.0,
                    "compound_return": 0.0,
                    "avg_hold_bars": 0.0,
                }
            ]
        )
    compound_return = float((1.0 + closed["gross_return"].fillna(0.0)).prod() - 1.0)
    return pd.DataFrame(
        [
            {
                "market": config.market,
                "symbol": config.symbol,
                "month": month,
                "closed_trades": int(len(closed)),
                "win_rate": float((closed["gross_return"] > 0).mean()),
                "avg_gross_return": float(closed["gross_return"].mean()),
                "compound_return": compound_return,
                "avg_hold_bars": float(closed["hold_bars"].mean()),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    config = MARKETS[args.market]
    workbook_path = Path(args.workbook or config.workbook).resolve()
    output_dir = Path(args.output_dir).resolve() / args.month
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, dataset = load_market_frames(config, workbook_path)
    actions = monthly_actions(predictions, dataset, config, args.month)
    action_rows, trades = build_trade_book(actions, predictions, config)
    summary = trade_summary(trades, args.month, config)

    csv_path = output_dir / f"{args.market}_{args.month}_trade_book.csv"
    xlsx_path = output_dir / f"{args.market}_{args.month}_trade_book.xlsx"

    trades.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        action_rows.to_excel(writer, sheet_name="actions", index=False)
        trades.to_excel(writer, sheet_name="trades", index=False)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote workbook: {xlsx_path}")


if __name__ == "__main__":
    main()
