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
    apply_wave_fib_overlay,
    archive_nq_contracts,
    build_archived_nq_main,
    build_cn_dataset,
    build_cn_live_frame,
    build_roll_adjusted_nq_main,
    build_signal_sessions,
    build_us_dataset,
    build_us_live_frame,
    fetch_yahoo_daily,
    make_price_features,
    pick_feature_columns,
)


@dataclass
class ForecastContext:
    market: str
    symbol: str
    sessions: pd.DataFrame
    dataset: pd.DataFrame
    live_frame: pd.DataFrame
    price_col: str


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Forecast market states for a window using only signals available up to a cutoff date.")
    parser.add_argument("--events", default=str(root_dir / "data/raw/wsquant_market_timeline.xlsx"))
    parser.add_argument("--cutoff-date", default="2026-03-23")
    parser.add_argument("--start-date", default="2026-03-23")
    parser.add_argument("--end-date", default="2026-03-28")
    parser.add_argument("--cn-symbol", default="159941.SZ")
    parser.add_argument("--output", default=str(root_dir / "data/forecasts/signal_forecast_2026-03-23_to_2026-03-28.xlsx"))
    parser.add_argument("--csv-output", default=str(root_dir / "data/forecasts/signal_forecast_2026-03-23_to_2026-03-28.csv"))
    parser.add_argument("--nq-archive-dir", default=str(root_dir / "data/contracts/nq_contract_archive"))
    parser.add_argument("--strict-nq-archive", action="store_true")
    parser.add_argument("--range", default="2y")
    return parser.parse_args()


def train_probability(train_df: pd.DataFrame, feature_cols: list[str], row: pd.DataFrame, market: str) -> tuple[float, float, str]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(train_df[feature_cols], train_df["target"].astype(int))
    raw_probability = float(pipeline.predict_proba(row[feature_cols])[0, 1])
    probability, overlay_reason = apply_wave_fib_overlay(raw_probability, row, market)
    return raw_probability, probability, overlay_reason


def action_from_states(current_state: int, recommended_state: int) -> str:
    if current_state == 0 and recommended_state == 1:
        return "买入"
    if current_state == 1 and recommended_state == 0:
        return "卖出"
    if recommended_state == 1:
        return "持有"
    return "空仓"


def build_contexts(
    events_path: Path,
    cutoff_date: pd.Timestamp,
    cn_symbol: str,
    date_range: str,
    archive_dir: Path,
    strict_nq_archive: bool,
) -> tuple[list[ForecastContext], pd.DataFrame, str]:
    us_sessions, cn_sessions = build_signal_sessions(events_path)
    us_sessions = us_sessions[us_sessions["session_date"] <= cutoff_date].copy()
    cn_sessions = cn_sessions[cn_sessions["session_date"] <= cutoff_date].copy()

    qqq_daily = fetch_yahoo_daily("QQQ", date_range)
    cn_daily = fetch_yahoo_daily(cn_symbol, date_range)

    archive_nq_status = archive_nq_contracts(
        archive_dir,
        range(pd.Timestamp.today().year - 1, pd.Timestamp.today().year + 2),
        date_range=date_range,
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
        if strict_nq_archive:
            raise RuntimeError("NQ archive incomplete for this forecast window.")
        nq_daily, _ = build_roll_adjusted_nq_main(date_range)
        nq_source = "roll-adjusted NQ=F fallback"

    qqq_features = make_price_features(qqq_daily, "qqq")
    cn_features = make_price_features(cn_daily, "cnetf")
    nq_features = make_price_features(nq_daily, "nq")

    us_dataset = build_us_dataset(us_sessions, qqq_features, nq_features, cn_features)
    cn_dataset = build_cn_dataset(cn_sessions, cn_features, qqq_features, nq_features)
    us_live = build_us_live_frame(us_sessions, qqq_features, nq_features, cn_features)
    cn_live = build_cn_live_frame(cn_sessions, cn_features, qqq_features, nq_features)

    contexts = [
        ForecastContext(
            market="场内",
            symbol=cn_symbol,
            sessions=cn_sessions,
            dataset=cn_dataset,
            live_frame=cn_live,
            price_col="cnetf_close",
        ),
        ForecastContext(
            market="美股",
            symbol="QQQ",
            sessions=us_sessions,
            dataset=us_dataset,
            live_frame=us_live,
            price_col="qqq_close",
        ),
    ]
    return contexts, archive_nq_status, nq_source


def forecast_market_window(
    context: ForecastContext,
    cutoff_date: pd.Timestamp,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    dataset = context.dataset.copy()
    live_frame = context.live_frame.copy()
    feature_cols = pick_feature_columns(dataset)
    if not feature_cols:
        raise RuntimeError(f"no feature columns available for {context.market}")

    dataset["target_date"] = pd.to_datetime(dataset["target_date"]).dt.normalize()
    live_frame["date"] = pd.to_datetime(live_frame["date"]).dt.normalize()
    sessions = context.sessions.copy()
    sessions["session_date"] = pd.to_datetime(sessions["session_date"]).dt.normalize()

    price_dates = sorted(pd.Timestamp(d).normalize() for d in live_frame["date"].unique())
    trading_dates = {d for d in price_dates if start_date <= d <= end_date}
    known_state_by_date = dict(zip(sessions["session_date"], sessions["state"].astype(int), strict=True))
    historical_dates = sorted(known_state_by_date)

    rows: list[dict[str, object]] = []
    prev_model_state: int | None = None

    for target_date in pd.date_range(start_date, end_date, freq="D"):
        target_date = pd.Timestamp(target_date).normalize()
        if target_date in trading_dates:
            date_index = price_dates.index(target_date)
            if date_index == 0:
                continue
            feature_date = price_dates[date_index - 1]
            row = live_frame[live_frame["date"].eq(feature_date)].sort_values("date").tail(1)
            train_cutoff = min(cutoff_date, target_date - pd.Timedelta(days=1))
            train = dataset[dataset["target_date"] <= train_cutoff].dropna(subset=feature_cols + ["target"]).copy()
            if train.empty or train["target"].nunique() < 2:
                raise RuntimeError(f"insufficient training labels for {context.market} on {target_date.date()}")

            raw_probability, probability_long, overlay_reason = train_probability(train, feature_cols, row, context.market)
            recommended_state = int(probability_long >= 0.5)

            if prev_model_state is None:
                prior_actual_dates = [d for d in historical_dates if d < target_date]
                current_state = int(known_state_by_date[prior_actual_dates[-1]]) if prior_actual_dates else 0
                current_state_source = "teacher_history"
            else:
                current_state = int(prev_model_state)
                current_state_source = "prior_model_forecast"

            actual_state = known_state_by_date.get(target_date)
            rows.append(
                {
                    "market": context.market,
                    "symbol": context.symbol,
                    "target_date": target_date,
                    "is_trading_day": 1,
                    "feature_date": feature_date,
                    "feature_price": float(row[context.price_col].iloc[0]),
                    "current_state": current_state,
                    "current_state_source": current_state_source,
                    "raw_probability": raw_probability,
                    "probability_long": probability_long,
                    "recommended_state": recommended_state,
                    "action": action_from_states(current_state, recommended_state),
                    "wave_overlay": overlay_reason,
                    "actual_state": int(actual_state) if actual_state is not None else np.nan,
                    "actual_match": bool(int(actual_state) == recommended_state) if actual_state is not None else np.nan,
                }
            )
            prev_model_state = recommended_state
            continue

        current_state = prev_model_state
        if current_state is None:
            prior_actual_dates = [d for d in historical_dates if d < target_date]
            current_state = int(known_state_by_date[prior_actual_dates[-1]]) if prior_actual_dates else 0
        rows.append(
            {
                "market": context.market,
                "symbol": context.symbol,
                "target_date": target_date,
                "is_trading_day": 0,
                "feature_date": pd.NaT,
                "feature_price": np.nan,
                "current_state": int(current_state),
                "current_state_source": "carry_forward",
                "raw_probability": np.nan,
                "probability_long": np.nan,
                "recommended_state": int(current_state),
                "action": "休市",
                "wave_overlay": "market_closed",
                "actual_state": np.nan,
                "actual_match": np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    events_path = Path(args.events).resolve()
    output_path = Path(args.output).resolve()
    csv_output_path = Path(args.csv_output).resolve()
    archive_dir = Path(args.nq_archive_dir).resolve()
    cutoff_date = pd.Timestamp(args.cutoff_date).normalize()
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()

    contexts, archive_nq_status, nq_source = build_contexts(
        events_path=events_path,
        cutoff_date=cutoff_date,
        cn_symbol=args.cn_symbol,
        date_range=args.range,
        archive_dir=archive_dir,
        strict_nq_archive=args.strict_nq_archive,
    )

    forecast_frames = [forecast_market_window(ctx, cutoff_date, start_date, end_date) for ctx in contexts]
    forecast_df = pd.concat(forecast_frames, ignore_index=True).sort_values(["market", "target_date"]).reset_index(drop=True)
    forecast_df.to_csv(csv_output_path, index=False)

    summary = (
        forecast_df[forecast_df["is_trading_day"].eq(1)]
        .groupby("market", as_index=False)
        .agg(
            trading_days=("target_date", "count"),
            buy_count=("action", lambda s: int((s == "买入").sum())),
            sell_count=("action", lambda s: int((s == "卖出").sum())),
            hold_count=("action", lambda s: int((s == "持有").sum())),
            empty_count=("action", lambda s: int((s == "空仓").sum())),
            known_actual_days=("actual_state", lambda s: int(s.notna().sum())),
            actual_match_days=("actual_match", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        )
    )
    notes = pd.DataFrame(
        [
            ("signal_cutoff_date", str(cutoff_date.date())),
            ("forecast_start_date", str(start_date.date())),
            ("forecast_end_date", str(end_date.date())),
            ("nq_source", nq_source),
            ("prediction_rule", "Each target date uses only labels known up to the cutoff date; states after the cutoff are chained from prior model forecasts."),
            ("action_rule", "0->1=买入, 1->0=卖出, 1->1=持有, 0->0=空仓, 非交易日=休市."),
        ],
        columns=["item", "value"],
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="forecast", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        archive_nq_status.to_excel(writer, sheet_name="nq_archive_status", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)

    print(f"Wrote workbook: {output_path}")
    print(f"Wrote CSV: {csv_output_path}")
    print("\nForecast:")
    print(forecast_df.to_string(index=False))


if __name__ == "__main__":
    main()
