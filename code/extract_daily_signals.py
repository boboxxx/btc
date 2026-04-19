#!/usr/bin/env python3
"""Build corrected market timelines and a buy/sell chart from raw JSON pages."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


FILE_RE = re.compile(r"currentPage%3d(\d+)$")

US_STRONG_HINTS = ("纽约", "美东时间", "纳斯达克交易所", "纽交所")
US_WEAK_HINTS = ("QQQ", "SPY", "纳斯达克100指数ETF", "标普500指数ETF")
ONSHORE_STRONG_HINTS = ("上海、深圳", "上交所、深交所", "（场内）", "(场内)", "上海时间")

US_STATE_PATTERNS = (
    (1, "100%", re.compile(r"100\s*%")),
    (0, "0%", re.compile(r"0\s*%")),
    (1, "满仓", re.compile(r"满仓")),
    (0, "空仓", re.compile(r"空仓")),
)

ONSHORE_PATTERNS = (
    (1, "买入", re.compile(r"(短线买入美股etf|短线美股etf直接买入|美股etf直接买入|直接买入美股etf)", re.I)),
    (0, "卖出", re.compile(r"(短线卖出美股etf|短线卖掉美股etf|卖掉美股etf|卖出美股etf)", re.I)),
    (1, "继续持有", re.compile(r"美股etf(继续持有|短线持有|暂时持有)", re.I)),
    (
        1,
        "满仓",
        re.compile(
            r"(A股场内的美股ETF已[^，。]{0,40}?满仓|美股ETF已[^，。]{0,40}?满仓|满仓美股ETF等通知|满仓A股场内的美股ETF等通知)",
            re.I,
        ),
    ),
    (
        0,
        "空仓",
        re.compile(
            r"(A股场内的美股ETF已[^，。]{0,40}?空仓|美股ETF已[^，。]{0,40}?空仓|空仓美股ETF等通知|空仓A股场内的美股ETF等通知|美股ETF短线空仓)",
            re.I,
        ),
    ),
    (1, "100%", re.compile(r"(上海、深圳|上交所、深交所|场内|上海时间).{0,100}?100\s*%", re.I)),
    (0, "0%", re.compile(r"(上海、深圳|上交所、深交所|场内|上海时间).{0,100}?0\s*%", re.I)),
    (1, "满仓", re.compile(r"(上海、深圳|上交所、深交所|场内|上海时间).{0,100}?满仓", re.I)),
    (0, "空仓", re.compile(r"(上海、深圳|上交所、深交所|场内|上海时间).{0,100}?空仓", re.I)),
)

UNSPLIT_PATTERNS = (
    re.compile(r"满仓等通知"),
    re.compile(r"空仓等通知"),
)


@dataclass
class ExtractedEvent:
    market: str
    signal: str
    state: int
    event_text: str


def parse_page_no(path: Path) -> int:
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse page number from {path.name}")
    return int(match.group(1))


def load_records(input_dir: Path) -> list[dict[str, object]]:
    files = sorted(input_dir.glob("getReview.php%3fcurrentPage%3d*"), key=parse_page_no)
    records: list[dict[str, object]] = []
    for page_no, path in ((parse_page_no(path), path) for path in files):
        raw = path.read_bytes().decode("utf-8-sig")
        obj = json.loads(raw)
        for row_in_page, item in enumerate(obj.get("data", {}).get("list", []), start=1):
            records.append(
                {
                    "page_no": page_no,
                    "row_in_page": row_in_page,
                    "md": str(item.get("createdAtYMD", "")),
                    "time": str(item.get("createdAt", "")),
                    "content": str(item.get("content", "")),
                }
            )
    return records


def assign_years(records: list[dict[str, object]], latest_year: int) -> None:
    year = latest_year
    prev_md: Optional[tuple[int, int]] = None

    for record in records:
        month, day = map(int, str(record["md"]).split("."))
        current_md = (month, day)
        if prev_md is not None and current_md > prev_md:
            year -= 1
        record["year"] = year
        record["date"] = f"{year:04d}-{month:02d}-{day:02d}"
        record["datetime"] = pd.to_datetime(f"{record['date']} {record['time']}")
        prev_md = current_md


def find_marker_text(content: str, markers: tuple[str, ...]) -> str:
    for marker in markers:
        idx = content.find(marker)
        if idx >= 0:
            return content[idx:]
    return content


def extract_us_event(content: str) -> Optional[ExtractedEvent]:
    has_onshore_context = any(hint in content for hint in ONSHORE_STRONG_HINTS) or "A股场内的美股ETF" in content
    has_us_context = any(hint in content for hint in US_STRONG_HINTS) or (
        any(hint in content for hint in US_WEAK_HINTS) and not has_onshore_context
    )
    if not has_us_context:
        return None

    for state, signal, pattern in US_STATE_PATTERNS:
        if pattern.search(content):
            return ExtractedEvent(
                market="美股",
                signal=signal,
                state=state,
                event_text=find_marker_text(content, US_STRONG_HINTS + US_WEAK_HINTS + ("纳斯达克指数", "标普500指数")),
            )
    return None


def extract_onshore_event(content: str) -> Optional[ExtractedEvent]:
    has_onshore_context = (
        any(hint in content for hint in ONSHORE_STRONG_HINTS)
        or "A股场内的美股ETF" in content
        or "美股ETF" in content
    )
    if not has_onshore_context:
        return None

    for state, signal, pattern in ONSHORE_PATTERNS:
        if pattern.search(content):
            return ExtractedEvent(
                market="场内",
                signal=signal,
                state=state,
                event_text=find_marker_text(
                    content,
                    ONSHORE_STRONG_HINTS + ("A股场内的美股ETF", "美股ETF", "纳斯达克指数", "标普500指数"),
                ),
            )
    return None


def is_unsplit_signal(content: str) -> bool:
    if extract_us_event(content) or extract_onshore_event(content):
        return False
    return any(pattern.search(content) for pattern in UNSPLIT_PATTERNS)


def build_events(records: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    events: list[dict[str, object]] = []
    unsplit: list[dict[str, object]] = []

    for seq_newest, record in enumerate(records):
        content = str(record["content"])
        for extracted in (extract_us_event(content), extract_onshore_event(content)):
            if extracted is None:
                continue
            events.append(
                {
                    "seq_newest": seq_newest,
                    "page_no": record["page_no"],
                    "row_in_page": record["row_in_page"],
                    "year": record["year"],
                    "date": record["date"],
                    "time": record["time"],
                    "datetime": record["datetime"],
                    "market": extracted.market,
                    "signal": extracted.signal,
                    "state": extracted.state,
                    "content": content,
                    "event_text": extracted.event_text,
                }
            )

        if is_unsplit_signal(content):
            unsplit.append(
                {
                    "page_no": record["page_no"],
                    "row_in_page": record["row_in_page"],
                    "year": record["year"],
                    "date": record["date"],
                    "time": record["time"],
                    "datetime": record["datetime"],
                    "content": content,
                }
            )

    events_df = pd.DataFrame(events)
    unsplit_df = pd.DataFrame(unsplit)

    if not events_df.empty:
        events_df = events_df.sort_values(["datetime", "seq_newest"], ascending=[True, False], kind="mergesort").reset_index(
            drop=True
        )
    if not unsplit_df.empty:
        unsplit_df = unsplit_df.sort_values(
            ["datetime", "page_no", "row_in_page"], ascending=[True, True, True], kind="mergesort"
        ).reset_index(drop=True)
    return events_df, unsplit_df


def build_transitions(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for market, group in events.groupby("market", sort=False):
        prev_state: Optional[int] = None
        for _, row in group.iterrows():
            event_type = "初始化" if prev_state is None else ("买入" if row["state"] == 1 else "卖出")
            if prev_state is None or row["state"] != prev_state:
                rows.append(
                    {
                        "market": market,
                        "year": row["year"],
                        "date": row["date"],
                        "time": row["time"],
                        "datetime": row["datetime"],
                        "event_type": event_type,
                        "signal": row["signal"],
                        "state": row["state"],
                        "event_text": row["event_text"],
                        "content": row["content"],
                    }
                )
                prev_state = int(row["state"])
    transitions = pd.DataFrame(rows)
    if transitions.empty:
        return transitions
    return transitions.sort_values(["datetime", "market"], ascending=[True, True], kind="mergesort").reset_index(drop=True)


def build_daily_status(events: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (date_value, market), group in events.groupby(["date", "market"], sort=False):
        ordered = group.sort_values(["datetime", "seq_newest"], ascending=[True, False], kind="mergesort")
        changes = transitions[
            (transitions["date"] == date_value) & (transitions["market"] == market) & (transitions["event_type"] != "初始化")
        ]
        rows.append(
            {
                "date": date_value,
                "market": market,
                "first_time": ordered.iloc[0]["time"],
                "first_signal": ordered.iloc[0]["signal"],
                "last_time": ordered.iloc[-1]["time"],
                "last_signal": ordered.iloc[-1]["signal"],
                "change_count": len(changes),
                "changes": " | ".join(f"{row.time} {row.event_type}" for row in changes.itertuples()),
                "last_event_text": ordered.iloc[-1]["event_text"],
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    return daily.sort_values(["date", "market"], ascending=[True, True], kind="mergesort").reset_index(drop=True)


def build_coverage(events: pd.DataFrame, unsplit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not events.empty:
        for market, group in events.groupby("market", sort=False):
            rows.append(
                {
                    "series": market,
                    "event_count": len(group),
                    "first_date": group["date"].min(),
                    "last_date": group["date"].max(),
                    "notes": "Explicit market-labeled records only.",
                }
            )
    if not unsplit.empty:
        rows.append(
            {
                "series": "未区分",
                "event_count": len(unsplit),
                "first_date": unsplit["date"].min(),
                "last_date": unsplit["date"].max(),
                "notes": "Generic signals like 满仓等通知 / 空仓等通知; not split into 美股 or 场内.",
            }
        )
    return pd.DataFrame(rows)


def save_plot(transitions: pd.DataFrame, output_path: Path) -> None:
    if transitions.empty:
        return

    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    market_config = [("美股", "US Market"), ("场内", "Onshore")]

    for ax, (market, title) in zip(axes, market_config):
        group = transitions[transitions["market"] == market].sort_values("datetime")
        if group.empty:
            ax.set_title(f"{title}: no explicit records")
            ax.set_yticks([0, 1], labels=["Flat", "Long"])
            ax.grid(True, axis="x", alpha=0.25)
            continue

        ax.step(group["datetime"], group["state"], where="post", linewidth=2, color="#3b82f6", label="state")
        buys = group[group["event_type"] == "买入"]
        sells = group[group["event_type"] == "卖出"]
        ax.scatter(buys["datetime"], buys["state"], marker="^", s=90, color="#16a34a", label="buy")
        ax.scatter(sells["datetime"], sells["state"], marker="v", s=90, color="#dc2626", label="sell")
        ax.set_title(title)
        ax.set_yticks([0, 1], labels=["Flat", "Long"])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    figure.suptitle("Buy/Sell Transition Timeline", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def autosize_sheet(writer: pd.ExcelWriter, sheet_name: str, widths: dict[str, float]) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract corrected market timelines and buy/sell points.")
    root_dir = Path(__file__).resolve().parent.parent
    parser.add_argument("--input-dir", type=Path, default=root_dir / "data/raw/reviews/pages", help="Directory containing getReview.php%%3fcurrentPage%%3d* files.")
    parser.add_argument("--latest-year", type=int, default=date.today().year, help="Year of the newest records.")
    parser.add_argument("--output", type=Path, default=root_dir / "data/raw/wsquant_market_timeline.xlsx", help="Output Excel workbook.")
    parser.add_argument(
        "--plot",
        type=Path,
        default=root_dir / "data/raw/wsquant_buy_sell_timeline.png",
        help="Output PNG chart for buy/sell points.",
    )
    args = parser.parse_args()

    records = load_records(args.input_dir)
    if not records:
        raise SystemExit(f"No JSON page files found in {args.input_dir}")

    assign_years(records, args.latest_year)
    events, unsplit = build_events(records)
    transitions = build_transitions(events)
    daily = build_daily_status(events, transitions)
    coverage = build_coverage(events, unsplit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        coverage.to_excel(writer, sheet_name="coverage", index=False)
        events.to_excel(writer, sheet_name="events", index=False)
        transitions.to_excel(writer, sheet_name="transitions", index=False)
        daily.to_excel(writer, sheet_name="daily_status", index=False)
        if not unsplit.empty:
            unsplit.to_excel(writer, sheet_name="unsplit", index=False)

        autosize_sheet(writer, "coverage", {"A": 12, "B": 12, "C": 14, "D": 14, "E": 55})
        autosize_sheet(
            writer,
            "events",
            {"A": 10, "B": 10, "C": 12, "D": 12, "E": 20, "F": 10, "G": 10, "H": 10, "I": 8, "J": 90, "K": 90},
        )
        autosize_sheet(
            writer,
            "transitions",
            {"A": 10, "B": 10, "C": 12, "D": 12, "E": 20, "F": 10, "G": 10, "H": 8, "I": 90, "J": 90},
        )
        autosize_sheet(
            writer,
            "daily_status",
            {"A": 12, "B": 10, "C": 10, "D": 10, "E": 10, "F": 10, "G": 12, "H": 55, "I": 90},
        )
        if not unsplit.empty:
            autosize_sheet(writer, "unsplit", {"A": 10, "B": 10, "C": 10, "D": 12, "E": 12, "F": 20, "G": 90})

    save_plot(transitions, args.plot)

    print(f"Wrote {len(events):,} explicit market events to {args.output}")
    print(f"Wrote {len(transitions):,} buy/sell transition points to {args.plot}")
    if not coverage.empty:
        print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
