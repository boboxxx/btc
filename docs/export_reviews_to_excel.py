#!/usr/bin/env python3
"""Export all `getReview.php%3fcurrentPage%3d*` JSON files to Excel.

The script scans the current directory by default, parses every page file,
and writes:

* `records`  - one row per review item
* `pages`    - one row per source file/page with basic metadata
* `errors`   - optional parse errors, if any files could not be read

Usage:
    python3 export_reviews_to_excel.py
    python3 export_reviews_to_excel.py --input-dir /path/to/ajax --output reviews.xlsx
"""

from __future__ import annotations

import argparse
import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd


FILE_RE = re.compile(r"currentPage%3d(\d+)$")
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def parse_page_no(path: Path) -> int:
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse page number from file name: {path.name}")
    return int(match.group(1))


def clean_text(value: Any) -> str:
    """Remove HTML tags and normalize whitespace for a readable text copy."""
    if value is None:
        return ""
    text = str(value)
    text = TAG_RE.sub("", text)
    text = unescape(text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def load_pages(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(
        input_dir.glob("getReview.php%3fcurrentPage%3d*"),
        key=parse_page_no,
    )

    records: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for path in files:
        page_no = parse_page_no(path)
        try:
            raw = path.read_bytes().decode("utf-8-sig")
            obj = json.loads(raw)
            data = obj.get("data", {})
            page_meta = data.get("page", {}) or {}
            items = data.get("list", []) or []
        except Exception as exc:  # noqa: BLE001 - collect and export parse errors
            errors.append(
                {
                    "page_no": page_no,
                    "source_file": path.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if items:
            first = items[0]
            last = items[-1]
            pages.append(
                {
                    "page_no": page_no,
                    "source_file": path.name,
                    "item_count": len(items),
                    "first_createdAtYMD": first.get("createdAtYMD", ""),
                    "first_createdAt": first.get("createdAt", ""),
                    "last_createdAtYMD": last.get("createdAtYMD", ""),
                    "last_createdAt": last.get("createdAt", ""),
                    "api_currentPage": page_meta.get("currentPage"),
                    "api_pageSize": page_meta.get("pageSize"),
                    "api_totalNum": page_meta.get("totalNum"),
                    "api_totalPage": page_meta.get("totalPage"),
                    "api_totalPageNum": page_meta.get("totalPageNum"),
                    "api_hasNextPage": page_meta.get("hasNextPage"),
                }
            )
        else:
            pages.append(
                {
                    "page_no": page_no,
                    "source_file": path.name,
                    "item_count": 0,
                    "first_createdAtYMD": "",
                    "first_createdAt": "",
                    "last_createdAtYMD": "",
                    "last_createdAt": "",
                    "api_currentPage": page_meta.get("currentPage"),
                    "api_pageSize": page_meta.get("pageSize"),
                    "api_totalNum": page_meta.get("totalNum"),
                    "api_totalPage": page_meta.get("totalPage"),
                    "api_totalPageNum": page_meta.get("totalPageNum"),
                    "api_hasNextPage": page_meta.get("hasNextPage"),
                }
            )

        for idx, item in enumerate(items, start=1):
            raw_content = item.get("content", "")
            records.append(
                {
                    "page_no": page_no,
                    "row_in_page": idx,
                    "source_file": path.name,
                    "api_currentPage": page_meta.get("currentPage"),
                    "api_pageSize": page_meta.get("pageSize"),
                    "api_totalNum": page_meta.get("totalNum"),
                    "api_totalPage": page_meta.get("totalPage"),
                    "api_totalPageNum": page_meta.get("totalPageNum"),
                    "api_hasNextPage": page_meta.get("hasNextPage"),
                    "createdAtYMD": item.get("createdAtYMD", ""),
                    "createdAt": item.get("createdAt", ""),
                    "id": item.get("id", ""),
                    "imgUrl": item.get("imgUrl", ""),
                    "content": raw_content,
                    "content_clean": clean_text(raw_content),
                }
            )

    return records, pages, errors


def autosize_columns(df: pd.DataFrame, worksheet) -> None:
    """Set practical Excel column widths without making long text columns huge."""
    from openpyxl.utils import get_column_letter

    long_text_columns = {"content", "content_clean", "imgUrl"}
    for idx, column in enumerate(df.columns, start=1):
        if column in long_text_columns:
            width = 80
        else:
            series = df[column].astype(str).fillna("")
            longest = max([len(column)] + [len(value) for value in series.tolist()] or [len(column)])
            width = min(max(longest + 2, 12), 36)
        worksheet.column_dimensions[get_column_letter(idx)].width = width


def main() -> int:
    parser = argparse.ArgumentParser(description="Export review JSON pages to Excel.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing getReview.php%%3fcurrentPage%%3d* files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wsquant_reviews.xlsx"),
        help="Output Excel file path.",
    )
    parser.add_argument(
        "--sort",
        choices=("newest", "oldest"),
        default="newest",
        help="Sort records by page order.",
    )
    args = parser.parse_args()

    records, pages, errors = load_pages(args.input_dir)
    if not records and not pages:
        raise SystemExit(f"No matching files found in {args.input_dir}")

    records_df = pd.DataFrame(records)
    pages_df = pd.DataFrame(pages)
    errors_df = pd.DataFrame(errors)

    if not records_df.empty:
        ascending = args.sort == "newest"
        records_df = records_df.sort_values(
            by=["page_no", "row_in_page"],
            ascending=[ascending, ascending],
            kind="mergesort",
        ).reset_index(drop=True)

    if not pages_df.empty:
        ascending = args.sort == "newest"
        pages_df = pages_df.sort_values(by=["page_no"], ascending=[ascending], kind="mergesort")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        records_df.to_excel(writer, sheet_name="records", index=False)
        pages_df.to_excel(writer, sheet_name="pages", index=False)
        if not errors_df.empty:
            errors_df.to_excel(writer, sheet_name="errors", index=False)

        workbook = writer.book
        for sheet_name, df in (
            ("records", records_df),
            ("pages", pages_df),
            ("errors", errors_df),
        ):
            if sheet_name not in writer.sheets:
                continue
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            autosize_columns(df, ws)

        # Make long text columns easier to inspect.
        if "records" in writer.sheets:
            ws = writer.sheets["records"]
            ws.sheet_view.zoomScale = 90

    print(f"Exported {len(records_df):,} records from {len(pages_df):,} pages to {args.output}")
    if not errors_df.empty:
        print(f"Skipped {len(errors_df)} file(s) with parse errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
