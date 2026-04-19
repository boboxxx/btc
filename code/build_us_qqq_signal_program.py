#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from build_5m_signal_program import main as core_main

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_ARGS = [
    "--market",
    "us",
    "--output",
    str(ROOT_DIR / "data/signals/us/us_qqq_signal_results.xlsx"),
    "--live-output",
    str(ROOT_DIR / "data/signals/us/us_qqq_live_signals.csv"),
    "--state-output",
    str(ROOT_DIR / "data/signals/us/us_qqq_signal_state.csv"),
    "--history-output",
    str(ROOT_DIR / "data/signals/us/us_qqq_signal_history.csv"),
    "--range",
    "60d",
    "--intraday-bars",
    "5000",
]


if __name__ == "__main__":
    core_main(DEFAULT_ARGS + sys.argv[1:])
