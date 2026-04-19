#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[$(timestamp)] Python not found in PATH"
  exit 1
fi

echo "[$(timestamp)] Using Python: ${PYTHON_BIN}"
tracked_changes="$(git diff --name-only)"
staged_changes="$(git diff --cached --name-only)"
if [ -n "${tracked_changes}" ] || [ -n "${staged_changes}" ]; then
  echo "[$(timestamp)] Local tracked changes detected; syncing with --autostash"
fi

echo "[$(timestamp)] Syncing latest origin/main"
git pull --rebase --autostash origin main

if [ -f "data/raw/wsquant_market_timeline.xlsx" ]; then
  echo "[$(timestamp)] Running code/build_open_nowcast.py"
  "$PYTHON_BIN" code/build_open_nowcast.py --market all
else
  echo "[$(timestamp)] Skipping code/build_open_nowcast.py (data/raw/wsquant_market_timeline.xlsx not found)"
fi

if [ -f "code/build_us_qqq_signal_program.py" ]; then
  echo "[$(timestamp)] Running code/build_us_qqq_signal_program.py"
  "$PYTHON_BIN" code/build_us_qqq_signal_program.py --print-mode all
else
  echo "[$(timestamp)] Running code/build_5m_signal_program.py --market us"
  "$PYTHON_BIN" code/build_5m_signal_program.py --market us --range 60d --intraday-bars 5000 --print-mode all \
    --output data/signals/us/us_qqq_signal_results.xlsx \
    --live-output data/signals/us/us_qqq_live_signals.csv \
    --state-output data/signals/us/us_qqq_signal_state.csv \
    --history-output data/signals/us/us_qqq_signal_history.csv
fi

if [ -f "code/build_cn_nasdaq_signal_program.py" ]; then
  echo "[$(timestamp)] Running code/build_cn_nasdaq_signal_program.py"
  "$PYTHON_BIN" code/build_cn_nasdaq_signal_program.py --print-mode all
else
  echo "[$(timestamp)] Running code/build_5m_signal_program.py --market cn"
  "$PYTHON_BIN" code/build_5m_signal_program.py --market cn --range 20d --intraday-bars 5000 --print-mode all \
    --output data/signals/cn/cn_nasdaq_signal_results.xlsx \
    --live-output data/signals/cn/cn_nasdaq_live_signals.csv \
    --state-output data/signals/cn/cn_nasdaq_signal_state.csv \
    --history-output data/signals/cn/cn_nasdaq_signal_history.csv
fi

if [ -f "code/build_btc_signal_program.py" ]; then
  echo "[$(timestamp)] Running code/build_btc_signal_program.py"
  "$PYTHON_BIN" code/build_btc_signal_program.py --range 60d --intraday-bars 5000 --print-mode all
else
  echo "[$(timestamp)] Skipping code/build_btc_signal_program.py (file not found)"
fi

echo "[$(timestamp)] Exporting docs/data/dashboard.json"
"$PYTHON_BIN" code/export_web_data.py --output docs/data/dashboard.json

echo "[$(timestamp)] Staging dashboard snapshot"
git add docs/data/dashboard.json

if git diff --cached --quiet; then
  echo "[$(timestamp)] No dashboard.json changes to commit"
else
  commit_message="Update dashboard data ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "[$(timestamp)] Committing: ${commit_message}"
  git commit -m "${commit_message}"
fi

echo "[$(timestamp)] Pushing to origin/main"
git push origin main

echo "[$(timestamp)] Dashboard update complete"
