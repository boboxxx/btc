#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Starting aligned 5-minute dashboard loop in $DIR"

while true; do
  now_epoch=$(date +%s)
  next_epoch=$(( ((now_epoch / 300) + 1) * 300 ))
  sleep_seconds=$(( next_epoch - now_epoch ))

  if next_run="$(date -d "@$next_epoch" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)"; then
    :
  else
    next_run="$(date -r "$next_epoch" '+%Y-%m-%d %H:%M:%S')"
  fi

  echo "[$(timestamp)] Waiting ${sleep_seconds}s until ${next_run}"
  sleep "${sleep_seconds}"

  echo "[$(timestamp)] Running update_dashboard.sh"
  if ! bash "$DIR/update_dashboard.sh"; then
    echo "[$(timestamp)] update_dashboard.sh failed, continuing to next 5-minute slot"
  fi
done
