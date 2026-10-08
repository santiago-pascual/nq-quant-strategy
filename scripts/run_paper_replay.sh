#!/bin/sh
set -eu
: "${PAPER_REPLAY_START:?Set PAPER_REPLAY_START in the protected environment file}"
: "${PAPER_REPLAY_END:?Set PAPER_REPLAY_END in the protected environment file}"
: "${PAPER_OUTPUT_DIR:=/var/lib/mnq-paper}"
: "${PAPER_COST_CONFIG:=/opt/mnq-paper/app/src/paper/config/topstepx_mnq_fees_2026-07.json}"

mkdir -p "$PAPER_OUTPUT_DIR"
PYTHON=/opt/mnq-paper/app/.venv/bin/python
set -- -m src.paper.run_realtime_paper --command run --mode PAPER \
  --replay-start "$PAPER_REPLAY_START" --replay-end "$PAPER_REPLAY_END" \
  --output-dir "$PAPER_OUTPUT_DIR" --cost-config "$PAPER_COST_CONFIG"
if [ -e "$PAPER_OUTPUT_DIR/paper_checkpoint.json" ]; then
  exec "$PYTHON" "$@" --resume
fi
if [ -e "$PAPER_OUTPUT_DIR/events.jsonl" ] || [ -e "$PAPER_OUTPUT_DIR/paper_analytics.sqlite3" ]; then
  echo "Existing Paper state has no resumable checkpoint; refusing to start" >&2
  exit 78
fi
exec "$PYTHON" "$@"
