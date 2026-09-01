#!/bin/bash
# Daily forward paper-trading runner for the multi-agent futures council.
#
# Runs independently of any app or interactive session: installed as a launchd
# job (see ops/com.abisha.futures-council.plist) so it survives reboots, needs
# no permission prompts, and catches up if the Mac was asleep at the trigger.
#
# This exists because the first app-scheduled run FIRED but produced nothing --
# a fresh unattended session stalls on tool-permission prompts. A study whose
# whole validity rests on an unbroken daily record cannot depend on that: a
# silent no-op is worse than a visible failure, because the log just stops
# growing and nobody notices.
#
# Never fabricates or backfills a decision. A missed day stays missed --
# a gap is honest, an invented data point is not.

set -uo pipefail

PROJECT="/Users/abish/projects/stock-xai-project"
OLLAMA="/opt/homebrew/opt/ollama/bin/ollama"
LOGDIR="$PROJECT/ops/logs"
mkdir -p "$LOGDIR"
STAMP="$(date '+%Y-%m-%d')"
LOG="$LOGDIR/run_$STAMP.log"

exec >>"$LOG" 2>&1
echo "===== run started $(date '+%Y-%m-%d %H:%M:%S %Z') ====="

# Single-instance lock. Without it, two runs can overlap (e.g. a manual run plus
# a launchd trigger) -- and the "already logged today" check below does NOT catch
# that, because neither has written its row yet. Observed live on 2026-09-01: two
# concurrent harnesses, which would have written duplicate rows for the same day
# and silently inflated the sample. flock is not on macOS by default, so this uses
# an atomic mkdir, which succeeds for exactly one caller.
LOCK="/tmp/futures-council.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run holds the lock ($LOCK) -- exiting to avoid duplicate rows"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# Skip weekends: futures don't settle, so a Saturday row would be a duplicate
# of Friday's close masquerading as a new observation.
DOW="$(date +%u)"   # 1=Mon .. 7=Sun
if [ "$DOW" -gt 5 ]; then
  echo "weekend (dow=$DOW) -- skipping, markets closed"
  exit 0
fi

# Don't double-log if a run already recorded today (e.g. manual run earlier).
LOGFILE="$PROJECT/data/paper_trades/agent_decisions.jsonl"
if [ -f "$LOGFILE" ] && grep -q "\"trade_date\": \"$STAMP\"" "$LOGFILE"; then
  echo "a decision for $STAMP is already logged -- skipping to avoid duplicate observations"
  exit 0
fi

# Ensure the local model server is up.
if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  echo "ollama not responding; starting it"
  OLLAMA_FLASH_ATTENTION="1" "$OLLAMA" serve >>"$LOGDIR/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    sleep 2
    curl -sf http://localhost:11434/api/version >/dev/null && break
  done
fi

if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  echo "FAILED: ollama unreachable after 60s -- no decision logged (correct: never fabricate)"
  exit 1
fi

cd "$PROJECT/backend" || exit 1
"$PROJECT/.venv/bin/python" scripts/paper_trade_agent.py --quiet
STATUS=$?
echo "harness exit status: $STATUS"

# Commit the decision so it is timestamped in git BEFORE the outcome is known.
# That is what makes the forward record tamper-evident.
if [ $STATUS -eq 0 ]; then
  cd "$PROJECT" || exit 1
  git add data/paper_trades/agent_decisions.jsonl
  if ! git diff --cached --quiet; then
    git commit -q -m "Paper-trade log: $STAMP" && git push -q origin main && echo "committed and pushed"
  else
    echo "no new rows to commit"
  fi
fi

echo "===== run finished $(date '+%H:%M:%S') ====="
