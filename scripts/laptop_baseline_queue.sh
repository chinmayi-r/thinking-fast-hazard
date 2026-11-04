#!/usr/bin/env bash
# Run the shipped agents one after another on the laptop, one fire episode each, after the current run ends.
# Usage: bash scripts/laptop_baseline_queue.sh "greedy mcts random" fire 1
set -uo pipefail
AGENTS="${1:-greedy mcts random}"; ENV_NAME="${2:-fire}"; EPISODES="${3:-1}"
ROOT=/g/iw-thinking-fast
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
# wait for the rule run (or any earlier runner) to release the build
while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 20; done
for A in $AGENTS; do
  echo "===== $(date +%H:%M:%S) start $ENV_NAME $A x$EPISODES ====="
  cd "$ROOT/HAZARD"
  "$ROOT/venv/Scripts/python.exe" -X utf8 "$ROOT/scripts/run_baseline.py" --env "$ENV_NAME" --agent "$A" --episodes "$EPISODES" --port 10826 ${EXTRA_ARGS:-} > "$ROOT/logs/baseline_${ENV_NAME}_${A}${LOG_TAG:-}_ep${EPISODES}.log" 2>&1
  echo "===== $(date +%H:%M:%S) end $A exit=$? ====="
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 5
done
echo "QUEUE DONE"
