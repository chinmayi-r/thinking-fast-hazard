#!/usr/bin/env bash
# Run the remaining baselines on the laptop, one simulator at a time (CR, Oct 16 16:20: "run all baselines").
# Order: wait for the running Claude episode; Claude under the clock if the first cost < $1.50; flood and wind for
# rule/greedy/mcts/random unclocked; then greedy and mcts under the clock on flood and wind.
set -uo pipefail
ROOT=/g/iw-thinking-fast
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
export ANTHROPIC_WORKSPACE_ID='wrkspc_01VPU7qLAqtHXFpUPSQ8DCgd'
PY="$ROOT/venv/Scripts/python.exe"
KEY="G:\\iw-thinking-fast\\.secrets\\anthropic_key.txt"
LOG="$ROOT/logs/all_baselines.log"
run() {  # run <tag> <args...>
  local tag="$1"; shift
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 10; done
  echo "===== $(date +%H:%M:%S) start $tag =====" | tee -a "$LOG"
  ( cd "$ROOT/HAZARD" && "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 "$@" > "$ROOT/logs/baseline_${tag}.log" 2>&1 )
  echo "===== $(date +%H:%M:%S) end $tag exit=$? =====" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 5
}
# 1. wait for the running Claude episode (attempt 3)
until grep -q "^exit=" "$ROOT/logs/baseline_fire_llm_opus_ep1_attempt3.log" 2>/dev/null; do sleep 20; done
# 2. Claude under the clock, only if the first episode's cost estimate was < $1.50 and it succeeded
COST=$("$PY" - <<'PYEOF'
import json, glob, os
ds = sorted(glob.glob(r"G:\iw-thinking-fast\results\fire_llm_ep1_*"))
try:
    sm = json.load(open(os.path.join(ds[-1], "summary.json")))
    print(sm.get("llm", {}).get("cost_usd_estimate", 99) if sm.get("exit") == 0 else 99)
except Exception:
    print(99)
PYEOF
)
echo "first Claude episode cost estimate: $COST USD" | tee -a "$LOG"
if "$PY" -c "import sys; sys.exit(0 if float('$COST') < 1.5 else 1)"; then
  run fire_llm_opus_tcm30_ep1 --env fire --agent llm --lm_source anthropic --lm_id claude-opus-5-5 --api_key_file "$KEY" --episodes 1 --clock tcm --fps 30
else
  echo "skipping clocked Claude run (cost $COST)" | tee -a "$LOG"
fi
# 3. flood and wind, shipped agents, unclocked (step 1 for the other two hazards)
for ENV_NAME in flood wind; do
  for A in rule greedy mcts random; do
    run ${ENV_NAME}_${A}_ep1 --env $ENV_NAME --agent $A --episodes 1
  done
done
# 4. greedy and mcts under the clock on flood and wind
for ENV_NAME in flood wind; do
  for A in greedy mcts; do
    run ${ENV_NAME}_${A}_tcm30_ep1 --env $ENV_NAME --agent $A --episodes 1 --clock tcm --fps 30
  done
done
echo "ALL BASELINES DONE $(date +%H:%M:%S)" | tee -a "$LOG"
