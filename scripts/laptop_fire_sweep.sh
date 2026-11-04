#!/usr/bin/env bash
# A19 verification on the failing case, then the 25-scene fire sweep (CR, Oct 16 18:3x: "apply A19, run the sweep").
# Resumable: rerun with the same results folders and HAZARD skips scenes that already have eval_result.json.
set -uo pipefail
ROOT=/g/iw-thinking-fast
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/fire_sweep.log"
run() {  # run <tag> <args...>
  local tag="$1"; shift
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 10; done
  echo "===== $(date +%H:%M:%S) start $tag =====" | tee -a "$LOG"
  ( cd "$ROOT/HAZARD" && "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 "$@" > "$ROOT/logs/baseline_${tag}.log" 2>&1 )
  local rc=$?
  echo "===== $(date +%H:%M:%S) end $tag exit=$rc =====" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 5
  return $rc
}
# (A19 verified Oct 16; verification step removed for the resumed sweep)
for A in rule greedy mcts random; do
  DIR="$ROOT/results/sweep_fire_${A}_ep25"
  mkdir -p "$DIR"
  # up to 3 attempts so a crash mid-sweep resumes at the next unfinished scene
  for attempt in 1 2 3; do
    run sweep_fire_${A}_ep25_attempt${attempt} --env fire --agent $A --episodes 25 --resume_dir "$DIR" && break
  done
done
echo "FIRE SWEEP DONE $(date +%H:%M:%S)" | tee -a "$LOG"
