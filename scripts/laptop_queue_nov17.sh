#!/usr/bin/env bash
# Laptop queue, Nov 17 (CR: "do everything on the list first"), one simulator at a time on port 10826:
#   1. rule on the five protocol kitchens with the A44 mapper fix (HAZARD_MAP_MAX_OBSTACLE_H=1.8), clock off, to compare
#      with the shipped laptop rule sweep (results/sweep_fire_rule_test25: 0.111 0.091 0.000 0.261 0.455 on these five);
#   2. honest-text Jev on mm_kitchen_3a-1 with A44, clock on (against the two shipped-map honest episodes: 0.167 slow layer,
#      0.000 explore-only);
#   3. if results/compiled/round1/policy_round1.py exists: the round-1 program on the five validation scenes, clock on,
#      shipped map (scripts/compiled/validate.py).
# Waits for any running HAZARD episode to finish first. Logs in logs/queue_nov17*.log.
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX
LOG="$ROOT/logs/queue_nov17.log"
echo "== $(date '+%F %T') queue start" | tee -a "$LOG"
while tasklist | grep -qi "TDW.exe"; do sleep 20; done
sleep 5
cd "$ROOT/HAZARD" || exit 1

# 1. rule x 5 kitchens, A44 on, clock off
export HAZARD_MAP_MAX_OBSTACLE_H=1.8
for SCENE in mm_kitchen_3a-1 mm_kitchen_3a-10 mm_kitchen_3a-11 mm_kitchen_3a-12 mm_kitchen_3a-13; do
  OUT="$ROOT/results/sweep_fire_rule_A44h1.8_test5"
  if [ -f "$OUT/$SCENE/eval_result.json" ]; then echo "skip $SCENE" | tee -a "$LOG"; continue; fi
  echo "== $(date '+%T') rule A44 $SCENE" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 2
  timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent rule --episodes 1 --port 10826 \
      --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
      --resume_dir "G:\iw-thinking-fast\results\sweep_fire_rule_A44h1.8_test5" \
      --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\\$SCENE" --clock none \
      > "$ROOT/logs/queue_nov17_rule_A44_$SCENE.log" 2>&1
  VR=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$OUT/$SCENE/eval_result.json" 2>/dev/null || echo none)
  echo "   exit=$? value_rate=$VR" | tee -a "$LOG"
done

# 2. honest Jev on 3a-1, A44 on, clock on
echo "== $(date '+%T') honest Jev A44 3a-1" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 2
timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent jev --honest_text \
    --api_key_file "G:\iw-thinking-fast\.secrets\typesafe_key.txt" --episodes 1 --port 10826 \
    --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
    --resume_dir "G:\iw-thinking-fast\results\sweep_fire_jev_honest_A44h1.8_test5_tcm30" \
    --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\mm_kitchen_3a-1" --clock tcm --fps 30 \
    > "$ROOT/logs/queue_nov17_jev_honest_A44_3a-1.log" 2>&1
VR=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$ROOT/results/sweep_fire_jev_honest_A44h1.8_test5_tcm30/mm_kitchen_3a-1/eval_result.json" 2>/dev/null || echo none)
echo "   value_rate=$VR" | tee -a "$LOG"

# 3. round-1 program on the validation scenes, shipped map, clock on
unset HAZARD_MAP_MAX_OBSTACLE_H
if [ -f "$ROOT/results/compiled/round1/policy_round1.py" ]; then
  echo "== $(date '+%T') validate round 1" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 2
  PYTHONPATH="$PYTHONPATH;G:/iw-thinking-fast/scripts" "$PY" -X utf8 "$ROOT/scripts/compiled/validate.py" \
      --program "G:\iw-thinking-fast\results\compiled\round1\policy_round1.py" --out "G:\iw-thinking-fast\results\compiled\round1\validation" \
      --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --clock tcm > "$ROOT/logs/queue_nov17_validate_round1.log" 2>&1
  echo "   exit=$?" | tee -a "$LOG"
else
  echo "no round-1 program yet; validation skipped" | tee -a "$LOG"
fi
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') queue done" | tee -a "$LOG"
