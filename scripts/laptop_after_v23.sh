#!/usr/bin/env bash
# After scripts/laptop_after_jev_la.sh prints "done" (logs/after_jev_la.log): rule with BOTH body fixes (A44 map 1.8 m + A50 v2
# lookahead 1.0 m), clock off: 3a-1 with the walk log (results/a44_a50_rule_3a-1), then the five protocol kitchens
# (results/sweep_fire_rule_A44_A50_test5), against shipped 0.184 and A44-only 0.281.
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/after_v23.log"
until grep -q "done" "$ROOT/logs/after_jev_la.log" 2>/dev/null; do sleep 30; done
sleep 10
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX
export HAZARD_MAP_MAX_OBSTACLE_H=1.8 HAZARD_WALK_LOOKAHEAD=1.0
run() {  # scene, out dir, walk log flag
  echo "== $(date '+%F %T') rule A44+A50v2 $1 -> $2" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
  HAZARD_WALK_LOG=$3 timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent rule --episodes 1 --port 10826 \
      --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
      --resume_dir "G:/iw-thinking-fast/results/$2" \
      --data_dir "G:/iw-thinking-fast/HAZARD/src/HAZARD/data/room_setup_fire/test_set/$1" --clock none \
      > "$ROOT/logs/after_v23_$2_$1.log" 2>&1
  VR=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$ROOT/results/$2/$1/eval_result.json" 2>/dev/null || echo none)
  echo "   exit=$? value_rate=$VR" | tee -a "$LOG"
}
run mm_kitchen_3a-1 a44_a50_rule_3a-1 1
for SCENE in mm_kitchen_3a-1 mm_kitchen_3a-10 mm_kitchen_3a-11 mm_kitchen_3a-12 mm_kitchen_3a-13; do
  run "$SCENE" sweep_fire_rule_A44_A50_test5 0
done
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') done" | tee -a "$LOG"
