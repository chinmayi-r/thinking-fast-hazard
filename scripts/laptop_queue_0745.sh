#!/usr/bin/env bash
# Restart of the chain the 07:40 app restart killed: (1) honest Jev v2.3, A44 on, shipped executor, clock on
# (results/jev_honest_v23_A44_3a-1); (2) rule with A44 + A50 v2 on 3a-1 with the walk log (results/a44_a50_rule_3a-1);
# (3) rule with A44 + A50 v2 on the five protocol kitchens, clock off (results/sweep_fire_rule_A44_A50_test5).
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/queue_0745.log"
echo "== $(date '+%F %T') queue start" | tee -a "$LOG"
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX HAZARD_WALK_LOG HAZARD_WALK_LOOKAHEAD
export HAZARD_MAP_MAX_OBSTACLE_H=1.8
vr() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$1" 2>/dev/null || echo none; }
echo "== $(date '+%T') honest Jev v2.3 A44 shipped executor 3a-1" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent jev --honest_text --save_maps 3 \
    --api_key_file "G:\iw-thinking-fast\.secrets\typesafe_key.txt" --episodes 1 --port 10826 \
    --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
    --resume_dir "G:/iw-thinking-fast/results/jev_honest_v23_A44_3a-1" \
    --data_dir "G:/iw-thinking-fast/HAZARD/src/HAZARD/data/room_setup_fire/test_set/mm_kitchen_3a-1" --clock tcm --fps 30 \
    > "$ROOT/logs/queue_0745_jev_v23.log" 2>&1
echo "   exit=$? value_rate=$(vr "$ROOT/results/jev_honest_v23_A44_3a-1/mm_kitchen_3a-1/eval_result.json")" | tee -a "$LOG"
export HAZARD_WALK_LOOKAHEAD=1.0
run() {
  echo "== $(date '+%T') rule A44+A50v2 $1 -> $2" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
  HAZARD_WALK_LOG=$3 timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent rule --episodes 1 --port 10826 \
      --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
      --resume_dir "G:/iw-thinking-fast/results/$2" \
      --data_dir "G:/iw-thinking-fast/HAZARD/src/HAZARD/data/room_setup_fire/test_set/$1" --clock none \
      > "$ROOT/logs/queue_0745_$2_$1.log" 2>&1
  echo "   exit=$? value_rate=$(vr "$ROOT/results/$2/$1/eval_result.json")" | tee -a "$LOG"
}
run mm_kitchen_3a-1 a44_a50_rule_3a-1 1
for SCENE in mm_kitchen_3a-1 mm_kitchen_3a-10 mm_kitchen_3a-11 mm_kitchen_3a-12 mm_kitchen_3a-13; do run "$SCENE" sweep_fire_rule_A44_A50_test5 0; done
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') queue done" | tee -a "$LOG"
