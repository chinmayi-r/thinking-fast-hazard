#!/usr/bin/env bash
# After scripts/laptop_after_jev22.sh prints "done" (logs/after_jev22.log): honest Jev v2.2 on 3a-1 with A44 and the A50 v2
# lookahead (first segment, 1.0 m), clock on -> results/jev_honest_v22_A44_A50_3a-1 (the first attempt skipped on a path slip).
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/after_a50v2rule.log"
until grep -q "done" "$ROOT/logs/after_jev22.log" 2>/dev/null; do sleep 30; done
sleep 10
echo "== $(date '+%F %T') honest Jev v2.2 A44 + A50 v2 lookahead 1.0, 3a-1" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX HAZARD_WALK_LOG
export HAZARD_MAP_MAX_OBSTACLE_H=1.8 HAZARD_WALK_LOOKAHEAD=1.0
timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent jev --honest_text --save_maps 3 \
    --api_key_file "G:\iw-thinking-fast\.secrets\typesafe_key.txt" --episodes 1 --port 10826 \
    --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
    --resume_dir "G:/iw-thinking-fast/results/jev_honest_v22_A44_A50_3a-1" \
    --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\mm_kitchen_3a-1" --clock tcm --fps 30 \
    > "$ROOT/logs/after_a50v2rule_jev.log" 2>&1
VR=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$ROOT/results/jev_honest_v22_A44_A50_3a-1/mm_kitchen_3a-1/eval_result.json" 2>/dev/null || echo none)
echo "   exit=$? value_rate=$VR" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') done" | tee -a "$LOG"
