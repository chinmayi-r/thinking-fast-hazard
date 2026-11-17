#!/usr/bin/env bash
# After the A46 walk-log episode (results/a46_walklog_rule/mm_kitchen_3a-1/summary.json), rerun honest Jev on 3a-1
# with A44 on, saving the question text per decision (jev_decisions.json) and the first three decisions' map arrays.
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/after_a46.log"
until [ -f "$ROOT/results/a46_walklog_rule/mm_kitchen_3a-1/summary.json" ]; do sleep 30; done
sleep 10
echo "== $(date '+%F %T') honest Jev A44 3a-1 with prompt log" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX HAZARD_WALK_LOG
export HAZARD_MAP_MAX_OBSTACLE_H=1.8
timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent jev --honest_text --save_maps 3 \
    --api_key_file "G:\iw-thinking-fast\.secrets\typesafe_key.txt" --episodes 1 --port 10826 \
    --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
    --resume_dir "G:\iw-thinking-fast\results\jev_honest_A44_3a-1_promptlog" \
    --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\mm_kitchen_3a-1" --clock tcm --fps 30 \
    > "$ROOT/logs/after_a46_jev_honest.log" 2>&1
echo "   exit=$?" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') done" | tee -a "$LOG"
