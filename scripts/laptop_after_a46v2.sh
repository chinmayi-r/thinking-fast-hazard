#!/usr/bin/env bash
# After scripts/laptop_after_jev.sh prints "done" (logs/after_jev.log): honest Jev on 3a-1 again, A44 on, clock on, with the
# honest text fixed (goal footprint not counted as a block; fire line from the per-object record), prompt log + 3 maps.
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/after_a46v2.log"
until grep -q "done" "$ROOT/logs/after_jev.log" 2>/dev/null; do sleep 30; done
sleep 10
echo "== $(date '+%F %T') honest Jev A44 3a-1, text v2.1 (goal footprint + object fire)" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX HAZARD_WALK_LOG
export HAZARD_MAP_MAX_OBSTACLE_H=1.8
timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent jev --honest_text --save_maps 3 \
    --api_key_file "G:\iw-thinking-fast\.secrets\typesafe_key.txt" --episodes 1 --port 10826 \
    --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
    --resume_dir "G:\iw-thinking-fast\results\jev_honest_A44_3a-1_v21" \
    --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\mm_kitchen_3a-1" --clock tcm --fps 30 \
    > "$ROOT/logs/after_a46v2_jev_honest.log" 2>&1
echo "   exit=$?" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') done" | tee -a "$LOG"
