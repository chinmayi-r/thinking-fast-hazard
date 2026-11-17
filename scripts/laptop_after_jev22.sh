#!/usr/bin/env bash
# After scripts/laptop_after_a50.sh prints "done" (logs/after_a50.log): rule on 3a-1 with A50 v2 (first-segment lookahead),
# shipped map, clock off, walk log on -> results/a50v2_lookahead_rule/mm_kitchen_3a-1.
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/after_jev22.log"
until grep -q "done" "$ROOT/logs/after_a50.log" 2>/dev/null; do sleep 30; done
sleep 10
echo "== $(date '+%F %T') A50 v2 lookahead 1.0 (first segment), rule 3a-1, walk log" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HAZARD_WALK_LOG=1 HAZARD_WALK_LOOKAHEAD=1.0
unset HAZARD_ASTAR_FIX HAZARD_MAP_MAX_OBSTACLE_H
timeout 1500 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent rule --episodes 1 --port 10826 \
    --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
    --resume_dir "G:\iw-thinking-fast\results\a50v2_lookahead_rule" \
    --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\mm_kitchen_3a-1" --clock none \
    > "$ROOT/logs/after_jev22_a50v2.log" 2>&1
echo "   exit=$? lines=$(wc -l < "$ROOT/results/a50v2_lookahead_rule/mm_kitchen_3a-1/walk_log.jsonl" 2>/dev/null)" | tee -a "$LOG"
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') done" | tee -a "$LOG"
