#!/usr/bin/env bash
# Nov 17 13:00: reflex alone (rrara_v1, reflector off) on 3a-1 with A44 (1.8 m) + A50 v3 (line-of-sight waypoint, 1.0 m) + A53,
# clock off then clock on -> results/reflex_A44_A50v3_A53_3a-1{,_tcm30}. Companion to the human sittings (0.222) and rule (0.111).
set -uo pipefail
ROOT=/g/iw-thinking-fast
PY="$ROOT/venv/Scripts/python.exe"
LOG="$ROOT/logs/reflex_a50v3.log"
# (no blanket TDW kill: CR's human page of 11:31 on port 10826 is still open)
cd "$ROOT/HAZARD" || exit 1
export PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HAZARD_ASTAR_FIX
export HAZARD_MAP_MAX_OBSTACLE_H=1.8 HAZARD_WALK_LOOKAHEAD=1.0 HAZARD_WALK_LOOKAHEAD_MODE=los HAZARD_MAP_DROP_OOB=1 HAZARD_WALK_LOG=1
for CLK in ${CLKS:-off tcm}; do
  if [ "$CLK" = off ]; then OUT=reflex_A44_A50v3_A53_3a-1; CARGS=""; PORT=10830; else OUT=reflex_A44_A50v3_A53_3a-1_tcm30; CARGS="--clock tcm --fps 30"; PORT=10831; fi
  echo "== $(date '+%F %T') reflex 3a-1 clock=$CLK -> $OUT" | tee -a "$LOG"
  timeout 3000 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent rrara_v1 --reflector_mode off --save_maps 3 \
      --episodes 1 --port $PORT --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
      --resume_dir "G:/iw-thinking-fast/results/$OUT" \
      --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\mm_kitchen_3a-1" $CARGS \
      > "$ROOT/logs/reflex_a50v3_$CLK.log" 2>&1
  RC=$?
  VR=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$ROOT/results/$OUT/mm_kitchen_3a-1/eval_result.json" 2>/dev/null || echo none)
  echo "   exit=$RC value_rate=$VR" | tee -a "$LOG"
  sleep 3
done
echo "== $(date '+%F %T') done" | tee -a "$LOG"
