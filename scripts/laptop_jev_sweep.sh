#!/usr/bin/env bash
# Jev (A37) on the laptop: the five protocol kitchens (scripts/human/PROTOCOL.md), HAZARD as shipped (no HAZARD_ASTAR_FIX,
# no --honest_text, HAZARD's prompt_v2.csv), clock on (frames = ceil(seconds x 30)), one process per kitchen on port 10826,
# resumable (kitchens with eval_result.json are skipped). The TypeSafe key stays in .secrets/typesafe_key.txt and is read
# only by JevClient through --api_key_file; never printed. CR's go: Nov 17 ("Jev 5 kitchens", "orii hazard no mod").
# Usage (Git Bash, from G:\iw-thinking-fast):   bash scripts/laptop_jev_sweep.sh
#   KITCHENS="mm_kitchen_3a-1 ..." to choose; CLOCK=none for the clock off; HONEST=1 adds --honest_text (A41) and a
#   separate sweep folder.
set -uo pipefail
ROOT=/g/iw-thinking-fast
CLOCK="${CLOCK:-tcm}"; HONEST="${HONEST:-0}"; CAP="${CAP:-1500}"
KITCHENS="${KITCHENS:-mm_kitchen_3a-1 mm_kitchen_3a-10 mm_kitchen_3a-11 mm_kitchen_3a-12 mm_kitchen_3a-13}"
TAG="jev$([ "$HONEST" = 1 ] && echo _honest)_test5$([ "$CLOCK" = tcm ] && echo _tcm30 || echo _noclock)"
SWEEP="fire_$TAG"
OUT="$ROOT/results/sweep_$SWEEP"; mkdir -p "$OUT" "$ROOT/logs"
EXTRA=""; [ "$HONEST" = 1 ] && EXTRA="--honest_text"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD"
unset HAZARD_ASTAR_FIX  # shipped planner
cd "$ROOT/HAZARD" || exit 1
echo "== $(date '+%F %T') sweep $SWEEP start (clock=$CLOCK honest=$HONEST)" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
for SCENE in $KITCHENS; do
  if [ -f "$OUT/$SCENE/eval_result.json" ]; then echo "skip $SCENE (done)" | tee -a "$ROOT/logs/sweep_$SWEEP.log"; continue; fi
  echo "== $(date '+%T') $SCENE" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 2
  timeout "$CAP" "$ROOT/venv/Scripts/python.exe" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent jev $EXTRA \
      --api_key_file "G:\iw-thinking-fast\.secrets\typesafe_key.txt" \
      --episodes 1 --port 10826 --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
      --resume_dir "G:\iw-thinking-fast\results\sweep_$SWEEP" --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\\$SCENE" \
      --clock "$CLOCK" --fps 30 > "$ROOT/logs/sweep_${SWEEP}_$SCENE.log" 2>&1
  RC=$?
  VR=$("$ROOT/venv/Scripts/python.exe" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$OUT/$SCENE/eval_result.json" 2>/dev/null || echo "none")
  echo "   exit=$RC value_rate=$VR" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
done
taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "== $(date '+%F %T') sweep $SWEEP done: $(ls "$OUT"/*/eval_result.json 2>/dev/null | wc -l)/$(echo $KITCHENS | wc -w) kitchens" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
