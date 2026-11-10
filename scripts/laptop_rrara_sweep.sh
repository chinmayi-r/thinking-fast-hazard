#!/usr/bin/env bash
# RRARA-v1 sweep on the laptop: the 25 fire test scenes, one process at a time (one TDW build on port 10826), resumable
# (scenes with eval_result.json are skipped by run_baseline.py --resume_dir). The reflector needs the API key, which stays
# on the laptop (.secrets/anthropic_key.txt, read by HAZARD's LLM class from the path; never printed), so the reflector
# modes run here and the reflector-off sweep runs on Della (scripts/della_scene_array.slurm with AGENT=rrara_v1).
# Usage (Git Bash, from G:\iw-thinking-fast):
#   MODE=faithful bash scripts/laptop_rrara_sweep.sh      # or MODE=worldtime, MODE=off; REFLEX=paper|hazard_rule
# Nov 9: about 10 min per scene (570-590 s) incl. a 60 s reset, so 25 scenes is roughly 4.5 h. Keep heavy apps closed
# (NOTES Oct 26: a 16 GB laptop hit its commit limit once during a sweep).
set -uo pipefail
ROOT=/g/iw-thinking-fast
MODE="${MODE:-faithful}"; REFLEX="${REFLEX:-paper}"; LM="${LM:-claude-haiku-5-5}"; CAP="${CAP:-1500}"
SWEEP="fire_rrara_v1_${REFLEX}_${MODE}_tcm30_test25"
OUT="$ROOT/results/sweep_$SWEEP"; mkdir -p "$OUT" "$ROOT/logs"
TEST="$ROOT/HAZARD/src/HAZARD/data/room_setup_fire/test_set"
export ANTHROPIC_WORKSPACE_ID="${ANTHROPIC_WORKSPACE_ID:-wrkspc_01VPU7qLAqtHXFpUPSQ8DCgd}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="G:/iw-thinking-fast/HAZARD;G:/iw-thinking-fast/HAZARD/src/HAZARD"
cd "$ROOT/HAZARD" || exit 1
echo "== $(date '+%F %T') sweep $SWEEP start" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
for SCENE in $(ls "$TEST" | sort | head -25); do
  if [ -f "$OUT/$SCENE/eval_result.json" ]; then echo "skip $SCENE (done)" | tee -a "$ROOT/logs/sweep_$SWEEP.log"; continue; fi
  echo "== $(date '+%T') $SCENE" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
  pkill -f "TDW.exe" 2>/dev/null; sleep 2
  timeout "$CAP" "$ROOT/venv/Scripts/python.exe" -X utf8 "$ROOT/scripts/run_baseline.py" --env fire --agent rrara_v1 \
      --reflex "$REFLEX" --reflector_mode "$MODE" --lm_id "$LM" --api_key_file "G:\iw-thinking-fast\.secrets\anthropic_key.txt" \
      --episodes 1 --port 10826 --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out "G:\iw-thinking-fast\results" \
      --resume_dir "G:\iw-thinking-fast\results\sweep_$SWEEP" --data_dir "G:\iw-thinking-fast\HAZARD\src\HAZARD\data\room_setup_fire\test_set\\$SCENE" \
      --clock tcm --fps 30 > "$ROOT/logs/sweep_${SWEEP}_$SCENE.log" 2>&1
  RC=$?
  VR=$("$ROOT/venv/Scripts/python.exe" -c "import json,sys; d=json.load(open(sys.argv[1])); print(round(d['avg_score']/d['avg_max_score'],3))" "$OUT/$SCENE/eval_result.json" 2>/dev/null || echo "none")
  echo "   exit=$RC value_rate=$VR" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
done
pkill -f "TDW.exe" 2>/dev/null
echo "== $(date '+%F %T') sweep $SWEEP done: $(ls "$OUT"/*/eval_result.json 2>/dev/null | wc -l)/25 scenes" | tee -a "$ROOT/logs/sweep_$SWEEP.log"
