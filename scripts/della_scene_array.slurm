#!/bin/bash
# HAZARD scene sweep on Della as a Slurm job array: one scene per task, own Xvfb display, own TDW build port, GPU via
# vglrun -d egl, assets from the local cache (compute nodes have no internet; NOTES Nov 8, A28). No --partition: Della
# routes GPU jobs by --gres and rejects "--partition=gpu" (sbatch error, Nov 8).
# Adapted from scripts/adroit_scene_array.slurm (never run on Adroit) with the Oct 26 lesson applied: OMP_NUM_THREADS=4,
# otherwise HAZARD's per-frame semantic-map update spreads over every core and runs 11x slower.
#
# Submit from della9 or della-vis1, cwd = /home/cr7998/iw-thinking-fast:
#   sbatch --export=ALL,AGENT=rule,ENV_NAME=fire scripts/della_scene_array.slurm
#   sbatch --export=ALL,AGENT=mcts,ENV_NAME=fire,EXTRA="--clock tcm --fps 30",SWEEP=fire_mcts_tcm30_test25 scripts/della_scene_array.slurm
# Resubmit the same line to resume: scenes with eval_result.json are skipped. Latencies here are "della" numbers (A15).
#SBATCH --job-name=hazard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-24%10
#SBATCH --output=logs/slurm_%x_%A_%a.out
set -uo pipefail
ROOT=/home/cr7998/iw-thinking-fast
AGENT="${AGENT:-rule}"; ENV_NAME="${ENV_NAME:-fire}"; SWEEP="${SWEEP:-${ENV_NAME}_${AGENT}_test25}"
EXTRA="${EXTRA:-}"   # e.g. "--clock tcm --fps 30"; LLM agents need --lm_source/--lm_id/--api_key_file and internet, so they do not run here
TEST="$ROOT/HAZARD/src/HAZARD/data/room_setup_${ENV_NAME/flood/fire}/test_set"
mapfile -t SCENES < <(ls "$TEST" | sort | head -25)
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
PORT=$((20000 + SLURM_ARRAY_TASK_ID))
DISP=$((200 + SLURM_ARRAY_TASK_ID))
export PYTHONPATH="$ROOT/HAZARD:$ROOT/HAZARD/src/HAZARD" TMPDIR="$ROOT/.cache/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="$ROOT/py310/bin/python"
echo "task $SLURM_ARRAY_TASK_ID scene $SCENE port $PORT display :$DISP host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ -f "$ROOT/results/sweep_${SWEEP}/$SCENE/eval_result.json" ]; then echo "already evaluated"; exit 0; fi
Xvfb ":$DISP" -screen 0 1280x720x24 > "$ROOT/logs/xvfb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 & XVFB=$!
for _ in $(seq 20); do [ -S "/tmp/.X11-unix/X$DISP" ] && break; sleep 1; done   # a build launched on a display that is not up never connects (A22 item 11)
[ -S "/tmp/.X11-unix/X$DISP" ] || { echo "XVFB FAILED: no socket after 20 s"; cat "$ROOT/logs/xvfb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"; exit 3; }
export DISPLAY=":$DISP"
echo "renderer: $(timeout 30 vglrun -d egl glxinfo -B 2>&1 | grep -E 'OpenGL renderer|error' | head -1)"   # must say NVIDIA; a Mesa/llvmpipe line means EGL did not get the GPU
cd "$ROOT/HAZARD"
timeout 55m "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port "$PORT" --env "$ENV_NAME" --agent "$AGENT" --episodes 1 \
    --build "$ROOT/tdw_build/TDW/TDW.x86_64" --launch "vglrun -d egl" --out "$ROOT/results" \
    --resume_dir "$ROOT/results/sweep_${SWEEP}" --data_dir "$TEST/$SCENE" --cached_assets     --build_args "-logFile $ROOT/logs/unity_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" $EXTRA
RC=$?
pkill -u "$USER" -f "TDW.x86_64 -port $PORT" 2>/dev/null; kill $XVFB 2>/dev/null
if [ -s "$ROOT/HAZARD/src/HAZARD/data/assets/download_assets.sh" ]; then echo "ASSETS MISSING: HAZARD wrote data/assets/download_assets.sh; run it on della-vis1 and resubmit"; fi
echo "exit=$RC"
exit $RC
