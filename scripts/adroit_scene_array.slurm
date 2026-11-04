#!/bin/bash
# HAZARD scene sweep on Adroit as a Slurm job array: one scene per task, each with its own Xvfb display and TDW
# build on its own port (the recipe verified on adroit-vis, Oct 16). Modelled on the CBM project's arrays and
# requeue pattern (G:\Princeton\CBM-Concept-Backwash\curated\train\mcbm_seeded_job.slurm).
# UNTESTED as of Oct 26: compute nodes have no internet, so HAZARD's asset cache must be filled first (A7).
#
# Submit (from the login node, cwd = /scratch/network/cr7998/iw-thinking-fast):
#   sbatch --export=ALL,AGENT=rule,ENV_NAME=fire,SWEEP=fire_rule_test25 scripts/adroit_scene_array.slurm
# Resubmit the identical line to resume: finished scenes (eval_result.json present) are skipped by the runner.
#SBATCH --job-name=hazard_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --array=0-24%4
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/slurm_%x_%A_%a.out
set -uo pipefail
ROOT=/scratch/network/cr7998/iw-thinking-fast
AGENT="${AGENT:-rule}"; ENV_NAME="${ENV_NAME:-fire}"; SWEEP="${SWEEP:-${ENV_NAME}_${AGENT}_test25}"
EXTRA="${EXTRA:-}"   # e.g. "--clock tcm --fps 30" or "--lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file $ROOT/.secrets/anthropic_key.txt"
TEST="$ROOT/HAZARD/src/HAZARD/data/room_setup_${ENV_NAME/flood/fire}/test_set"
mapfile -t SCENES < <(ls "$TEST" | grep -E "kitchen|craftroom|suburb" | sort | head -25)
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
PORT=$((20000 + SLURM_ARRAY_TASK_ID))
DISP=$((200 + SLURM_ARRAY_TASK_ID))
export PYTHONPATH="$ROOT/HAZARD:$ROOT/HAZARD/src/HAZARD" TMPDIR="$ROOT/.cache/tmp" ANTHROPIC_WORKSPACE_ID='wrkspc_01VPU7qLAqtHXFpUPSQ8DCgd'
PY="$ROOT/py310/bin/python"
trap 'echo "[requeue before wall] $SLURM_JOB_ID"; pkill -u "$USER" -f "TDW.x86_64 -port $PORT"; scontrol requeue "$SLURM_JOB_ID"' USR1
echo "task $SLURM_ARRAY_TASK_ID scene $SCENE port $PORT display :$DISP host $(hostname)"
if [ -f "$ROOT/results/sweep_${SWEEP}/$SCENE/eval_result.json" ]; then echo "already evaluated"; exit 0; fi
Xvfb ":$DISP" -screen 0 1280x720x24 >/dev/null 2>&1 & XVFB=$!
sleep 2; export DISPLAY=":$DISP"
cd "$ROOT/HAZARD"
# the runner starts the build itself with the VirtualGL EGL prefix (A3 on Linux) and times every decision
timeout 40m "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port "$PORT" --env "$ENV_NAME" --agent "$AGENT" --episodes 1 \
    --build "$ROOT/tdw_build/TDW/TDW.x86_64" --launch "vglrun -d egl" --out "$ROOT/results" \
    --resume_dir "$ROOT/results/sweep_${SWEEP}" --data_dir "$TEST/$SCENE" $EXTRA &
wait $!
RC=$?
pkill -u "$USER" -f "TDW.x86_64 -port $PORT" 2>/dev/null; kill $XVFB 2>/dev/null
echo "exit=$RC"
exit $RC
