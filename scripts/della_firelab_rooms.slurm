#!/bin/bash
# firelab rooms on Della as a Slurm job array: one room per task, one core per task (CR, Nov 24). CPU only.
# Written on the laptop; NOT yet run on Della (A80). Needs scripts/della_firelab_setup.sh to have been run once.
#
# Submit from a login node, cwd = /home/cr7998/iw-thinking-fast:
#   sbatch --export=ALL,TAG=run3_measured,AGENTS="always commit moving",CLOCK=measured,EXTRA="--no-board" scripts/della_firelab_rooms.slurm
# Resubmit the same line to resume: rooms already in results/firelab_sim/$TAG/ are skipped.
# When the array is done, on the login node:
#   PYTHONPATH=$PWD pyk310/bin/python -m firelab.run_rooms --tag run3_measured --report-only
# The measured clock charges real seconds, so each task gets one core to itself and BLAS is held to one thread;
# other jobs on the same node can still slow memory access a little. Every result file records the node's CPU
# and the time of one fixed reference plan.
#SBATCH --job-name=firelab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/slurm_%x_%A_%a.out
set -uo pipefail
ROOT=/home/cr7998/iw-thinking-fast
TAG="${TAG:?set TAG, the name of the run}"
AGENTS="${AGENTS:-random reflex}"; CLOCK="${CLOCK:-off}"; EXTRA="${EXTRA:-}"
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT" || exit 1
echo "task $SLURM_ARRAY_TASK_ID host $(hostname) cpu $(lscpu | grep 'Model name' | sed 's/Model name: *//') tag $TAG agents $AGENTS clock $CLOCK"
# shellcheck disable=SC2086
"$ROOT/pyk310/bin/python" -m firelab.run_rooms --tag "$TAG" --first "$SLURM_ARRAY_TASK_ID" --rooms 1 \
    --agents $AGENTS --clock "$CLOCK" $EXTRA 2>&1 | grep -v "inertial\|b3Warning\|_link$"
