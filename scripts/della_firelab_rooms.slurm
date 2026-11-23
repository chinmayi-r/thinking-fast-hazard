#!/bin/bash
# firelab rooms on Della as a Slurm job array: one room per task, one core per task (CR, Nov 23). CPU only.
# Written on the laptop; NOT yet run on Della (A80, A86). Needs scripts/della_firelab_setup.sh to have been run once.
#
# Submit from a login node, from inside the git clone (on Della: ~/thinking-fast-hazard). The job uses the folder
# it was submitted from, so no folder name is written into it:
#   cd ~/thinking-fast-hazard
#   sbatch --export=ALL,TAG=run4_items10_declared,AGENTS="random reflex always commit moving",CLOCK=declared,EXTRA="--items 10 --no-board" scripts/della_firelab_rooms.slurm
# Resubmit the same line to resume: rooms already in results/firelab_sim/$TAG/ are skipped.
# When the array is done, on the login node, from the same folder:
#   PYTHONPATH=$PWD pyk310/bin/python -m firelab.run_rooms --tag run4_items10_declared --report-only
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
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
[ -d "$ROOT/firelab" ] && [ -x "$ROOT/pyk310/bin/python" ] || { echo "submit from the clone after the setup: no firelab/ or pyk310/ in $ROOT"; exit 1; }
TAG="${TAG:?set TAG, the name of the run}"
AGENTS="${AGENTS:-random reflex}"; CLOCK="${CLOCK:-off}"; EXTRA="${EXTRA:-}"
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT" || exit 1
echo "task $SLURM_ARRAY_TASK_ID host $(hostname) cpu $(lscpu | grep 'Model name' | sed 's/Model name: *//') tag $TAG agents $AGENTS clock $CLOCK"
# shellcheck disable=SC2086
"$ROOT/pyk310/bin/python" -m firelab.run_rooms --tag "$TAG" --first "$SLURM_ARRAY_TASK_ID" --rooms 1 \
    --agents $AGENTS --clock "$CLOCK" $EXTRA 2>&1 | grep -v "inertial\|b3Warning\|_link$"
