#!/usr/bin/env bash
# One-time setup of the kinder simulator and firelab on Della. CPU only, no display, no GPU.
# Written on the laptop; NOT yet run to the end on Della (A80, A86). Run on a login node (it needs the internet),
# from the git clone, which on Della is ~/thinking-fast-hazard (NOT ~/iw-thinking-fast: that is the HAZARD work
# folder and is not a repository; A86):
#   ssh -m hmac-sha2-512 cr7998@della.princeton.edu
#   cd ~/thinking-fast-hazard && git pull && bash scripts/della_firelab_setup.sh 2>&1 | tee logs/della_firelab_setup.log
# The script finds the repository from its own location, so no folder name is written into it. Inside the clone it
# makes a Python 3.10 environment (pyk310) and a clone of kindergarden, both ignored by git, installs the versions
# the laptop has (venv_kinder, Nov 22), and ends with the firelab test suite on a compute node.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$ROOT/firelab" ] && [ -e "$ROOT/.git" ] || { echo "$ROOT is not the thinking-fast-hazard clone (no firelab/ or no .git)"; exit 1; }
mkdir -p "$ROOT/.cache/pip" "$ROOT/.cache/tmp" "$ROOT/.conda_pkgs" "$ROOT/logs"
cd "$ROOT" || exit 1
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp" CONDA_PKGS_DIRS="$ROOT/.conda_pkgs"
echo "== host $(hostname), $(lscpu | grep 'Model name' | sed 's/Model name: *//'), repository $ROOT at $(git rev-parse --short HEAD)"

echo "== [1/4] python 3.10 environment pyk310"
export PS1="${PS1:-}"; module load anaconda3/2025.6   # PS1: 'module load' aborts under 'set -u' without it (Nov 8)
[ -x pyk310/bin/python ] || conda create -y -q -p "$ROOT/pyk310" python=3.10 || exit 1
PY="$ROOT/pyk310/bin/python"

echo "== [2/4] kindergarden at 86c55a4 (the commit the laptop uses; public repository, no large-file storage)"
[ -d kindergarden ] || git clone -q https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git || exit 1
git -C kindergarden checkout -q 86c55a4 || exit 1

echo "== [3/4] packages, pinned to the laptop's versions"
"$PY" -m pip install -q numpy==1.26.4 gymnasium==1.4.0 scipy==1.14.0 shapely==2.1.2 dill==0.4.1 matplotlib pillow \
    relational_structs==0.0.1 prpl_utils==0.0.7 pybullet_helpers==0.1.2 pytest==9.2.0 || exit 1
# The laptop has the package 'pybullet-arm64' 3.2.8 (kindergarden's own pin). If Linux has no build of it, plain
# 'pybullet' provides the same module; either may compile from source for several minutes.
"$PY" -m pip install -q pybullet-arm64==3.2.8 || "$PY" -m pip install -q pybullet || exit 1
"$PY" -m pip install -q --no-deps -e kindergarden || exit 1
export PYTHONPATH="$ROOT"
"$PY" -c "import sys, numpy, gymnasium, pybullet, kinder, firelab; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'gymnasium', gymnasium.__version__, 'pybullet api', pybullet.getAPIVersion()); print('ids', firelab.ENV_IDS)" || exit 1

echo "== [4/4] the firelab tests on one core of a compute node (a few minutes)"
srun --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
    env PYTHONPATH="$ROOT" "$PY" -m pytest -p no:cacheprovider -q firelab 2>&1 | grep -v "inertial\|b3Warning\|_link$" | tail -15
echo "== done; paste the lines from '== host' down"
