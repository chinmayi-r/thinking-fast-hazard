#!/usr/bin/env bash
# HAZARD on adroit-vis: install into the scratch py310 env (mirrors scripts/install_laptop.sh, A1-A5, A10, A11, A14)
# and run ONE fire episode with the random agent, TDW build started by hand on the A100 (Xvfb + vglrun -d egl, A3).
# Usage on adroit-vis:  bash adroit_hazard_first_run.sh 2>&1 | tee logs/hazard_first_run.log
set -uo pipefail
ROOT=/scratch/network/cr7998/iw-thinking-fast
cd "$ROOT" || exit 1
mkdir -p logs results
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp" TORCH_HOME="$ROOT/.cache/torch" HF_HOME="$ROOT/.cache/hf"
PY="$ROOT/py310/bin/python"
PORT=10826

echo "== [1/5] torch 2.1.2 + cu118, mmcv 2.1.0, mmdet 3.x (same pins as the laptop)"
"$PY" -m pip install -q "numpy<2" "setuptools<81" torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 || exit 1
"$PY" -m pip install -q "numpy<2" mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html || exit 1
"$PY" -m pip install -q "numpy<2" "mmdet>=3.0,<3.4" mmengine || exit 1

echo "== [2/5] rest of HAZARD/requirements.txt (opencv headless on a server; gym 0.26.2)"
"$PY" -m pip install -q "numpy<2" opencv-python-headless gym==0.26.2 pyastar2d tiktoken openai backoff inflect streamlit || exit 1

echo "== [3/5] HAZARD at the laptop's commit"
[ -d HAZARD ] || git clone -q https://github.com/UMass-Foundation-Model/HAZARD.git
git -C HAZARD checkout -q d35c71ebd24c08783552bab246df8ac82190e1e2
"$PY" -m pip install -q -e HAZARD --no-deps || exit 1
"$PY" -c "import torch, mmcv, mmdet, tdw, gym, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, '| tdw', tdw.__version__, '| gym', gym.__version__)" || exit 1

echo "== [4/5] display + build (output to logs/build.log so the shell never waits on it)"
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; pkill -u "$USER" Xvfb 2>/dev/null; sleep 1
Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 &
sleep 2; export DISPLAY=:99
vglrun -d egl "$ROOT/tdw_build/TDW/TDW.x86_64" "-port $PORT" > logs/build.log 2>&1 &
sleep 3

echo "== [5/5] one fire episode, random agent, no perception (A6, A9). HAZARD imports need both paths (A10)."
export PYTHONPATH="$ROOT/HAZARD:$ROOT/HAZARD/src/HAZARD"
cd "$ROOT/HAZARD"
OUT="$ROOT/results/fire_random_ep1_$(date +%Y%m%d_%H%M%S)"
START=$(date +%s)
timeout 1800 "$PY" -c "
import time; t0=time.perf_counter()
from HAZARD import submit
submit(output_dir='$OUT', env_name='fire', agent='random', port=$PORT, max_test_episode=1)
print(f'submit() returned after {time.perf_counter()-t0:.1f}s')
"
RC=$?
echo "exit=$RC  wall=$(( $(date +%s) - START ))s  output_dir=$OUT"
ls -la "$OUT" 2>/dev/null | head -20
echo "== build log tail:"; tail -5 "$ROOT/logs/build.log"
echo "== Player.log errors:"; grep -i -E "error|exception" ~/.config/unity3d/MIT/TDW/Player.log 2>/dev/null | head -5
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; pkill -u "$USER" Xvfb 2>/dev/null
echo "== done"
