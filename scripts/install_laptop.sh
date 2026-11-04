#!/usr/bin/env bash
# Laptop install for HAZARD + ThreeDWorld on G, per ASSUMPTIONS.md A1-A5, A10, A11 (approved Oct 16 2026).
# Run from Git Bash:  bash scripts/install_laptop.sh 2>&1 | tee logs/install_laptop.log
# Python: CPython 3.10.11 from the official NuGet package, extracted to G:\iw-thinking-fast\python310
# (the Store 3.10 on this machine cannot be launched from a non-interactive shell, error 0x80070005).
set -euo pipefail
ROOT=/g/iw-thinking-fast
export PIP_CACHE_DIR='G:\iw-thinking-fast\.cache\pip'
export TORCH_HOME='G:\iw-thinking-fast\.cache\torch'
export HF_HOME='G:\iw-thinking-fast\.cache\hf'
# pip unpacks wheels in %TEMP%, which is on C (A4): move the temp dir to G as well
export TMP='G:\iw-thinking-fast\.cache\tmp'
export TEMP='G:\iw-thinking-fast\.cache\tmp'
export TMPDIR='G:\iw-thinking-fast\.cache\tmp'
PY="$ROOT/venv/Scripts/python.exe"

echo "== [1/6] numpy<2 (torch 2.1 and the mmcv 2.1 wheels were built against numpy 1.x)"
"$PY" -m pip install "numpy<2"

echo "== [2/6] torch 2.1.2 + torchvision 0.16.2, CUDA 11.8 wheels (A5)"
"$PY" -m pip install "numpy<2" torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo "== [3/6] mmcv 2.1.0 prebuilt for cu118/torch2.1 (A5; mmdet needs mmcv<2.2)"
"$PY" -m pip install "numpy<2" mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

echo "== [4/6] mmdet 3.x + mmengine"
"$PY" -m pip install "numpy<2" "mmdet>=3.0,<3.4" mmengine

echo "== [5/6] tdw 1.13.0.0 (A2) and the rest of HAZARD/requirements.txt (A11: gym 0.26.2)"
"$PY" -m pip install "numpy<2" tdw==1.13.0.0 opencv-python gym==0.26.2 pyastar2d tiktoken openai backoff inflect streamlit

echo "== [6/6] HAZARD editable install (A10)"
"$PY" -m pip install -e "$ROOT/HAZARD" --no-deps

echo "== setuptools<81: setuptools 84 dropped pkg_resources, which tdw/librarian.py:3 imports (found on adroit-vis, Oct 16)"
"$PY" -m pip install "setuptools<81"

echo "== pip check"
"$PY" -m pip check || true

echo "== import smoke test"
"$PY" -c "import numpy, torch, torchvision, mmcv, mmdet, tdw, gym, pyastar2d, cv2; print('numpy', numpy.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda); print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__); print('tdw', tdw.__version__, 'gym', gym.__version__, 'cv2', cv2.__version__)"
echo "== done"
