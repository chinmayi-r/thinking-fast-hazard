#!/usr/bin/env bash
# One-time setup of HAZARD + ThreeDWorld on Princeton Della. Run on della-vis1 (has internet and vglrun).
# Mirrors scripts/adroit_tdw_hello.sh + adroit_hazard_first_run.sh (Oct 16/26), with Della paths (NOTES Nov 8).
# Everything lives on Tom's fileset, nothing in /home. Ends with a TDW hello-world on the A100 via Xvfb + vglrun -d egl.
# Usage:  ssh -m hmac-sha2-512 cr7998@della-vis1.princeton.edu
#         mkdir -p /home/cr7998/iw-thinking-fast/logs && cd /home/cr7998/iw-thinking-fast
#         bash scripts/della_setup.sh 2>&1 | tee logs/della_setup.log
set -uo pipefail
ROOT=/home/cr7998/iw-thinking-fast
mkdir -p "$ROOT/.cache/pip" "$ROOT/.cache/tmp" "$ROOT/.conda_pkgs" "$ROOT/tdw_build" "$ROOT/logs" "$ROOT/results"
cd "$ROOT" || exit 1
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp" CONDA_PKGS_DIRS="$ROOT/.conda_pkgs"
export TORCH_HOME="$ROOT/.cache/torch" HF_HOME="$ROOT/.cache/hf"
echo "== host $(hostname), $(grep PRETTY_NAME /etc/os-release | cut -d= -f2)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -2

echo "== [1/6] python 3.10 env on scratch (A1: same version as the laptop and adroit)"
export PS1="${PS1:-}"; module load anaconda3/2025.6
[ -x py310/bin/python ] || conda create -y -q -p "$ROOT/py310" python=3.10
PY="$ROOT/py310/bin/python"
"$PY" -m pip install -q "numpy<2" "setuptools<81" tdw==1.13.0.0 || exit 1   # setuptools 84 dropped pkg_resources, which tdw imports

echo "== [2/6] torch 2.1.2 + cu118, mmcv 2.1.0, mmdet 3.x, rest of HAZARD/requirements.txt (same pins as laptop and adroit)"
"$PY" -m pip install -q "numpy<2" torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 || exit 1
"$PY" -m pip install -q "numpy<2" mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html || exit 1
"$PY" -m pip install -q "numpy<2" "mmdet>=3.0,<3.4" mmengine || exit 1
"$PY" -m pip install -q "numpy<2" opencv-python-headless gym==0.26.2 pyastar2d tiktoken openai anthropic backoff inflect streamlit pandas || exit 1

echo "== [3/6] TDW Linux build v1.13.0 on scratch (A3: started by hand, never by tdw's launcher)"
cd tdw_build
[ -f TDW_Linux.tar.gz ] || curl -sSL -o TDW_Linux.tar.gz https://github.com/threedworld-mit/tdw/releases/download/v1.13.0/TDW_Linux.tar.gz
[ -x TDW/TDW.x86_64 ] || tar -xzf TDW_Linux.tar.gz
echo "build version $(cat TDW/version.txt)"; cd "$ROOT"

echo "== [4/6] HAZARD at the laptop's commit d35c71e, plus the three local patches (A16, A19, A18/A20; all apply cleanly, checked Nov 8)"
[ -d HAZARD ] || git clone -q https://github.com/UMass-Foundation-Model/HAZARD.git
git -C HAZARD checkout -q d35c71ebd24c08783552bab246df8ac82190e1e2
if git -C HAZARD diff --quiet; then
  git -C HAZARD apply "$ROOT/scripts/A16_challenge_pickup_guard.diff" "$ROOT/scripts/A19_previous_action_chain.diff" "$ROOT/scripts/A18_A20_llm_anthropic_openai_vlm_v3.diff" || exit 1
  echo "patches applied"
else
  echo "HAZARD already has local changes; patches not re-applied"
fi
"$PY" -m pip install -q -e HAZARD --no-deps || exit 1
# HAZARD's cached-asset mode resolves <cwd>/data/assets; the runner's cwd is HAZARD/, the data lives in src/HAZARD/data (A28)
[ -e HAZARD/data ] || ln -s "$ROOT/HAZARD/src/HAZARD/data" HAZARD/data
mkdir -p HAZARD/src/HAZARD/data/assets
"$PY" -c "import torch, mmcv, mmdet, tdw, gym, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, '| tdw', tdw.__version__, '| gym', gym.__version__)" || exit 1

echo "== [5/6] display: Xvfb for the window, VirtualGL EGL back end for the GPU (the adroit-vis recipe; no :0 server, no sudo)"
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; pkill -u "$USER" Xvfb 2>/dev/null; sleep 1
Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 & XVFB=$!; sleep 2; export DISPLAY=:99
echo "renderer: $(vglrun -d egl glxinfo -B 2>/dev/null | grep 'OpenGL renderer' || echo 'EGL path failed')"

echo "== [6/6] hello world: build on the GPU, controller connects, three empty steps (same test as adroit_tdw_hello.sh)"
cat > "$ROOT/scripts/tdw_hello_linux.py" <<'PYEOF'
import subprocess, sys, time
build = sys.argv[1]; port = 10826; launch = sys.argv[2:]
t0 = time.perf_counter()
p = subprocess.Popen(launch + [build, f"-port {port}"])
from tdw.controller import Controller
c = Controller(port=port, check_version=False, launch_build=False)
print(f"connected after {time.perf_counter()-t0:.2f}s; build {c._tdw_version} unity {c._unity_version}")
for i in range(3):
    t = time.perf_counter(); c.communicate([]); print(f"empty communicate #{i}: {(time.perf_counter()-t)*1000:.1f} ms")
print("Hello world!"); c.communicate({"$type": "terminate"})
try: p.wait(timeout=15)
except subprocess.TimeoutExpired: p.kill(); print("build did not exit; killed")
PYEOF
timeout 300 "$PY" "$ROOT/scripts/tdw_hello_linux.py" "$ROOT/tdw_build/TDW/TDW.x86_64" vglrun -d egl 2>&1 || echo "HELLO FAILED (exit $?)"
echo "== Player.log renderer lines:"; grep -i -E "renderer|vendor|opengl|vulkan" ~/.config/unity3d/MIT/TDW/Player.log 2>/dev/null | head -5
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; kill $XVFB 2>/dev/null
echo "== done. Next: bash scripts/della_cache_assets.sh"
