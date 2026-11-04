#!/usr/bin/env bash
# ThreeDWorld feasibility test on Princeton Adroit, run on adroit-vis (2x A100 80 GB, internet, no scheduler;
# probe Oct 16: vglrun, xpra, Xvfb, glxinfo present; DISPLAY unset over plain SSH).
# Everything goes to /scratch/network/cr7998 (home is at 15.7 of 19.5 GiB). Port 10826 as in HAZARD/starter_code.py.
# Usage on adroit-vis:  bash adroit_tdw_hello.sh 2>&1 | tee hello.log
set -uo pipefail
ROOT=/scratch/network/cr7998/iw-thinking-fast
mkdir -p "$ROOT/.cache/pip" "$ROOT/.cache/tmp" "$ROOT/.conda_pkgs" "$ROOT/tdw_build" "$ROOT/logs"
cd "$ROOT"
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp" CONDA_PKGS_DIRS="$ROOT/.conda_pkgs"
echo "== host $(hostname), $(grep PRETTY_NAME /etc/os-release | cut -d= -f2)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -2
echo "== X sockets on this node: $(ls /tmp/.X11-unix/ 2>/dev/null | tr '\n' ' ')"

echo "== python 3.10 env on scratch (same version as the laptop, A1)"
module load anaconda3/2025.6
[ -x py310/bin/python ] || conda create -y -q -p "$ROOT/py310" python=3.10
PY="$ROOT/py310/bin/python"
"$PY" -m pip install -q tdw==1.13.0.0 "setuptools<81"   # setuptools 84 dropped pkg_resources, which tdw/librarian.py imports
"$PY" -c "import tdw, sys; print('python', sys.version.split()[0], 'tdw', tdw.__version__)"

echo "== TDW Linux build v1.13.0 (141.8 MB), kept on scratch, started by hand (A3)"
cd tdw_build
[ -f TDW_Linux.tar.gz ] || curl -sSL -o TDW_Linux.tar.gz https://github.com/threedworld-mit/tdw/releases/download/v1.13.0/TDW_Linux.tar.gz
[ -x TDW/TDW.x86_64 ] || tar -xzf TDW_Linux.tar.gz
echo "build version $(cat TDW/version.txt)"; cd "$ROOT"

echo "== display: Xvfb :99 for the window; VirtualGL on :0 for the GPU if a GPU-backed X server exists (tdw xpra.md:14-20)"
Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 & XVFB=$!; sleep 2; export DISPLAY=:99
# adroit-vis has no :0 X server (Oct 16: sockets X1..X15 and per-user ones only), so use VirtualGL's EGL back end, which needs none
if vglrun -d egl glxinfo -B 2>/dev/null | grep -q "OpenGL renderer"; then LAUNCH="vglrun -d egl"
elif [ -S /tmp/.X11-unix/X0 ]; then LAUNCH="vglrun -d :0"
else LAUNCH=""; echo "no VirtualGL path works: software rendering only"; fi
echo "renderer via '${LAUNCH:-plain}': $($LAUNCH glxinfo -B 2>/dev/null | grep 'OpenGL renderer' || echo unknown)"

cat > tdw_hello_adroit.py <<'PYEOF'
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
timeout 300 "$PY" tdw_hello_adroit.py "$ROOT/tdw_build/TDW/TDW.x86_64" $LAUNCH || echo "HELLO FAILED (exit $?)"
echo "== Player.log renderer lines:"; grep -i -E "renderer|vendor|opengl|vulkan|gpu|error" ~/.config/unity3d/MIT/TDW/Player.log 2>/dev/null | head -10
kill $XVFB 2>/dev/null || true
echo "== done"
