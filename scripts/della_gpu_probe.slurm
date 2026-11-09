#!/bin/bash
# Diagnose why the TDW build does not come up on a Della compute node (array 17136282, Nov 8: every task sat at
# "Resetting Environment" until the 55 min kill, exit=124; build never connected). One task, 10 minutes, prints:
# the GPU the cgroup exposes, the DRI/EGL devices, Xvfb's own output, vglrun -d egl glxinfo, then the hello-world
# with the build's stdout and Unity's Player.log written to logs/ (not ~/.config, which every task would share).
# Submit from ~/iw-thinking-fast:  sbatch scripts/della_gpu_probe.slurm   then   cat logs/gpu_probe_<jobid>.out
#SBATCH --job-name=gpuprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_probe_%j.out
set -uo pipefail
ROOT=/home/cr7998/iw-thinking-fast
PY="$ROOT/py310/bin/python"
export PYTHONPATH="$ROOT/HAZARD:$ROOT/HAZARD/src/HAZARD" TMPDIR="$ROOT/.cache/tmp" OMP_NUM_THREADS=4
DISP=250; PORT=20250
echo "== node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi -L; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
echo "== devices"; ls -la /dev/nvidia* /dev/dri 2>&1 | head -20
echo "== X socket dir"; ls -ld /tmp/.X11-unix 2>&1; ls /tmp/.X11-unix 2>&1
echo "== Xvfb :$DISP (its own output follows)"
Xvfb ":$DISP" -screen 0 1280x720x24 > "$ROOT/logs/xvfb_probe.log" 2>&1 & XVFB=$!
for _ in 1 2 3 4 5 6 7 8; do [ -S "/tmp/.X11-unix/X$DISP" ] && break; sleep 1; done
[ -S "/tmp/.X11-unix/X$DISP" ] && echo "socket up" || echo "NO SOCKET after 8 s"
cat "$ROOT/logs/xvfb_probe.log"; export DISPLAY=":$DISP"
echo "== plain glxinfo (software path)"; timeout 30 glxinfo -B 2>&1 | grep -E "renderer|vendor|error|Error" | head -5
echo "== vglrun -d egl glxinfo"; timeout 30 vglrun -d egl glxinfo -B 2>&1 | grep -E "renderer|vendor|error|Error|VGL" | head -8
echo "== vglrun +v (verbose) first lines"; timeout 30 vglrun +v -d egl glxinfo -B 2>&1 | head -12
echo "== EGL device list via eglinfo (if present)"; command -v eglinfo >/dev/null && timeout 30 eglinfo 2>&1 | grep -E "device|Device|renderer|vendor" | head -8 || echo "no eglinfo"
echo "== hello world: build stdout -> logs/build_probe.log, Unity log -> logs/unity_probe.log, 120 s cap"
cat > "$ROOT/.cache/tmp/hello_probe.py" <<'PYEOF'
import subprocess, sys, time
build, port, logf = sys.argv[1], int(sys.argv[2]), sys.argv[3]; launch = sys.argv[4:]
t0 = time.perf_counter()
p = subprocess.Popen(launch + [build, f"-port {port}", "-logFile", logf], stdout=open(logf + ".stdout", "w"), stderr=subprocess.STDOUT)
from tdw.controller import Controller
c = Controller(port=port, check_version=False, launch_build=False)
print(f"connected after {time.perf_counter()-t0:.2f}s; build {c._tdw_version} unity {c._unity_version}", flush=True)
for i in range(3):
    t = time.perf_counter(); c.communicate([]); print(f"empty communicate #{i}: {(time.perf_counter()-t)*1000:.1f} ms", flush=True)
print("Hello world!"); c.communicate({"$type": "terminate"})
try: p.wait(timeout=15)
except subprocess.TimeoutExpired: p.kill(); print("build did not exit; killed")
PYEOF
timeout 120 "$PY" "$ROOT/.cache/tmp/hello_probe.py" "$ROOT/tdw_build/TDW/TDW.x86_64" $PORT "$ROOT/logs/unity_probe.log" vglrun -d egl 2>&1; echo "hello exit=$?"
pkill -u "$USER" -f "TDW.x86_64 -port $PORT" 2>/dev/null
echo "== build process stdout:"; cat "$ROOT/logs/unity_probe.log.stdout" 2>/dev/null | head -20
echo "== Unity Player.log (renderer / error lines, then last 15 lines):"
grep -i -E "renderer|vendor|opengl|vulkan|error|fail|display" "$ROOT/logs/unity_probe.log" 2>/dev/null | head -15; echo ...; tail -15 "$ROOT/logs/unity_probe.log" 2>/dev/null
kill $XVFB 2>/dev/null; echo "== done"
