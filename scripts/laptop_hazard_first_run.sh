#!/usr/bin/env bash
# One HAZARD fire episode, random agent, on the laptop. Build started by hand from G (A3), port 10826 (starter_code.py).
# Usage (Git Bash):  bash scripts/laptop_hazard_first_run.sh [env] [agent]   e.g. fire random | flood random | wind random
set -uo pipefail
ROOT=/g/iw-thinking-fast
ENV_NAME="${1:-fire}"; AGENT="${2:-random}"; PORT=10826
export TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp' TORCH_HOME='G:\iw-thinking-fast\.cache\torch' HF_HOME='G:\iw-thinking-fast\.cache\hf'
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD'   # A10: both paths
PY="$ROOT/venv/Scripts/python.exe"
mkdir -p "$ROOT/logs" "$ROOT/results"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="G:\\iw-thinking-fast\\results\\${ENV_NAME}_${AGENT}_ep1_${STAMP}"

echo "== build: TDW.exe -port $PORT (output to logs/build_${STAMP}.log)"
taskkill //F //IM TDW.exe >/dev/null 2>&1 || true
"$ROOT/tdw_build/TDW/TDW.exe" "-port $PORT" > "$ROOT/logs/build_${STAMP}.log" 2>&1 &
BUILD_PID=$!
sleep 5

echo "== submit(env_name=$ENV_NAME, agent=$AGENT, max_test_episode=1, port=$PORT) -> $OUT"
cd "$ROOT/HAZARD"
START=$(date +%s)
"$PY" -c "
import time; t0 = time.perf_counter()
from HAZARD import submit
submit(output_dir=r'$OUT', env_name='$ENV_NAME', agent='$AGENT', port=$PORT, max_test_episode=1)
print(f'submit() returned after {time.perf_counter() - t0:.1f}s')
"
RC=$?
echo "exit=$RC  wall=$(( $(date +%s) - START ))s  output_dir=$OUT"
ls -la "$OUT" 2>/dev/null | head -30
echo "== build log tail:"; tail -5 "$ROOT/logs/build_${STAMP}.log"
echo "== Player.log errors:"; grep -i -E "error|exception" "/c/Users/Chinm_user/AppData/LocalLow/MIT/TDW/Player.log" 2>/dev/null | head -5
taskkill //F //IM TDW.exe >/dev/null 2>&1 || true
echo "== done"
