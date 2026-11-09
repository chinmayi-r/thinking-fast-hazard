#!/usr/bin/env bash
# Run ONE fire test scene in cached-asset mode on della-vis1 with everything visible: HAZARD's output, the build's
# Unity log (own file via -logFile), and whether an eval_result.json came out. Replaces the silent probe in
# della_cache_assets.sh, which only looked for download_assets.sh and so passed a run that hung on
# "Failed to read data for the AssetBundle 'fire'" (array 17136282, Nov 8).
# Usage (della-vis1):  cd ~/iw-thinking-fast && bash scripts/della_cache_check.sh 2>&1 | tee logs/della_cache_check.log
set -uo pipefail
ROOT=/home/cr7998/iw-thinking-fast
ENV_NAME="${ENV_NAME:-fire}"; SCENE="${SCENE:-mm_kitchen_3a-1}"; CAP="${CAP:-900}"
PY="$ROOT/py310/bin/python"; DATA="$ROOT/HAZARD/src/HAZARD/data"; ASSETS="$DATA/assets"
TEST="$DATA/room_setup_${ENV_NAME/flood/fire}/test_set"
export PYTHONPATH="$ROOT/HAZARD:$ROOT/HAZARD/src/HAZARD" TMPDIR="$ROOT/.cache/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "== assets dir: $(ls "$ASSETS" | wc -l) files; newest 4:"; ls -lt "$ASSETS" | head -5
echo "== the add-on bundles HAZARD needs besides the scene's own (fire effect, replicant, fireman):"
for f in fire replicant_0 fireman download_assets.sh; do printf "%-20s " "$f"; ls -la "$ASSETS/$f" 2>&1 | awk '{print $5, $6, $7, $8}'; done
echo "== run $SCENE cached, ${CAP}s cap, unity log -> logs/unity_cache_check.log"
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; pkill -u "$USER" Xvfb 2>/dev/null; sleep 1
Xvfb :98 -screen 0 1280x720x24 >/dev/null 2>&1 & XVFB=$!; sleep 2; export DISPLAY=:98
rm -f "$ASSETS/download_assets.sh" "$ROOT/logs/unity_cache_check.log"; OUT="$ROOT/results/cache_check_${ENV_NAME}"; rm -rf "$OUT"
cd "$ROOT/HAZARD"
timeout "$CAP" "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 --env "$ENV_NAME" --agent rule --episodes 1 \
    --build "$ROOT/tdw_build/TDW/TDW.x86_64" --launch "vglrun -d egl" --out "$OUT" --data_dir "$TEST/$SCENE" --cached_assets \
    --build_args "-logFile $ROOT/logs/unity_cache_check.log" 2>&1 | grep -v -E "UserWarning|pkg_resources|Gym|gymnasium|logger.warn|migration" | tail -25
RC=${PIPESTATUS[0]}; echo "runner exit=$RC (124 = hit the cap)"
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; kill $XVFB 2>/dev/null
echo "== unity log, asset/url/error lines:"; grep -i -E "AssetBundle|amazonaws|file://|error|fail|Renderer" "$ROOT/logs/unity_cache_check.log" 2>/dev/null | cut -c1-200 | head -20
[ -s "$ASSETS/download_assets.sh" ] && { echo "== HAZARD asked for:"; cat "$ASSETS/download_assets.sh"; }
if ls "$OUT"/eval_result.json "$OUT"/*/eval_result.json >/dev/null 2>&1; then echo "== CACHE OK: eval_result.json written"; else echo "== CACHE NOT OK: no eval_result.json"; fi
