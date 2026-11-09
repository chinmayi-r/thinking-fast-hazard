#!/usr/bin/env bash
# Fill HAZARD's local asset cache for the fire test set, on della-vis1 (internet). Compute nodes have none (NOTES Nov 8).
# How HAZARD's cache works (envs/fire/fireagent_controller.py:387-423): with use_cached_assets, every S3 url is rewritten
# to file://<cwd>/data/assets/<basename>; if a file is missing HAZARD writes data/assets/download_assets.sh (wget lines),
# terminates the build and exits 0. So: 1) pre-fetch every url named in the 25 scenes' log.txt, 2) run one scene in cached
# mode a few times to catch the add-on assets (replicant, fireman, obi) until it stops asking.
# Usage (on della-vis1, after della_setup.sh):  cd /home/cr7998/iw-thinking-fast && bash scripts/della_cache_assets.sh 2>&1 | tee logs/della_cache.log
set -uo pipefail
ROOT=/home/cr7998/iw-thinking-fast
ENV_NAME="${ENV_NAME:-fire}"
PY="$ROOT/py310/bin/python"
DATA="$ROOT/HAZARD/src/HAZARD/data"
ASSETS="$DATA/assets"
TEST="$DATA/room_setup_${ENV_NAME/flood/fire}/test_set"
mkdir -p "$ASSETS" "$ROOT/logs"
export PYTHONPATH="$ROOT/HAZARD:$ROOT/HAZARD/src/HAZARD" TMPDIR="$ROOT/.cache/tmp"

echo "== [1/2] urls from every test scene's log.txt (linux platform), fetched flat into data/assets"
grep -ho '"url": "[^"]*amazonaws[^"]*"' "$TEST"/*/log.txt | cut -d'"' -f4 | sed -E 's#/(windows|osx)/#/linux/#' | sort -u > "$ROOT/logs/asset_urls_${ENV_NAME}.txt"
echo "$(wc -l < "$ROOT/logs/asset_urls_${ENV_NAME}.txt") unique urls"
cd "$ASSETS" && wget -q -nc -i "$ROOT/logs/asset_urls_${ENV_NAME}.txt" 2>&1 | tail -3
echo "$(ls "$ASSETS" | wc -l) files in data/assets, $(du -sh "$ASSETS" | cut -f1)"

echo "== [2/2] one scene in cached mode until HAZARD stops writing download_assets.sh (add-on assets)"
SCENE="$(ls "$TEST" | sort | head -1)"
pkill -u "$USER" -f TDW.x86_64 2>/dev/null; pkill -u "$USER" Xvfb 2>/dev/null; sleep 1
Xvfb :98 -screen 0 1280x720x24 >/dev/null 2>&1 & XVFB=$!; sleep 2; export DISPLAY=:98
cd "$ROOT/HAZARD"
for i in 1 2 3 4 5; do
  rm -f "$ASSETS/download_assets.sh"
  OUT="$ROOT/results/cache_probe_${ENV_NAME}_$i"; rm -rf "$OUT"
  # rule agent, no clock, 15-minute cap, output kept in logs/cache_probe_$i.log (Nov 8 lesson: a silent probe passed a run
  # that hung on an unreadable bundle). The pass counts only if the episode finished (eval_result.json).
  OMP_NUM_THREADS=4 timeout 900 "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 --env "$ENV_NAME" --agent rule --episodes 1       --build "$ROOT/tdw_build/TDW/TDW.x86_64" --launch "vglrun -d egl" --out "$OUT" --data_dir "$TEST/$SCENE" --cached_assets       --build_args "-logFile $ROOT/logs/unity_cache_probe_$i.log" > "$ROOT/logs/cache_probe_$i.log" 2>&1
  pkill -u "$USER" -f TDW.x86_64 2>/dev/null
  if [ -s "$ASSETS/download_assets.sh" ]; then
    echo "pass $i: HAZARD asked for $(wc -l < "$ASSETS/download_assets.sh") more files; fetching"
    (cd "$ASSETS" && bash download_assets.sh >/dev/null 2>&1)
  elif ls "$OUT"/eval_result.json "$OUT"/*/eval_result.json >/dev/null 2>&1; then
    echo "pass $i: episode finished from the cache; complete for $SCENE"; break
  else
    echo "pass $i: nothing requested but no eval_result.json either; see logs/cache_probe_$i.log and logs/unity_cache_probe_$i.log"
    grep -i -E "AssetBundle|amazonaws|error|fail" "$ROOT/logs/unity_cache_probe_$i.log" 2>/dev/null | cut -c1-160 | head -8; break
  fi
done
kill $XVFB 2>/dev/null
echo "$(ls "$ASSETS" | wc -l) files, $(du -sh "$ASSETS" | cut -f1) in data/assets"
echo "== done. Next: sbatch --export=ALL,AGENT=rule,ENV_NAME=fire scripts/della_scene_array.slurm"
