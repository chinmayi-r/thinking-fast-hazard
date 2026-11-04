#!/usr/bin/env bash
# Oct 27 19:05: after the v3 lane ends, redo vlm / Opus scenes that have no eval_result.json (one retry each),
# e.g. 3a-16 (transient Anthropic connection error 18:34) and 3a-17 (hung at environment reset, 30-min cap).
set -uo pipefail
ROOT=/g/iw-thinking-fast
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
export ANTHROPIC_WORKSPACE_ID='wrkspc_01VPU7qLAqtHXFpUPSQ8DCgd'
PY="$ROOT/venv/Scripts/python.exe"
KEY="G:\\iw-thinking-fast\\.secrets\\anthropic_key.txt"
LOG="$ROOT/logs/sweep_v2.log"
TEST="$ROOT/HAZARD/src/HAZARD/data/room_setup_fire/test_set"
mapfile -t SCENES < <(ls "$TEST" | grep -E "kitchen|craftroom" | sort | head -25)
win() { echo "$1" | sed 's#^/g/#G:\\#; s#/#\\#g'; }
until grep -q "LANE V3 DONE" "$LOG" 2>/dev/null; do sleep 30; done
sleep 10; taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "===== $(date +%H:%M:%S) v3 pass 2: redo vlm/opus scenes without results =====" | tee -a "$LOG"
run_scene() {
  local tag="$1"; local scene="$2"; shift 2
  local dir="$ROOT/results/sweep_${tag}"; mkdir -p "$dir"
  if [ -f "$dir/$scene/eval_result.json" ]; then return 0; fi
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 10; done
  echo "===== $(date +%H:%M:%S) start $tag $scene (pass 2) =====" | tee -a "$LOG"
  ( cd "$ROOT/HAZARD" && timeout 30m "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 --episodes 1 \
      --resume_dir "$(win "$dir")" --data_dir "$(win "$TEST/$scene")" "$@" > "$ROOT/logs/sweep_${tag}_${scene}_pass2.log" 2>&1 )
  local rc=$?
  echo "===== $(date +%H:%M:%S) end $tag $scene exit=$rc (pass 2) =====" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
}
for scene in "${SCENES[@]}"; do run_scene fire_vlm_haiku_test25 "$scene" --env fire --agent vlm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file "$KEY"; done
for scene in "${SCENES[@]:0:5}"; do run_scene fire_llm_opus_test5 "$scene" --env fire --agent llm --lm_source anthropic --lm_id claude-opus-5-5 --api_key_file "$KEY"; done
echo "LANE V3 PASS2 DONE $(date +%H:%M:%S)" | tee -a "$LOG"
echo "SWEEP V2 ALL DONE (release the Oct 26 stay-awake helper) $(date +%H:%M:%S)" | tee -a "$LOG"
