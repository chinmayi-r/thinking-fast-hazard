#!/usr/bin/env bash
# Oct 27 reorder (CR: "yes, any way to make it shorter?"): after the Haiku llm sweep that v2 is running now,
# take over from v2: redo the shipped-agent scenes without results, record the two demo episodes and encode
# them, THEN the Haiku vlm 25 and Opus 5 sweeps unless logs/SKIP_LAPTOP_VLM exists (set when Adroit runs them).
# Same per-scene runner, same 30-minute cap, same results layout as v2.
set -uo pipefail
ROOT=/g/iw-thinking-fast
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
export ANTHROPIC_WORKSPACE_ID='wrkspc_01VPU7qLAqtHXFpUPSQ8DCgd'
PY="$ROOT/venv/Scripts/python.exe"
KEY="G:\\iw-thinking-fast\\.secrets\\anthropic_key.txt"
LOG="$ROOT/logs/sweep_v2.log"   # one lane log for the whole Oct 26-27 run
TEST="$ROOT/HAZARD/src/HAZARD/data/room_setup_fire/test_set"
mapfile -t SCENES < <(ls "$TEST" | grep -E "kitchen|craftroom" | sort | head -25)
win() { echo "$1" | sed 's#^/g/#G:\\#; s#/#\\#g'; }

# 1. wait for v2's Haiku llm sweep, then stop v2 and its pass-2 watcher so they never start the vlm/opus sweeps
until grep -q "sweep fire_llm_haiku_test25 done" "$LOG" 2>/dev/null; do sleep 15; done
for pid in $(ps -W | grep -E "bash" | awk '{print $1}'); do
  if cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | grep -qE "laptop_sweep_v2|SWEEP V2 ALL DONE"; then kill "$pid" 2>/dev/null; fi
done
for pid in $(ps -W | grep -iE "python" | awk '{print $1}'); do
  if cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | grep -qE "run_baseline.py.*(agent vlm|opus)"; then kill "$pid" 2>/dev/null; fi
done
sleep 5; taskkill //F //IM TDW.exe >/dev/null 2>&1
echo "===== $(date +%H:%M:%S) v3 takes over: v2 driver and pass-2 watcher stopped =====" | tee -a "$LOG"

run_scene() {  # run_scene <sweep tag> <scene> <args...>
  local tag="$1"; local scene="$2"; shift 2
  local dir="$ROOT/results/sweep_${tag}"; mkdir -p "$dir"
  if [ -f "$dir/$scene/eval_result.json" ]; then return 0; fi
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 10; done
  echo "===== $(date +%H:%M:%S) start $tag $scene =====" | tee -a "$LOG"
  ( cd "$ROOT/HAZARD" && timeout 30m "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 --episodes 1 \
      --resume_dir "$(win "$dir")" --data_dir "$(win "$TEST/$scene")" "$@" > "$ROOT/logs/sweep_${tag}_${scene}.log" 2>&1 )
  local rc=$?
  echo "===== $(date +%H:%M:%S) end $tag $scene exit=$rc =====" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
}
sweep() { local tag="$1"; local n="$2"; shift 2; for scene in "${SCENES[@]:0:$n}"; do run_scene "$tag" "$scene" "$@"; done; echo "----- sweep $tag done $(date +%H:%M:%S) -----" | tee -a "$LOG"; }

# 2. redo shipped-agent scenes without results (one retry each; a second failure stays missing)
sweep fire_rule_test25 25 --env fire --agent rule
sweep fire_greedy_test25 25 --env fire --agent greedy
sweep fire_random_test25 25 --env fire --agent random
sweep fire_llm_haiku_test25 25 --env fire --agent llm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file "$KEY"

# 3. demo videos on the first test scene
DEMO="$(win "$TEST/${SCENES[0]}")"
for spec in "mcts --agent mcts" "vlm_haiku --agent vlm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file $KEY"; do
  tag="${spec%% *}"; args="${spec#* }"
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 10; done
  echo "===== $(date +%H:%M:%S) start video_$tag =====" | tee -a "$LOG"
  ( cd "$ROOT/HAZARD" && timeout 40m "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 --env fire --episodes 1 --record --data_dir "$DEMO" $args > "$ROOT/logs/video_${tag}.log" 2>&1 )
  echo "===== $(date +%H:%M:%S) end video_$tag exit=$? =====" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
done
"$PY" "$ROOT/scripts/make_videos.py" 2>&1 | tee -a "$LOG"
echo "VIDEOS DONE $(date +%H:%M:%S)" | tee -a "$LOG"

# 4. the remaining Claude sweeps, unless Adroit took them
if [ -f "$ROOT/logs/SKIP_LAPTOP_VLM" ]; then
  echo "SKIP_LAPTOP_VLM present: vlm/opus sweeps left to Adroit" | tee -a "$LOG"
else
  sweep fire_vlm_haiku_test25 25 --env fire --agent vlm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file "$KEY"
  sweep fire_llm_opus_test5 5 --env fire --agent llm --lm_source anthropic --lm_id claude-opus-5-5 --api_key_file "$KEY"
fi
echo "LANE V3 DONE $(date +%H:%M:%S)" | tee -a "$LOG"
echo "SWEEP V2 ALL DONE (markers for the Oct 26 stay-awake helpers) $(date +%H:%M:%S)" | tee -a "$LOG"; echo "SWEEP V2 PASS2 DONE" | tee -a "$LOG"
