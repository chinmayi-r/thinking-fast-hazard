#!/usr/bin/env bash
# Second lane (CR approved Oct 26 11:0x): after the shipped-agent fire sweep, run Claude Haiku 5.5 on 25 fire scenes
# as llm and as vlm, then Claude Opus 5.5 on 5 scenes as llm, then two recorded demo episodes and ffmpeg videos.
set -uo pipefail
ROOT=/g/iw-thinking-fast
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
export ANTHROPIC_WORKSPACE_ID='wrkspc_01VPU7qLAqtHXFpUPSQ8DCgd'
PY="$ROOT/venv/Scripts/python.exe"
KEY="G:\\iw-thinking-fast\\.secrets\\anthropic_key.txt"
LOG="$ROOT/logs/llm_lane.log"
SCENE="G:\\iw-thinking-fast\\HAZARD\\src\\HAZARD\\data\\room_setup_fire\\mm_craftroom_2a-1"
run() {  # run <tag> <args...>
  local tag="$1"; shift
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 10; done
  echo "===== $(date +%H:%M:%S) start $tag =====" | tee -a "$LOG"
  ( cd "$ROOT/HAZARD" && "$PY" -X utf8 "$ROOT/scripts/run_baseline.py" --port 10826 "$@" > "$ROOT/logs/baseline_${tag}.log" 2>&1 )
  local rc=$?
  echo "===== $(date +%H:%M:%S) end $tag exit=$rc =====" | tee -a "$LOG"
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 5
  return $rc
}
sweep() {  # sweep <tag> <episodes> <args...>: resumable, up to 3 attempts
  local tag="$1"; local eps="$2"; shift 2
  local dir="$ROOT/results/sweep_${tag}_ep${eps}"; mkdir -p "$dir"
  for attempt in 1 2 3; do
    run ${tag}_ep${eps}_attempt${attempt} --episodes "$eps" --resume_dir "$dir" "$@" && break
  done
}
until grep -q "FIRE SWEEP DONE" "$ROOT/logs/fire_sweep.log" 2>/dev/null; do sleep 120; done
echo "shipped-agent sweep done; starting LLM lane $(date +%H:%M:%S)" | tee -a "$LOG"
sweep fire_llm_haiku 25 --env fire --agent llm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file "$KEY"
sweep fire_vlm_haiku 25 --env fire --agent vlm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file "$KEY"
sweep fire_llm_opus 5 --env fire --agent llm --lm_source anthropic --lm_id claude-opus-5-5 --api_key_file "$KEY"
# demo videos: one mcts and one Haiku vlm episode on the first scene, every frame captured
run video_fire_mcts --env fire --agent mcts --episodes 1 --record --data_dir "$SCENE"
run video_fire_vlm_haiku --env fire --agent vlm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file "$KEY" --episodes 1 --record --data_dir "$SCENE"
"$PY" "$ROOT/scripts/make_videos.py" 2>&1 | tee -a "$LOG"
echo "LLM LANE DONE $(date +%H:%M:%S)" | tee -a "$LOG"
