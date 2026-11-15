#!/usr/bin/env bash
# Human baseline for HAZARD, file 4 of 4 (ASSUMPTIONS.md A36): one sitting = one person plays five fire kitchens, one
# HAZARD process per kitchen, through the page (scripts/human/page.html). Protocol: scripts/human/PROTOCOL.md.
#
# Usage (Git Bash, from G:\iw-thinking-fast):
#   CAMERA=off scripts/human/play.sh cr_text            # text-only sitting: exactly what the Haiku agent gets
#   CAMERA=on  scripts/human/play.sh cr_cam             # same five kitchens with the robot's camera on the page
#   scripts/human/play.sh cr_text mm_kitchen_3a-2       # other kitchens, by name (HAZARD data/room_setup_fire/test_set)
#   TABLE_ONLY=1 scripts/human/play.sh cr_text          # just print the sitting's table against the laptop sweeps
#   DRY_RUN=1 ...                                       # print the commands, run nothing
# Env: CAMERA on|off (on), PORT (8765), CLOCK none|tcm (none: the person's seconds are not charged, CR Nov 12),
#      HOLD seconds the score stays on the page (20), PY, BUILD, RESULTS (results/).
# Output: $RESULTS/human_<sitting>/<kitchen>/ with summary.json (block "human"), decisions.jsonl, 0/human_log.json,
#         0/<frame>_human.json, 0/<frame>_human_frame.png (camera on). A kitchen with eval_result.json is skipped, so
#         a sitting interrupted mid-kitchen resumes at that kitchen when the same command is run again.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SITTING="${1:?usage: play.sh <sitting> [kitchen ...]}"; shift || true
KITCHENS=("$@")
if [ ${#KITCHENS[@]} -eq 0 ]; then  # the first five of HAZARD's sorted test set (run_experiments.py:176), as --episodes 5 would take
  KITCHENS=(mm_kitchen_3a-1 mm_kitchen_3a-10 mm_kitchen_3a-11 mm_kitchen_3a-12 mm_kitchen_3a-13)
fi
CAMERA="${CAMERA:-on}"; PORT="${PORT:-8765}"; CLOCK="${CLOCK:-none}"; HOLD="${HOLD:-20}"
PY="${PY:-$ROOT/venv/Scripts/python.exe}"
BUILD="${BUILD:-$ROOT/tdw_build/TDW/TDW.exe}"
RESULTS="${RESULTS:-$ROOT/results}"
DRY_RUN="${DRY_RUN:-0}"; TABLE_ONLY="${TABLE_ONLY:-0}"

w() { cygpath -w "$1" 2>/dev/null || echo "$1"; }  # Windows Python wants G:\... paths, not /g/...
WROOT="$(w "$ROOT")"
export PYTHONPATH="$WROOT\\HAZARD;$WROOT\\HAZARD\\src\\HAZARD" TMP="$WROOT\\.cache\\tmp" TEMP="$WROOT\\.cache\\tmp"
OUT="$RESULTS/human_${SITTING}"
mkdir -p "$OUT" "$ROOT/logs" "$ROOT/.cache/tmp"

table() {  # the sitting's kitchens next to the laptop sweeps on the same kitchens (d35c71e, clock off; results/sweep_fire_*)
  "$PY" -X utf8 - "$OUT" "$RESULTS" "${KITCHENS[@]}" <<'EOF'
import json, os, sys
out, results, kitchens = sys.argv[1], sys.argv[2], sys.argv[3:]
def rate(p):
    try:
        e = json.load(open(p)); return e["avg_score"] / e["avg_max_score"]
    except Exception:
        return None
def fmt(x): return "  -  " if x is None else f"{x:.3f}"
rows = [("human " + os.path.basename(out)[6:], [rate(os.path.join(out, k, "eval_result.json")) for k in kitchens])]
for sw, label in [("sweep_fire_random_test25", "random"), ("sweep_fire_rule_test25", "rule"), ("sweep_fire_greedy_test25", "greedy"),
                  ("sweep_fire_mcts_test25", "mcts"), ("sweep_fire_llm_haiku_test25", "Haiku llm (text)"),
                  ("sweep_fire_vlm_haiku_test25", "Haiku vlm (camera)")]:
    d = os.path.join(results, sw)
    if os.path.isdir(d):
        rows.append((label, [rate(os.path.join(d, k, "eval_result.json")) for k in kitchens]))
print(f"{'agent':22s} " + " ".join(f"{k[11:]:>6s}" for k in kitchens) + "   mean (played)")
for label, r in rows:
    have = [x for x in r if x is not None]
    mean = f"{sum(have)/len(have):.3f} ({len(have)})" if have else "-"
    print(f"{label:22s} " + " ".join(f"{fmt(x):>6s}" for x in r) + f"   {mean}")
print()
for k in kitchens:  # the person's side of it
    p = os.path.join(out, k, "summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p)); h = s.get("human", {})
    print(f"{k}: score {h.get('score')}, decisions {h.get('decisions')} ({h.get('asked')} asked, {h.get('automatic')} automatic), "
          f"think {h.get('wall_s_mean')} s mean / {h.get('wall_s_max')} s max, frames the thinking would cost at 30 fps: "
          f"{s.get('frames_at_30fps_total')}, episode frames {s.get('final_frame_count')}, wall {s.get('wall_s_total')} s")
EOF
}

if [ "$TABLE_ONLY" = 1 ]; then table; exit 0; fi

CAM_FLAG=(); [ "$CAMERA" = off ] && CAM_FLAG=(--no_camera)
first=1
for k in "${KITCHENS[@]}"; do
  if [ -f "$OUT/$k/eval_result.json" ]; then echo "== $k already played, skipping"; continue; fi
  BROWSER_FLAG=(); [ $first -eq 0 ] && BROWSER_FLAG=(--no_browser)  # the first kitchen opens the tab; it reconnects for the rest
  CMD=("$PY" -X utf8 -u "$(w "$ROOT/scripts/run_baseline.py")" --env fire --agent human --episodes 1 --port 10826
       --build "$(w "$BUILD")" --out "$(w "$RESULTS")" --resume_dir "$(w "$OUT")"
       --data_dir "$WROOT\\HAZARD\\src\\HAZARD\\data\\room_setup_fire\\test_set\\$k"
       --clock "$CLOCK" --human_port "$PORT" --human_hold "$HOLD" "${CAM_FLAG[@]}" "${BROWSER_FLAG[@]}")
  first=0
  if [ "$DRY_RUN" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; continue; fi
  echo "== $(date +%H:%M:%S) $k (camera $CAMERA, clock $CLOCK): page at http://127.0.0.1:$PORT/ once the kitchen has loaded"
  while tasklist 2>/dev/null | grep -q -i "TDW.exe"; do sleep 5; done
  ( cd "$ROOT/HAZARD" && "${CMD[@]}" 2>&1 | grep --line-buffered -v "^current_step\|^Target status\|FireObjectState" \
      | tee "$ROOT/logs/human_${SITTING}_${k}.log" | grep --line-buffered "human page\|Traceback\|Error\|avg_score" )
  taskkill //F //IM TDW.exe >/dev/null 2>&1; sleep 3
  [ -f "$OUT/$k/eval_result.json" ] && echo "== $k done" || echo "== $k did not finish (no eval_result.json); rerun the same command to replay it"
done
[ "$DRY_RUN" = 1 ] || table
