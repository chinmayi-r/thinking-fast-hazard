#!/usr/bin/env bash
# Oct 27 18:2x: the laptop vlm sweep stops after mm_kitchen_3a-19; adroit-vis runs the other 14 scenes and Opus.
LOG=/g/iw-thinking-fast/logs/sweep_v2.log
until grep -q "end fire_vlm_haiku_test25 mm_kitchen_3a-19" "$LOG" 2>/dev/null; do sleep 10; done
for pid in $(ps -W | awk '{print $1}'); do
  if cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | grep -qE "laptop_lane_v3|run_baseline.py.*vlm_haiku"; then kill "$pid" 2>/dev/null; fi
done
sleep 3; taskkill //F //IM TDW.exe >/dev/null 2>&1
rm -rf /g/iw-thinking-fast/results/sweep_fire_vlm_haiku_test25/mm_kitchen_3a-2   # partial, if 3a-2 had started; adroit runs it
echo "===== $(date +%H:%M:%S) laptop lane stopped after vlm 3a-19; remaining vlm scenes and Opus on adroit-vis =====" | tee -a "$LOG"
echo "LANE V3 DONE $(date +%H:%M:%S) (stopped by laptop_stop_after_3a19.sh)" | tee -a "$LOG"
echo "SWEEP V2 PASS2 DONE" | tee -a "$LOG"
