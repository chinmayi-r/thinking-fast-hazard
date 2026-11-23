#!/usr/bin/env bash
# Copy the kinder-era work from the working folder into the git clone thinking-fast-hazard/ (CR, Nov 24, decision 5).
# It copies and stages; it does not commit and never pushes (CR pushes). Run from Git Bash:
#   bash /g/iw-thinking-fast/scripts/sync_to_clone.sh
set -euo pipefail
SRC=/g/iw-thinking-fast
DST=$SRC/thinking-fast-hazard
mkdir -p "$DST/firelab" "$DST/scripts" "$DST/results/firelab_gate" "$DST/results/firelab_sim" "$DST/results/slides_nov23"
cp "$SRC"/firelab/*.py "$DST/firelab/"
mkdir -p "$DST/firelab/examples" && cp "$SRC"/firelab/examples/*.py "$DST/firelab/examples/"
cp "$SRC"/PLAN.md "$SRC"/DESIGN.md "$SRC"/NOTES.md "$SRC"/ASSUMPTIONS.md "$DST/"
cp "$SRC"/scripts/kinder_*.py "$SRC"/scripts/firelab_*.py "$SRC"/scripts/first_run_trace.py "$SRC"/scripts/gate_exit_rules.py "$SRC"/scripts/seconds_per_step.py "$SRC"/scripts/run_summary.py "$SRC"/scripts/show_game.py "$SRC"/scripts/model_run_summary.py \
   "$SRC"/scripts/della_firelab_*.s* "$SRC"/scripts/sync_to_clone.sh "$DST/scripts/"
cp "$SRC"/results/firelab_gate/*.csv "$DST/results/firelab_gate/"
cp "$SRC"/results/firelab_sim/*.csv "$DST/results/firelab_sim/"
for run in "$SRC"/results/firelab_sim/*/; do  # each run folder: its rows, not its saved games
  mkdir -p "$DST/results/firelab_sim/$(basename "$run")" && cp "$run"rows_*.csv "$DST/results/firelab_sim/$(basename "$run")/"
done
cp "$SRC"/results/slides_nov23/time_planner.py "$DST/results/slides_nov23/"
git -C "$DST" add firelab scripts PLAN.md DESIGN.md NOTES.md ASSUMPTIONS.md results/firelab_gate results/firelab_sim results/slides_nov23/time_planner.py
git -C "$DST" status --short | head -60
