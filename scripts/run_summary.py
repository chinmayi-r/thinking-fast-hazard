"""Summary of a planner run against the reflex and the clock-off planner, room by room (rooms 0-49, Nov 24).

Reads results/firelab_sim/<tag>/rows_*.csv, the first run (reflex, random) and run1_off. Plays nothing.
Run: venv_kinder\\Scripts\\python.exe scripts\\run_summary.py run2_declared   (PYTHONPATH = repo root)
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

from firelab.gate import paired_interval

SIM = Path(__file__).resolve().parents[1] / "results" / "firelab_sim"
read = lambda pattern: [r for p in sorted(glob.glob(str(SIM / pattern))) for r in csv.DictReader(open(p, newline=""))]
by_room = lambda rows, agent: {int(r["room"]): r for r in rows if r["agent"] == agent}
tag = sys.argv[1]
rows = read(f"{tag}/rows_*.csv")
reflex, off = by_room(read("first_run.csv"), "reflex"), by_room(read("run1_off/rows_*.csv"), "always")
for name in ("always", "commit", "moving"):
    mine = by_room(rows, name)
    rooms = sorted(set(mine) & set(reflex))
    if not rooms:
        continue
    col = lambda src, key: np.array([float(src[k][key]) for k in rooms])
    sim, base, free = col(mine, "sim"), col(reflex, "sim"), np.array([float(off[k]["sim"]) for k in rooms if k in off])
    lo, hi = paired_interval(sim - base)
    line = (f"{name:6s} ({mine[rooms[0]]['clock']}): {len(rooms)} rooms, simulator {sim.mean():.3f}, reflex {base.mean():.3f}, "
            f"gap {np.mean(sim - base):+.3f} [{lo:+.3f}, {hi:+.3f}], ahead/level/behind {int((sim - base > 0.005).sum())}/"
            f"{int((abs(sim - base) <= 0.005).sum())}/{int((sim - base < -0.005).sum())}, zeros {int((sim == 0).sum())}, deaths {int((col(mine, 'health') <= 0).sum())}")
    if len(free) == len(rooms):
        line += f"; against the clock-off planner {np.mean(sim - free):+.3f}"
    if all(mine[k]["rule"] != "" for k in rooms):
        rule = col(mine, "rule")
        line += f"; rule check {rule.mean():.3f} (rooms differing by more than 0.01: {int((abs(sim - rule) > 0.01).sum())})"
    print(line)
    print(f"        per episode: thoughts {col(mine, 'thoughts').mean():.1f}, search steps {col(mine, 'steps').mean():.0f}, "
          f"declared {col(mine, 'declared_s').mean():.2f} s, measured {col(mine, 'measured_s').mean():.2f} s, "
          f"charged {col(mine, 'charged_ticks').mean():.1f} ticks (most {col(mine, 'charged_ticks').max():.0f}), "
          f"estimate off by {np.mean([float(mine[k]['estimate_error']) for k in rooms if mine[k]['estimate_error'] != '']):.1f} ticks per thought, "
          f"episode {col(mine, 'sim_ticks').mean():.0f} ticks")
