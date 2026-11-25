"""The like-for-like check on the rule check's board (Nov 23): planner without walking through fire against the reflex.

No simulator. For rooms 0-49 of the main rule set (8 items, values 1 2 3 5 8) it plays, on firelab/gate.py's board:
the reflex (gate.reflex), and our planner with the clock off (firelab.thinkers.thinker on GateSkills) in eight
settings: any bag or a guarded bag (1 or 2 items), and any bag with a cautious forecast (fire 1.5 times faster),
each with and without walking through fire.
gate.py and the old result files are not touched; rows go to a NEW file, firelab/results/firelab_gate/nofire_check_nov23.csv.
Run: venv_kinder\\Scripts\\python.exe scripts\\gate_nofire_check.py   (PYTHONPATH = repo root; a few minutes)
"""
import csv
from pathlib import Path

import numpy as np

from firelab import gate
from firelab.agents import GateSkills
from firelab.clock import Clock
from firelab.thinkers import thinker

MAIN = gate.Task(items=8, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, pick_ticks=10, hp_loss=2.0)
SETTINGS = {f"planner_{bag or 'any'}bag_{'fire' if fire else 'nofire'}": {"max_bag": bag, "allow_fire": fire}
            for bag in (None, 1, 2) for fire in (True, False)}
SETTINGS.update({f"cautious15_anybag_{'fire' if fire else 'nofire'}": {"caution": 1.5, "allow_fire": fire} for fire in (True, False)})
OUT = Path(__file__).resolve().parents[1] / "firelab" / "results" / "firelab_gate" / "nofire_check_nov23.csv"

rows = []
for room in range(50):
    world = gate.make_world(MAIN, room)
    row = {"room": room, "reflex": round(gate.reflex(world), 4)}
    for name, setting in SETTINGS.items():
        board = GateSkills(world)
        thinker(board, Clock("off"), **setting)
        row[name] = round(board.score(), 4)
    rows.append(row)
    print(room, row, flush=True)
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
reflex = np.array([r["reflex"] for r in rows])
print(f"\nrule check's board, 50 rooms: reflex {reflex.mean():.3f}")
for name in SETTINGS:
    score = np.array([r[name] for r in rows])
    lo, hi = gate.paired_interval(score - reflex)
    print(f"{name:26s} {score.mean():.3f}  gap to the reflex {np.mean(score - reflex):+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"ahead/level/behind {int((score - reflex > 0.005).sum())}/{int((abs(score - reflex) <= 0.005).sum())}/{int((score - reflex < -0.005).sum())}  zeros {int((score == 0).sum())}")
print("wrote", OUT)
