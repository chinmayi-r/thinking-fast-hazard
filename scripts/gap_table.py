"""The gap table (Nov 23): every simulator run on rooms 0-49 against the reflex-like agent that has the same commands.

Reads rows files only; plays nothing. Two gaps per row. "To the reflex" is the one that counts since CR's decision
of Nov 23 about 12:00 (A98): the reflex is the baseline and stays out of fire; smarter agents may walk through it.
"To the same-commands agent" is the earlier like-for-like reading, kept for the record: the reflex for a row without
fire, Jev before every command with the fire menu (jev_run50) for a row with fire.
Run: venv_kinder\\Scripts\\python.exe scripts\\gap_table.py   (PYTHONPATH = repo root)
"""
import csv
import glob
from pathlib import Path

import numpy as np

from firelab.gate import paired_interval

SIM = Path(__file__).resolve().parents[1] / "results" / "firelab_sim"


def load(tag: str, agent: str, key: str = "sim") -> dict:
    files = [SIM / tag] if tag.endswith(".csv") else sorted(glob.glob(str(SIM / tag / "rows_*.csv")))
    return {int(r["room"]): float(r[key]) for p in files for r in csv.DictReader(open(p, newline="")) if r["agent"] == agent}


reflex, jev_fire = load("first_run.csv", "reflex"), load("jev_run50", "model")
ROWS = [  # label, tag, agent, baseline
    ("planner, any bag, no fire", "run1_off_firefree", "always", reflex),
    ("planner, bank after every pick, no fire", "run5_guard1_nofire", "always", reflex),
    ("planner, bank after two, no fire", "run5_guard2_nofire", "always", reflex),
    ("Jev before every command, no fire", "jev_run50_firefree", "model", reflex),
    ("RRARA v1 Jev, strict, no fire", "jev_run50_firefree", "rrara1", reflex),
    ("RRARA v1 Jev, free, no fire", "jev_free_nofire", "rrara1", reflex),
    ("cautious 1.5, any bag, no fire", "run6_c15_any_nofire", "always", reflex),
    ("cautious 1.5, bank after two, no fire", "run6_c15_bag2_nofire", "always", reflex),
    ("cautious 2.0, bank after two, no fire", "run6_c20_bag2_nofire", "always", reflex),
    ("planner, any bag, fire", "run1_off", "always", jev_fire),
    ("planner, bank after every pick, fire", "run5_guard1_fire", "always", jev_fire),
    ("planner, bank after two, fire", "run5_guard2_fire", "always", jev_fire),
    ("cautious 1.5, any bag, fire", "run6_c15_any_fire", "always", jev_fire),
    ("cautious 1.5, bank after two, fire", "run6_c15_bag2_fire", "always", jev_fire),
    ("cautious 2.0, bank after two, fire", "run6_c20_bag2_fire", "always", jev_fire),
    ("RRARA v1 Jev, strict, fire", "jev_run50", "rrara1", jev_fire),
    ("RRARA v1 Jev, free, fire", "jev_free_fire", "rrara1", jev_fire),
    ("Opus before every command, fire", "opus_run50_fire", "model", jev_fire),  # Della, declared 0.15 s per call (A100)
    ("RRARA v1 Opus, free, fire", "opus_run50_fire", "rrara1", jev_fire),
    ("Opus before every command, real time, fire", "opus_run50_fire_measured", "model", jev_fire),  # Della (A101)
]
print(f"reflex (no fire) {np.mean(list(reflex.values())):.3f}; Jev before every command with fire routes {np.mean(list(jev_fire.values())):.3f}")
for label, tag, agent, base in ROWS:
    mine = load(tag, agent)
    rooms = sorted(set(mine) & set(base))
    if len(rooms) < 50:
        print(f"{label:42s} only {len(rooms)} rooms yet")
        continue
    a, b, r = (np.array([d[k] for k in rooms]) for d in (mine, base, reflex))
    lo, hi = paired_interval(a - b)
    rlo, rhi = paired_interval(a - r)
    print(f"{label:42s} {a.mean():.3f}  to the reflex {np.mean(a - r):+.3f} [{rlo:+.3f}, {rhi:+.3f}]  |  to the same-commands agent {np.mean(a - b):+.3f} [{lo:+.3f}, {hi:+.3f}]  ahead/level/behind {int((a - b > 0.005).sum())}/"
          f"{int((abs(a - b) <= 0.005).sum())}/{int((a - b < -0.005).sum())}  zeros {int((a == 0).sum())}  behind the reflex in {int((a - r < -0.005).sum())}")
