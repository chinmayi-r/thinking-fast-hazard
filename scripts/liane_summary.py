"""Liane's method on the fire task: the test file of the kept program beside the other agents on rooms 50-99 (A99).

Reads files only; plays nothing. The test file is what firelab/validate.py writes with --first 50 --rooms 50.
For every other agent that has played rooms 50-99 (the held_* runs) it prints that agent's score and the program's
score minus it, room by room, with a 95 % interval.
Run: venv_kinder\\Scripts\\python.exe scripts\\liane_summary.py results\\firelab_compiled\\v1\\test_rooms_50_99.json
     (PYTHONPATH = repo root)
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

from firelab.gate import paired_interval

SIM = Path(__file__).resolve().parents[1] / "results" / "firelab_sim"
OTHERS = [  # label, run folder, agent
    ("reflex", "held_reflex", "reflex"),
    ("random", "held_reflex", "random"),
    ("Jev before every command", "held_jev_fire", "model"),
    ("planner, clock off", "held_plain_fire", "always"),
    ("cautious planner, clock off", "held_c15_fire", "always"),
]


def held(tag: str, agent: str) -> dict:
    return {int(r["room"]): float(r["sim"]) for p in sorted(glob.glob(str(SIM / tag / "rows_*.csv")))
            for r in csv.DictReader(open(p, newline="")) if r["agent"] == agent}


def main(path: str) -> int:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    mine = {int(k): v["score"] for k, v in record["rooms"].items()}
    print(f"program {record['program']}: {record['n']} rooms, score {record['mean']:.3f}, deaths {record['deaths']}, "
          f"illegal commands {record['illegal']}, errors {record['raised']}, {1000 * record['seconds_per_decision']:.3f} ms per decision "
          f"(slowest {1000 * record['slowest_decision']:.2f} ms), zeros {sum(v == 0 for v in mine.values())}")
    for label, tag, agent in OTHERS:
        other = held(tag, agent)
        rooms = sorted(set(mine) & set(other))
        if not rooms:
            print(f"{label:30s} no rooms in common")
            continue
        a, b = (np.array([d[k] for k in rooms]) for d in (mine, other))
        lo, hi = paired_interval(a - b)
        print(f"{label:30s} {b.mean():.3f}   program minus it {np.mean(a - b):+.3f} [{lo:+.3f}, {hi:+.3f}]   program ahead/level/behind "
              f"{int((a - b > 0.005).sum())}/{int((abs(a - b) <= 0.005).sum())}/{int((a - b < -0.005).sum())}   ({len(rooms)} rooms)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
