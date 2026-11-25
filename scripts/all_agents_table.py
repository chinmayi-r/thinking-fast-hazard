"""One table of every agent on rooms 0-49, 8 items (Nov 23): score, gap to the reflex, ticks spent standing to think.

Reads rows files and the program's validation file only; plays nothing. Fire-free model rows are left out (A98).
Run: venv_kinder\Scripts\python.exe scripts\all_agents_table.py   (PYTHONPATH = repo root)
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

from firelab.gate import paired_interval

ROOT = Path(__file__).resolve().parents[1]
SIM, PROGRAM = ROOT / "firelab" / "results" / "firelab_sim", ROOT / "firelab" / "results" / "firelab_compiled" / "v1" / "rooms_0_49_laptop.json"
ROWS = [  # label, run, agent
    ("random", "first_run.csv", "random"),
    ("reflex", "first_run.csv", "reflex"),
    ("planner before every command, clock off", "run1_off", "always"),
    ("planner before every command, declared clock", "run2_declared", "always"),
    ("planner, plan once and follow, declared clock", "run2_declared", "commit"),
    ("reflex acts while the planner plans, declared clock", "run2_declared", "moving"),
    ("cautious planner, clock off", "run6_c15_any_fire", "always"),
    ("Jev before every command, real time", "jev_run50", "model"),
    ("RRARA v1, Jev checking, strict", "jev_run50", "rrara1"),
    ("RRARA v1, Jev checking, free", "jev_free_fire", "rrara1"),
    ("Opus before every command, 0.15 s declared", "opus_run50_fire", "model"),
    ("Opus before every command, real time", "opus_run50_fire_measured", "model"),
    ("RRARA v1, Opus checking, free", "opus_run50_fire", "rrara1"),
]


def load(tag: str, agent: str) -> dict:
    files = [SIM / tag] if tag.endswith(".csv") else sorted(glob.glob(str(SIM / tag / "rows_*.csv")))
    return {int(r["room"]): r for p in files for r in csv.DictReader(open(p, newline="")) if r["agent"] == agent and int(r["room"]) < 50}


def line(label: str, scores, base, standing, deaths) -> None:
    gap = scores - base
    lo, hi = paired_interval(gap)
    print(f"{label:52s} {scores.mean():.3f}  {gap.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  ahead/level/behind {int((gap > 0.005).sum())}/"
          f"{int((abs(gap) <= 0.005).sum())}/{int((gap < -0.005).sum())}  zeros {int((scores == 0).sum())}  standing {standing}  deaths {deaths}")


def main() -> None:
    reflex = load("first_run.csv", "reflex")
    rooms = range(50)
    base = np.array([float(reflex[k]["sim"]) for k in rooms])
    for label, tag, agent in ROWS:
        rows = load(tag, agent)
        if len(rows) < 50:
            print(f"{label:52s} only {len(rows)} rooms")
            continue
        standing = f"{np.mean([float(rows[k].get('charged_ticks') or 0) for k in rooms]):.0f}"
        line(label, np.array([float(rows[k]["sim"]) for k in rooms]), base, standing, sum(float(rows[k]["health"]) <= 0 for k in rooms))
    if PROGRAM.exists():
        program = json.loads(PROGRAM.read_text(encoding="utf-8"))["rooms"]
        line("Liane's method: the kept program (round 2)", np.array([program[str(k)]["score"] for k in rooms]), base, "0", sum(program[str(k)]["died"] for k in rooms))


if __name__ == "__main__":
    main()
