"""Summary of a model run (agents `model` and `rrara1`) against the reflex, room by room. Reads files only.

Reads firelab/results/firelab_sim/<tag>/rows_*.csv and its games, and the reflex rows of the first run (rooms 0-49).
Run: venv_kinder\\Scripts\\python.exe scripts\\model_run_summary.py jev_run50   (PYTHONPATH = repo root)
"""
import collections
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

from firelab.gate import paired_interval

SIM = Path(__file__).resolve().parents[1] / "firelab" / "results" / "firelab_sim"
tag = sys.argv[1]
rows = [r for p in sorted(glob.glob(str(SIM / tag / "rows_*.csv"))) for r in csv.DictReader(open(p, newline=""))]
reflex = {int(r["room"]): float(r["sim"]) for r in csv.DictReader(open(SIM / "first_run.csv", newline="")) if r["agent"] == "reflex"}
for name in ("model", "rrara1"):
    mine = {int(r["room"]): r for r in rows if r["agent"] == name}
    rooms = sorted(set(mine) & set(reflex))
    if not rooms:
        continue
    sim, base = np.array([float(mine[k]["sim"]) for k in rooms]), np.array([reflex[k] for k in rooms])
    lo, hi = paired_interval(sim - base)
    col = lambda key: np.array([float(mine[k][key]) for k in rooms])
    print(f"{name} ({mine[rooms[0]]['clock']}): {len(rooms)} rooms, score {sim.mean():.3f}, reflex {base.mean():.3f}, gap {np.mean(sim - base):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}], ahead/level/behind {int((sim - base > 0.005).sum())}/{int((abs(sim - base) <= 0.005).sum())}/"
          f"{int((sim - base < -0.005).sum())}, deaths {int((col('health') <= 0).sum())}")
    print(f"     per episode: {col('thoughts').mean():.1f} calls, {col('charged_ticks').mean():.1f} ticks of call time, "
          f"tokens in {col('tokens_in').mean():.0f} out {col('tokens_out').mean():.0f}; total {col('cost_usd').sum():.4f} dollars")
    calls, seconds, checks, differ = collections.Counter(), [], collections.Counter(), []
    for room in rooms:
        game = [json.loads(line) for line in (SIM / tag / "games" / f"room_{room:03d}_{name}.jsonl").open()]
        for t in (g for g in game if g["type"] == "thought"):
            seconds.append(t["measured_s"])
            calls["no answer: " + t["error"] if t["choice"] is None else "valid" if name == "rrara1" and t["choice"] == 0 else "invalid" if name == "rrara1" else "answered"] += 1
            if name == "rrara1":
                checks[t["previous_check"]] += 1
        if abs(float(mine[room]["sim"]) - reflex[room]) > 0.005:
            differ.append((room, float(mine[room]["sim"]), reflex[room]))
    seconds.sort()
    print(f"     calls: {dict(calls)}; seconds median {seconds[len(seconds) // 2]:.3f}, 90th percentile {seconds[int(0.9 * len(seconds))]:.3f}, slowest {seconds[-1]:.3f}")
    if checks:
        print(f"     what became of each check at the next decision: {dict(checks)}")
    print(f"     rooms that differ from the reflex (room, this agent, reflex): {differ}")
