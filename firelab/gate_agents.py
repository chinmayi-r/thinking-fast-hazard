"""Three ways to spend thinking time, on the chosen layout: who still beats the reflex at which cost?

Layout (CR, Nov 23): one partition with one 0.8 m doorway, robot width counted, items picked from a
neighbouring cell. Agents, all planning from the fire as it is now (firelab.gate.thinker):
  always  think before every action, standing still        (the always-deliberate baseline)
  commit  think once, follow the plan, think again only when blocked   (agent A)
  moving  as commit, but the reflex acts during every thought          (agent B)
One CSV row per room; the report has one row per rule set.

Run:    venv_kinder\\Scripts\\python.exe -m firelab.gate_agents --workers 10
Report: venv_kinder\\Scripts\\python.exe -m firelab.gate_agents --report-only
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from firelab.gate import Task, make_world, paired_interval, reflex, thinker, unreachable

TASK = {"layout": "one_wide", "clearance": True, "reach": 1, "bag": True, "pick_ticks": 10, "hp_loss": 2.0}
RULES = [(items, values) for items in (6, 8) for values in ((5, 1), (1, 2, 3, 5, 8))]
COSTS = (0, 2, 5, 10, 20, 100)  # ticks per thought; 10 ticks = 1 s
AGENTS = {"always": {}, "commit": {"commit": True}, "moving": {"commit": True, "moving": True}}
COLUMNS = ["items", "values", "seed", "unreachable", "reflex"] + [f"{a}_{c}" for a in AGENTS for c in COSTS]
OUT = Path(__file__).resolve().parent / "results" / "firelab_gate" / "agents_seeds.csv"


def run_chunk(job: tuple) -> list:
    items, values, seeds = job
    rows = []
    for seed in seeds:
        w = make_world(Task(items=items, values=values, **TASK), seed)
        row = {"items": items, "values": values, "seed": seed, "unreachable": unreachable(w), "reflex": round(reflex(w), 4)}
        for name, kind in AGENTS.items():
            for cost in COSTS:
                same = name == "moving" and cost == 0  # with no thinking time, moving and commit are the same agent
                row[f"{name}_{cost}"] = row["commit_0"] if same else round(thinker(w, cost, **kind), 4)
        rows.append(row)
    return rows


def report(rows: list) -> None:
    for items, values in RULES:
        rs = [r for r in rows if r["items"] == str(items) and r["values"] == str(values)]
        if not rs:
            continue
        a = {k: np.array([float(r[k]) for r in rs]) for k in COLUMNS[3:]}
        print(f"\n{items} items, values {values}: {len(rs)} rooms, reflex {a['reflex'].mean():.2f}, "
              f"unreachable items per room {a['unreachable'].mean():.2f}")
        print("agent  | " + " | ".join(f"cost {c:>3}" for c in COSTS) + " | beats reflex up to | gap at 20 [95 %]")
        for name in AGENTS:
            means = [a[f"{name}_{c}"].mean() for c in COSTS]
            ahead = [c for c, m in zip(COSTS, means) if m > a["reflex"].mean()]
            gap = a[f"{name}_20"] - a["reflex"]
            lo, hi = paired_interval(gap)
            print(f"{name:6s} | " + " | ".join(f"{m:8.2f}" for m in means)
                  + f" | {str(max(ahead)) + ' ticks' if ahead else 'never':>18s} | {gap.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=5, help="rooms per job")
    ap.add_argument("--items", type=int, nargs="*", default=None, help="run only these item counts")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(OUT.open(newline=""))) if OUT.exists() else []
    done = {(r["items"], r["values"], r["seed"]) for r in rows}
    jobs = []
    for items, values in sorted(RULES, reverse=True):  # the slow 8-item rooms first
        if args.items and items not in args.items:
            continue
        todo =[s for s in range(args.seeds) if (str(items), str(values), str(s)) not in done]
        jobs += [(items, values, todo[n:n + args.chunk]) for n in range(0, len(todo), args.chunk)]
    if jobs and not args.report_only:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        fresh = not OUT.exists()
        with OUT.open("a", newline="") as f, Pool(args.workers) as pool:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if fresh:
                writer.writeheader()
            for n, chunk in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
                writer.writerows(chunk)
                f.flush()
                rows += [{k: str(v) for k, v in r.items()} for r in chunk]
                print(f"[{n}/{len(jobs)}] items={chunk[0]['items']} values={chunk[0]['values']}", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
