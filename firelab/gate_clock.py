"""The finalist rule sets on several wall layouts, many seeds, with the clock.

Rule sets: bag on, 10-tick picks, every combination of RULES. Variants: (layout, clearance) pairs.
The current-fire planner is charged 0, 2, 20 and 100 ticks per decision. One CSV row per (variant,
rule set, seed) holds the score of every agent in that room, so single rooms can be inspected; the
report averages them. A stopped run resumes from the rows already written.

Run:    venv_kinder\\Scripts\\python.exe -m firelab.gate_clock --workers 10
Report: venv_kinder\\Scripts\\python.exe -m firelab.gate_clock --report-only
Slow fire: add  --p-spread 0.005 --max-ticks 1500 --tag slow --seeds 25
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from firelab.gate import EXIT_RULES, Task, evaluate, paired_interval

RULES = {"items": (6, 8), "values": ((5, 1), (1, 2, 3, 5, 8)), "drain": (0.02, 1.0), "end_coverage": (0.5, 0.8)}
FIXED = {"bag": True, "pick_ticks": 10, "hp_loss": 2.0}
VARIANTS = (("two_narrow", False), ("two_wide", False), ("two_wide", True), ("one_wide", True))  # (layout, clearance)
CHARGES = (0, 2, 20, 100)  # ticks per decision; 10 ticks = 1 s
NOW = ["planner_now" if c == 0 else f"planner_now_{c}" for c in CHARGES]
KEYS = ["layout", "clearance"] + list(RULES) + ["seed"]
SCORES = ["random"] + list(EXIT_RULES) + ["reflex_vpt"] + NOW + ["planner_all", "t_end", "unreachable"]
CHUNK = 10  # seeds per job, so the slow rule sets spread over the workers
OUT_DIR = Path(__file__).resolve().parent / "results" / "firelab_gate"


def run_chunk(job: tuple) -> list:
    (layout, clearance), knobs, seeds, extra = job
    r = evaluate(Task(layout=layout, clearance=clearance, **FIXED, **knobs, **extra), seeds, CHARGES)
    base = {"layout": layout, "clearance": clearance, **knobs}
    return [{**base, "seed": seed, **{k: round(float(r[k][n]), 4) for k in SCORES}} for n, seed in enumerate(seeds)]


def report(rows: list) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEYS[:-1])].append(r)
    lines = []
    for key, rs in groups.items():
        a = {k: np.array([float(r[k]) for r in rs]) for k in SCORES}
        gap = a["planner_now"] - a["reflex"]
        lo, hi = paired_interval(gap)
        best = max(EXIT_RULES, key=lambda k: a[k].mean())
        loses = next((str(c) for c, k in zip(CHARGES, NOW) if a[k].mean() < a["reflex"].mean()), "never")
        cells = list(key) + [str(len(rs))] + [f"{a[k].mean():.2f}" for k in ("random", "reflex")]
        cells += [f"{a[best].mean():.2f} ({best})", f"{a['reflex_vpt'].mean():.2f}", f"{a['planner_all'].mean():.2f}"]
        cells += [f"{a[k].mean():.2f}" for k in NOW]
        cells += [f"{gap.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{a['planner_now'].mean() - a[best].mean():+.2f}"]
        cells += [f"{(gap > 0).sum()}/{(gap == 0).sum()}/{(gap < 0).sum()}", f"{a['reflex'].std():.2f}", loses,
                  f"{a['unreachable'].mean():.2f}"]
        lines.append((-gap.mean(), " | ".join(cells)))
    print(" | ".join(KEYS[:-1] + ["seeds", "random", "reflex", "best exit rule", "rfx_vpt", "pl_all"]
                     + [f"now@{c}" for c in CHARGES]
                     + ["gap [95 %]", "gap vs best exit", "rooms won/tied/lost", "reflex sd", "loses at", "unreachable items"]))
    for _, line in sorted(lines):
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="fast", help="names the output file: clock_seeds_<tag>.csv")
    ap.add_argument("--p-spread", type=float, default=0.015)
    ap.add_argument("--max-ticks", type=int, default=500)
    ap.add_argument("--variants", type=int, nargs="*", default=None, help="indices into VARIANTS (default: all)")
    ap.add_argument("--items", type=int, nargs="*", default=None, help="restrict the item counts")
    ap.add_argument("--drain", type=float, nargs="*", default=None, help="restrict the drain values")
    ap.add_argument("--end-coverage", type=float, nargs="*", default=None, help="restrict the end coverages")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out = OUT_DIR / f"clock_seeds_{args.tag}.csv"
    rows = list(csv.DictReader(out.open(newline=""))) if out.exists() else []
    done = {tuple(r[k] for k in KEYS) for r in rows}
    extra = {"p_spread": args.p_spread, "max_ticks": args.max_ticks}
    jobs = []
    for v in [VARIANTS[i] for i in (args.variants if args.variants is not None else range(len(VARIANTS)))]:
        for combo in itertools.product(*RULES.values()):
            knobs = dict(zip(RULES, combo))
            limits = {"items": args.items, "drain": args.drain, "end_coverage": args.end_coverage}
            if any(allowed and knobs[k] not in allowed for k, allowed in limits.items()):
                continue
            key = (v[0], str(v[1])) + tuple(str(x) for x in combo)
            todo = [s for s in range(args.seeds) if key + (str(s),) not in done]
            jobs += [(v, knobs, todo[n:n + CHUNK], extra) for n in range(0, len(todo), CHUNK)]
    if jobs and not args.report_only:
        jobs.sort(key=lambda j: -j[1]["items"])  # the slow 8-item jobs first
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fresh = not out.exists()
        with out.open("a", newline="") as f, Pool(args.workers) as pool:
            writer = csv.DictWriter(f, fieldnames=KEYS + SCORES)
            if fresh:
                writer.writeheader()
            for n, chunk in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
                writer.writerows(chunk)
                f.flush()
                rows += [{k: str(v) for k, v in r.items()} for r in chunk]
                print(f"[{n}/{len(jobs)}] {chunk[0]['layout']} clearance={chunk[0]['clearance']} items={chunk[0]['items']}", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
