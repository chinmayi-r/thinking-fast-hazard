"""Search the rule sets for ones that pass the gate, instead of hand-picking rows.

Every combination of the knobs in GRID is played by the five agents of firelab/gate.py on the same
seeds. One CSV row per rule set is written as soon as it finishes, so a stopped run resumes where it
left off. The report sorts by the gap between the planner that sees only the current fire and the
reflex, and shows only rule sets where the reflex scores at or below --max-reflex.

Run:    venv_kinder\\Scripts\\python.exe -m firelab.gate_grid --workers 6
Report: venv_kinder\\Scripts\\python.exe -m firelab.gate_grid --report-only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

from firelab.gate import Task, evaluate, paired_interval

GRID = {
    "items": (6, 8),
    "values": ((5, 1), (1, 2, 3, 5, 8)),
    "pick_ticks": (10, 50, 100),
    "drain": (0.02, 1.0),
    "end_coverage": (0.5, 0.8),
    "bag": (False, True),
}
FIXED = {"layout": "two_narrow", "hp_loss": 2.0}  # the Nov 22 grid ran on the 0.4 m doorways (not buildable, A69)
AGENTS = ("random", "reflex", "reflex_vpt", "planner_now", "planner_all")
EXTRA = ("gap_now", "gap_now_lo", "gap_now_hi", "gap_all", "gap_now_vpt", "end_s", "seconds")
OUT = Path(__file__).resolve().parents[1] / "results" / "firelab_gate" / "grid.csv"


def run_one(job: tuple) -> dict:
    knobs, seeds = job
    began = time.time()
    r = evaluate(Task(**FIXED, **knobs), range(seeds))
    row = {k: str(v) for k, v in knobs.items()}
    row.update({a: round(float(r[a].mean()), 3) for a in AGENTS})
    lo, hi = paired_interval(r["planner_now"] - r["reflex"])
    row.update(
        gap_now=round(float((r["planner_now"] - r["reflex"]).mean()), 3),  # the gap the report sorts by
        gap_now_lo=round(lo, 3),
        gap_now_hi=round(hi, 3),
        gap_all=round(float((r["planner_all"] - r["reflex"]).mean()), 3),  # against the all-knowing bound
        gap_now_vpt=round(float((r["planner_now"] - r["reflex_vpt"]).mean()), 3),  # does it survive the diagnostic reflex?
        end_s=round(float(r["t_end"].mean()) / 10, 1),
        seconds=round(time.time() - began, 1),
    )
    return row


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def report(rows: list, max_reflex: float) -> None:
    keep = sorted((r for r in rows if float(r["reflex"]) <= max_reflex), key=lambda r: -float(r["gap_now"]))
    print(f"{len(keep)} of {len(rows)} rule sets have the reflex at or below {max_reflex}; sorted by gap_now")
    head = ("items", "values", "pick", "drain", "end", "bag", "random", "reflex", "rfx_vpt", "pl_now", "pl_all",
            "gap_now [95 %]", "gap_all", "gap_vs_vpt")
    print(" | ".join(head))
    for r in keep:
        cells = [r[k] for k in GRID] + [f"{float(r[a]):.2f}" for a in AGENTS]
        cells.append(f"{float(r['gap_now']):+.2f} [{float(r['gap_now_lo']):+.2f}, {float(r['gap_now_hi']):+.2f}]")
        cells += [f"{float(r['gap_all']):+.2f}", f"{float(r['gap_now_vpt']):+.2f}"]
        print(" | ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None, help="run only the first n unfinished rule sets (for timing)")
    ap.add_argument("--max-reflex", type=float, default=0.6)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rows = read_rows(args.out)
    done = {tuple(r[k] for k in GRID) for r in rows}
    todo = [dict(zip(GRID, combo)) for combo in itertools.product(*GRID.values())]
    todo = [k for k in todo if tuple(str(v) for v in k.values()) not in done][: args.limit]
    if todo and not args.report_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fresh = not args.out.exists()
        with args.out.open("a", newline="") as f, Pool(args.workers) as pool:
            writer = csv.DictWriter(f, fieldnames=list(GRID) + list(AGENTS) + list(EXTRA))
            if fresh:
                writer.writeheader()
            for n, row in enumerate(pool.imap_unordered(run_one, [(k, args.seeds) for k in todo]), 1):
                writer.writerow(row)
                f.flush()
                rows.append({k: str(v) for k, v in row.items()})
                print(f"[{n}/{len(todo)}] {row['seconds']:6.1f} s  " + " ".join(f"{k}={row[k]}" for k in GRID), flush=True)
    report(rows, args.max_reflex)


if __name__ == "__main__":
    main()
