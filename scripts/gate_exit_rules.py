"""Diagnostic for the gate grid: how much of each bag-on gap is the reflex's exit rule (A68a)?

For every bag-on rule set in results/firelab_gate/grid.csv this replays the official reflex and records
how its episodes end (load lost to the clock, trapped, dead), then tries three other simple exit rules
on the same worlds: bank after every pick, bank after every two picks, and the official rule plus
"head home when fewer than pick_ticks + 100 ticks remain before the tick cap". None of these is the
official reflex; they show which gaps would survive a better exit rule.

Run: venv_kinder\\Scripts\\python.exe scripts\\gate_exit_rules.py     (reflex only, under a minute)
"""

import ast
import csv

import numpy as np

from firelab.gate import LEAVE_MARGIN, State, Task, act, coverage, live, make_world, over, score, value
from firelab.gate_grid import FIXED, GRID, OUT


def reflex_exit(w, every=None, margin_ticks=0):
    """the Nov 22 coverage-rule reflex with two extra exit switches; returns the final state and how it ended."""
    s, skip, how = State(w.start), set(), "ok"
    while not over(w, s):
        todo = [k for k in live(w, s) if k not in skip]
        leave = coverage(w, s.t) >= w.task.end_coverage - LEAVE_MARGIN or w.task.max_ticks - s.t <= margin_ticks
        if s.carrying and ((every is not None and s.carrying >= every) or leave or not todo):
            if not act(w, s, ("home", None, False)):
                how = "dead" if s.hp <= 0 else ("clock" if over(w, s) else "trapped")
                break
        elif todo and not leave:
            near = lambda k: -abs(w.cells[k][0] - s.pos[0]) - abs(w.cells[k][1] - s.pos[1])
            k = max(todo, key=lambda k: (value(w, k, s.t), near(k)))
            if not act(w, s, ("get", k, False)) and not over(w, s):
                skip.add(k)
        else:
            break
    if how == "ok" and s.carrying:
        how = "dead" if s.hp <= 0 else "clock"
    return s, how


def main() -> None:
    rows = [r for r in csv.DictReader(OUT.open(newline="")) if r["bag"] == "True"]
    out = []
    for r in rows:
        knobs = {k: ast.literal_eval(r[k]) for k in GRID}
        worlds = [make_world(Task(**FIXED, **knobs), seed) for seed in range(25)]
        official = [reflex_exit(w) for w in worlds]
        ends = [how for _, how in official]
        variants = {
            "official": np.mean([score(w, s) for w, (s, _) in zip(worlds, official)]),
            "bank every pick": np.mean([score(w, reflex_exit(w, every=1)[0]) for w in worlds]),
            "bank every 2": np.mean([score(w, reflex_exit(w, every=2)[0]) for w in worlds]),
            "watch the cap": np.mean([score(w, reflex_exit(w, margin_ticks=knobs["pick_ticks"] + 100)[0]) for w in worlds]),
        }
        best = max(variants, key=variants.get)
        out.append((float(r["planner_now"]) - variants[best], r, variants, best, ends))
    print("items | values | pick | drain | end | official reflex | lost load: clock/trapped/dead of 25 | "
          "every pick | every 2 | watch cap | planner_now | gap official | gap vs best exit rule")
    for gap_best, r, v, best, ends in sorted(out, key=lambda x: -x[0]):
        lost = "/".join(str(ends.count(h)) for h in ("clock", "trapped", "dead"))
        print(f"{r['items']} | {r['values']} | {r['pick_ticks']} | {r['drain']} | {r['end_coverage']} | "
              f"{v['official']:.2f} | {lost} | {v['bank every pick']:.2f} | {v['bank every 2']:.2f} | "
              f"{v['watch the cap']:.2f} | {float(r['planner_now']):.2f} | {float(r['gap_now']):+.2f} | "
              f"{gap_best:+.2f} ({best})")


if __name__ == "__main__":
    main()
