"""Seconds per search step on Della, from CR's timing of Nov 23 (della-r3c3n13, Xeon Gold 6242 at 2.80 GHz).

CR's results/slides_nov23/time_planner.py printed the seconds of the first plan in rooms 0-9 (6 and 8 items) and
rooms 0-2 (10 items). The number of search steps of each of those plans does not depend on the machine, so it is
recounted here and divided into her seconds. About a minute on the laptop (the 10-item plans are the slow part).
Run: venv_kinder\\Scripts\\python.exe scripts\\seconds_per_step.py   (PYTHONPATH = repo root)
"""
import statistics

from firelab import gate
from firelab.planner import plan

DELLA_SECONDS = {  # pasted by CR, job 18197402
    6: [0.021, 0.105, 0.013, 0.012, 0.012, 0.250, 0.275, 0.017, 0.025, 0.034],
    8: [0.088, 1.034, 0.448, 0.134, 3.751, 2.070, 3.971, 0.028, 0.839, 1.951],
    10: [26.185, 9.490, 6.675],
}
RULES = {"values": (1, 2, 3, 5, 8), "layout": "one_wide", "clearance": True, "reach": 1, "bag": True, "pick_ticks": 10, "hp_loss": 2.0}

for items, seconds in DELLA_SECONDS.items():
    steps = []
    for room in range(len(seconds)):
        w = gate.make_world(gate.Task(items=items, **RULES), room)
        steps.append(plan(gate.forecast(w, 0), gate.State(w.start))[2])
    per_room = [1000 * s / n for s, n in zip(seconds, steps)]
    print(f"items {items}: steps {steps}")
    print(f"   ms per step by room: {[round(x, 3) for x in per_room]}")
    print(f"   all rooms together: {sum(seconds):.3f} s / {sum(steps)} steps = {1000 * sum(seconds) / sum(steps):.4f} ms per step; "
          f"median room {statistics.median(per_room):.4f}; median steps {statistics.median(steps):.0f}, median seconds {statistics.median(seconds):.3f}")
