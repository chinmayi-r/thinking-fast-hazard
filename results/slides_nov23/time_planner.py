"""How long does one plan take in real seconds, by number of items?

Times the first thought of the current-fire planner (all items still on the floor, the largest search)
in a few rooms of the rule check. It plays no episode.
Run:  venv_kinder\\Scripts\\python.exe results\\slides_nov23\\time_planner.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from firelab.gate import State, Task, forecast, make_world, plan

TASK = {"layout": "one_wide", "clearance": True, "reach": 1, "bag": True, "pick_ticks": 10, "hp_loss": 2.0}

for items, rooms in ((6, 10), (8, 10), (10, 3)):
    times = []
    for seed in range(rooms):
        w = make_world(Task(items=items, values=(1, 2, 3, 5, 8), **TASK), seed)
        believed = forecast(w, 0)
        t0 = time.perf_counter()
        plan(believed, State(w.start))
        times.append(time.perf_counter() - t0)
        print(f"  items {items} room {seed}: {times[-1]:.3f} s", flush=True)
    times.sort()
    print(f"items {items}: median {times[len(times) // 2]:.3f} s, slowest {times[-1]:.3f} s over {rooms} rooms", flush=True)
