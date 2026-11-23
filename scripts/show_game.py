"""Print saved games decision by decision, side by side by agent. Reads files only; plays nothing.

Run: venv_kinder\\Scripts\\python.exe scripts\\show_game.py run2_declared 6 commit moving
"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "results" / "firelab_sim"
tag, room, agents = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
for agent in agents:
    game = [json.loads(line) for line in (RUNS / tag / "games" / f"room_{room:03d}_{agent}.jsonl").open()]
    head, end = game[0], game[-1]
    first = next(g for g in game if g["type"] == "thought")
    print(f"\n=== room {room}, {agent} ({head['clock']}): score {end['score']}, ended tick {end['ticks']}, health {end['health']}")
    print("    items: " + ", ".join(f"{i['id']} at {tuple(i['cell'])} worth {i['start']:.0f}" for i in first["seen"]["items"])
          + f"; fire starts at {[tuple(c) for c in first['seen']['burning']]}")
    events = [(g["seen"]["tick"], 1, g) for g in game if g["type"] == "thought"] + [(g["ended_tick"], 0, g) for g in game if g["type"] == "command"]
    for tick, _, g in sorted(events, key=lambda e: e[:2]):
        if g["type"] == "thought":
            s = g["seen"]
            plan = " > ".join(f"{a[0]} {'' if a[1] is None else a[1]}{' THROUGH FIRE' if a[2] else ''}".replace("  ", " ") for a in g["plan"]) or "(nothing)"
            print(f"  tick {tick:3d} THINK at {tuple(s['cell'])}, health {s['health']:.0f}, bag {s['carrying']} worth {s['held']:.1f}, banked {s['saved']:.1f}, "
                  f"{len(s['burning'])} cells burning, forecast end tick {s['end_forecast']:.0f}; expects {g['estimate']} ticks, charged {g['charged']} "
                  f"({g['steps']} steps); plan worth {g['plan_value']:.1f}: {plan}")
        else:
            print(f"  tick {tick:3d}   did {g['name']} {'' if g['item'] is None else g['item']} in {g['ticks']} ticks: {'ok' if g['done'] else g['reason']}".replace("  in", " in"))
