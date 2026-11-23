"""Trace rooms where the reflex's two logs part (first run, Nov 23): both logs in full, and the fire around the parting.

Replays the given rooms once in each world (about 5 s a room after a 6 s build).
Run: venv_kinder\\Scripts\\python.exe scripts\\first_run_trace.py 3 9 21   (PYTHONPATH = repo root)
"""
import sys

import firelab  # noqa: F401
from firelab import gate
from firelab.agents import GateSkills, reflex
from firelab.env import FireLab3DEnv
from firelab.run_rooms import parted
from firelab.skills import Skills

MAIN = gate.Task(items=8, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)
env = FireLab3DEnv(level=2, realistic_bg=False)
skills = Skills(env)
for room in [int(a) for a in sys.argv[1:]]:
    w = gate.make_world(MAIN, room)
    board = GateSkills(w)
    rule_log = reflex(board)
    env.reset(seed=room)
    skills.start()
    sim_log = reflex(skills)
    n = parted(sim_log, rule_log)
    print(f"\nroom {room}: sim {skills.score():.4f} (ends tick {skills.ticks}), rule {board.score():.4f} (ends tick {board.ticks}), "
          f"episode end tick {w.t_end}, logs part at command {n}")
    print("  items: " + ", ".join(f"{k}:{w.cells[k]} worth {w.values[k]:.0f} lit@{w.burn[w.cells[k]]:.0f}" for k in range(8)))
    for i in range(max(len(sim_log), len(rule_log))):
        s = sim_log[i] if i < len(sim_log) else None
        r = rule_log[i] if i < len(rule_log) else None
        show = lambda e: "-" if e is None else f"{e[0]} {'' if e[1] is None else e[1]} tick {e[2]} ({e[3].ticks} ticks) {e[3].reason}"
        print(f"  {i}{' <-- part' if i == n else ''}: sim [{show(s)}]   rule [{show(r)}]")
    if n is not None and n > 0:
        t_sim, t_rule = sim_log[n - 1][2], rule_log[n - 1][2]
        print(f"  the choice at command {n} was made at tick {t_sim} (sim) and {t_rule} (rule); item values then, by the rule check's formula:")
        for k in range(8):
            print(f"    item {k}: at tick {t_sim} {gate.value(w, k, t_sim):.3f}, at tick {t_rule} {gate.value(w, k, t_rule):.3f}")
env.close()
