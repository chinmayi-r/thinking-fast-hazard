"""Probe (Nov 23): on which cell centres can the base really stand, and does the rule check's rule agree?

The rule check let the robot centre on a cell only if none of the 8 cells around it is a wall or furniture
(gate.build, `clearance`). Here the base is put on every free cell centre, facing the cabinets as at home,
and kinder is asked whether it collides. One room, about 15 s.
Run: venv_kinder\\Scripts\\python.exe scripts\\kinder_probe_clear_cells.py   (PYTHONPATH = repo root)
"""
import numpy as np
from pybullet_helpers.geometry import SE2Pose

import firelab  # noqa: F401
from firelab import gate
from firelab.env import FireLab3DEnv
from firelab.fire import BLOCKED

env = FireLab3DEnv(level=2, realistic_bg=False)
inner = env.unwrapped._object_centric_env  # pylint: disable=protected-access
env.reset(seed=0)
free = {(int(i), int(j)) for i, j in np.argwhere(inner.fire.state != BLOCKED)}
rule = gate.make_world(gate.Task(layout="one_wide", clearance=True), 0).open  # cells the rule check allowed
stands = set()
for cell in sorted(free):
    x, y = inner.fire.center_of(*cell)
    inner.robot.set_base(SE2Pose(x, y, np.pi / 2))
    if not inner._robot_or_held_object_collision_exists():  # pylint: disable=protected-access
        stands.add(cell)
print(f"free cells {len(free)}, rule check allows {len(rule)}, simulator allows {len(stands)}")
print("allowed by the rule check but colliding in the simulator:", sorted(rule - stands))
print("free in the simulator but not allowed by the rule check:", sorted(stands - rule))
for j in range(14, -1, -1):  # a map: # wall or furniture, o both allow, + simulator only, x rule check only, . neither
    print(f"{j:2d} " + "".join(
        "#" if (i, j) not in free else "o" if (i, j) in rule & stands else "+" if (i, j) in stands
        else "x" if (i, j) in rule else "." for i in range(16)))
env.close()
