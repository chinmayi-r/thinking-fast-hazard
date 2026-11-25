"""One picture of the level-2 room (chunk 3e): kinder's render, and the floor plan the fire and the base use.

Builds one room and steps it 150 ticks with the robot standing still (a few seconds after a 6 s build).
Run: venv_kinder\\Scripts\\python.exe scripts\\firelab_room.py   (PYTHONPATH = repo root)
Out: firelab/results/firelab_show/level2_room.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import firelab  # noqa: F401
from firelab.env import FireLab3DEnv

OUT = Path(__file__).resolve().parents[1] / "firelab/results/firelab_show/level2_room.png"
env = FireLab3DEnv(level=2, render_mode="rgb_array")
inner = env.unwrapped._object_centric_env  # pylint: disable=protected-access
env.reset(seed=0)
for _ in range(150):
    env.step(np.zeros(11, dtype=np.float32))
picture, c, pose = env.render(), inner.fire.config, inner.robot.base.get_pose()
fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))
left.imshow(picture)
left.set_title("kinder's render: the two grey boxes are the partition")
left.axis("off")
right.imshow(inner.fire.state.T, origin="lower", cmap=ListedColormap(["#eeeeee", "#f28c28", "#4a4a4a", "#7d93b2"]),
             vmin=0, vmax=3, extent=[c.x_min, c.x_max, c.y_min, c.y_max], interpolation="nearest")
right.axhline(inner.fl.safe_y, color="#2e8b57", linewidth=2)
right.text(c.x_min + 0.05, inner.fl.safe_y + 0.05, "safe strip (y >= 0.7)", color="#2e8b57")
right.add_patch(plt.Rectangle((pose.x - 0.255, pose.y - 0.275), 0.51, 0.55, fill=False, edgecolor="#1f6feb", linewidth=2))
right.text(pose.x + 0.3, pose.y, "robot base, to scale", color="#1f6feb")
for k, cell in enumerate(inner.items.cells):  # each item with what it is worth now
    x, y = inner.fire.center_of(*cell)
    right.text(x, y, f"{inner.items.value[k]:.1f}", ha="center", va="center", fontsize=9, fontweight="bold", color="#0b3d91")
right.set_title(f"floor plan, room 0 at tick 150 ({100 * inner.fire.coverage():.0f} % burning)\norange = burning, blue = partition and furniture, numbers = items and their value now", fontsize=10)
right.set_xlabel("x (m)")
right.set_ylabel("y (m)")
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("wrote", OUT)
