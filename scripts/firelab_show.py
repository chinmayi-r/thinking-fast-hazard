"""One picture of what firelab is: a HAZARD frame, kinder's lab room, and our fire grid on that room.

Runs one 300-tick episode with the robot standing still (about 2 s after a 6 s build).
Run: venv_kinder\Scripts\python.exe scripts\firelab_show.py
Out: results/firelab_show/what_we_are_building.png
"""

import argparse
from pathlib import Path

import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

import firelab  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
HAZARD_FRAME = ROOT / "results/fire_human_ep1_20261113_115543/0/0553_human_frame.png"
COLORS = ["#eeeeee", "#f28c28", "#4a4a4a", "#7d93b2"]  # normal, burning, burnt, blocked
LABELS = ["floor", "burning", "burnt", "furniture (never burns)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, nargs="+", default=[0, 100, 200, 300])
    ap.add_argument("--out", default=str(ROOT / "results/firelab_show/what_we_are_building.png"))
    args = ap.parse_args()

    env = gymnasium.make("firelab/FireLab3D-l1-v0", render_mode="rgb_array")
    inner = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    env.reset(seed=args.seed)
    room = env.render()
    c = inner.fire.config
    snaps = {}
    zero = np.zeros(11, dtype=np.float32)
    for t in range(max(args.ticks) + 1):
        if t in args.ticks:
            snaps[t] = (inner.fire.state.copy(), inner.fire.coverage())
        env.step(zero)
    pose = inner.robot.base.get_pose()
    env.close()

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, len(args.ticks), height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(Image.open(HAZARD_FRAME))
    ax.set_title("1. HAZARD (what we used until now)\nThreeDWorld, Unity, needs a GPU and a display", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1:])
    ax.imshow(room)
    ax.set_title("2. kinder's PrplLab3D room (the lab's simulator)\nthe PRPL lab and its TidyBot; no fire, no clock", fontsize=10)
    ax.axis("off")
    cmap = ListedColormap(COLORS)
    for k, t in enumerate(args.ticks):
        state, cov = snaps[t]
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(state.T, origin="lower", cmap=cmap, vmin=0, vmax=3,
                  extent=[c.x_min, c.x_max, c.y_min, c.y_max], interpolation="nearest")
        ax.set_xticks(np.arange(c.x_min, c.x_max + 1e-9, c.cell_size), minor=True)
        ax.set_yticks(np.arange(c.y_min, c.y_max + 1e-9, c.cell_size), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        ax.plot(pose.x, pose.y, marker="o", markersize=9, color="#1f6feb", markeredgecolor="white")
        ax.annotate("robot", (pose.x, pose.y), textcoords="offset points", xytext=(8, 6), fontsize=9, color="#1f6feb")
        ax.set_title(f"tick {t} = {t * 0.1:.0f} s, {100 * cov:.0f}% burnt or burning", fontsize=10)
        ax.set_xlabel("x (m)")
        if k == 0:
            ax.set_ylabel("3. what we add: the fire grid, seen from above\ny (m)")
    seen = set(np.unique(np.concatenate([v[0].ravel() for v in snaps.values()])))  # BURNT is unused so far
    fig.legend(handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in range(4) if k in seen],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out, "robot at", round(pose.x, 2), round(pose.y, 2),
          "coverage", {t: round(v[1], 3) for t, v in snaps.items()})


if __name__ == "__main__":
    main()
