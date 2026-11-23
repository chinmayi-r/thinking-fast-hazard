"""Skills: what a robot in the level-2 room can be told to do. Every agent acts through these.

A skill runs env.step once per tick until it is finished and returns Result(ticks, done, reason), so
nothing here is free: walking one 0.2 m cell costs 4 ticks, a pick 10, a drop 5, as in the rule check
(firelab/gate.py, whose route, walk, pick and drop these mirror). A skill never claims success it did
not have: done is true only if the robot got there, the item is in the bag, or the bag was banked.

The skills read the fire grid and the items from the env object directly; the same numbers are in the
state vector (A74).
"""

from __future__ import annotations

import math
import time
from typing import Callable, NamedTuple

import numpy as np

from firelab.fire import BLOCKED, BURNING
from firelab.items import BAG, FLOOR, SAVED
from firelab.planner import board_from

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
Cell = tuple[int, int]


class Result(NamedTuple):
    ticks: int  # env steps the skill used
    done: bool
    reason: str  # "ok", "no route", "blocked", "refused", "deadline" or "episode over"


class Skills:
    """Wraps one FireLab3DEnv. Call env.reset first, then start(); skills step the env themselves."""

    def __init__(self, env) -> None:
        self.env = env
        self.inner = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        self.fire, self.items = self.inner.fire, self.inner.items
        nx, ny = self.fire.config.shape
        solid = {(int(i), int(j)) for i, j in np.argwhere(self.fire.state == BLOCKED)}
        # Cells the base may centre on: none of the 9 cells around and under it is a wall or furniture.
        # The rule check's rule; scripts/kinder_probe_clear_cells.py found the simulator allows the same 161.
        self.open = {
            (i, j) for i in range(nx) for j in range(ny)
            if not any((i + a, j + b) in solid for a in (-1, 0, 1) for b in (-1, 0, 1))
        }
        self.start()

    def start(self) -> None:
        """Call after every env.reset."""
        self.over, self.ticks, self.info = False, 0, {}
        self.sim_seconds = 0.0  # wall-clock time spent inside env.step, so a runner can tell thinking from simulating

    def cell(self) -> Cell:
        pose = self.inner.robot.base.get_pose()
        return self.fire.cell_of(pose.x, pose.y)

    def _step(self, dx: float = 0.0, dy: float = 0.0, gripper: float = 0.0) -> None:
        action = np.zeros(11, dtype=np.float32)
        action[0], action[1], action[-1] = dx, dy, gripper
        began = time.perf_counter()
        *_, terminated, truncated, self.info = self.env.step(action)
        self.sim_seconds += time.perf_counter() - began
        self.ticks += 1
        self.over = terminated or truncated

    def _finish(self, since: int, done: bool, reason: str) -> Result:
        return Result(self.ticks - since, done, "ok" if done else "episode over" if self.over else reason)

    def route(self, goal: Callable[[Cell], bool], cross: bool = False) -> list[Cell] | None:
        """Breadth-first search from the robot's cell: the shortest list of cells to one where goal is true.

        cross=False keeps out of cells that are burning now. None if there is no way.
        """
        start = self.cell()
        prev, layer = {start: None}, [start]
        while layer:
            for c in layer:
                if goal(c):
                    path = []
                    while c != start:
                        path.append(c)
                        c = prev[c]
                    return path[::-1]
            nxt = []
            for i, j in layer:
                for di, dj in NEIGHBOURS:
                    n = (i + di, j + dj)
                    if n in prev or n not in self.open or (not cross and self.fire.state[n] == BURNING):
                        continue
                    prev[n] = (i, j)
                    nxt.append(n)
            layer = nxt
        return None

    def _drive(self, cell: Cell) -> bool:
        """Drive the base to the centre of a neighbouring cell. False if kinder refuses a move."""
        tx, ty = self.fire.center_of(*cell)
        step = self.inner.fl.max_base_step
        while not self.over:
            pose = self.inner.robot.base.get_pose()
            dx, dy = tx - pose.x, ty - pose.y
            if max(abs(dx), abs(dy)) < 1e-4:
                return True
            self._step(float(np.clip(dx, -step, step)), float(np.clip(dy, -step, step)))
            now = self.inner.robot.base.get_pose()
            if abs(now.x - pose.x) + abs(now.y - pose.y) < 1e-9:
                return False  # the move was reverted: something solid is in the way
        return False

    def walk(self, goal: Callable[[Cell], bool], cross: bool = False, until: float = math.inf) -> Result:
        """Walk until goal(cell) is true. The route is found again after every cell, since the fire moves.

        until: a deadline tick. Once the clock has reached it the walk stops at the cell it is on ("deadline").
        """
        since = self.ticks
        while not self.over and not goal(self.cell()):
            if self.ticks >= until:
                return self._finish(since, False, "deadline")
            path = self.route(goal, cross)
            if path is None:
                return self._finish(since, False, "no route")
            if not self._drive(path[0]):
                return self._finish(since, False, "blocked")
        return self._finish(since, not self.over, "")

    def wait(self, ticks: int = 1) -> Result:
        since = self.ticks
        while not self.over and self.ticks - since < ticks:
            self._step()
        return self._finish(since, self.ticks - since == ticks, "")

    # what an agent may look at

    @property
    def count(self) -> int:
        return len(self.items.cells)

    def live(self) -> list[int]:
        """Items still on the floor with some value left."""
        return [k for k, place in enumerate(self.items.where) if place == FLOOR and self.items.value[k] > 0]

    def value(self, k: int) -> float:
        return self.items.value[k]

    def carrying(self) -> int:
        return self.items.where.count(BAG)

    def item_cell(self, k: int) -> Cell:
        return self.items.cells[k]

    def score(self) -> float:
        return self.items.score()

    def believe(self, caution: float = 1.0) -> tuple:
        """(world, state): the rule check's board as the robot sees it now, with the forecast fire (firelab.planner)."""
        return board_from(self, caution)

    # the two goals agents walk to

    def near(self, k: int) -> Callable[[Cell], bool]:
        """True for every cell from which item k's cell is in reach (whether or not the item is still there)."""
        (ci, cj), r = self.items.cells[k], self.items.rules.reach
        return lambda c: abs(c[0] - ci) <= r and abs(c[1] - cj) <= r

    def safe(self, c: Cell) -> bool:
        return self.inner.in_safe_strip(*self.fire.center_of(*c))

    # pick and drop, alone and after a walk

    def pick(self, k: int | None = None) -> Result:
        """Pick item k from here (10 ticks); with no k, the lowest-numbered item in reach. done = it is in the bag."""
        since, near = self.ticks, self.items.in_reach(self.cell())
        k = near[0] if k is None and near else k
        if k is None or not self.items.start_pick(self.cell(), k):  # by name; a gripper "close" takes the most valuable
            return self._finish(since, False, "refused")
        while not self.over and self.items.job:
            self._step()
        return self._finish(since, self.items.where[k] == BAG, "")

    def drop(self) -> Result:
        """Empty the bag here (5 ticks). Refused outside the safe strip or with nothing carried. done = it is banked."""
        since = self.ticks
        carried = [k for k, place in enumerate(self.items.where) if place == BAG]
        if not carried or not self.safe(self.cell()):
            return self._finish(since, False, "refused")
        self._step(gripper=1.0)  # gripper "open" starts the drop; this is its first tick
        while not self.over and self.items.job:
            self._step()
        return self._finish(since, all(self.items.where[k] == SAVED for k in carried), "refused")

    def get(self, k: int, cross: bool = False, until: float = math.inf) -> Result:
        """Walk into reach of item k, then pick it. A deadline stops the walk, never a pick that has begun."""
        since = self.ticks
        if self.items.where[k] != FLOOR:
            return self._finish(since, False, "refused")  # already picked
        walked = self.walk(self.near(k), cross, until)
        if not walked.done:
            return self._finish(since, False, walked.reason)
        return self._finish(since, self.pick(k).done, "refused")

    def home(self, cross: bool = False, until: float = math.inf) -> Result:
        """Walk to the safe strip, then empty the bag there. A deadline stops the walk, never a drop that has begun."""
        since = self.ticks
        walked = self.walk(self.safe, cross, until)
        if not walked.done:
            return self._finish(since, False, walked.reason)
        return self._finish(since, self.drop().done, "refused")
