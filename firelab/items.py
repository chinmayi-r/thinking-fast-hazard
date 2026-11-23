"""The items of level 2: where each one is and what it is worth. No simulator behind it.

Like firelab/fire.py this file knows only grid cells and ticks; firelab/env.py calls it once per
tick (chunk 3f-2). The rules are those of the rule check, firelab/gate.py, so that room `room` here has
the same items as room `room` there (A72):
  - an item lying on a burning cell loses `drain` of its starting value per tick;
  - a pick takes `pick_ticks`, from the item's cell or one next to it, and puts the item in the bag;
  - a drop takes `drop_ticks`, only in the safe strip, and banks everything in the bag;
  - only banked items count. Score = value banked / value on the floor at the start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR, BAG, SAVED, LOST = 0, 1, 2, 3  # where an item is; LOST = in the bag when the episode ended


@dataclass(frozen=True)
class ItemRules:
    """The settings. The defaults are the main rule set: 8 items since Nov 23 (A75); it was 6 (A71)."""

    count: int = 8
    values: tuple = (1, 2, 3, 5, 8)  # an item is worth one of these, drawn uniformly
    drain: float = 0.02  # share of its starting value an item on a burning cell loses per tick
    pick_ticks: int = 10
    drop_ticks: int = 5
    reach: int = 1  # pick from this many cells away: 1 = the item's own cell and the 8 around it


class Items:
    """State of every item, plus the one pick or drop the robot is in the middle of."""

    def __init__(self, rules: ItemRules = ItemRules()) -> None:
        self.rules = rules
        self.cells: list[tuple[int, int]] = []  # item k lies in cells[k] until it is picked
        self.start: list[float] = []  # what item k was worth at reset
        self.value: list[float] = []  # what it is worth now; frozen once it leaves the floor
        self.where: list[int] = []
        self.job: tuple[str, int] | None = None  # ("pick", k) or ("drop", -1) while one is under way
        self.left = 0  # ticks until the job is done

    def place(self, floor: list[tuple[int, int]], room: int) -> None:
        """Put the items on distinct cells of `floor` and draw their values, as gate.make_world does."""
        rng = np.random.default_rng([room, 1])  # the items' own stream: the fire uses default_rng(room)
        self.cells = [floor[q] for q in rng.choice(len(floor), size=self.rules.count, replace=False)]
        self.start = [float(v) for v in rng.choice(self.rules.values, size=self.rules.count)]
        self.value, self.where = list(self.start), [FLOOR] * self.rules.count
        self.job, self.left = None, 0

    def burn(self, burning) -> None:
        """One tick of fire. `burning` holds the cells that were on fire when the tick began."""
        for k, cell in enumerate(self.cells):
            if self.where[k] == FLOOR and cell in burning:
                self.value[k] = max(0.0, self.value[k] - self.rules.drain * self.start[k])

    def in_reach(self, cell: tuple[int, int]) -> list[int]:
        """Items still on the floor that a robot standing in `cell` can pick."""
        r = self.rules.reach
        return [
            k for k, (i, j) in enumerate(self.cells)
            if self.where[k] == FLOOR and abs(i - cell[0]) <= r and abs(j - cell[1]) <= r
        ]

    def start_pick(self, cell: tuple[int, int], k: int | None = None) -> bool:
        """Begin picking item k, or the most valuable item in reach. False if busy or out of reach."""
        near = self.in_reach(cell)
        if self.job or not near or (k is not None and k not in near):
            return False
        best = max(near, key=lambda q: self.value[q])
        self.job, self.left = ("pick", best if k is None else k), self.rules.pick_ticks
        return True

    def start_drop(self, in_safe: bool) -> bool:
        """Begin emptying the bag. False if busy, outside the safe strip, or the bag is empty."""
        if self.job or not in_safe or BAG not in self.where:
            return False
        self.job, self.left = ("drop", -1), self.rules.drop_ticks
        return True

    def tick(self, running: bool) -> None:
        """One tick of the job in hand. On its last tick the job takes effect.

        running: the episode has not ended this tick. A drop that ends too late banks nothing.
        """
        if self.job is None:
            return
        self.left -= 1
        if self.left > 0:
            return
        (kind, k), self.job = self.job, None
        if kind == "pick":
            self.where[k] = BAG
        else:
            self.where = [(SAVED if running else LOST) if place == BAG else place for place in self.where]

    def total(self, place: int) -> float:
        return sum(v for v, at in zip(self.value, self.where) if at == place)

    def score(self) -> float:
        return self.total(SAVED) / sum(self.start)
