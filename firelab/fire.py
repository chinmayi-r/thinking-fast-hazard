"""The fire grid: a 2-D cellular spread model with no simulator behind it.

PLAN.md decisions D4 and D5. One call to step() is one tick (0.1 s, D2 / A57). A cell is
NORMAL, BURNING, BURNT or BLOCKED. At level 1 nothing becomes BURNT: HAZARD's floor fires burn
indefinitely too (HAZARD envs/fire/object.py:53-59, burning_time = -1); BURNT is reserved for
the extinguisher (chunk 8). BLOCKED marks floor under furniture; it never burns and is never
counted as burnable.

Spread rule (D4): each tick, every BURNING cell ignites each of its 4 neighbours that is NORMAL
with probability p. A NORMAL cell with k burning neighbours therefore ignites with probability
1 - (1 - p) ** k, which is what step() draws, one uniform per cell per tick. Cells ignited in a
tick start spreading from the next tick. HAZARD's rule differs (grow for 140 frames, then a
2 % per 10-frame draw that ramps to certainty by 250 frames, one child per direction, fire.py
146-156, 183-195); it is a candidate second rule, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMAL, BURNING, BURNT, BLOCKED = 0, 1, 2, 3
STATE_NAMES = {NORMAL: "normal", BURNING: "burning", BURNT: "burnt", BLOCKED: "blocked"}


@dataclass(frozen=True)
class FireGridConfig:
    """Grid geometry and the spread probability (A57, A58)."""

    x_min: float = -1.5  # prpl-tidybot conf/config.yaml preview.floor_bounds
    x_max: float = 1.7
    y_min: float = -1.5
    y_max: float = 1.5
    cell_size: float = 0.2  # HAZARD's FIRE_SPREAD_SIZE (fire_utils.py:21)
    p_spread: float = 0.015  # per burning neighbour per tick; see measure_spread.py

    @property
    def shape(self) -> tuple[int, int]:
        """(nx, ny): cells along x and along y."""
        nx = int(round((self.x_max - self.x_min) / self.cell_size))
        ny = int(round((self.y_max - self.y_min) / self.cell_size))
        return nx, ny


class FireGrid:
    """State array plus the rule. Index order is state[i, j] with i along x, j along y."""

    def __init__(self, config: FireGridConfig = FireGridConfig(), rng=None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = np.full(config.shape, NORMAL, dtype=np.int8)
        self.tick = 0
        self.ignition: tuple[int, int] | None = None

    # geometry

    def in_bounds(self, i: int, j: int) -> bool:
        nx, ny = self.config.shape
        return 0 <= i < nx and 0 <= j < ny

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        """Cell containing world point (x, y), or None if the point is off the grid."""
        c = self.config
        i = int(np.floor((x - c.x_min) / c.cell_size))
        j = int(np.floor((y - c.y_min) / c.cell_size))
        return (i, j) if self.in_bounds(i, j) else None

    def center_of(self, i: int, j: int) -> tuple[float, float]:
        c = self.config
        return (c.x_min + (i + 0.5) * c.cell_size, c.y_min + (j + 0.5) * c.cell_size)

    # editing

    def block_box(self, x0: float, x1: float, y0: float, y1: float) -> int:
        """Mark every cell whose centre lies in [x0, x1] x [y0, y1] BLOCKED; return the count."""
        xs, ys = self._centers()
        inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        newly = inside & (self.state != BLOCKED)
        self.state[inside] = BLOCKED
        return int(newly.sum())

    def reset(self, ignition: tuple[int, int] | None = None) -> tuple[int, int]:
        """Clear fire (keep BLOCKED), ignite one cell: the given one or a uniform burnable cell."""
        self.state[self.state != BLOCKED] = NORMAL
        self.tick = 0
        if ignition is None:
            burnable = np.argwhere(self.state == NORMAL)
            if len(burnable) == 0:
                raise ValueError("no burnable cell to ignite")
            ignition = tuple(int(v) for v in burnable[self.rng.integers(len(burnable))])
        i, j = ignition
        if not self.in_bounds(i, j) or self.state[i, j] == BLOCKED:
            raise ValueError(f"cannot ignite cell {ignition}")
        self.state[i, j] = BURNING
        self.ignition = (i, j)
        return self.ignition

    def step(self) -> int:
        """Advance one tick with the D4 rule; return the number of newly ignited cells."""
        burning = self.state == BURNING
        k = np.zeros(self.state.shape, dtype=np.int8)
        k[1:, :] += burning[:-1, :]
        k[:-1, :] += burning[1:, :]
        k[:, 1:] += burning[:, :-1]
        k[:, :-1] += burning[:, 1:]
        p_cell = 1.0 - (1.0 - self.config.p_spread) ** k
        draws = self.rng.random(self.state.shape)
        new = (self.state == NORMAL) & (k > 0) & (draws < p_cell)
        self.state[new] = BURNING
        self.tick += 1
        return int(new.sum())

    # queries

    def is_burning(self, x: float, y: float) -> bool:
        cell = self.cell_of(x, y)
        return cell is not None and self.state[cell] == BURNING

    def burnable(self) -> int:
        return int((self.state != BLOCKED).sum())

    def coverage(self) -> float:
        """Fraction of burnable cells that are BURNING or BURNT; the level-2 stop rule reads it."""
        on_fire = ((self.state == BURNING) | (self.state == BURNT)).sum()
        return float(on_fire) / max(1, self.burnable())

    def burning_cells(self) -> list[tuple[int, int]]:
        return [tuple(int(v) for v in ij) for ij in np.argwhere(self.state == BURNING)]

    def features(self) -> np.ndarray:
        """(nx * ny, 3) array of x, y, state per cell, row-major in (i, j); the env's view."""
        xs, ys = self._centers()
        return np.stack([xs.ravel(), ys.ravel(), self.state.astype(float).ravel()], axis=1)

    def _centers(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny = self.config.shape
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        c = self.config
        return c.x_min + (ii + 0.5) * c.cell_size, c.y_min + (jj + 0.5) * c.cell_size
