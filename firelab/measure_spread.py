"""Chunk 1 measurement: how many ticks until the fire covers half the room, per p_spread?

Usage (from G:\\iw-thinking-fast):
    venv_kinder\\Scripts\\python.exe -m firelab.measure_spread [--seeds 100] [--p 0.005 0.015 ...]

No furniture is blocked here (the env blocks cells in chunk 2), so these are upper bounds on
the open floor. The table fixes p_spread in ASSUMPTIONS.md A57. Ticks are 0.1 s (D2).
"""

from __future__ import annotations

import argparse

import numpy as np

from firelab.fire import FireGrid, FireGridConfig


def ticks_to_coverage(p: float, seed: int, target: float, max_ticks: int) -> tuple[int | None, int | None]:
    """(ticks until coverage >= target or None, ticks until the first spread or None)."""
    grid = FireGrid(FireGridConfig(p_spread=p), rng=np.random.default_rng(seed))
    grid.reset()
    first_spread = None
    for t in range(1, max_ticks + 1):
        if grid.step() > 0 and first_spread is None:
            first_spread = t
        if grid.coverage() >= target:
            return t, first_spread
    return None, first_spread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=[0.005, 0.01, 0.015, 0.03, 0.1])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--max-ticks", type=int, default=5000)
    args = ap.parse_args()

    shape = FireGridConfig().shape
    print(f"grid {shape[0]} x {shape[1]} = {shape[0] * shape[1]} cells, {args.seeds} seeds, "
          f"target coverage {args.target:.0%}, cap {args.max_ticks} ticks (1 tick = 0.1 s)")
    print(f"{'p':>6} | {'reached':>7} | {'ticks to target: p10 / median / p90':>36} | {'seconds (median)':>16} | {'first spread (median)':>21}")
    for p in args.p:
        hits, firsts = [], []
        for seed in range(args.seeds):
            t, f = ticks_to_coverage(p, seed, args.target, args.max_ticks)
            if t is not None:
                hits.append(t)
            if f is not None:
                firsts.append(f)
        if hits:
            q10, med, q90 = np.percentile(hits, [10, 50, 90])
            ticks = f"{q10:6.0f} / {med:6.0f} / {q90:6.0f}"
            secs = f"{med / 10:.1f}"
        else:
            ticks, secs = "never within cap", "-"
        first = f"{np.median(firsts):.0f}" if firsts else "-"
        print(f"{p:6.3f} | {len(hits):3d}/{args.seeds:<3d} | {ticks:>36} | {secs:>16} | {first:>21}")


if __name__ == "__main__":
    main()
