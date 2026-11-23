"""Chunk 0 measurement: how fast does kinder's PrplLab3D env step and render on this machine?

Usage (venv_kinder active or via its python.exe):
    python scripts/kinder_step_rate.py [--env kinder/PrplLab3D-o1-v0] [--steps 200]

Prints: construction time, reset time, random-action steps per second, zero-action steps per
second, and the time of one overview render. Numbers go into NOTES.md with the machine name.
"""

import argparse
import sys
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="kinder/PrplLab3D-o1-v0")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    import gymnasium
    import kinder

    if sys.platform == "win32":
        # A63 (Nov 22): prpl3d._suppress_urdf_warnings redirects fd 1 into a pipe while the URDF
        # loads and joins a drainer thread; on Windows the join never returns (verified with
        # scripts/kinder_probe_hang.py: hangs as shipped, 6 s with the redirect disabled).
        import contextlib
        from kinder.envs.kinematic3d import prpl3d
        prpl3d._suppress_urdf_warnings = contextlib.nullcontext  # type: ignore[attr-defined]

    kinder.register_all_environments()
    t = time.perf_counter()
    env = gymnasium.make(args.env, render_mode="rgb_array")
    t_make = time.perf_counter() - t

    t = time.perf_counter()
    obs, _ = env.reset(seed=0)
    t_reset = time.perf_counter() - t
    print(f"env={args.env} obs_dim={obs.shape[0]} action_dim={env.action_space.shape[0]}")
    print(f"make {t_make:.2f} s, reset {t_reset:.3f} s")

    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(args.steps):
        a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        env.step(a)
    dt = time.perf_counter() - t
    print(f"random actions: {args.steps} steps in {dt:.2f} s = {args.steps / dt:.0f} steps/s")

    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    t = time.perf_counter()
    for _ in range(args.steps):
        env.step(zero)
    dt = time.perf_counter() - t
    print(f"zero action:    {args.steps} steps in {dt:.2f} s = {args.steps / dt:.0f} steps/s")

    t = time.perf_counter()
    img = env.render()
    dt = time.perf_counter() - t
    print(f"render: {img.shape} in {dt:.3f} s")
    env.close()


if __name__ == "__main__":
    main()
