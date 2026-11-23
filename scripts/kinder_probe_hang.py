"""Chunk 0 diagnosis: does kinder's PrplLab3D env hang at construction on Windows, and is the
stdout-redirect in prpl3d._suppress_urdf_warnings the cause?

Usage: python -u scripts/kinder_probe_hang.py [--no-suppress]
Prints a timestamped line at each stage so the stage that never returns is visible in the log.
"""

import argparse
import contextlib
import sys
import time

T0 = time.perf_counter()


def mark(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.2f} s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-suppress", action="store_true",
                    help="replace prpl3d._suppress_urdf_warnings with a no-op before building")
    args = ap.parse_args()

    mark("importing kinder")
    import kinder  # noqa: F401
    from kinder.envs.kinematic3d import prpl3d
    mark("imported")
    if args.no_suppress:
        prpl3d._suppress_urdf_warnings = contextlib.nullcontext  # type: ignore[attr-defined]
        mark("suppressor replaced with nullcontext")

    mark("constructing ObjectCentricPrplLab3DEnv(num_cubes=1, realistic_bg=False)")
    env = prpl3d.ObjectCentricPrplLab3DEnv(num_cubes=1, use_gui=False, realistic_bg=False)
    mark("constructed")
    obs, _ = env.reset(seed=0)
    mark(f"reset ok, {len(list(obs))} objects")
    env.close()
    mark("closed")
    sys.exit(0)


if __name__ == "__main__":
    main()
