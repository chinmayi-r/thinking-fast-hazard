"""firelab: a spreading-fire task in the PRPL lab room, on kinder's PrplLab3D simulator.

Importing this package registers the gymnasium ids below, so
    import firelab; gymnasium.make("firelab/FireLab3D-l1-v0")
works, and so does gymnasium's import-prefix form "firelab:firelab/FireLab3D-l1-v0" with no
import of our own (A56; how prpl-tidybot's KinderSimEnv will reach us, to verify in chunk 7).
"""

import sys

from gymnasium.envs.registration import register, registry

if sys.platform == "win32":
    # A63: kinder's prpl3d._suppress_urdf_warnings redirects fd 1 into a pipe while the lab URDF
    # loads and joins a drainer thread; on Windows the join never returns, so the env never
    # finishes constructing. Replacing it with a no-op before any env is built is the whole fix;
    # the cost is that pybullet's "No inertial data" warnings print. Nothing happens on Linux.
    import contextlib

    from kinder.envs.kinematic3d import prpl3d

    prpl3d._suppress_urdf_warnings = contextlib.nullcontext  # type: ignore[attr-defined]

ENV_IDS: list[str] = []


def register_environments() -> None:
    """Register every firelab variant once (gymnasium warns on a repeat)."""
    variants = {
        "firelab/FireLab3D-l1-v0": {"level": 1, "num_cubes": 0},
        "firelab/FireLab3D-l2-v0": {"level": 2, "num_cubes": 0},  # main rule set; num_items= and values= change it
    }
    for env_id, kwargs in variants.items():
        if env_id not in registry:
            register(id=env_id, entry_point="firelab.env:FireLab3DEnv", kwargs=kwargs)
        if env_id not in ENV_IDS:
            ENV_IDS.append(env_id)


register_environments()
