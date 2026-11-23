"""Chunk 2 probe: what does the PRPL lab URDF put on the floor, and where?

Prints every link of the lab body with its world AABB (x, y extents and z range), so we can
decide which links block fire cells. Usage: venv_kinder\\Scripts\\python.exe -u scripts/kinder_probe_lab_links.py
"""

import contextlib

import pybullet as p
from kinder.envs.kinematic3d import prpl3d

prpl3d._suppress_urdf_warnings = contextlib.nullcontext  # A63 Windows shim


def main() -> None:
    env = prpl3d.ObjectCentricPrplLab3DEnv(num_cubes=0, use_gui=False, realistic_bg=False)
    cid, lab = env.physics_client_id, env._lab_id  # pylint: disable=protected-access
    n = p.getNumJoints(lab, physicsClientId=cid)
    print(f"lab body {lab}: {n} joints/links; robot home {env.config.robot_base_home_pose}")
    print(f"{'link':>4} {'name':34s} {'x0':>6} {'x1':>6} {'y0':>6} {'y1':>6} {'z0':>6} {'z1':>6}")
    rows = []
    for link in range(-1, n):
        name = "base" if link == -1 else p.getJointInfo(lab, link, physicsClientId=cid)[12].decode()
        (x0, y0, z0), (x1, y1, z1) = p.getAABB(lab, link, physicsClientId=cid)
        rows.append((link, name, x0, x1, y0, y1, z0, z1))
    for r in sorted(rows, key=lambda r: r[6]):
        print(f"{r[0]:>4} {r[1]:34s} " + " ".join(f"{v:6.2f}" for v in r[2:]))
    env.close()


if __name__ == "__main__":
    main()
