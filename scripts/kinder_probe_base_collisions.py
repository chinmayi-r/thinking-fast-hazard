"""Probe (Nov 23): with check_base_collisions on, where can the base go in the lab room?

Run: venv_kinder\Scripts\python.exe scripts\kinder_probe_base_collisions.py   (PYTHONPATH = repo root)
"""
import dataclasses
import time

import numpy as np
import pybullet as p
from kinder.envs.kinematic3d.prpl3d import PrplLab3DEnvConfig

import firelab  # noqa: F401
from firelab.env import ObjectCentricFireLab3DEnv

t0 = time.time()
cfg = dataclasses.replace(PrplLab3DEnvConfig(), check_base_collisions=True)
env = ObjectCentricFireLab3DEnv(config=cfg, realistic_bg=False)
env.reset(seed=0)
cid, base = env.physics_client_id, env.robot.base.robot_id
print(f"built in {time.time() - t0:.1f} s; check_base_collisions = {env.config.check_base_collisions}")
print("collision at home:", env._robot_or_held_object_collision_exists())
lo, hi = p.getAABB(base, -1, physicsClientId=cid)
for link in range(p.getNumJoints(base, physicsClientId=cid)):
    a, b = p.getAABB(base, link, physicsClientId=cid)
    lo, hi = np.minimum(lo, a), np.maximum(hi, b)
print("base AABB at home, min:", np.round(lo, 3), "max:", np.round(hi, 3))


def drive(dx, dy, n=80):
    env.reset(seed=0)
    for _ in range(n):
        a = np.zeros(11, dtype=np.float32)
        a[0], a[1] = dx, dy
        env.step(a)
    q = env.robot.base.get_pose()
    return round(float(q.x), 3), round(float(q.y), 3)


for name, d in {"+y (to the cabinets)": (0, 0.05), "-y": (0, -0.05), "+x": (0.05, 0), "-x": (-0.05, 0)}.items():
    print(f"80 ticks {name}: stops at {drive(*d)}")
