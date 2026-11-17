"""Heading-frame check (A52): does HAZARD's get_facing (envs/fire/agent.py:270-275, inverse rotation applied to +z) give the
body's true heading, and which way does Replicant.turn_by(+90) turn? Pure TDW, no HAZARD. Launches the build on a port,
spawns a Replicant, reads transform.forward and the HAZARD formula at yaw 0, after turn_by(+90), after turn_by(+90) again,
and after move_by(1.0). Prints a table; nothing is written. Run from G:\\iw-thinking-fast:
  venv/Scripts/python.exe -X utf8 scripts/hazard_fixes/facing_check.py --build "G:\\iw-thinking-fast\\tdw_build\\TDW\\TDW.exe" --port 10830
"""
import argparse
import subprocess
import time

import numpy as np
from tdw.controller import Controller
from tdw.add_ons.replicant import Replicant
from tdw.quaternion_utils import QuaternionUtils
from tdw.replicant.action_status import ActionStatus


def hazard_facing(rotation):
    v = QuaternionUtils.multiply_by_vector(QuaternionUtils.get_inverse(rotation), np.array([0, 0, 1.0]))
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arctan2(v[2], v[0])))


def true_facing(forward):
    return float(np.degrees(np.arctan2(forward[2], forward[0])))


def run_until_done(c, r):
    for _ in range(600):
        c.communicate([])
        if r.action.status != ActionStatus.ongoing:
            return r.action.status
    return r.action.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--port", type=int, default=10830)
    a = ap.parse_args()
    proc = subprocess.Popen([a.build, f"-port={a.port}", "-screenWidth=256", "-screenHeight=256"])
    time.sleep(8)
    c = Controller(port=a.port, launch_build=False)
    r = Replicant(position={"x": 0, "y": 0, "z": 0})
    c.add_ons.append(r)
    from tdw.tdw_utils import TDWUtils
    c.communicate([{"$type": "load_scene", "scene_name": "ProcGenScene"}, TDWUtils.create_empty_room(12, 12)])
    c.communicate([])
    rows = []

    def snap(label):
        d = r.dynamic
        p = d.transform.position
        rows.append((label, true_facing(d.transform.forward), hazard_facing(d.transform.rotation), round(float(p[0]), 3), round(float(p[2]), 3)))

    snap("start")
    r.turn_by(90)
    run_until_done(c, r)
    snap("after turn_by(+90)")
    r.turn_by(90)
    run_until_done(c, r)
    snap("after turn_by(+90) again")
    r.move_by(1.0)
    st = run_until_done(c, r)
    snap(f"after move_by(1.0) [{st}]")
    print(f"{'step':32s} {'true heading (deg, atan2(fwd.z, fwd.x))':>42s} {'HAZARD get_facing (deg)':>24s} {'x':>8s} {'z':>8s}")
    for lab, t, h, x, z in rows:
        print(f"{lab:32s} {t:42.1f} {h:24.1f} {x:8.3f} {z:8.3f}")
    c.communicate({"$type": "terminate"})
    proc.terminate()


if __name__ == "__main__":
    main()
