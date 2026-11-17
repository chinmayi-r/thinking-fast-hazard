"""A43 verification: save the robot's map arrays and the camera inputs that built them, at a decision, without touching
HAZARD. Called by run_baseline.py's TimedAgent when --save_maps K is given (the first K decisions of each episode).
Writes <out_dir>/maps/ep<e>_dec<n>_frame<f>.npz with: explored, height, id (sem_map, challenge.py:301-303 after the
finished-object erasure), other (temperature layer, when present), goal_map, depth and seg_mask from state["raw"],
camera_matrix and agent position / facing from the controller (fireagent_controller.py:361, 384, envs/fire/agent.py:270).
Analysis: scripts/hazard_fixes/analyze_maps.py."""
import os

import numpy as np


def save_decision_maps(state, out_dir, episode, decision, frame, controller=None):
    os.makedirs(out_dir, exist_ok=True)
    sem = state.get("sem_map", {}) or {}
    raw = state.get("raw", {}) or {}
    arrays = {}
    for k in ("explored", "height", "id", "other"):
        v = sem.get(k)
        if v is not None:
            arrays[k] = np.asarray(v)
    if state.get("goal_map") is not None:
        arrays["goal_map"] = np.asarray(state["goal_map"])
    for k in ("depth", "seg_mask", "log_temp"):
        v = raw.get(k)
        if v is not None:
            arrays[k] = np.asarray(v)
    if controller is not None:
        try:
            ag = controller.agents[0]
            arrays["camera_matrix"] = np.asarray(ag.dynamic.camera_matrix, dtype=np.float64)
            arrays["agent_position"] = np.asarray(ag.dynamic.transform.position, dtype=np.float64)
            arrays["agent_facing_rad"] = np.asarray([float(ag.get_facing())])
            sm = controller.sem_map
            arrays["map_offset"] = np.asarray(sm.map_offset, dtype=np.int64)
            arrays["grid_size"] = np.asarray([float(sm.grid_size)])
        except Exception as e:  # never let bookkeeping stop an episode
            arrays["controller_error"] = np.asarray([repr(e)])
    path = os.path.join(out_dir, f"ep{episode}_dec{decision}_frame{frame}.npz")
    np.savez_compressed(path, **arrays)
    return path
