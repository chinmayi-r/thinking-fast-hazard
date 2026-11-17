"""A46: walk-step log inside HAZARD's agent_walk_to (policy/env_actions.py), active only when HAZARD_WALK_LOG=1. Each
loop iteration appends one record to env_actions.WALK_LOG: frame, the robot's position, the distance left to the target,
the planned next cell, and the Replicant's action status after the key frame (success / ongoing / collision / ...), which
HAZARD computes and discards (env_actions.py:124-134, fireagent_controller.py:276-300). Logging only; no behaviour change.
The runner dumps WALK_LOG per decision when the flag is on. Purpose: tell apart the causes of a capped walk (a walk that
never moves = the body refusing move_to; a walk that moves and stalls = an obstacle; a walk that reaches the counter's edge
and cannot satisfy the 0.5 m arrival test = A29 geometry). Idempotent. Diff: scripts/A46_walk_log.diff.
"""
import ast
import io
import os
import sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "policy", "env_actions.py")
s = io.open(p, encoding="utf-8").read()
if "WALK_LOG" in s:
    print("already patched")
    sys.exit(0)

anchor = '"""this action is environment independent"""\n# def agent_walk_to('
assert s.count(anchor) == 1
helper = '''WALK_LOG = []  # A46 (iw-thinking-fast, 2026-11-17): per-step records of agent_walk_to when HAZARD_WALK_LOG=1


def _walk_log_on():
    import os as _os
    return _os.environ.get("HAZARD_WALK_LOG", "0").strip() == "1"


def _walk_log(env, target, agent_pos, target_pos, path, status):
    try:
        WALK_LOG.append({"frame": int(env.controller.frame_count), "target": int(target) if isinstance(target, int) else None,
                         "pos": [float(agent_pos[0]), float(agent_pos[2])],
                         "dist_m": float(np.linalg.norm(agent_pos[[0, 2]] - target_pos[[0, 2]])),
                         "next_cell": [int(path[1][0]), int(path[1][1])] if path is not None and len(path) > 1 else None,
                         "status": [str(x) for x in status] if status is not None else None})
    except Exception as e:  # logging must never stop a walk
        WALK_LOG.append({"error": repr(e)})


'''
s = s.replace(anchor, helper + anchor)

old = '''        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            env.controller.next_key_frame(force_direction=None)
        elif task == "wind":
            wind_v = env.controller.manager.wind_v
            env.controller.next_key_frame(force_direction=wind_v)
        else:
            assert env.controller.manager.flood_manager.source_from == 'x_max'
            env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))
'''
new = '''        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            _kf = env.controller.next_key_frame(force_direction=None)
        elif task == "wind":
            wind_v = env.controller.manager.wind_v
            _kf = env.controller.next_key_frame(force_direction=wind_v)
        else:
            assert env.controller.manager.flood_manager.source_from == 'x_max'
            _kf = env.controller.next_key_frame(force_direction=np.array([-1, 0, 0]))
        if _walk_log_on():  # A46
            _walk_log(env, target, env.controller.agents[0].dynamic.transform.position,
                      env.controller.manager.objects[target].position if isinstance(target, int) else np.array(target), path,
                      _kf[0] if isinstance(_kf, tuple) and len(_kf) > 0 else None)
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A46 applied, syntax OK")
