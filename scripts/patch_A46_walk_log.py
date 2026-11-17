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

HITS_HELPER = '''def _walk_hits(env):
    """A54 (2026-11-17): object ids the Replicant's body parts are in contact with after the key frame (TDW
    ReplicantDynamic.collisions, body parts excluded), with HAZARD's class for each ("Object", "Fire", "Agent", or "?")."""
    try:
        dyn = env.controller.agents[0].dynamic
        ids = sorted({int(o) for lst in dyn.collisions.values() for o in lst if o not in dyn.body_parts})
        objs = env.controller.manager.objects
        return [[i, objs[i].name if i in objs else "?"] for i in ids]
    except Exception as e:
        return [["error", repr(e)]]


'''

p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "policy", "env_actions.py")
s = io.open(p, encoding="utf-8").read()
if "_walk_hits" in s:
    print("already patched (with hits)")
    sys.exit(0)
if "WALK_LOG" in s:  # A46 without the A54 "hits" field: upgrade in place
    old_line = '"status": [str(x) for x in status] if status is not None else None})'
    assert s.count(old_line) == 1 and s.count("def _walk_log(") == 1
    s = s.replace(old_line, '"status": [str(x) for x in status] if status is not None else None,\n                         "hits": _walk_hits(env)})')
    s = s.replace("def _walk_log(", HITS_HELPER + "def _walk_log(", 1)
    ast.parse(s)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("A46 upgraded with hits, syntax OK")
    sys.exit(0)

anchor = '"""this action is environment independent"""\n# def agent_walk_to('
assert s.count(anchor) == 1
helper = '''WALK_LOG = []  # A46 (iw-thinking-fast, 2026-11-17): per-step records of agent_walk_to when HAZARD_WALK_LOG=1


def _walk_log_on():
    import os as _os
    return _os.environ.get("HAZARD_WALK_LOG", "0").strip() == "1"


def _walk_hits(env):
    """A54 (2026-11-17): object ids the Replicant's body parts are in contact with after the key frame (TDW
    ReplicantDynamic.collisions, body parts excluded), with HAZARD's class for each ("Object", "Fire", "Agent", or "?")."""
    try:
        dyn = env.controller.agents[0].dynamic
        ids = sorted({int(o) for lst in dyn.collisions.values() for o in lst if o not in dyn.body_parts})
        objs = env.controller.manager.objects
        return [[i, objs[i].name if i in objs else "?"] for i in ids]
    except Exception as e:
        return [["error", repr(e)]]


def _walk_log(env, target, agent_pos, target_pos, path, status):
    try:
        WALK_LOG.append({"frame": int(env.controller.frame_count), "target": int(target) if isinstance(target, int) else None,
                         "pos": [float(agent_pos[0]), float(agent_pos[2])],
                         "dist_m": float(np.linalg.norm(agent_pos[[0, 2]] - target_pos[[0, 2]])),
                         "next_cell": [int(path[1][0]), int(path[1][1])] if path is not None and len(path) > 1 else None,
                         "status": [str(x) for x in status] if status is not None else None,
                         "hits": _walk_hits(env)})
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
if s.count(old) == 1:  # d35c71e
    s = s.replace(old, new)
else:  # 5a50c2d (policy/env_actions.py:99-104): bare next_key_frame() ends agent_walk_to, right before agent_walk_to_single_step
    old2 = "        env.controller.next_key_frame()\n\ndef agent_walk_to_single_step("
    new2 = '''        _kf = env.controller.next_key_frame()
        if _walk_log_on():  # A46
            _walk_log(env, target, env.controller.agents[0].dynamic.transform.position,
                      env.controller.manager.objects[target].position if isinstance(target, int) else np.array(target), path,
                      _kf[0] if isinstance(_kf, tuple) and len(_kf) > 0 else None)

def agent_walk_to_single_step('''
    assert s.count(old2) == 1, ("no A46 anchor in this env_actions.py", s.count(old2))
    s = s.replace(old2, new2)
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A46 applied, syntax OK")
