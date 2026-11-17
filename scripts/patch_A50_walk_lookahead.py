"""A50: lookahead waypoint for HAZARD's walk executor (policy/env_actions.py, agent_walk_to), active only when
HAZARD_WALK_LOOKAHEAD=<metres> is set and positive. Shipped: every key frame sends the Replicant a move_to to the planned
path's first corner (path[1], usually the next cell) with arrived_at=0.5, so MoveTo is "arrived" as soon as its turn phase
ends: 5 frames and 0.05 m per step, about 1 m per 100-frame walk budget (A49, NOTES Nov 17 06:30). With the flag the
move_to target is the point `lookahead` metres along the polyline (pure pursuit), so the inner 0.5 m test is met only
after real walking. The outer arrival test, the 100-frame budget, the planner, the map and the replanning are unchanged.
Idempotent. Diff: scripts/A50_walk_lookahead.diff.
"""
import ast
import io
import os
import sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "policy", "env_actions.py")
s = io.open(p, encoding="utf-8").read()
if "_walk_waypoint" in s:
    print("already patched")
    sys.exit(0)

anchor = '"""this action is environment independent"""\n# def agent_walk_to('
assert s.count(anchor) == 1
helper = '''def _walk_lookahead_cells(env):
    """A50 (iw-thinking-fast, 2026-11-17): HAZARD_WALK_LOOKAHEAD in metres -> map cells; 0 when unset (shipped behaviour)."""
    import os as _os
    try:
        m = float(_os.environ.get("HAZARD_WALK_LOOKAHEAD", "0") or 0)
    except ValueError:
        return 0.0
    if m <= 0:
        return 0.0
    return m / float(env.controller.sem_map.grid_size)


def _walk_waypoint(path, lookahead_cells):
    """The point `lookahead_cells` along the planned polyline `path` (corner points, map cells) from its start; the path's
    end when it is shorter; the shipped path[1] when lookahead_cells <= 0. Returns [row, col] ints."""
    if lookahead_cells <= 0 or len(path) < 2:
        return [int(path[1][0]), int(path[1][1])]
    acc = 0.0
    prev = np.asarray(path[0], dtype=float)
    for i in range(1, len(path)):
        p = np.asarray(path[i], dtype=float)
        seg = float(np.linalg.norm(p - prev))
        if seg > 0 and acc + seg >= lookahead_cells:
            t = (lookahead_cells - acc) / seg
            q = prev + t * (p - prev)
            return [int(round(q[0])), int(round(q[1]))]
        acc += seg
        prev = p
    return [int(path[-1][0]), int(path[-1][1])]


'''
s = s.replace(anchor, helper + anchor)

old = '''        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            _kf = env.controller.next_key_frame(force_direction=None)
'''
new = '''        _wp = _walk_waypoint(path, _walk_lookahead_cells(env))  # A50: path[1] unless HAZARD_WALK_LOOKAHEAD is set
        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(_wp)),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            _kf = env.controller.next_key_frame(force_direction=None)
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A50 applied, syntax OK")
