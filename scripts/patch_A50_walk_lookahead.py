"""A50: lookahead waypoint for HAZARD's walk executor (policy/env_actions.py, agent_walk_to), active only when
HAZARD_WALK_LOOKAHEAD=<metres> is set and positive. Shipped: every key frame sends the Replicant a move_to to the planned
path's first corner (path[1], usually the next cell) with arrived_at=0.5, so MoveTo is "arrived" as soon as its turn phase
ends: 5 frames and 0.05 m per step, about 1 m per 100-frame walk budget (A49, NOTES Nov 17 06:30). With the flag the
move_to target is the point `lookahead` metres along the polyline (pure pursuit), so the inner 0.5 m test is met only
after real walking. The outer arrival test, the 100-frame budget, the planner, the map and the replanning are unchanged.
v3 (HAZARD_WALK_LOOKAHEAD_MODE=los, Nov 17 12:45): the target is the FARTHEST planned path cell within the lookahead whose
straight line from the body crosses only cells the planner treats as passable (height <= 0.5 m, astar.py:59), i.e. string
pulling over the planned path; the v2 first-segment rule (mode seg, the default) collapses to the shipped corner whenever the
first segment is short, which in the A44 map it usually is (Della column 2: 1.20 m per 100 frames, NOTES Nov 17 12:30).
Idempotent; re-run on an already patched file upgrades v2 helpers to v3. Diff: scripts/A50_walk_lookahead.diff.
"""
import ast
import io
import os
import re
import sys

p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "policy", "env_actions.py")
s = io.open(p, encoding="utf-8").read()
if "_walk_los_clear" in s:
    print("already patched (v3)")
    sys.exit(0)
if "_walk_waypoint" in s:  # v2 applied: strip its helpers and call sites, then apply v3 below
    s = re.sub(r"def _walk_lookahead_cells\(env\):.*?(?=\"\"\"this action is environment independent\"\"\")", "", s, flags=re.S)
    s = s.replace("_wp = _walk_waypoint(path, _walk_lookahead_cells(env))  # A50: path[1] unless HAZARD_WALK_LOOKAHEAD is set\n        ", "")
    s = s.replace("grid_to_real(_wp)", "grid_to_real(path[1])")
    assert "_walk_" not in s.replace("_walk_log", "").replace("agent_walk_to", ""), "v2 strip failed"

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


def _walk_los_clear(height, a, b, max_h=0.5):
    """True when every map cell on the straight line a -> b (cells, inclusive) has height <= max_h: the planner's own
    passability rule (astar.py:59, cells above 0.5 m are charged as obstacles). Sampled at half-cell steps."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(1, int(np.ceil(2 * np.linalg.norm(b - a))))
    w, h = height.shape
    for t in np.linspace(0.0, 1.0, n + 1):
        q = a + t * (b - a)
        r, c = int(round(q[0])), int(round(q[1]))
        if r < 0 or c < 0 or r >= w or c >= h or height[r, c] > max_h:
            return False
    return True


def _walk_lookahead_mode():
    import os as _os
    return _os.environ.get("HAZARD_WALK_LOOKAHEAD_MODE", "seg").strip().lower()


def _walk_waypoint(path, lookahead_cells, height=None):
    """Move_to target for one walk step, in map cells ([row, col] ints). lookahead_cells <= 0: the shipped path[1].
    Mode seg (v2, default): the point `lookahead_cells` along the FIRST segment path[0] -> path[1], or path[1] when that
    segment is shorter. (v1 followed the polyline round corners: the body cut corners into counters, 31 of 63 steps
    "detected_obstacle", NOTES Nov 17 06:45.)
    Mode los (v3): the farthest path cell within `lookahead_cells` of path[0] whose straight line from path[0] crosses
    only passable cells of `height` (string pulling); path[1] when none passes or no height map is given."""
    if lookahead_cells <= 0 or len(path) < 2:
        return [int(path[1][0]), int(path[1][1])]
    a = np.asarray(path[0], dtype=float)
    if _walk_lookahead_mode() == "los" and height is not None:
        best = None
        for i in range(1, len(path)):
            b = np.asarray(path[i], dtype=float)
            if np.linalg.norm(b - a) > lookahead_cells:
                break
            if _walk_los_clear(height, a, b):
                best = path[i]
        if best is not None:
            return [int(best[0]), int(best[1])]
        return [int(path[1][0]), int(path[1][1])]
    b = np.asarray(path[1], dtype=float)
    seg = float(np.linalg.norm(b - a))
    if seg <= lookahead_cells or seg == 0:
        return [int(path[1][0]), int(path[1][1])]
    q = a + (lookahead_cells / seg) * (b - a)
    return [int(round(q[0])), int(round(q[1]))]


'''
s = s.replace(anchor, helper + anchor)

old = '''        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            _kf = env.controller.next_key_frame(force_direction=None)
'''
new = '''        _wp = _walk_waypoint(path, _walk_lookahead_cells(env), sem_map["height"])  # A50: path[1] unless HAZARD_WALK_LOOKAHEAD is set
        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(_wp)),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.05 if record_mode else 0.5})
        if not effect_on_agents or task == "fire":
            _kf = env.controller.next_key_frame(force_direction=None)
'''
if s.count(old) == 1:  # d35c71e
    s = s.replace(old, new)
else:  # 5a50c2d (policy/env_actions.py:74-103): no record_mode / task arguments, bare next_key_frame(); anchor on the
    # "walk to first point" block of agent_walk_to (agent_walk_to_single_step has the same move_to lines but no such comment)
    old2 = '''        # walk to first point
        if path is None or len(path) <= 1:
            return False, "don't know, maybe out of bounds"
        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(path[1])),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.5})
'''
    new2 = '''        # walk to first point
        if path is None or len(path) <= 1:
            return False, "don't know, maybe out of bounds"
        _wp = _walk_waypoint(path, _walk_lookahead_cells(env), sem_map["height"])  # A50: path[1] unless HAZARD_WALK_LOOKAHEAD is set
        env.controller.do_action(agent_idx=0, action="move_to", params={"target": TDWUtils.array_to_vector3(env.controller.grid_to_real(_wp)),
                                                                        "reset_arms": reset_arms,
                                                                        "arrived_at": 0.5})
'''
    assert s.count(old2) == 1, ("no A50 anchor in this env_actions.py", s.count(old2))
    s = s.replace(old2, new2)
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A50 v3 applied, syntax OK")
