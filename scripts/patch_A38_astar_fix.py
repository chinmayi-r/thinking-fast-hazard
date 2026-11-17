"""A38: optional planner fix in HAZARD's policy/astar.py, active only when HAZARD_ASTAR_FIX=1 (ASSUMPTIONS.md A38).
(a) unexplored cells within HAZARD_ASTAR_WALL_REACH (2) cells of a ranged obstacle taller than 0.5 m are priced as a
    2.5 m wall; (b) unexplored floor costs HAZARD_ASTAR_UNEXPLORED (300 when the fix is on, 30 when off).
Idempotent; flag off leaves the weights byte-identical to the shipped function. Tests: scripts/hazard_fixes/test_astar_fix.py.
"""
import ast
import io
import os
import sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "policy", "astar.py")
s = io.open(p, encoding="utf-8").read()
if "_astar_fix_config" in s:
    print("already patched")
    sys.exit(0)

old_import = "import pyastar2d\n"
assert s.count(old_import) == 1
s = s.replace(old_import, "import os\nimport pyastar2d\n")

helper = '''
def _astar_fix_config():
    """A38 (iw-thinking-fast, 2026-11-16): planner fix, off unless HAZARD_ASTAR_FIX=1. Returns (on, unexplored cost, reach)."""
    on = os.environ.get("HAZARD_ASTAR_FIX", "0").strip() == "1"
    unexplored = float(os.environ.get("HAZARD_ASTAR_UNEXPLORED", "300" if on else "30"))
    reach = int(os.environ.get("HAZARD_ASTAR_WALL_REACH", "2"))
    return on, unexplored, reach

'''
anchor = "# sem_map: [channel, height, width]\ndef get_astar_weight(sem_map, origin, destination):\n"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor)

old_unexplored = "    weight = np.ones((w, h))\n    weight[explored == 0] = 30\n"
new_unexplored = ("    fix_on, unexplored_cost, reach = _astar_fix_config()  # A38\n"
                  "    weight = np.ones((w, h))\n    weight[explored == 0] = unexplored_cost  # 30 as shipped unless HAZARD_ASTAR_FIX=1\n")
assert s.count(old_unexplored) == 1
s = s.replace(old_unexplored, new_unexplored)

old_tail = "    weight[height > 0.1] += 10\n"
new_tail = old_tail + '''    if fix_on:  # A38 (c): the robot cannot pass anything taller than 0.5 m, so such a cell is never cheaper than a wall
        blocked = (explored == 1) & (height > 0.5)
        weight[blocked] = np.maximum(weight[blocked], 1e4)
        # A38 (a): a ranged obstacle taller than 0.5 m continues, at its own cost, into the unexplored cells next to it
        tall_w = np.where(blocked, weight, 0.0)
        near_w = np.zeros((w, h))
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                d = near_w[max(0, i): min(w, w + i), max(0, j): min(h, h + j)]
                np.maximum(d, tall_w[max(0, -i): min(w, w - i), max(0, -j): min(h, h - j)], out=d)
        extend = (explored == 0) & (near_w > 0)
        weight[extend] = np.maximum(weight[extend], near_w[extend])
'''
assert s.count(old_tail) == 1
s = s.replace(old_tail, new_tail)

ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A38 applied, syntax OK")
