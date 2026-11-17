"""A44: optional mapper fix in HAZARD's utils/model.py (SemanticMap.forward), active only when HAZARD_MAP_MAX_OBSTACLE_H is
set: depth points higher than that (metres above the floor) are dropped before the per-cell maximum, so the ceiling, wall
cabinets and upper walls seen from below no longer turn floor cells into 2 m obstacles (A43: 28-31 % of explored cells on
mm_kitchen_3a-1, planner weight median 24,000 against 700 with a 1.8 m cut). The shipped cut is Y < 2.5 (model.py:279-281);
unset, the code is byte-identical in behaviour. This is Navigation2's max_obstacle_height rule (default 2.0 m there).
Idempotent. Tests: scripts/hazard_fixes/test_map_clearance.py.
"""
import ast
import io
import os
import sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "utils", "model.py")
s = io.open(p, encoding="utf-8").read()
if "_map_max_obstacle_height" in s:
    print("already patched")
    sys.exit(0)

old_import = "import numpy as np\n"
assert s.count(old_import) == 1, s.count(old_import)
s = s.replace(old_import, "import os\nimport numpy as np\n", 1)

helper = '''
def _map_max_obstacle_height():
    """A44 (iw-thinking-fast, 2026-11-17): points above this height (m) are not floor-level obstacles. HAZARD's shipped
    cut is 2.5 m, which keeps the ceiling and wall cabinets as obstacles on the floor cells below them (A43). Set
    HAZARD_MAP_MAX_OBSTACLE_H (e.g. 1.8, the body's clearance) to change it; unset = 2.5 as shipped."""
    v = os.environ.get("HAZARD_MAP_MAX_OBSTACLE_H", "").strip()
    return float(v) if v else 2.5

'''
anchor = "class PointCloud"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)

old_cut = "        zipped = zipped[Y < 2.5]\n        id_map = id_map[Y < 2.5]\n        Y = Y[Y < 2.5]\n"
new_cut = ("        _hmax = _map_max_obstacle_height()  # A44: 2.5 as shipped unless HAZARD_MAP_MAX_OBSTACLE_H is set\n"
           "        zipped = zipped[Y < _hmax]\n        id_map = id_map[Y < _hmax]\n        Y = Y[Y < _hmax]\n")
assert s.count(old_cut) == 1, s.count(old_cut)
s = s.replace(old_cut, new_cut)

ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A44 applied, syntax OK")
