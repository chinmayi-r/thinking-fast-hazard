"""A53: drop out-of-map points in HAZARD's mapper instead of clamping them to the border (utils/model.py, Semantic_Mapping
.forward). Shipped: every projected point whose cell falls outside the 512 x 512 map is clamped to the nearest border cell
(model.py:286-287, torch.max(torch.min(XZ, bound_high), bound_low)); the pixels at the far plane (sky, windows, the
silhouette edge of every object: depth ~99.5 m) therefore paint the border with whatever segmentation id they carry, so a
target 2 m away also has phantom id cells 25 m away, and everything that locates an object by the centre of its id cells
(HAZARD's LLM prompt distance llm.py:434-450, the honest text, the program API) reads it 20-30 m away (NOTES Nov 17 10:00,
A53). Active only when HAZARD_MAP_DROP_OOB=1; off = byte-identical to shipped. Idempotent. Diff: scripts/A53_map_drop_oob.diff.
"""
import ast
import io
import os
import sys

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HAZARD", "src", "HAZARD", "utils", "model.py")
s = io.open(p, encoding="utf-8").read()
if "_map_drop_oob" in s:
    print("already patched")
    sys.exit(0)

anchor = "def _map_max_obstacle_height():"
assert s.count(anchor) == 1, "A44 must be applied first (it defines _map_max_obstacle_height)"
helper = '''def _map_drop_oob():
    """A53 (iw-thinking-fast, 2026-11-17): HAZARD_MAP_DROP_OOB=1 drops points outside the map instead of clamping them to its border."""
    import os as _os
    return _os.environ.get("HAZARD_MAP_DROP_OOB", "0").strip() == "1"


'''
s = s.replace(anchor, helper + anchor)

old = '''        XZ = torch.max(torch.min(XZ, bound_high), bound_low)
        XZ = XZ.long()
'''
new = '''        _inb = ((XZ >= bound_low) & (XZ <= bound_high)).all(0)  # A53: which points fall inside the map before clamping
        XZ = torch.max(torch.min(XZ, bound_high), bound_low)
        XZ = XZ.long()
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

old = '''        zipped = zipped[Y < _hmax]
        id_map = id_map[Y < _hmax]
        Y = Y[Y < _hmax]
'''
new = '''        _keep = Y < _hmax
        if _map_drop_oob():  # A53: a point outside the map is not evidence about any border cell
            _keep = _keep & _inb
        zipped = zipped[_keep]
        id_map = id_map[_keep]
        Y = Y[_keep]
'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A53 applied, syntax OK")
