"""A19: bound the Replicant's _previous_action chain in HAZARD's three controllers.

TDW 1.13 deep-copies a finished action into agent._previous_action (tdw/add_ons/replicant_base.py:181) and every
built-in move/turn action stores its predecessor (tdw/replicant/actions/move_to.py:105), so the copied chain grows by
one link per high-level action and overflows Python's recursion limit late in an episode (seen Oct 16: flood rule at
decision 15, wind greedy at frame 2827). Keeping one link preserves what TDW reads (the immediately previous action).
Idempotent. Run from G:\iw-thinking-fast:  ./venv/Scripts/python.exe scripts/patch_A19_previous_action_chain.py
"""
import io
import sys

ROOT = r"G:\iw-thinking-fast\HAZARD\src\HAZARD\envs"
FILES = [ROOT + r"\fire\fireagent_controller.py", ROOT + r"\flood\floodagent_controller.py",
         ROOT + r"\wind\windagent_controller.py"]
ANCHOR = "        resp = super().communicate(commands)\n"
INSERT = (
    "        # A19 (iw-thinking-fast, 2026-10-16): TDW 1.13 deep-copies a finished action into _previous_action\n"
    "        # (tdw/add_ons/replicant_base.py:181) and every move/turn action stores its predecessor, so the copied\n"
    "        # chain grows by one link per action and overflows the recursion limit late in an episode. Keep one link.\n"
    "        for _agent in getattr(self, 'agents', []):\n"
    "            _prev = getattr(_agent, '_previous_action', None)\n"
    "            if _prev is not None and getattr(_prev, '_previous_action', None) is not None:\n"
    "                _prev._previous_action = None\n"
)
for p in FILES:
    s = io.open(p, encoding="utf-8").read()
    if "A19 (iw-thinking-fast" in s:
        print("already patched:", p)
        continue
    assert s.count(ANCHOR) == 1, f"{p}: anchor count {s.count(ANCHOR)}"
    s = s.replace(ANCHOR, ANCHOR + INSERT)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    import ast
    ast.parse(s)
    print("patched:", p)
