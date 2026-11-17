"""Tests for the compiler (scripts/compiled/compile.py): prompt assembly, the dry run, reply parsing, the audit gate, and
the command line, with a scripted caller in place of the model. No API, no simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest compiled.test_compile -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from compiled import compile as comp
from compiled.api import ProgramAgent
from compiled.test_traces import make_episode

HERE = os.path.dirname(os.path.abspath(__file__))
GOOD = ('A short note on the episodes.\n\n```python\n"""round one"""\nfrom compiled.api import Paths\n\n\nclass Policy:\n'
        '    def __init__(self, task, target_values):\n        self.task = task\n        self.seen = {}\n\n'
        '    def decide(self, obs):\n        if obs["holding"]:\n            return ("drop", None)\n'
        '        if obs["at_object"] and obs["at_object"]["is_target"]:\n            return ("pick_up", obs["at_object"]["id"])\n'
        '        t = [o for o in obs["objects"] if o["is_target"] and o["cell"] is not None]\n'
        '        if not t:\n            return ("explore", None)\n'
        '        t.sort(key=lambda o: (self.seen.get(o["id"], 0), -o["value"], o["straight_cells"]))\n'
        '        self.seen[t[0]["id"]] = self.seen.get(t[0]["id"], 0) + 1\n        return ("walk_to", t[0]["id"])\n```\n')
BAD = "Here you go:\n```python\nimport random\nclass Policy:\n    def __init__(self, task, target_values): pass\n    def decide(self, obs): return ('explore', None)\n```\n"


class TestPrompt(unittest.TestCase):
    def test_sections_in_order(self):
        p = comp.build_prompt("API TEXT", "class Policy: ...", "EPISODES", previous_src="OLD", validation={"mean": 0.1}, notes="NOTE")
        order = [p.index(s) for s in ("# The API", "API TEXT", "# A worked example", "class Policy", "# Recorded episodes", "EPISODES",
                                      "# The previous round", "OLD", "# How the previous program scored", '"mean": 0.1', "# Notes", "NOTE", "# Task")]
        self.assertEqual(order, sorted(order))
        p1 = comp.build_prompt("A", "B", "C")
        self.assertNotIn("previous round", p1)
        self.assertNotIn("Notes from", p1)

    def test_extract_module(self):
        self.assertTrue(comp.extract_module(GOOD).startswith('"""round one"""'))
        self.assertEqual(comp.extract_module("no block at all"), "no block at all\n")
        self.assertEqual(comp.extract_module("```\nx = 1\n```\nthen ```python\ny = 2\n```"), "y = 2\n")


class TestRound(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()

    def test_dry_run_writes_the_prompt_only(self):
        rec = comp.compile_round(self.out, 1, "EPISODES", dry_run=True)
        self.assertTrue(rec["dry_run"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "prompt_round1.md")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "policy_round1.py")))
        prompt = open(os.path.join(self.out, "prompt_round1.md"), encoding="utf-8").read()
        self.assertIn("class Policy", prompt)  # the reflex example is in
        self.assertIn("## 4. Actions and their legality", prompt)  # the API document is in
        self.assertIn("EPISODES", prompt)

    def test_accepted_program_loads_and_plays(self):
        seen = {}
        rec = comp.compile_round(self.out, 1, "EPISODES", caller=lambda p: seen.setdefault("prompt", p) and GOOD)
        self.assertTrue(rec["accepted"], rec["audit"])
        self.assertEqual(rec["policy"], os.path.join(self.out, "policy_round1.py"))
        self.assertTrue(os.path.exists(os.path.join(self.out, "reply_round1.md")))
        self.assertIn("# Task", seen["prompt"])
        agent = ProgramAgent(task="fire", program_path=rec["policy"])
        agent.reset(goal_objects=["hairbrush", "purse"], objects_info={"hairbrush": {"value": 1}, "purse": {"value": 5}})
        from compiled.test_api import PI_FIRST, scene
        self.assertEqual(agent.choose_target(scene(objects={66: (50, 40), 56: (30, 30)}), PI_FIRST), ("walk_to", "56"))  # highest value first
        self.assertEqual(agent.choose_target(scene(objects={66: (50, 40), 56: (30, 30)}), PI_FIRST), ("walk_to", "66"))  # then the unseen one
        meta = json.load(open(os.path.join(self.out, "compile_round1.json")))
        self.assertEqual((meta["round"], meta["accepted"], meta["audit"]), (1, True, []))

    def test_rejected_program_is_kept_but_not_accepted(self):
        rec = comp.compile_round(self.out, 2, "EPISODES", previous_src="old", validation={"mean": 0.2}, caller=lambda p: BAD)
        self.assertFalse(rec["accepted"])
        self.assertTrue(any("'random'" in a for a in rec["audit"]))
        self.assertTrue(os.path.exists(os.path.join(self.out, "policy_round2.py")))
        prompt = open(os.path.join(self.out, "prompt_round2.md"), encoding="utf-8").read()
        self.assertIn("# The previous round's program", prompt)

    def test_no_key_no_caller_is_refused(self):
        with self.assertRaises(ValueError):
            comp.compile_round(self.out, 1, "EPISODES")


class TestCommandLine(unittest.TestCase):
    def test_dry_run_from_a_sweep_folder(self):
        root = tempfile.mkdtemp()
        make_episode(root)
        out = os.path.join(root, "round1")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in [env.get("PYTHONPATH", ""), os.path.dirname(HERE)] if p)
        r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "compile.py"), "--traces", root, "--out", out, "--round", "1",
                            "--dry_run", "--max_decisions", "3", "--with_reasoning"], capture_output=True, text=True, timeout=180, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        prompt = open(os.path.join(out, "prompt_round1.md"), encoding="utf-8").read()
        self.assertIn("### Episode mm_kitchen_test: final value rate 0.167", prompt)
        self.assertIn("scorer pays 1", prompt)  # the scorer's values by name were loaded from HAZARD's table
        self.assertIn("reasoning: Because the bag is worth 5.", prompt)
        self.assertNotIn("decision 4", prompt)


if __name__ == "__main__":
    unittest.main()
