"""Tests for the A46 walk-step log (scripts/patch_A46_walk_log.py applied to HAZARD/src/HAZARD/policy/env_actions.py) and
the runner's dump of it. A fake controller stands in for the simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest hazard_fixes.test_walk_log -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from policy import env_actions  # HAZARD module, patched


class FakeEnv:
    class C:
        frame_count = 120
    controller = C()


class TestWalkLog(unittest.TestCase):
    def setUp(self):
        env_actions.WALK_LOG.clear()
        self._saved = os.environ.pop("HAZARD_WALK_LOG", None)

    def tearDown(self):
        env_actions.WALK_LOG.clear()
        os.environ.pop("HAZARD_WALK_LOG", None)
        if self._saved is not None:
            os.environ["HAZARD_WALK_LOG"] = self._saved

    def test_flag(self):
        self.assertFalse(env_actions._walk_log_on())
        os.environ["HAZARD_WALK_LOG"] = "1"
        self.assertTrue(env_actions._walk_log_on())

    def test_record(self):
        env_actions._walk_log(FakeEnv(), 12345, np.array([1.0, 0.0, 2.0]), np.array([1.3, 0.9, 2.4]), [[10, 20], [11, 20]], ["ActionStatus.ongoing"])
        r = env_actions.WALK_LOG[-1]
        self.assertEqual((r["frame"], r["target"], r["pos"], r["next_cell"], r["status"]), (120, 12345, [1.0, 2.0], [11, 20], ["ActionStatus.ongoing"]))
        self.assertAlmostEqual(r["dist_m"], 0.5, places=6)
        env_actions._walk_log(FakeEnv(), np.array([0, 0, 0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), None, None)
        self.assertEqual((env_actions.WALK_LOG[-1]["target"], env_actions.WALK_LOG[-1]["next_cell"], env_actions.WALK_LOG[-1]["status"]), (None, None, None))

    def test_runner_dump(self):
        from run_baseline import TimedAgent

        class Inner:
            agent_type = "rule"
            def reset(self, *a, **k): pass
            def choose_target(self, s, p): return ("explore", None)
        d = Path(tempfile.mkdtemp())
        t = TimedAgent(Inner(), d / "decisions.jsonl")
        t.reset()
        env_actions.WALK_LOG.extend([{"frame": 1, "dist_m": 2.0}, {"frame": 2, "dist_m": 1.9}])
        t.choose_target({"goal_map": np.zeros((4, 4))}, {"step": 3})
        lines = [json.loads(l) for l in open(d / "walk_log.jsonl", encoding="utf-8")]
        self.assertEqual([(l["frame"], l["decision"], l["episode"]) for l in lines], [(1, 0, 1), (2, 0, 1)])
        self.assertEqual(env_actions.WALK_LOG, [])  # moved, not copied
        t.choose_target({"goal_map": np.zeros((4, 4))}, {"step": 4})  # nothing new: no extra lines
        self.assertEqual(len(open(d / "walk_log.jsonl", encoding="utf-8").readlines()), 2)
        # HAZARD's own module name (src.HAZARD.policy.env_actions, challenge.py:11) is a different module object: drained too
        import sys, types
        fake = types.ModuleType("src.HAZARD.policy.env_actions")
        fake.WALK_LOG = [{"frame": 7, "dist_m": 1.0}]
        sys.modules["src.HAZARD.policy.env_actions"] = fake
        try:
            t.choose_target({"goal_map": np.zeros((4, 4))}, {"step": 5})
        finally:
            del sys.modules["src.HAZARD.policy.env_actions"]
        lines = [json.loads(l) for l in open(d / "walk_log.jsonl", encoding="utf-8")]
        self.assertEqual([(l["frame"], l["decision"]) for l in lines][-1], (7, 2))
        self.assertEqual(fake.WALK_LOG, [])


if __name__ == "__main__":
    unittest.main()
