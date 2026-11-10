"""Tests for the referee and the background runner with made-up inputs; no simulator, no API.
Run from G:\\iw-thinking-fast:  PYTHONPATH=scripts venv/Scripts/python.exe -m unittest scripts.rrara.test_rrara -v"""
import time
import unittest

from rrara.background import BackgroundJob
from rrara.reconcile import decide, legal_actions, same_action

PI = {"step": 300, "holding_objects": [], "action_result": False, "action_info": "",
      "nearest_object": [{"name": "hairbrush", "category": "hairbrush", "id": "64"}],
      "explored_object_name_list": [{"name": "hairbrush", "category": "hairbrush", "id": "64"},
                                    {"name": "bag", "category": "bag", "id": "71"},
                                    {"name": "table", "category": "table", "id": "9"}]}
TARGETS = ["hairbrush", "bag"]


class TestLegal(unittest.TestCase):
    def test_not_holding(self):
        acts = legal_actions(PI, TARGETS)
        self.assertIn(("walk_to", "64"), acts)
        self.assertIn(("walk_to", "71"), acts)
        self.assertNotIn(("walk_to", "9"), acts)  # the table is not a target
        self.assertIn(("pick_up", "64"), acts)
        self.assertIn(("explore", None), acts)
        self.assertNotIn(("drop", None), acts)

    def test_holding(self):
        pi = dict(PI, holding_objects=[{"id": "64", "category": "hairbrush"}])
        acts = legal_actions(pi, TARGETS)
        self.assertEqual(acts, [("explore", None), ("drop", None)])


class TestDecide(unittest.TestCase):
    legal = legal_actions(PI, TARGETS)

    def test_no_verdict_means_reflex(self):
        a, rec = decide(("walk_to", "64"), None, self.legal)
        self.assertEqual(a, ("walk_to", "64"))
        self.assertFalse(rec["override"])

    def test_valid_means_reflex(self):
        v = {"valid": True, "replacement": None, "wall_s": 1.0}
        a, rec = decide(("walk_to", "64"), v, self.legal, mode="faithful")
        self.assertEqual(a, ("walk_to", "64"))
        self.assertEqual(rec["path"], "reflex (verdict valid)")

    def test_invalid_overrides_when_legal(self):
        v = {"valid": False, "replacement": ["walk_to", "71"], "wall_s": 1.0}
        a, rec = decide(("walk_to", "64"), v, self.legal, mode="faithful")
        self.assertEqual(a, ("walk_to", "71"))
        self.assertTrue(rec["override"])

    def test_invalid_but_illegal_replacement(self):
        v = {"valid": False, "replacement": ["drop", None], "wall_s": 1.0}  # not holding anything
        a, rec = decide(("walk_to", "64"), v, self.legal, mode="faithful")
        self.assertEqual(a, ("walk_to", "64"))
        self.assertEqual(rec["path"], "reflex (replacement not legal now)")

    def test_worldtime_late(self):
        v = {"valid": False, "replacement": ["walk_to", "71"], "wall_s": 3.0}  # 90 frames needed
        a, rec = decide(("walk_to", "64"), v, self.legal, mode="worldtime", action_frames=80)
        self.assertEqual(a, ("walk_to", "64"))
        self.assertEqual(rec["path"], "reflex (verdict late)")
        self.assertEqual(rec["frames_needed"], 90)
        self.assertFalse(rec["in_time"])

    def test_worldtime_in_time(self):
        v = {"valid": False, "replacement": ["walk_to", "71"], "wall_s": 2.0}  # 60 frames needed
        a, rec = decide(("walk_to", "64"), v, self.legal, mode="worldtime", action_frames=80)
        self.assertEqual(a, ("walk_to", "71"))
        self.assertTrue(rec["in_time"])

    def test_faithful_ignores_lateness(self):
        v = {"valid": False, "replacement": ["walk_to", "71"], "wall_s": 30.0}
        a, rec = decide(("walk_to", "64"), v, self.legal, mode="faithful", action_frames=1)
        self.assertEqual(a, ("walk_to", "71"))

    def test_reflector_error(self):
        a, rec = decide(("walk_to", "64"), {"error": "boom"}, self.legal)
        self.assertEqual(a, ("walk_to", "64"))
        self.assertEqual(rec["path"], "reflex (reflector error)")

    def test_same_action_str_int(self):
        self.assertTrue(same_action(("walk_to", 64), ("walk_to", "64")))
        self.assertFalse(same_action(("walk_to", 64), ("pick_up", "64")))


class TestBackground(unittest.TestCase):
    def test_never_blocks(self):
        job = BackgroundJob()
        job.start(lambda: (time.sleep(0.3), "done")[1], tag=7)
        self.assertTrue(job.pending())
        self.assertIsNone(job.collect())  # not finished yet: returns at once with nothing
        time.sleep(0.5)
        res, wall, tag = job.collect()
        self.assertEqual((res, tag), ("done", 7))
        self.assertGreaterEqual(wall, 0.3)
        self.assertFalse(job.pending())
        job.shutdown()

    def test_wall_is_time_to_finish_not_time_to_collect(self):
        # Nov 10 regression: the agent collects at the NEXT decision, after the simulator played the whole action out;
        # the verdict's wall must be how long it took to exist, not how long it sat waiting to be read.
        job = BackgroundJob()
        job.start(lambda: (time.sleep(0.2), "done")[1])
        time.sleep(0.8)
        res, wall, tag = job.collect()
        self.assertEqual(res, "done")
        self.assertGreaterEqual(wall, 0.2)
        self.assertLess(wall, 0.6)
        job.shutdown()

    def test_error_is_reported_not_raised(self):
        job = BackgroundJob()
        job.start(lambda: 1 / 0)
        time.sleep(0.1)
        res, wall, tag = job.collect()
        self.assertIn("error", res)
        job.shutdown()


if __name__ == "__main__":
    unittest.main()
