"""Tests for the A50 lookahead waypoint (scripts/patch_A50_walk_lookahead.py applied to HAZARD/src/HAZARD/policy/env_actions.py).

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest hazard_fixes.test_walk_lookahead -v
"""
import os
import unittest

from policy import env_actions  # HAZARD module, patched


class FakeEnv:
    class C:
        class S:
            grid_size = 0.1
        sem_map = S()
    controller = C()


class TestWalkLookahead(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("HAZARD_WALK_LOOKAHEAD", None)

    def tearDown(self):
        os.environ.pop("HAZARD_WALK_LOOKAHEAD", None)
        if self._saved is not None:
            os.environ["HAZARD_WALK_LOOKAHEAD"] = self._saved

    def test_flag_off_is_shipped_first_corner(self):
        self.assertEqual(env_actions._walk_lookahead_cells(FakeEnv()), 0.0)
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 11], [10, 40]], 0.0), [10, 11])
        os.environ["HAZARD_WALK_LOOKAHEAD"] = "garbage"
        self.assertEqual(env_actions._walk_lookahead_cells(FakeEnv()), 0.0)
        os.environ["HAZARD_WALK_LOOKAHEAD"] = "-1"
        self.assertEqual(env_actions._walk_lookahead_cells(FakeEnv()), 0.0)

    def test_metres_to_cells(self):
        os.environ["HAZARD_WALK_LOOKAHEAD"] = "1.0"
        self.assertAlmostEqual(env_actions._walk_lookahead_cells(FakeEnv()), 10.0)

    def test_straight_path_cut_at_the_lookahead(self):
        # one segment of 30 cells: the waypoint is 10 cells along it, not the far corner
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 40]], 10.0), [10, 20])

    def test_l_path_stops_at_the_first_corner(self):
        # 4 cells right, then 20 cells down: the waypoint is the corner, never a point the straight line to which cuts it
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 14], [30, 14]], 10.0), [10, 14])
        # a long first segment is cut at the lookahead even when corners follow
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 40], [30, 40]], 10.0), [10, 20])

    def test_short_first_segment_is_the_shipped_corner(self):
        # a first segment shorter than the lookahead ends at its corner (the shipped behaviour for that step)
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 12], [12, 12]], 10.0), [10, 12])
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 12]], 10.0), [10, 12])

    def test_diagonal_segment_rounds_to_cells(self):
        wp = env_actions._walk_waypoint([[0, 0], [30, 30]], 10.0)
        self.assertEqual(wp, [7, 7])  # 10 cells along the diagonal = 7.07 in each axis


if __name__ == "__main__":
    unittest.main()
