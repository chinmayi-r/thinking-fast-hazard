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
        self._saved_mode = os.environ.pop("HAZARD_WALK_LOOKAHEAD_MODE", None)

    def tearDown(self):
        os.environ.pop("HAZARD_WALK_LOOKAHEAD", None)
        os.environ.pop("HAZARD_WALK_LOOKAHEAD_MODE", None)
        if self._saved is not None:
            os.environ["HAZARD_WALK_LOOKAHEAD"] = self._saved
        if self._saved_mode is not None:
            os.environ["HAZARD_WALK_LOOKAHEAD_MODE"] = self._saved_mode

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


def _room(w=50, h=50):
    import numpy as np
    return np.zeros((w, h))


class TestWalkLookaheadLOS(unittest.TestCase):
    """v3: HAZARD_WALK_LOOKAHEAD_MODE=los, string pulling over the planned path against the height map."""

    def setUp(self):
        os.environ["HAZARD_WALK_LOOKAHEAD_MODE"] = "los"

    def tearDown(self):
        os.environ.pop("HAZARD_WALK_LOOKAHEAD_MODE", None)

    def test_mode_default_is_seg(self):
        os.environ.pop("HAZARD_WALK_LOOKAHEAD_MODE", None)
        self.assertEqual(env_actions._walk_lookahead_mode(), "seg")
        # seg mode with a height map still cuts the first segment, never jumps a corner
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 12], [20, 12]], 10.0, _room()), [10, 12])

    def test_open_room_jumps_the_corners(self):
        # staircase path of 2-cell segments across an empty room: the waypoint is the farthest cell within 10 of the start
        path = [[10, 10], [10, 12], [12, 12], [12, 14], [14, 14], [14, 16], [16, 16], [16, 18], [18, 18], [18, 30]]
        wp = env_actions._walk_waypoint(path, 10.0, _room())
        self.assertEqual(wp, [16, 18])  # |(16,18)-(10,10)| = 10.0, inclusive; (18,18) is 11.3, out

    def test_lookahead_radius_is_inclusive(self):
        path = [[10, 10], [10, 20], [10, 30]]
        self.assertEqual(env_actions._walk_waypoint(path, 10.0, _room()), [10, 20])
        self.assertEqual(env_actions._walk_waypoint(path, 9.0, _room()), [10, 20])  # no cell within 9: falls back to path[1]

    def test_counter_blocks_the_shortcut(self):
        # L-path round a counter: start (10,10), corner (10,16), end (16,16); counter cells (12..16, 11..15) are 0.9 m tall
        hm = _room()
        hm[12:17, 11:16] = 0.9
        path = [[10, 10], [10, 16], [16, 16]]
        self.assertEqual(env_actions._walk_waypoint(path, 10.0, hm), [10, 16])
        # with the counter removed the straight line to the end (8.5 cells) is clear
        self.assertEqual(env_actions._walk_waypoint(path, 10.0, _room()), [16, 16])

    def test_low_object_is_passable(self):
        hm = _room()
        hm[12:17, 11:16] = 0.3  # a rug, below the planner's 0.5 m obstacle rule
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 16], [16, 16]], 10.0, hm), [16, 16])

    def test_no_height_map_falls_back_to_seg(self):
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 12], [20, 12]], 10.0, None), [10, 12])

    def test_flag_off_is_shipped(self):
        self.assertEqual(env_actions._walk_waypoint([[10, 10], [10, 12], [20, 12]], 0.0, _room()), [10, 12])

    def test_line_off_the_map_is_not_clear(self):
        self.assertFalse(env_actions._walk_los_clear(_room(20, 20), [1, 1], [1, 25]))
        self.assertTrue(env_actions._walk_los_clear(_room(20, 20), [1, 1], [1, 19]))


if __name__ == "__main__":
    unittest.main()
