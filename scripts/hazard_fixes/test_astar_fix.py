"""Tests for the A38 planner fix (scripts/patch_A38_astar_fix.py applied to HAZARD/src/HAZARD/policy/astar.py).
Synthetic maps in HAZARD's own shape (sem_map['explored'], sem_map['height'], 0.1 m cells). No simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -m unittest hazard_fixes.test_astar_fix -v
"""
import os
import unittest

import numpy as np

from policy.astar import get_astar_path, get_astar_weight  # HAZARD module (src/HAZARD on PYTHONPATH)

FLAGS = ("HAZARD_ASTAR_FIX", "HAZARD_ASTAR_UNEXPLORED", "HAZARD_ASTAR_WALL_REACH")


def shipped_weight(sem_map):
    """The function as HAZARD ships it (astar.py at d35c71e, lines 36-58), for the flag-off equality test."""
    explored, height = sem_map["explored"], sem_map["height"]
    w, h = explored.shape
    weight = np.ones((w, h))
    weight[explored == 0] = 30
    weight[height > 0.5] += np.exp(height[height > 0.5] * 3) * 7
    weight[height > 0.1] += 10
    conv = np.zeros((w, h))
    for i in range(-2, 3):
        for j in range(-2, 3):
            conv[max(0, i): min(w, w + i), max(0, j): min(h, h + j)] += weight[max(0, -i): min(w, w - i), max(0, -j): min(h, h - j)] * 1.0 / (abs(i) + abs(j) + 1)
    return conv.astype(np.float32)


def room_with_wall(n=40, wall_col=20, door_rows=range(0, 8), ranged_rows=range(0, 26)):
    """A 4 m x 4 m explored floor split by a 2.5 m wall at `wall_col` with a door at `door_rows`. The wall is ranged
    (explored, height set) only on `ranged_rows`; the rest of the wall column is unexplored with no height, which is
    how a wall the depth camera never hit looks in HAZARD's map."""
    explored = np.ones((n, n), dtype=np.int64)
    height = np.zeros((n, n), dtype=np.float32)
    for r in range(n):
        if r in door_rows:
            continue
        if r in ranged_rows:
            height[r, wall_col] = 2.5
        else:
            explored[r, wall_col] = 0
    return {"explored": explored, "height": height}


def crossings(path, wall_col):
    """Rows at which the path steps across the wall column."""
    rows = []
    for a, b in zip(path[:-1], path[1:]):
        lo, hi = sorted((a[1], b[1]))
        if lo <= wall_col <= hi:
            rows.append(int(a[0]))
    return rows


class Base(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in FLAGS}
        for k in FLAGS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestFlagOff(Base):
    def test_weights_identical_to_the_shipped_function(self):
        rng = np.random.default_rng(1)
        sem = {"explored": (rng.random((48, 48)) > 0.3).astype(np.int64),
               "height": np.where(rng.random((48, 48)) > 0.8, rng.random((48, 48)) * 2.5, 0.0).astype(np.float32)}
        np.testing.assert_array_equal(get_astar_weight(sem, (0, 0), (47, 47)), shipped_weight(sem))

    def test_shipped_planner_goes_through_the_unranged_wall(self):
        sem = room_with_wall()
        w = get_astar_weight(sem, (36, 5), (36, 35))
        path = get_astar_path(w, (36, 5), (36, 35))
        rows = crossings(path, 20)
        self.assertTrue(rows and all(r >= 26 for r in rows), rows)  # crosses where the wall is unranged, not at the door


class TestFlagOn(Base):
    def setUp(self):
        super().setUp()
        os.environ["HAZARD_ASTAR_FIX"] = "1"

    def test_path_uses_the_door_instead_of_the_unranged_wall(self):
        sem = room_with_wall()
        w = get_astar_weight(sem, (36, 5), (36, 35))
        path = get_astar_path(w, (36, 5), (36, 35))
        rows = crossings(path, 20)
        self.assertTrue(rows and all(r < 8 for r in rows), rows)  # every crossing is inside the door rows 0..7

    def test_unranged_wall_next_to_a_ranged_part_is_priced_as_wall(self):
        os.environ["HAZARD_ASTAR_UNEXPLORED"] = "30"  # isolate the wall extension from the unexplored cost
        sem = room_with_wall()
        w = get_astar_weight(sem, (36, 5), (36, 35))
        self.assertGreater(w[27, 20], 5000)  # 2 cells beyond the last ranged wall cell (row 25): extended
        self.assertLess(w[35, 20], 200)  # far from any ranged part: plain unexplored, as shipped

    def test_unexplored_cost_defaults_to_300_and_is_configurable(self):
        sem = {"explored": np.zeros((10, 10), dtype=np.int64), "height": np.zeros((10, 10), dtype=np.float32)}
        self.assertAlmostEqual(float(get_astar_weight(sem, (0, 0), (9, 9))[5, 5]), 300 * (1 + 4 * 0.5 + 4 / 3 + 4 / 3 + 8 / 4 + 4 / 5), places=2)
        os.environ["HAZARD_ASTAR_UNEXPLORED"] = "50"
        self.assertAlmostEqual(float(get_astar_weight(sem, (0, 0), (9, 9))[5, 5]), 50 * (1 + 4 * 0.5 + 4 / 3 + 4 / 3 + 8 / 4 + 4 / 5), places=2)

    def test_all_unexplored_map_is_still_plannable(self):
        sem = {"explored": np.zeros((40, 40), dtype=np.int64), "height": np.zeros((40, 40), dtype=np.float32)}
        path = get_astar_path(get_astar_weight(sem, (2, 2), (30, 30)), (2, 2), (30, 30))
        self.assertIsNotNone(path)
        self.assertEqual((tuple(path[0]), tuple(path[-1])), ((2, 2), (30, 30)))

    def test_a_sixty_centimetre_gap_between_ranged_counters_stays_open(self):
        n = 40
        explored = np.ones((n, n), dtype=np.int64)
        height = np.zeros((n, n), dtype=np.float32)
        height[0:17, 20] = 0.9  # counter, ranged
        height[23:40, 20] = 0.9  # counter, ranged; rows 17..22 are a 6-cell (0.6 m) gap
        explored[17:23, 20] = 0  # the gap itself was never ranged
        sem = {"explored": explored, "height": height}
        w = get_astar_weight(sem, (10, 5), (10, 35))
        self.assertGreater(w[16, 20], 10000)  # A38 (c): a counter is impassable, not the shipped ~114
        self.assertGreater(w[17, 20], 10000)  # A38 (a): the gap edge within 2 cells of the counter continues it
        self.assertLess(w[19, 20], w[17, 20])  # the middle of the gap is the cheapest way through
        self.assertLess(w[20, 20], w[22, 20])
        path = get_astar_path(w, (10, 5), (10, 35))
        self.assertTrue(path is not None and all(18 <= r <= 21 for r in crossings(path, 20)), crossings(path, 20))

    def test_shipped_planner_would_rather_cross_a_counter_than_unexplored_floor(self):
        # the defect behind A38 (c): with the shipped curve a 0.9 m counter (about 114) is cheaper than 300 of unexplored
        os.environ.pop("HAZARD_ASTAR_FIX")
        os.environ["HAZARD_ASTAR_UNEXPLORED"] = "300"  # the cost knob alone, without (a) and (c)
        explored = np.ones((40, 40), dtype=np.int64)
        height = np.zeros((40, 40), dtype=np.float32)
        height[:, 20] = 0.9
        explored[17:23, 20] = 0
        w = get_astar_weight({"explored": explored, "height": height}, (10, 5), (10, 35))
        self.assertLess(w[10, 20], w[19, 20])  # a counter cell (about 313 smoothed) is cheaper than the unexplored gap
        path = get_astar_path(w, (10, 5), (10, 35))
        self.assertFalse(any(17 <= r <= 22 for r in crossings(path, 20)))  # so the path climbs the counter instead


if __name__ == "__main__":
    unittest.main()
