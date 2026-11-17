"""Tests for the A44 mapper fix (scripts/patch_A44_map_clearance.py applied to HAZARD/src/HAZARD/utils/model.py).
A synthetic point cloud stands in for the depth camera (PointCloud.get is replaced), so the per-cell maximum logic of
SemanticMap.forward runs as shipped. No simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -m unittest hazard_fixes.test_map_clearance -v
"""
import os
import unittest
from unittest import mock

import numpy as np
import torch

from utils import model as hazard_model  # HAZARD module (src/HAZARD on PYTHONPATH)

FLAG = "HAZARD_MAP_MAX_OBSTACLE_H"


def fake_points(n_cells=10, floor_y=0.0, overhead_y=2.4, counter_y=0.9, counter_cells=(3, 4)):
    """A (1, 3, H, W) point cloud: row 0 = floor points over n_cells cells along z, row 1 = overhead points (ceiling or
    wall cabinet) over the same cells, row 2 = counter-top points over `counter_cells`."""
    H, W = 3, n_cells
    pts = torch.zeros((1, 3, H, W))
    for j in range(W):
        pts[0, 0, :, j] = 0.0  # x: one row of cells at x = 0
        pts[0, 2, :, j] = 0.1 * j + 0.02  # z: inside cell j ((z + 0.05) // 0.1 = j, model.py:275)
    pts[0, 1, 0, :] = floor_y
    pts[0, 1, 1, :] = overhead_y
    pts[0, 1, 2, :] = -1.0  # row 2 unused unless a counter cell
    for j in counter_cells:
        pts[0, 1, 2, j] = counter_y
    return pts


class Base(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(FLAG, None)
        self.sm = hazard_model.Semantic_Mapping(device=torch.device("cpu"), map_size_h=32, map_size_v=32, grid_size=0.1)
        self.sm.map_offset = [0, 0]
        n = 32
        self.last = {"explored": torch.zeros((n, n), dtype=torch.int64), "height": torch.zeros((n, n)),
                     "id": torch.zeros((n, n), dtype=torch.int64), "other": None}
        self.obs = torch.zeros((1, 4, 3, 10))  # rgb + depth, unused once PointCloud.get is replaced
        self.id_map = torch.zeros((3, 10), dtype=torch.int64)

    def tearDown(self):
        os.environ.pop(FLAG, None)
        if self._saved is not None:
            os.environ[FLAG] = self._saved

    def run_forward(self, pts):
        with mock.patch.object(hazard_model.PointCloud, "get", return_value=pts):
            out = self.sm.forward(obs=self.obs, id_map=self.id_map, camera_matrix=torch.eye(4), maps_last=self.last,
                                  position=None, targets=None)
        return out["height"].numpy(), out["explored"].numpy()


class TestClearance(Base):
    def test_shipped_keeps_the_overhead_points_as_floor_obstacles(self):
        h, ex = self.run_forward(fake_points())
        self.assertEqual(ex[0, :10].tolist(), [1] * 10)
        self.assertTrue(np.allclose(h[0, :10], 2.4))  # every floor cell reads 2.4 m: the A43 artifact

    def test_cut_drops_overhead_points_and_keeps_counters(self):
        os.environ[FLAG] = "1.8"
        h, ex = self.run_forward(fake_points())
        self.assertEqual(ex[0, :10].tolist(), [1] * 10)  # the cells are still explored (the floor points remain)
        expect = [0.0] * 10
        expect[3] = expect[4] = 0.9
        self.assertTrue(np.allclose(h[0, :10], expect), h[0, :10])

    def test_cut_value_is_the_flag(self):
        os.environ[FLAG] = "2.45"
        h, _ = self.run_forward(fake_points(overhead_y=2.4))
        self.assertTrue(np.allclose(h[0, :10][[0, 1, 2, 5]], 2.4))  # 2.4 < 2.45: kept
        os.environ[FLAG] = "2.3"
        h, _ = self.run_forward(fake_points(overhead_y=2.4))
        self.assertTrue(np.allclose(h[0, :10][[0, 1, 2, 5]], 0.0))

    def test_shipped_cut_is_two_point_five(self):
        self.assertEqual(hazard_model._map_max_obstacle_height(), 2.5)
        h, _ = self.run_forward(fake_points(overhead_y=2.6))
        self.assertTrue(np.allclose(h[0, :10][[0, 1, 2]], 0.0))  # above the shipped cut: dropped as shipped


if __name__ == "__main__":
    unittest.main()
