"""Tests for A53 (scripts/patch_A53_map_drop_oob.py applied to HAZARD/src/HAZARD/utils/model.py): a point whose cell falls
outside the map is clamped to the border as shipped, and dropped when HAZARD_MAP_DROP_OOB=1. Same synthetic point cloud
as test_map_clearance (PointCloud.get replaced). No simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest hazard_fixes.test_map_drop_oob -v
"""
import os
import unittest
from unittest import mock

import torch

from utils import model as hazard_model

FLAG = "HAZARD_MAP_DROP_OOB"


def points_with_one_far_object(far_z=99.5, oid=7):
    """A 1 x 2 point cloud: pixel 0 is floor inside the map at cell (0, 3); pixel 1 is a far-plane point (z = 99.5 m,
    way beyond a 32-cell map) that carries object id `oid` in the segmentation mask, as the silhouette pixels of a real
    object do (NOTES Nov 17 10:00: 760 of 760 pixels of hairbrush 64 at depth 99.51 m in the first frame of 3a-1)."""
    pts = torch.zeros((1, 3, 1, 2))
    pts[0, 2, 0, 0] = 0.32  # z inside cell 3
    pts[0, 2, 0, 1] = far_z
    pts[0, 1, 0, :] = 0.0  # floor height
    ids = torch.zeros((1, 2), dtype=torch.int64)
    ids[0, 1] = oid
    return pts, ids


class TestDropOutOfBounds(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(FLAG, None)
        self.sm = hazard_model.Semantic_Mapping(device=torch.device("cpu"), map_size_h=32, map_size_v=32, grid_size=0.1)
        self.sm.map_offset = [0, 0]
        n = 32
        self.last = {"explored": torch.zeros((n, n), dtype=torch.int64), "height": torch.zeros((n, n)),
                     "id": torch.zeros((n, n), dtype=torch.int64), "other": None}
        self.obs = torch.zeros((1, 4, 1, 2))

    def tearDown(self):
        os.environ.pop(FLAG, None)
        if self._saved is not None:
            os.environ[FLAG] = self._saved

    def run_forward(self, pts, ids):
        with mock.patch.object(hazard_model.PointCloud, "get", return_value=pts):
            out = self.sm.forward(obs=self.obs, id_map=ids, camera_matrix=torch.eye(4), maps_last=self.last,
                                  position=None, targets=None)
        return out["id"].numpy(), out["explored"].numpy()

    def test_flag(self):
        self.assertFalse(hazard_model._map_drop_oob())
        os.environ[FLAG] = "1"
        self.assertTrue(hazard_model._map_drop_oob())

    def test_shipped_clamps_the_far_point_to_the_border_with_its_id(self):
        pts, ids = points_with_one_far_object()
        id_map, explored = self.run_forward(pts, ids)
        self.assertEqual(int(id_map[0, 31]), 7)  # the far-plane pixel landed on the last column, carrying the object id
        self.assertEqual(int(explored[0, 31]), 1)
        self.assertEqual(int(explored[0, 3]), 1)  # the floor point is mapped either way

    def test_flag_drops_the_far_point_and_keeps_the_rest(self):
        os.environ[FLAG] = "1"
        pts, ids = points_with_one_far_object()
        id_map, explored = self.run_forward(pts, ids)
        self.assertEqual(int(id_map[0, 31]), 0)
        self.assertEqual(int(explored[0, 31]), 0)
        self.assertEqual(int(explored[0, 3]), 1)
        self.assertEqual(int(id_map.sum()), 0)  # the object has no cell at all until a pixel with real depth sees it

    def test_a_point_inside_the_map_is_untouched_by_the_flag(self):
        os.environ[FLAG] = "1"
        pts, ids = points_with_one_far_object(far_z=1.52, oid=9)  # the "far" pixel is now inside the map at cell 15
        id_map, explored = self.run_forward(pts, ids)
        self.assertEqual(int(id_map[0, 15]), 9)
        self.assertEqual(int(explored[0, 15]), 1)


if __name__ == "__main__":
    unittest.main()
