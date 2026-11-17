"""Tests for scripts/hazard_fixes/save_maps.py (A43 verification hook). No simulator."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hazard_fixes.save_maps import save_decision_maps

RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_baseline.py")


class FakeController:
    class _Dyn:
        camera_matrix = np.arange(16, dtype=np.float64)
        class transform:
            position = np.array([1.0, 0.0, 2.0])
    class _Agent:
        dynamic = None
        def get_facing(self):
            return 1.5
    class _SemMap:
        map_offset = [256, 256]
        grid_size = 0.1

    def __init__(self):
        a = self._Agent(); a.dynamic = self._Dyn()
        self.agents = [a]
        self.sem_map = self._SemMap()


class TestSaveMaps(unittest.TestCase):
    def test_arrays_land_in_one_npz(self):
        d = tempfile.mkdtemp()
        state = {"sem_map": {"explored": np.ones((8, 8), dtype=np.int64), "height": np.full((8, 8), 2.4, dtype=np.float32),
                             "id": np.zeros((8, 8), dtype=np.int64), "other": None},
                 "goal_map": np.zeros((8, 8)), "raw": {"depth": np.ones((4, 4), dtype=np.float32), "seg_mask": np.zeros((4, 4), dtype=np.int64)}}
        path = save_decision_maps(state, os.path.join(d, "maps"), 1, 2, 120, controller=FakeController())
        self.assertTrue(path.endswith(os.path.join("maps", "ep1_dec2_frame120.npz")))
        z = np.load(path)
        self.assertEqual(sorted(z.files), ["agent_facing_rad", "agent_position", "camera_matrix", "depth", "explored", "goal_map",
                                           "grid_size", "height", "id", "map_offset", "seg_mask"])
        self.assertEqual(float(z["height"].max()), np.float32(2.4))
        self.assertEqual(list(z["agent_position"]), [1.0, 0.0, 2.0])
        self.assertEqual(float(z["agent_facing_rad"][0]), 1.5)
        self.assertEqual(list(z["map_offset"]), [256, 256])

    def test_without_controller_and_missing_layers(self):
        d = tempfile.mkdtemp()
        state = {"sem_map": {"explored": np.ones((4, 4), dtype=np.int64), "id": np.zeros((4, 4), dtype=np.int64)}, "raw": {}}
        z = np.load(save_decision_maps(state, d, 1, 1, 43))
        self.assertEqual(sorted(z.files), ["explored", "id"])

    def test_runner_flag(self):
        out = subprocess.run([sys.executable, RUNNER, "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        self.assertIn("--save_maps", out.stdout)


if __name__ == "__main__":
    unittest.main()
