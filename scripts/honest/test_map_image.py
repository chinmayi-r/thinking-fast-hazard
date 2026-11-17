"""Tests for the map picture (scripts/honest/map_image.py). Synthetic states, no simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest honest.test_map_image -v
"""
import os
import tempfile
import unittest

import numpy as np

from compiled.api import build_observation
from honest.map_image import (COL_FLOOR, COL_LOW, COL_PATH_BLOCKED, COL_PATH_UNSEEN, COL_ROBOT, COL_TALL, COL_TARGET,
                              COL_UNSEEN, crop_box, render_map)
from honest.test_honest_text import NAME_VALUES, TARGET_VALUES, make_input, make_state


def px(im, info, cell):
    """Colour of the pixel at the centre of a map cell."""
    r0, _, c0, _ = info["crop"]
    s = info["scale"]
    x, y = int((cell[1] - c0 + 0.5) * s), int((cell[0] - r0 + 0.5) * s)
    return im.getpixel((x, y))[:3]


class TestCrop(unittest.TestCase):
    def test_box_covers_explored_and_robot(self):
        ex = np.zeros((64, 64), dtype=np.int64)
        ex[20:30, 20:40] = 1
        self.assertEqual(crop_box(ex, agent=(50, 10)), (12, 59, 2, 48))
        self.assertEqual(crop_box(np.zeros((16, 16), dtype=np.int64)), (0, 16, 0, 16))  # nothing explored: the whole map


class TestRender(unittest.TestCase):
    def test_colours_and_file(self):
        st = make_state(unexplored=(0, 64, 20, 24), wall=(0, 40, 30))
        st["sem_map"]["height"][45:50, 5] = 0.3  # a counter
        obs = build_observation(st, make_input(), "fire", TARGET_VALUES)
        out = os.path.join(tempfile.mkdtemp(), "map.png")
        im, info = render_map(st, obs, out, px=320, frame=43)
        self.assertTrue(os.path.exists(out))
        with open(out, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
        self.assertLessEqual(max(im.size), 320)
        self.assertEqual(px(im, info, (10, 22)), COL_UNSEEN)  # the unseen band
        self.assertEqual(px(im, info, (10, 5)), COL_FLOOR)
        self.assertEqual(px(im, info, (10, 30)), COL_TALL)  # the wall
        self.assertEqual(px(im, info, (47, 5)), COL_LOW)  # the counter
        self.assertEqual(px(im, info, (32, 40)), COL_TARGET)  # the comb's cell
        self.assertEqual(px(im, info, (32, 10)), COL_ROBOT)  # the robot
        # the heading tick points to +col (right in the image) for heading 90
        x, y = (10 - info["crop"][2] + 0.5) * info["scale"], (32 - info["crop"][0] + 0.5) * info["scale"]
        self.assertEqual(im.getpixel((int(x + info["scale"] * 4), int(y)))[:3], COL_ROBOT)

    def test_paths_and_missing_layers(self):
        st = make_state()
        del st["sem_map"]["height"]  # the human tests' state has no height layer
        obs = build_observation(st, make_input(), "fire", TARGET_VALUES)
        im, info = render_map(st, obs, None, paths={"66": [(32, 10), (32, 40)]})
        self.assertGreater(im.size[0], 0)
        self.assertEqual(px(im, info, (10, 5)), COL_FLOOR)
        im2, _ = render_map(st, None, None)  # no observation: floor and objects only
        self.assertEqual(im2.size, im.size)

    def test_path_marks_by_class(self):
        st = make_state(unexplored=(0, 64, 20, 24))
        obs = build_observation(st, make_input(), "fire", TARGET_VALUES)
        marks = {"66": [((32, 15), "known"), ((32, 21), "unseen"), ((32, 35), "tall")]}
        im, info = render_map(st, obs, None, paths={"66": [(32, 10), (32, 40)]}, path_marks=marks)
        self.assertEqual(px(im, info, (32, 21)), COL_PATH_UNSEEN)
        self.assertEqual(px(im, info, (32, 35)), COL_PATH_BLOCKED)
        self.assertNotEqual(px(im, info, (32, 15)), COL_PATH_UNSEEN)  # known stretches keep the plain path colour

    def test_all_unexplored(self):
        st = make_state()
        st["sem_map"]["explored"][:] = 0
        obs = build_observation(st, make_input(), "fire", TARGET_VALUES)
        im, info = render_map(st, obs, None, px=128)
        self.assertEqual(info["crop"], (0, 64, 0, 64))
        self.assertEqual(px(im, info, (5, 5)), COL_UNSEEN)


if __name__ == "__main__":
    unittest.main()
