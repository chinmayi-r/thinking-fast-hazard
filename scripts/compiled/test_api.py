"""Tests for the program API (scripts/compiled/api.py, API.md) and the worked example (examples/reflex_program.py).
Synthetic maps in HAZARD's own shape plus processed_input dicts copied from the Nov 16 Jev episode on mm_kitchen_3a-1.
No simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest compiled.test_api -v
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from compiled import api
from compiled.api import (ProgramAgent, Paths, audit_program, build_observation, check_action, legal_actions,
                          load_program, normalize_action)
from rrara.reflex import PaperReflex

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "examples", "reflex_program.py")
RUNNER = os.path.join(os.path.dirname(HERE), "run_baseline.py")

# ---- fixtures: objects_info as challenge.py:100-108 hands it over, processed_input entries as challenge.py:144-148 ----
GOALS = ["hairbrush", "purse"]
OBJECTS_INFO = {"hairbrush": {"value": 1, "inflammable": 1}, "purse": {"value": 5, "inflammable": 1}}
PI_FIRST = {  # decision 1 of the Jev episode (input43.json): nothing held, nowhere, action_result False with empty info
    "explored_object_name_list": [
        {"name": "cabinet_36_two_door_wood_oak_white_composite", "category": "cabinet", "id": "1"},
        {"name": "b04_comb", "category": "hairbrush", "id": "66"},
        {"name": "b05_comb", "category": "hairbrush", "id": "64"},
        {"name": "bag_3", "category": "purse", "id": "56"},
    ],
    "holding_objects": [], "nearest_object": [], "step": 43, "action_result": False, "action_info": "", "save_dir": "x"}
PI_ARRIVED = dict(PI_FIRST, step=257, action_result=True, action_info="success",
                  nearest_object=[{"name": "b04_comb", "category": "hairbrush", "id": "66"}])
PI_HOLDING = dict(PI_FIRST, step=299, action_result=True, action_info="success, reach for success",
                  holding_objects=[{"name": "b04_comb", "category": "hairbrush", "id": "66"}], nearest_object=[])
PI_FAILED = dict(PI_FIRST, step=145, action_result=False, action_info="max steps reached")


def scene(n=64, agent=(50, 8), facing=(50, 9), objects=None, walls=(), unexplored_cols=()):
    """A HAZARD-shaped state: sem_map explored/height/id arrays and a goal_map with the robot (-2) and facing (-1) marks.
    `objects` = {id: (r, c)} painted as 2 x 2 explored blocks; `walls` = (r0, r1, col, height) explored segments."""
    explored = np.ones((n, n), dtype=np.int64)
    height = np.zeros((n, n), dtype=np.float32)
    ids = np.zeros((n, n), dtype=np.int64)
    for oid, (r, c) in (objects or {}).items():
        ids[r:r + 2, c:c + 2] = int(oid)
    for r0, r1, col, h in walls:
        height[r0:r1, col] = h
    for col in unexplored_cols:
        explored[:, col] = 0
    goal = np.zeros((n, n))
    if agent is not None:
        goal[agent] = -2
    if facing is not None:
        goal[facing] = -1
    return {"sem_map": {"explored": explored, "height": height, "id": ids, "other": None}, "goal_map": goal,
            "raw": {}, "RL": {}}


class TestObservation(unittest.TestCase):
    def test_fields_types_and_sources(self):
        st = scene(objects={66: (50, 40), 64: (10, 40), 56: (30, 30)})
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertEqual(tuple(obs), api.OBS_KEYS)
        self.assertEqual((obs["task"], obs["frame"], obs["frames_left"]), ("fire", 43, 1457))
        self.assertEqual((obs["agent_cell"], obs["facing_cell"]), ((50, 8), (50, 9)))
        self.assertEqual(obs["holding"], [])
        self.assertIsNone(obs["at_object"])
        self.assertIsNone(obs["last_action"])
        self.assertIsNone(obs["last_result"])  # first decision: HAZARD's False/"" is not a result
        by_id = {o["id"]: o for o in obs["objects"]}
        self.assertEqual(set(by_id), {"1", "66", "64", "56"})
        self.assertEqual((by_id["56"]["is_target"], by_id["56"]["value"], by_id["56"]["category"]), (True, 5.0, "purse"))
        self.assertEqual((by_id["1"]["is_target"], by_id["1"]["value"], by_id["1"]["cell"]), (False, None, None))
        self.assertEqual(by_id["66"]["cell"], (50, 40))  # mean of the 2x2 block rounds to its top-left here
        self.assertAlmostEqual(by_id["66"]["straight_cells"], 32.0)
        self.assertIsNone(by_id["1"]["straight_cells"])
        for k in ("explored", "height", "id"):
            self.assertIs(obs["map"][k], st["sem_map"][k])  # the robot's own arrays, no copy, nothing else

    def test_object_cell_uses_only_explored_cells(self):
        st = scene(objects={66: (50, 40)})
        st["sem_map"]["explored"][50:52, 40:42] = 0  # on the id layer but never explored: not on the robot's map
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertIsNone({o["id"]: o for o in obs["objects"]}["66"]["cell"])

    def test_at_object_holding_last_result(self):
        st = scene(objects={66: (50, 10)})
        obs = build_observation(st, PI_ARRIVED, "fire", {"hairbrush": 1.0, "purse": 5.0}, last_action=("walk_to", "66"))
        self.assertEqual((obs["at_object"]["id"], obs["at_object"]["is_target"], obs["at_object"]["value"]), ("66", True, 1.0))
        self.assertEqual(obs["last_action"], ("walk_to", "66"))
        self.assertEqual(obs["last_result"], {"ok": True, "info": "success"})
        obs = build_observation(st, PI_HOLDING, "fire", {"hairbrush": 1.0, "purse": 5.0}, last_action=("pick_up", 66))
        self.assertEqual(obs["holding"][0]["id"], "66")
        self.assertEqual(obs["last_action"], ("pick_up", "66"))  # ids normalized to strings
        obs = build_observation(st, PI_FAILED, "fire", {"hairbrush": 1.0, "purse": 5.0}, last_action=("walk_to", "64"))
        self.assertEqual(obs["last_result"], {"ok": False, "info": "max steps reached"})

    def test_robot_off_the_map_edge(self):
        st = scene(agent=None, facing=None, objects={66: (50, 40)})  # fireagent_controller.py:557 leaves no -2 mark
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertIsNone(obs["agent_cell"])
        self.assertIsNone(obs["facing_cell"])
        self.assertIsNone(obs["objects"][1]["straight_cells"])

    def test_heading_is_the_mirror_of_the_facing_mark(self):
        """A52: HAZARD's facing mark is 180 - the true heading (measured in TDW, scripts/hazard_fixes/facing_check.py)."""
        from compiled.api import heading_deg, bearing_deg
        self.assertEqual(heading_deg((32, 10), (32, 11)), 90.0)   # mark +col: unchanged (90 is its own mirror)
        self.assertEqual(heading_deg((32, 10), (32, 9)), -90.0)   # mark -col: unchanged
        self.assertEqual(heading_deg((32, 10), (33, 10)), 180.0)  # mark +row (HAZARD says 0): the body faces -row
        self.assertEqual(heading_deg((32, 10), (31, 10)), 0.0)    # mark -row (HAZARD says 180): the body faces +row
        self.assertEqual(heading_deg((32, 10), (33, 11)), 135.0)  # mark +45: true 135
        # a body facing +row (mark at -row) has an object at a larger row straight ahead, one at a larger col on its left
        self.assertEqual(bearing_deg((32, 10), heading_deg((32, 10), (31, 10)), (40, 10)), 0.0)
        self.assertEqual(bearing_deg((32, 10), heading_deg((32, 10), (31, 10)), (32, 20)), 90.0)
        self.assertEqual(bearing_deg((32, 10), heading_deg((32, 10), (31, 10)), (24, 10)), 180.0)

    def test_heading_range_and_bearing(self):
        # heading +col (facing (50, 9) from (50, 8)) = 90 deg in the body's arctan2(z, x) convention
        st = scene(objects={66: (50, 40), 64: (10, 8), 56: (90, 8) if False else (60, 8)})
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertEqual(obs["heading_deg"], 90.0)
        by_id = {o["id"]: o for o in obs["objects"]}
        self.assertEqual((by_id["66"]["range_m"], by_id["66"]["bearing_deg"]), (3.2, 0.0))  # straight ahead
        self.assertEqual(by_id["64"]["bearing_deg"], 90.0)  # smaller row = world -x = the robot's left (A41)
        self.assertEqual(by_id["56"]["bearing_deg"], -90.0)  # larger row = right
        self.assertEqual(by_id["1"]["bearing_deg"], None)
        self.assertEqual(api.wrap_deg(270.0), -90.0)
        self.assertEqual(api.wrap_deg(-180.0), 180.0)
        st = scene(agent=(50, 8), facing=(49, 7), objects={66: (50, 40)})  # diagonal mark at -135: true heading -45 (A52)
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertEqual(obs["heading_deg"], -45.0)
        self.assertEqual({o["id"]: o for o in obs["objects"]}["66"]["bearing_deg"], 135.0)  # +col is behind-left of -45

    def test_object_cells_one_pass_equals_per_object(self):
        rng = np.random.default_rng(3)
        ids = np.zeros((128, 128), dtype=np.int64)
        for i in range(30):
            r, c = int(rng.integers(0, 124)), int(rng.integers(0, 124))
            ids[r:r + int(rng.integers(1, 5)), c:c + int(rng.integers(1, 5))] = 100 + i
        explored = (rng.random((128, 128)) > 0.3).astype(np.int64)
        sem = {"explored": explored, "id": ids}
        cells = api.object_cells(sem)
        present = {int(v) for v in np.unique(explored * ids) if v != 0}
        self.assertEqual(set(cells), present)
        for oid in present:
            self.assertEqual(cells[oid], api.object_cell(sem, oid))
        self.assertEqual(api.object_cells({"explored": np.zeros((4, 4), dtype=np.int64), "id": ids[:4, :4]}), {})

    def test_cropped_planning_matches_the_full_map(self):
        # a 512 map explored only in the middle, as a kitchen is; the crop (explored area + 4 m) must give the same path
        n = 512
        explored = np.zeros((n, n), dtype=np.int64)
        explored[200:320, 200:330] = 1
        height = np.zeros((n, n), dtype=np.float32)
        height[200:300, 260] = 2.5  # a wall with a gap at rows 300-319
        ids = np.zeros((n, n), dtype=np.int64)
        ids[240:242, 310:312] = 66
        goal = np.zeros((n, n)); goal[240, 210] = -2; goal[240, 211] = -1
        st = {"sem_map": {"explored": explored, "height": height, "id": ids}, "goal_map": goal}
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        full = Paths(obs, coarse=4, crop=False)
        cropped = Paths(obs, coarse=4, crop=True)
        self.assertEqual(cropped.path("66"), full.path("66"))
        self.assertEqual(cropped.cells("66"), full.cells("66"))
        self.assertLess(cropped._weight.shape[0], full._weight.shape[0])  # the crop is smaller than 128 pooled rows
        self.assertEqual(cropped._origin, (160, 160))  # explored box minus 40 cells, aligned to the pooling
        self.assertEqual(api.crop_box(np.zeros((8, 8), dtype=np.int64), []), (0, 8, 0, 8))

    def test_path_points_are_full_map_cells(self):
        st = scene(objects={66: (50, 40)})
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        path = Paths(obs, coarse=4).path("66")
        self.assertEqual(path[0], (48, 8))  # the robot's pooled cell, scaled back
        self.assertEqual(path[-1], (48, 40))
        self.assertTrue(all((r % 4 == 0 and c % 4 == 0) for r, c in path))
        self.assertEqual(len(path), 2)  # a straight run: HAZARD's A* keeps the corner points only (astar.py:16-34, path_cleanup)
        self.assertIsNone(Paths(obs).path("99"))

    def test_wind_episode_length(self):
        obs = build_observation(scene(), dict(PI_FIRST, step=100), "wind", {})
        self.assertEqual(obs["frames_left"], 2900)


class TestPaths(unittest.TestCase):
    def test_open_floor_and_walled_off(self):
        st = scene(objects={66: (50, 40), 64: (10, 40)}, walls=[(0, 64, 30, 2.5)])  # a full wall between robot and both
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        p = Paths(obs, coarse=4)
        # HAZARD's planner never refuses: a 2.5 m wall costs about 12,000 per cell (astar.py:41-42), not infinity, so the
        # path goes straight through it and the length is the straight line. That is the A38 finding; the program sees
        # the same weights the body walks by.
        self.assertEqual(p.cells("66"), 32.0)
        st = scene(objects={66: (50, 40), 64: (10, 40)}, walls=[(0, 40, 30, 2.5)])  # the wall stops at row 40
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        p = Paths(obs, coarse=4)
        d66, d64 = p.cells("66"), p.cells("64")
        self.assertGreaterEqual(d66, 32.0)  # at least the straight line, in full cells
        self.assertGreater(d64, d66)  # 64 is behind the wall: round the end of it, not through it
        self.assertIsNone(p.cells("99"))  # unknown id
        self.assertIsNone(p.cells("1"))  # seen object without a cell

    def test_same_cell_is_zero_and_no_robot_is_none(self):
        st = scene(objects={66: (50, 8)})
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertEqual(Paths(obs).cells("66"), 0.0)
        st = scene(agent=None, objects={66: (50, 40)})
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        self.assertIsNone(Paths(obs).cells("66"))

    def test_weight_is_built_once_per_observation(self):
        st = scene(objects={66: (50, 40), 64: (10, 40)})
        obs = build_observation(st, PI_FIRST, "fire", {"hairbrush": 1.0, "purse": 5.0})
        p = Paths(obs, coarse=4)
        p.cells("66")
        w = p._weight
        p.cells("64")
        self.assertIs(p._weight, w)
        self.assertEqual(w.shape, (16, 16))


class TestActions(unittest.TestCase):
    def obs(self, pi, **kw):
        return build_observation(scene(objects={66: (50, 40), 64: (10, 40), 56: (30, 30)}), pi, "fire",
                                 {"hairbrush": 1.0, "purse": 5.0}, **kw)

    def test_menu_matches_hazards_agents(self):
        self.assertEqual(legal_actions(self.obs(PI_FIRST)),
                         [("explore", None), ("walk_to", "66"), ("walk_to", "64"), ("walk_to", "56")])
        self.assertEqual(legal_actions(self.obs(PI_ARRIVED)),
                         [("explore", None), ("pick_up", "66"), ("walk_to", "66"), ("walk_to", "64"), ("walk_to", "56")])
        self.assertEqual(legal_actions(self.obs(PI_HOLDING)), [("explore", None), ("drop", None)])
        wind = build_observation(scene(objects={66: (50, 40)}), PI_HOLDING, "wind", {"hairbrush": 1.0})
        self.assertEqual(legal_actions(wind), [("explore", None)])  # drop into the bag is fire/flood only

    def test_check_action(self):
        o = self.obs(PI_FIRST)
        self.assertEqual(check_action(("walk_to", "66"), o), (True, "ok"))
        self.assertEqual(check_action(("walk_to", 66), o), (True, "ok"))  # int id accepted
        self.assertEqual(check_action(("explore", None), o), (True, "ok"))
        self.assertFalse(check_action(("walk_to", "1"), o)[0])  # the cabinet is not a target
        self.assertFalse(check_action(("walk_to", "99"), o)[0])
        self.assertFalse(check_action(("pick_up", "66"), o)[0])  # not standing at it
        self.assertFalse(check_action(("drop", None), o)[0])  # holding nothing
        self.assertFalse(check_action(("stop", None), o)[0])
        self.assertFalse(check_action("explore", o)[0])
        self.assertFalse(check_action(("walk_to", "66", "x"), o)[0])
        self.assertFalse(check_action(("walk_to", {"id": 66}), o)[0])
        o = self.obs(PI_ARRIVED)
        self.assertTrue(check_action(("pick_up", "66"), o)[0])
        self.assertTrue(check_action(("pick_up", None), o)[0])  # challenge.py:385-389: the object just walked to
        self.assertFalse(check_action(("pick_up", "64"), o)[0])
        o = self.obs(PI_HOLDING)
        self.assertTrue(check_action(("drop", None), o)[0])
        self.assertFalse(check_action(("walk_to", "64"), o)[0])

    def test_normalize(self):
        self.assertEqual(normalize_action(["walk_to", 7]), ("walk_to", "7"))
        self.assertEqual(normalize_action(("drop", None)), ("drop", None))
        self.assertEqual(normalize_action(("walk_to", np.int64(7))), ("walk_to", "7"))


class TestAudit(unittest.TestCase):
    def write(self, src):
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        f.write(src)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_example_passes(self):
        self.assertEqual(audit_program(EXAMPLE), [])

    def test_forbidden_imports_and_shape(self):
        p = self.write("import random, os\nfrom anthropic import Anthropic\nclass Policy:\n    def __init__(self, task, target_values): pass\n")
        problems = audit_program(p)
        self.assertEqual(len(problems), 4, problems)  # random, os, anthropic, no decide
        self.assertTrue(any("'random'" in x for x in problems))
        self.assertTrue(any("anthropic" in x for x in problems))
        self.assertTrue(any("decide" in x for x in problems))
        p = self.write("import numpy as np\ndef decide(obs): return ('explore', None)\n")
        self.assertEqual(audit_program(p), ["no `class Policy` at module level"])
        p = self.write("class Policy(\n")
        self.assertTrue(audit_program(p)[0].startswith("cannot parse"))
        with self.assertRaises(ValueError):
            load_program(p)
        with self.assertRaises(ValueError):
            load_program(self.write("import socket\nclass Policy:\n    def __init__(self, t, v): pass\n    def decide(self, o): pass\n"))


class TestProgramAgent(unittest.TestCase):
    def make(self, path=EXAMPLE):
        a = ProgramAgent(task="fire", program_path=path)
        a.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        return a

    def test_is_a_custom_agent(self):
        a = self.make()
        self.assertEqual(a.agent_type, "custom")  # challenge.py:257-258 accepts the label
        self.assertEqual(a.target_values, {"hairbrush": 1.0, "purse": 5.0})

    def test_episode_on_the_example(self):
        a = self.make()
        st = scene(objects={66: (50, 40), 64: (10, 40), 56: (30, 30)})
        # purse 56: path ~31 cells / value 5 = 6.4; comb 66: 32 / 1 = 33  -> the purse
        self.assertEqual(a.choose_target(st, PI_FIRST), ("walk_to", "56"))
        # the walk failed: cost doubles, still the cheapest (12.8 < 33)
        self.assertEqual(a.choose_target(st, PI_FAILED), ("walk_to", "56"))
        # three more failures: 6.4 * 16 = 102 > 33 -> the comb
        for _ in range(3):
            a.choose_target(st, PI_FAILED)
        self.assertEqual(a.choose_target(st, PI_FAILED), ("walk_to", "66"))
        self.assertEqual(a.last_action, ("walk_to", "66"))
        self.assertEqual(a.choose_target(st, PI_ARRIVED), ("pick_up", "66"))
        self.assertEqual(a.choose_target(st, PI_HOLDING), ("drop", None))
        # nothing left on the map that is a target -> explore
        empty = dict(PI_FIRST, explored_object_name_list=[PI_FIRST["explored_object_name_list"][0]])
        self.assertEqual(a.choose_target(scene(), empty), ("explore", None))
        self.assertEqual(len(a.decision_log), 9)
        self.assertTrue(all(r["legal"] for r in a.decision_log))
        s = a.summary()
        self.assertEqual((s["decisions"], s["illegal_or_crashed"]), (9, 0))
        self.assertEqual(s["actions"], {"walk_to": 6, "pick_up": 1, "drop": 1, "explore": 1})
        self.assertEqual(a.decision_log[1]["last_result"], {"ok": False, "info": "max steps reached"})
        # a new episode starts from clean memory
        a.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        self.assertEqual(a.choose_target(st, PI_FIRST), ("walk_to", "56"))

    def test_illegal_and_crashing_programs_fall_back_to_explore(self):
        src = ("class Policy:\n    def __init__(self, task, target_values): self.n = 0\n"
               "    def decide(self, obs):\n        self.n += 1\n"
               "        if self.n == 1: return ('pick_up', '66')\n"
               "        if self.n == 2: return 'walk_to'\n"
               "        if self.n == 3: raise KeyError('oops')\n        return ('walk_to', '66')\n")
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        f.write(src)
        f.close()
        self.addCleanup(os.unlink, f.name)
        a = self.make(f.name)
        st = scene(objects={66: (50, 40)})
        self.assertEqual(a.choose_target(st, PI_FIRST), ("explore", None))
        self.assertEqual(a.choose_target(st, PI_FIRST), ("explore", None))
        self.assertEqual(a.choose_target(st, PI_FIRST), ("explore", None))
        self.assertEqual(a.choose_target(st, PI_FIRST), ("walk_to", "66"))
        log = a.decision_log
        self.assertEqual([r["legal"] for r in log], [False, False, False, True])
        self.assertIn("not on the menu", log[0]["reason"])
        self.assertIn("(name, param) pair", log[1]["reason"])
        self.assertIn("KeyError", log[2]["reason"])
        self.assertEqual(a.summary()["illegal_or_crashed"], 3)

    def test_decision_time_on_a_full_size_map(self):
        a = self.make()
        rng = np.random.default_rng(0)
        st = scene(n=512, agent=(256, 60), facing=(256, 61), objects={66: (256, 400), 64: (40, 400), 56: (470, 300)},
                   walls=[(0, 200, 230, 2.5), (260, 512, 230, 2.5)])
        st["sem_map"]["height"] += (rng.random((512, 512)) > 0.97) * 0.9  # clutter
        t = time.perf_counter()
        action = a.choose_target(st, PI_FIRST)
        dt = time.perf_counter() - t
        self.assertEqual(action[0], "walk_to")
        self.assertLess(dt, 1.0, dt)  # 30 frames at most even on a loaded laptop; the measured figure is reported in NOTES
        print(f"\n  decision on a 512x512 map with 3 targets: {dt*1000:.1f} ms")


class TestExampleEqualsPaperReflex(unittest.TestCase):
    """The worked example and scripts/rrara/reflex.py must choose the same action at every decision, so the reflex's
    measured scores (0.757 on 5a50c2d, 0.199 on d35c71e, clocked) describe the program."""

    def test_random_scenes(self):
        rng = np.random.default_rng(7)
        agreements = 0
        for seed in range(40):
            n = 64
            k = int(rng.integers(1, 5))
            objs, entries = {}, [PI_FIRST["explored_object_name_list"][0]]
            for i in range(k):
                oid = 60 + i
                objs[oid] = (int(rng.integers(2, n - 3)), int(rng.integers(2, n - 3)))
                cat = "purse" if rng.random() < 0.4 else "hairbrush"
                entries.append({"name": f"obj_{oid}", "category": cat, "id": str(oid)})
            walls = [(int(rng.integers(0, 30)), int(rng.integers(34, n)), int(rng.integers(5, n - 5)), 2.5) for _ in range(int(rng.integers(0, 3)))]
            st = scene(n=n, agent=(int(rng.integers(2, n - 3)), int(rng.integers(2, n - 3))), facing=None, objects=objs, walls=walls,
                       unexplored_cols=[int(rng.integers(0, n))] if rng.random() < 0.5 else ())
            reflex = PaperReflex(task="fire", coarse=4, max_searches=6)
            reflex.reset(GOALS, OBJECTS_INFO)
            prog = ProgramAgent(task="fire", program_path=EXAMPLE)
            prog.reset(GOALS, OBJECTS_INFO)
            last = None
            for step in range(5):  # a few decisions with random outcomes, so the failure memory is exercised too
                if step == 0:
                    pi = dict(PI_FIRST, explored_object_name_list=entries)
                elif last is not None and last[0] == "walk_to" and rng.random() < 0.6:
                    pi = dict(PI_FAILED, explored_object_name_list=entries)
                elif last is not None and last[0] == "walk_to":
                    tgt = next(e for e in entries if e["id"] == last[1])
                    pi = dict(PI_ARRIVED, explored_object_name_list=entries, nearest_object=[tgt])
                elif last is not None and last[0] == "pick_up":
                    tgt = next(e for e in entries if e["id"] == last[1])
                    pi = dict(PI_HOLDING, explored_object_name_list=entries, holding_objects=[tgt])
                else:
                    pi = dict(PI_FIRST, explored_object_name_list=entries, action_result=True, action_info="success")
                if last is not None:
                    reflex.note_result(last, pi["action_result"])
                a_ref = reflex.choose_target(st, pi)
                a_prog = prog.choose_target(st, pi)
                self.assertEqual(a_prog, a_ref, (seed, step, pi["action_info"], reflex.last_costs))
                agreements += 1
                last = a_prog
        self.assertEqual(agreements, 200)


class TestRunnerWiring(unittest.TestCase):
    def test_help_lists_program(self):
        out = subprocess.run([sys.executable, RUNNER, "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        self.assertIn("--program", out.stdout)
        self.assertIn("| program", out.stdout)

    def test_program_agent_needs_a_path(self):
        out = subprocess.run([sys.executable, RUNNER, "--agent", "program", "--no_build"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 2, out.stderr[-500:])
        self.assertIn("--program is required", out.stderr)


if __name__ == "__main__":
    unittest.main()
