"""Tests for honest text v2 (scripts/honest/honest_text.py). Synthetic HAZARD-shaped states, no simulator.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest honest.test_honest_text -v
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from honest.honest_text import (EXPLORE_FRAMES, Honest, apply_honest_text, direction_text, load_template,
                                name_value_table)
from human.human_agent import HumanAgent
from jev.client import JevClient
from jev.jev_agent import JevAgent

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "prompt_honest.csv")
HAZARD_PROMPT = os.path.join(HERE, "..", "..", "HAZARD", "src", "HAZARD", "policy", "llm_configs", "prompt_v2.csv")
RUNNER = os.path.join(os.path.dirname(HERE), "run_baseline.py")

GOALS = ["hairbrush", "purse"]
OBJECTS_INFO = {"hairbrush": {"value": 1, "inflammable": 1}, "purse": {"value": 5, "inflammable": 1}}
TARGET_VALUES = {"hairbrush": 1.0, "purse": 5.0}
OBJECTS = [{"name": "cabinet_36_two_door_wood_oak_white_composite", "category": "cabinet", "id": "1"},
           {"name": "b04_comb", "category": "hairbrush", "id": "66"},
           {"name": "bag_3", "category": "purse", "id": "56"},
           {"name": "small_purse", "category": "purse", "id": "62"}]
NAME_VALUES = {"bag_3": 1, "small_purse": 5}


def make_state(n=64, agent=(32, 10), facing=(32, 11), cells=None, wall=None, unexplored=None, hot=None):
    """cells: {id: (r, c)} 2x2 blocks; wall: (r0, r1, col); unexplored: (r0, r1, c0, c1); hot: ((r, c), celsius)."""
    cells = cells if cells is not None else {1: (2, 2), 66: (32, 40), 56: (10, 40), 62: (50, 12)}
    explored = np.ones((n, n), dtype=np.int64)
    height = np.zeros((n, n), dtype=np.float32)
    ids = np.zeros((n, n), dtype=np.int64)
    for oid, (r, c) in cells.items():
        ids[r:r + 2, c:c + 2] = oid
    if wall:
        height[wall[0]:wall[1], wall[2]] = 2.5
    if unexplored:
        explored[unexplored[0]:unexplored[1], unexplored[2]:unexplored[3]] = 0
    other = np.full((1, n, n), np.log(25.0), dtype=np.float32)
    if hot:
        (r, c), t = hot
        other[0, r, c] = np.log(t)
    goal = np.zeros((n, n))
    if agent:
        goal[agent] = -2
    if facing:
        goal[facing] = -1
    seg = np.zeros((8, 8), dtype=np.int64)
    return {"goal_map": goal, "sem_map": {"explored": explored, "height": height, "id": ids, "other": other},
            "raw": {"seg_mask": seg, "log_temp": np.full((8, 8), np.log(25.0), dtype=np.float32),
                    "rgb": np.zeros((3, 8, 8), dtype=np.float32)}, "RL": {}}


def make_input(step=43, holding=None, nearest=None, result=False, info="", objects=None, save_dir=None):
    return {"explored_object_name_list": list(objects if objects is not None else OBJECTS),
            "holding_objects": holding or [], "nearest_object": nearest or [], "step": step,
            "action_result": result, "action_info": info, "save_dir": save_dir or tempfile.mkdtemp()}


class TestTables(unittest.TestCase):
    def test_scorer_values_by_name(self):
        t = name_value_table()
        self.assertEqual((t["bag_3"], t["small_purse"]), (1, 5))  # the Nov 16 decoy and the real purse
        self.assertEqual(t["b04_comb"], 1)
        self.assertNotIn("no_such_object", t)  # a name not in the file is worth 1 to the scorer (challenge.py:202-203)

    def test_direction_text(self):
        self.assertEqual(direction_text(0), "straight ahead")
        self.assertEqual(direction_text(-12), "straight ahead")
        self.assertEqual(direction_text(40), "40 deg to my left")
        self.assertEqual(direction_text(-100), "100 deg to my right")
        self.assertEqual(direction_text(175), "behind me")
        self.assertEqual(direction_text(None), "direction unknown")

    def test_template_has_the_slots(self):
        for task in ("fire", "flood", "wind"):
            t = load_template(TEMPLATE, task)
            for slot in ("$TARGET_OBJECTS$", "$STATE$", "$ACTION_HISTORY$", "$AVAILABLE_ACTIONS$"):
                self.assertIn(slot, t)
            self.assertNotIn("$OBJECT_HISTORY$", t)


class TestHonestState(unittest.TestCase):
    def test_first_decision_text(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        st = make_state(unexplored=(0, 64, 20, 24), hot=((32, 30), 400))  # an unseen band across the whole map
        h.begin_decision(st, make_input())
        s = h.state_text()
        self.assertIn("Frame 43 of 1500, 1457 frames left.", s)
        self.assertIn("Holding: nothing.", s)
        self.assertIn("Standing at: no item", s)
        lines = [l for l in s.splitlines() if l.startswith("- ")]
        self.assertEqual(len(lines), 3)  # the cabinet is not a target
        self.assertTrue(lines[0].startswith('- purse "small_purse" (id 62)'), lines[0])  # nearest by path first
        comb = next(l for l in lines if "(id 66)" in l)
        # the comb is straight ahead (heading +col, object on the same row, 3 m away) and beyond the unseen band
        self.assertTrue(comb.startswith('- hairbrush "b04_comb" (id 66): worth 1; 3.0 m away, straight ahead; path'), comb)
        self.assertIn("over unseen floor", comb)
        self.assertIn("walk time unknown yet (cap 100 frames)", comb)
        self.assertIn('"bag_3" (id 56): worth 1', s)  # the scorer's value, not the category's 5
        self.assertIn('"small_purse" (id 62): worth 5', s)
        self.assertIn("Fire: hottest spot on my map 400 C, 2.0 m away, straight ahead.", s)
        self.assertIn("Unseen floor: nearest 1.0 m away", s)
        self.assertIn("Saved so far: nothing.", s)
        self.assertEqual(h.history_text(), "(nothing yet)")

    def test_category_values_when_asked(self):
        h = Honest("fire", TARGET_VALUES, name_values=None)
        h.begin_decision(make_state(), make_input())
        self.assertIn('"bag_3" (id 56): worth 5', h.state_text())
        self.assertIn("name: purse, value: 5", h.targets_text())

    def test_bearing_sides(self):
        # heading +col (east on the map); an object at a smaller row is at +90 deg = left (A41 convention)
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        h.begin_decision(make_state(cells={56: (10, 10), 62: (50, 10)}, unexplored=None), make_input())
        s = h.state_text()
        self.assertIn('(id 56): worth 1; 2.2 m away, 90 deg to my left', s)
        self.assertIn('(id 62): worth 5; 1.8 m away, 90 deg to my right', s)

    def test_no_fire_no_unseen(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        h.begin_decision(make_state(), make_input())
        s = h.state_text()
        self.assertIn("Fire: nothing hot on my map yet.", s)
        self.assertIn("Unseen floor: none within 3 m; 100 % of the floor within 3 m is mapped.", s)


class TestRadar(unittest.TestCase):
    def test_eight_sectors(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        # robot at (32, 10) facing +col; a wall at col 30 on rows 0-40 (2 m ahead); map 64 x 64
        h.begin_decision(make_state(wall=(0, 40, 30)), make_input())
        facts = {f["sector"]: f for f in h.radar_facts()}
        self.assertEqual([f["sector"] for f in h.radar_facts()],
                         ["ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right"])
        self.assertEqual((facts["ahead"]["kind"], facts["ahead"]["dist_m"]), ("wall", 2.0))
        self.assertEqual((facts["behind"]["kind"], facts["behind"]["dist_m"]), ("map edge", 1.1))  # col -1 at step 11
        self.assertEqual((facts["left"]["kind"], facts["left"]["dist_m"]), ("map edge", 3.3))  # left = -row (A41)
        self.assertEqual((facts["right"]["kind"], facts["right"]["dist_m"]), ("map edge", 3.2))
        self.assertEqual(facts["ahead-left"]["kind"], "wall")  # the diagonal toward smaller rows meets the wall (rows < 40)
        self.assertEqual(facts["ahead-right"]["kind"], "map edge")  # the other diagonal passes below the wall's end (row 40)
        self.assertGreater(facts["ahead-right"]["dist_m"], 4.0)
        text = h.radar_text()
        self.assertTrue(text.startswith("Around me, the first thing in each direction: ahead: wall at 2.0 m; ahead-left: wall at"), text)
        self.assertIn("Around me", h.state_text())

    def test_unseen_and_low_obstacle_and_open(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        st = make_state(unexplored=(0, 64, 20, 24))
        st["sem_map"]["height"][40:64, 10] = 0.3  # a counter 0.8 m to the right (+row)
        h.begin_decision(st, make_input())
        facts = {f["sector"]: f for f in h.radar_facts()}
        self.assertEqual((facts["ahead"]["kind"], facts["ahead"]["dist_m"]), ("unseen floor", 1.0))
        self.assertEqual((facts["right"]["kind"], facts["right"]["dist_m"]), ("low obstacle", 0.8))
        self.assertIn("right: low obstacle at 0.8 m", h.radar_text())
        big = make_state(n=160, agent=(80, 80), facing=(80, 81), cells={66: (150, 150)})
        h.begin_decision(big, make_input(objects=OBJECTS[:2]))
        self.assertTrue(all(f["kind"] == "open" for f in h.radar_facts()))
        self.assertIn("ahead: open for 5 m", h.radar_text())

    def test_no_heading_no_radar(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        h.begin_decision(make_state(facing=None), make_input())
        self.assertIsNone(h.radar_facts())
        self.assertNotIn("Around me", h.state_text())


class TestResults(unittest.TestCase):
    def run_walk(self, h, st_before, st_after, step0=43, step1=149, result=False, info="max steps reached"):
        h.begin_decision(st_before, make_input(step=step0))
        h.end_decision(("walk_to", "66"))
        h.begin_decision(st_after, make_input(step=step1, result=result, info=info))
        return h.log[-1]

    def test_capped_walk_that_got_closer_teaches_the_speed(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        rec = self.run_walk(h, make_state(agent=(32, 10), facing=(32, 11)), make_state(agent=(32, 30), facing=(32, 31)))
        self.assertEqual((rec["moved_m"], rec["frames"], rec["was_m"], rec["remaining_m"]), (2.0, 106, 3.0, 1.0))
        self.assertEqual(rec["text"], 'walked toward hairbrush "b04_comb" (id 66): moved 2.0 m in 106 frames; ended 1.0 m from it (was 3.0 m): ran out of the 100-frame walk budget')
        self.assertEqual(h.frames_per_m(), 53.0)
        f = h.facts["66"]
        self.assertEqual(f["pred_frames"], math.ceil(f["path_m"] * 53.0))  # the learned speed prices the next walk
        self.assertIn(f"about {f['pred_frames']} frames (cap 100)", h.state_text())
        self.assertIn("walks tried 1, last ended 1.0 m short", h.state_text())
        self.assertEqual(h.history_text(), "1. frame 43: " + rec["text"])

    def test_walk_beyond_the_cap_is_said_so(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        # a slow walk: 1.0 m in 106 frames -> 106 frames per metre; the comb is then 3.2 m = 340 frames away
        self.run_walk(h, make_state(agent=(32, 10), facing=(32, 11)), make_state(agent=(32, 20), facing=(32, 21)))
        f = h.facts["66"]
        self.assertEqual(f["need_frames"], math.ceil(f["path_m"] * 106.0))
        self.assertGreater(f["need_frames"], 100)
        self.assertIn(f"about {f['need_frames']} frames at my speed so far: more than one walk's cap of 100, so one walk will stop short and it takes about {math.ceil(f['need_frames'] / 100)} walks", h.state_text())
        self.assertNotIn("about 100 frames (cap 100)", h.state_text())

    def test_walk_that_did_not_get_going(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        rec = self.run_walk(h, make_state(), make_state(agent=(32, 11), facing=(32, 12)))
        self.assertIn("moved 0.1 m in 106 frames", rec["text"])
        self.assertIn("did not get going (blocked right where I stood)", rec["text"])
        self.assertIsNone(h.frames_per_m())

    def test_walk_that_got_no_closer_and_unseen_path(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        before = make_state(unexplored=(0, 64, 20, 24))
        after = make_state(agent=(20, 10), facing=(20, 11), unexplored=(0, 64, 20, 24))
        rec = self.run_walk(h, before, after)
        self.assertIn("walked but got no closer", rec["text"])
        self.assertIn("the planned path crossed", rec["text"])
        self.assertIn("m of unseen floor", rec["text"])

    def test_arrival_pick_up_drop_and_saved_total(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        st = make_state()
        rec = self.run_walk(h, st, make_state(agent=(32, 38), facing=(32, 39)), result=True, info="success")
        self.assertIn("arrived (within 0.5 m)", rec["text"])
        comb = {"name": "b04_comb", "category": "hairbrush", "id": "66"}
        h.end_decision(("pick_up", "66"))
        h.begin_decision(st, make_input(step=190, result=True, info="success, reach for success", holding=[comb]))
        self.assertEqual(h.log[-1]["text"], 'picked up hairbrush "b04_comb" (id 66) (41 frames)')
        self.assertIn('Holding: hairbrush "b04_comb" (id 66), worth 1.', h.state_text())
        text, n, plans, actions = h.menu()
        self.assertEqual(actions[0], ("drop", None))
        self.assertTrue(plans[0].startswith('put hairbrush "b04_comb" (id 66) in my bag: saves it (worth 1)'))
        h.end_decision(("drop", None))
        h.begin_decision(st, make_input(step=240, result=True, info="successfully drop and put away"))
        self.assertEqual(h.log[-1]["text"], 'put hairbrush "b04_comb" (id 66) in my bag (50 frames): saved, worth 1')
        self.assertIn("Saved so far: hairbrush (id 66) worth 1 (total 1).", h.state_text())
        self.assertEqual(h.summary()["saved_value"], 1.0)

    def test_explore_turn_step_results(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        st = make_state()
        h.begin_decision(st, make_input(objects=OBJECTS[:2]))
        h.end_decision(("explore", None))
        h.begin_decision(st, make_input(step=92, result=True, info="success"))
        self.assertEqual(h.log[-1]["text"], "looked around (49 frames): 2 new objects on my map")
        h.end_decision(("low_level.turn_by", {"angle": -90.0}))
        h.begin_decision(st, make_input(step=96, result=True, info="success"))
        self.assertEqual(h.log[-1]["text"], "turned left 90 deg (4 frames)")
        h.end_decision(("low_level.move_by", {"distance": 1.0}))
        h.begin_decision(make_state(agent=(32, 20), facing=(32, 21)), make_input(step=110, result=True, info="success"))
        self.assertEqual(h.log[-1]["text"], "stepped forward: moved 1.0 m (14 frames)")
        self.assertEqual([r["n"] for r in h.log], [1, 2, 3])

    def test_clock_accounting(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES, fps=30)
        h.begin_decision(make_state(), make_input(step=43))
        time.sleep(0.07)  # the "model" thinks 70 ms = 3 frames
        h.end_decision(("walk_to", "66"))
        h.begin_decision(make_state(agent=(32, 30), facing=(32, 31)), make_input(step=152, result=False, info="max steps reached"))
        rec = h.log[-1]
        self.assertGreaterEqual(rec["think_frames"], 3)
        self.assertEqual(rec["frames"] + rec["think_frames"], 109)
        self.assertIn("frames of thinking", rec["text"])


class TestMenu(unittest.TestCase):
    def test_menu_shape_and_order(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        h.begin_decision(make_state(), make_input())
        text, n, plans, actions = h.menu()
        self.assertEqual(n, len(plans), len(actions))
        self.assertEqual([a[0] for a in actions], ["walk_to", "walk_to", "walk_to", "walk_to_single", "explore",
                                                   "low_level.turn_by", "low_level.turn_by", "low_level.move_by"])
        self.assertEqual([a[1] for a in actions[:3]], ["62", "66", "56"])  # nearest by path first
        self.assertEqual(actions[3], ("walk_to_single", "62"))
        self.assertEqual(actions[5], ("low_level.turn_by", {"angle": -90.0}))  # left = counter-clockwise (A41)
        self.assertEqual(actions[6], ("low_level.turn_by", {"angle": 90.0}))
        self.assertEqual(actions[7], ("low_level.move_by", {"distance": 1.0}))
        self.assertTrue(text.startswith('A. walk to purse "small_purse" (id 62): path'))
        self.assertIn(f"look around: turn a full circle, {EXPLORE_FRAMES} frames", text)
        self.assertEqual(text.count("\n"), 8)
        for i, line in enumerate(text.splitlines()):
            self.assertTrue(line.startswith(f"{chr(ord('A') + i)}. "))

    def test_menu_without_extras_and_cap(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES, single_steps=False, turns=False)
        objs = [OBJECTS[0]] + [{"name": f"obj{i}", "category": "hairbrush", "id": str(100 + i)} for i in range(15)]
        cells = {100 + i: (2 + 4 * (i % 15), 50) for i in range(15)}
        h.begin_decision(make_state(cells=cells), make_input(objects=objs))
        text, n, plans, actions = h.menu()
        self.assertEqual(n, 12)
        self.assertEqual(actions[-1], ("explore", None))
        self.assertEqual(sum(1 for a in actions if a[0] == "walk_to"), 11)

    def test_nothing_seen(self):
        h = Honest("fire", TARGET_VALUES, name_values=NAME_VALUES)
        h.begin_decision(make_state(cells={1: (2, 2)}), make_input(objects=OBJECTS[:1]))
        text, n, plans, actions = h.menu()
        self.assertEqual(actions[0], ("explore", None))
        self.assertEqual(n, 4)
        self.assertIn("Target items I have seen: none yet.", h.state_text())


class ScriptedChooser:
    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def __call__(self, decision):
        self.seen.append(json.loads(json.dumps(decision)))
        return self.answers.pop(0)


class TestMixinOnHumanAgent(unittest.TestCase):
    def test_prompt_menu_history_and_auto_actions(self):
        chooser = ScriptedChooser(["A", "E", "B", "C"])
        agent = HumanAgent(task="fire", prompt_template_path=HAZARD_PROMPT, chooser=chooser, camera=False)
        apply_honest_text(agent, TEMPLATE, name_values=NAME_VALUES)
        self.assertEqual(type(agent).__name__, "HonestHumanAgent")
        agent.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        st = make_state()
        d = tempfile.mkdtemp()
        a1 = agent.choose_target(st, make_input(step=43, save_dir=d))
        self.assertEqual(a1, ("walk_to", "62"))  # letter A = nearest target by path
        p = chooser.seen[0]["prompt"]
        self.assertNotIn("$", p)  # every slot filled
        self.assertIn("Me and the map now:\nFrame 43 of 1500", p)
        self.assertIn("(nothing yet)", p)
        self.assertIn("purse: each item is worth 5 or 1", p)
        self.assertTrue(any(m["text"].startswith("look around") for m in chooser.seen[0]["menu"]))
        # the walk arrives: the parent picks up automatically, no question asked; then drops automatically
        purse = {"name": "small_purse", "category": "purse", "id": "62"}
        a2 = agent.choose_target(make_state(agent=(50, 10), facing=(50, 11)),
                                 make_input(step=120, result=True, info="success", nearest=[purse], save_dir=d))
        self.assertEqual(a2, ("pick_up", "62"))
        a3 = agent.choose_target(make_state(agent=(50, 10), facing=(50, 11)),
                                 make_input(step=160, result=True, info="success, reach for success", holding=[purse], save_dir=d))
        self.assertEqual(a3, ("drop", None))
        self.assertEqual(len(chooser.seen), 1)
        # fourth decision: the history holds all three earlier actions with numbers
        a4 = agent.choose_target(make_state(agent=(50, 10), facing=(50, 11)),
                                 make_input(step=210, result=True, info="successfully drop and put away", save_dir=d))
        self.assertEqual(a4, ("explore", None))  # letter E = look around (3 walks + single step before it)
        p = chooser.seen[1]["prompt"]
        self.assertIn('1. frame 43: walked toward purse "small_purse" (id 62): moved 1.8 m in 77 frames, arrived (within 0.5 m)', p)
        self.assertIn('2. frame 120: picked up purse "small_purse" (id 62) (40 frames)', p)
        self.assertIn('3. frame 160: put purse "small_purse" (id 62) in my bag (50 frames): saved, worth 5', p)
        self.assertIn("Saved so far: purse (id 62) worth 5 (total 5).", p)
        self.assertEqual(agent.honest.summary()["saved_value"], 5.0)

    def test_history_is_never_trimmed(self):
        chooser = ScriptedChooser(["E"] * 14)  # look around, 14 times
        agent = HumanAgent(task="fire", prompt_template_path=HAZARD_PROMPT, chooser=chooser, camera=False)
        apply_honest_text(agent, TEMPLATE, name_values=NAME_VALUES)
        agent.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        d = tempfile.mkdtemp()
        for i in range(14):
            agent.choose_target(make_state(), make_input(step=43 + 49 * i, result=i > 0, info="success" if i else "", save_dir=d))
        p = chooser.seen[-1]["prompt"]
        for i in range(1, 14):
            self.assertIn(f"{i}. frame {43 + 49 * (i - 1)}: looked around (49 frames)", p)  # HAZARD keeps only the last 10

    def test_reset_clears_the_episode(self):
        chooser = ScriptedChooser(["A", "A"])
        agent = HumanAgent(task="fire", prompt_template_path=HAZARD_PROMPT, chooser=chooser, camera=False)
        apply_honest_text(agent, TEMPLATE, name_values=NAME_VALUES)
        agent.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        d = tempfile.mkdtemp()
        agent.choose_target(make_state(), make_input(save_dir=d))
        agent.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        agent.choose_target(make_state(), make_input(save_dir=d))
        self.assertIn("(nothing yet)", chooser.seen[1]["prompt"])


class TestMixinOnJevAgent(unittest.TestCase):
    def test_jev_reads_the_honest_menu(self):
        from jev.test_jev import FakeTransport
        client = JevClient(api_key_file="", transport=FakeTransport(["C"]), sleep=lambda s: None)
        agent = JevAgent(task="fire", prompt_template_path=HAZARD_PROMPT, client=client)
        apply_honest_text(agent, TEMPLATE, name_values=NAME_VALUES)
        agent.reset(goal_objects=GOALS, objects_info=OBJECTS_INFO)
        action = agent.choose_target(make_state(), make_input(save_dir=tempfile.mkdtemp()))
        self.assertEqual(action, ("walk_to", "56"))  # letter C of the honest menu
        req = client.transport.requests[0]
        self.assertEqual(len(req["questions"]["action"]["criteria"]), 8)
        self.assertIn("Frame 43 of 1500", req["state"])
        self.assertIn("look around: turn a full circle", req["questions"]["action"]["criteria"]["E"])


class TestRunnerWiring(unittest.TestCase):
    def test_help_lists_the_flags(self):
        out = subprocess.run([sys.executable, RUNNER, "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        for flag in ("--honest_text", "--category_values", "--no_single_steps", "--no_turns"):
            self.assertIn(flag, out.stdout)


if __name__ == "__main__":
    unittest.main()
