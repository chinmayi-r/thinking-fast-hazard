"""Tests for scripts/human/human_agent.py (A36). No simulator, no network: a hand-made state in the shape HAZARD's
challenge.py hands to the agent (process_input, challenge.py:134-163), the real policy.llm.LLM code underneath.

Run from G:\\iw-thinking-fast (the venv has HAZARD's dependencies):
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -m unittest human.test_human_agent -v
"""
import json
import os
import tempfile
import unittest

import numpy as np

from human.human_agent import HumanAgent

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = os.path.join(HERE, "..", "..", "HAZARD", "src", "HAZARD", "policy", "llm_configs", "prompt_v2.csv")

GOALS = ["hairbrush", "bag"]
OBJECTS_INFO = {"hairbrush": {"value": 1, "waterproof": 0}, "bag": {"value": 5, "waterproof": 0}}


def make_state(h=16, w=16):
    """goal_map: -2 marks the agent; sem_map id/explored and raw seg_mask carry object ids 64 (hairbrush), 71 (bag)."""
    goal_map = np.zeros((h, w), dtype=np.int64)
    goal_map[2, 2] = -2
    ids = np.zeros((h, w), dtype=np.int64)
    ids[10:12, 10:12] = 64
    ids[4:6, 12:14] = 71
    ids[14:16, 0:2] = 9  # the table: explored, not a target, so it never appears in the menu
    explored = np.ones((h, w), dtype=np.int64)
    seg = np.zeros((h, w), dtype=np.int64)
    seg[0:2, 0:2] = 64  # hairbrush in view, bag not
    log_temp = np.full((h, w), np.log(30.0))
    rgb = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32)
    return {"goal_map": goal_map, "sem_map": {"id": ids, "explored": explored},
            "raw": {"seg_mask": seg, "log_temp": log_temp, "rgb": rgb}}


def make_input(save_dir, step=120, holding=None, nearest=None):
    return {"explored_object_name_list": [{"name": "hairbrush", "category": "hairbrush", "id": "64"},
                                          {"name": "bag", "category": "bag", "id": "71"},
                                          {"name": "table", "category": "table", "id": "9"}],
            "holding_objects": holding or [], "nearest_object": nearest or [], "step": step,
            "action_result": True, "action_info": "", "save_dir": save_dir}


class ScriptedChooser:
    """Returns the scripted answers in order and remembers every decision it was shown."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.seen = []

    def __call__(self, decision):
        self.seen.append(json.loads(json.dumps(decision)))  # snapshot
        return self.answers.pop(0)


class TestHumanAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def make(self, answers, camera=True):
        chooser = ScriptedChooser(answers)
        agent = HumanAgent(task="fire", prompt_template_path=PROMPT, chooser=chooser, camera=camera)
        agent.reset(GOALS, OBJECTS_INFO)
        return agent, chooser

    def test_no_model_is_ever_called(self):
        agent, _ = self.make(["A"])
        self.assertIsNone(agent.anthropic_client)
        self.assertEqual(agent.generator, agent._ask_person)
        self.assertEqual(agent.agent_type, "llm")  # challenge.py:257, 308 dispatch on this

    def test_menu_is_hazards_and_letter_picks_it(self):
        agent, chooser = self.make(["B"])
        action = agent.choose_target(make_state(), make_input(self.tmp))
        d = chooser.seen[0]
        # the same options the LLM agent would list (llm.py:487-520): walk to each target, then look around
        self.assertEqual([m["text"] for m in d["menu"]],
                         ["go pick up object <hairbrush> (64)", "go pick up object <bag> (71)", "look around"])
        self.assertEqual(d["letters"], ["A", "B", "C"])
        self.assertEqual(action, ("walk_to", "71"))  # B = the bag
        # the same scene text Haiku gets (progress2text, llm.py:382-430)
        self.assertIn("Target objects currently seen:", d["prompt"])
        self.assertIn("name: hairbrush, id: 64, value: 1", d["prompt"])
        self.assertIn("temperature: 30.0 Celsius", d["prompt"])
        self.assertIn("Target objects previously seen:", d["prompt"])
        self.assertIn("name: bag, id: 71, value: 5", d["prompt"])
        self.assertIn("A. go pick up object <hairbrush> (64)", d["prompt"])  # the menu is in the prompt text too
        self.assertNotIn("step by step", d["prompt"])  # no chain-of-thought suffix (A36 point 2)
        self.assertEqual(d["frame"], 120)
        self.assertEqual(d["frames_total"], 1500)

    def test_bad_input_is_asked_again_and_logged(self):
        agent, chooser = self.make(["Z", "", " c "])
        action = agent.choose_target(make_state(), make_input(self.tmp))
        self.assertEqual(action, ("explore", None))
        self.assertEqual(len(chooser.seen), 3)
        rec = agent.records[-1]
        self.assertEqual(rec["rejected"], ["Z", ""])
        self.assertEqual(rec["choice"], "C")
        self.assertEqual(agent.summary()["rejected_inputs"], 2)

    def test_files_written(self):
        agent, _ = self.make(["A"])
        agent.choose_target(make_state(), make_input(self.tmp, step=7))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "0007_human.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "0007_human_frame.png")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "0007_info.json")))  # HAZARD's own record, llm.py:584-586
        with open(os.path.join(self.tmp, "human_log.json"), encoding="utf-8") as f:
            log = json.load(f)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["action"], ["walk_to", "64"])
        self.assertEqual(log[0]["frame_png"], os.path.join(self.tmp, "0007_human_frame.png"))
        self.assertGreaterEqual(log[0]["wall_s"], 0)
        from PIL import Image
        with Image.open(log[0]["frame_png"]) as im:
            self.assertEqual(im.size, (32, 32))

    def test_camera_off_writes_no_png(self):
        agent, chooser = self.make(["A"], camera=False)
        agent.choose_target(make_state(), make_input(self.tmp, step=7))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "0007_human_frame.png")))
        self.assertIsNone(chooser.seen[0]["frame_png"])

    def test_pick_up_is_automatic_as_for_haiku(self):
        agent, chooser = self.make([])  # any ask would raise IndexError
        nearest = [{"name": "hairbrush", "category": "hairbrush", "id": "64"}]
        action = agent.choose_target(make_state(), make_input(self.tmp, nearest=nearest))
        self.assertEqual(action, ("pick_up", "64"))  # llm.py:564-567, before the menu exists
        self.assertEqual(chooser.seen, [])
        self.assertTrue(agent.records[-1]["automatic"])
        self.assertEqual(agent.summary(), {"decisions": 1, "asked": 0, "automatic": 1, "rejected_inputs": 0,
                                           "wall_s_mean": None, "wall_s_max": None})

    def test_drop_is_automatic_as_for_haiku(self):
        agent, chooser = self.make([])
        holding = [{"name": "bag", "category": "bag", "id": "71"}]
        action = agent.choose_target(make_state(), make_input(self.tmp, holding=holding))
        self.assertEqual(action, ("drop", None))  # llm.py:560-563
        self.assertEqual(chooser.seen, [])

    def test_history_and_result_wording_reach_the_person(self):
        agent, chooser = self.make(["A", "C"])
        agent.choose_target(make_state(), make_input(self.tmp, step=1))
        pi = make_input(self.tmp, step=101)
        pi["action_result"] = False
        pi["action_info"] = "max steps reached"  # challenge.py's info when a walk ran 100 frames
        agent.choose_target(make_state(), pi)
        self.assertIn("go pick up object <hairbrush> (64) (paused after taking 100 steps)", chooser.seen[1]["prompt"])
        self.assertEqual(agent.n, 2)
        self.assertEqual(len(agent.records), 2)

    def test_reset_clears_the_episode(self):
        agent, _ = self.make(["A"])
        agent.choose_target(make_state(), make_input(self.tmp))
        agent.reset(GOALS, OBJECTS_INFO)
        self.assertEqual(agent.records, [])
        self.assertEqual(agent.n, 0)


if __name__ == "__main__":
    unittest.main()
