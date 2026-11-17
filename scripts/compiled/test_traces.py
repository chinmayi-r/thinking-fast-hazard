"""Tests for the trace converter (scripts/compiled/traces.py). A synthetic episode folder in HAZARD's layout; the real
Haiku sweep is used when it is on disk (results/ is not in the repo).

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest compiled.test_traces -v
"""
import json
import os
import tempfile
import unittest

from compiled.traces import (episode_stats, load_episode, load_sweep, parse_actions_txt, parse_plan,
                             parse_prompt_objects, render_episode)

PROMPT = ("...\nCurrent State:\nTarget objects currently seen:\nname: hairbrush, id: 64, value: 1, distance: 65.84 m, temperature: 20.04 Celsius\n"
          "Target objects previously seen:\nname: bag, handbag, pocketbook, purse, id: 56, value: 5, distance: 32.79 m, temperature: unknown\n"
          "Previous actions:\nlook around (success)\n\nObjects states history:\n$OBJECT_HISTORY$\nAvailable actions:\n"
          "A. go pick up object <bag, handbag, pocketbook, purse> (56)\nB. go pick up object <hairbrush> (64)\nC. look around\n\nAnswer: ")
REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "sweep_fire_llm_haiku_test25", "mm_kitchen_3a-1")


def make_episode(root):
    ep = os.path.join(root, "mm_kitchen_test")
    os.makedirs(os.path.join(ep, "0"))
    objs = [{"name": "cabinet_x", "category": "cabinet", "id": "1"}, {"name": "b04_comb", "category": "hairbrush", "id": "64"},
            {"name": "bag_3", "category": "bag, handbag, pocketbook, purse", "id": "56"}]
    def inp(step, result, info, holding=(), nearest=()):
        json.dump({"explored_object_name_list": objs, "holding_objects": list(holding), "nearest_object": list(nearest),
                   "step": step, "action_result": result, "action_info": info, "save_dir": "x"},
                  open(os.path.join(ep, "0", f"input{step}.json"), "w"))
    def info(step, letter, plan, cot="Because the bag is worth 5."):
        json.dump({"cot_outputs": [cot], "cot_usage": 0.0, "num_available_actions": 3, "prompts": PROMPT,
                   "outputs": [letter], "plan": plan, "total_cost": 0.0}, open(os.path.join(ep, "0", f"{step:04d}_info.json"), "w"))
    inp(43, False, "")
    info(43, "C", "look around")
    inp(92, True, "success")
    info(92, "A", "go pick up object <bag, handbag, pocketbook, purse> (56)")
    inp(195, False, "max steps reached")
    info(195, "B", "go pick up object <hairbrush> (64)")
    inp(250, True, "success", nearest=[{"name": "b04_comb", "category": "hairbrush", "id": "64"}])  # automatic pick-up
    inp(290, True, "success, reach for success", holding=[{"name": "b04_comb", "category": "hairbrush", "id": "64"}])  # automatic drop
    with open(os.path.join(ep, "actions.txt"), "w") as f:
        f.write("step 0 action ('explore', None)\nstep 0 action ('walk_to', '56')\nstep 0 action ('walk_to', '64')\n"
                "step 0 action ('pick_up', 64) b04_comb 12345\nstep 0 action ('drop', None)\n")
    json.dump({"avg_score": 1.0, "avg_max_score": 6.0}, open(os.path.join(ep, "eval_result.json"), "w"))
    return ep


class TestParsers(unittest.TestCase):
    def test_plan_to_action(self):
        self.assertEqual(parse_plan("go pick up object <bag, handbag, pocketbook, purse> (56)"), ("walk_to", "56"))
        self.assertEqual(parse_plan("go to object <table> (9)"), ("walk_to", "9"))
        self.assertEqual(parse_plan("put the holding object in my bag"), ("drop", None))
        self.assertEqual(parse_plan("look around"), ("explore", None))
        self.assertIsNone(parse_plan("something else"))
        self.assertIsNone(parse_plan(None))

    def test_prompt_objects(self):
        d = parse_prompt_objects(PROMPT)
        self.assertEqual(d["56"], {"category": "bag, handbag, pocketbook, purse", "value_shown": 5.0, "distance_cells": 32.79})
        self.assertEqual(d["64"]["distance_cells"], 65.84)

    def test_actions_txt(self):
        p = os.path.join(tempfile.mkdtemp(), "actions.txt")
        with open(p, "w") as f:
            f.write("step 0 action ('explore', None)\nstep 0 action ('walk_to', '56')\nstep 0 action ('pick_up', 64) b04_comb 1\nstep 0 action ('drop', None)\n")
        self.assertEqual(parse_actions_txt(p), [("explore", None), ("walk_to", "56"), ("pick_up", "64"), ("drop", None)])
        self.assertEqual(parse_actions_txt(p + ".missing"), [])


class TestEpisode(unittest.TestCase):
    def setUp(self):
        self.ep = make_episode(tempfile.mkdtemp())

    def test_records_align_decisions_outcomes_and_automatic_actions(self):
        ep = load_episode(self.ep, name_values={"bag_3": 0, "small_purse": 1})
        self.assertEqual((ep["kitchen"], ep["score"], len(ep["decisions"])), ("mm_kitchen_test", 0.167, 5))
        r1, r2, r3, r4, r5 = ep["decisions"]
        self.assertEqual((r1["action"], r1["automatic"], r1["prev_result"]), (["explore", None], False, None))
        self.assertEqual((r1["outcome"], r1["frames_used"]), ({"ok": True, "info": "success"}, 49))
        self.assertEqual((r2["action"], r2["chosen"]["letter"]), (["walk_to", "56"], "A"))
        self.assertEqual(r2["outcome"], {"ok": False, "info": "max steps reached"})
        self.assertEqual(r2["frames_used"], 103)
        bag = next(o for o in r2["objects"] if o["id"] == "56")
        self.assertEqual((bag["value_shown"], bag["value_scorer"], bag["distance_m"], bag["is_target"]), (5.0, 1, 3.28, True))
        cab = next(o for o in r2["objects"] if o["id"] == "1")
        self.assertEqual((cab["is_target"], cab["distance_m"]), (False, None))
        self.assertEqual(r2["menu"][0], "A. go pick up object <bag, handbag, pocketbook, purse> (56)")
        self.assertEqual(r2["reasoning"], "Because the bag is worth 5.")
        self.assertEqual((r4["action"], r4["automatic"], r4["at_object"]["id"]), (["pick_up", "64"], True, "64"))
        self.assertEqual((r5["action"], r5["holding"][0]["id"], r5["outcome"]), (["drop", None], "64", None))
        st = episode_stats(ep)
        self.assertEqual((st["walks"], st["walks_capped"], st["walks_arrived"], st["explores"], st["most_walked"]), (2, 1, 1, 1, ("56", 1)))

    def test_render(self):
        ep = load_episode(self.ep, name_values={"bag_3": 0})
        text = render_episode(ep, with_reasoning=True)
        self.assertTrue(text.startswith("### Episode mm_kitchen_test: final value rate 0.167"))
        self.assertIn("- decision 2, frame 92: holding nothing; previous action: ok (success)", text)
        self.assertIn("targets seen: hairbrush (id 64) shown worth 1, scorer pays 1, 6.58 m straight line; bag, handbag, pocketbook, purse (id 56) shown worth 5, scorer pays 1, 3.28 m straight line", text)
        self.assertIn("action: ('walk_to', '56') (letter A)", text)
        self.assertIn("reasoning: Because the bag is worth 5.", text)
        self.assertIn("outcome: failed (max steps reached), 103 frames", text)
        self.assertIn("action: ('pick_up', '64') (automatic)", text)
        short = render_episode(ep, max_decisions=2)
        self.assertNotIn("decision 3", short)
        self.assertNotIn("reasoning", short)

    def test_load_sweep(self):
        root = os.path.dirname(self.ep)
        eps = load_sweep(root)
        self.assertEqual([e["kitchen"] for e in eps], ["mm_kitchen_test"])
        self.assertEqual(load_sweep(root, kitchens=["other"]), [])


@unittest.skipUnless(os.path.isdir(os.path.join(REAL, "0")), "the Nov 10 Haiku sweep is not on this machine")
class TestRealHaikuEpisode(unittest.TestCase):
    def test_haiku_3a_1(self):
        ep = load_episode(REAL)
        self.assertEqual(ep["kitchen"], "mm_kitchen_3a-1")
        self.assertEqual(len(ep["decisions"]), 15)
        self.assertEqual(ep["decisions"][0]["action"], ["explore", None])
        st = episode_stats(ep)
        self.assertGreaterEqual(st["walks_capped"], 10)  # the decoy-bag episode: Haiku walked to bag 56 again and again
        self.assertEqual(st["most_walked"][0], "56")
        text = render_episode(ep)
        self.assertIn("(id 56) shown worth 5", text)


if __name__ == "__main__":
    unittest.main()
