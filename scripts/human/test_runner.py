"""Tests for scripts/human/runner.py (A36, file 3). A fake Challenge object stands in for HAZARD's (only target_status
and data_dir are read), the real HumanAgent runs on the file-1 fixture, the page is a real server on an ephemeral port.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest human.test_runner -v
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

from human.page import PageChooser
from human.runner import CountedChooser, LiveCounters, finish_human_run, make_human_agent, score_from_evals
from human.test_human_agent import GOALS, OBJECTS_INFO, PROMPT, make_input, make_state

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeChallenge:
    def __init__(self, scene="mm_kitchen_3a-1"):
        self.data_dir = os.path.join("G:\\", "x", "room_setup_fire", scene)
        self.target_status = {4: False, 9: False, 13: False}


class TestCounters(unittest.TestCase):
    def test_before_the_first_episode_there_is_nothing(self):
        self.assertEqual(LiveCounters(lambda: None)(), {})

    def test_saved_counts_targets_dropped_in_the_bag_and_names_the_kitchen(self):
        ch = FakeChallenge()
        c = LiveCounters(lambda: ch)
        self.assertEqual(c(), {"saved": "0 of 3 targets", "kitchen": "mm_kitchen_3a-1"})
        ch.target_status[9] = 412  # challenge.py:180 stores the frame of the drop
        self.assertEqual(c()["saved"], "1 of 3 targets")

    def test_counted_chooser_pushes_counters_then_asks(self):
        calls = []

        class Page:
            def set_info(self, **k): calls.append(("info", k))
            def __call__(self, d): calls.append(("ask", d["n"])); return "A"

        ch = FakeChallenge("mm_kitchen_3a-7")
        self.assertEqual(CountedChooser(Page(), LiveCounters(lambda: ch))({"n": 3}), "A")
        self.assertEqual(calls, [("info", {"saved": "0 of 3 targets", "kitchen": "mm_kitchen_3a-7"}), ("ask", 3)])

    def test_score_is_the_mean_value_rate_over_eval_files(self):
        self.assertIsNone(score_from_evals({}))
        self.assertIsNone(score_from_evals({"a": "unreadable: x"}))
        evals = {"k1/eval_result.json": {"avg_score": 3.0, "avg_max_score": 12.0},
                 "k2/eval_result.json": {"avg_score": 6.0, "avg_max_score": 8.0}, "bad": "unreadable"}
        self.assertEqual(score_from_evals(evals), 0.5)  # (0.25 + 0.75) / 2


class TestEndToEnd(unittest.TestCase):
    """HumanAgent + CountedChooser + PageChooser as run_baseline.py wires them, driven through HTTP like a person."""

    def setUp(self):
        self.ch = FakeChallenge()
        self.agent, self.page = make_human_agent(task="fire", prompt_path=PROMPT, get_challenge=lambda: self.ch,
                                                 port=0, camera=False, open_browser=False)
        self.base = self.page.url.rstrip("/")
        self.save_dir = tempfile.mkdtemp()
        self.agent.reset(GOALS, OBJECTS_INFO)

    def tearDown(self):
        try:
            self.page.close()
        except Exception:
            pass

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return json.loads(r.read())

    def post(self, obj):
        req = urllib.request.Request(self.base + "/choose", data=json.dumps(obj).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def decide_in_thread(self, step):
        box = {}

        def run():
            box["action"] = self.agent.choose_target(make_state(), make_input(self.save_dir, step=step))
        t = threading.Thread(target=run, daemon=True)
        t.start()
        for _ in range(500):
            if self.page.state()["status"] == "waiting":
                break
            time.sleep(0.01)
        return t, box

    def test_agent_is_the_llm_agent_and_the_page_shows_counters_with_the_menu(self):
        self.assertEqual(self.agent.agent_type, "llm")  # challenge.py:257 dispatch, same path as Haiku
        self.assertIsNone(self.agent.anthropic_client)
        t, box = self.decide_in_thread(120)
        st = self.get("/state")
        self.assertEqual(st["status"], "waiting")
        self.assertEqual(st["info"], {"saved": "0 of 3 targets", "kitchen": "mm_kitchen_3a-1"})
        self.assertEqual([m["text"] for m in st["decision"]["menu"]],
                         ["go pick up object <hairbrush> (64)", "go pick up object <bag> (71)", "look around"])
        self.assertEqual(st["decision"]["frame"], 120)
        self.assertFalse(st["decision"]["has_frame"])  # camera=False
        self.assertEqual(self.post({"seq": 1, "letter": "A"})["ok"], True)
        t.join(5)
        self.assertEqual(box["action"], ("walk_to", "64"))
        self.assertEqual(self.agent.records[-1]["choice"], "A")

    def test_counters_move_between_decisions(self):
        t, box = self.decide_in_thread(120)
        self.post({"seq": 1, "letter": "C"})
        t.join(5)
        self.ch.target_status[4] = 300
        t, box = self.decide_in_thread(300)
        self.assertEqual(self.get("/state")["info"]["saved"], "1 of 3 targets")
        self.post({"seq": 2, "letter": "A"})
        t.join(5)
        self.assertEqual(self.page.choices, [(1, "C"), (2, "A")])

    def test_finish_puts_the_score_on_the_page_and_returns_the_summary(self):
        t, box = self.decide_in_thread(120)
        self.post({"seq": 1, "letter": "B"})
        t.join(5)
        self.ch.target_status[9] = 500
        evals = {"eval_result.json": {"avg_score": 5.5, "avg_max_score": 11.0}}
        before = time.perf_counter()
        out = finish_human_run(self.page, self.agent, evals, counters=LiveCounters(lambda: self.ch), hold_s=0)
        self.assertLess(time.perf_counter() - before, 1.0)  # hold_s=0 does not sleep
        self.assertEqual(out["score"], 0.5)
        self.assertEqual(out["clicks"], [{"seq": 1, "letter": "B"}])
        self.assertEqual((out["decisions"], out["asked"], out["automatic"], out["camera"]), (1, 1, 0, False))
        self.assertEqual(out["url"], self.page.url)
        with self.assertRaises(Exception):  # the server is closed once the hold is over
            self.get("/state")

    def test_finish_state_is_visible_while_held(self):
        self.ch.target_status[9] = 500
        evals = {"eval_result.json": {"avg_score": 1.0, "avg_max_score": 4.0}}
        box = {}

        def run():
            box["out"] = finish_human_run(self.page, self.agent, evals, counters=LiveCounters(lambda: self.ch), hold_s=0.5)
        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(0.1)
        st = self.get("/state")
        self.assertEqual((st["status"], st["decision"]), ("done", None))
        self.assertEqual(st["info"], {"saved": "1 of 3 targets", "kitchen": "mm_kitchen_3a-1", "score": 0.25})
        th.join(5)
        self.assertEqual(box["out"]["score"], 0.25)


class TestRunnerFlags(unittest.TestCase):
    def test_run_baseline_lists_the_human_flags(self):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_baseline.py"), "--help"],
                             capture_output=True, text=True, timeout=120).stdout
        for flag in ("--human_port", "--no_camera", "--no_browser", "--human_hold"):
            self.assertIn(flag, out)


if __name__ == "__main__":
    unittest.main()
