"""Tests for Jev as RRARA's reflector (scripts/jev/jev_reflector.py) and its wiring in RRARAv1Agent. A scripted transport
stands in for the API; HAZARD's LLM text code runs for real on a synthetic state. No simulator, no network.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -W error::RuntimeWarning -m unittest jev.test_jev_reflector -v
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from human.test_human_agent import GOALS, OBJECTS_INFO, make_input
from human.test_human_agent import make_state as _make_state


def make_state():
    st = _make_state()
    st['sem_map']['height'] = np.zeros_like(st['sem_map']['explored'], dtype=np.float32)  # the reflex plans on the height layer
    return st
from jev.client import JevClient
from jev.jev_reflector import INSTRUCTIONS, VALID_LETTER, JevReflector
from jev.test_jev import FakeTransport
from rrara.agent import RRARAv1Agent

RUNNER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_baseline.py")


def client(items):
    return JevClient(api_key_file="", transport=FakeTransport(items), sleep=lambda s: None)


class TestJevReflector(unittest.TestCase):
    def test_valid_verdict_and_request_shape(self):
        c = client([VALID_LETTER])
        r = JevReflector(task="fire", client=c)
        r.reset(GOALS, OBJECTS_INFO)
        v = r.validate(make_state(), make_input(tempfile.mkdtemp()), ("walk_to", "64"))
        self.assertTrue(v["valid"])
        self.assertIsNone(v["replacement"])
        self.assertEqual(v["proposed"], ["walk_to", "64"])
        req = c.transport.requests[0]
        crit = req["questions"]["action"]["criteria"]
        self.assertEqual(list(crit)[0], VALID_LETTER)  # VALID first, then the menu letters
        self.assertIn("A", crit)
        self.assertEqual(req["questions"]["action"]["instructions"], INSTRUCTIONS)
        self.assertIn("ALREADY STARTED this action", req["state"])  # the reflector's own prompt template
        self.assertIsNone(r.anthropic_client)
        self.assertEqual(r.summary()["calls"], 1)

    def test_letter_verdict_replaces_and_same_letter_is_valid(self):
        # the menu for this state (rule_based/llm order): A = walk to hairbrush 64, B = walk to bag 71, C = look around
        c = client(["B", "A"])
        r = JevReflector(task="fire", client=c)
        r.reset(GOALS, OBJECTS_INFO)
        v = r.validate(make_state(), make_input(tempfile.mkdtemp()), ("walk_to", "64"))
        self.assertFalse(v["valid"])
        self.assertEqual(v["replacement"], ["walk_to", "71"])
        v2 = r.validate(make_state(), make_input(tempfile.mkdtemp()), ("walk_to", "64"))
        self.assertTrue(v2["valid"])  # it named the action already started
        self.assertEqual(len(r.verdicts), 2)
        self.assertGreater(r.summary()["frames_at_30fps_total"], 0)


class TestAgentWiring(unittest.TestCase):
    def test_agent_uses_jev_reflector_and_overrides(self):
        c = client(["B"] * 5)
        agent = RRARAv1Agent(task="fire", mode="faithful", reflector="jev", reflector_client=c)
        agent.reset(GOALS, OBJECTS_INFO)
        self.assertEqual(type(agent.reflector).__name__, "JevReflector")
        d = tempfile.mkdtemp()
        a1 = agent.choose_target(make_state(), make_input(d))  # reflex proposes; the check starts in the background
        self.assertEqual(a1[0], "walk_to")
        time.sleep(0.3)  # the fake transport answers at once; let the thread finish
        a2 = agent.choose_target(make_state(), make_input(d, step=160))
        rec = agent.decision_log[-1]
        self.assertEqual(rec["verdict_status"], "used")
        s = agent.rrara_summary()
        self.assertEqual(s["reflector_kind"], "jev")
        self.assertGreaterEqual(s["reflector"]["calls"], 1)
        self.assertIn("frames_at_30fps_total", s["reflector"])
        agent.job.shutdown()

    def test_mode_off_builds_no_reflector(self):
        agent = RRARAv1Agent(task="fire", mode="off", reflector="jev")
        self.assertIsNone(agent.reflector)

    def test_runner_flags(self):
        out = subprocess.run([sys.executable, RUNNER, "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-500:])
        self.assertIn("--reflector ", out.stdout.replace("\n", " "))
        self.assertIn("--reflector_key_file", out.stdout)


if __name__ == "__main__":
    unittest.main()
