"""Tests for scripts/jev/ (A37): the request/response mapping, the client with a fake transport (no network, no key),
the bearer header, and the agent on the real policy.llm.LLM code with HAZARD's own menu builder and answer parser.

Run from G:\\iw-thinking-fast:
  PYTHONPATH="HAZARD;HAZARD/src/HAZARD;scripts" venv/Scripts/python.exe -X utf8 -m unittest jev.test_jev -v
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error

import numpy as np

from jev.client import (JevClient, JevError, UrllibTransport, build_request, parse_menu, parse_response, read_key,
                        PRICE_PER_M_INPUT_USD, DEFAULT_MODEL)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROMPT_CSV = os.path.join(ROOT, "HAZARD", "src", "HAZARD", "policy", "llm_configs", "prompt_v2.csv")

# the text of decision 3 of the Nov 16 sitting (results/human_cr_text/mm_kitchen_3a-1), as HAZARD printed it
SAMPLE_PROMPT = """Suppose I am wearing fireproof suits and my house is on fire. I want to save as many valuable target objects as possible by picking them up and putting them into my bag.
Target objects:
name: bag, handbag, pocketbook, purse, value: 5, attribute: None
name: hairbrush, value: 1, attribute: None

Current State:
Target objects currently seen:
name: hairbrush, id: 66, value: 1, distance: 8.73 m, temperature: 107.2 Celsius

Previous actions:
look around (success), go pick up object <hairbrush> (66) (paused after taking 100 steps)

Objects states history:
$OBJECT_HISTORY$
Available actions:
A. go pick up object <bag, handbag, pocketbook, purse> (56)
B. go pick up object <hairbrush> (57)
C. go pick up object <hairbrush> (64)
D. go pick up object <hairbrush> (66)
E. look around

Answer:"""
SAMPLE_MENU = [("A", "go pick up object <bag, handbag, pocketbook, purse> (56)"), ("B", "go pick up object <hairbrush> (57)"),
               ("C", "go pick up object <hairbrush> (64)"), ("D", "go pick up object <hairbrush> (66)"), ("E", "look around")]


def body_for(letter, menu, model="jev-1.13.0", in_tok=900):
    letters = [l for l, _ in menu]
    probs = {l: (0.8 if l == letter else 0.2 / max(1, len(letters) - 1)) for l in letters}
    return {"model": model, "answers": {"action": {"choice": letter, "confidence": 0.8, "probabilities": probs}},
            "usage": {"input_tokens": in_tok, "output_tokens": 0}}


class FakeTransport:
    """Scripted responses; an item may be a JevError to raise instead. Records every request it was given."""
    def __init__(self, items):
        self.items = list(items)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(json.dumps(request)))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        menu = [(l, t) for l, t in request["questions"]["action"]["criteria"].items()]
        return body_for(item, menu) if isinstance(item, str) else item


class TestMenuAndRequest(unittest.TestCase):
    def test_menu_is_the_lettered_block_of_the_prompt(self):
        self.assertEqual(parse_menu(SAMPLE_PROMPT), SAMPLE_MENU)

    def test_menu_ignores_lines_outside_the_block(self):
        self.assertEqual(parse_menu("A. not a menu\nAvailable actions:\nB. yes\nAnswer:\nC. no"), [("B", "yes")])
        self.assertEqual(parse_menu("no menu here"), [])

    def test_request_is_one_choice_question_over_the_menu(self):
        req = build_request(SAMPLE_PROMPT, SAMPLE_MENU)
        self.assertEqual(req["model"], DEFAULT_MODEL)
        self.assertEqual(req["state"], SAMPLE_PROMPT)
        q = req["questions"]["action"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(q["criteria"], dict(SAMPLE_MENU))
        self.assertTrue(q["instructions"])

    def test_examples_are_prepended_to_the_state(self):
        req = build_request(SAMPLE_PROMPT, SAMPLE_MENU, examples="Example: ... -> B\n")
        self.assertTrue(req["state"].startswith("Example: ... -> B\n\nSuppose I am"))

    def test_empty_menu_is_refused(self):
        with self.assertRaises(ValueError):
            build_request(SAMPLE_PROMPT, [])


class TestParseResponse(unittest.TestCase):
    def test_letter_confidence_probabilities_usage(self):
        out = parse_response(body_for("D", SAMPLE_MENU), SAMPLE_MENU)
        self.assertEqual(out["letter"], "D")
        self.assertEqual(out["confidence"], 0.8)
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)
        self.assertEqual((out["input_tokens"], out["output_tokens"], out["model"]), (900, 0, "jev-1.13.0"))

    def test_choice_outside_the_menu_is_an_error(self):
        with self.assertRaises(JevError):
            parse_response(body_for("Z", SAMPLE_MENU), SAMPLE_MENU)

    def test_missing_answer_is_an_error(self):
        with self.assertRaises(JevError):
            parse_response({"answers": {}}, SAMPLE_MENU)


class TestClient(unittest.TestCase):
    def make(self, items):
        t = FakeTransport(items)
        slept = []
        c = JevClient(transport=t, sleep=slept.append)
        return c, t, slept

    def test_decide_returns_the_letter_and_logs_the_call_in_a18_format(self):
        c, t, _ = self.make(["B"])
        out = c.decide(SAMPLE_PROMPT)
        self.assertEqual(out["letter"], "B")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(t.requests[0]["questions"]["action"]["criteria"], dict(SAMPLE_MENU))  # menu parsed from the prompt
        self.assertEqual(len(c.call_log), 1)
        log = c.call_log[0]
        for k in ("wall_s", "frames_at_30fps", "input_tokens", "output_tokens", "stop_reason", "model", "confidence"):
            self.assertIn(k, log)
        self.assertEqual(log["stop_reason"], "choice")
        self.assertEqual(log["model"], "jev-1.13.0")  # the id the API reported, not the alias
        self.assertGreaterEqual(log["frames_at_30fps"], 1)
        self.assertAlmostEqual(c.total_cost, 900 * PRICE_PER_M_INPUT_USD / 1e6)

    def test_retryable_error_is_retried_with_backoff(self):
        c, t, slept = self.make([JevError("HTTP 503", status=503, retryable=True), JevError("HTTP 429", status=429, retryable=True), "E"])
        out = c.decide(SAMPLE_PROMPT)
        self.assertEqual(out["letter"], "E")
        self.assertEqual(out["attempts"], 3)
        self.assertEqual(slept, [0.5, 1.0])
        self.assertEqual(len(c.call_log), 1)

    def test_non_retryable_error_raises_at_once(self):
        c, t, slept = self.make([JevError("HTTP 400", status=400, retryable=False), "A"])
        with self.assertRaises(JevError):
            c.decide(SAMPLE_PROMPT)
        self.assertEqual(slept, [])
        self.assertEqual(c.call_log, [])  # nothing logged for a failed decision

    def test_gives_up_after_retries(self):
        c, t, slept = self.make([JevError("x", retryable=True)] * 4)
        with self.assertRaises(JevError):
            c.decide(SAMPLE_PROMPT)
        self.assertEqual(len(slept), 3)

    def test_no_key_and_no_transport_is_refused(self):
        with self.assertRaises(ValueError):
            JevClient()


class TestTransportAndKey(unittest.TestCase):
    def test_key_file_first_non_empty_line(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n  \ntest-key-not-real-123\nsecond\n")
        try:
            self.assertEqual(read_key(f.name), "test-key-not-real-123")
        finally:
            os.unlink(f.name)

    def test_bearer_header_carries_the_key_and_the_body_does_not(self):
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=None):
            seen["headers"] = dict(req.header_items())
            seen["body"] = req.data.decode("utf-8")
            seen["url"] = req.full_url
            return Resp(json.dumps(body_for("C", SAMPLE_MENU)).encode("utf-8"))

        t = UrllibTransport("test-key-not-real-123", urlopen=fake_urlopen)
        body = t(build_request(SAMPLE_PROMPT, SAMPLE_MENU))
        self.assertEqual(body["answers"]["action"]["choice"], "C")
        self.assertEqual(seen["headers"].get("Authorization"), "Bearer test-key-not-real-123")
        self.assertNotIn("test-key-not-real-123", seen["body"])
        self.assertTrue(seen["url"].startswith("https://api.typesafe.ai/"))

    def test_http_errors_map_to_retryable_flags(self):
        def fail(code):
            def fake_urlopen(req, timeout=None):
                raise urllib.error.HTTPError(req.full_url, code, "err", {}, io.BytesIO(b"{}"))
            return fake_urlopen
        for code, retryable in ((429, True), (503, True), (400, False), (401, False)):
            with self.assertRaises(JevError) as cm:
                UrllibTransport("k", urlopen=fail(code))(build_request(SAMPLE_PROMPT, SAMPLE_MENU))
            self.assertEqual((cm.exception.status, cm.exception.retryable), (code, retryable))


# --- the agent on HAZARD's real LLM code (needs HAZARD/src/HAZARD on PYTHONPATH, as the human tests do) ---
try:
    from jev.jev_agent import JevAgent
    from human.test_human_agent import make_state, make_input, GOALS, OBJECTS_INFO
    HAVE_HAZARD = True
except Exception:  # pragma: no cover
    HAVE_HAZARD = False


@unittest.skipUnless(HAVE_HAZARD, "HAZARD not importable (set PYTHONPATH as in the docstring)")
class TestJevAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def make(self, answers):
        t = FakeTransport(answers)
        client = JevClient(transport=t)
        agent = JevAgent(task="fire", prompt_template_path=PROMPT_CSV, client=client)
        agent.reset(GOALS, OBJECTS_INFO)
        return agent, t

    def test_menu_is_hazards_and_the_letter_picks_the_action(self):
        agent, t = self.make(["B"])
        action = agent.choose_target(make_state(), make_input(self.tmp))
        self.assertEqual(action, ("walk_to", "71"))  # B = the bag, through HAZARD's parse_answer
        req = t.requests[0]
        self.assertEqual(list(req["questions"]["action"]["criteria"].values()),
                         ["go pick up object <hairbrush> (64)", "go pick up object <bag> (71)", "look around"])
        self.assertIn("Target objects currently seen:", req["state"])
        self.assertIn("A. go pick up object <hairbrush> (64)", req["state"])
        self.assertNotIn("step by step", req["state"])  # cot off: one call, one letter (A37)
        self.assertEqual(agent.decisions[0]["letter"], "B")
        self.assertEqual(agent.decisions[0]["plan"], "go pick up object <bag> (71)")
        self.assertIs(agent.call_log, agent.client.call_log)
        self.assertEqual(len(agent.call_log), 1)

    def test_pick_up_and_drop_are_automatic_without_a_call(self):
        agent, t = self.make([])
        nearest = [{"name": "hairbrush", "category": "hairbrush", "id": "64"}]
        self.assertEqual(agent.choose_target(make_state(), make_input(self.tmp, nearest=nearest)), ("pick_up", "64"))
        holding = [{"name": "bag", "category": "bag", "id": "71"}]
        self.assertEqual(agent.choose_target(make_state(), make_input(self.tmp, holding=holding)), ("drop", None))
        self.assertEqual(t.requests, [])
        self.assertEqual(agent.summary()["decisions"], 0)

    def test_summary_and_reset(self):
        agent, t = self.make(["C", "A"])
        agent.choose_target(make_state(), make_input(self.tmp))
        agent.choose_target(make_state(), make_input(self.tmp, step=200))
        s = agent.summary()
        self.assertEqual(s["decisions"], 2)
        self.assertEqual(s["confidence_mean"], 0.8)
        self.assertEqual(s["models"], ["jev-1.13.0"])
        self.assertAlmostEqual(s["cost_usd"], 2 * 900 * PRICE_PER_M_INPUT_USD / 1e6, places=9)
        agent.reset(GOALS, OBJECTS_INFO)
        self.assertEqual(agent.decisions, [])

    def test_api_failure_ends_the_episode_instead_of_guessing(self):
        agent, t = self.make([JevError("HTTP 401", status=401, retryable=False)])
        with self.assertRaises(JevError):
            agent.choose_target(make_state(), make_input(self.tmp))


class TestRunnerFlag(unittest.TestCase):
    def test_runner_help_lists_jev(self):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_baseline.py"), "--help"],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--jev_examples", out.stdout)
        self.assertIn("jev", out.stdout)


if __name__ == "__main__":
    unittest.main()
