"""Tests for scripts/human/page.py (A36, file 2). A real server on an ephemeral port, a real HTTP client (urllib),
the chooser called in a thread the way HAZARD's loop would call it. No HAZARD import needed.

Run from G:\\iw-thinking-fast:
  PYTHONPATH=scripts venv/Scripts/python.exe -X utf8 -m unittest human.test_page -v
"""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from human.page import PageChooser


def decision(n=1, png=None, rejected=(), map_png=None):
    return {"n": n, "frame": 120 + 40 * n, "frames_total": 1500, "automatic": False, "prompt": f"PROMPT {n}\nAvailable actions:\nA. x\nB. y\n", "map_png": map_png,
            "menu": [{"letter": "A", "text": "go pick up object <hairbrush> (64)", "action": ["walk_to", "64"]},
                     {"letter": "B", "text": "look around", "action": ["explore", "None"]}],
            "letters": ["A", "B"], "frame_png": png, "holding": [], "rejected": list(rejected)}


class TestPage(unittest.TestCase):
    def setUp(self):
        self.pc = PageChooser(port=0)  # ephemeral port, no browser
        self.base = self.pc.url.rstrip("/")

    def tearDown(self):
        self.pc.close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def ask_in_thread(self, d):
        box = {}
        def run():
            box["letter"] = self.pc(d)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        for _ in range(100):  # until the server shows it
            if self.pc.state()["status"] == "waiting":
                break
            time.sleep(0.01)
        return t, box

    def test_page_and_idle_state(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"paused after taking 100 steps", body)  # the rules name HAZARD's own wording
        self.assertIn(b"Picking up is automatic", body)
        st = json.loads(self.get("/state")[2])
        self.assertEqual(st, {"status": "idle", "seq": 0, "decision": None, "info": {}})

    def test_no_frame_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/frame.png")
        self.assertEqual(cm.exception.code, 404)

    def test_click_unblocks_the_agent_with_that_letter(self):
        t, box = self.ask_in_thread(decision(1))
        st = json.loads(self.get("/state")[2])
        self.assertEqual(st["status"], "waiting")
        self.assertEqual(st["seq"], 1)
        self.assertEqual(st["decision"]["prompt"], "PROMPT 1\nAvailable actions:\nA. x\nB. y\n")
        self.assertEqual([m["letter"] for m in st["decision"]["menu"]], ["A", "B"])
        self.assertNotIn("frame_png", st["decision"])  # local paths are not sent to the page
        self.assertFalse(st["decision"]["has_frame"])
        code, j = self.post("/choose", {"seq": 1, "letter": "b"})
        self.assertEqual((code, j["ok"]), (200, True))
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(box["letter"], "B")
        self.assertEqual(self.pc.state()["status"], "playing")
        self.assertEqual(self.pc.choices, [(1, "B")])

    def test_stale_or_bad_clicks_are_refused_and_do_not_unblock(self):
        code, j = self.post("/choose", {"seq": 1, "letter": "A"})  # nothing asked yet
        self.assertEqual(code, 409)
        self.assertIn("nothing", j["why"])
        t, box = self.ask_in_thread(decision(1))
        self.assertEqual(self.post("/choose", {"seq": 0, "letter": "A"})[0], 409)  # old page
        self.assertEqual(self.post("/choose", {"seq": 1, "letter": "Z"})[0], 409)  # not listed
        self.assertEqual(self.post("/choose", {"seq": 1})[0], 409)  # no letter
        time.sleep(0.05)
        self.assertTrue(t.is_alive())  # still blocked
        self.assertEqual(self.post("/choose", {"seq": 1, "letter": "A"})[0], 200)
        t.join(2)
        self.assertEqual(box["letter"], "A")

    def test_second_decision_gets_a_new_seq_and_the_frame(self):
        png = os.path.join(tempfile.mkdtemp(), "f.png")
        with open(png, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfake")
        t, box = self.ask_in_thread(decision(1))
        self.post("/choose", {"seq": 1, "letter": "A"})
        t.join(2)
        t2, box2 = self.ask_in_thread(decision(2, png=png))
        st = json.loads(self.get("/state")[2])
        self.assertEqual(st["seq"], 2)
        self.assertTrue(st["decision"]["has_frame"])
        code, ctype, body = self.get("/frame.png?seq=2")
        self.assertEqual((code, ctype), (200, "image/png"))
        self.assertEqual(body, b"\x89PNG\r\n\x1a\nfake")
        self.assertEqual(self.post("/choose", {"seq": 1, "letter": "A"})[0], 409)  # the old seq no longer counts
        self.post("/choose", {"seq": 2, "letter": "B"})
        t2.join(2)
        self.assertEqual(box2["letter"], "B")

    def test_info_and_finish(self):
        self.pc.set_info(saved=2)
        self.assertEqual(self.pc.state()["info"], {"saved": 2})
        t, box = self.ask_in_thread(decision(1))
        self.post("/choose", {"seq": 1, "letter": "A"})
        t.join(2)
        self.pc.finish(score=0.4)
        st = json.loads(self.get("/state")[2])
        self.assertEqual(st["status"], "done")
        self.assertIsNone(st["decision"])
        self.assertEqual(st["info"], {"saved": 2, "score": 0.4})
        self.assertEqual(self.post("/choose", {"seq": 1, "letter": "A"})[0], 409)

    def test_map_is_served_like_the_frame(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/map.png")
        self.assertEqual(cm.exception.code, 404)
        png = os.path.join(tempfile.mkdtemp(), "m.png")
        with open(png, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nmap")
        t, box = self.ask_in_thread(decision(1, map_png=png))
        st = json.loads(self.get("/state")[2])
        self.assertTrue(st["decision"]["has_map"])
        self.assertFalse(st["decision"]["has_frame"])
        self.assertNotIn("map_png", st["decision"])  # local paths are not sent to the page
        code, ctype, body = self.get("/map.png?seq=1")
        self.assertEqual((code, ctype, body), (200, "image/png", b"\x89PNG\r\n\x1a\nmap"))
        self.post("/choose", {"seq": 1, "letter": "A"})
        t.join(2)
        self.assertEqual(box["letter"], "A")

    def test_bad_json_is_400(self):
        req = urllib.request.Request(self.base + "/choose", data=b"{not json", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
