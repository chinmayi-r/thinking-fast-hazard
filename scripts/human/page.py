"""Human baseline for HAZARD, file 2 of 4: the page (ASSUMPTIONS.md A36).

PageChooser is a chooser for HumanAgent (file 1): a tiny HTTP server in a background thread that shows the current
decision at http://127.0.0.1:<port>/ and blocks the agent until the person clicks a letter. One process: HAZARD, the
agent and this server share it, so nothing is copied anywhere. The page (page.html, next to this file) has three
panels: the camera frame (toggleable; off = the text-only run), the prompt text and lettered buttons exactly as the
Haiku agent gets them, and the rules written for a person plus live counters.

Endpoints: GET /            the page
           GET /state       JSON {"status": idle|waiting|playing|done, "seq": n, "decision": {...} or null, "info": {...}}
           GET /frame.png   the current decision's camera frame (404 when there is none)
           POST /choose     JSON {"seq": n, "letter": "B"}; ignored unless seq is the decision being waited on

The runner (file 3) may push extra counters with set_info(saved=..., score=...) and must call finish() at the end.
"""
import json
import os
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "page.html")


class PageChooser:
    def __init__(self, host="127.0.0.1", port=8765, open_browser=False):
        self._lock = threading.Lock()
        self._answers = queue.Queue()
        self.seq = 0  # increases per decision shown; a click must quote it
        self.status = "idle"
        self.decision = None
        self.info = {}
        self.choices = []  # (seq, letter, wall_s) for the record
        chooser = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep HAZARD's console readable
                pass

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    with open(PAGE, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                elif path == "/state":
                    self._send(200, json.dumps(chooser.state()).encode("utf-8"))
                elif path in ("/frame.png", "/map.png"):
                    png = chooser._frame_path() if path == "/frame.png" else chooser._map_path()
                    if png and os.path.exists(png):
                        with open(png, "rb") as f:
                            self._send(200, f.read(), "image/png")
                    else:
                        self._send(404, b"no frame" if path == "/frame.png" else b"no map", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path.split("?")[0] != "/choose":
                    return self._send(404, b"not found", "text/plain")
                n = int(self.headers.get("Content-Length", "0"))
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    return self._send(400, b'{"ok": false, "why": "bad json"}')
                ok, why = chooser.submit(body.get("seq"), body.get("letter"))
                self._send(200 if ok else 409, json.dumps({"ok": ok, "why": why}).encode("utf-8"))

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.host, self.port = self.server.server_address[:2]
        self.url = f"http://{self.host}:{self.port}/"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="human-page")
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)

    # --- what the page reads ---
    def state(self):
        with self._lock:
            d = None
            if self.decision is not None:
                d = {k: v for k, v in self.decision.items() if k not in ("frame_png", "map_png")}
                d["has_frame"] = bool(self.decision.get("frame_png"))
                d["has_map"] = bool(self.decision.get("map_png"))
            return {"status": self.status, "seq": self.seq, "decision": d, "info": dict(self.info)}

    def _frame_path(self):
        with self._lock:
            return self.decision.get("frame_png") if self.decision else None

    def _map_path(self):
        with self._lock:
            return self.decision.get("map_png") if self.decision else None

    # --- what the page sends ---
    def submit(self, seq, letter):
        with self._lock:
            if self.status != "waiting":
                return False, "nothing is being asked"
            if seq != self.seq:
                return False, f"stale: page shows {seq}, agent is at {self.seq}"
            letter = (letter or "").strip().upper()
            if letter not in (self.decision or {}).get("letters", []):
                return False, f"not an option: {letter!r}"
        self._answers.put((seq, letter))
        return True, "ok"

    # --- the chooser protocol (HumanAgent calls this, in HAZARD's thread, and blocks) ---
    def __call__(self, decision):
        with self._lock:
            self.seq += 1
            self.decision = decision
            self.status = "waiting"
            seq = self.seq
        while True:
            got_seq, letter = self._answers.get()
            if got_seq == seq:
                break
        with self._lock:
            self.status = "playing"
            self.choices.append((seq, letter))
        return letter

    # --- for the runner (file 3) ---
    def set_info(self, **counters):
        with self._lock:
            self.info.update(counters)

    def finish(self, **counters):
        with self._lock:
            self.info.update(counters)
            self.status = "done"
            self.decision = None

    def close(self):
        self.server.shutdown()
        self.server.server_close()
