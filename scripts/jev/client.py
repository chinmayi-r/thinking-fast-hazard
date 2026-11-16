"""Jev (TypeSafe AI's "System One" decision model) as a fast brain for HAZARD. ASSUMPTIONS.md A37.

One request = HAZARD's prompt text as the `state` plus one `choice` question whose criteria are the lettered menu
lines of that prompt. The answer is a letter, which HAZARD's own parse_answer (policy/llm.py:342-380) accepts.
Vendor docs read Nov 16 2026: POST https://api.typesafe.ai/v1/systemone, bearer key, model alias `jev-latest`;
response `answers.<name>` with `choice`, `confidence`, `probabilities`; `usage.input_tokens/output_tokens`.

No network in tests: the HTTP layer is a callable `transport(request_dict) -> body_dict`, injectable.
The key is read from a file and only ever goes into the Authorization header; it is never logged or printed.
"""
import json
import math
import re
import time
import urllib.error
import urllib.request

JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
PRICE_PER_M_INPUT_USD = 0.042  # vendor list price, Nov 2026 (A37); output tokens are free
DEFAULT_INSTRUCTIONS = ("Pick the one available action that saves the most value of the target objects before they burn. "
                        "The state describes the house, the targets seen so far, what the robot is holding, the previous "
                        "actions and their results, and the available actions.")
_MENU_LINE = re.compile(r"^([A-Z])\.\s+(.+?)\s*$")


class JevError(RuntimeError):
    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def read_key(path):
    """First non-empty line of the key file. The value is returned, never printed."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    raise ValueError(f"no key in {path}")


def parse_menu(prompt_text):
    """[(letter, text)] from the 'Available actions:' block of a HAZARD prompt (lines 'A. ...' up to 'Answer')."""
    out, inside = [], False
    for raw in prompt_text.splitlines():
        s = raw.strip()
        if s.startswith("Available actions"):
            inside = True
            continue
        if not inside:
            continue
        if s.startswith("Answer"):
            break
        m = _MENU_LINE.match(s)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def build_request(prompt_text, menu, model=DEFAULT_MODEL, instructions=DEFAULT_INSTRUCTIONS, examples=None):
    """The JSON body. `examples`, if given, is prepended to the state (in-context examples, Liane's ablation)."""
    if not menu:
        raise ValueError("empty menu")
    state = prompt_text if not examples else f"{examples.rstrip()}\n\n{prompt_text}"
    return {"model": model, "state": state,
            "questions": {"action": {"type": "choice", "instructions": instructions,
                                     "criteria": {letter: text for letter, text in menu}}}}


def parse_response(body, menu):
    """letter / confidence / probabilities / usage / model from a response body; raises JevError if unusable."""
    try:
        ans = body["answers"]["action"]
    except (KeyError, TypeError):
        raise JevError(f"no 'answers.action' in response: {json.dumps(body)[:300]}")
    letter = ans.get("choice")
    letters = [l for l, _ in menu]
    if letter not in letters:
        raise JevError(f"choice {letter!r} is not one of {letters}")
    usage = body.get("usage") or {}
    return {"letter": letter, "confidence": ans.get("confidence"), "probabilities": ans.get("probabilities") or {},
            "input_tokens": int(usage.get("input_tokens") or 0), "output_tokens": int(usage.get("output_tokens") or 0),
            "model": body.get("model")}


class UrllibTransport:
    """POST the request as JSON with the bearer key. Raises JevError; 429 and 5xx are retryable."""

    def __init__(self, key, url=JEV_URL, timeout=30.0, urlopen=None):
        self._key = key
        self.url = url
        self.timeout = timeout
        self._urlopen = urlopen or urllib.request.urlopen

    def __call__(self, request_dict):
        data = json.dumps(request_dict).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "Authorization": f"Bearer {self._key}"})
        try:
            with self._urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = ""
            try:
                text = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise JevError(f"HTTP {e.code}: {text}", status=e.code, retryable=(e.code == 429 or e.code >= 500))
        except urllib.error.URLError as e:
            raise JevError(f"connection error: {e.reason}", retryable=True)
        except TimeoutError:
            raise JevError("timeout", retryable=True)


class JevClient:
    """decide(prompt_text) -> dict(letter, confidence, probabilities, wall_s, ...); every call appended to call_log
    in the A18 format (wall_s, input_tokens, output_tokens, stop_reason, model) plus confidence and frames_at_30fps."""

    def __init__(self, api_key_file=None, model=DEFAULT_MODEL, transport=None, timeout=30.0, retries=3,
                 instructions=DEFAULT_INSTRUCTIONS, sleep=time.sleep):
        if transport is None:
            if not api_key_file:
                raise ValueError("api_key_file is required (the key is never taken from the environment)")
            transport = UrllibTransport(read_key(api_key_file), timeout=timeout)
        self.transport = transport
        self.model = model
        self.retries = retries
        self.instructions = instructions
        self._sleep = sleep
        self.call_log = []
        self.total_cost = 0.0

    def decide(self, prompt_text, menu=None, examples=None):
        menu = menu or parse_menu(prompt_text)
        req = build_request(prompt_text, menu, model=self.model, instructions=self.instructions, examples=examples)
        attempt = 0
        t0 = time.perf_counter()
        while True:
            try:
                body = self.transport(req)
                break
            except JevError as e:
                attempt += 1
                if not e.retryable or attempt > self.retries:
                    raise
                self._sleep(0.5 * 2 ** (attempt - 1))
        dt = time.perf_counter() - t0
        out = parse_response(body, menu)
        out["wall_s"] = round(dt, 4)
        out["attempts"] = attempt + 1
        cost = out["input_tokens"] * PRICE_PER_M_INPUT_USD / 1e6
        self.total_cost += cost
        out["cost_usd"] = cost
        self.call_log.append({"wall_s": out["wall_s"], "frames_at_30fps": int(math.ceil(dt * 30)),
                              "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
                              "stop_reason": "choice", "model": out["model"] or self.model,
                              "confidence": out["confidence"], "attempts": out["attempts"]})
        return out
