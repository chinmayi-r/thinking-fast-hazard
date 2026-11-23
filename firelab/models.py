"""One way to ask a language model (chunk 8a): the text of the room and lettered options in, an Answer out.

Two models stand behind it:
  Jev     TypeSafe's "System One" choice model. A port of scripts/jev/client.py (A37): one request carries the
          text as `state` and one `choice` question whose criteria are the lettered options.
  Claude  through the official `anthropic` package, written from the claude-api skill's Python reference
          (read Nov 24). claude-opus-5-5 cannot have thinking switched off, so the request leaves the thinking
          setting out and sets only output_config.effort; no temperature. The stop reason is checked BEFORE the
          answer is read: only "end_turn" is read as an answer. A refusal, a cut-off (max_tokens) or an API error
          is recorded as no answer, and no second model is asked in its place (CR, Nov 24).
A key is given as a file path only; its first line is read, handed to the client, and never printed or stored.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import NamedTuple

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
JEV_URL = "https://api.typesafe.ai/v1/systemone"
# Dollars per million tokens (input, output). Jev: vendor list price, Nov 2026 (A37). Opus 5.5: the claude-api
# skill's table cached Nov 3 2026 (thinking is billed as output). Check both on the day of a run.
PRICES = {"jev": (0.042, 0.0), "claude-opus-5-5": (4.0, 20.0), "fake": (0.0, 0.0)}


class Answer(NamedTuple):
    choice: int | None  # index of the option chosen; None = no answer
    raw: str  # what the model wrote (Jev: the letter)
    seconds: float  # wall time of the whole call, retries included
    tokens_in: int
    tokens_out: int
    stop_reason: str
    error: str  # "" or why there is no answer


def read_key(path) -> str:
    """The first non-empty line of the key file. Returned, never printed."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return line.strip()
    raise ValueError(f"no key in {path}")


def parse_letter(raw: str, n_options: int) -> int | None:
    """The option a written answer names: its first stand-alone capital letter that is one of the options."""
    for match in re.finditer(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", raw or ""):
        index = LETTERS.index(match.group(1))
        if index < n_options:
            return index
    return None


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES["jev" if model.startswith("jev") else "fake" if model.startswith("fake") else model]
    return (tokens_in * price_in + tokens_out * price_out) / 1e6


class Jev:
    def __init__(self, key_file=None, model: str = "jev-latest", transport=None, timeout: float = 30.0, retries: int = 3) -> None:
        self.model, self.retries = model, retries
        self.transport = transport or self._post
        self._key, self._timeout = (None if transport else read_key(key_file)), timeout

    def _post(self, body: dict) -> dict:
        request = urllib.request.Request(JEV_URL, data=json.dumps(body).encode("utf-8"), method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {self._key}"})
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def ask(self, text: str, options: list, instructions: str) -> Answer:
        body = {"model": self.model, "state": text, "questions": {"action": {
            "type": "choice", "instructions": instructions, "criteria": {LETTERS[n]: o for n, o in enumerate(options)}}}}
        began, attempt = time.perf_counter(), 0
        while True:
            try:
                reply = self.transport(body)
                break
            except (urllib.error.URLError, TimeoutError, ValueError) as e:  # HTTPError is a URLError
                status = getattr(e, "code", None)
                attempt += 1
                if attempt > self.retries or (status is not None and status != 429 and status < 500):
                    return Answer(None, "", time.perf_counter() - began, 0, 0, "error", f"{type(e).__name__} {status or ''}".strip())
                time.sleep(0.5 * 2 ** (attempt - 1))
        seconds, usage = time.perf_counter() - began, reply.get("usage") or {}
        tokens = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        letter = ((reply.get("answers") or {}).get("action") or {}).get("choice")
        if not isinstance(letter, str) or letter not in LETTERS[:len(options)]:
            return Answer(None, str(letter), seconds, *tokens, "choice", "answer is not one of the options")
        return Answer(LETTERS.index(letter), letter, seconds, *tokens, "choice", "")


class Claude:
    def __init__(self, key_file=None, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 16000, client=None) -> None:
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.errors: tuple = (Exception,)  # with a stand-in client (tests) any failure is "no answer"
        if client is None:
            import anthropic  # needed only where this model is run (della-vis1), so not at the top of the file

            client = anthropic.Anthropic(api_key=read_key(key_file))
            self.errors = (anthropic.APIStatusError, anthropic.APIConnectionError)  # after the SDK's own retries
        self.client = client

    def ask(self, text: str, options: list, instructions: str) -> Answer:
        began = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,  # room for the thinking, which counts against the limit
                system=instructions + " Answer with the letter of one option and nothing else.",
                output_config={"effort": self.effort},  # no `thinking` setting: it cannot be disabled on this model
                messages=[{"role": "user", "content": text}],
            )
        except self.errors as e:
            return Answer(None, "", time.perf_counter() - began, 0, 0, "error", f"{type(e).__name__} {getattr(e, 'status_code', '')}".strip())
        seconds = time.perf_counter() - began
        tokens = int(response.usage.input_tokens), int(response.usage.output_tokens)
        if response.stop_reason != "end_turn":  # refusal, max_tokens, ...: the content is not read
            return Answer(None, "", seconds, *tokens, str(response.stop_reason), "no answer: " + str(response.stop_reason))
        raw = "".join(block.text for block in response.content if block.type == "text").strip()
        choice = parse_letter(raw, len(options))
        return Answer(choice, raw, seconds, *tokens, "end_turn", "" if choice is not None else "no option letter in the answer")


class Fake:
    """A stand-in that always answers the same option (or none), for tests and dry runs. No network."""

    def __init__(self, choice: int | None = 0, seconds: float = 0.0) -> None:
        self.model, self.choice, self.seconds, self.asked = "fake", choice, seconds, []

    def ask(self, text: str, options: list, instructions: str) -> Answer:
        self.asked.append((text, list(options), instructions))
        choice = self.choice if self.choice is not None and self.choice < len(options) else None
        return Answer(choice, "" if choice is None else LETTERS[choice], self.seconds, 0, 0, "choice", "" if choice is not None else "no answer")


def make_model(name: str, key_file=None):
    """ "jev" or "jev-...", "claude-opus-5-5" (or another Claude model id), or "fake"."""
    if name.startswith("jev"):
        return Jev(key_file, name if name != "jev" else "jev-latest")
    if name.startswith("claude"):
        return Claude(key_file, name)
    if name == "fake":
        return Fake()
    raise ValueError(f"unknown model {name!r}")
