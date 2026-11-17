"""Jev as RRARA's reflector (A37 "not done" item; Tom Nov 10: "looking at Jev or other things that are designed to be fast").
The reflector of RRARA-v1 (scripts/rrara/reflector.py) builds the same scene text as the LLM baseline and asks one
question: is the action the reflex just started fine (VALID), or which lettered action should replace it. With Haiku
that answer takes about 4 s, so under the worldtime rule 24 % of verdicts arrive too late (NOTES Nov 10). Here the same
question goes to Jev as one typed "choice" question whose criteria are VALID plus the menu letters; Jev answers in
about 0.1-0.3 s. Everything else (prompt, history, parsing, the verdict record) is the Reflector's own code: this class
only replaces the generator, as JevAgent does for the LLM agent. Tests: scripts/jev/test_jev_reflector.py.
"""
from jev.client import DEFAULT_MODEL, JevClient, parse_menu
from rrara.reflector import Reflector

VALID_LETTER = "VALID"
VALID_TEXT = "the action I just started is fine; keep going"
INSTRUCTIONS = ("A robot has just started the action named under 'My proposed action'. Judge it against the scene. Answer "
                "VALID to keep it, or the letter of a better available action to replace it.")


class JevReflector(Reflector):
    def __init__(self, task, api_key_file="", model=DEFAULT_MODEL, client=None):
        super().__init__(task=task, lm_id=model, source="anthropic", api_key_file="", max_tokens=8)
        self.anthropic_client = None  # no Anthropic call can go out (llm.py:95-98 built one from an empty key)
        self.client = client or JevClient(api_key_file=api_key_file, model=model, instructions=INSTRUCTIONS)
        self.client.instructions = INSTRUCTIONS  # the reflector's question, whichever client was handed in
        self.call_log = self.client.call_log
        self.generator = self._judge  # Reflector.validate calls self.generator(messages, sampling_params)

    def _judge(self, messages, sampling_params):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        menu = [(VALID_LETTER, VALID_TEXT)] + parse_menu(prompt)
        r = self.client.decide(prompt, menu=menu)
        letter = r["letter"]
        # Reflector.validate reads "VALID" from the text, else parses a letter (parse_answer accepts a bare letter)
        return [VALID_LETTER if letter == VALID_LETTER else letter], r.get("cost_usd", 0.0)

    def summary(self):
        w = [c["wall_s"] for c in self.call_log]
        return {"model": self.lm_id, "calls": len(w), "wall_s_mean": round(sum(w) / len(w), 4) if w else None,
                "wall_s_max": round(max(w), 4) if w else None,
                "frames_at_30fps_total": sum(c["frames_at_30fps"] for c in self.call_log),
                "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in self.call_log), 6)}
