"""Tests for firelab.models (chunk 8a). No network and no simulator: both models are driven through stand-ins.

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_models.py
"""

import urllib.error
from types import SimpleNamespace

import pytest

from firelab.models import Answer, Claude, Fake, Jev, cost_usd, make_model, parse_letter, read_key

OPTIONS = ["get item 3", "go home", "wait 10 ticks"]


def test_read_key_takes_the_first_line_and_parse_letter_finds_the_option(tmp_path):
    key = tmp_path / "key.txt"
    key.write_text("\n  sk-not-a-real-key  \nsecond line\n")
    assert read_key(key) == "sk-not-a-real-key"
    assert [parse_letter(raw, 3) for raw in ("B", "B.", " b ", "Answer: C", "I pick A because B burns", "D", "", "Get it")] == \
           [1, 1, None, 2, 0, None, None, None]


def test_jev_sends_one_choice_question_and_reads_the_letter():
    sent = []

    def transport(body):
        sent.append(body)
        return {"answers": {"action": {"choice": "B", "confidence": 0.9}}, "usage": {"input_tokens": 812, "output_tokens": 1}}

    answer = Jev(transport=transport).ask("THE ROOM", OPTIONS, "Pick the best.")
    assert sent == [{"model": "jev-latest", "state": "THE ROOM", "questions": {"action": {
        "type": "choice", "instructions": "Pick the best.", "criteria": {"A": "get item 3", "B": "go home", "C": "wait 10 ticks"}}}}]
    assert answer[:2] == (1, "B") and (answer.tokens_in, answer.tokens_out, answer.error) == (812, 1, "") and answer.seconds >= 0
    off_menu = Jev(transport=lambda body: {"answers": {"action": {"choice": "Q"}}}).ask("T", OPTIONS, "I")
    assert off_menu.choice is None and off_menu.error == "answer is not one of the options"


def test_jev_gives_no_answer_on_an_error_it_cannot_retry():
    def refused(body):
        raise urllib.error.HTTPError("u", 401, "unauthorized", None, None)

    answer = Jev(transport=refused).ask("T", OPTIONS, "I")
    assert answer.choice is None and answer.stop_reason == "error" and answer.error == "HTTPError 401"


class _Client:
    """Stands in for anthropic.Anthropic: records the request, returns a canned response."""

    def __init__(self, response=None, error=None):
        self.request, self._response, self._error = None, response, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.request = kwargs
        if self._error:
            raise self._error
        return self._response


class _Unreadable:
    """A response whose content must not be touched."""

    def __init__(self, stop_reason):
        self.stop_reason, self.usage = stop_reason, SimpleNamespace(input_tokens=900, output_tokens=40)

    @property
    def content(self):
        raise AssertionError("the content was read before the stop reason was checked")


def test_claude_sets_effort_and_nothing_about_thinking_and_reads_only_text_blocks():
    blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=" B\n")]
    client = _Client(SimpleNamespace(stop_reason="end_turn", content=blocks, usage=SimpleNamespace(input_tokens=900, output_tokens=120)))
    answer = Claude(client=client).ask("THE ROOM", OPTIONS, "Pick the best.")
    assert answer == Answer(1, "B", answer.seconds, 900, 120, "end_turn", "")
    request = client.request
    assert request["model"] == "claude-opus-5-5" and request["output_config"] == {"effort": "low"} and request["max_tokens"] == 16000
    assert request["messages"] == [{"role": "user", "content": "THE ROOM"}] and request["system"].startswith("Pick the best.")
    assert not {"thinking", "temperature", "top_p", "tool_choice", "fallbacks"} & set(request)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_claude_checks_the_stop_reason_before_reading_the_answer(stop_reason):
    answer = Claude(client=_Client(_Unreadable(stop_reason))).ask("T", OPTIONS, "I")
    assert answer.choice is None and answer.stop_reason == stop_reason and answer.error == "no answer: " + stop_reason
    assert (answer.tokens_in, answer.tokens_out) == (900, 40)  # still counted: they are billed


def test_claude_api_error_and_wordy_answer_and_the_rest():
    failed = Claude(client=_Client(error=RuntimeError("boom"))).ask("T", OPTIONS, "I")
    assert failed.choice is None and failed.stop_reason == "error" and failed.error == "RuntimeError"
    wordy = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="none of these")], usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    assert Claude(client=_Client(wordy)).ask("T", OPTIONS, "I").error == "no option letter in the answer"
    assert Fake(1).ask("T", OPTIONS, "I").choice == 1 and Fake(None).ask("T", OPTIONS, "I").choice is None and Fake(7).ask("T", OPTIONS, "I").choice is None
    assert cost_usd("jev-latest", 1_000_000, 5) == pytest.approx(0.042) and cost_usd("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(6.0)
    assert isinstance(make_model("fake"), Fake)
    with pytest.raises(ValueError):
        make_model("gpt")
