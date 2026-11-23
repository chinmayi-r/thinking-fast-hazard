"""Tests for firelab.compile and firelab.validate (Liane's method, offline; A99). No network.

The writing step is driven by a stand-in for the model; one 6-item room is built to play three rooms (about 20 s).
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_compile.py
"""

import json
from pathlib import Path

import pytest

from firelab import gate
from firelab.compile import EXAMPLE, build_prompt, compile_round, extract_module, main, render_game
from firelab.env import FireLab3DEnv
from firelab.models import Fake, cost_usd
from firelab.run_rooms import play
from firelab.skills import Skills
from firelab.validate import play_rooms

SIX = gate.Task(items=6, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)


@pytest.fixture(scope="module")
def games(tmp_path_factory):
    """Two saved games of a stand-in "slow model" that always takes the first option."""
    folder = tmp_path_factory.mktemp("games")
    env = FireLab3DEnv(level=2, num_items=6, realistic_bg=False)
    try:
        skills = Skills(env)
        for room in (100, 101):
            play(env, skills, SIX, room, "model", games=folder, model=Fake(0, seconds=2.0), options={"call_time": "measured"})
    finally:
        env.close()
    return folder


def test_a_game_is_rendered_decision_by_decision(games):
    text = render_game(games / "room_100_model.jsonl", max_decisions=3)
    assert text.startswith("## Room 100: final score ") and "Decision 1\nTime: tick 0;" in text and "Decision 4" not in text
    assert "\nYou: cell (9, 7), health 100 of 100." in text and "\nItems on the floor:\n  item 0 at" in text
    assert "\nChose: get item " in text and "after 2.0 s of thinking (20 ticks standing)\nResult: ok, " in text
    assert "later decisions not shown)" in text and "Map (" not in text and "Costs:" not in text


def test_the_prompt_carries_the_api_the_example_the_games_and_the_last_round():
    prompt = build_prompt("API TEXT", "class Policy: pass", "GAMES", previous_src="OLD PROGRAM", validation={"mean": 0.5}, full_text="FULL")
    order = [prompt.index(s) for s in ("API TEXT", "class Policy: pass", "FULL", "GAMES", "OLD PROGRAM", '"mean": 0.5', "# Task")]
    assert order == sorted(order)
    assert "previous round" not in build_prompt("A", "B", "C") and extract_module("talk\n```python\nx = 1\n```\nmore\n```python\ny = 2\n```") == "y = 2\n"


def test_a_round_writes_the_prompt_the_reply_and_an_audited_program(games, tmp_path):
    dry = tmp_path / "dry"
    assert main(["--games", str(games), "--out", str(dry), "--round", "1", "--dry-run"]) == 0
    prompt = (dry / "prompt_round1.md").read_text(encoding="utf-8")
    assert "# The program API for the fire task" in prompt and "## Room 100:" in prompt and "## Room 101:" in prompt and "Map (" in prompt
    assert not (dry / "policy_round1.py").exists() and json.loads((dry / "compile_round1.json").read_text())["dry_run"] is True

    good = "Ten lines of talk.\n```python\n" + Path(EXAMPLE).read_text(encoding="utf-8") + "```\n"
    record = compile_round(tmp_path / "ok", 1, "GAMES", caller=lambda prompt: good)
    assert record["accepted"] and record["audit"] == [] and (tmp_path / "ok" / "reply_round1.md").exists()
    bad = compile_round(tmp_path / "bad", 1, "GAMES", caller=lambda prompt: "```python\nimport os\nclass Policy:\n    pass\n```")
    assert not bad["accepted"] and "import of 'os' is not allowed" in bad["audit"]
    none = compile_round(tmp_path / "none", 1, "GAMES", caller=lambda prompt: None)
    assert not none["accepted"] and none["audit"] == ["no program: the reply did not end normally"]


def test_validation_plays_the_program_and_the_reflex_in_each_room(tmp_path):
    record = play_rooms(EXAMPLE, [110, 111, 112], items=6)
    assert record["n"] == 3 and set(record["rooms"]) == {"110", "111", "112"}
    for row in record["rooms"].values():  # the example is the reflex written as a program: the same score in every room
        assert row["score"] == row["reflex"] and row["illegal"] == 0 and row["raised"] == 0 and row["seconds_per_decision"] < 0.1
    assert record["gap_to_reflex"] == 0.0 and record["ahead_level_behind"] == [0, 3, 0] and record["mean"] == record["reflex_mean"]
    refused = tmp_path / "refused.py"
    refused.write_text("import socket\nclass Policy:\n    def __init__(self, rules):\n        pass\n    def decide(self, obs):\n        return None\n")
    with pytest.raises(ValueError, match="program refused"):
        play_rooms(refused, [110])


def test_haiku_has_a_price():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
