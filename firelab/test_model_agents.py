"""Tests for firelab.model_agents (chunk 8b). One 6-item room is built; room 3 is played with stand-in models, no network.

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_model_agents.py
"""

import json

import pytest

import firelab  # noqa: F401
from firelab import gate
from firelab.agents import reflex
from firelab.env import FireLab3DEnv
from firelab.interface import Session
from firelab.model_agents import call_cost, model_every_command, rrara_v1
from firelab.models import Answer, Fake
from firelab.run_rooms import COLUMNS, play
from firelab.skills import Skills

SIX = gate.Task(items=6, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)


@pytest.fixture(scope="module")
def robot():
    env = FireLab3DEnv(level=2, num_items=6, realistic_bg=False)
    yield env, Skills(env)
    env.close()


def _fresh(robot) -> Session:
    env, skills = robot
    env.reset(seed=3)
    skills.start()
    return Session(skills)


def test_call_cost_gives_both_times_and_charges_by_the_toggle():
    answer = Answer(0, "A", 1.23, 900, 5, "end_turn", "")
    assert call_cost(answer, "measured", 0.3) == {"measured_s": 1.23, "declared_s": 0.3, "ticks_measured": 13, "ticks_declared": 3, "charged": 13}
    assert call_cost(answer, "declared", 0.3)["charged"] == 3


def test_a_checker_that_always_says_valid_leaves_the_rule_alone(robot):
    env, skills = robot
    env.reset(seed=3)
    skills.start()
    ours = [(name, k, tick, result.done) for name, k, tick, result in reflex(skills)]
    score = skills.score()
    records = []
    log = rrara_v1(_fresh(robot), Fake(0, seconds=5.0), accounting="strict", records=records)
    assert [(name, k, tick, result.done) for name, k, tick, result in log] == ours and skills.score() == score
    assert len(records) == len(log) and records[0]["previous_check"] == "none" and {r["previous_check"] for r in records[1:]} <= {"valid", "late"}
    first = records[0]
    assert first["options"][0].startswith("VALID: keep the rule's command, get item") and "A fast rule has ALREADY STARTED this command: get item" in first["text"]
    assert first["charged"] == 50 and first["ready_tick"] == 50 and first["rule"] == first["command"]
    json.dumps(records)  # a saved game is plain JSON


def test_an_invalid_answer_replaces_the_rules_next_choice_unless_it_comes_too_late(robot):
    records = []
    log = rrara_v1(_fresh(robot), Fake(1), accounting="free", records=records)  # always "invalid, take the first other option"
    assert records[1]["previous_check"] == "override" and records[1]["command"] != records[1]["rule"]
    assert log[0][0] == "get" and len(log) >= 2
    late = []
    rrara_v1(_fresh(robot), Fake(1, seconds=100.0), accounting="strict", records=late)  # 1000 ticks: never in time
    assert {r["previous_check"] for r in late[1:]} == {"late"} and all(r["command"] == r["rule"] for r in late)
    declared = []
    rrara_v1(_fresh(robot), Fake(1, seconds=100.0), accounting="strict", call_time="declared", declared_seconds=0.2, records=declared)
    assert declared[1]["previous_check"] == "override" and declared[0]["charged"] == 2  # the same model, declared fast


def test_no_answer_means_the_rule_stands(robot):
    env, skills = robot
    env.reset(seed=3)
    skills.start()
    ours = [(name, k, tick) for name, k, tick, _ in reflex(skills)]
    records = []
    log = rrara_v1(_fresh(robot), Fake(None), records=records)
    assert [(name, k, tick) for name, k, tick, _ in log] == ours and {r["previous_check"] for r in records[1:]} == {"no answer"}


def test_the_model_before_every_command_stands_while_it_answers(robot):
    records = []
    session = _fresh(robot)
    log = model_every_command(session, Fake(0, seconds=2.0), records=records)  # always the first option: the most valuable item
    assert log[0][0] == "get" and records[0]["charged"] == 20 and records[0]["command"][0] == "get"
    assert log[0][2] >= 20 + log[0][3].ticks and "Options:\nA. get item" in records[0]["text"]
    silent, session = [], _fresh(robot)
    log = model_every_command(session, Fake(None), records=silent)
    assert all(name == "wait" for name, *_ in log) and session.skills.over and session.skills.score() == 0.0


def test_the_runner_plays_both_model_agents_and_saves_what_the_model_saw(robot, tmp_path):
    env, skills = robot
    options = {"accounting": "strict", "call_time": "declared", "declared_seconds": 0.3}
    rows = [play(env, skills, SIX, 3, name, games=tmp_path, model=Fake(0), options=options) for name in ("model", "rrara1")]
    assert all(list(row) == COLUMNS and row["rule"] == "" and row["thoughts"] > 0 and row["cost_usd"] == 0.0 for row in rows)
    assert [row["clock"] for row in rows] == ["declared", "declared-strict"]
    game = [json.loads(line) for line in (tmp_path / "room_003_rrara1.jsonl").read_text().splitlines()]
    call = next(g for g in game if g["type"] == "thought")
    assert {"text", "options", "raw", "choice", "previous_check", "measured_s", "declared_s", "charged", "tokens_in", "stop_reason"} <= set(call)
