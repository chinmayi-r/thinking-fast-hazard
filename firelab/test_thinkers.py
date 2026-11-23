"""Tests for firelab.clock and firelab.thinkers (chunk 6b).

On the rule check's board the three agents must score exactly what gate.thinker scores at the same charge.
In the simulator each plays one 6-item room and the saved game is checked. One room is built (about 6 s).
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_thinkers.py
"""

import json

import pytest

import firelab  # noqa: F401
from firelab import gate
from firelab.agents import GateSkills
from firelab.clock import Clock, ticks
from firelab.env import FireLab3DEnv
from firelab.skills import Skills
from firelab.thinkers import AGENTS, thinker

SIX = gate.Task(items=6, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)


def test_the_clock_gives_both_costs_and_charges_by_its_mode():
    assert (ticks(0.0), ticks(0.05), ticks(0.1), ticks(0.11), ticks(1.8)) == (0, 1, 1, 2, 18)
    cost = Clock("declared", seconds_per_step=0.001).cost(measured_s=0.42, steps=1500)
    assert (cost["ticks_measured"], cost["declared_s"], cost["ticks_declared"], cost["charged"]) == (5, 1.5, 15, 15)
    assert Clock("measured").cost(0.42, 1500)["charged"] == 5 and Clock("off").cost(0.42, 1500)["charged"] == 0
    assert Clock("fixed", 20).cost(0.42, 1500)["charged"] == 20


def test_the_estimate_is_exact_when_known_and_otherwise_the_last_cost():
    assert Clock("off").estimate(8) == 0 and Clock("fixed", 20).estimate(8) == 20
    clock = Clock("declared", seconds_per_step=0.001)
    assert clock.estimate(8) == 10 and clock.estimate(6) == 1  # first thought: the timing table, else 1 tick
    clock.cost(0.0, 700)
    assert clock.estimate(8) == 7  # afterwards: what the last thought was charged
    clock.start()
    assert clock.estimate(8) == 10  # a new episode forgets it


@pytest.mark.parametrize("name", list(AGENTS))
@pytest.mark.parametrize("charge", [0, 20])
def test_on_the_board_each_agent_scores_as_the_rule_checks_thinker(name, charge):
    for room in range(4):
        w = gate.make_world(SIX, room)
        board = GateSkills(w)
        thinker(board, Clock("fixed", charge), **AGENTS[name])
        assert board.score() == gate.thinker(w, charge, **AGENTS[name]), (name, charge, room)


@pytest.fixture(scope="module")
def robot():
    env = FireLab3DEnv(level=2, num_items=6, realistic_bg=False)
    yield env, Skills(env)
    env.close()


@pytest.mark.parametrize("name", list(AGENTS))
def test_in_the_simulator_each_agent_plays_a_room_and_leaves_a_saved_game(robot, name):
    env, skills = robot
    env.reset(seed=3)
    skills.start()
    thoughts = []
    log = thinker(skills, Clock("fixed", 20), thoughts=thoughts, **AGENTS[name])
    assert 0.0 <= skills.score() <= 1.0 and thoughts and log
    first = thoughts[0]
    assert first["estimate"] == 20 and first["charged"] == 20 and first["steps"] > 0 and first["measured_s"] > 0
    assert first["seen"]["tick"] == 0 and len(first["seen"]["items"]) == 6 and len(first["seen"]["burning"]) == 1
    assert first["plan"] and first["plan"][0][0] in ("get", "home")
    json.dumps({"thoughts": thoughts, "log": [[n, k, t, list(r)] for n, k, t, r in log]})  # a saved game is plain JSON
    if name == "moving":  # the reflex acted during the first thought: its command ends by the deadline or just after
        assert log[0][2] <= 20 + 15
    else:  # the robot stood for 20 ticks, then acted
        assert log[0][2] > 20
    if name == "always":
        assert len(thoughts) >= len(log)  # one thought per action
