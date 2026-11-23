"""Tests for firelab.agents (chunk 5a): the two agents are true copies, and the two worlds agree when the fire is still.

One room is built (about 6 s), with the fire frozen (p_spread = 0).
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_agents.py
"""

import dataclasses

import pytest

import firelab  # noqa: F401
from firelab import gate
from firelab.agents import GateSkills, random_agent, reflex
from firelab.env import LEVEL2, FireLab3DEnv
from firelab.fire import FireGridConfig
from firelab.skills import Skills

MAIN = gate.Task(items=8, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)
STILL = dataclasses.replace(MAIN, p_spread=0.0)  # the same rooms with a fire that never spreads


@pytest.fixture(scope="module")
def robot():
    env = FireLab3DEnv(firelab_config=dataclasses.replace(LEVEL2, grid=FireGridConfig(p_spread=0.0)), realistic_bg=False)
    yield env, Skills(env)
    env.close()


@pytest.mark.parametrize("room", range(10))
def test_on_the_rule_checks_board_the_copies_score_exactly_as_the_originals(room):
    w = gate.make_world(MAIN, room)
    a, b = GateSkills(w), GateSkills(w)
    reflex(a)
    random_agent(b, 10_000 + room)
    assert a.score() == gate.reflex(w) and b.score() == gate.random_agent(w, 10_000 + room)


@pytest.mark.parametrize("room, score", [(2, 0.875), (5, 0.774)])
def test_with_a_still_fire_the_reflex_does_the_same_in_both_worlds(robot, room, score):
    env, skills = robot
    board = GateSkills(gate.make_world(STILL, room))
    on_board = reflex(board)
    env.reset(seed=room)
    skills.start()
    in_sim = reflex(skills)
    assert len(in_sim) == len(on_board) > 8
    for (name, k, tick, result), (name2, k2, tick2, result2) in zip(in_sim[:-1], on_board[:-1]):
        # the same command on the same item with the same outcome; the simulator is 2 ticks ahead throughout,
        # because its first step from home is half a cell (A74 c)
        assert (name, k, result.done, tick) == (name2, k2, result2.done, tick2 - 2)
    assert skills.score() == pytest.approx(board.score()) == pytest.approx(score, abs=1e-3)
    assert not in_sim[-1][3].done and in_sim[-1][3].reason == "episode over" and skills.ticks == 500


def test_random_runs_to_the_end_and_its_log_accounts_for_every_tick(robot):
    env, skills = robot
    env.reset(seed=2)
    skills.start()
    log = random_agent(skills, 10_002)
    assert skills.over and skills.ticks == 500 and sum(result.ticks for *_, result in log) == 500
    assert 0.0 <= skills.score() <= 1.0 and {name for name, *_ in log} <= {"go", "go home", "pick", "drop", "wait"}
    assert 0.0 < skills.sim_seconds < 30.0  # time inside env.step is counted
