"""Tests for firelab.planner (chunk 6a): the counted search is the rule check's, and the board built in the simulator is too.

One 6-item room is built (about 6 s); 6 items keep each plan at a few hundredths of a second.
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_planner.py
"""

import math

import pytest

import firelab  # noqa: F401
from firelab import gate
from firelab.agents import GateSkills
from firelab.env import FireLab3DEnv
from firelab.planner import plan
from firelab.skills import Skills

RULES = {"values": (1, 2, 3, 5, 8), "layout": "one_wide", "clearance": True, "reach": 1, "bag": True, "hp_loss": 2.0}
SIX = gate.Task(items=6, **RULES)


@pytest.fixture(scope="module")
def robot():
    env = FireLab3DEnv(level=2, num_items=6, realistic_bg=False)
    yield env, Skills(env)
    env.close()


@pytest.mark.parametrize("room", range(10))
def test_the_counted_search_gives_the_rule_checks_plan(room):
    board = GateSkills(gate.make_world(SIX, room))
    world, state = board.believe()
    value, actions, steps = plan(world, state)
    assert (value, actions) == gate.plan(world, state)
    assert steps > 0 and plan(world, state)[2] == steps  # the count is a property of the search, not of the machine


def test_the_copy_also_holds_at_eight_items():
    world, state = GateSkills(gate.make_world(gate.Task(items=8, **RULES), 0)).believe()
    value, actions, steps = plan(world, state)
    assert (value, actions) == gate.plan(world, state) and steps == 586  # room 0; the same on any machine


@pytest.mark.parametrize("room", range(3))
def test_the_simulators_board_is_the_rule_checks_forecast(robot, room):
    env, skills = robot
    board = GateSkills(gate.make_world(SIX, room))
    env.reset(seed=room)
    skills.start()
    for wait in (0, 60, 60):  # at tick 0, 60 and 120, the robot standing still in both worlds
        skills.wait(wait)
        board.wait(wait)
        (ws, ss), (wb, sb) = skills.believe(), board.believe()
        now = skills.ticks
        assert now == board.ticks == sb.t == ss.t and (ss.pos, ss.hp, ss.taken) == (sb.pos, sb.hp, sb.taken)
        assert ws.open == wb.open and ws.safe == wb.safe and ws.cells == wb.cells and ws.t_end == wb.t_end
        assert {c for c, lit in ws.burn.items() if lit <= now} == {c for c, lit in wb.burn.items() if lit <= now}
        assert {c: lit for c, lit in ws.burn.items() if lit > now} == {c: lit for c, lit in wb.burn.items() if lit > now}
        for t in (now, now + 40, now + 200):  # item values now and later
            assert [gate.value(ws, k, t) for k in range(6)] == pytest.approx([gate.value(wb, k, t) for k in range(6)])
        assert plan(ws, ss) == plan(wb, sb) and not math.isinf(ws.t_end)
