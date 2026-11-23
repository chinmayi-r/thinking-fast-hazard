"""Tests for firelab.skills (chunk 4): routes, walking, get, home, and honest failures.

One room is built (about 6 s). Its fire is frozen (p_spread = 0), so every tick count below is exact.
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_skills.py
"""

import dataclasses

import pytest

import firelab  # noqa: F401
from firelab import gate
from firelab.env import LEVEL2, FireLab3DEnv
from firelab.fire import BURNING, NORMAL, FireGridConfig
from firelab.items import BAG, SAVED, ItemRules
from firelab.skills import Result, Skills

ROOM = 2  # with 6 items: fire at (13, 3); items at (13, 10), (4, 9), (3, 3), (13, 4), (2, 1), (4, 7) worth 3, 3, 1, 8, 5, 5


@pytest.fixture(scope="module")
def robot():
    frozen = dataclasses.replace(LEVEL2, items=ItemRules(count=6), grid=FireGridConfig(p_spread=0.0))
    env = FireLab3DEnv(firelab_config=frozen, realistic_bg=False)
    yield env, Skills(env)
    env.close()


def _fresh(env, skills):
    """Reset, then step onto a cell centre: home (0.3, 0.0) is 0.1 m from the centre of (8, 7), so 2 ticks."""
    env.reset(seed=ROOM)
    skills.start()
    assert skills.walk(lambda c: c == (8, 7)) == Result(2, True, "ok")


def test_the_walker_may_use_the_cells_the_rule_check_allowed(robot):
    _, skills = robot
    assert skills.open == gate.make_world(gate.Task(layout="one_wide", clearance=True), 0).open and len(skills.open) == 161


def test_walk_goes_through_the_doorway_at_four_ticks_a_cell(robot):
    env, skills = robot
    _fresh(env, skills)
    path = skills.route(lambda c: c == (7, 2))
    assert len(path) == 6 and {c for c in path if c[1] == 5} <= {(7, 5), (8, 5)}  # 6 cells, the wall row crossed in the doorway
    assert skills.walk(lambda c: c == (7, 2)) == Result(24, True, "ok") and skills.cell() == (7, 2)


def test_fire_in_the_doorway_means_no_route_unless_told_to_cross(robot):
    env, skills = robot
    _fresh(env, skills)
    skills.fire.state[7, 5] = skills.fire.state[8, 5] = BURNING
    try:
        assert skills.route(lambda c: c == (7, 2)) is None
        assert skills.walk(lambda c: c == (7, 2)) == Result(0, False, "no route")
        assert skills.walk(lambda c: c == (7, 2), cross=True) == Result(24, True, "ok")
        assert skills.info["health"] < 100.0  # crossing cost health
    finally:
        skills.fire.state[7, 5] = skills.fire.state[8, 5] = NORMAL


def test_get_then_home_banks_the_item(robot):
    env, skills = robot
    _fresh(env, skills)
    assert skills.get(1) == Result(4 * 4 + 10, True, "ok")  # 4 cells to (5, 8), next to the item at (4, 9); pick 10
    assert skills.cell() == (5, 8) and skills.items.where[1] == BAG and skills.info["score"] == 0.0
    assert skills.home() == Result(3 * 4 + 5, True, "ok")  # 3 cells to row 11, the safe strip; drop 5
    assert skills.items.where[1] == SAVED and skills.info["score"] == pytest.approx(3 / 25)


def test_get_reaches_an_item_behind_the_partition(robot):
    env, skills = robot
    _fresh(env, skills)
    result = skills.get(2)  # the item at (3, 3)
    assert result.done and skills.items.where[2] == BAG and skills.cell()[1] < 5


def test_skills_say_so_when_they_fail(robot):
    env, skills = robot
    _fresh(env, skills)
    assert skills.home() == Result(4 * 4, False, "refused")  # walked to the strip with an empty bag
    assert skills.get(1).done and skills.get(1) == Result(0, False, "refused")  # already picked
    _fresh(env, skills)
    assert skills.walk(lambda c: c == (11, 7)).done
    assert not skills._drive((11, 6)) and skills.cell() == (11, 7)  # pylint: disable=protected-access  # the wall stops it


def test_nothing_runs_after_the_episode_is_over(robot):
    env, skills = robot
    _fresh(env, skills)
    assert skills.wait(600) == Result(498, False, "episode over")  # the cap is 500 ticks; 2 were used
    assert skills.get(1) == Result(0, False, "episode over") and skills.home() == Result(0, False, "episode over")
