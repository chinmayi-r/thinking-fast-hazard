"""Tests for level 2 in the simulator (chunk 3f-2): items in the state, pick and drop, the end, the score.

Two rooms are built (about 6 s each). Run:
venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_level2.py
"""

import dataclasses

import numpy as np
import pytest

import firelab
from firelab import gate
from firelab.env import LEVEL2, FireLab3DEnv, level_config
from firelab.items import BAG, FLOOR, SAVED

MAIN = gate.Task(items=8, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)
STAY, NORTH, CLOSE, OPEN = (np.zeros(11, dtype=np.float32) for _ in range(4))
NORTH[1], CLOSE[-1], OPEN[-1] = 0.05, -1.0, 1.0


@pytest.fixture(scope="module")
def room():
    env = FireLab3DEnv(level=2, realistic_bg=False, allow_state_access=True)
    yield env, env.unwrapped._object_centric_env  # pylint: disable=protected-access
    env.close()


def _steps(env, action, n):
    for _ in range(n):
        out = env.step(action)
    return out


def _item_next_to_home(env, inner):
    """Room 0 with item 0 moved to the cell in front of the robot, so a test needs no route."""
    env.reset(seed=0)
    inner.items.cells[0] = (9, 8)
    inner._show_items()  # pylint: disable=protected-access


def test_the_id_is_registered_and_the_settings_can_be_changed():
    assert "firelab/FireLab3D-l2-v0" in firelab.ENV_IDS
    assert level_config(2).items.count == 8 and level_config(2).items.values == (1, 2, 3, 5, 8)
    other = level_config(2, num_items=6, values=(5, 1))
    assert (other.items.count, other.items.values, other.partitions) == (6, (5, 1), LEVEL2.partitions)
    assert level_config(1).items is None and not LEVEL2.items_block_base


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_room_s_has_the_rule_checks_items_and_they_are_in_the_vector(room, seed):
    env, inner = room
    vec, _ = env.reset(seed=seed)
    w = gate.make_world(MAIN, seed)
    assert inner.items.cells == w.cells and inner.items.start == w.values
    assert vec.shape == (19 + 7 + 240 * 3 + 8 * 5 + 7,)  # robot, lab, cells, 8 items, status
    state = env.observation_space.devectorize(vec)
    first = state.get_object_from_name("item_00")
    assert (state.get(first, "value"), state.get(first, "where")) == (w.values[0], FLOOR)


def test_pick_takes_ten_ticks_standing_still_and_a_drop_in_the_safe_strip_banks(room):
    env, inner = room
    _item_next_to_home(env, inner)
    _steps(env, NORTH, 2)  # the item's box overlaps the base here: moving at all shows the base passes over it
    assert inner.robot.base.get_pose().y == pytest.approx(0.1)
    *_, info = env.step(CLOSE)
    assert info["job"] == ("pick", 0) and info["job_left"] == 9
    *_, info = _steps(env, NORTH, 8)  # pushing during a pick does nothing
    assert inner.robot.base.get_pose().y == pytest.approx(0.1) and inner.items.where[0] == FLOOR
    env.step(NORTH)
    assert inner.items.where[0] == BAG and inner.items.job is None
    *_, info = env.step(OPEN)  # not in the safe strip: refused
    assert info["job"] is None and info["score"] == 0.0
    _steps(env, NORTH, 13)
    assert inner.robot.base.get_pose().y >= 0.7
    rewards = [env.step(OPEN)[1]] + [env.step(STAY)[1] for _ in range(4)]
    worth = inner.items.value[0]
    assert inner.items.where[0] == SAVED and rewards == [0, 0, 0, 0, pytest.approx(worth)]
    *_, info = env.step(STAY)
    assert info["saved"] == pytest.approx(worth) and info["score"] == pytest.approx(worth / sum(inner.items.start))


def test_set_state_restores_the_items_and_the_job(room):
    env, inner = room
    _item_next_to_home(env, inner)
    env.step(CLOSE)
    vec, *_ = _steps(env, STAY, 3)
    _steps(env, STAY, 20)
    assert inner.items.where[0] == BAG
    env.set_state(vec)
    assert np.allclose(env.get_state(), vec)
    assert inner.items.where[0] == FLOOR and inner.items.job == ("pick", 0) and inner.items.left == 6


@pytest.mark.parametrize("seed", [0, 1])
def test_the_episode_ends_at_the_rule_checks_tick(seed):
    """With a robot that cannot die, standing still: the end comes from the fire or the cap alone."""
    env = FireLab3DEnv(firelab_config=dataclasses.replace(LEVEL2, health=1e9), realistic_bg=False)
    try:
        w = gate.make_world(MAIN, seed)
        env.reset(seed=seed)
        for _ in range(500):
            _, _, terminated, truncated, info = env.step(STAY)
            if terminated or truncated:
                break
        assert info["tick"] == w.t_end and terminated == (info["coverage"] >= 0.5) and info["score"] == 0.0
    finally:
        env.close()
