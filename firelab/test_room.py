"""Tests for the level-2 room (chunk 3e): partition, base collisions, the fence, the safe strip, ignition.

One room is built (about 6 s) and shared by every test.
Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_room.py
"""

import numpy as np
import pytest

import firelab  # noqa: F401  registers the ids and applies the Windows shim
from firelab import gate
from firelab.env import LEVEL2, ONE_DOORWAY, FireLab3DEnv
from firelab.fire import BLOCKED, BURNING

STEP = 0.05  # metres per tick, the base's top speed


@pytest.fixture(scope="module")
def room():
    env = FireLab3DEnv(level=2, realistic_bg=False)
    yield env, env.unwrapped._object_centric_env  # pylint: disable=protected-access
    env.close()


def _drive(env, inner, dx, dy, ticks):
    """Push the base (dx, dy) metres per tick for some ticks; return where it ends up and the last info."""
    action = np.zeros(11, dtype=np.float32)
    action[0], action[1] = dx, dy
    for _ in range(ticks):
        *_, info = env.step(action)
    pose = inner.robot.base.get_pose()
    return float(pose.x), float(pose.y), info


def _cells(inner, state):
    return {(int(i), int(j)) for i, j in np.argwhere(inner.fire.state == state)}


def test_floor_plan_is_the_rule_checks_one_doorway_room(room):
    _, inner = room
    assert ONE_DOORWAY == gate.LAYOUTS["one_wide"] and LEVEL2.health_loss_per_tick == 2.0
    blocked = _cells(inner, BLOCKED)
    assert {i for i, j in blocked if j == 5} == set(range(16)) - {6, 7, 8, 9}  # row 5 is the wall, columns 6-9 the doorway
    free = {(i, j) for i in range(16) for j in range(15)} - blocked
    assert free == set(gate.fire_history(0, "one_wide", 0.015, 500))  # the same free cells as the rule check
    assert inner.config.check_base_collisions and len(inner._partition_ids) == 2  # pylint: disable=protected-access


def test_the_partition_stops_the_base_and_the_doorway_lets_it_through(room):
    env, inner = room
    env.reset(seed=0)
    x, y, _ = _drive(env, inner, 0.0, -STEP, 30)  # straight back from home: x = 0.3 puts the base's edge behind the wall
    assert x == pytest.approx(0.3) and y > -0.1
    env.reset(seed=0)
    _drive(env, inner, -STEP, 0.0, 4)  # 0.2 m to the left: x = 0.1, the middle of the doorway
    x, y, _ = _drive(env, inner, 0.0, -STEP, 24)
    assert x == pytest.approx(0.1) and y == pytest.approx(-1.2)  # through, 0.8 m behind the wall


def test_the_fence_and_the_cabinets_stop_the_base(room):
    env, inner = room
    env.reset(seed=0)
    x, _, _ = _drive(env, inner, -STEP, 0.0, 80)  # 4 m of pushing; the floor ends at x = -1.5
    assert -1.5 <= x < -1.4
    env.reset(seed=0)
    _, y, _ = _drive(env, inner, 0.0, STEP, 80)
    assert 0.9 < y < 1.1  # stopped by the cabinets (the probe measured 1.05), not by the fence at 1.5


def test_the_safe_strip_starts_at_y_0_7(room):
    env, inner = room
    env.reset(seed=0)
    _, y, info = _drive(env, inner, 0.0, STEP, 13)
    assert y == pytest.approx(0.65) and not info["in_safe"]
    _, y, info = _drive(env, inner, 0.0, STEP, 2)
    assert y == pytest.approx(0.75) and info["in_safe"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_room_s_burns_as_room_s_of_the_rule_check(room, seed):
    env, inner = room
    burn = gate.fire_history(seed, "one_wide", 0.015, 500)  # cell -> tick it catches fire
    env.reset(seed=seed)
    assert inner.room == seed and _cells(inner, BURNING) == {c for c, t in burn.items() if t == 0}
    _drive(env, inner, 0.0, 0.0, 60)
    assert _cells(inner, BURNING) == {c for c, t in burn.items() if t <= 60}


def test_without_a_seed_the_env_draws_a_room(room):
    env, inner = room
    env.reset(seed=7)
    env.reset()
    assert inner.room != 7 and _cells(inner, BURNING) <= set(inner.ignitable())
