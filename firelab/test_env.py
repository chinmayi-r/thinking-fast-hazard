"""Tests for firelab.env (chunk 2). Each env takes about 6 s to build, so three are shared.

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_env.py
"""

import gymnasium
import numpy as np
import pytest

import firelab  # noqa: F401  registers the ids and applies the Windows shim
from firelab.env import FireLab3DEnv, FireLabConfig, FireLabObjectCentricState
from firelab.fire import BLOCKED, BURNING, FireGridConfig

ZERO = np.zeros(11, dtype=np.float32)


def _state(env, vec) -> FireLabObjectCentricState:
    s = env.observation_space.devectorize(vec)
    return FireLabObjectCentricState(s.data, s.type_features)


@pytest.fixture(scope="module")
def env():
    e = gymnasium.make("firelab/FireLab3D-l1-v0", realistic_bg=False)
    yield e
    e.close()


@pytest.fixture(scope="module")
def fast_env():
    """p = 1: the fire fills the room in a few ticks. State access on, for the set_state test."""
    cfg = FireLabConfig(grid=FireGridConfig(p_spread=1.0))
    e = FireLab3DEnv(firelab_config=cfg, realistic_bg=False, allow_state_access=True)
    yield e
    e.close()


def test_reset_gives_one_burning_cell_full_health_and_the_expected_vector(env):
    vec, _ = env.reset(seed=0)
    assert vec.shape == (19 + 7 + 240 * 3 + 7,)  # robot, lab, 240 cells, status; no cubes or items at level 1
    s = _state(env, vec)
    assert len(s.burning_cells()) == 1 and s.health == 100.0 and s.tick == 0
    assert s.coverage == pytest.approx(1 / (240 - len(_blocked(env))))


def _blocked(env):
    return [ij for ij in zip(*np.where(env.unwrapped._object_centric_env.fire.state == BLOCKED))]  # pylint: disable=protected-access


def test_furniture_blocks_cells_along_the_far_wall_only(env):
    inner = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    blocked = _blocked(env)
    assert 0 < len(blocked) < 60 and inner.blocked_links  # cabinets, sink, doors
    ys = [inner.fire.center_of(i, j)[1] for i, j in blocked]
    assert min(ys) > 1.0  # the cabinet run is at y 1.2 to 2.0 (kinder_probe_lab_links.log)
    home = inner.config.robot_base_home_pose
    assert inner.fire.state[inner.fire.cell_of(home.x, home.y)] != BLOCKED


def test_same_seed_same_ignition(env):
    a = _state(env, env.reset(seed=3)[0]).burning_cells()
    b = _state(env, env.reset(seed=3)[0]).burning_cells()
    assert a == b


def test_zero_action_advances_fire_and_ticks(fast_env):
    fast_env.reset(seed=1)
    for t in range(3):
        vec, reward, terminated, truncated, info = fast_env.step(ZERO)
        assert info["tick"] == t + 1 and not terminated and not truncated
    s = _state(fast_env, vec)
    assert s.tick == 3 and len(s.burning_cells()) >= 4 and info["coverage"] > 1 / 240
    assert reward == 0.0 or info["in_fire"]


def test_fire_reaching_the_robot_drains_health(fast_env):
    fast_env.reset(seed=1)
    hit = False
    for _ in range(80):
        vec, reward, terminated, truncated, info = fast_env.step(ZERO)
        if info["in_fire"]:
            hit = True
            assert reward == pytest.approx(-0.15)
    assert hit and _state(fast_env, vec).health < 100.0 and info["score"] < 1.0


def test_set_state_restores_the_fire_and_the_counters(fast_env):
    fast_env.reset(seed=2)
    for _ in range(10):
        vec10, *_ = fast_env.step(ZERO)
    for _ in range(20):
        fast_env.step(ZERO)
    fast_env.set_state(vec10)
    assert np.allclose(fast_env.get_state(), vec10)


def test_health_zero_terminates_and_the_cap_truncates():
    cfg = FireLabConfig(health=0.3, max_ticks=6, grid=FireGridConfig(p_spread=1.0))
    e = FireLab3DEnv(firelab_config=cfg, realistic_bg=False)
    try:
        e.reset(seed=0)
        outcome = None
        for _ in range(6):
            _, _, terminated, truncated, info = e.step(ZERO)
            if terminated or truncated:
                outcome = (terminated, truncated, info)
                break
        assert outcome is not None
        terminated, truncated, info = outcome
        if terminated:
            assert info["health"] <= 0.0  # two ticks in fire at 0.15 per tick drain 0.3
        else:
            assert truncated and info["tick"] == 6
    finally:
        e.close()
