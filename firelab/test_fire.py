"""Tests for firelab.fire (chunk 1). Run: venv_kinder\\Scripts\\python.exe -m pytest firelab -q"""

import numpy as np
import pytest

from firelab.fire import BLOCKED, BURNING, NORMAL, FireGrid, FireGridConfig


def _grid(p: float, seed: int = 0) -> FireGrid:
    return FireGrid(FireGridConfig(p_spread=p), rng=np.random.default_rng(seed))


def test_default_shape_is_the_240_cell_room():
    cfg = FireGridConfig()
    assert cfg.shape == (16, 15)
    assert FireGrid(cfg).features().shape == (240, 3)


def test_cell_of_and_center_of_round_trip_and_off_grid():
    g = _grid(0.0)
    for (i, j) in [(0, 0), (15, 14), (7, 3)]:
        x, y = g.center_of(i, j)
        assert g.cell_of(x, y) == (i, j)
    assert g.cell_of(-1.5, -1.5) == (0, 0)  # lower corner is inside
    assert g.cell_of(1.7, 1.5) is None  # upper corner is the next cell, off grid
    assert g.cell_of(5.0, 0.0) is None
    assert g.is_burning(5.0, 0.0) is False


def test_reset_ignites_exactly_one_burnable_cell():
    g = _grid(0.015)
    ij = g.reset()
    assert g.state[ij] == BURNING
    assert (g.state == BURNING).sum() == 1
    assert g.tick == 0 and g.ignition == ij
    assert g.reset(ignition=(3, 4)) == (3, 4) and g.state[3, 4] == BURNING
    assert (g.state == BURNING).sum() == 1  # the earlier fire was cleared


def test_reset_rejects_blocked_or_off_grid_ignition():
    g = _grid(0.015)
    g.block_box(-1.5, -1.0, -1.5, -1.0)
    with pytest.raises(ValueError):
        g.reset(ignition=(0, 0))
    with pytest.raises(ValueError):
        g.reset(ignition=(99, 0))


def test_same_seed_same_fire():
    a, b = _grid(0.05, seed=7), _grid(0.05, seed=7)
    assert a.reset() == b.reset()
    for _ in range(50):
        a.step(), b.step()
    assert np.array_equal(a.state, b.state) and a.tick == b.tick == 50


def test_p_zero_never_spreads():
    g = _grid(0.0)
    g.reset(ignition=(8, 7))
    assert sum(g.step() for _ in range(100)) == 0
    assert (g.state == BURNING).sum() == 1 and g.tick == 100


def test_p_one_fills_the_four_neighbourhood_in_one_tick():
    g = _grid(1.0)
    g.reset(ignition=(8, 7))
    assert g.step() == 4
    burning = set(g.burning_cells())
    assert burning == {(8, 7), (7, 7), (9, 7), (8, 6), (8, 8)}  # no diagonals
    assert g.step() == 8  # the ring grows by Manhattan distance


def test_new_fire_spreads_only_from_the_next_tick():
    g = _grid(1.0)
    g.reset(ignition=(0, 0))  # corner: 2 neighbours
    assert g.step() == 2 and g.step() == 3  # (1,1), (2,0), (0,2); not more


def test_blocked_cells_never_burn_and_are_not_burnable():
    g = _grid(1.0)
    n_blocked = g.block_box(-1.5, 0.05, -1.5, 1.5)  # the left half: centres -1.4 ... 0.0 (0.05 absorbs float error)
    assert n_blocked == 8 * 15 and g.burnable() == 240 - n_blocked
    g.reset(ignition=(12, 7))
    for _ in range(60):
        g.step()
    assert (g.state == BLOCKED).sum() == n_blocked
    assert g.coverage() == pytest.approx(1.0)  # every burnable cell burning
    assert not g.is_burning(*g.center_of(0, 0))


def test_coverage_and_features_track_state():
    g = _grid(1.0)
    g.reset(ignition=(8, 7))
    assert g.coverage() == pytest.approx(1 / 240)
    g.step()
    assert g.coverage() == pytest.approx(5 / 240)
    f = g.features()
    k = 8 * 15 + 7  # row-major index of (8, 7)
    assert tuple(f[k, :2]) == pytest.approx(g.center_of(8, 7)) and f[k, 2] == BURNING
    assert (f[:, 2] == NORMAL).sum() == 235
