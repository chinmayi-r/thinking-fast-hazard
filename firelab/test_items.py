"""Tests for firelab.items (chunk 3f-1): every rule is compared with the rule check, firelab/gate.py.

No simulator. Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_items.py
"""

import math

import pytest

from firelab import gate
from firelab.items import BAG, FLOOR, LOST, SAVED, ItemRules, Items

MAIN = gate.Task(items=8, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)


def _placed(room: int):
    """Room `room` of the rule check, and our items placed on the same floor."""
    w = gate.make_world(MAIN, room)
    floor = [c for c in w.burn if c[1] < gate.SAFE_ROW and c != gate.START and w.burn[c] != 0]
    items = Items()
    items.place(floor, room)
    return w, items


def _by_hand(cells, values) -> Items:
    items = Items(ItemRules(count=len(cells)))
    items.cells, items.start, items.value, items.where = list(cells), list(values), list(values), [FLOOR] * len(cells)
    return items


def _run(items: Items, w, t: int, ticks: int, running: bool = True) -> int:
    """Advance `ticks` ticks from tick t: the fire burns the items, then the job counts down."""
    for now in range(t, t + ticks):
        items.burn({c for c, lit in w.burn.items() if lit <= now})  # cells on fire when this tick began
        items.tick(running)
    return t + ticks


@pytest.mark.parametrize("room", range(5))
def test_place_gives_the_rule_checks_items(room):
    w, items = _placed(room)
    assert items.cells == w.cells and items.start == w.values and items.where == [FLOOR] * 8


@pytest.mark.parametrize("room", range(5))
def test_burning_drains_value_as_in_the_rule_check(room):
    w, items = _placed(room)
    t = 0
    for until in (50, 150, 300):
        t = _run(items, w, t, until - t)
        assert items.value == pytest.approx([gate.value(w, k, t) for k in range(8)])


@pytest.mark.parametrize("room", range(5))
def test_pick_then_drop_scores_as_the_rule_check(room):
    for k in range(8):
        w, items = _placed(room)
        lit = w.burn[w.cells[k]]
        t0 = 0 if math.isinf(lit) else int(lit) + 5  # start the pick while the item is already burning
        s = gate.State(w.cells[k], t=t0)
        assert gate.pick(w, s, k) and s.t == t0 + 10
        t = _run(items, w, 0, t0)
        assert items.start_pick(w.cells[k], k)
        t = _run(items, w, t, 10)
        assert items.where[k] == BAG and items.total(BAG) == pytest.approx(s.held)
        s.pos, s.hp = next(iter(w.safe)), 100.0  # carried to the safe strip; health is the env's business
        w.t_end = 10**6  # this test is about value, not about the episode's end
        assert gate.drop(w, s) and items.start_drop(in_safe=True)
        _run(items, w, t, 5)
        assert items.where[k] == SAVED and items.score() == pytest.approx(gate.score(w, s))


def test_reach_is_the_cell_and_its_eight_neighbours():
    items = _by_hand([(5, 5)], [3.0])
    assert all(items.in_reach((5 + a, 5 + b)) == [0] for a in (-1, 0, 1) for b in (-1, 0, 1))
    assert items.in_reach((7, 5)) == [] and not items.start_pick((7, 5))


def test_pick_takes_the_most_valuable_in_reach_unless_one_is_named():
    items = _by_hand([(5, 5), (6, 5), (9, 9)], [2.0, 8.0, 5.0])
    assert items.start_pick((5, 5)) and items.job == ("pick", 1)
    items = _by_hand([(5, 5), (6, 5), (9, 9)], [2.0, 8.0, 5.0])
    assert not items.start_pick((5, 5), 2) and items.start_pick((5, 5), 0) and items.job == ("pick", 0)


def test_one_job_at_a_time_and_a_picked_item_stops_burning():
    items = _by_hand([(5, 5), (5, 6)], [2.0, 8.0])
    assert items.start_pick((5, 5), 0) and not items.start_pick((5, 5), 1) and not items.start_drop(True)
    for _ in range(10):
        assert items.where[0] == FLOOR
        items.tick(True)
    assert items.where[0] == BAG and items.job is None
    items.burn({(5, 5), (5, 6)})
    assert items.value == pytest.approx([2.0, 8.0 * 0.98])


def test_drop_needs_the_safe_strip_and_a_bag_and_banks_only_in_time():
    items = _by_hand([(5, 5), (5, 6)], [2.0, 8.0])
    assert not items.start_drop(True)  # nothing in the bag
    items.where[1] = BAG
    assert not items.start_drop(False) and items.start_drop(True)
    for _ in range(5):
        items.tick(running=False)  # the episode ended before the drop did
    assert items.where == [FLOOR, LOST] and items.score() == 0.0
