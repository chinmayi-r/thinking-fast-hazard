"""Tests for firelab.gate. Pure Python, no simulator.

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_gate.py
"""

import math

from firelab.gate import (
    FRONT_TICKS, START, State, Task, build, by_value_per_tick, forecast, make_world,
    planner_all, planner_now, random_agent, reflex, route, spend, value,
)


def _corridor(task: Task, burn_at: dict | None = None, items=((2, 0),), values=(5.0,), length=5):
    """A row of cells; (0, 0) is not the safe strip, so tests that need a strip pass their own world."""
    burn = {(i, 0): math.inf for i in range(length)}
    burn.update(burn_at or {})
    return build(task, burn, list(items), list(values), start=(0, 0))


def _with_strip(task: Task, items, values, length=5):
    """A corridor running up the grid's j axis, whose far end (j = 11) is the safe strip; start next to it."""
    burn = {(0, j): math.inf for j in range(11 - length, 12)}
    return build(task, burn, [(0, 11 - d) for d in items], list(values), start=(0, 11))


def test_same_seed_same_world_and_the_fire_ignores_the_item_knobs():
    a, b = make_world(Task(items=6, layout="two_narrow"), 3), make_world(Task(items=6, layout="two_narrow"), 3)
    assert a.burn == b.burn and a.cells == b.cells and a.values == b.values and a.t_end == b.t_end
    c = make_world(Task(items=8, layout="two_narrow", values=(1, 2, 3, 5, 8), bag=True, pick_ticks=100), 3)
    assert c.burn == a.burn and c.t_end == a.t_end


def test_world_layout():
    w = make_world(Task(items=6), 0)
    assert len(w.burn) == 240 - 19 and START in w.burn  # furniture cells are absent
    assert w.lit[0] == 0 and w.lit[1] > 0  # exactly the ignition cell burns at tick 0
    assert all(c[1] < 11 and c not in w.safe and w.burn[c] != 0 for c in w.cells)
    assert len(make_world(Task(layout="two_narrow"), 0).burn) == 240 - 19 - 17
    assert make_world(Task(end_coverage=0.8), 0).t_end >= w.t_end


def test_item_value_drains_per_tick_or_is_lost_at_once():
    slow = _corridor(Task(drain=0.02), {(2, 0): 100})
    assert value(slow, 0, 99) == 5.0 and value(slow, 0, 110) == 5.0 * 0.8 and value(slow, 0, 150) == 0.0
    fast = _corridor(Task(drain=1.0), {(2, 0): 100})
    assert value(fast, 0, 100) == 5.0 and value(fast, 0, 101) == 0.0


def test_spend_charges_health_only_for_ticks_in_fire():
    w, s = _corridor(Task(hp_loss=2.0), {(1, 0): 2}), State((1, 0))
    spend(w, s, 4)  # ticks 0..3; the cell burns from tick 2, so two of them hurt
    assert s.t == 4 and s.hp == 100.0 - 2 * 2.0


def test_route_goes_through_fire_only_when_told_to_and_foresight_sees_arrival_time():
    at_item = lambda c: c == (2, 0)
    w = _corridor(Task(), {(1, 0): 0})
    assert route(w, (0, 0), 0, at_item, cross=False, foresight=False) is None
    assert route(w, (0, 0), 0, at_item, cross=True, foresight=False) == [(1, 0), (2, 0)]
    w = _corridor(Task(), {(1, 0): 3})  # not burning now, burning by the time a 4-tick step ends
    assert route(w, (0, 0), 0, at_item, cross=False, foresight=False) == [(1, 0), (2, 0)]
    assert route(w, (0, 0), 0, at_item, cross=False, foresight=True) is None


def test_no_fire_means_everything_is_saved_with_or_without_a_bag():
    for bag in (False, True):
        w = make_world(Task(items=3, p_spread=0.0, bag=bag), 1)
        assert reflex(w) == 1.0 and planner_all(w) == 1.0 and planner_now(w) == 1.0


def test_a_bag_saves_two_items_where_two_trips_do_not_fit():
    # items 2 and 4 cells from the strip. One tour: 16 + 2 x 10 + 16 + 5 = 57 ticks. Two trips: 31 + 47 = 78.
    for bag, expected in ((True, 1.0), (False, 0.5)):
        w = _with_strip(Task(max_ticks=60, bag=bag), items=(2, 4), values=(1.0, 1.0))
        assert planner_all(w) == expected


def test_a_bag_out_at_the_end_is_lost():
    w = _with_strip(Task(max_ticks=40, bag=True), items=(2, 4), values=(1.0, 5.0))
    assert reflex(w) == 0.0  # it goes for the 5 first and cannot be back by tick 40
    assert planner_all(w) == 1 / 6  # the near item alone fits: 8 + 10 + 8 + 5 = 31


def test_value_per_tick_prefers_the_near_item():
    w = _with_strip(Task(), items=(1, 5), values=(2.0, 5.0), length=6)
    s = State(w.start)
    assert by_value_per_tick(w, s, 0) > by_value_per_tick(w, s, 1)  # 2 / 4 ticks against 5 / 20 ticks


def test_forecast_keeps_the_present_and_advances_the_front_a_cell_at_a_time():
    w = _corridor(Task(), {(0, 0): 5, (1, 0): 400})
    f = forecast(w, 10)
    assert f.burn[(0, 0)] == 5 and f.burn[(1, 0)] == 10 + FRONT_TICKS and f.burn[(3, 0)] == 10 + 3 * FRONT_TICKS


def test_scores_are_fractions_and_random_is_repeatable():
    w = make_world(Task(items=6, layout="two_narrow", hp_loss=2.0, bag=True), 2)
    for score in (reflex(w), reflex(w, by_value_per_tick), planner_all(w), planner_now(w), random_agent(w, 7)):
        assert 0.0 <= score <= 1.0
    assert random_agent(w, 7) == random_agent(w, 7)


def test_layout_cell_counts_and_clearance_changes_movement_only():
    assert len(make_world(Task(layout="two_wide"), 0).burn) == 240 - 19 - 13
    assert len(make_world(Task(layout="one_wide"), 0).burn) == 240 - 19 - 12
    a, b = make_world(Task(layout="two_wide"), 4), make_world(Task(layout="two_wide", clearance=True), 4)
    assert a.burn == b.burn and a.cells == b.cells and a.t_end == b.t_end and b.open < a.open


def test_clearance_keeps_the_robot_a_cell_from_walls_and_closes_narrow_doorways():
    wide = make_world(Task(layout="two_wide", clearance=True), 0)
    # the left doorway is columns 2-5 of row 5; only its two middle cells are a cell away from both wall ends
    assert [(i, 5) in wide.open for i in (2, 3, 4, 5)] == [False, True, True, False]
    assert (3, 4) in wide.open and (7, 3) not in wide.open  # (7, 3) touches the wall at column 8
    back_room = lambda c: c == (3, 2)
    assert route(wide, START, 0, back_room, True, False) is not None
    narrow = make_world(Task(layout="two_narrow", clearance=True), 0)
    assert route(narrow, START, 0, back_room, True, False) is None  # a 0.4 m doorway has no enterable cell


def test_exit_rules_and_the_clock():
    w = _with_strip(Task(max_ticks=60, bag=True), items=(2, 4), values=(1.0, 1.0))
    assert reflex(w) == 0.5 and reflex(w, every=2) == 1.0  # the official reflex banks every pick: one trip fits
    assert planner_now(w) == 1.0 and planner_now(w, charge=20) < 1.0  # thinking 20 ticks per decision costs an item


def test_reach_lets_the_robot_pick_from_a_neighbouring_cell_and_nothing_is_unreachable():
    from firelab.gate import act, unreachable, within_reach
    w = _corridor(Task(reach=1), items=((3, 0),))
    s = State((0, 0))
    assert within_reach(w, 0)((2, 0)) and not within_reach(w, 0)((1, 0))
    assert act(w, s, ("get", 0, False)) and s.pos == (2, 0) and s.carrying == 1  # stopped one cell short
    for seed in range(10):
        kept = dict(items=8, layout="one_wide", clearance=True)
        assert unreachable(make_world(Task(reach=1, **kept), seed)) == 0
    assert sum(unreachable(make_world(Task(reach=0, **kept), seed)) for seed in range(10)) > 0


def test_plan_returns_every_action_in_order():
    from firelab.gate import plan
    w = _with_strip(Task(max_ticks=60, bag=True), items=(2, 4), values=(1.0, 1.0))
    best, actions = plan(w, State(w.start))
    assert best == 2.0 and [a[0] for a in actions] == ["get", "get", "home"]


def test_three_ways_to_think_agree_when_thinking_is_free_and_split_when_it_costs():
    from firelab.gate import thinker
    calm = make_world(Task(items=3, p_spread=0.0, bag=True), 1)
    for kind in ({}, {"commit": True}, {"commit": True, "moving": True}):
        assert thinker(calm, 0, **kind) == 1.0 and thinker(calm, 20, **kind) == 1.0  # no fire, time to spare
    tight = _with_strip(Task(max_ticks=60, bag=True), items=(2, 4), values=(1.0, 1.0))
    always, commit, moving = (thinker(tight, 20, **k) for k in ({}, {"commit": True}, {"commit": True, "moving": True}))
    assert always == 0.0  # 20 ticks before each of three actions: the clock runs out with the bag still out
    assert commit == 0.5  # one thought, then the near item fits
    assert moving >= commit  # the reflex has already fetched the near item when the plan arrives
