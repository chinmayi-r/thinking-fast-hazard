"""The planner for the simulator: the rule check's exhaustive search, counted, on a board built from what the robot sees.

Two pieces (PLAN.md section 6, chunk 6a):
  plan(world, state)   a copy of gate.plan that also counts its search steps (one step = one state expanded).
                       gate.py itself is not touched (CR, Nov 24, decision 1); test_planner.py holds the copy
                       to the original.
  board_from(skills)   the rule check's board (gate.World, gate.State) built from the simulator's present state:
                       which cells burn now, each item's value now and at reset, the robot's cell, health, bag
                       and banked value. The future fire is the rule check's forecast (gate.forecast: the front
                       advances one cell per 30 ticks at p = 0.015).
"""

from __future__ import annotations

import copy
import math

from firelab import gate
from firelab.fire import BLOCKED, BURNING
from firelab.items import BAG, FLOOR, SAVED


def plan(w: gate.World, start: gate.State, foresight: bool = True, cross_fire: bool = True) -> tuple:
    """Exhaustive search from `start` in world w. Returns (most value banked, the actions in order, search steps).

    cross_fire=False: no action of the plan may walk through fire, which is what the reflex is held to (A94).
    """
    best, first, fronts, steps = start.saved, (), {}, 0

    def search(s: gate.State, done: tuple) -> None:
        nonlocal best, first, steps
        steps += 1  # the one line gate.plan does not have
        if s.saved > best:
            best, first = s.saved, done
        todo = gate.live(w, s)
        if s.saved + s.held + sum(gate.value(w, k, s.t) for k in todo) <= best:
            return  # even banking everything left cannot beat the best found so far
        labels = fronts.setdefault((s.taken, s.pos, s.carrying), [])
        if any(t <= s.t and hp >= s.hp and held >= s.held and saved >= s.saved for t, hp, held, saved in labels):
            return  # same items taken, same place, and an earlier visit was at least as good in every way
        labels.append((s.t, s.hp, s.held, s.saved))
        options = [("get", k) for k in sorted(todo, key=lambda k: -gate.value(w, k, s.t))] if w.task.bag or not s.carrying else []
        for kind, k in options + ([("home", None)] if s.carrying else []):
            seen = set()
            for cross in ((False, True) if cross_fire else (False,)):
                s2 = copy.copy(s)
                if gate.act(w, s2, (kind, k, cross), foresight) and (s2.t, s2.hp) not in seen:
                    seen.add((s2.t, s2.hp))
                    search(s2, done + ((kind, k, cross),))

    search(copy.copy(start), ())
    return best, first, steps


def board_from(skills) -> tuple:
    """(gate.World, gate.State): the forecast board as seen from the simulator right now."""
    fire, items, fl = skills.fire, skills.items, skills.inner.fl
    rules, now = items.rules, skills.ticks
    task = gate.Task(
        items=rules.count, layout="one_wide", clearance=True, hp_loss=fl.health_loss_per_tick, values=rules.values,
        pick_ticks=rules.pick_ticks, drain=rules.drain, end_coverage=fl.end_coverage, bag=True, reach=rules.reach,
        p_spread=fire.config.p_spread, max_ticks=fl.max_ticks,
    )
    nx, ny = fire.config.shape
    burn = {(i, j): (now if fire.state[i, j] == BURNING else math.inf)
            for i in range(nx) for j in range(ny) if fire.state[i, j] != BLOCKED}
    for k, cell in enumerate(items.cells):  # how long an item has burned is read off the value it has lost
        if items.where[k] == FLOOR and burn[cell] <= now:
            burn[cell] = now - round((1.0 - items.value[k] / items.start[k]) / rules.drain)
    seen = gate.build(task, burn, list(items.cells), list(items.start), skills.cell())
    state = gate.State(
        pos=skills.cell(), t=now, hp=skills.info.get("health", fl.health), held=items.total(BAG),
        carrying=items.where.count(BAG), saved=items.total(SAVED),
        taken=frozenset(k for k, place in enumerate(items.where) if place != FLOOR),
    )
    return gate.forecast(seen, now), state
