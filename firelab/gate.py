"""Gate check: on a given set of task rules, do random, the reflex and an unlimited-time planner separate?

CR's three conditions (PLAN.md 0a, A65), clock off: random does badly, the reflex does clearly worse
than an agent with unlimited time, the reflex is not near the ceiling. This file models the task on the
bare fire grid, with no simulator and no LLM, so a set of rules can be rejected before anything is
built in kinder. The agents: random; the reflex (most valuable item, bank after every pick) and three
other exit rules; a diagnostic reflex that picks by value per tick of travel; a planner that knows the
future of the fire (an upper bound); and a planner that sees only the fire as it is now, forecasts the
rest, and can be charged ticks for every decision (the clock).

Runners: firelab/gate_grid.py (rule-set grid), firelab/gate_clock.py (layouts, 100 seeds, the clock).
Simplifications against the kinder env: A66, A68, A69.
"""

from __future__ import annotations

import bisect
import copy
import functools
import math
from dataclasses import dataclass

import numpy as np

from firelab.fire import BLOCKED, BURNING, FireGrid, FireGridConfig

NX, NY = FireGridConfig().shape
STEP_TICKS = 4  # one 0.2 m cell at 0.05 m per tick (A57)
DROP_TICKS = 5  # A60
SAFE_ROW = 11  # cells with j >= 11 have centre y >= 0.8: the safe strip y > 0.6 (D6)
START = (9, 7)  # the cell under the robot home pose (0.3, 0.0)
FURNITURE = ((-1.1, 1.7, 1.3, 1.5), (-1.1, -0.1, 1.1, 1.3))  # the 19 cells chunk 2 measured (A58 addendum)
# Partition layouts, as boxes (x0, x1, y0, y1) for FireGrid.block_box. Every layout has a wall across
# the room at y = -0.4 (grid row 5); the "two" layouts add a wall splitting the back half in two.
BACK_SPLIT = (0.1, 0.3, -1.5, -0.5)
LAYOUTS = {
    "open": (),
    # two doorways 0.4 m wide: narrower than the 0.51 m x 0.55 m base, so NOT buildable; kept for comparison
    "two_narrow": ((-1.5, -0.9, -0.5, -0.3), (-0.5, 0.7, -0.5, -0.3), (1.1, 1.7, -0.5, -0.3), BACK_SPLIT),
    # two doorways 0.8 m wide (columns 2-5 and 10-13)
    "two_wide": ((-1.5, -1.1, -0.5, -0.3), (-0.3, 0.5, -0.5, -0.3), (1.3, 1.7, -0.5, -0.3), BACK_SPLIT),
    # one partition, one doorway 0.8 m wide in the middle (columns 6-9)
    "one_wide": ((-1.5, -0.3, -0.5, -0.3), (0.5, 1.7, -0.5, -0.3)),
}
LEAVE_MARGIN = 0.10  # the "coverage" exit rule heads home once coverage is this close to the end (A68)
FRONT_TICKS = 30  # forecast: ticks for the fire front to advance one cell at p = 0.015 (measured, A68)
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Task:
    """The knobs under test. The defaults are PLAN.md D6 as first written."""

    items: int = 3
    layout: str = "open"  # a key of LAYOUTS
    clearance: bool = False  # True: the robot cannot centre on a cell that touches a wall or furniture
    hp_loss: float = 0.15  # health lost per tick in a burning cell
    values: tuple = (5, 1)  # an item is worth one of these, drawn uniformly
    pick_ticks: int = 10
    drain: float = 0.02  # share of its value an item on a burning cell loses per tick (1.0 = lost at once)
    end_coverage: float = 0.5
    bag: bool = False  # off: one item per trip. on: any number, banked when the robot reaches the safe strip
    reach: int = 0  # pick an item from this many cells away: 0 = its own cell, 1 = any neighbouring cell (A70)
    p_spread: float = 0.015
    max_ticks: int = 500


@dataclass
class World:
    """One episode, fixed facts. The fire never depends on the robot, so its whole history is computed once."""

    task: Task
    burn: dict  # cell -> tick at which it catches fire (inf = never); walls and furniture are absent
    open: set  # cells the robot may stand in: all of burn, or fewer with clearance
    safe: set  # cells of the safe strip
    cells: list  # item k lies in cells[k]
    values: list  # item k is worth values[k] at the start
    lit: list  # the finite ignition ticks, sorted, so coverage at tick t is a bisect
    t_end: int  # tick at which the episode ends (coverage reached or the cap)
    start: tuple = START


@dataclass
class State:
    """Everything that changes as an agent acts."""

    pos: tuple
    t: int = 0
    hp: float = 100.0
    held: float = 0.0  # worth of what is in hand or in the bag
    carrying: int = 0  # number of items in hand or in the bag
    saved: float = 0.0
    taken: frozenset = frozenset()


@functools.lru_cache(maxsize=None)
def fire_history(seed: int, layout: str, p_spread: float, max_ticks: int) -> dict:
    """Run the fire once to the cap; return {free cell: tick it ignites, inf if never}."""
    rng = np.random.default_rng(seed)
    grid = FireGrid(FireGridConfig(p_spread=p_spread), rng)
    for box in FURNITURE + LAYOUTS[layout]:
        grid.block_box(*box)
    floor = [(int(i), int(j)) for i, j in np.argwhere(grid.state != BLOCKED) if j < SAFE_ROW and (i, j) != START]
    grid.reset(floor[int(rng.integers(len(floor)))])
    when = np.full(grid.state.shape, math.inf)
    for t in range(max_ticks + 1):
        when[(grid.state == BURNING) & np.isinf(when)] = t
        grid.step()
    return {(int(i), int(j)): float(when[i, j]) for i, j in np.argwhere(grid.state != BLOCKED)}


def build(task: Task, burn: dict, cells: list, values: list, start=START) -> World:
    lit = sorted(b for b in burn.values() if b < math.inf)
    need = math.ceil(task.end_coverage * len(burn))  # burning cells that end the episode
    t_end = int(min(task.max_ticks, lit[need - 1])) if need <= len(lit) else task.max_ticks
    enterable = set(burn)
    if task.clearance:  # the base is about half a metre wide: keep its centre one cell away from obstacles
        solid = {(i, j) for i in range(NX) for j in range(NY)} - enterable
        enterable = {(i, j) for i, j in burn if not any((i + a, j + b) in solid for a in (-1, 0, 1) for b in (-1, 0, 1))}
    return World(task, burn, enterable, {c for c in burn if c[1] >= SAFE_ROW}, cells, values, lit, t_end, start)


def make_world(task: Task, seed: int) -> World:
    burn = fire_history(seed, task.layout, task.p_spread, task.max_ticks)
    rng = np.random.default_rng([seed, 1])  # items have their own stream: every rule set sees the same fire
    floor = [c for c in burn if c[1] < SAFE_ROW and c != START and burn[c] != 0]
    cells = [floor[q] for q in rng.choice(len(floor), size=task.items, replace=False)]
    values = [float(v) for v in rng.choice(task.values, size=task.items)]
    return build(task, burn, cells, values)


def coverage(w: World, t: int) -> float:
    return bisect.bisect_right(w.lit, t) / len(w.burn)


def value(w: World, k: int, t: int) -> float:
    """What item k is worth at tick t if it is still on the floor."""
    return w.values[k] * max(0.0, 1.0 - w.task.drain * max(0, t - w.burn[w.cells[k]]))


def live(w: World, s: State) -> list:
    return [k for k in range(len(w.cells)) if k not in s.taken and value(w, k, s.t) > 0]


def over(w: World, s: State) -> bool:
    return s.hp <= 0 or s.t >= w.t_end


def score(w: World, s: State) -> float:
    return s.saved / sum(w.values)


def spend(w: World, s: State, n: int) -> None:
    """Stay n ticks in the current cell; pay health for the ticks in which it burns."""
    in_fire = min(n, max(0, s.t + n - max(w.burn[s.pos], s.t)))
    s.hp -= w.task.hp_loss * in_fire
    s.t += n


def route(w: World, start, t0: int, goal, cross: bool, foresight: bool):
    """Breadth-first search: the shortest list of cells from start to a cell where goal(cell) is true."""
    prev, layer, t = {start: None}, [start], t0
    while layer:
        for c in layer:
            if goal(c):
                path = []
                while c != start:
                    path.append(c)
                    c = prev[c]
                return path[::-1]
        t += STEP_TICKS
        nxt = []
        for i, j in layer:
            for di, dj in NEIGHBOURS:
                n = (i + di, j + dj)
                if n in prev or n not in w.open:
                    continue  # seen already, a wall, furniture, off the grid, or too close to one (clearance)
                if not cross and w.burn[n] <= (t if foresight else t0):
                    continue  # on fire: when we would get there (foresight) or right now (no foresight)
                prev[n] = (i, j)
                nxt.append(n)
        layer = nxt
    return None


def within_reach(w: World, k: int):
    """Goal test for item k: true for every cell the robot can pick it from (task.reach cells away, corners too)."""
    (ci, cj), r = w.cells[k], w.task.reach
    return lambda c: abs(c[0] - ci) <= r and abs(c[1] - cj) <= r


def walk(w: World, s: State, goal, cross: bool, foresight: bool, until: float = math.inf) -> bool:
    """Walk until goal(pos). Without foresight the route is recomputed after every cell, as a robot would.

    until: give up (return False) once the clock reaches this tick, for a body that acts only while a thought lasts.
    """
    while not goal(s.pos):
        if s.t >= until:
            return False
        path = route(w, s.pos, s.t, goal, cross, foresight)
        if path is None:
            return False
        for c in path if foresight else path[:1]:
            s.pos = c
            spend(w, s, STEP_TICKS)
            if over(w, s):
                return False
    return True


def pick(w: World, s: State, only: int | None = None) -> bool:
    """Pick item `only`, or any item, if the robot is within reach of it."""
    if s.carrying and not w.task.bag:
        return False  # no bag: one item at a time
    for k in range(len(w.cells)) if only is None else [only]:
        if k not in s.taken and within_reach(w, k)(s.pos):
            spend(w, s, w.task.pick_ticks)
            s.taken, s.carrying, s.held = s.taken | {k}, s.carrying + 1, s.held + value(w, k, s.t)
            return True
    return False


def drop(w: World, s: State) -> bool:
    """Bank everything carried. Only in the safe strip, and only if the robot is still alive and in time."""
    if not s.carrying or s.pos not in w.safe:
        return False
    spend(w, s, DROP_TICKS)
    if not over(w, s):
        s.saved += s.held
    s.held, s.carrying = 0.0, 0
    return True


def act(w: World, s: State, action: tuple, foresight: bool = False, until: float = math.inf) -> bool:
    """Carry out ("get", k, cross) or ("home", None, cross). True if it was completed in time."""
    kind, k, cross = action
    goal = within_reach(w, k) if kind == "get" else (lambda c: c in w.safe)
    if not walk(w, s, goal, cross, foresight, until):
        return False
    return (pick(w, s, k) if kind == "get" else drop(w, s)) and not over(w, s)


def unreachable(w: World) -> int:
    """Items the robot could never get to, even through fire (walls, furniture, clearance)."""
    return sum(route(w, w.start, 0, within_reach(w, k), True, False) is None for k in range(len(w.cells)))


def random_agent(w: World, seed: int) -> float:
    """Uniform over the commands: go to item k, go home, pick, drop, wait. No 'done', which would only hurt it."""
    rng, s, n = np.random.default_rng(seed), State(w.start), len(w.cells)
    while not over(w, s):
        c, before = int(rng.integers(n + 4)), s.t
        if c < n:
            walk(w, s, within_reach(w, c), False, False)
        elif c == n:
            walk(w, s, lambda cell: cell in w.safe, False, False)
        elif c == n + 1:
            pick(w, s)
        elif c == n + 2:
            drop(w, s)
        if s.t == before:
            spend(w, s, 1)  # a command that did nothing still costs a tick
    return score(w, s)


def by_value(w: World, s: State, k: int) -> float:
    return value(w, k, s.t)


def by_value_per_tick(w: World, s: State, k: int) -> float:
    """Diagnostic only: value divided by the ticks of walking it takes to reach the item."""
    path = route(w, s.pos, s.t, within_reach(w, k), False, False)
    return -1.0 if path is None else value(w, k, s.t) / (STEP_TICKS * max(1, len(path)))


# When a reflex with a bag turns for home. "reflex" is the official one (CR, Nov 22): bank after every pick.
EXIT_RULES = {
    "reflex": {"every": 1},
    "reflex_two": {"every": 2},
    "reflex_cov": {"every": None, "by_coverage": True},
    "reflex_covcap": {"every": None, "by_coverage": True, "by_cap": True},
}


def reflex(w: World, key=by_value, **rule) -> float:
    """The reflex as in HAZARD and RRARA: go for the most valuable item, never step into fire."""
    s = State(w.start)
    run_reflex(w, s, key=key, **rule)
    return score(w, s)


def run_reflex(w: World, s: State, key=by_value, every=1, by_coverage=False, by_cap=False, until=math.inf) -> None:
    """Let the reflex act from state s until the episode ends, it has nothing left to do, or the clock reaches `until`."""
    skip = set()
    while not over(w, s) and s.t < until:
        todo = [k for k in live(w, s) if k not in skip]
        leave = by_coverage and coverage(w, s.t) >= w.task.end_coverage - LEAVE_MARGIN
        leave = leave or (by_cap and w.task.max_ticks - s.t <= w.task.pick_ticks + 100)
        full = not w.task.bag or (every is not None and s.carrying >= every)
        if s.carrying and (full or leave or not todo):
            if not act(w, s, ("home", None, False), until=until):
                break  # trapped with the load, dead, out of time, or told to stop
        elif todo and not leave:
            near = lambda k: -abs(w.cells[k][0] - s.pos[0]) - abs(w.cells[k][1] - s.pos[1])
            k = max(todo, key=lambda k: (key(w, s, k), near(k)))
            if not act(w, s, ("get", k, False), until=until) and not over(w, s) and s.t < until:
                skip.add(k)  # fire only grows, so it stays unreachable
        else:
            break


def plan(w: World, start: State, foresight: bool = True) -> tuple:
    """Exhaustive search from `start` in world w. Returns (most value banked, the actions of that plan in order)."""
    best, first, fronts = start.saved, (), {}

    def search(s: State, done: tuple) -> None:
        nonlocal best, first
        if s.saved > best:
            best, first = s.saved, done
        todo = live(w, s)
        if s.saved + s.held + sum(value(w, k, s.t) for k in todo) <= best:
            return  # even banking everything left cannot beat the best found so far
        labels = fronts.setdefault((s.taken, s.pos, s.carrying), [])
        if any(t <= s.t and hp >= s.hp and held >= s.held and saved >= s.saved for t, hp, held, saved in labels):
            return  # same items taken, same place, and an earlier visit was at least as good in every way
        labels.append((s.t, s.hp, s.held, s.saved))
        options = [("get", k) for k in sorted(todo, key=lambda k: -value(w, k, s.t))] if w.task.bag or not s.carrying else []
        for kind, k in options + ([("home", None)] if s.carrying else []):
            seen = set()
            for cross in (False, True):
                s2 = copy.copy(s)
                if act(w, s2, (kind, k, cross), foresight) and (s2.t, s2.hp) not in seen:
                    seen.add((s2.t, s2.hp))
                    search(s2, done + ((kind, k, cross),))

    search(copy.copy(start), ())
    return best, first


def planner_all(w: World) -> float:
    """Upper bound: every order of rescues, every cross-the-fire choice, knowing the future fire."""
    return plan(w, State(w.start))[0] / sum(w.values)


def forecast(w: World, t: int) -> World:
    """The world as seen at tick t: what burns now is known; the front is assumed to advance at its average speed."""
    # ticks per cell; the measured 30 is for p = 0.015 and the speed scales with p (p = 0: the fire never moves)
    front = round(FRONT_TICKS * 0.015 / w.task.p_spread) if w.task.p_spread > 0 else math.inf
    burn = {c: (b if b <= t else math.inf) for c, b in w.burn.items()}
    layer, d = [c for c, b in burn.items() if b <= t], 0
    while layer and front < math.inf:
        d, nxt = d + 1, []
        for i, j in layer:
            for di, dj in NEIGHBOURS:
                n = (i + di, j + dj)
                if burn.get(n) == math.inf:
                    burn[n] = t + d * front
                    nxt.append(n)
        layer = nxt
    return build(w.task, burn, w.cells, w.values, w.start)


def stand(w: World, s: State, until: float) -> None:
    """What the body does during a thought, version 1: nothing."""
    spend(w, s, max(0, int(until) - s.t))


def reflex_until(w: World, s: State, until: float) -> None:
    """What the body does during a thought, version 2: the reflex keeps acting, then waits if it runs out of things to do."""
    run_reflex(w, s, until=until)
    if not over(w, s):
        stand(w, s, until)


def thinker(w: World, charge: int = 0, commit: bool = False, moving: bool = False) -> float:
    """The agents that plan from the fire as it is now. Each thought costs `charge` ticks of world time.

    commit False: do the first action of the plan, then think again (always deliberate; planner_now).
    commit True:  follow the whole plan and think again only when an action cannot be carried out
                  (no fire-free route, item gone) or the plan has run out (agent A, plan and commit).
    moving True:  the reflex acts during the thought instead of the robot standing (agent B).
    The planner knows its cost and what the body does meanwhile, so it plans from where the robot will be.
    """
    s, body = State(w.start), (reflex_until if moving else stand)
    while not over(w, s):
        before, believed, after = s.t, forecast(w, s.t), copy.copy(s)
        body(believed, after, before + charge)  # where the planner expects the robot to be when the thought ends
        actions = plan(believed, after)[1]
        if not actions and not s.carrying:
            break  # nothing in the forecast is worth going for
        body(w, s, before + charge)  # the thought happens in the real world: the fire advances
        for action in (actions or (("home", None, True),))[: None if commit else 1]:
            if over(w, s) or not act(w, s, action):
                break  # blocked, or the item is gone: back to thinking
        if s.t == before:
            spend(w, s, 1)
    return score(w, s)


def planner_now(w: World, charge: int = 0) -> float:
    """Always deliberate: forecast, plan, do one action, repeat; `charge` ticks per thought with the robot standing."""
    return thinker(w, charge)


def evaluate(task: Task, seeds=range(25), charges=(0,)) -> dict:
    """Per-seed scores of every agent on one rule set. Clock charges other than 0 add planner_now_<charge>."""
    worlds = [make_world(task, seed) for seed in seeds]
    out = {"random": [random_agent(w, 10_000 + seed) for w, seed in zip(worlds, seeds)]}
    for name, rule in EXIT_RULES.items():
        out[name] = [reflex(w, **rule) for w in worlds]
    out["reflex_vpt"] = [reflex(w, by_value_per_tick) for w in worlds]
    for c in charges:
        out["planner_now" if c == 0 else f"planner_now_{c}"] = [planner_now(w, c) for w in worlds]
    out = {name: np.array(v) for name, v in out.items()}
    out["planner_all"] = np.max([np.array([planner_all(w) for w in worlds])] + list(out.values()), axis=0)  # a bound
    out["t_end"] = np.array([w.t_end for w in worlds])
    out["unreachable"] = np.array([unreachable(w) for w in worlds])
    return out


def paired_interval(diff: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """95 % bootstrap interval of the mean of per-seed differences."""
    rng = np.random.default_rng(0)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
