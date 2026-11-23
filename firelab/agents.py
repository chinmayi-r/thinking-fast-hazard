"""The two agents every result is measured against: random and the reflex. Copies of the rule check's.

Each agent takes a "body": something with the skills of firelab.skills.Skills (walk, get, home, pick,
drop, wait) and the few things an agent may look at (live, value, carrying, cell, item_cell, over,
ticks). Two bodies exist: Skills, which is the robot in the simulator, and GateSkills below, which is
the robot on the rule check's board (firelab/gate.py). So the same agent code runs in both worlds, and
each run leaves a log, one line per command: (command, item, tick when it ended, Result).
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np

from firelab import gate
from firelab.skills import Result


def reflex(body, log: list | None = None, until: float = math.inf) -> list:
    """The official reflex (CR, Nov 22): most valuable item first, bank after every pick, never enter fire.

    A copy of gate.run_reflex with the "bank after every pick" exit rule. `until` is a deadline tick: the reflex
    stops once the clock reaches it, after finishing a pick or drop it has begun (used while a plan is being made).
    """
    log = [] if log is None else log
    skip: set[int] = set()
    while not body.over and body.ticks < until:
        choice = reflex_choice(body, skip)
        if choice is None:
            break  # nothing carried and nothing left worth fetching
        if choice[0] == "home":
            result = body.home(until=until)
            log.append(("home", None, body.ticks, result))
            if not result.done:
                break  # trapped with the load, dead, or out of time
        else:
            result = body.get(choice[1], until=until)
            log.append(("get", choice[1], body.ticks, result))
            if not result.done and not body.over and body.ticks < until:
                skip.add(choice[1])  # fire only grows, so the item stays out of reach
    return log


def reflex_choice(body, skip: set) -> tuple | None:
    """The rule as one decision: ("home",) with anything in the bag, else ("get", k) for the most valuable item
    left (the nearer one on a tie), else None. `skip` holds the items a get could not reach."""
    if body.carrying():
        return ("home",)
    todo = [k for k in body.live() if k not in skip]
    if not todo:
        return None
    here = body.cell()
    far = lambda k: abs(body.item_cell(k)[0] - here[0]) + abs(body.item_cell(k)[1] - here[1])
    return ("get", max(todo, key=lambda k: (body.value(k), -far(k))))


def random_agent(body, seed: int, log: list | None = None) -> list:
    """Uniform over the commands: go to item k, go home, pick, drop, wait. A copy of gate.random_agent."""
    log = [] if log is None else log
    rng, n = np.random.default_rng(seed), body.count
    while not body.over:
        c, before = int(rng.integers(n + 4)), body.ticks
        if c < n:
            name, result = "go", body.walk(body.near(c))
        elif c == n:
            name, result = "go home", body.walk(body.safe)
        elif c == n + 1:
            name, result = "pick", body.pick()
        elif c == n + 2:
            name, result = "drop", body.drop()
        if c == n + 3 or body.ticks == before:
            name, result = ("wait" if c == n + 3 else name), body.wait(1)  # a command that did nothing still costs a tick
        log.append((name, c if c < n else None, body.ticks, result))
    return log


class GateSkills:
    """The same skills on the rule check's board: every method hands over to the function in firelab/gate.py."""

    def __init__(self, world: gate.World) -> None:
        self.w, self.s = world, gate.State(world.start)

    def _do(self, call) -> Result:
        since = self.s.t
        done = bool(call())
        return Result(self.s.t - since, done, "ok" if done else "episode over" if self.over else "failed")

    over = property(lambda self: gate.over(self.w, self.s))
    ticks = property(lambda self: self.s.t)
    count = property(lambda self: len(self.w.cells))

    def cell(self):
        return self.s.pos

    def live(self):
        return gate.live(self.w, self.s)

    def value(self, k):
        return gate.value(self.w, k, self.s.t)

    def carrying(self):
        return self.s.carrying

    def item_cell(self, k):
        return self.w.cells[k]

    def score(self):
        return gate.score(self.w, self.s)

    def believe(self, caution: float = 1.0):
        task = dataclasses.replace(self.w.task, p_spread=self.w.task.p_spread * caution)  # a faster fire in the forecast only
        return gate.forecast(dataclasses.replace(self.w, task=task), self.s.t), copy.copy(self.s)

    def near(self, k):
        return gate.within_reach(self.w, k)

    def safe(self, c):
        return c in self.w.safe

    def walk(self, goal, cross=False, until=math.inf):
        return self._do(lambda: gate.walk(self.w, self.s, goal, cross, False, until))

    def pick(self, k=None):
        return self._do(lambda: gate.pick(self.w, self.s, k))

    def drop(self):
        return self._do(lambda: gate.drop(self.w, self.s))

    def wait(self, ticks=1):
        return self._do(lambda: gate.spend(self.w, self.s, ticks) is None)

    def get(self, k, cross=False, until=math.inf):
        return self._do(lambda: gate.act(self.w, self.s, ("get", k, cross), until=until))

    def home(self, cross=False, until=math.inf):
        return self._do(lambda: gate.act(self.w, self.s, ("home", None, cross), until=until))
