"""The three agents that plan: one loop, two switches. A port of gate.thinker onto a body (chunk 6b).

  commit False, moving False   plan before every action, standing still while it thinks ("always").
  commit True,  moving False   plan once and follow the plan; think again only when an action cannot be carried out
                               or the plan has run out ("commit").
  commit True,  moving True    as commit, but the reflex acts while the plan is being made ("moving", the
                               reflex-while-planning agent).
None of the three is RRARA (Zheng et al. 2025): RRARA version 1 is a rule that acts plus a language model that
checks the rule's action in the background; its spec for this task is PLAN.md section 7 (CR, Nov 24).
Each thought is costed by a Clock (firelab/clock.py) and recorded with what the agent saw, for the saved games.
"""

from __future__ import annotations

import time

from firelab import gate
from firelab.agents import reflex
from firelab.clock import Clock
from firelab.planner import plan


def seen(world: gate.World, state: gate.State) -> dict:
    """What the agent knows when a thought begins, as plain numbers and lists (one record of a saved game)."""
    now = state.t
    return {
        "tick": now, "cell": list(state.pos), "health": state.hp, "carrying": state.carrying, "held": state.held,
        "saved": state.saved, "taken": sorted(state.taken), "end_forecast": world.t_end,
        "items": [{"id": k, "cell": list(world.cells[k]), "start": world.values[k], "value": round(gate.value(world, k, now), 4)}
                  for k in range(len(world.cells))],
        "burning": sorted(list(c) for c, lit in world.burn.items() if lit <= now),
    }


def thinker(body, clock: Clock, commit: bool = False, moving: bool = False, log: list | None = None,
            thoughts: list | None = None, allow_fire: bool = True) -> list:
    """Run one episode. Commands go to `log` as in firelab.agents; one dict per thought goes to `thoughts`."""
    log = [] if log is None else log
    thoughts = [] if thoughts is None else thoughts
    predict = gate.reflex_until if moving else gate.stand  # what the body is expected to do during the thought
    clock.start()
    while not body.over:
        before, began = body.ticks, time.perf_counter()
        believed, state = body.believe()
        record = {"seen": seen(believed, state), "estimate": clock.estimate(len(gate.live(believed, state)))}
        predict(believed, state, before + record["estimate"])  # where the robot should be when the thought ends
        value, actions, steps = plan(believed, state, cross_fire=allow_fire)
        record.update(clock.cost(time.perf_counter() - began, steps), plan=[list(a) for a in actions], plan_value=value)
        thoughts.append(record)
        if not actions and not body.carrying():
            break  # nothing in the forecast is worth going for
        until = before + record["charged"]  # the thought takes place in the world: the fire advances meanwhile
        if moving:
            reflex(body, log, until=until)
        if not body.over:
            body.wait(max(0, until - body.ticks))
        for kind, k, cross in (actions or (("home", None, allow_fire),))[: None if commit else 1]:
            if body.over:
                break
            result = body.get(k, cross) if kind == "get" else body.home(cross)
            log.append((kind, k, body.ticks, result))
            if not result.done:
                break  # blocked, or the item is gone: back to thinking
        if body.ticks == before:
            body.wait(1)
    return log


AGENTS = {  # name -> the two switches
    "always": {"commit": False, "moving": False},
    "commit": {"commit": True, "moving": False},
    "moving": {"commit": True, "moving": True},
}
