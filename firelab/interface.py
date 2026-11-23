"""The interface every outside agent sits on (chunk 7a): the room as one dictionary, and the commands.

A Session wraps the robot in the simulator (firelab.skills.Skills). An agent that is not ours (a compiled
program, a language model, RRARA's checker) never touches the simulator: it reads `observe()` and answers with
a command, which `do()` carries out through the skills and answers with Result(ticks, done, reason).

Commands are tuples (lists are accepted):
  ("get", k)            walk into reach of item k by a fire-free route and pick it      ("get", k, True): may cross fire
  ("home",)             walk to the safe strip and bank the bag                         ("home", True): may cross fire
  ("go", (i, j))        walk to cell (i, j)                                             ("go", (i, j), True)
  ("pick", k) / ("pick",)   pick item k, or any item in reach, without walking
  ("drop",)             bank the bag here (only in the safe strip)
  ("wait", n)           stand n ticks (1 to 100)
A command that is not well-formed or names something that is not there is not carried out: it costs one tick,
is recorded as illegal with the reason, and the episode goes on (as in scripts/compiled/API.md).
The room is fully observed: every item and every burning cell is in the dictionary (PLAN.md section 6).
"""

from __future__ import annotations

from firelab.fire import BLOCKED, BURNING
from firelab.items import BAG, FLOOR, LOST, SAVED
from firelab.skills import Result, Skills

API_VERSION = "1"
WHERE = {FLOOR: "floor", BAG: "bag", SAVED: "saved", LOST: "lost"}
MAX_WAIT = 100


class Session:
    """One episode, seen from outside. Make a new one after every env.reset (and call skills.start())."""

    def __init__(self, skills: Skills, allow_fire: bool = True) -> None:
        self.skills = skills
        # False: no command may walk through fire, which is what the reflex is held to (A93 a, CR Nov 24). The menu
        # then lists only what a fire-free route reaches, and a "through fire" command is illegal.
        self.allow_fire = allow_fire
        self.history: list = []  # (command, Result), oldest first

    def observe(self) -> dict:
        """Everything an agent may know, as plain numbers, strings and lists (it survives json.dumps)."""
        s = self.skills
        fire, items, fl, rules = s.fire, s.items, s.inner.fl, s.items.rules
        pose, cell = s.inner.robot.base.get_pose(), s.cell()
        cells_of = lambda state: sorted([int(i), int(j)] for i, j in zip(*(fire.state == state).nonzero()))
        ticks_per_cell = round(fire.config.cell_size / fl.max_base_step)
        home = s.route(s.safe)
        listed = []
        for k in range(s.count):
            on_floor = items.where[k] == FLOOR
            path = s.route(s.near(k)) if on_floor else None
            listed.append({
                "id": k, "cell": list(items.cells[k]), "value": float(items.value[k]), "start": items.start[k],
                "where": WHERE[items.where[k]], "burning": bool(on_floor and fire.state[items.cells[k]] == BURNING),
                "cells_away": None if path is None else len(path),
                "ticks_to_get": None if path is None else ticks_per_cell * len(path) + rules.pick_ticks,
            })
        last = self.history[-1] if self.history else None
        return {
            "api_version": API_VERSION, "tick": s.ticks, "ticks_left": fl.max_ticks - s.ticks, "fire_routes_allowed": self.allow_fire,
            "health": s.info.get("health", fl.health), "max_health": fl.health,
            "burning_share": round(fire.coverage(), 4), "end_share": fl.end_coverage,
            "robot": {"cell": list(cell), "x": round(float(pose.x), 3), "y": round(float(pose.y), 3),
                      "in_safe": s.safe(cell), "in_fire": bool(fire.state[cell] == BURNING)},
            "bag": [k for k in range(s.count) if items.where[k] == BAG], "bag_value": round(items.total(BAG), 4),
            "banked_value": round(items.total(SAVED), 4), "total_value": sum(items.start), "score": round(s.score(), 4),
            "items": listed,
            "home": {"cells_away": None if home is None else len(home),
                     "ticks": None if home is None else ticks_per_cell * len(home) + rules.drop_ticks},
            "grid": {"nx": fire.config.shape[0], "ny": fire.config.shape[1], "cell_m": fire.config.cell_size,
                     "burning": cells_of(BURNING), "walls": cells_of(BLOCKED),
                     "safe_rows_from": min(j for i, j in s.open if s.safe((i, j))), "open": sorted(list(c) for c in s.open)},
            "job": None if items.job is None else {"kind": items.job[0], "item": items.job[1], "ticks_left": items.left},
            "last_command": None if last is None else _plain(last[0]),
            "last_result": None if last is None else {"ticks": last[1].ticks, "done": last[1].done, "reason": last[1].reason},
            "costs": {"ticks_per_cell": ticks_per_cell, "pick_ticks": rules.pick_ticks, "drop_ticks": rules.drop_ticks,
                      "item_loss_per_tick": rules.drain, "health_loss_per_tick": fl.health_loss_per_tick},
        }

    def legal(self, obs: dict | None = None) -> list:
        """The menu: the commands worth offering now. Items by value; through fire only where no fire-free route exists."""
        obs = obs or self.observe()
        live = [i for i in obs["items"] if i["where"] == "floor" and i["value"] > 0]
        menu = [("get", i["id"]) if i["cells_away"] is not None else ("get", i["id"], True)
                for i in sorted(live, key=lambda i: -i["value"])]
        if obs["bag"]:
            menu.append(("home",) if obs["home"]["cells_away"] is not None else ("home", True))
        if not self.allow_fire:
            menu = [c for c in menu if c[-1] is not True]
        return menu + [("wait", 10)]

    def check(self, command) -> str:
        """"" if the command can be attempted, else the reason it is illegal."""
        try:
            return self._check(command[0], list(command[1:]))
        except (TypeError, IndexError, ValueError):
            return "not a well-formed command"

    def _check(self, name, args: list) -> str:
        if args and isinstance(args[-1], bool):
            if args[-1] and not self.allow_fire:
                return "routes through fire are not allowed in this run"
            args = args[:-1]  # the "may cross fire" flag
        items = self.skills.items
        if name in ("home", "drop"):
            return "" if not args else f"{name} takes no argument"
        if name == "get" or (name == "pick" and args):
            ok = len(args) == 1 and isinstance(args[0], int) and 0 <= args[0] < self.skills.count
            return "" if ok and items.where[args[0]] == FLOOR else "no such item on the floor"
        if name == "pick":
            return ""
        if name == "go":
            ok = len(args) == 1 and len(args[0]) == 2 and tuple(args[0]) in self.skills.open
            return "" if ok else "not a cell the robot can stand on"
        if name == "wait":
            ok = len(args) == 1 and isinstance(args[0], int) and 1 <= args[0] <= MAX_WAIT
            return "" if ok else f"wait takes a number of ticks from 1 to {MAX_WAIT}"
        return f"unknown command {name!r}"

    def do(self, command) -> Result:
        """Carry out one command through the skills. Illegal: one tick is spent and nothing else happens."""
        s, reason = self.skills, self.check(command)
        if reason:
            result = Result(s.wait(1).ticks, False, "illegal: " + reason)
        else:
            name, args = command[0], list(command[1:])
            cross = bool(args.pop()) if args and isinstance(args[-1], bool) else False
            if name == "get":
                result = s.get(args[0], cross)
            elif name == "home":
                result = s.home(cross)
            elif name == "go":
                target = tuple(args[0])
                result = s.walk(lambda c: c == target, cross)
            elif name == "pick":
                result = s.pick(args[0] if args else None)
            elif name == "drop":
                result = s.drop()
            else:
                result = s.wait(args[0])
        self.history.append((command, result))
        return result


def _plain(command) -> list:
    """A command as JSON-friendly lists."""
    if not isinstance(command, (list, tuple)):
        return [str(command)]
    return [list(part) if isinstance(part, tuple) else part for part in command]
