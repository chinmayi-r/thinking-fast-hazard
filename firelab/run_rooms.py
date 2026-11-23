"""Play rooms in the simulator with any of the agents, optionally the same rooms on the rule check's board.

Agents: random, reflex (firelab/agents.py); always, commit, moving (firelab/thinkers.py), under a clock
(firelab/clock.py: off, declared, measured, fixed).

A run is a folder results/firelab_sim/<tag>/ holding
  rows_<first room>.csv   one row per room and agent: the scores, the ticks, the thinking in both costs;
  games/room_NNN_<agent>.jsonl   the game, decision by decision: what the agent saw, what it planned, how long
                          it thought (measured and declared), every command and its result (CR, Nov 23).
Each invocation appends to its own rows file, so rooms can run in parallel, one process per room (Della).

Run:    venv_kinder\\Scripts\\python.exe -m firelab.run_rooms --tag run1_off --agents always --clock off --rooms 50
Report: venv_kinder\\Scripts\\python.exe -m firelab.run_rooms --tag run1_off --report-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

import firelab  # noqa: F401
from firelab import gate
from firelab.agents import GateSkills, random_agent, reflex
from firelab.clock import SECONDS_PER_STEP, Clock
from firelab.env import FireLab3DEnv
from firelab.interface import Session
from firelab.model_agents import AGENTS as MODEL_AGENTS
from firelab.models import cost_usd, make_model
from firelab.planner import plan
from firelab.skills import Skills
from firelab.thinkers import AGENTS as THINKERS
from firelab.thinkers import thinker

AGENTS = ["random", "reflex", *THINKERS, *MODEL_AGENTS]
COLUMNS = ["room", "agent", "clock", "sim", "rule", "sim_ticks", "rule_ticks", "health", "commands", "thoughts",
           "charged_ticks", "measured_s", "declared_s", "steps", "estimate_error", "think_s", "think_ticks",
           "tokens_in", "tokens_out", "cost_usd",
           "parted_at", "sim_line", "rule_line"]
RUNS = Path(__file__).resolve().parents[1] / "results" / "firelab_sim"


def run_agent(name: str, body, room: int, clock: Clock, thoughts: list, allow_fire: bool = True, max_bag: int | None = None,
              caution: float = 1.0) -> list:
    if name == "random":
        return random_agent(body, 10_000 + room)  # the rule check's seed (gate.py:408)
    if name == "reflex":
        return reflex(body)
    return thinker(body, clock, thoughts=thoughts, allow_fire=allow_fire, max_bag=max_bag, caution=caution, **THINKERS[name])


def reference(task: gate.Task) -> dict:
    """Time one fixed plan (room 0, first thought), so every result file says how fast this machine is."""
    world = gate.make_world(task, 0)
    began = time.perf_counter()
    steps = plan(gate.forecast(world, 0), gate.State(world.start))[2]
    seconds = time.perf_counter() - began
    return {"host": platform.node(), "cpu": platform.processor(), "reference_steps": steps,
            "reference_s": round(seconds, 4), "reference_ms_per_step": round(1000 * seconds / steps, 4)}


def parted(sim_log: list, rule_log: list):
    """Index of the first command where the two logs differ in command, item or outcome; None if they never do."""
    for n, (a, b) in enumerate(zip(sim_log, rule_log)):
        if (a[0], a[1], a[3].done) != (b[0], b[1], b[3].done):
            return n
    return None if len(sim_log) == len(rule_log) else min(len(sim_log), len(rule_log))


def _line(log: list, n) -> str:
    if n is None or n >= len(log):
        return ""
    name, k, tick, result = log[n]
    return f"{name} {'' if k is None else k} ends tick {tick}: {result.reason}".replace("  ", " ")


def save_game(path: Path, head: dict, thoughts: list, log: list, end: dict) -> None:
    """One JSON object per line: the episode, then every thought, then every command, then the outcome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"type": "episode", **head}) + "\n")
        for record in thoughts:
            f.write(json.dumps({"type": "thought", **record}) + "\n")
        for name, k, tick, result in log:
            f.write(json.dumps({"type": "command", "name": name, "item": k, "ended_tick": tick, "ticks": result.ticks,
                                "done": result.done, "reason": result.reason}) + "\n")
        f.write(json.dumps({"type": "end", **end}) + "\n")


def play(env, skills: Skills, task: gate.Task, room: int, name: str, clock: Clock | None = None,
         board: bool = True, games: Path | None = None, head: dict | None = None, model=None, options: dict | None = None) -> dict:
    """One room, one agent: in the simulator, and (board=True) on the rule check's board under the same clock."""
    clock = clock or Clock()
    rule_log, rule_body, options = [], None, options or {}
    if name in MODEL_AGENTS:
        board = False  # the interface the models use exists in the simulator only
    if board:
        rule_body = GateSkills(gate.make_world(task, room))
        rule_log = run_agent(name, rule_body, room, clock, [], options.get("allow_fire", True), options.get("max_bag"), options.get("caution", 1.0))
    env.reset(seed=room)
    skills.start()
    thoughts: list = []
    began = time.perf_counter()
    if name in MODEL_AGENTS:  # thoughts holds one record per call: what the model saw, what it answered, both times
        sim_log = MODEL_AGENTS[name](Session(skills, allow_fire=options.get("allow_fire", True)), model, records=thoughts, **options)
    else:
        sim_log = run_agent(name, skills, room, clock, thoughts, options.get("allow_fire", True), options.get("max_bag"), options.get("caution", 1.0))
    tokens = [sum(t.get(k, 0) for t in thoughts) for k in ("tokens_in", "tokens_out")]
    label = clock.mode if clock.mode != "fixed" else f"fixed{clock.fixed_ticks}"
    if name in MODEL_AGENTS:
        label = options.get("call_time", "measured") + ("-" + options.get("accounting", "free") if name == "rrara1" else "")
    think = max(0.0, time.perf_counter() - began - skills.sim_seconds)
    n = parted(sim_log, rule_log) if board else None
    row = {
        "room": room, "agent": name, "clock": label,
        "sim": round(skills.score(), 4), "rule": round(rule_body.score(), 4) if board else "",
        "sim_ticks": skills.ticks, "rule_ticks": rule_body.ticks if board else "",
        "health": skills.info.get("health", 100.0), "commands": len(sim_log), "thoughts": len(thoughts),
        "charged_ticks": sum(t["charged"] for t in thoughts), "measured_s": round(sum(t["measured_s"] for t in thoughts), 4),
        "declared_s": round(sum(t["declared_s"] for t in thoughts), 4), "steps": sum(t["steps"] for t in thoughts),
        "estimate_error": round(float(np.mean([abs(t["estimate"] - t["charged"]) for t in thoughts])), 2) if thoughts else "",
        "think_s": round(think, 4), "think_ticks": math.ceil(think * 10),
        "tokens_in": tokens[0], "tokens_out": tokens[1], "cost_usd": round(cost_usd(model.model, *tokens), 6) if model is not None and name in MODEL_AGENTS else 0.0,
        "parted_at": "" if n is None else n, "sim_line": _line(sim_log, n), "rule_line": _line(rule_log, n),
    }
    if games is not None:
        end = {"score": row["sim"], "ticks": skills.ticks, "health": row["health"]}
        save_game(games / f"room_{room:03d}_{name}.jsonl", {**(head or {}), "room": room, "agent": name, "clock": row["clock"]},
                  thoughts, sim_log, end)
    return row


def report(rows: list) -> None:
    names = [a for a in AGENTS if any(r["agent"] == a for r in rows)]
    cell = {(int(r["room"]), r["agent"]): r for r in rows}
    print("room | " + " | ".join(f"{a} sim  rule" for a in names))
    for room in sorted({int(r["room"]) for r in rows}):
        show = lambda r: "     -     " if r is None else f"{float(r['sim']):.2f}  {float(r['rule']):.2f}" if r["rule"] != "" else f"{float(r['sim']):.2f}     -"
        print(f"{room:4d} | " + " | ".join(show(cell.get((room, a))).rjust(len(a) + 10) for a in names))
    for name in names:
        rs = [r for r in rows if r["agent"] == name]
        sim = np.array([float(r["sim"]) for r in rs])
        line = f"{name} ({rs[0]['clock']}): {len(rs)} rooms, simulator {sim.mean():.3f}"
        if all(r["rule"] != "" for r in rs):
            rule = np.array([float(r["rule"]) for r in rs])
            line += f", rule check {rule.mean():.3f}, mean difference {np.mean(sim - rule):+.3f}, rooms differing by more than 0.01: {int(np.sum(np.abs(sim - rule) > 0.01))}"
        if int(rs[0]["thoughts"] or 0) or any(int(r["thoughts"] or 0) for r in rs):
            f = lambda k: np.mean([float(r[k]) for r in rs])
            line += (f"; per episode: {f('thoughts'):.1f} thoughts, {f('steps'):.0f} search steps, measured {f('measured_s'):.2f} s, "
                     f"declared {f('declared_s'):.2f} s, charged {f('charged_ticks'):.1f} ticks")
        if any(int(r["tokens_in"] or 0) for r in rs):
            line += f"; tokens in {f('tokens_in'):.0f}, out {f('tokens_out'):.0f}, {sum(float(r['cost_usd']) for r in rs):.4f} dollars for these rooms"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="the run's folder under results/firelab_sim/")
    ap.add_argument("--agents", nargs="+", default=["random", "reflex"], choices=AGENTS)
    ap.add_argument("--clock", default="off", choices=["off", "declared", "measured", "fixed"])
    ap.add_argument("--fixed-ticks", type=int, default=0)
    ap.add_argument("--seconds-per-step", type=float, default=SECONDS_PER_STEP, help="the declared cost of one search step")
    ap.add_argument("--first", type=int, default=0, help="first room")
    ap.add_argument("--rooms", type=int, default=50, help="number of rooms")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--values", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--no-board", action="store_true", help="skip the rule check's board (halves the planning time)")
    ap.add_argument("--model", default="fake", help="for the agents model and rrara1: jev, claude-opus-5-5 or fake")
    ap.add_argument("--key-file", default=None, help="path of the file holding the model's key (the key is never printed)")
    ap.add_argument("--accounting", default="free", choices=["free", "strict"], help="rrara1: is the checker's time counted")
    ap.add_argument("--call-time", default="measured", choices=["measured", "declared"], help="what a model call is charged")
    ap.add_argument("--seconds-per-call", type=float, default=0.3, help="the declared time of one model call")
    ap.add_argument("--caution", type=float, default=1.0, help="planner agents: how much faster than average the forecast fire spreads")
    ap.add_argument("--max-bag", type=int, default=None, help="planner agents: bank once this many items are carried (1 = as the reflex)")
    ap.add_argument("--fire-free-menu", action="store_true",
                    help="no agent may walk through fire, as the reflex never does: model menus and the planner's plans")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    folder = RUNS / args.tag
    read = lambda: [r for p in sorted(folder.glob("rows_*.csv")) for r in csv.DictReader(p.open(newline=""))]
    done = {(r["room"], r["agent"]) for r in read()}
    todo = [(room, name) for room in range(args.first, args.first + args.rooms) for name in args.agents if (str(room), name) not in done]
    if todo and not args.report_only:
        values = tuple(args.values)
        task = gate.Task(items=args.items, values=values, layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)
        head = {"items": args.items, "values": values, "seconds_per_step": args.seconds_per_step, **reference(task)}
        uses_model = any(a in MODEL_AGENTS for a in args.agents)
        model = make_model(args.model, args.key_file) if uses_model else None
        options = {"accounting": args.accounting, "call_time": args.call_time, "declared_seconds": args.seconds_per_call,
                   "allow_fire": not args.fire_free_menu, "max_bag": args.max_bag, "caution": args.caution}
        head.update(allow_fire=options["allow_fire"], max_bag=args.max_bag, caution=args.caution)
        if uses_model:
            head.update(model=args.model, **options)
        print("machine:", head, flush=True)
        env = FireLab3DEnv(level=2, num_items=args.items, values=values, realistic_bg=False)
        skills = Skills(env)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"rows_{args.first:03d}.csv"
        fresh = not out.exists()
        with out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if fresh:
                writer.writeheader()
            for room, name in todo:
                clock = Clock(args.clock, args.fixed_ticks, args.seconds_per_step)
                writer.writerow(play(env, skills, task, room, name, clock, not args.no_board, folder / "games", head, model, options))
                f.flush()
        env.close()
    report(read())


if __name__ == "__main__":
    main()
