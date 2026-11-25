"""The validation step of Liane's method on the fire task (A99; a port of scripts/compiled/validate.py).

A program plays a list of rooms alone, with no model. For every room: the score, whether the robot died, how many
of its commands were illegal or raised, the number of decisions and the time `decide` took. Our reflex plays the
same rooms, so each room shows both scores. The file written is what the next writing round reads, and with
--first 50 --rooms 50 it is also the final test.

Run:  python -m firelab.validate --program firelab/results/firelab_compiled/v1/policy_round1.py --first 110 --rooms 10 \\
          --out firelab/results/firelab_compiled/v1/validation_round1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

import firelab  # noqa: F401
from firelab.agents import reflex
from firelab.env import FireLab3DEnv
from firelab.gate import paired_interval
from firelab.interface import Session
from firelab.programs import load_program, run_program
from firelab.skills import Skills


def play_rooms(program_path, rooms, items: int = 8, values: tuple = (1, 2, 3, 5, 8), with_reflex: bool = True) -> dict:
    """Play each room with the program (and the reflex). Returns the validation record."""
    module = load_program(program_path)  # raises if the audit fails: an unfit program never plays
    env = FireLab3DEnv(level=2, num_items=items, values=values, realistic_bg=False)
    skills, per_room = Skills(env), {}
    try:
        for room in rooms:
            row = {}
            if with_reflex:
                env.reset(seed=room)
                skills.start()
                reflex(skills)
                row["reflex"] = round(skills.score(), 4)
            env.reset(seed=room)
            skills.start()
            log = run_program(Session(skills), module)
            seconds = [r["seconds"] for r in log]
            row.update(
                score=round(skills.score(), 4), ticks=skills.ticks, health=skills.info.get("health", 100.0),
                died=skills.info.get("health", 100.0) <= 0, decisions=len(log),
                illegal=sum(r["reason"].startswith("illegal") for r in log), raised=sum(bool(r["error"]) for r in log),
                failed=sum((not r["done"]) and not r["reason"].startswith("illegal") for r in log),
                seconds_per_decision=round(statistics.mean(seconds), 5) if seconds else 0.0,
                slowest_decision=round(max(seconds), 5) if seconds else 0.0,
                commands=[" ".join(str(part) for part in r["command"]) + ("" if r["done"] else f" [{r['reason']}]") for r in log][:40],
            )
            per_room[str(room)] = row
    finally:
        env.close()
    return summarise(str(program_path), per_room)


def summarise(program: str, per_room: dict) -> dict:
    scores = np.array([r["score"] for r in per_room.values()])
    out = {"program": program, "rooms": per_room, "n": len(scores), "mean": round(float(scores.mean()), 4),
           "deaths": sum(r["died"] for r in per_room.values()), "illegal": sum(r["illegal"] for r in per_room.values()),
           "raised": sum(r["raised"] for r in per_room.values()),
           "seconds_per_decision": round(statistics.mean(r["seconds_per_decision"] for r in per_room.values()), 5),
           "slowest_decision": max(r["slowest_decision"] for r in per_room.values())}
    if all("reflex" in r for r in per_room.values()):
        base = np.array([r["reflex"] for r in per_room.values()])
        lo, hi = paired_interval(scores - base)
        out.update(reflex_mean=round(float(base.mean()), 4), gap_to_reflex=round(float(np.mean(scores - base)), 4),
                   gap_interval=[round(lo, 4), round(hi, 4)],
                   ahead_level_behind=[int((scores - base > 0.005).sum()), int((abs(scores - base) <= 0.005).sum()), int((scores - base < -0.005).sum())])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--first", type=int, default=110, help="first room")
    ap.add_argument("--rooms", type=int, default=10, help="number of rooms")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--values", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    record = play_rooms(args.program, range(args.first, args.first + args.rooms), args.items, tuple(args.values))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "rooms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
