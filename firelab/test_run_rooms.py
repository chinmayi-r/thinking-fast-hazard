"""Tests for firelab.run_rooms (chunks 5b and 6c). One 6-item room is built; room 3 is played by four agents (about 20 s).

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_run_rooms.py
"""

import json

from firelab import gate
from firelab.clock import Clock
from firelab.env import FireLab3DEnv
from firelab.run_rooms import COLUMNS, parted, play, reference, report
from firelab.skills import Result, Skills

SIX = gate.Task(items=6, values=(1, 2, 3, 5, 8), layout="one_wide", clearance=True, reach=1, bag=True, hp_loss=2.0)


def test_parted_finds_the_first_command_that_differs():
    ok, bad = Result(4, True, "ok"), Result(4, False, "no route")
    log = [("get", 1, 10, ok), ("home", None, 20, ok)]
    assert parted(log, log) is None
    assert parted(log, [("get", 1, 12, ok), ("home", None, 22, ok)]) is None  # ticks alone do not count as parting
    assert parted(log, [("get", 1, 12, ok), ("home", None, 22, bad)]) == 1
    assert parted(log, [("get", 2, 10, ok)]) == 0 and parted(log, log[:1]) == 1


def test_play_gives_a_full_row_and_a_saved_game_for_each_kind_of_agent(tmp_path, capsys):
    head = reference(SIX)
    assert head["reference_steps"] == 257 and head["reference_s"] > 0  # room 0 at 6 items: the same count everywhere
    env = FireLab3DEnv(level=2, num_items=6, realistic_bg=False)
    try:
        skills = Skills(env)
        rows = [play(env, skills, SIX, 3, name, games=tmp_path, head=head) for name in ("random", "reflex")]
        rows.append(play(env, skills, SIX, 3, "commit", Clock("declared"), games=tmp_path, head=head))
        rows.append(play(env, skills, SIX, 3, "always", Clock("off"), board=False, games=tmp_path, head=head))
    finally:
        env.close()
    for row in rows:
        assert list(row) == COLUMNS and 0.0 <= row["sim"] <= 1.0 and row["sim_ticks"] <= 500
    assert rows[1]["rule"] == round(gate.reflex(gate.make_world(SIX, 3)), 4)  # the rule check's own number
    commit, always = rows[2], rows[3]
    assert commit["clock"] == "declared" and commit["thoughts"] >= 1 and commit["steps"] > 0 and commit["rule"] != ""
    assert always["clock"] == "off" and always["charged_ticks"] == 0 and always["rule"] == "" and always["thoughts"] >= always["commands"]
    game = [json.loads(line) for line in (tmp_path / "room_003_commit.jsonl").read_text().splitlines()]
    assert [g["type"] for g in game][0] == "episode" and game[-1] == {"type": "end", "score": commit["sim"], "ticks": commit["sim_ticks"], "health": commit["health"]}
    thought = next(g for g in game if g["type"] == "thought")
    assert {"seen", "plan", "estimate", "measured_s", "steps", "declared_s", "ticks_measured", "ticks_declared", "charged"} <= set(thought)
    assert game[0]["reference_steps"] == 257 and any(g["type"] == "command" for g in game)
    report([{k: str(v) for k, v in row.items()} for row in rows])
    printed = capsys.readouterr().out
    assert "reflex (off): 1 rooms" in printed and "commit (declared): 1 rooms" in printed and "search steps" in printed
