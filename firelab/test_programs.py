"""Tests for firelab.programs (chunk 7c). One 8-item room is built; room 0 is played twice (about 15 s).

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_programs.py
"""

from pathlib import Path

import pytest

import firelab  # noqa: F401
from firelab.agents import reflex
from firelab.env import FireLab3DEnv
from firelab.interface import Session
from firelab.programs import audit_program, load_program, run_program
from firelab.skills import Skills

EXAMPLE = Path(__file__).parent / "examples" / "reflex_program.py"


def test_the_loader_refuses_programs_that_reach_outside(tmp_path):
    assert audit_program(EXAMPLE) == []
    bad = tmp_path / "bad.py"
    bad.write_text("import os\nfrom urllib import request\nimport numpy as np\nclass Policy:\n    def decide(self, obs):\n        return None\n")
    problems = audit_program(bad)
    assert problems == ["import of 'os' is not allowed", "import of 'urllib' is not allowed", "Policy has no `__init__` method"]
    with pytest.raises(ValueError, match="program refused"):
        load_program(bad)
    (tmp_path / "none.py").write_text("x = 1\n")
    assert audit_program(tmp_path / "none.py") == ["no `class Policy` at module level"]


def test_the_reflex_program_plays_exactly_as_the_reflex_and_a_broken_one_does_not_end_the_episode(tmp_path):
    env = FireLab3DEnv(level=2, realistic_bg=False)
    try:
        skills = Skills(env)
        env.reset(seed=0)
        skills.start()
        ours = reflex(skills)
        ours_score, ours_ticks = skills.score(), skills.ticks
        env.reset(seed=0)
        skills.start()
        log = run_program(Session(skills), load_program(EXAMPLE))
        assert [(r["command"], r["done"], r["tick"] + r["ticks"]) for r in log] == \
               [([name] if k is None else [name, k], result.done, tick) for name, k, tick, result in ours]
        assert (skills.score(), skills.ticks) == (ours_score, ours_ticks) and ours_score > 0
        assert all(r["seconds"] < 0.5 and r["error"] == "" for r in log)

        broken = tmp_path / "broken.py"
        broken.write_text("class Policy:\n    def __init__(self, rules):\n        self.n = 0\n    def decide(self, obs):\n"
                          "        self.n += 1\n        if self.n == 1:\n            return ('drop',)\n        if self.n == 2:\n"
                          "            return 1 / 0\n        if self.n == 3:\n            return ('fly',)\n        return None\n")
        env.reset(seed=0)
        skills.start()
        log = run_program(Session(skills), load_program(broken))
        assert [(r["reason"], r["error"] != "") for r in log] == [("refused", False), ("ok", True), ("illegal: unknown command 'fly'", False)]
        assert skills.ticks == 2 and not skills.over  # the refused drop was free; the error and the illegal command cost a tick each
    finally:
        env.close()
