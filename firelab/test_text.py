"""Tests for firelab.text (chunk 7b). One 6-item room with the fire frozen is built (about 6 s).

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_text.py
"""

import dataclasses

import pytest

import firelab  # noqa: F401
from firelab.env import LEVEL2, FireLab3DEnv
from firelab.fire import BURNING, FireGridConfig
from firelab.interface import Session
from firelab.items import ItemRules
from firelab.skills import Skills
from firelab.text import describe, room_map, say


@pytest.fixture(scope="module")
def session():
    frozen = dataclasses.replace(LEVEL2, items=ItemRules(count=6), grid=FireGridConfig(p_spread=0.0))
    env = FireLab3DEnv(firelab_config=frozen, realistic_bg=False)
    env.reset(seed=2)  # fire at (13, 3); items at (13, 10), (4, 9), (3, 3), (13, 4), (2, 1), (4, 7) worth 3, 3, 1, 8, 5, 5
    yield Session(Skills(env))
    env.close()


def test_the_map_shows_walls_fire_items_the_strip_and_the_robot(session):
    rows = room_map(session.observe()).split("\n")
    assert len(rows) == 15 and all(len(row) == 3 + 16 for row in rows)
    assert rows[0].startswith("14 ") and rows[-1].startswith(" 0 ")
    assert rows[14 - 7][3 + 9] == "R" and rows[14 - 3][3 + 13] == "F" and rows[14 - 9][3 + 4] == "1"
    assert rows[14 - 5] == " 5 ######....######" and rows[14 - 11] == "11 ================"


def test_the_text_states_values_distances_costs_and_the_options(session):
    obs = session.observe()
    menu = session.legal(obs)
    text = describe(obs, [say(c, obs) for c in menu], session.history)
    assert "Time: tick 0; at most 500 ticks are left" in text and "when 50 % of the floor burns; 0 % burns now" in text
    assert "You: cell (9, 7), health 100 of 100. At health 0" in text and "Bag: empty. Banked so far: 0.0 of 25.0." in text
    assert "  item 3 at (13, 4): worth 8.0 (started at 8); 9 cells away, about 46 ticks to walk there and pick it" in text
    assert "Safe strip (rows 11 and above): 4 cells away, about 21 ticks to walk there and bank" in text
    assert "What you did so far:\n  nothing yet" in text
    assert "Options:\nA. get item 3 (worth 8.0 now; about 46 ticks)" in text and text.endswith("G. wait 10 ticks")


def test_the_text_follows_the_episode(session):
    session.do(("get", 1))
    session.do(("fly",))
    session.skills.fire.state[4, 7] = BURNING  # item 5's cell, by hand
    try:
        obs = session.observe()
        text = describe(obs, [say(c, obs) for c in session.legal(obs)], session.history, note="A NOTE.")
    finally:
        session.skills.fire.state[4, 7] = 0
    assert "Bag: 1 item(s) worth 3.0, not banked yet." in text and "  item 1 at" not in text
    assert "  1. get 1: ok, " in text and "  2. fly: illegal: unknown command 'fly', 1 ticks" in text
    assert "item 5 at (4, 7): worth 5.0 (started at 5), BURNING: loses 0.10 per tick, worth nothing in 50 ticks" in text
    assert "\n\nA NOTE.\n\nOptions:\n" in text and "go to the safe strip and bank the bag (worth 3.0; about" in text
