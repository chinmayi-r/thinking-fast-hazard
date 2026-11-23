"""Tests for firelab.interface (chunk 7a). One 6-item room with the fire frozen is built (about 6 s).

Run: venv_kinder\\Scripts\\python.exe -m pytest -p no:cacheprovider -q firelab/test_interface.py
"""

import dataclasses
import json

import pytest

import firelab  # noqa: F401
from firelab.env import LEVEL2, FireLab3DEnv
from firelab.fire import FireGridConfig
from firelab.interface import Session
from firelab.items import ItemRules
from firelab.skills import Result, Skills

ROOM = 2  # with 6 items: fire at (13, 3); items at (13, 10), (4, 9), (3, 3), (13, 4), (2, 1), (4, 7) worth 3, 3, 1, 8, 5, 5


@pytest.fixture(scope="module")
def env():
    frozen = dataclasses.replace(LEVEL2, items=ItemRules(count=6), grid=FireGridConfig(p_spread=0.0))
    e = FireLab3DEnv(firelab_config=frozen, realistic_bg=False)
    yield e
    e.close()


def _session(env) -> Session:
    env.reset(seed=ROOM)
    return Session(Skills(env))


def test_observe_is_plain_data_with_the_room_in_it(env):
    obs = _session(env).observe()
    assert json.loads(json.dumps(obs)) == obs  # nothing but numbers, strings, lists and dictionaries
    assert (obs["tick"], obs["ticks_left"], obs["health"], obs["score"], obs["bag"]) == (0, 500, 100.0, 0.0, [])
    assert obs["robot"] == {"cell": [9, 7], "x": 0.3, "y": 0.0, "in_safe": False, "in_fire": False}
    assert [(i["cell"], i["start"], i["where"]) for i in obs["items"]][:2] == [([13, 10], 3.0, "floor"), ([4, 9], 3.0, "floor")]
    assert obs["items"][1]["cells_away"] == 5 and obs["items"][1]["ticks_to_get"] == 5 * 4 + 10
    assert obs["grid"]["burning"] == [[13, 3]] and obs["grid"]["safe_rows_from"] == 11 and len(obs["grid"]["open"]) == 161
    assert obs["home"] == {"cells_away": 4, "ticks": 4 * 4 + 5} and obs["total_value"] == 25.0
    assert obs["costs"] == {"ticks_per_cell": 4, "pick_ticks": 10, "drop_ticks": 5, "item_loss_per_tick": 0.02, "health_loss_per_tick": 2.0}
    assert obs["last_command"] is None and obs["job"] is None


def test_the_menu_lists_items_by_value_and_home_only_with_a_bag(env):
    session = _session(env)
    assert session.legal() == [("get", 3), ("get", 4), ("get", 5), ("get", 0), ("get", 1), ("get", 2), ("wait", 10)]
    assert session.do(("get", 1)).done
    assert session.legal() == [("get", 3), ("get", 4), ("get", 5), ("get", 0), ("get", 2), ("home",), ("wait", 10)]


def test_do_carries_commands_out_and_the_observation_follows(env):
    session = _session(env)
    got = session.do(["get", 1])  # a list is accepted
    assert got.done and got.ticks <= session.history[0][1].ticks == got.ticks <= 30
    obs = session.observe()
    assert obs["bag"] == [1] and obs["bag_value"] == 3.0 and obs["items"][1]["where"] == "bag" and obs["items"][1]["cells_away"] is None
    assert obs["last_command"] == ["get", 1] and obs["last_result"] == {"ticks": got.ticks, "done": True, "reason": "ok"}
    assert session.do(("home",)).done and session.observe()["score"] == pytest.approx(3 / 25)
    assert session.do(("go", (8, 7))).done and session.observe()["robot"]["cell"] == [8, 7]
    assert session.do(("wait", 3)) == Result(3, True, "ok") and session.do(("drop",)) == Result(0, False, "refused")
    assert session.do(("pick",)) == Result(0, False, "refused")  # nothing in reach here


@pytest.mark.parametrize("command", [("get", 99), ("get", "1"), ("fly",), ("wait", 0), ("wait", 500), ("go", (0, 5)), ("go", 3), None, ("home", 4)])
def test_an_illegal_command_costs_one_tick_and_the_episode_goes_on(env, command):
    session = _session(env)
    result = session.do(command)
    assert result.ticks == 1 and not result.done and result.reason.startswith("illegal: ")
    assert session.skills.ticks == 1 and session.do(("wait", 1)).done
    assert session.observe()["last_result"]["done"] is True and len(session.history) == 2
