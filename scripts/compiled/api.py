"""A39 step (i): the observation a compiled program receives, the actions it may return, and the loader that runs such a
program in HAZARD's custom-agent slot. The contract is written out in scripts/compiled/API.md; this file is the code
behind every statement there. Tests: scripts/compiled/test_api.py.

A program is a Python module that defines

    class Policy:
        def __init__(self, task, target_values): ...   # task "fire" | "flood" | "wind"; {category: value}
        def decide(self, obs) -> (action_name, param)   # obs built by build_observation() below

and imports nothing outside ALLOWED_IMPORTS (no file, network or model calls while it plays; Liane's "no model calls
during play"). `ProgramAgent` wraps it as HAZARD's `agent_type == "custom"` object (policy/custom.py:11-19,
challenge.py:257-258): reset(goal_objects, objects_info) once per episode, choose_target(state, processed_input) once
per decision. An illegal or crashing decision is replaced by ("explore", None) and logged, so one compiler mistake
does not end a 10-minute episode.
"""
import ast
import importlib.util
import math
import os
import time

import numpy as np

from policy.astar import get_astar_path, get_astar_weight  # HAZARD's own planner (src/HAZARD on PYTHONPATH)

API_VERSION = "1"
CELL_M = 0.1  # challenge.py:48: grid_size=0.1, map 512 x 512
EPISODE_FRAMES = {"fire": 1500, "flood": 1500, "wind": 3000}  # challenge.py:53-59; the loop ends at :421
ACTIONS = ("walk_to", "pick_up", "drop", "explore")
OBS_KEYS = ("api_version", "task", "frame", "frames_left", "agent_cell", "facing_cell", "holding", "at_object",
            "objects", "last_action", "last_result", "map")
ALLOWED_IMPORTS = ("math", "collections", "heapq", "itertools", "functools", "typing", "dataclasses", "numpy", "compiled",
                   "compiled.api")


# ---------------------------------------------------------------- map helpers (the conventions of HAZARD's own agents)
def agent_cell(goal_map):
    """The robot's cell: goal_map == -2 (fireagent_controller.py:558). None when the robot is off the map edge (:557)."""
    pts = np.argwhere(goal_map == -2)
    return None if len(pts) == 0 else (int(pts[0][0]), int(pts[0][1]))


def facing_cell(goal_map):
    """The cell one step ahead of the robot in its facing direction: goal_map == -1 (fireagent_controller.py:559-565)."""
    pts = np.argwhere(goal_map == -1)
    return None if len(pts) == 0 else (int(pts[0][0]), int(pts[0][1]))


def object_cell(sem_map, obj_id):
    """Mean cell of an object's explored footprint, as the LLM agent computes it (llm.py:544-549). An id is meaningful
    only where explored == 1 (challenge.py:136). None when the object is not on the explored map."""
    pts = np.argwhere((sem_map["explored"] * sem_map["id"]) == int(obj_id))
    return None if len(pts) == 0 else (int(round(pts[:, 0].mean())), int(round(pts[:, 1].mean())))


class Paths:
    """Shortest-path lengths on the robot's own map, by HAZARD's A* (policy/astar.py, the functions the walk executor
    itself uses, env_actions.py:118-119). The weight is pooled by `coarse` x `coarse` cells (block max, so obstacles stay
    obstacles) because one full-map pyastar2d call costs 0.1-0.2 s (rrara/reflex.py notes); lengths are returned in full
    cells (0.1 m). The weight is computed once per observation and reused for every object."""

    def __init__(self, obs, coarse=4):
        self.obs = obs
        self.coarse = max(1, int(coarse))
        self._weight = None

    def _weight_map(self, dest):
        if self._weight is None:
            sem = self.obs["map"]
            w = get_astar_weight(sem_map={"explored": sem["explored"], "height": sem["height"]},
                                 origin=self.obs["agent_cell"], destination=dest)
            c = self.coarse
            if c > 1:
                h, wd = w.shape
                h2, w2 = h // c, wd // c
                w = w[:h2 * c, :w2 * c].reshape(h2, c, w2, c).max(axis=(1, 3)).astype(np.float32)
            self._weight = w
        return self._weight

    def cells(self, obj_id):
        """Path length in cells from the robot to the object's cell, or None when the object has no cell, the robot has
        none, or A* returns nothing. HAZARD's weights are finite everywhere (a 2.5 m wall costs about 12,000 per cell,
        astar.py:41-42), so in practice a path always exists and a wall shows up as a long or expensive path, not as
        None. A program that wants "reachable" must compare the length with the straight line itself."""
        origin = self.obs["agent_cell"]
        dest = None
        for o in self.obs["objects"]:
            if o["id"] == str(obj_id):
                dest = o["cell"]
                break
        if origin is None or dest is None:
            return None
        c = self.coarse
        o, d = (origin[0] // c, origin[1] // c), (dest[0] // c, dest[1] // c)
        if o == d:
            return 0.0
        path = get_astar_path(self._weight_map(dest), o, d)
        if path is None or len(path) == 0:
            return None
        pts = np.asarray(path, dtype=float)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) * c if len(pts) > 1 else 0.0


# ---------------------------------------------------------------- the observation
def _obj(entry, target_values, agent=None, sem_map=None):
    cat = entry["category"]
    is_target = cat in target_values
    rec = {"id": str(entry["id"]), "name": entry["name"], "category": cat, "is_target": is_target,
           "value": float(target_values[cat]) if is_target else None}
    if sem_map is not None:
        cell = object_cell(sem_map, int(entry["id"]))
        rec["cell"] = cell
        rec["straight_cells"] = (None if (cell is None or agent is None)
                                 else float(np.hypot(cell[0] - agent[0], cell[1] - agent[1])))
    return rec


def build_observation(state, processed_input, task, target_values, last_action=None):
    """Everything a program may act on, from HAZARD's `state` (fireagent_controller.py:535-566) and `processed_input`
    (challenge.py:134-163). Nothing here comes from the scene file or the simulator's ground truth: the object list is
    what the robot has segmented and mapped, values are the per-category values every shipped agent gets at reset
    (challenge.py:100-108), the map is the robot's own."""
    sem_map, goal_map = state["sem_map"], state["goal_map"]
    agent = agent_cell(goal_map)
    frame = int(processed_input["step"])  # env.controller.frame_count at decision time (challenge.py:160)
    objects = [_obj(o, target_values, agent, sem_map) for o in processed_input["explored_object_name_list"]]
    holding = [_obj(o, target_values) for o in processed_input["holding_objects"]]
    near = processed_input.get("nearest_object") or []
    at_object = _obj(near[0], target_values) if near else None  # set only after a walk_to arrived (challenge.py:357-360)
    last_result = None
    if last_action is not None:
        last_result = {"ok": bool(processed_input["action_result"]), "info": str(processed_input["action_info"])}
    return {
        "api_version": API_VERSION,
        "task": task,
        "frame": frame,
        "frames_left": max(0, EPISODE_FRAMES[task] - frame),
        "agent_cell": agent,
        "facing_cell": facing_cell(goal_map),
        "holding": holding,
        "at_object": at_object,
        "objects": objects,
        "last_action": None if last_action is None else (last_action[0], None if last_action[1] is None else str(last_action[1])),
        "last_result": last_result,
        "map": {"explored": sem_map["explored"], "height": sem_map["height"], "id": sem_map["id"]},
    }


# ---------------------------------------------------------------- actions
def legal_actions(obs):
    """The menu HAZARD's own agents choose from (rule_based.py:40-62, the LLM agent's get_available_plans): walk_to a
    seen target when not holding; pick_up the object just walked to when it is a target and nothing is held; drop when
    holding (fire/flood: into the bag, which is what scores); explore always."""
    menu = [("explore", None)]
    if obs["holding"] and obs["task"] in ("fire", "flood"):
        menu.append(("drop", None))
    if not obs["holding"]:
        at = obs["at_object"]
        if at is not None and at["is_target"]:
            menu.append(("pick_up", at["id"]))
        for o in obs["objects"]:
            if o["is_target"]:
                menu.append(("walk_to", o["id"]))
    return menu


def normalize_action(action):
    """(name, param) with the param as a string id or None, which is what the reflex returns and challenge.py int()s."""
    if not isinstance(action, (tuple, list)) or len(action) != 2:
        raise ValueError(f"an action is a (name, param) pair, got {action!r}")
    name, param = action
    if not isinstance(name, str):
        raise ValueError(f"action name must be a string, got {name!r}")
    if param is not None and not isinstance(param, (str, int, np.integer)):
        raise ValueError(f"action param must be an object id or None, got {param!r}")
    return name, None if param is None else str(int(param)) if not isinstance(param, str) else param


def check_action(action, obs):
    """(ok, reason). ok when the normalized action is on legal_actions(obs)."""
    try:
        name, param = normalize_action(action)
    except ValueError as e:
        return False, str(e)
    if name not in ACTIONS:
        return False, f"unknown action {name!r}; the actions are {ACTIONS}"
    menu = legal_actions(obs)
    if (name, param) in menu:
        return True, "ok"
    if name == "pick_up" and param is None and any(a[0] == "pick_up" for a in menu):
        return True, "ok"  # challenge.py:385-389 picks up the object just walked to
    return False, f"{(name, param)} is not on the menu {menu}"


# ---------------------------------------------------------------- loading a program
def audit_program(path):
    """Problems with a program's source that make it unfit to play: imports outside ALLOWED_IMPORTS (files, network,
    model clients, randomness), no `class Policy`, no `decide`. Empty list = fit."""
    problems = []
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except (OSError, SyntaxError) as e:
        return [f"cannot parse: {e}"]
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            if n not in ALLOWED_IMPORTS and n.split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import of {n!r} is not allowed (allowed: {', '.join(ALLOWED_IMPORTS)})")
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Policy"]
    if not classes:
        problems.append("no `class Policy` at module level")
    else:
        methods = {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)}
        for m in ("__init__", "decide"):
            if m not in methods:
                problems.append(f"Policy has no `{m}` method")
    return problems


def load_program(path):
    """Import a program module from a file path after audit_program() passes."""
    problems = audit_program(path)
    if problems:
        raise ValueError("program refused: " + "; ".join(problems))
    name = "compiled_program_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProgramAgent:
    """A compiled program in HAZARD's custom slot. Decision wall time is measured here as well as by the runner's
    TimedAgent (run_baseline.py), with frames = ceil(seconds x fps), RRARA's convention."""

    def __init__(self, task, program_path, fps=30):
        assert task in EPISODE_FRAMES
        self.task = task
        self.fps = fps
        self.agent_type = "custom"
        self.program_path = str(program_path)
        self.module = load_program(self.program_path)
        self.policy = None
        self.decision_log = []
        self._episode_start()

    def _episode_start(self):
        self.n = 0
        self.last_action = None

    def reset(self, goal_objects, objects_info):
        self.goal_objects = list(goal_objects)
        self.target_values = {c: float(objects_info.get(c, {}).get("value", 1)) for c in self.goal_objects}
        self.policy = self.module.Policy(task=self.task, target_values=dict(self.target_values))
        self._episode_start()

    def choose_target(self, state, processed_input):
        obs = build_observation(state, processed_input, self.task, self.target_values, self.last_action)
        t = time.perf_counter()
        error = ""
        try:
            action = self.policy.decide(obs)
        except Exception as e:  # a compiler mistake must not end the episode
            action, error = None, f"{type(e).__name__}: {e}"
        wall = time.perf_counter() - t
        if error:
            ok, reason = False, error
        else:
            ok, reason = check_action(action, obs)
        if ok:
            action = normalize_action(action)
        else:
            action = ("explore", None)
        self.n += 1
        self.decision_log.append({
            "decision": self.n, "frame": obs["frame"], "action": list(action), "legal": ok,
            "reason": None if ok else reason, "wall_s": round(wall, 6), "frames_at_fps": math.ceil(wall * self.fps),
            "objects_seen": len(obs["objects"]), "targets_seen": sum(o["is_target"] for o in obs["objects"]),
            "holding": len(obs["holding"]), "last_result": obs["last_result"],
        })
        self.last_action = action
        return action

    def summary(self):
        walls = [r["wall_s"] for r in self.decision_log]
        return {
            "program": self.program_path, "api_version": API_VERSION, "decisions": len(walls),
            "illegal_or_crashed": sum(1 for r in self.decision_log if not r["legal"]),
            "wall_s_mean": round(sum(walls) / len(walls), 4) if walls else None,
            "wall_s_max": round(max(walls), 4) if walls else None,
            "frames_at_fps_total": sum(r["frames_at_fps"] for r in self.decision_log),
            "actions": {a: sum(1 for r in self.decision_log if r["action"][0] == a) for a in ACTIONS},
        }
