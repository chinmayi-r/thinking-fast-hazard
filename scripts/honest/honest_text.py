"""Honest text v2 (A32, A41): what HAZARD's LLM-class agents (Haiku, Jev, the human page) are told, rewritten from the
facts the robot actually has. CR's Nov 17 spec: every option with its predicted effect and cost, every object with
range and direction relative to me, every result in numbers with a reason, the full history, the true worth of each
item, and the controls the body already supports (one step toward an item, a 90-degree turn, a 1 m step).

It is a mixin over policy.llm.LLM (`apply_honest_text(agent, template)`): it replaces progress2text (the $STATE$ slot),
get_available_plans (the menu), objects_list2text (the target list) and pre-fills $ACTION_HISTORY$ with the complete
log, so LLM.run (llm.py:589-651) and the subclasses (HumanAgent, JevAgent) are untouched. The facts come from
compiled.api.build_observation, the same observation a compiled program gets. HAZARD's body is untouched: the three
extra controls are actions its dispatcher already executes (challenge.py:361-383).

Numbers: distances in metres on the robot's own map (0.1 m cells, challenge.py:48), directions in degrees relative to
the heading (positive = left, A41), frames at the simulator's rate. Walk time per metre is learned from this
episode's own walks (unknown until the first one), since it is not published.
"""
import csv
import json
import math
import os
import time

import numpy as np

import policy  # HAZARD package (src/HAZARD on PYTHONPATH), to find data/meta_data/value.json
from compiled.api import CELL_M, EPISODE_FRAMES, Paths, build_observation

HIGH_VALUE, LOW_VALUE = 5, 1  # challenge.py:89-90
EXPLORE_FRAMES = 49  # measured, every explore (NOTES Oct 28)
WALK_CAP_FRAMES = 100  # challenge.py:346
ARRIVE_M = 0.5  # challenge.py:346 (d35c71e; 1.25 on 5a50c2d, A29)
TURN_DEG = 90.0
STEP_M = 1.0
MENU_CAP = 12
MOVED_EPS_M = 0.3  # below this a walk "did not get going"
PLAN_NEAREST = 8  # targets planned per decision (A* calls), the nearest by straight line


def name_value_table(path=None):
    """The scorer's table (challenge.py:190-203, data/meta_data/value.json): object NAME -> 5 or 1; names not in the
    file are worth 1. The name is in the robot's own object list (challenge.py:144-148), so this is not privileged."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(policy.__file__)), "..", "data", "meta_data", "value.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {name: (HIGH_VALUE if v == 1 else LOW_VALUE) for name, v in raw.items()}


def direction_text(bearing):
    """bearing in degrees, positive = left (A41)."""
    if bearing is None:
        return "direction unknown"
    a = abs(bearing)
    if a <= 15:
        return "straight ahead"
    if a >= 165:
        return "behind me"
    return f"{int(round(a))} deg to my {'left' if bearing > 0 else 'right'}"


def _dist_cells(a, b):
    return None if a is None or b is None else float(np.hypot(a[0] - b[0], a[1] - b[1]))


def unseen_cells_along(path, explored):
    """Number of 1-cell steps along the polyline `path` (corner points, full-map cells) that land on unexplored cells."""
    if path is None or len(path) < 2:
        return 0
    n = 0
    H, W = explored.shape
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        steps = max(1, int(round(float(np.hypot(r1 - r0, c1 - c0)))))
        for k in range(1, steps + 1):
            r = int(round(r0 + (r1 - r0) * k / steps))
            c = int(round(c0 + (c1 - c0) * k / steps))
            if explored[min(max(r, 0), H - 1), min(max(c, 0), W - 1)] == 0:
                n += 1
    return n


def load_template(path, task):
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["type"] == task:
                return row["prompt"]
    raise KeyError(f"no prompt for task {task!r} in {path}")


class Honest:
    """Per-episode bookkeeping and text. begin_decision(state, processed_input) before the prompt is built,
    end_decision(action) after the agent has chosen; state_text() / history_text() / menu() in between."""

    def __init__(self, task, target_values, name_values=None, coarse=4, single_steps=True, turns=True, fps=0):
        assert task in EPISODE_FRAMES
        self.task = task
        self.target_values = dict(target_values)
        self.name_values = name_values  # None: category values, as HAZARD's prompt; a table: the scorer's values
        self.coarse = coarse
        self.single_steps = single_steps
        self.turns = turns
        self.fps = fps  # 0 when the clock is off: thinking costs no frames
        self.reset()

    def reset(self):
        self.n = 0
        self.log = []  # finished decisions, one dict each (see _finish_prev)
        self.prev = None  # the decision whose result has not arrived yet
        self.speed = []  # (moved_m, frames) from walks, for the frames-per-metre estimate
        self.saved = []  # items put in the bag: (id, name, value)
        self.obs = None
        self.state = None
        self.facts = {}  # target id -> path facts this decision
        self.t_start = None

    # ---- values
    def value_of(self, obj):
        if self.name_values is not None:
            return float(self.name_values.get(obj["name"], LOW_VALUE))
        return float(obj["value"]) if obj.get("value") is not None else float(LOW_VALUE)

    def frames_per_m(self):
        moved = sum(m for m, _ in self.speed)
        return None if moved < MOVED_EPS_M else sum(f for _, f in self.speed) / moved

    # ---- the decision boundary
    def begin_decision(self, state, processed_input):
        self.n += 1
        last = None
        if self.prev is not None:
            name, param = self.prev["action"]
            last = (name, param if isinstance(param, (str, int)) or param is None else None)
        obs = build_observation(state, processed_input, self.task, self.target_values, last)
        self.obs, self.state = obs, state
        if self.prev is not None:
            self._finish_prev(obs)
        self._compute_facts(obs)
        self.t_start = time.perf_counter()
        return obs

    def _target_range(self, obs, oid):
        for o in obs["objects"]:
            if o["id"] == oid:
                return o["range_m"]
        return None

    def _finish_prev(self, obs):
        p = self.prev
        res = obs["last_result"] or {"ok": False, "info": ""}
        frames_total = obs["frame"] - p["frame"]
        think = p["think_frames"]
        frames = max(0, frames_total - think)
        moved_c = _dist_cells(obs["agent_cell"], p["agent_cell"])
        moved = None if moved_c is None else round(moved_c * CELL_M, 2)
        name, param = p["action"]
        rec = {"n": p["n"], "frame": p["frame"], "action": [name, param if not isinstance(param, dict) else param],
               "ok": bool(res["ok"]), "info": str(res["info"]), "frames": frames, "think_frames": think,
               "moved_m": moved, "target": p["target"], "was_m": p["was_m"], "remaining_m": None, "new_objects": None}
        if name in ("walk_to", "walk_to_single") and p["target"] is not None:
            rec["remaining_m"] = self._target_range(obs, p["target"])
            if moved is not None and moved >= MOVED_EPS_M and frames > 0:
                self.speed.append((moved, frames))
        elif name == "explore":
            rec["new_objects"] = len(obs["objects"]) - p["n_objects"]
        elif name == "drop" and res["ok"] and p["holding"]:
            h = p["holding"][0]
            self.saved.append((h["id"], h["category"], self.value_of(h)))
        rec["text"] = self.result_text(rec, p)
        self.log.append(rec)
        self.prev = None

    def end_decision(self, action):
        think_wall = time.perf_counter() - self.t_start if self.t_start else 0.0
        obs = self.obs
        name, param = action[0], action[1]
        target = None
        if name in ("walk_to", "walk_to_single", "pick_up") and param is not None and not isinstance(param, dict):
            target = str(param)
        self.prev = {"n": self.n, "frame": obs["frame"], "agent_cell": obs["agent_cell"], "action": (name, param),
                     "target": target, "was_m": self._target_range(obs, target) if target else None,
                     "n_objects": len(obs["objects"]), "holding": obs["holding"],
                     "think_wall_s": round(think_wall, 4),
                     "think_frames": math.ceil(think_wall * self.fps) if self.fps else 0,
                     "planned": dict(self.facts.get(target, {})) if target else None,
                     "menu": [list(a) if not isinstance(a[1], dict) else [a[0], a[1]] for a in self._last_menu_actions]
                     if hasattr(self, "_last_menu_actions") else None}

    # ---- facts about the map
    def _compute_facts(self, obs):
        self.facts = {}
        targets = [o for o in obs["objects"] if o["is_target"] and o["cell"] is not None]
        targets.sort(key=lambda o: (o["straight_cells"], o["id"]))
        fpm = self.frames_per_m()
        tries = {}
        for r in self.log:
            if r["action"][0] == "walk_to" and r["target"] is not None:
                tries.setdefault(r["target"], []).append(r)
        paths = Paths(obs, coarse=self.coarse) if obs["agent_cell"] is not None else None
        explored = obs["map"]["explored"]
        for o in targets[:PLAN_NEAREST]:
            path = paths.path(o["id"]) if paths is not None else None
            fact = {"path_m": None, "unseen_m": None, "pred_frames": None, "need_frames": None,
                    "tries": len(tries.get(o["id"], [])),
                    "last_short_m": (tries[o["id"]][-1]["remaining_m"] if tries.get(o["id"]) else None)}
            if path is not None:
                pts = np.asarray(path, dtype=float)
                length_c = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
                fact["path_m"] = round(length_c * CELL_M, 2)
                # HAZARD's A* returns the path's corner points only (astar.py:16-34, path_cleanup), so sample every segment cell by cell
                fact["unseen_m"] = round(unseen_cells_along(path, explored) * CELL_M, 2)
                if fpm is not None:
                    fact["need_frames"] = int(math.ceil(fact["path_m"] * fpm))  # at my speed so far, uncapped
                    fact["pred_frames"] = int(min(WALK_CAP_FRAMES, fact["need_frames"]))
            self.facts[o["id"]] = fact
        for o in targets[PLAN_NEAREST:]:
            self.facts[o["id"]] = {"path_m": None, "unseen_m": None, "pred_frames": None, "need_frames": None,
                                   "tries": len(tries.get(o["id"], [])),
                                   "last_short_m": (tries[o["id"]][-1]["remaining_m"] if tries.get(o["id"]) else None)}

    def fire_facts(self):
        """Hottest mapped spot, from the robot's own temperature layer (sem_map['other'][0] = mapped log temperature,
        fireagent_controller.py:528-540); None when the layer is absent or nothing is mapped."""
        st = self.state
        other = None if st is None else st["sem_map"].get("other")
        if other is None or self.obs["agent_cell"] is None:
            return None
        layer = np.asarray(other[0] if other.ndim == 3 else other, dtype=float)
        mask = (self.obs["map"]["explored"] == 1) & np.isfinite(layer)
        if not mask.any():
            return None
        idx = np.argmax(np.where(mask, layer, -np.inf))
        r, c = np.unravel_index(idx, layer.shape)
        temp_c = float(np.exp(layer[r, c]))
        if temp_c < 60.0:
            return None
        from compiled.api import bearing_deg
        agent = self.obs["agent_cell"]
        return {"temp_c": round(temp_c), "range_m": round(_dist_cells(agent, (r, c)) * CELL_M, 1),
                "bearing_deg": bearing_deg(agent, self.obs["heading_deg"], (int(r), int(c)))}

    def unseen_facts(self, radius_m=3.0):
        """Nearest unexplored cell and the mapped fraction within radius_m of the robot."""
        obs = self.obs
        agent = obs["agent_cell"]
        if agent is None:
            return None
        ex = obs["map"]["explored"]
        R = int(radius_m / CELL_M)
        r0, r1 = max(0, agent[0] - R), min(ex.shape[0], agent[0] + R + 1)
        c0, c1 = max(0, agent[1] - R), min(ex.shape[1], agent[1] + R + 1)
        win = ex[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        d = np.hypot(rr - agent[0], cc - agent[1])
        inside = d <= R
        frac = float(win[inside].mean()) if inside.any() else 0.0
        unseen = inside & (win == 0)
        if not unseen.any():
            return {"mapped_frac": round(frac, 2), "nearest_m": None, "bearing_deg": None}
        k = np.argmin(np.where(unseen, d, np.inf))
        r, c = np.unravel_index(k, d.shape)
        from compiled.api import bearing_deg
        return {"mapped_frac": round(frac, 2), "nearest_m": round(float(d[r, c]) * CELL_M, 1),
                "bearing_deg": bearing_deg(agent, obs["heading_deg"], (int(rr[r, c]), int(cc[r, c])))}

    # ---- text
    def item_label(self, o):
        return f"{o['category']} \"{o['name']}\" (id {o['id']})"

    def walk_cost_text(self, fact):
        if fact is None or fact["path_m"] is None:
            return "no path on my map"
        s = f"path {fact['path_m']} m"
        if fact["unseen_m"]:
            s += f" ({fact['unseen_m']} m over unseen floor)"
        need = fact.get("need_frames")
        if need is None:
            s += f", walk time unknown yet (cap {WALK_CAP_FRAMES} frames)"
        elif need <= WALK_CAP_FRAMES:
            s += f", about {need} frames (cap {WALK_CAP_FRAMES})"
        else:
            walks = math.ceil(need / WALK_CAP_FRAMES)
            s += (f", about {need} frames at my speed so far: more than one walk's cap of {WALK_CAP_FRAMES}, "
                  f"so one walk will stop short and it takes about {walks} walks")
        return s

    def state_text(self, temps=None):
        obs = self.obs
        total = EPISODE_FRAMES[self.task]
        lines = [f"Frame {obs['frame']} of {total}, {obs['frames_left']} frames left."]
        if obs["holding"]:
            h = obs["holding"][0]
            lines.append(f"Holding: {self.item_label(h)}, worth {self.value_of(h):g}.")
        else:
            lines.append("Holding: nothing.")
        at = obs["at_object"]
        if at is not None:
            lines.append(f"Standing at: {self.item_label(at)} (a pick-up will work now).")
        else:
            lines.append("Standing at: no item (I must walk to an item before I can pick it up).")
        targets = [o for o in obs["objects"] if o["is_target"]]
        targets.sort(key=lambda o: (self.facts.get(o["id"], {}).get("path_m") or 1e9, o["range_m"] or 1e9, o["id"]))
        if targets:
            lines.append("Target items I have seen, nearest first:")
            for o in targets:
                f = self.facts.get(o["id"])
                s = f"- {self.item_label(o)}: worth {self.value_of(o):g}"
                if o["range_m"] is not None:
                    s += f"; {o['range_m']} m away, {direction_text(o['bearing_deg'])}"
                s += f"; {self.walk_cost_text(f)}"
                if temps and o["id"] in temps and temps[o["id"]]:
                    s += f"; temperature {np.exp(temps[o['id']][-1]):.0f} C"
                if f and f["tries"]:
                    s += f"; walks tried {f['tries']}"
                    if f["last_short_m"] is not None:
                        s += f", last ended {f['last_short_m']} m short"
                lines.append(s + ".")
        else:
            lines.append("Target items I have seen: none yet.")
        fire = self.fire_facts()
        if self.task == "fire":
            if fire is None:
                lines.append("Fire: nothing hot on my map yet.")
            else:
                lines.append(f"Fire: hottest spot on my map {fire['temp_c']} C, {fire['range_m']} m away, {direction_text(fire['bearing_deg'])}.")
        un = self.unseen_facts()
        if un is not None:
            if un["nearest_m"] is None:
                lines.append(f"Unseen floor: none within 3 m; {int(un['mapped_frac'] * 100)} % of the floor within 3 m is mapped.")
            else:
                lines.append(f"Unseen floor: nearest {un['nearest_m']} m away, {direction_text(un['bearing_deg'])}; "
                             f"{int(un['mapped_frac'] * 100)} % of the floor within 3 m is mapped.")
        if self.saved:
            tot = sum(v for _, _, v in self.saved)
            lines.append("Saved so far: " + ", ".join(f"{c} (id {i}) worth {v:g}" for i, c, v in self.saved) + f" (total {tot:g}).")
        else:
            lines.append("Saved so far: nothing.")
        return "\n".join(lines) + "\n"

    def result_text(self, rec, p):
        name, param = p["action"]
        fr = f"{rec['frames']} frames"
        if rec["think_frames"]:
            fr += f" plus {rec['think_frames']} frames of thinking"
        tgt = self._label_for(p["target"]) if p["target"] else None
        if name in ("walk_to", "walk_to_single"):
            verb = "walked toward" if name == "walk_to" else "took one step toward"
            moved = rec["moved_m"]
            was, rem = rec["was_m"], rec["remaining_m"]
            s = f"{verb} {tgt}: "
            s += f"moved {moved} m in {fr}" if moved is not None else f"{fr}"
            if rec["ok"] and rec["info"] == "success":
                s += ", arrived (within 0.5 m)" if name == "walk_to" else ", arrived"
            elif rec["ok"]:
                s += f"; now {rem} m from it" if rem is not None else ""
            else:
                if rem is not None and was is not None:
                    s += f"; ended {rem} m from it (was {was} m)"
                elif rem is not None:
                    s += f"; ended {rem} m from it"
                s += ": " + self.failure_reason(rec, p)
            return s
        if name == "explore":
            k = rec["new_objects"]
            s = f"looked around ({fr})"
            s += f": {k} new object{'s' if k != 1 else ''} on my map" if k is not None else ""
            return s if rec["ok"] else s + f"; failed: {rec['info']}"
        if name == "pick_up":
            return f"picked up {tgt} ({fr})" if rec["ok"] else f"tried to pick up {tgt} ({fr}): failed, {rec['info']}"
        if name == "drop":
            if rec["ok"]:
                h = p["holding"][0] if p["holding"] else None
                what = self.item_label(h) if h else "the item"
                v = self.value_of(h) if h else 0
                return f"put {what} in my bag ({fr}): saved, worth {v:g}"
            return f"tried to put the item in my bag ({fr}): failed, {rec['info']}"
        if name == "low_level.turn_by":
            ang = float(param.get("angle", 0)) if isinstance(param, dict) else 0.0
            return f"turned {'right' if ang > 0 else 'left'} {abs(ang):g} deg ({fr})"
        if name == "low_level.move_by":
            return f"stepped forward: moved {rec['moved_m']} m ({fr})" + ("" if rec["ok"] else f"; {rec['info']}")
        return f"{name} ({fr}): {'ok' if rec['ok'] else rec['info']}"

    def failure_reason(self, rec, p):
        info, moved, was, rem = rec["info"], rec["moved_m"], rec["was_m"], rec["remaining_m"]
        planned = p.get("planned") or {}
        if info == "max steps reached":
            if moved is not None and moved < MOVED_EPS_M:
                why = "did not get going (blocked right where I stood)"
            elif rem is not None and was is not None and rem >= was - 0.2:
                why = "walked but got no closer (the path led around or away)"
            else:
                why = f"ran out of the {WALK_CAP_FRAMES}-frame walk budget"
            if planned.get("unseen_m"):
                why += f"; the planned path crossed {planned['unseen_m']} m of unseen floor"
            return why
        if info == "target not in vision or memory":
            return "the item is no longer on my map"
        if info.startswith("don't know"):
            return "no path on my map"
        return info or "unknown"

    def _label_for(self, oid):
        for o in self.obs["objects"]:
            if o["id"] == oid:
                return self.item_label(o)
        for r in reversed(self.log):
            if r.get("target") == oid and r.get("target_label"):
                return r["target_label"]
        return f"item id {oid}"

    def history_text(self):
        if not self.log:
            return "(nothing yet)"
        return "\n".join(f"{r['n']}. frame {r['frame']}: {r['text']}" for r in self.log)

    def targets_text(self):
        cats = sorted(self.target_values)
        if self.name_values is None:
            return "\n".join(f"name: {c}, value: {self.target_values[c]:g}" for c in cats)
        return "\n".join(f"{c}: each item is worth {HIGH_VALUE} or {LOW_VALUE}; the worth of every item I have seen is listed below"
                         for c in cats)

    def menu(self):
        obs = self.obs
        plans, actions = [], []
        if obs["holding"] and self.task in ("fire", "flood"):
            h = obs["holding"][0]
            plans.append(f"put {self.item_label(h)} in my bag: saves it (worth {self.value_of(h):g}), a few dozen frames")
            actions.append(("drop", None))
        elif obs["holding"] and self.task == "wind":
            for o in obs["objects"]:
                if o["category"] == "shopping cart":
                    plans.append(f"go put the item into shopping cart (id {o['id']})")
                    actions.append(("walk_to", o["id"]))
        else:
            targets = [o for o in obs["objects"] if o["is_target"]]
            targets.sort(key=lambda o: (self.facts.get(o["id"], {}).get("path_m") or 1e9, o["range_m"] or 1e9, o["id"]))
            room = MENU_CAP - (1 + (1 if self.single_steps else 0) + (3 if self.turns else 0))
            for o in targets[:room]:
                f = self.facts.get(o["id"])
                plans.append(f"walk to {self.item_label(o)}: {self.walk_cost_text(f)}; arriving lets me pick it up")
                actions.append(("walk_to", o["id"]))
            if self.single_steps and targets:
                o = targets[0]
                plans.append(f"one step toward {self.item_label(o)}: a few frames; does not count as arriving")
                actions.append(("walk_to_single", o["id"]))
        plans.append(f"look around: turn a full circle, {EXPLORE_FRAMES} frames; adds what I see to my map")
        actions.append(("explore", None))
        if self.turns:
            plans.append(f"turn left {TURN_DEG:g} deg: a few frames")
            actions.append(("low_level.turn_by", {"angle": -TURN_DEG}))
            plans.append(f"turn right {TURN_DEG:g} deg: a few frames")
            actions.append(("low_level.turn_by", {"angle": TURN_DEG}))
            plans.append(f"step forward {STEP_M:g} m: a few frames, straight ahead only")
            actions.append(("low_level.move_by", {"distance": STEP_M}))
        plans, actions = plans[:MENU_CAP], actions[:MENU_CAP]
        self._last_menu_actions = actions
        text = "".join(f"{chr(ord('A') + i)}. {p}\n" for i, p in enumerate(plans))
        return text, len(plans), plans, actions

    def summary(self):
        walks = [r for r in self.log if r["action"][0] == "walk_to"]
        return {"decisions": self.n, "logged": len(self.log), "name_values": self.name_values is not None,
                "single_steps": self.single_steps, "turns": self.turns, "fps": self.fps,
                "frames_per_m": None if self.frames_per_m() is None else round(self.frames_per_m(), 1),
                "walks": len(walks), "walks_arrived": sum(1 for r in walks if r["ok"] and r["info"] == "success"),
                "walks_capped": sum(1 for r in walks if r["info"] == "max steps reached"),
                "saved": [list(s) for s in self.saved], "saved_value": sum(v for _, _, v in self.saved),
                "actions": {}}


class HonestTextMixin:
    """Put in front of an LLM subclass by apply_honest_text(). Overrides only text builders and the decision boundary."""
    _honest_opts = {}
    _honest_template = None

    def reset(self, goal_objects, objects_info):
        super().reset(goal_objects, objects_info)
        self.honest = Honest(self.task, {c: float(objects_info[c]["value"]) for c in goal_objects}, **self._honest_opts)

    def choose_target(self, state, processed_input):
        self.honest.begin_decision(state, processed_input)
        self.prompt_template = self._honest_template.replace("$ACTION_HISTORY$", self.honest.history_text() + "\n")
        action = super().choose_target(state, processed_input)
        self.honest.end_decision(action)
        return action

    def progress2text(self):
        return self.honest.state_text(temps=getattr(self, "env_change_record", None))

    def get_available_plans(self):
        return self.honest.menu()

    def objects_list2text(self):
        return self.honest.targets_text()


def apply_honest_text(agent, template_path, name_values="scorer", **opts):
    """Turn an LLM-class agent (LLM, HumanAgent, JevAgent, VLMAgent) into its honest-text version in place.
    name_values: "scorer" (the scorer's per-name table), None (HAZARD's per-category values), or a dict."""
    if name_values == "scorer":
        name_values = name_value_table()
    cls = type(agent)
    honest_cls = type("Honest" + cls.__name__, (HonestTextMixin, cls), {})
    agent.__class__ = honest_cls
    agent._honest_opts = dict(opts, name_values=name_values)
    agent._honest_template = load_template(template_path, agent.task)
    agent.prompt_template = agent._honest_template
    if getattr(agent, "objects_info", None) and getattr(agent, "target_objects", None):  # reset already happened
        agent.honest = Honest(agent.task, {c: float(agent.objects_info[c]["value"]) for c in agent.target_objects},
                              **agent._honest_opts)
    return agent
