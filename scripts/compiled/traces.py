"""A39 step (ii): turn a recorded HAZARD episode of an LLM-class agent (Haiku, Jev, the person) into traces a compiler can
read, in the program API's vocabulary (scripts/compiled/API.md). Sources per episode folder <ep>/ (one kitchen):
  <ep>/0/input<frame>.json      processed_input at every decision (challenge.py:309-311): objects seen, holding, the object
                                just walked to, the frame, and how the PREVIOUS action went (action_result / action_info)
  <ep>/0/<frame>_info.json      only for decisions that asked the model (llm.py:585-587): the prompt text, the model's
                                chain of thought, its letter and the plan text it chose
  <ep>/actions.txt              every action in order, automatic pick-ups and drops included (challenge.py:325-331)
  <ep>/eval_result.json         the kitchen's score
The prompt text carries what the model was told (per-category values, distances in cells printed as metres, llm.py:
400-452); the trace keeps those as the model saw them and adds the honest readings (metres, the scorer's value by name)
so the compiler can see both. No simulator, no model call. Tests: scripts/compiled/test_traces.py.
"""
import glob
import json
import os
import re

HIGH_VALUE, LOW_VALUE = 5, 1  # challenge.py:89-90
CELL_M = 0.1
_PLAN_WALK = re.compile(r"go (?:pick up object|to object|put object into) <(?P<cat>[^>]+)> \((?P<id>\d+)\)")
_PROMPT_OBJ = re.compile(r"name: (?P<cat>.+?), id: (?P<id>\d+), value: (?P<value>[\d.]+), distance: (?P<dist>[\d.]+) m")


def parse_plan(plan):
    """Plan text (llm.py:487-520) -> (action, param), as HAZARD's run() maps it (the letter picks the plan)."""
    if plan is None:
        return None
    m = _PLAN_WALK.search(plan)
    if m:
        return ("walk_to", m.group("id"))
    if plan.startswith("put the holding object"):
        return ("drop", None)
    if plan.startswith("look around"):
        return ("explore", None)
    return None


def parse_prompt_objects(prompt):
    """{id: {category, value_shown, distance_cells}} from the 'Target objects ... seen' lines of a shipped prompt."""
    out = {}
    for m in _PROMPT_OBJ.finditer(prompt or ""):
        out[m.group("id")] = {"category": m.group("cat").strip(), "value_shown": float(m.group("value")),
                              "distance_cells": float(m.group("dist"))}
    return out


def parse_actions_txt(path):
    """[(action, param)] in order from actions.txt lines like "step 0 action ('walk_to', '56')"."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"action \('([a-z_.]+)', (?:'([^']*)'|(None)|(\d+))\)", line)
        if m:
            name = m.group(1)
            param = m.group(2) if m.group(2) is not None else (None if m.group(3) else m.group(4))
            out.append((name, param))
    return out


def load_episode(folder, name_values=None):
    """Traces of one recorded episode: {"kitchen", "score", "decisions": [record, ...]} where a record has
    n, frame, frames_used (to the next decision), holding, at_object, objects (id, name, category, value_shown,
    value_scorer, distance_m as the prompt implied), prev_result, menu, chosen (letter, plan, action), automatic (no
    model call), reasoning (the model's chain of thought, trimmed), outcome (the next decision's prev_result)."""
    sub = os.path.join(folder, "0")
    inputs = sorted(glob.glob(os.path.join(sub, "input*.json")), key=lambda p: int(os.path.basename(p)[5:-5]))
    infos = {int(os.path.basename(p)[:4]): p for p in glob.glob(os.path.join(sub, "*_info.json"))}
    actions = parse_actions_txt(os.path.join(folder, "actions.txt"))
    score = None
    ev = os.path.join(folder, "eval_result.json")
    if os.path.exists(ev):
        e = json.load(open(ev))
        score = round(e["avg_score"] / e["avg_max_score"], 3) if e.get("avg_max_score") else None
    recs = []
    for i, path in enumerate(inputs):
        pi = json.load(open(path, encoding="utf-8"))
        frame = int(pi["step"])
        info = json.load(open(infos[frame], encoding="utf-8")) if frame in infos else None
        shown = parse_prompt_objects(info["prompts"]) if info else {}
        objects = []
        for o in pi["explored_object_name_list"]:
            s = shown.get(str(o["id"]))
            objects.append({"id": str(o["id"]), "name": o["name"], "category": o["category"],
                            "value_shown": s["value_shown"] if s else None,
                            "value_scorer": (HIGH_VALUE if name_values.get(o["name"], 0) == 1 else LOW_VALUE) if name_values is not None else None,
                            "distance_m": round(s["distance_cells"] * CELL_M, 2) if s else None,
                            "is_target": bool(s)})
        chosen = None
        if info:
            chosen = {"letter": (info.get("outputs") or [None])[0], "plan": info.get("plan"), "action": parse_plan(info.get("plan"))}
        action = chosen["action"] if chosen and chosen["action"] else (actions[i] if i < len(actions) else None)
        cot = info.get("cot_outputs") if info else None
        reasoning = (cot[0] if isinstance(cot, list) and cot else None)
        recs.append({"n": i + 1, "frame": frame, "frames_used": None,
                     "holding": [{"id": str(h["id"]), "category": h["category"], "name": h["name"]} for h in pi["holding_objects"]],
                     "at_object": ({"id": str(pi["nearest_object"][0]["id"]), "category": pi["nearest_object"][0]["category"]}
                                   if pi.get("nearest_object") else None),
                     "objects": objects,
                     "prev_result": None if i == 0 else {"ok": bool(pi["action_result"]), "info": str(pi["action_info"])},
                     "menu": [l.strip() for l in (info["prompts"].split("Available actions:")[-1].split("Answer")[0].strip().splitlines()) if l.strip()] if info else [],
                     "chosen": chosen, "action": list(action) if action else None, "automatic": info is None,
                     "reasoning": (reasoning[:600] + "...") if reasoning and len(reasoning) > 600 else reasoning,
                     "outcome": None})
    for a, b in zip(recs[:-1], recs[1:]):
        a["frames_used"] = b["frame"] - a["frame"]
        a["outcome"] = b["prev_result"]
    return {"kitchen": os.path.basename(os.path.normpath(folder)), "score": score, "decisions": recs}


def load_sweep(sweep_dir, kitchens=None, name_values=None):
    """Episodes of a sweep folder (one subfolder per kitchen), optionally only the named kitchens."""
    eps = []
    for k in sorted(os.listdir(sweep_dir)):
        if kitchens and k not in kitchens:
            continue
        if os.path.isdir(os.path.join(sweep_dir, k, "0")):
            eps.append(load_episode(os.path.join(sweep_dir, k), name_values=name_values))
    return eps


def _obj_line(o):
    s = f"{o['category']} (id {o['id']})"
    if o["value_shown"] is not None:
        s += f" shown worth {o['value_shown']:g}"
    if o["value_scorer"] is not None:
        s += f", scorer pays {o['value_scorer']}"
    if o["distance_m"] is not None:
        s += f", {o['distance_m']} m straight line"
    return s


def render_episode(ep, max_decisions=None, with_reasoning=False):
    """Compact text of one episode for a compiler prompt: one block per decision in the API's vocabulary."""
    lines = [f"### Episode {ep['kitchen']}: final value rate {ep['score'] if ep['score'] is not None else 'unknown'}"]
    for r in ep["decisions"][: max_decisions or None]:
        head = f"- decision {r['n']}, frame {r['frame']}: holding {', '.join(h['category'] + ' (id ' + h['id'] + ')' for h in r['holding']) or 'nothing'}"
        if r["at_object"]:
            head += f"; standing at {r['at_object']['category']} (id {r['at_object']['id']})"
        if r["prev_result"] is not None:
            head += f"; previous action: {'ok' if r['prev_result']['ok'] else 'failed'}" + (f" ({r['prev_result']['info']})" if r['prev_result']['info'] else "")
        lines.append(head)
        targets = [o for o in r["objects"] if o["is_target"]]
        if targets:
            lines.append("  targets seen: " + "; ".join(_obj_line(o) for o in targets))
        if r["action"]:
            how = "automatic" if r["automatic"] else f"letter {r['chosen']['letter']}"
            lines.append(f"  action: {tuple(r['action'])} ({how})")
        if with_reasoning and r.get("reasoning"):
            lines.append(f"  reasoning: {r['reasoning']}")
        if r["outcome"] is not None:
            lines.append(f"  outcome: {'ok' if r['outcome']['ok'] else 'failed'}" + (f" ({r['outcome']['info']})" if r['outcome']['info'] else "")
                         + (f", {r['frames_used']} frames" if r["frames_used"] is not None else ""))
    return "\n".join(lines)


def episode_stats(ep):
    walks = [r for r in ep["decisions"] if r["action"] and r["action"][0] == "walk_to"]
    capped = [r for r in walks if r["outcome"] and r["outcome"]["info"] == "max steps reached"]
    arrived = [r for r in walks if r["outcome"] and r["outcome"]["ok"]]
    targets = {}
    for r in walks:
        targets[r["action"][1]] = targets.get(r["action"][1], 0) + 1
    return {"kitchen": ep["kitchen"], "score": ep["score"], "decisions": len(ep["decisions"]), "walks": len(walks),
            "walks_capped": len(capped), "walks_arrived": len(arrived),
            "most_walked": max(targets.items(), key=lambda kv: kv[1]) if targets else None,
            "explores": sum(1 for r in ep["decisions"] if r["action"] and r["action"][0] == "explore")}
