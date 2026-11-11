"""The referee: which brain acts this turn. Pure functions, no simulator, no network, no threads (RRARA_V1_PLAN.md file 5).

Paper, eq. (9): the action at t' is a_t if valid, a_new otherwise. Our additions, each an open assumption in ASSUMPTIONS.md:
  A30 (mode): "faithful" = background compute is free, a finished verdict always counts (the paper's accounting);
             "worldtime" = the verdict exists in the world only after ceil(wall_s x fps) frames, so it counts only if that
             is no more than the frames the checked action actually took.
  A31: the caller passes a verdict only if it is about the immediately preceding decision; older ones are never passed.
       A replacement must still be legal now, otherwise the reflex stands.
"""
import math


def same_action(a, b):
    return a[0] == b[0] and str(a[1]) == str(b[1])


def legal_actions(processed_input, target_categories):
    """What HAZARD's executor can carry out from this observation (challenge.py:337-411): walk to any explored target,
    pick up when standing at one and holding nothing, drop when holding, explore always."""
    acts = [("explore", None)]
    holding = processed_input["holding_objects"]
    if holding:
        acts.append(("drop", None))
    else:
        for obj in processed_input["explored_object_name_list"]:
            if obj["category"] in target_categories:
                acts.append(("walk_to", str(obj["id"])))
        if processed_input["nearest_object"]:
            acts.append(("pick_up", str(processed_input["nearest_object"][0]["id"])))
    return acts


def decide(proposed, verdict, legal, mode="worldtime", action_frames=0, fps=30):
    """proposed: the reflex's action for this turn. verdict: None, or the reflector's record about the previous action
    ({valid, replacement, wall_s}). legal: legal_actions(...) now. Returns (action, record)."""
    rec = {"path": "reflex", "override": False, "in_time": None, "frames_needed": None, "action_frames": action_frames}
    if verdict is None or "error" in verdict:
        rec["path"] = "reflex (no verdict)" if verdict is None else "reflex (reflector error)"
        return proposed, rec
    frames_needed = math.ceil(verdict["wall_s"] * fps)
    in_time = True if mode == "faithful" else frames_needed <= action_frames
    rec.update(in_time=in_time, frames_needed=frames_needed)
    if not in_time:
        rec["path"] = "reflex (verdict late)"
        return proposed, rec
    if verdict["valid"]:
        rec["path"] = "reflex (verdict valid)"
        return proposed, rec
    rep = verdict["replacement"]
    # The verdict record stores the replacement's parameter as a string (it is JSON-logged), so return the matching
    # legal action, which carries the native parameter: ("drop", None), never ("drop", "None"). 5a50c2d's loop does
    # int() on a drop parameter unless it is Python None (challenge.py:345), so the string crashed it (Della, Nov 10).
    match = None if rep is None else next((a for a in legal if same_action(rep, a)), None)
    if match is None:
        rec["path"] = "reflex (replacement not legal now)"
        return proposed, rec
    rec.update(path="override", override=True)
    return (match[0], match[1]), rec
