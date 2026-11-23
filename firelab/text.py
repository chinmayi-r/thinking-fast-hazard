"""The room as text (chunk 7b): what a language model is shown. Built only from the observation dictionary.

The rules it follows are CR's for the honest text (A32, A41): every item with its true worth, what it is losing
and how far it is; every command with what it will cost; the last results in numbers with their reasons, the
whole history; and a small map. Nothing is in the text that is not in firelab.interface.Session.observe().
"""

from __future__ import annotations

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
INTRO = ("You control a robot in a room where a fire is spreading. Items lie on the floor. Carry them to the safe strip "
         "and bank them there: only banked items count, each at the value it has when you pick it up.")


def say(command, obs: dict) -> str:
    """One command in words, with its predicted cost in ticks where the observation gives one."""
    name, args = command[0], list(command[1:])
    cross = bool(args.pop()) if args and isinstance(args[-1], bool) else False
    fire = ", walking through fire" if cross else ""
    if name == "get":
        item = obs["items"][args[0]]
        cost = f"about {item['ticks_to_get']} ticks" if item["ticks_to_get"] is not None else "no fire-free route, time unknown"
        return f"get item {args[0]}{fire} (worth {item['value']:.1f} now; {cost})"
    if name == "home":
        cost = f"about {obs['home']['ticks']} ticks" if obs["home"]["ticks"] is not None else "no fire-free route, time unknown"
        return f"go to the safe strip{fire} and bank the bag (worth {obs['bag_value']:.1f}; {cost})"
    if name == "wait":
        return f"wait {args[0]} ticks"
    return " ".join(str(part) for part in command)


def room_map(obs: dict) -> str:
    """The floor as characters, top row first: # wall or furniture, F fire, = safe strip, . floor, R robot, digit item."""
    g = obs["grid"]
    walls, burning = {tuple(c) for c in g["walls"]}, {tuple(c) for c in g["burning"]}
    items = {tuple(i["cell"]): str(i["id"]) for i in obs["items"] if i["where"] == "floor"}
    rows = []
    for j in range(g["ny"] - 1, -1, -1):
        row = ""
        for i in range(g["nx"]):
            c = (i, j)
            row += ("R" if list(c) == obs["robot"]["cell"] else items[c] if c in items else "#" if c in walls
                    else "F" if c in burning else "=" if j >= g["safe_rows_from"] else ".")
        rows.append(f"{j:2d} {row}")
    return "\n".join(rows)


def describe(obs: dict, options: list, history: list | None = None, note: str = "") -> str:
    """The full text: the state, the history, the map, an optional note, and the lettered options."""
    c, r = obs["costs"], obs["robot"]
    lines = [INTRO, ""]
    lines.append(f"Time: tick {obs['tick']}; at most {obs['ticks_left']} ticks are left (one tick is 0.1 s). The episode also ends "
                 f"when {100 * obs['end_share']:.0f} % of the floor burns; {100 * obs['burning_share']:.0f} % burns now.")
    here = f"You: cell ({r['cell'][0]}, {r['cell'][1]}), health {obs['health']:.0f} of {obs['max_health']:.0f}."
    if r["in_fire"]:
        here += f" You are standing IN FIRE and lose {c['health_loss_per_tick']:.0f} health per tick."
    lines.append(here + " At health 0 the episode ends and the bag is lost.")
    bag = f"{len(obs['bag'])} item(s) worth {obs['bag_value']:.1f}, not banked yet" if obs["bag"] else "empty"
    lines.append(f"Bag: {bag}. Banked so far: {obs['banked_value']:.1f} of {obs['total_value']:.1f}.")
    lines.append(f"Costs: walking {c['ticks_per_cell']} ticks per cell, a pick {c['pick_ticks']} ticks, banking {c['drop_ticks']} ticks. "
                 f"An item on a burning cell loses {100 * c['item_loss_per_tick']:.0f} % of its starting value per tick. "
                 f"In fire you lose {c['health_loss_per_tick']:.0f} health per tick.")
    lines += ["", "Items on the floor:"]
    for item in obs["items"]:
        if item["where"] != "floor":
            continue
        text = f"  item {item['id']} at ({item['cell'][0]}, {item['cell'][1]}): worth {item['value']:.1f} (started at {item['start']:.0f})"
        if item["burning"]:
            loss = c["item_loss_per_tick"] * item["start"]
            text += f", BURNING: loses {loss:.2f} per tick, worth nothing in {item['value'] / loss:.0f} ticks"
        far = (f"{item['cells_away']} cells away, about {item['ticks_to_get']} ticks to walk there and pick it"
               if item["cells_away"] is not None else "no fire-free route to it")
        lines.append(f"{text}; {far}")
    home = obs["home"]
    lines.append(f"Safe strip (rows {obs['grid']['safe_rows_from']} and above): "
                 + (f"{home['cells_away']} cells away, about {home['ticks']} ticks to walk there and bank" if home["cells_away"] is not None
                    else "no fire-free route to it"))
    lines += ["", "What you did so far:"]
    for n, (command, result) in enumerate(history or [], 1):
        lines.append(f"  {n}. {' '.join(str(p) for p in command)}: {'ok' if result.done else result.reason}, {result.ticks} ticks")
    if not history:
        lines.append("  nothing yet")
    lines += ["", f"Map (the number on the left is the row; the top row is row {obs['grid']['ny'] - 1}, columns run 0 to "
              f"{obs['grid']['nx'] - 1} left to right; # wall or furniture, F fire, = safe strip, . floor, R you, a digit is that item):",
              room_map(obs)]
    if note:
        lines += ["", note]
    lines += ["", "Options:"] + [f"{LETTERS[n]}. {option}" for n, option in enumerate(options)]
    return "\n".join(lines)
