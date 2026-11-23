# The program API for the fire task, version 1

This is what a program-writer reads before writing a program that plays the fire task with no model call.
Everything here is backed by `firelab/interface.py` (the observation and the commands) and `firelab/programs.py`
(the loader and the runner). The worked example is `firelab/examples/reflex_program.py`.

## 1. What a program is

A Python module that defines

```python
class Policy:
    def __init__(self, rules):   # rules: the parts of the observation that never change (section 3, "rules")
        ...
    def decide(self, obs):       # obs: the dictionary of section 3
        return ("get", 3)        # one command of section 4, or None to stop for good
```

- It may import only: math, collections, heapq, itertools, functools, typing, dataclasses, numpy. No files, network,
  model clients, clock or randomness. A file that breaks this is refused before it plays.
- `decide` may keep memory in `self`. A new `Policy` is made for every episode.
- If `decide` raises, or returns a command that is illegal, one tick is lost, the fact is recorded, and the episode
  goes on. A program must never crash: every branch returns a command or None.
- Returning None ends the program's part in the episode; what is banked is kept.

## 2. The episode

- A room with a fire that starts in one cell and spreads. The floor is a grid of 16 columns (i, 0 to 15) by 15 rows
  (j, 0 to 14); a cell is 0.2 m. One partition runs across row 5 with a doorway; rows 11 and above are the safe strip.
- Items lie on the floor. Score = value banked / value on the floor at the start. Only banked items count.
- To bank: pick items into the bag, walk to the safe strip, drop. A drop banks the whole bag at once.
- An item on a burning cell loses 2 % of its starting value per tick until picked. A picked item keeps its value.
- The robot has health 100 and loses 2 per tick while its cell burns. At health 0 the episode ends and the bag is lost.
- The episode also ends when half the floor burns, or after 500 ticks. Whatever is in the bag then is lost.
- Time is in ticks (0.1 s). Walking costs 4 ticks per cell, a pick 10 ticks, a drop 5 ticks.

## 3. The observation (`Session.observe()`), plain numbers, strings and lists

| key | meaning |
|---|---|
| `tick`, `ticks_left` | time now, and ticks until the 500-tick cap |
| `health`, `max_health` | the robot's health |
| `burning_share`, `end_share` | share of the floor burning now; the share that ends the episode (0.5) |
| `robot` | `{"cell": [i, j], "x", "y", "in_safe": bool, "in_fire": bool}` |
| `bag`, `bag_value` | ids of the items carried, and what they are worth together |
| `banked_value`, `total_value`, `score` | banked so far; value at the start; banked / total |
| `items` | one record per item, in id order (below) |
| `home` | `{"cells_away", "ticks"}`: a fire-free route to the safe strip and the ticks to walk it and bank; both None if there is none |
| `grid` | `{"nx": 16, "ny": 15, "cell_m": 0.2, "burning": [[i, j], ...], "walls": [[i, j], ...], "safe_rows_from": 11, "open": [[i, j], ...]}`; `open` lists the cells the robot can stand on |
| `job` | None, or the pick or drop under way |
| `last_command`, `last_result` | the previous command as a list, and `{"ticks", "done", "reason"}`; both None at the start |
| `costs` | `{"ticks_per_cell": 4, "pick_ticks": 10, "drop_ticks": 5, "item_loss_per_tick": 0.02, "health_loss_per_tick": 2.0}` |

An item record: `{"id", "cell": [i, j], "value" (now), "start" (at the beginning), "where": "floor" | "bag" | "saved" | "lost",
"burning": bool, "cells_away", "ticks_to_get"}`. `cells_away` and `ticks_to_get` describe a fire-free route to a cell from
which the item can be picked (its own cell or one of the 8 around it); both are None when there is no such route or
the item is not on the floor.

`rules` (given to `__init__`) holds `api_version`, `costs`, `end_share`, `max_health`, `total_value`.

The whole room is visible: every item and every burning cell. What is NOT visible: where the fire will be later.

## 4. Commands

| command | what the robot does | result when it cannot |
|---|---|---|
| `("get", k)` | walks by a fire-free route into reach of item k and picks it | "no route" (0 ticks if there is none from the start) |
| `("get", k, True)` | the same, but the route may cross fire (costs health) | "episode over" if it dies on the way |
| `("home",)` | walks by a fire-free route to the safe strip and banks the bag | "no route"; "refused" with an empty bag |
| `("home", True)` | the same, through fire if it must | |
| `("go", (i, j))` / `("go", (i, j), True)` | walks to a cell of `grid["open"]` | "no route" |
| `("pick", k)` / `("pick",)` | picks item k, or any item in reach, without walking | "refused" |
| `("drop",)` | banks the bag here | "refused" outside the safe strip or with an empty bag |
| `("wait", n)` | stands n ticks, 1 to 100 | |

A walk re-plans after every cell, so it adapts to the fire. A command returns only when it is finished or has failed;
`decide` is called again after it. A legal command that cannot be done costs what it used (often nothing).

## 5. What the reflex example does, and where it loses

It fetches the most valuable item left, banks after every pick, never walks through fire, and stops when it cannot
get home. It scores about 0.60 to 0.63 on average. It loses value by walking home after every single pick, by
never crossing fire even when the health cost would be small, and by ignoring how soon an item or a route will burn.
