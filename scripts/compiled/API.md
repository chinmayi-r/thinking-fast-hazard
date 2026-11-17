# The program API for HAZARD (compiled policy, A39 step i), version 1.1

This is the document a compiler (Opus, or a person) reads before writing a program that plays HAZARD's fire task
without any model call during play. Every statement below is backed by `scripts/compiled/api.py` and by the HAZARD
line it cites. The worked example is `scripts/compiled/examples/reflex_program.py`, RRARA's reflex written against
this API; tests in `scripts/compiled/test_api.py` show it decides exactly as `scripts/rrara/reflex.py` does.

## 1. What a program is

A Python module that defines

```python
class Policy:
    def __init__(self, task, target_values):   # task: "fire" | "flood" | "wind"; target_values: {category: value}
        ...
    def decide(self, obs):                      # obs: the dict in section 3
        return ("walk_to", "71")                # one of the actions in section 4
```

Rules the loader (`api.load_program`) enforces before the first episode:

- Imports only from: `math, collections, heapq, itertools, functools, typing, dataclasses, numpy, compiled.api`.
  No files, sockets, subprocesses, model clients, `random`, `time`. A program is deterministic and plays alone
  (Liane's "no model calls during play").
- `class Policy` with `__init__` and `decide` at module level.

Rules enforced at every decision (`api.ProgramAgent.choose_target`):

- `decide` may keep memory in `self` across decisions; a new `Policy` is built per episode (`reset`).
- The returned action must be on the legal menu (section 4); otherwise, or if `decide` raises, the body does
  `("explore", None)` and the decision is logged as illegal with the reason. The episode goes on.
- Wall time is measured and, under the clock, charged to the world: frames = ceil(seconds x 30) (RRARA's rule,
  `scripts/run_baseline.py` TimedAgent). The reflex decides in 8-20 ms (1 frame). One A* search on the full
  512 x 512 map costs 0.1-0.2 s (3-6 frames); `Paths` (section 5) pools the map 4 x 4 so a search is a few ms.
  Budget to design for: under 33 ms per decision, so thinking never costs more than a frame.

## 2. The episode

- Fire and flood: 1500 frames; wind: 3000 (`challenge.py:53-59`). The loop ends when `frame_count >= max_steps`
  (`challenge.py:421`). At 30 fps that is 50 s of world time.
- Score: value saved / value of all targets (`challenge.py:187-230`). A target counts once it is dropped into the
  bag (fire, flood). A burning target loses its value (prompt_v2.csv: "objects lose their value once they start
  burning").
- The values the program gets are per CATEGORY (`challenge.py:100-108`: 5 if the category is in
  `scenes/scene_configs/value.json` else 1), the same table every shipped agent gets at reset. The scorer pays per
  object NAME (`challenge.py:190-203`, `data/meta_data/value.json`). The two disagree for some objects (Nov 16: a
  `bag_3` reads 5 and pays 1). This is HAZARD as shipped; exposing name values is an open point (section 7).

## 3. The observation (`api.build_observation`)

Built from HAZARD's `state` (`fireagent_controller.py:535-566`) and `processed_input` (`challenge.py:134-163`) at
each decision. Nothing comes from the scene file or ground truth (A39 point 9): the object list is what the robot's
camera has segmented and put on its map; the map is the robot's own.

| key | type | meaning | source |
|---|---|---|---|
| `api_version` | str | "1" | |
| `task` | str | "fire" / "flood" / "wind" | |
| `frame` | int | world frame at decision time | `processed_input["step"]`, `challenge.py:160` |
| `frames_left` | int | `EPISODE_FRAMES[task] - frame`, floor 0 | `challenge.py:53-59` |
| `agent_cell` | (r, c) or None | the robot's map cell; None when off the map edge | `goal_map == -2`, `fireagent_controller.py:557-558` |
| `facing_cell` | (r, c) or None | the cell one step ahead in the facing direction (8 directions) | `goal_map == -1`, `fireagent_controller.py:559-565` |
| `heading_deg` | float or None | heading on the map from the facing mark: angle of (facing - agent) from +row (world x) toward +col (world z), the body's `get_facing()` convention (`envs/fire/agent.py:270-275`), quantized to 45 degrees | v1.1 |
| `holding` | list of object | what the robot holds (0 or 1 item) | `processed_input["holding_objects"]`, `challenge.py:149, 167-172` |
| `at_object` | object or None | the object the robot just walked to; set only after a `walk_to` arrived, cleared otherwise | `processed_input["nearest_object"]`, `challenge.py:150-159, 357-360` |
| `objects` | list of object | every object currently on the explored map, all categories | `processed_input["explored_object_name_list"]`, `challenge.py:136-148` |
| `last_action` | (name, id or None) or None | the action the program returned last decision | kept by `ProgramAgent` |
| `last_result` | {"ok": bool, "info": str} or None | how the body executed it; None on the first decision | `processed_input["action_result"/"action_info"]`, `challenge.py:161-162` |
| `map` | dict of 3 arrays | `explored` (0/1), `height` (metres), `id` (object ids, meaningful where explored == 1); 512 x 512 cells of 0.1 m | `state["sem_map"]`, `challenge.py:48`, `fireagent_controller.py:535-540` |

An **object** is `{"id": str, "name": str, "category": str, "is_target": bool, "value": float or None}`; objects
in `objects` also carry `"cell": (r, c) or None` (mean of the object's explored cells, `llm.py:544-549`),
`"straight_cells": float or None` (straight-line cells from the robot), `"range_m"` (the same in metres) and
`"bearing_deg"` (direction relative to the heading, (-180, 180], positive = the robot's left; A41: the sign is derived
from `turn_by`'s "positive = clockwise" and `get_facing`'s arctan2(z, x) and is to be confirmed on the first camera
episode). `value` is the category value for targets
and None for everything else. Ids are strings (`challenge.py:144` builds them as str; the body does `int()` on them).

What is NOT in the observation, by design: the camera image, the temperature layer (`state["sem_map"]["other"]`,
`state["raw"]["log_temp"]`), objects already saved (erased from the map before the decision, `challenge.py:301-303`),
and anything about objects the camera has not seen. The first two are open points (section 7).

## 4. Actions and their legality (`api.legal_actions`, `api.check_action`)

The legal menu is the one HAZARD's own agents choose from (`policy/rule_based.py:40-62`, the LLM agent's
`get_available_plans`), so a program competes on the same terms as Haiku and the rule agent:

| action | when legal | what the body does | frames | failure `info` strings |
|---|---|---|---|---|
| `("explore", None)` | always | 12 turns of 30 degrees in place; each view updates the map | 49 measured (Oct 28 run) | "can not turn around at this time" (`env_actions.py:246-254`) |
| `("walk_to", id)` | a seen target, not holding | one A* step per loop on the current map toward the object's true position, replanning each step, until within `arrived_at` or the cap | up to 100 (`challenge.py:346-355`); the cap is the common outcome on d35c71e (A29) | "max steps reached" (`env_actions.py:106`), "target not in vision or memory" (`:115`), "don't know, maybe out of bounds" (`:123`) |
| `("pick_up", id)` or `("pick_up", None)` | `at_object` is a target and nothing is held | reach for and grasp the object | a few dozen | "You need to walk to an object first." (`challenge.py:389`), "cannot grasp, maybe too far" (`env_actions.py:241`), "failed to grasp" |
| `("drop", None)` | holding, fire/flood | puts the held object into the bag: this is what scores | a few dozen | "not holding an object" (`env_actions.py:261`) |

`arrived_at` is 0.5 m on d35c71e (`challenge.py:346, 351`) and 1.25 m on the paper's snapshot 5a50c2d (A29); on
the laptop snapshot most walks end at the cap. A program should expect `walk_to` to fail and plan for it
(the example multiplies the cost of a target by 2 per failed walk, A34).

Actions the body supports but this menu does not offer yet (open point): `walk_to_single` (one A* step,
`challenge.py:371-383`), `low_level.turn_by` / `move_by` (`challenge.py:361-369`, `env_actions.py:137-156`),
walking to a map position rather than an object (`env_actions.py:87` accepts a position; `challenge.py:345` passes ids
only).

## 5. Helpers a program may use (`from compiled.api import Paths`)

- `Paths(obs, coarse=4)`: `.path(id)` returns the planned path's corner points as full-map cells (HAZARD's A* returns
  turning points only, `astar.py:16-34, path_cleanup`) or None;
  `.cells(id)` its length. It builds HAZARD's own A* weight on the robot's map once (`policy/astar.py`, the function the
  walk executor uses, `env_actions.py:118-119`), pooled `coarse` x `coarse` by block max so obstacles stay obstacles.
  `.cells(id)` returns the path length in full cells (0.1 m) from the robot to that object's cell, or None when the
  robot or the object has no cell. HAZARD's weights are finite everywhere (a 2.5 m wall costs about 12,000 per
  cell, `astar.py:41-42`), so A* always finds a path: a wall in the way shows as a path that goes round it, or
  straight through it when going round is dearer than 12,000 per cell. "Reachable" is the program's own judgement
  (for example, path length against straight line). Costs a few ms per object after the first.
- `obs["objects"][i]["straight_cells"]`: straight-line distance, free; good for choosing which objects to plan.
- Anything else the program needs it computes from `obs["map"]` with numpy.

The map knows nothing a real robot would not: unexplored cells cost 30 in the shipped planner (`astar.py:40`) and
the robot walks through them as if they were floor; with `HAZARD_ASTAR_FIX=1` (A38) they cost 300 and unranged
continuations of walls are priced as walls. A program's `Paths` sees the same weights the body walks by.

## 6. The worked example

`examples/reflex_program.py` is RRARA's rapid-reflex policy (Zheng et al. 2025 eq. 2, our `scripts/rrara/reflex.py`,
A34) in program form: drop if holding; pick up if standing at a target; else walk to the target with the least
(path cells + 1) / value, doubled per earlier failed walk, planning only the six nearest by straight line; else explore.
`test_api.py` checks on random maps that it returns the same action as `PaperReflex` at every decision, so the reflex's
measured scores apply to it: 0.757 (5a50c2d, clocked, 25 test kitchens) and 0.199 (d35c71e, clocked), Della, Nov 10.
The program itself has not played an episode yet (no runs without CR's go).

## 7. Open points (ASSUMPTIONS.md A40)

1. Value per name: the name is already in the robot's object list, so a name value table is not privileged
   information; exposing it changes the text every agent gets. CR's call (A32).
2. The temperature layer and the camera image are observed by the robot but not exposed in v1.
3. The three extra controls (bounded walk, turn, walk to a position) exist in the body but not on the menu.
4. The gate threshold for validation (A39 point 5) and the snapshot to compile on (point 10) are open with Tom.

## 8. Running a program (not yet run; one kitchen first, when CR says)

```
cd G:\iw-thinking-fast\HAZARD
..\venv\Scripts\python.exe ..\scripts\run_baseline.py --env fire --agent program --program ..\scripts\compiled\examples\reflex_program.py --episodes 1 --data_dir <kitchen> --clock tcm --out <results dir>
```

The summary carries a `program` block (decisions, illegal count, wall per decision, frames at 30 fps, action counts) and
`program_decisions.json` has one record per decision.
