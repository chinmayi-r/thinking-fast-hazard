# Human-played episode, HAZARD fire, `mm_kitchen_3a-1`

Episode folder: `G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1\`. Played on CHIN_WIN (summary.json `machine`), clock `none`, 1 episode, final frame count 1565 (summary.json `final_frame_count`), 1564 steps and score 2.0 of max 9.0 (eval_result.json `avg_steps`, `avg_score`, `avg_max_score`).

Thinking-time fields are meaningless for this episode: decisions were left open for hours (decisions.jsonl `wall_s` ranges 0.096 s to 35,277 s; summary.json `wall_s_total` 69,136 s, `frames_at_30fps_total` 2,048,852). They are ignored below except where stated.

Every number below names its source file. HAZARD code citations are to `G:\iw-thinking-fast\HAZARD\src\HAZARD\` (paths relative to that).

## 0. What HAZARD saved in `0/` and how ids and distances work

`0/` holds (141 files): `human_log.json` (one record per decision; prompt, menu, letter, `automatic` flag, `holding`, `wall_s`), one `<frame>_human.json` per decision (the same record), `<frame>_info.json` for the 14 asked decisions (keys `cot_usage`, `num_available_actions`, `prompts`, `outputs`, `plan`, `total_cost`; written by `policy/llm.py:585`), `input<frame>.json` for the 14 asked decisions (keys `explored_object_name_list`, `holding_objects`, `nearest_object`, `step`, `action_result`, `action_info`, `save_dir`; written at `challenge.py:308-311`), and five PNGs per decision frame: `sem_map_`, `rgb_`, `obj_map_`, `height_map_`, `explore_` (written by `visualize_obs`, `policy/env_actions.py:79-83`, called at `challenge.py:312` for agent type llm/llmv2). No file carries the agent position as a number; see section 2 for how it was recovered.

Runtime ids. The manager renumbers objects in the order `add_object` commands appear in the scene file, starting the list at `[0]` (`envs/fire/manager.py:24,33` `id_list = [0]`; `manager.py:114-118` `add_object` appends and sets `id_renumbering[idx] = len(id_list)-1`; the scene loop that calls it is `envs/fire/fireagent_controller.py:160-170`). So runtime id = 0-based position among the 66 `add_object` entries in `data/room_setup_fire/test_set/mm_kitchen_3a-1/log.txt` + 1. Verified against `actions.txt` (`pick_up 64 b03_hair_comb_2010 8657745`, `pick_up 66 b04_comb 15599511`).

| runtime id | scene id | name | category | position (x, z) m | prompt value (by category, `challenge.py:106`) | scorer value (by name, `challenge.py:193-201`, `data/meta_data/value.json`) |
|---|---|---|---|---|---|---|
| 56 | 14901620 | bag_3 | bag, handbag, pocketbook, purse | (-1.427, -2.454) | 5 | **1** (`value.json["bag_3"] = 0` -> `low_value`) |
| 57 | 8265913 | b04_comb | hairbrush | (-0.677, -2.204) | 1 | 1 |
| 62 | 3819180 | small_purse | bag, handbag, pocketbook, purse | (0.323, 1.546) | 5 | **5** (`value.json["small_purse"] = 1` -> `high_value`) |
| 64 | 8657745 | b03_hair_comb_2010 | hairbrush | (0.823, 2.796) | 1 | 1 |
| 66 | 15599511 | b04_comb | hairbrush | (-1.427, 3.296) | 1 | 1 |

Max score 9 = 1 + 1 + 5 + 1 + 1 (`challenge.py:226` sums the per-name values), which is why eval_result.json says `avg_max_score` 9.0 and not 13. `high_value = 5`, `low_value = 1` at `challenge.py:89-90`. The prompt the person read says "bag, handbag, pocketbook, purse, value: 5" (human_log.json, every prompt), because the prompt's value table is built per category at `challenge.py:106`, while scoring looks the object name up at `challenge.py:193-201`. Runtime 62 (the one target actually worth 5) never appeared in any of the 14 prompts (human_log.json; `input<frame>.json` `explored_object_name_list` never lists id 62).

Agent spawn (-0.25, 0.0, 0.75); initial fires at (0.5, 2.25), (-0.25, 2.5), (0.5, 1.5) (scene `info.json`). The 5-point purse (62) is 0.18 m from the fire at (0.5, 1.5).

Distances in the prompt are in 0.1 m map cells, not metres: `policy/llm.py:432-452` takes the centroid of the cells of `sem_map['explored'] * sem_map['id']` equal to the object id and returns the Euclidean norm to `self.agent_pos` in grid units; the grid is 0.1 m (`challenge.py:48` `grid_size=0.1`, map 512 x 512, `map_size_h=512, map_size_v=512`). The "m" suffix in the prompt is wrong. The centroid is of whatever cells carry the id, so it can be far off when cells are mis-projected: hairbrush 57 is reported at 324-369 cells (32-37 m) in every prompt from decision 2 on, while its true distance from the robot was 5.0-6.1 m (section 2); hairbrush 64 is reported at 356.45 cells at decision 1 and 122.39 at decision 3 while the robot was 1.3 m from it at decision 3.

Walk primitive: `agent_walk_to` (`policy/env_actions.py:93-136`) replans A* every step on the current semantic map (`env_actions.py:110-117`), moves to the first waypoint (`env_actions.py:124-126`), succeeds when the planar distance to the target is below `arrived_at` (`env_actions.py:102-105`; the challenge passes `arrived_at=0.5`, `max_steps=100`, `challenge.py:345-356`) and fails with "max steps reached" when `frame_count - start_frame > max_steps` (`env_actions.py:107-110`). Prompt wording for that failure is "paused after taking 100 steps" (human_log.json). A* weights (`policy/astar.py:36-58`): explored free cell 1, unexplored 30, +10 for height > 0.1 m, + 7 * exp(3 * height) for height > 0.5 m (a 2.5 m wall cell is about 12,600), then a 5 x 5 weighted blur (`astar.py:51-57`); the search itself is 4-connected `pyastar2d.astar_path` (`astar.py:32`). Weights are finite, so a path always exists, including through walls and unexplored space.

Frame counter: `frame_count` is incremented once per `communicate()` in `envs/fire/fire.py:219` (inside `fire_step`, `fire.py:158`). The episode's `log.txt` (840 KB, 1569 lines) is the TDW `Logger` add-on output, one JSON line of commands per `communicate()` (`venv/Lib/site-packages/tdw/add_ons/logger.py:64-67`); line number = frame + 6 (calibrated on the first `object_look_at_position` after each walk decision: deltas of 5-6 at decisions 2, 4, 5, 6, 8, 9, 10, 13, 16).

## 1. Per-decision table

Sources: decisions.jsonl (`frame_count_before`, `frame_count_after_charge`, `action`), human_log.json (`choice`, menu text, `automatic`, `holding`, the "Previous actions" tail of the next prompt for HAZARD's result wording), output.log (`finish:` wording), eval_result.json `target_status` (saved frames 293 and 493). Frames consumed = next decision's `frame_count_before` minus this one's; the last uses `final_frame_count` 1565. Decisions 4, 5, 7, 8 carry `"automatic": true` and no prompt or menu in human_log.json (summary.json `human.automatic` = 4, `asked` = 14): the runner issued pick-up after a successful walk and drop after a pick-up without asking.

| # | frame before -> after | chosen (letter: menu text) | HAZARD result (output.log / next prompt) | frames consumed | saved so far |
|---|---|---|---|---|---|
| 1 | 43 -> 92 | B: look around | success | 49 | 0 |
| 2 | 92 -> 197 | D: go pick up object <hairbrush> (66) | fail, max steps reached ("paused after taking 100 steps") | 105 | 0 |
| 3 | 197 -> 234 | C: go pick up object <hairbrush> (64) | success | 37 | 0 |
| 4 | 234 -> 248 | AUTOMATIC: pick_up 64 | success (grasp of 8657745 at log line 246) | 14 | 0 |
| 5 | 248 -> 294 | AUTOMATIC: drop (holding hairbrush) | success; 8657745 teleported away at log line 280; `target_status[8657745] = 293` | 46 | 1 |
| 6 | 294 -> 387 | C: go pick up object <hairbrush> (66) | success | 93 | 1 |
| 7 | 387 -> 429 | AUTOMATIC: pick_up 66 | success (grasp of 15599511 at log line 414) | 42 | 1 |
| 8 | 429 -> 494 | AUTOMATIC: drop (holding hairbrush) | success; 15599511 teleported away at log line 480; `target_status[15599511] = 493` | 65 | 2 |
| 9 | 494 -> 600 | A: go pick up object <bag, handbag, pocketbook, purse> (56) | fail, max steps reached | 106 | 2 |
| 10 | 600 -> 703 | A: go pick up object <bag ...> (56) | fail, max steps reached | 103 | 2 |
| 11 | 703 -> 809 | B: go pick up object <hairbrush> (57) | fail, max steps reached; **robot leaves the house during this walk (frame ~788)** | 106 | 2 |
| 12 | 809 -> 921 | B: go pick up object <hairbrush> (57) | fail, max steps reached | 112 | 2 |
| 13 | 921 -> 1024 | A: go pick up object <bag ...> (56) | fail, max steps reached | 103 | 2 |
| 14 | 1024 -> 1130 | A: go pick up object <bag ...> (56) | fail, max steps reached | 106 | 2 |
| 15 | 1130 -> 1246 | A: go pick up object <bag ...> (56) | fail, max steps reached | 116 | 2 |
| 16 | 1246 -> 1352 | A: go pick up object <bag ...> (56) | fail, max steps reached | 106 | 2 |
| 17 | 1352 -> 1458 | A: go pick up object <bag ...> (56) | fail, max steps reached | 106 | 2 |
| 18 | 1458 -> 1565 (end) | A: go pick up object <bag ...> (56) | fail, max steps reached; episode ends at max_steps 1500 (`challenge.py:54`) | 107 | 2 |

Menus offered (human_log.json): decision 1 {A: 64, B: look around}; decision 2 {A: 56, B: 57, C: 64, D: 66}; decision 3 the same plus E: look around; decision 6 {A: 56, B: 57, C: 66, D: look around}; decisions 9-18 {A: 56, B: 57, C: look around}. Target 62 was never in a menu. Clicks recorded in summary.json `human.clicks` (B, D, C, C, A, A, B, B, A, A, A, A, A, A) match.

Frame costs measured here: look around 49; successful walks 37 and 93; failed walks 103-116 (the 100-frame cap plus the granularity of one `move_to` per replanning step, which is 5 frames in the log: `object_look_at_position` lines recur every 5 lines, e.g. 130, 135, 147, 152 ... in log.txt); automatic pick-ups 14 and 42; automatic drops 46 and 65. Twelve failed walks consumed 1,277 of the 1,522 frames after the first decision.

Prompt temperatures: hairbrush 66 was reported at 107.2 C from decision 3 on (human_log.json), yet it was saved with end state 0 = NORMAL (eval_result.json `final_states`), so both saved hairbrushes scored full value (`challenge.py:205-208`).

## 2. Trajectory

No JSON carries the agent position. Two sources recover it:

1. `sem_map_<frame>.png`: `visualize_obs` paints a 3 x 3 red block at the robot's grid cell `[int(x // 0.1 + 256), int(z // 0.1 + 256)]` (`env_actions.py:52-59`; `map_offset = [256, 256]` from `utils/model.py:217-219` with the default `origin_pos=[0,0]` since `fireagent_controller.py:71-72` passes none). Read back here with `png_analysis.py` (scratchpad, see end).
2. The episode `log.txt`: every `move_to` step sends `send_overlap_box` for replicant 2171609 at `transform.position + forward * collision_avoidance_distance` with y raised (`venv/Lib/site-packages/tdw/replicant/actions/move_by.py:188-198`). Subtracting 0.5 m along the rotation's forward vector reproduces the spawn (-0.25, 0.75) exactly at log lines 99-100, so 0.5 m is taken as that replicant's `collision_avoidance_distance` (it comes from the replicant record, `tdw/librarian.py:263`; not read directly). 449 such rows exist (only while walking; none during look-around, pick-up or drop). Extracted to the scratchpad `robot_track.csv`.

The two agree to within 0.1 m at every decision frame (table below: PNG cell vs track row at line = frame + 6).

Room bounding box from the scene file's object positions (`data/room_setup_fire/test_set/mm_kitchen_3a-1/log.txt`, `add_object` entries): x in [-1.576, 1.856] (painting 14707150 to candlestick 4003324), z in [-3.204, 3.296] (toothbrush 14098052 to comb 15599511); padded by 0.3 m to x in [-1.88, 2.16], z in [-3.50, 3.60]. No floor or wall objects are in the scene file (the walls come with `add_scene mm_kitchen_3a`); the high-height band in the reconstructed height map (section 4) agrees with this box.

| # | frame | robot (x, z) m, track | robot, PNG cell | target | distance to target, m (track) | prompt's "distance" (cells) | inside room box |
|---|---|---|---|---|---|---|---|
| 1 | 43 | (-0.25, 0.75) spawn | (-0.3, 0.7) | - | - | 64: 356.45 | yes |
| 2 | 92 | (-0.25, 0.75) | (-0.3, 0.7) | 66 | 2.83 | 25.71 | yes |
| 3 | 197 | (-0.50, 2.77) | (-0.6, 2.7) | 64 | 1.32 | 122.39 | yes |
| 4 | 234 | (0.29, 2.82) | (0.3, 2.8) | 64 | 0.53 | - | yes |
| 5 | 248 | (0.29, 2.82) | (0.3, 2.8) | - | - | - | yes |
| 6 | 294 | (0.29, 2.82) | (0.3, 2.8) | 66 | 1.78 | 17.36 | yes |
| 7 | 387 | (-1.49, 2.79) | (-1.5, 2.8) | 66 | 0.51 | - | yes |
| 8 | 429 | (-1.49, 2.79) | (-1.5, 2.8) | - | - | - | yes |
| 9 | 494 | (-1.49, 2.79) | (-1.5, 2.8) | 56 | 5.25 | 50.75 | yes |
| 10 | 600 | (-0.12, 2.79) | (-0.1, 2.7) | 56 | 5.41 | 62.34 | yes |
| 11 | 703 | (-0.84, 2.79) | (-0.9, 2.7) | 57 | 4.99 | 335.63 | yes |
| 12 | 809 | (-1.31, 3.85) | (-1.4, 3.8) | 57 | 6.08 | 342.29 | **no** |
| 13 | 921 | (0.20, 4.40) | (0.2, 4.4) | 56 | 7.05 | 78.58 | no |
| 14 | 1024 | (2.59, 4.36) | (2.6, 4.3) | 56 | 7.91 | 86.18 | no |
| 15 | 1130 | (3.01, 3.73) | (3.0, 3.6) | 56 | 7.61 | 83.14 | no |
| 16 | 1246 | (3.28, 2.01) | (3.2, 1.9) | 56 | 6.49 | 70.91 | no |
| 17 | 1352 | (3.31, 0.95) | (3.3, 0.8) | 56 | 5.83 | 64.05 | no |
| 18 | 1458 | (2.62, 0.60) | (2.6, 0.5) | 56 | 5.07 | 56.81 | no |
| end | ~1560 | (2.60, -1.37) | - | 56 | 4.16 | - | no |

Path length walked per decision (sum of consecutive track deltas): 2.25 m (dec. 2), 0.84 (3), 1.88 (6), 1.87 (9), 1.59 (10), 1.34 (11), 1.97 (12), 2.34 (13), 0.97 (14), 1.79 (15), 0.97 (16), 0.97 (17), 1.93 (18). A failed 100-frame walk moves the robot about 1-2.3 m.

Frame range outside the room box: from about frame 788 (log line 794, robot at (-1.30, 3.61), during decision 11's walk) to the end (line 1566, frame ~1560, (2.60, -1.37)); 242 of the 449 walking rows are outside. Route: north through the north-west corner at x about -1.3 (frames 703-809), east along z about 4.4 outside the north wall (809-1024), then south along x about 3.3 outside the east wall (1024-1565), i.e. wall-following clockwise around the house while the bag sat in the south-west corner inside.

What the walks did inside, frames 92-703: the robot stayed on the z = 2.5-2.85 strip along the north part of the kitchen the whole time (track x from -1.49 to 0.38). That strip is the top edge of the fire belt: the initial fires are at z 1.5-2.5 and the 90 `add_visual_effect` fire spawns in log.txt cover x in [-1.13, 1.38], z in [0.62, 3.16] by the end. The south half of the room (z < 0, where 56 and 57 are) is unexplored in every map saved (white in the PNGs).

Decision 2's first walk (to 66) failed at the cap while the robot moved only 2.25 m and ended 0.5 m short of 66's line: the `object_look_at_position` waypoints in log.txt lines 130-197 oscillate between z 2.1 and 2.9 at x -0.3 to -0.7, consistent with the door-frame bump reported during play; the second attempt (decision 6) reached it in 93 frames.

## 3. Where the value went

| scene id | runtime | name | scorer value | outcome | frame |
|---|---|---|---|---|---|
| 8657745 | 64 | b03_hair_comb_2010 | 1 | saved, end state NORMAL (0) | 293 (eval_result.json `target_status`) |
| 15599511 | 66 | b04_comb | 1 | saved, end state NORMAL (0) | 493 |
| 14901620 | 56 | bag_3 | 1 (shown as 5 in the prompt) | never reached; 10 failed walks (decisions 9, 10, 13-18) | - |
| 8265913 | 57 | b04_comb | 1 | never reached; 2 failed walks (decisions 11, 12) | - |
| 3819180 | 62 | small_purse | 5 | never shown in any prompt; never attempted | - |

Score 2.0 of 9.0 (eval_result.json). The ten walks toward 56 were spent on an item worth 1 to the scorer, chosen because the prompt said 5. The only 5-point item sat 1.0 m from the spawn (from (-0.25, 0.75) to (0.323, 1.546)) and 0.18 m from an initial fire; the random agent on the same kitchen picked it up with a blind `pick_up` from the spawn at frame 242 (section 5), so it was gettable in the first 250 frames.

## 4. Semantic map renders

The arrays themselves are not saved. `visualize_obs` writes only PNGs (`env_actions.py:79-83`), called at `challenge.py:312`. What can be recovered: `explore_<f>.png` is exactly `explored * 255` (`env_actions.py:74-78, 83`; pixel values are only 0 and 255 in all 18 files). `height_map_<f>.png` is `height * int(255 / height.max())` cast to uint8 (`env_actions.py:40-45, 82`), and `sem_map_<f>.png` grey is `200 - 200 * min(height/2, 1)` on explored cells with height > 0.3 (`env_actions.py:47-49`); solving both gives the scale `k = int(255/hmax) = 102` px/m (hmax 2.50 m) at every frame, so height is recovered to 0.01 m with saturation at 2.5 m. The id map is not recoverable (`obj_map` is binary, `env_actions.py:38-39`). To save the arrays, add `np.save` of `sem_map["explored"]`, `["height"]`, `["id"]` next to the `cv2.imwrite` calls in `visualize_obs` (`env_actions.py:79-83`), or dump `state['sem_map']` beside the `input<frame>.json` write at `challenge.py:309-311`.

Rendered with the reconstructed height and the exact `get_astar_weight` (`astar.py:36-58`), 4-connected `pyastar2d` path from the robot to bag 56 (destination = the scene position of 56, as `agent_walk_to` uses `manager.objects[target].position`, `env_actions.py:101`), robot in red, targets as blue squares with runtime ids, initial fires orange, room box blue, image x = world x rightwards, image y = world z upwards:

- `astar_cost_frame703.png`: decision 11, the state the planner saw when the walk that left the house started. A* path 202 cells, 137 of them outside the room box: west along the explored north strip, out of the north-west corner, around the outside of the north and east walls, in from the south. This is the route the robot then followed over the next 850 frames.
- `astar_cost_frame809.png`: decision 12, robot already outside the north-west corner (its own cell sits in the wall band, weight about 3,600). Path 188 cells, 136 outside.
- `astar_cost_frame1458.png`: last decision, robot outside the east wall at (2.6, 0.5). Path 133 cells, 128 outside, continuing south around the house.

Caveats: height is quantized and saturated at 2.5 m, the real planner replanned every 5 frames and the destination can differ if the bag moved (it is non-kinematic, `fireagent_controller.py:244-246`), and the real walk executes only the first waypoint each time. The reconstructed path is indicative, but it reproduces the observed exit and direction at all three frames.

Why the planner leaves: everything the robot has seen is on the north strip and outside the house (cells seen through the door and windows are explored = cost 1), while the interior south half is unexplored (cost 30 per cell before the blur) and the table-chair cluster around (0, 0) is a high-height block. A route of 130-200 explored cells around the outside is cheaper than 60 unexplored interior cells, and as the robot walks outside it explores more outside cells, which keeps the outside route the cheapest at every replanning. Nothing in the weights distinguishes outside from inside.

## 5. Comparison with the shipped agents on the same kitchen

Sources: `G:\iw-thinking-fast\results\sweep_fire_<agent>_test25\mm_kitchen_3a-1\` `output.log` ("Executing step ... action ..., finish: ..."), `actions.txt`, `eval_result.json`, `decisions.jsonl` (line counts 49, 20, 20, 15). The random agent's `decisions.jsonl` records integer actions mapped by `envs/fire/fire_gym.py:138-150` (`walk_to_single` = one A* step to the nearest unfinished target, `pick_up` nearest, drop, explore), and its "walk_to_single" is `agent_walk_to_single_step` (`env_actions.py:140-185`), one replanning step rather than a 100-frame walk. The mcts and rule agents save the same PNG set (20 decision frames each, `0/`); the haiku run saves PNGs plus `input*.json` and `*_info.json` (15 frames).

| agent | decisions | walks | failed walks | pick-ups (ok / fail) | drops | explores | targets saved (frame) | score / 9 | steps |
|---|---|---|---|---|---|---|---|---|---|
| human (this episode) | 18 | 14 `walk_to` | 12 max steps | 2 (2 / 0), automatic | 2 | 1 | 8657745 (293), 15599511 (493) | 2.0 | 1564 |
| random | 49 | 19 `walk_to_single` | 1 (target not in vision or memory) | 9 (3 / 6 failed to grasp) | 5 (3 ok, 2 not holding) | 16 | 3819180 (242, state 0 -> 5), 8657745 (858, state 1 -> 0.5), 15599511 (1350, state 1 -> 0.5) | 6.0 | 1524 |
| mcts | 20 | 14 `walk_to` | 13 max steps | 1 (1 / 0) | 1 | 4 | 15599511 (411) | 1.0 | 1598 |
| rule | 20 | 12 `walk_to` | 10 max steps | 1 (1 / 0) | 1 | 6 | 8657745 (497) | 1.0 | 1537 |
| llm_haiku | 15 | 14 `walk_to` (all to 56) | 14 max steps | 0 | 0 | 1 | none | 0.0 | 1558 |

Half credit for random's hairbrushes is `challenge.py:205-208` (saved but burning at the end, `final_states` 1). Targets 14901620 (bag_3, 56) and 8265913 (hairbrush 57), both in the south-west corner, were saved by nobody; the rule agent's tenth walk to 56 did return success at decision 19 (output.log) but the episode ended before a pick-up. The mcts agent lost its first 100 frames to the same door-frame failure on 64 that the person had on 66, then failed 13 walks; haiku chose the bag at every one of its 14 decisions and never got there. The random agent scored highest because its action space bypasses both failure modes: its single-step walks never trigger the 100-frame cap, and its blind nearest-target `pick_up` took the 5-point purse that the prompt never listed.

## 6. Findings for the advisor

The person did what the menu allowed well: one look-around, then the two reachable 1-point hairbrushes in the first 494 frames (rule and mcts each got one, haiku none), then a commitment to the only high-value item the interface showed. The interface misled them twice: the prompt values targets by category (`challenge.py:106`), so `bag_3` read "value 5" while the scorer values by name (`challenge.py:193-201`, `value.json` `bag_3 = 0` -> 1 point), and the real 5-point `small_purse` (runtime 62, 1.0 m from the spawn, 0.18 m from a fire) never appeared in any prompt although the random agent picked it up blind at frame 242. What the body prevented was everything after that: all twelve walks toward the south half of the kitchen died at the 100-frame cap (`env_actions.py:107-110`), and from frame ~788 the A* planner (explored 1, unexplored 30, `astar.py:36-58`, no inside/outside notion) routed the robot out of the north-west corner and clockwise around the outside of the house, where it stayed for the last 777 frames, a route the same planner reproduces on the reconstructed maps at frames 703, 809 and 1458. Decision latency was irrelevant in this kitchen: twelve failed walks cost 1,277 frames, whereas haiku's 4-5 s per decision would cost about 130 frames each under the 30 FPS rule, so the binding constraint was the walk primitive and the map, not thinking time, and no gate between fast and slow thinkers can recover value that the menu never offered and the controller cannot reach.

## Files created by this analysis

- `G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1\ANALYSIS.md` (this file)
- `G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1\astar_cost_frame703.png`
- `G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1\astar_cost_frame809.png`
- `G:\iw-thinking-fast\results\human_cr_text\mm_kitchen_3a-1\astar_cost_frame1458.png`
- Scratch (session temp, not in the project): `C:\Users\Chinm_user\AppData\Local\Temp\claude\4f7394ba4b6593ea\f22c7979-d273-41b9-bf02-0f5298f33f24\scratchpad\png_analysis.py`, `...\scratchpad\log_extract.py`, `...\scratchpad\render_maps.py`, `...\scratchpad\robot_track.csv`. `render_maps.py` is the one worth copying into the project if the renders are to be regenerated.

No existing file was modified, the simulator was not run, nothing was committed.
