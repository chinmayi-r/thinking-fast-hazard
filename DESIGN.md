# DESIGN: the fire task on the lab simulator, chunk by chunk

One section per chunk of PLAN.md. Each says what the code does, where it lives and why there, and what the alternatives were. Written as the chunks land, Nov 22, 2026 onward.

## Chunk 0: environments on disk (no task code)

**What it does.** Puts kinder's lab-room env on the laptop in a form that can be stepped and tested, measures how fast it is, and records two defects met on the way. Files: `scripts/kinder_step_rate.py` (make, reset, 200 random steps, 200 zero steps, one render, all timed), `scripts/kinder_probe_hang.py` (timestamped construction stages with a switch that disables kinder's stdout redirect), `logs/install_kinder.log`, `logs/kinder_*.log`, NOTES.md Nov 22, ASSUMPTIONS.md A62 to A64.

**Where it lives and why.** The venv is `venv_kinder/` beside the HAZARD `venv/`, not inside it: HAZARD's venv is Python 3.10.11 with torch and mmcv pinned to it, prpl-tidybot wants 3.10.14 or newer, and the two stacks never need to import each other. The measurement scripts sit in `scripts/` with the HAZARD-era scripts because they are tooling about the simulator, not part of the task package `firelab/` that Liane will install.

**Decisions and alternatives.**
- *uv-managed interpreter versus conda or the NuGet Python we used for HAZARD.* uv was already on the machine, fetches a 3.10.20 build onto G in one command, and is what kindergarden's README recommends. conda would have added a second package manager; the NuGet route gave 3.10.11, below prpl-tidybot's floor.
- *kinematic3d backend only versus `pip install -e ".[develop]"`.* The full install pulls MuJoCo, pygame, pymunk, OpenCV and the linters, none of which the PrplLab3D env needs, and the laptop's C drive has 1 GB free, so every cache is on G and the install is kept small. Cost: the lab's requirements file for this subset is incomplete (A64) and we found out the hard way; benefit: a 57-package venv that installs in 100 s and a precise list of what the fire env depends on.
- *A monkeypatch shim for the Windows hang versus editing the kindergarden clone versus a fork.* The clone is pinned to the commit prpl-tidybot depends on and must stay byte-identical so Liane and we run the same simulator. A fork would carry a one-line change for one platform. The shim (replace `prpl3d._suppress_urdf_warnings` with `contextlib.nullcontext` on win32 before the env is built) is five lines, lives in our package, and does nothing on Linux. Cost: 115 harmless URDF warning pairs print on Windows.
- *Measuring the step rate versus taking "fast to render" on trust.* Liane said it was fast; the number decides the plan. 127 to 183 steps per second means a 500-tick episode simulates in about 3 s without rendering, and one render costs 0.86 s, so videos must be strided, as prpl-tidybot already does. These are the figures the clock's tick length and the Della budget rest on.

## Chunk 1: the fire grid, no simulator (firelab/fire.py)

**What it does.** A 16 x 15 array of int8 cell states over the lab floor, one method per verb the env will need: `cell_of` and `center_of` convert between metres and cell indices; `block_box` marks the floor under furniture; `reset` clears the fire and ignites one cell (a given one or a uniform draw over burnable cells, from the grid's own RNG so the env can hand it gymnasium's seeded generator); `step` applies the spread rule once and returns how many cells caught; `is_burning`, `coverage`, `burning_cells`, `features` are the queries the env, the agents and the stop rule read. `measure_spread.py` runs the grid alone over seeds and probabilities and prints the time-to-half-the-room table that fixed p (A57 addendum). `test_fire.py` holds ten tests, each pinning one rule.

**Where it lives and why.** `firelab/fire.py` is the first file of the package Liane will install, and it imports only numpy. Keeping the fire out of the env class means the spread rule can be tested in half a second with no pybullet, measured over hundreds of seeds in seconds, swapped for HAZARD's rule later without touching the env, and reused for a flood (a different rule, a different drain, the same grid). The env (chunk 2) owns a FireGrid and asks it questions; it never reaches into the array.

**Decisions and alternatives.**
- *Vectorised neighbour count versus a loop over burning cells.* The rule "each burning cell ignites each NORMAL neighbour with probability p" is equivalent to "a NORMAL cell with k burning neighbours ignites with probability 1 - (1 - p)^k". The second form is four shifted additions and one array of uniforms per tick, the same work whether 1 or 200 cells burn, and it draws exactly one random number per cell per tick, which makes two grids with the same seed agree for ever. The loop form would have been easier to read but 100 times slower on a full room and its random stream would depend on iteration order.
- *New fire spreads from the next tick.* `step` computes the neighbour count from the state before any ignition in that tick, so a fire front advances one cell per tick at most. HAZARD has the same property through its candidate list (fire.py:154-155, 173-181: a candidate becomes a fire only after the simulator confirms the spot on the next frame). The test with a corner ignition (2 then 3 new cells) pins it.
- *No burnout at level 1.* HAZARD's floor fires never go out (object.py:53-59). An object that burns out is a property of objects, not cells, and belongs to the valuables in chunk 3. `BURNT` exists in the enum so the extinguisher (chunk 8) has somewhere to go without changing the feature encoding.
- *One cell state integer, not a per-cell temperature.* HAZARD's objects carry a temperature that diffuses with 1/d^2 weights (object.py:84-113). That gives a smooth front and per-object ignition thresholds, but it costs an O(n^2) pass every 5 frames and nobody has ever read a temperature from an agent's observation; the agents read "is this cell on fire". A state integer is what the agent sees, so it is what the model keeps.
- *Grid geometry from tidybot's floor bounds and HAZARD's cell size.* The bounds are the box prpl-tidybot's plan preview refuses to leave (conf/config.yaml preview.floor_bounds), so an agent that stays on our grid is one the real robot could follow; 0.2 m is HAZARD's spread step (fire_utils.py:21), so one of our cells is one HAZARD fire box.
- *p chosen by measurement.* The call said 0.1; the table says that burns half the room in 5 s. 0.015 gives 31 s on the open floor, inside the 50 s episode, so the 50 % stop rule binds in most seeds and the robot has time to act but not to dawdle. Furniture will shrink the burnable set and the number is re-measured in chunk 2.
- *Line budget.* 308 raw lines, 205 lines of code once blanks, comments and docstrings are removed (counted with an ast pass). The docstrings are long on purpose: this is the file Liane reads first.

## Chunk 2: the level 1 env (firelab/types.py, firelab/env.py, firelab/__init__.py)

**What it does.** Wraps kinder's lab room so that every `step` is a tick: the parent moves the robot (and reverts it on a collision), the FireGrid spreads once, the robot's base cell is checked, health is charged, counters advance, and a new observation is built that contains kinder's robot and lab features plus one FireCell object per grid cell and one status object. `FireLab3DEnv` is the gymnasium face: a 750-float vector over a fixed object list. `firelab/__init__.py` registers `firelab/FireLab3D-l1-v0` and applies the Windows shim before any env can be built. Seven tests cover the vector shape, the furniture blocking, seeding, spread, health drain, state restore, and the two end conditions.

**Where it lives and why.**
- *`types.py` separate from `env.py`.* The type table is what agents, Liane's program API and the trace converter import to read a state; none of them should import pybullet to do it.
- *Two env classes, as kinder does it.* kinder's own envs come as an inner object-centric env and a `ConstantObjectKinDEREnv` wrapper that fixes the object list and vectorises; prpl-tidybot's `KinderSimEnv` reaches the inner one through `env.unwrapped._object_centric_env` (sim_env.py:97-99) and calls `set_state` on it. Keeping their shape means their pipeline, their recorder and their shadow sim work on our env without a special case.
- *A copy of kinder's type table, not an edit.* `Kinematic3DEnvTypeFeatures` is a module-level dict kinder's other envs read; adding our types to it would change every kinder env's observation space in the same process. `dict(...)` plus two entries costs nothing and leaves kinder untouched.

**Decisions and alternatives.**
- *Fire cells as 240 small objects versus one object with 240 features versus an `info` array.* 240 objects keeps each feature list short and lets any state reader ask "the cells whose state is BURNING" with kinder's own `get_objects`, which is how `burning_cells()` is written; one object would need positional decoding and would break `create_markdown_description`'s one-row-per-feature table into 720 rows; an `info` array is invisible to everything that reads the state vector (the program API, the recorder, Liane's compiler). Cost: a 750-dimensional vector where PrplLab3D had 38. kinder vectorises in numpy, and the step rate did not visibly move.
- *Blocking by link AABB with a z cut, versus the collision mesh, versus a hand-drawn list.* The probe showed every floor-standing link has its box bottom below 0.3 m and every wall cabinet above 1.2 m, so one threshold separates them, and the URDF has no floor or wall link that would block the whole room (a half-room-area guard is kept against a future URDF). AABBs over-block at furniture edges (A58): a robot could stand on a cell we call blocked. The mesh would be exact but costs a point-in-mesh test per cell per link; a hand list would rot when the lab file changes.
- *Health charged from the base position, not the robot's footprint.* One lookup per tick, and it matches how the agents will reason (one cell under the robot). A footprint test would make the 0.5 m-wide base touch fire half a cell earlier; worth revisiting when the three conditions are checked.
- *`goal_reached` is False.* kinder's `step` ends an episode when the parent's goal holds (cubes on the counter, prpl3d.py:262-273), which is not our task. Level 1 ends by health or the tick cap; level 2 adds its own rule in chunk 3. Overriding the hook is cheaper than reimplementing the parent's step.
- *Seeding through gymnasium's generator.* `_reset_objects` hands `self.np_random` to the grid, so `reset(seed=k)` fixes the ignition cell and every spread draw along with kinder's cube placement. Liane's train, validation and test splits (chunk 5) are then lists of integers.
- *The parent `step`'s observation is thrown away.* Calling `super().step` keeps kinder's collision-revert and grasp logic intact at the price of one extra `_get_obs` per tick. Copying the parent's 150-line step to save that call would make us maintain kinder's physics rules; not worth it at 180 steps per second.
- *`max_action_mag` as the speed cap.* kinder uses one number to clip the base deltas (metres) and the joint deltas (radians) (base_env.py:523-540). Setting it to 0.05 gives the 0.5 m/s base speed of A57 and, as a side effect, a slow arm; irrelevant while picks are magic (A60), to be split into two caps if real grasps arrive.
- *Line budget.* 370 raw, 251 code lines by the same ast count as chunk 1; one over the 250 on the code count, well over if docstrings count. If the budget is raw lines, the tests (115) would move to a chunk of their own next time.

## Chunk 3a: the gate check (Nov 22)

**What it does.** `firelab/gate.py` answers one question about a set of task rules before they are built in kinder: with the clock off, do random, the reflex and an agent with unlimited time separate (PLAN.md 0a, A65)? It models the task on the bare fire grid: the robot is a cell, a step to a neighbouring cell costs 4 ticks, a pick 10, a drop 5; items sit in cells, lose 2 % of their value per tick once their cell burns, and count as saved when dropped in the safe strip; health drops per tick in a burning cell; the episode ends at 50 % coverage, 500 ticks or health 0. Three agents play the same worlds: `random_agent` (uniform over go-to-item, go-home, pick, drop, wait), `reflex` (most valuable item first, never enters fire, one item at a time) and `planner` (tries every order of rescues and every cross-the-fire choice, knowing the future of the fire). `main` prints, per rule set, the three mean scores over 25 seeds, the planner-minus-reflex gap with a paired bootstrap interval, and a verdict against the A65 thresholds.

**Where it lives.** `firelab/gate.py` (288 lines, 218 of code), `firelab/test_gate.py` (65, 41 of code), both pure Python on top of `firelab/fire.py`; no kinder import of its own. Log: `logs/gate_nov22.log`.

**Why this way, against the alternatives.**
- *The fire is computed once per world, not stepped during play.* The fire never depends on the robot, so `make_world` runs FireGrid to the end and stores the tick at which each cell ignites. Every question an agent asks (is this cell burning at tick t, what is this item worth at tick t) is then arithmetic, and the planner can try thousands of rescue orders in seconds. Alternative: step a FireGrid inside every trial, which is the same answer about 100 times slower. This stops being valid the day an agent can change the fire (extinguisher, fire door); then the search has to carry the fire state.
- *The planner knows the future.* That makes it an upper bound, not a stand-in for an LLM or a human: if even it cannot beat the reflex by the margin, nothing can, and the rules are rejected without spending a token. If it passes, condition 2 still has to be shown with a strong LLM and CR playing. Alternative: a planner that samples possible fires, which is the more honest proxy and the next thing to add if a rule set passes the bound.
- *The reflex is kept as it was in HAZARD and RRARA* (most valuable first; CR, Nov 22), plus "never step into fire" and re-routing after every cell. It is not tuned per rule set and not replaced by a best-of-several-rules family.
- *Random has no "done" command.* Including it would end most random episodes at once and make condition 1 trivially true.
- *The grid model is cruder than the kinder env will be*: 4-neighbour moves, pick only from the item cell (not within 0.5 m), no arm. It is for rejecting rule sets, not for reporting scores (A66).

**What it found.** No rule set made of the existing knobs passes with room to spare; see NOTES.md Nov 22 and the A65 addendum.

**The reflex on the real robot (CR asked, Nov 22).** Read from prpl-tidybot, not measured: the real pipeline is ceiling cameras and ArUco markers into a marker-detector server (`src/prpl_tidybot/marker_detector/`, README "Launching the servers"), a perceiver that turns detections into the same object-centric state the sim gives (`real_sim/perceivers/kinematic3d.py`), and a 10 Hz control loop (`third_party/constants.py:48-49`). The real arm and camera interfaces for the lab env still raise NotImplementedError (README, the `prpl3d-o1 mode=real` example). So the same reflex rule can run there unchanged, because it needs only the object list, a value table keyed by object id and the skills. What changes around it: a safety layer first (stop or back off inside a margin of the hazard or an obstacle, every control period, whatever the planner is doing: Tom's "keep it alive while it compiles", Nov 17); margins instead of "not in a burning cell" (pose noise, stopping distance); skills that take seconds and sometimes fail, so the rule needs a retry-or-abandon clause; and a fire that is virtual, an overlay in the state, since nothing burns in the lab.

## Chunk 3b: bag and exit, the two planners, and the grid (Nov 22)

**What changed.** `firelab/gate.py` gained five knobs on `Task` (values, pick_ticks, drain, end_coverage, bag), a bag (any number of items, banked when the robot drops in the safe strip alive and in time; with the bag off the old one-item-per-trip rule), one search `plan()` used by two planners, a `forecast()`, a diagnostic reflex, and `evaluate()`. `firelab/gate_grid.py` runs every combination of the knobs and writes one CSV row per rule set. `scripts/gate_exit_rules.py` is a diagnostic on the reflex exit rule.

**Where it lives.** `firelab/gate.py` (343 lines, 254 of code), `firelab/gate_grid.py` (106, 87), `firelab/test_gate.py` (101, 67), `scripts/gate_exit_rules.py` (72). Results in `results/firelab_gate/grid.csv`.

**Why this way, against the alternatives.**
- *One fire per seed for every rule set.* `fire_history` is cached on (seed, walls, p, cap) and items come from their own random stream, so two rule sets differ only in the rules, and gaps are paired by seed. Alternative: a fresh world per rule set, which adds fire-to-fire noise to every comparison.
- *Two planners from one search.* `planner_all` runs `plan()` on the true future; `planner_now` runs the same `plan()` on a forecast built from the fire as it is now, does only the first action, and replans. The forecast is the simplest that uses the rules: each cell ignites 30 ticks per cell of distance from the burning cells (the measured median front speed). Alternative: sampling many possible fires and planning against the average, which is closer to what a careful agent would do and many times the cost; worth it only for finalists.
- *Pruning by dominance instead of a different algorithm.* With a bag the orders of 8 items are 8! before choices of when to go home. `plan()` keeps, per (items taken, place, load count), the visits seen so far and drops a new visit that is no better in time, health, load value and banked value. It stays exact because fire only grows and values only fall, so arriving earlier with more is never worse. Alternative: a forward dynamic programme over subsets, the same idea with more code; or Della, not needed (the whole grid is about 32 CPU-minutes).
- *The diagnostics are not agents of record.* The value-per-tick reflex and the alternative exit rules exist to test whether a gap is a property of the task or of one weak rule.

**What it found.** NOTES.md Nov 22 (late), A65 addendum 2, A68 addendum: the bag is what opens a gap; most of the largest gaps are the reflex exit rule; the candidate that survives is 8 items, bag, fast picks.

## Chunk 3c: buildable layouts, robot width, exit rules and the clock (Nov 22)

**What changed.** `firelab/gate.py`: the `walls` switch became `layout` (four layouts in `LAYOUTS`) plus `clearance`; `World.open` holds the cells the robot may stand in and `route` uses it; `reflex` takes its exit rule as arguments and `EXIT_RULES` names four of them, with "bank after every pick" as the official one; `planner_now` takes a `charge` in ticks per decision; `unreachable` counts items no route can reach; `evaluate` returns every agent's score for every seed. New `firelab/gate_clock.py` runs layout variants x rule sets x seeds in chunks of 10 seeds and writes one CSV row per room.

**Where it lives.** `firelab/gate.py` (388 lines, 287 of code), `firelab/gate_clock.py` (115, 96), `firelab/test_gate.py` (125, 84). Results: `results/firelab_gate/clock_seeds_fast.csv`, `clock_seeds_slow.csv`.

**Why this way, against the alternatives.**
- *Robot width as a set of enterable cells, not a wider robot.* The base is 0.51 m x 0.55 m, so its centre cannot come within about a cell of an obstacle. Removing every free cell that touches a wall or furniture cell from `World.open` gives that with one set and no change to the search. The fire keeps using the real walls (`World.burn`). Alternative: a robot occupying 3 x 3 cells, which needs footprint checks in every move and in the health rule.
- *The clock as ticks spent standing before each action.* It is RRARA's rule (decision time becomes world time with the body still) with the cost as a knob instead of a measured wall time, so one planner can be run at several costs. The planner knows its cost and plans from where it will be when the thinking ends; otherwise it would be handicapped twice, by the wait and by a stale plan. Alternative not built: thinking once for several actions, or thinking while moving, which is Tom's fast-and-slow idea and the next thing to test in this model if we stay in it.
- *One row per room.* The report is recomputed from the per-room CSV, so a new column or a different grouping needs no rerun, and single rooms can be inspected.

**What it found.** NOTES.md Nov 22 (23:30): the gap survives 0.8 m doorways and the robot-width rule (+0.12 to +0.23); the planner loses to the reflex somewhere between 0.2 s and 2 s per decision; slowing the fire makes 2 s decisions pay only by shrinking the gap.

## Chunk 3d: reach, plan-and-commit, think-while-moving (Nov 23)

**What changed.** In `firelab/gate.py`: a `reach` knob and `within_reach`, so an item can be picked from a neighbouring cell; `walk` and `act` take an `until` tick; the reflex loop moved into `run_reflex`, which can start from any state and stop at a tick; `plan` returns the whole list of actions; `stand`, `reflex_until` and `thinker` replace the body of `planner_now`. New `firelab/gate_agents.py` runs the three thinking agents at six costs on the chosen layout and writes one row per room.

**Where it lives.** `firelab/gate.py` (changed functions only, about 60 new lines), `firelab/gate_agents.py` (new, 96 lines), `firelab/test_gate.py` (3 new tests, 35 lines). Results: `results/firelab_gate/agents_seeds.csv`.

**Why this way, against the alternatives.**
- *One loop for three agents.* `thinker` has two switches. `commit` decides how much of the plan is carried out before the next thought (one action or all of it). `moving` decides what the body does during a thought (`stand` or `reflex_until`). The same body function is run twice per thought: once on the forecast, to predict where the robot will be when the thought ends, and once on the real world. Alternative: three separate functions, which would repeat the forecast-plan-act loop three times and let the agents drift apart in small ways.
- *The reflex during a thought is the official reflex, unchanged.* `run_reflex` is the old loop with a stop tick, so agent B's body does exactly what the reflex agent would do. Alternative: a simpler stand-in such as "walk toward the safe strip", which would make B's result depend on a second, invented rule.
- *"Blocked" means an action cannot be carried out.* A committed plan is a list of actions; walking re-routes around new fire without a thought, as the reflex's walking does, and a thought is spent only when no fire-free route is left, the item is gone, or the plan has run out. Alternative: storing the planned path cell by cell and re-thinking when the next planned cell burns, which is closer to CR's wording ("when the next step would enter fire") and thinks more often; noted as A70.
- *Reach as a goal test.* `within_reach(w, k)` returns the test "is this cell close enough to item k", and walking, picking, the diagnostic reflex and the unreachable count all use it. Alternative CR offered: never place items next to walls, which changes where items can be and hides the constraint the real robot has.

**What it found.** NOTES.md Nov 23: unreachable items are 0; always-deliberate loses at 2 s per thought; plan-and-commit still wins at 2 s; think-while-moving still wins at 10 s on means. All four rule sets have been run (6-item at 00:35 on battery, 8-item at 02:05 on mains); the pattern is the same in all four.

## Chunk 3e: the level-2 room in kinder (Nov 23)

**What it is.** The room the rule check settled on, built in the simulator: one partition across the room at y = -0.4 with one 0.8 m doorway, the base stopped by partition, furniture and the edge of the floor, a safe strip (y >= 0.7), and a fire that starts and spreads exactly as in room `seed` of the rule check. No items, bag or score yet (chunk 3f); the `l2` gymnasium id is registered there, when the task is whole.

**Where it lives.** `firelab/env.py` (62 new lines): three settings on `FireLabConfig`, the constants `ONE_DOORWAY` and `LEVEL2`, and seven small methods on `ObjectCentricFireLab3DEnv`. `firelab/test_room.py` (93 lines, 9 tests). `scripts/kinder_probe_base_collisions.py` (40 lines, the measurement behind the fence). `scripts/firelab_room.py` (42 lines, the picture `results/firelab_show/level2_room.png`).

**Why this way, against the alternatives.**
- *The partition is a list of boxes in the config, and each box does two jobs.* `_build_partitions` turns a box into a pybullet body (what the base hits) and into BLOCKED cells (what the fire cannot enter). One list, so the wall the robot meets and the wall the fire meets cannot drift apart. Alternative: a second URDF for the room, which is more to maintain and gives the fire grid nothing.
- *A fence, not outer walls.* The probe showed the kinder room has no outer walls: the base drove 4 m off the floor. `_robot_or_held_object_collision_exists` now also answers "yes" when the base centre is off the grid, and kinder's own step then reverts the move (base_env.py:578-585). That is the rule check's confinement exactly. Alternative: four wall bodies, which need invented positions and would stop the 0.51 m base from centring on edge cells that the rule check let it use.
- *Base collisions on at every level.* One line in `__init__` (`check_base_collisions=True`); kinder then tests the base against the lab, the cubes and, through `_get_collision_object_ids`, the partition. Alternative: level 2 only, which leaves a level 1 where the robot drives through cabinets.
- *The seed names the room.* At level 2 the fire gets its own generator `default_rng(seed)` and picks its first cell the way `gate.fire_history` does, so a score in the simulator can be set next to the rule check's score for the same room. Alternative: gymnasium's shared generator, as at level 1; then kinder's own draws come first and no room would match.
- *The room is defined in env.py, not imported from gate.py.* The env should not depend on the planning model. A test (`test_floor_plan_is_the_rule_checks_one_doorway_room`) holds the two together instead.

**What the tests show.** Straight back from home the base is stopped by the wall (its edge is at x = 0.555, the doorway ends at 0.5); 0.2 m to the left it passes and reaches y = -1.2. The fence holds at x = -1.5. The cabinets stop it near y = 1.05. Rooms 0 to 3 have the same burning cells at tick 0 and tick 60 as the rule check.

**Open (A71 g).** With base collisions on, the base collides with cubes on the floor; in the rule check the robot could cross an item's cell. To settle before chunk 3f.

## Chunk 3f-1: the item rules, no simulator (Nov 23)

**What it is.** `firelab/items.py`: where each item is (floor, bag, saved, lost), what it is worth, and the one pick or drop the robot is in the middle of. It knows only grid cells and ticks, like `fire.py`. Chunk 3f-2 will call it once per tick from the env.

**Where it lives.** `firelab/items.py` (103 lines): `ItemRules` (the settings; the defaults are CR's main rule set) and `Items` with `place`, `burn`, `in_reach`, `start_pick`, `start_drop`, `tick`, `total`, `score`. `firelab/test_items.py` (104 lines, 19 tests).

**Why this way, against the alternatives.**
- *Our own items, not kinder's cubes.* kinder's grasp holds one object and needs the arm to touch it (base_env.py:595-621). The task has a bag and a magic pick, so none of that code fits. Alternative: subclass around kinder's `_grasped_object`, which would fight it at every step.
- *A file with no simulator.* The rules can be tested against the rule check in four seconds, with no room to build, and CR can read them without knowing pybullet. Alternative: methods on the env, as the room chunk did; fine for ten lines, not for the task's scoring.
- *A pick or drop is a job with a countdown.* `start_pick` records ("pick", k) and 10 ticks; `tick` counts down once per env step and applies the job on its last tick. One env step stays one tick (D2). Alternative: one env step that silently runs 10 ticks, as the rule check's `spend` does; simpler, but then a step would no longer mean a tick, and the clock chunk charges time in ticks.
- *Value is stored and decremented, not computed from the tick the cell caught fire.* The current value is then part of the state, so kinder's set_state (which prpl-tidybot uses to carry out a skill) restores it with nothing hidden. The tests check that the decrement gives the rule check's formula.
- *Unnamed pick = the most valuable item in reach.* The rule check took the lowest-numbered one (gate.py:215), which only mattered to its random agent. A named item is supported for the skills chunk.

**What the tests show.** For rooms 0 to 4: the same cells and values as `gate.make_world`; the same values as `gate.value` at ticks 50, 150 and 300; and for each of the 6 items, a pick begun while the item is already burning followed by a drop gives the same held value and the same score as `gate.pick` and `gate.drop`.

## Chunk 3f-2: the items in the room (Nov 23)

**What it is.** Level 2 as a whole task in the simulator: the items of `firelab/items.py` are placed in the room, shown in the state vector, picked and dropped through the gripper signal, and scored. `firelab/FireLab3D-l2-v0` is registered with the main rule set; `num_items=` and `values=` change it.

**Where it lives.** `firelab/env.py` (79 new lines: three settings, `level_config`, `_show_items`, and additions to `__init__`, `_reset_objects`, `_get_obs`, `_set_object_states`, `step`, the wrapper). `firelab/types.py` (11 lines: `ItemType`, `item_name`, three job features on the status). `firelab/__init__.py` (the id). `firelab/test_level2.py` (109 lines, 9 tests).

**Why this way, against the alternatives.**
- *Pick and drop ride on the gripper number kinder already has.* The last number of every action is the gripper: below -0.5 is "close", above 0.5 is "open" (base_env.py:588-593). "Close" starts a pick, "open" starts a drop. Every agent, including a random one, can therefore do everything through the ordinary `step(action)`. Alternative: separate methods `env.pick(k)` and `env.drop()`; clearer to call, but invisible to gymnasium wrappers and to prpl-tidybot's sim pipeline.
- *The robot is frozen during a job by zeroing the action.* One line in `step`; kinder then moves nothing. Alternative: let it move and cancel the job, which adds a rule the rule check never had.
- *Each item is an object in the state with five numbers* (x, y, value, start, where), and the job is three numbers on the status object. Everything an agent needs is in the vector, and kinder's set_state restores it (tested mid-pick). Alternative: `info` only, which prpl-tidybot's skill execution by set_state would lose.
- *Each item also has a small blue box in pybullet.* It is there to be seen, and to be bumped into if `items_block_base` is switched on. With the switch off the boxes are simply left out of the list kinder checks, so the base passes over them.
- *Order inside one step:* start a job, move, fire spreads, health, then the items burn (using the cells that were on fire when the tick began, which gives the rule check's formula), then the job counts down. A drop that ends on the tick the episode ends banks nothing, as in the rule check.

**What the tests show.** Rooms 0 to 3 have the rule check's items. A pick takes exactly 10 steps and pushing during it moves nothing; a drop outside the safe strip is refused; a drop inside takes 5 steps and the reward on the fifth is the item's value. With a robot that cannot die standing still, rooms 0 and 1 end at exactly the rule check's tick.

**Not done yet.** No agent can walk to an item: that needs the skills chunk (routes around fire, walls and furniture). The first run of random and the reflex comes after it.

## Chunk 4: skills (Nov 23)

**What it is.** `firelab/skills.py`: the things a robot in the level-2 room can be told to do. `walk` to a goal, `get` item k, go `home` and bank, `wait`. Each one steps the env tick by tick and returns `Result(ticks, done, reason)`. Random, the reflex and every later agent act only through these.

**Where it lives.** `firelab/skills.py` (150 lines): `Result` and `Skills` with `cell`, `route`, `walk`, `wait`, `get`, `home` and three helpers (`_step`, `_finish`, `_drive`). `firelab/test_skills.py` (92 lines, 7 tests). `scripts/kinder_probe_clear_cells.py` (34 lines, the measurement behind the walkable cells). 276 new lines in all, 26 over the 250 limit: the probe was not in my estimate.

**Why this way, against the alternatives.**
- *The walker uses the rule check's rule for where the base may stand, after measuring it.* The probe put the base on every free cell centre and asked kinder whether it collides: 161 cells are free, exactly the 161 the rule check allowed (no cell around or under the base is a wall or furniture). So the rule is kept as one set built at start-up. Alternative: ask kinder about every candidate cell while planning, which is slow and gives the same answer.
- *Routes are the rule check's breadth-first search over cells*, with the same neighbour order, so ties are broken the same way. A route keeps out of cells burning now unless told to cross. Alternative: A* over metres with the base's real shape; more faithful to a real planner, but then a difference from the rule check could come from the route finder and not from the simulator.
- *The route is found again after every cell.* The fire moves while the robot walks; this is what the rule check's robot without foresight did.
- *Every skill reports what happened, not what was asked.* `done` is true only if the robot is at the goal, the item is in the bag, or what was carried is now saved. A move kinder reverts is reported as "blocked". This is the lesson of HAZARD's silent failures (A32).
- *`get` names its item by calling `items.start_pick(cell, k)`*; a gripper "close" through `step` would take the most valuable item in reach, which is not always the one asked for. `home` uses the ordinary gripper "open".

**What the tests show.** In a room with the fire frozen: 6 cells through the doorway take exactly 24 ticks; fire in the doorway gives "no route", and with `cross=True` the robot goes through and loses health; `get` then `home` costs 26 and 17 ticks and scores 3/25; an empty-bag `home`, a second `get` of the same item, a drive into the wall and anything after the episode's end all report failure.

## Chunk 5a: random and the reflex as agents (Nov 23)

**What it is.** `firelab/agents.py`: `reflex` and `random_agent`, copied from the rule check, written against a "body" (anything with the skills and a few look-ups). Two bodies exist: `Skills` (the robot in the simulator) and `GateSkills` (the robot on the rule check's board). The runner and the first run are chunk 5b.

**Where it lives.** `firelab/agents.py` (120 lines). `firelab/skills.py` (+49 lines: `pick` and `drop` split out of `get` and `home`; the goals `near` and `safe`; the look-ups `count`, `live`, `value`, `carrying`, `item_cell`, `score`; a counter of time spent inside `env.step`). `firelab/test_agents.py` (62 lines, 13 tests). 231 new lines.

**Why this way, against the alternatives.**
- *One agent, two worlds.* CR asked for the simulator's score beside the rule check's for every room, with the differences explained. If the reflex in the simulator were a second piece of code, a difference could be the code. With one `reflex` run on two bodies, a difference can only be the world, and the two logs show the first command where they part. Alternative: call `gate.reflex` for the rule check's number and a separate simulator reflex; simpler, but it cannot say why two scores differ.
- *`GateSkills` hands every call to the existing function in `gate.py`* and changes nothing there; the rule check stays closed. A test holds it to the originals: on rooms 0 to 9 the copies score exactly what `gate.reflex` and `gate.random_agent` score.
- *Agents look only through the body's look-ups* (`live`, `value`, `carrying`, `cell`, `item_cell`), never into the env. That is the list of things a real-robot version must provide.
- *Each agent returns a log*, one line per command: command, item, tick when it ended, Result. It is what the comparison reads, and later what the clock charges against.
- *`Skills` counts the seconds spent inside `env.step`.* A runner can subtract it from the wall time of an episode to get the agent's own thinking time (CLAUDE.md rule 5, latency from day one).

**What the tests show.** With the fire frozen, the reflex in the simulator and on the rule check's board give the same commands on the same items with the same outcomes and the same score (0.875 in room 2, 0.774 in room 5). The simulator is exactly 2 ticks ahead throughout, because its first step from home is half a cell (A74 c).

## Chunk 5b: the runner (Nov 23)

**What it is.** `firelab/run_rooms.py`: plays each room with random and the reflex in the simulator, plays the same room with the same agent on the rule check's board, and writes one CSV row per room and agent. `--report-only` reprints the table from the CSV.

**Where it lives.** `firelab/run_rooms.py` (122 lines): `AGENTS`, `parted`, `play`, `report`, `main`. `firelab/test_run_rooms.py` (34 lines, 2 tests). Output: `results/firelab_sim/first_run.csv`.

**Why this way, against the alternatives.**
- *Both worlds in one row.* The comparison CR asked for is per room, so the pair is made at the moment of the run and cannot be mismatched later. Alternative: read the rule check's old CSVs; those were made with 6 items or without the pick-from-neighbour rule, so they are not the current numbers.
- *`parted` finds the first command where the two logs differ* in command, item or outcome (ticks alone do not count: the simulator is known to run 2 ticks ahead). The row stores that command from both sides, which is the starting point for saying why a room's scores differ.
- *One room is built once and reset per episode*, in one process. At about 5 s an episode there is nothing to gain from workers for 100 episodes, and one process keeps the timing honest.
- *Thinking time is measured, not charged.* Each row has the agent's own seconds (episode wall time minus time inside `env.step`) and what that would be in ticks at 10 ticks per second. The clock chunk will charge it; this run only records it.
- *Resume by (room, agent).* A stopped run continues where it was; rows already in the CSV are not replayed.

## Chunk 6a: the planner for the simulator (Nov 24)

**What it is.** `firelab/planner.py`: `plan`, the rule check's exhaustive search with a count of its search steps, and `board_from`, which builds the rule check's board from the simulator's present state. Both bodies gain `believe()`, which returns that board: in the simulator through `board_from`, on the rule check's own board through `gate.forecast`.

**Where it lives.** `firelab/planner.py` (73 lines), `firelab/test_planner.py` (61 lines, 14 tests), 3 lines each in `firelab/skills.py` and `firelab/agents.py`. 140 new lines. Also written now, outside the chunk: `scripts/della_firelab_setup.sh` (38 lines) and `scripts/sync_to_clone.sh` (17 lines).

**Why this way, against the alternatives.**
- *The search is a copy, not an edit.* CR's decision 1: `gate.py` stays byte for byte what produced the Nov 22 to 23 numbers. The copy differs by one line (`steps += 1`); tests hold it to `gate.plan` on rooms 0 to 9 at 6 items and room 0 at 8.
- *A search step is one state expanded* (one call of `search`). It is what the declared cost will charge for. At 8 items the first plan takes 586 to 20 738 steps over rooms 0 to 5, and a step costs 0.16 to 0.35 ms on the laptop, so steps track seconds closely enough to stand in for them.
- *The board is rebuilt from the state the robot sees, not carried over from the rule check.* Burning cells come from the fire grid. How long an item has burned is not in the state, so it is read off the value the item has lost (value now against value at reset, 2 % per tick). The future fire is `gate.forecast`, unchanged. Alternative: let the planner read the simulator's true future fire; that is the all-knowing planner, which no real agent can be.
- *Checked equal, not assumed equal.* With the robot standing still in both worlds, the simulator's board and the rule check's forecast are compared at ticks 0, 60 and 120 in rooms 0 to 2: same cells burning, same forecast for every other cell, same end tick, same item values now and later, same plan and the same number of search steps.

## Chunk 6b: the clock and the three thinking agents (Nov 24)

**What it is.** `firelab/clock.py`: what a thought costs, two ways (measured seconds; declared = search steps times a fixed number of seconds), both recorded for every thought, one of them charged. `firelab/thinkers.py`: the three planning agents as one loop with two switches, a port of `gate.thinker` onto a body, and `seen`, the record of what the agent knew when a thought began.

**Where it lives.** `firelab/clock.py` (59 lines), `firelab/thinkers.py` (73), `firelab/test_thinkers.py` (76, 11 tests), a deadline tick on `walk`, `get`, `home` in `firelab/skills.py` (+8) and on `reflex` and `GateSkills` in `firelab/agents.py` (+4). 220 new lines.

**Why this way, against the alternatives.**
- *One loop for three agents*, as in the rule check: `commit` decides how much of a plan is carried out before the next thought, `moving` decides what the body does during a thought. On the rule check's board each agent scores exactly what `gate.thinker` scores at charges 0 and 20 in rooms 0 to 3 (tested), so the port adds nothing of its own.
- *The estimate* (CR's decision 2). Before a thought the agent says how many ticks it expects to think and plans from the state it predicts for that moment: standing, or the reflex run forward on its believed board. Then the true cost is charged. With the clock off or fixed the estimate is exact; otherwise it is the timing table for the first thought and the last charge afterwards. Estimate and charge are both in the record.
- *"The reflex acts while the plan is made" is done in order*: the plan is made and timed, then the world is advanced by the charged ticks with the reflex acting. The fire does not depend on the planner and the planner's input was fixed when the thought began, so this equals running them side by side, and it is repeatable. Alternative: real threads in real time, which is not repeatable.
- *A deadline on the skills, not a second reflex.* `reflex(body, until=...)` is the same reflex; `walk` stops at the cell it is on when the clock reaches the deadline, and a pick or drop that has begun is finished (A70 c).
- *The saved game is built from the believed board*, the exact input the planner had: tick, cell, health, bag, every item with value now and at reset, the burning cells, the forecast end. Alternative: the raw state vector, which is 800 numbers and not what the planner reads.

## Chunk 6c: the runner for any agent and clock (Nov 24)

**What it is.** `firelab/run_rooms.py` now plays any of random, reflex, always, commit, moving under any clock; writes each run into its own folder (`results/firelab_sim/<tag>/`), one rows file per invocation and one game file per episode; and times one fixed reference plan at the start so every result names the machine's speed. `scripts/della_firelab_rooms.slurm` runs one room per array task on one core.

**Where it lives.** `firelab/run_rooms.py` (185 lines, was 122), `firelab/test_run_rooms.py` (50, was 34), `scripts/della_firelab_rooms.slurm` (29). 108 new lines.

**Why this way, against the alternatives.**
- *A folder per run with a rows file per invocation.* Fifty Della tasks can write at once without sharing a file; the report reads every rows file in the folder. Alternative: one CSV with locking.
- *One game file per episode*, JSON lines: the episode (rule set, machine, reference plan), every thought (seen, estimate, plan, both costs, ticks charged), every command with its result, the outcome. This is what CR asked to keep for Liane's method.
- *The rule check's board stays optional* (`--no-board`). Under the clock off, fixed or declared the board run is exact and comparable; it doubles the planning time. Under the measured clock it is not comparable and should be off.
