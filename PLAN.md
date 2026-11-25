# PLAN: the fire task on the lab simulator

Written Nov 22, 2026 from Liane's Nov 19 call, the Nov 10 and Nov 17 meetings, and a read of HAZARD, kindergarden (commit 86c55a4) and prpl-tidybot (main). Each chunk is at most 250 lines of new code, tests included. Cloning or installing does not count. After every chunk: a DESIGN.md section, a file-by-file walkthrough, then stop for CR.

## 0. What we are building, in one paragraph

A fire task in the PRPL lab room as kinder simulates it, in levels. Level 1: a fire starts in one random floor cell and spreads; the robot must stay out of it. Level 2: valuables lie on the floor, fire drains their value and the robot's health, the robot collects what it can and reaches the safe zone before the fire covers half the room. Level 3 (later): an extinguisher. Every agent runs under RRARA's clock (thinking time is charged as world time) with a switch to turn it off. Random, reflex and an unlimited-time agent are run first to check the three conditions. The README lets Liane install it and run her ablations without us.

## 0a. Why we left HAZARD, and the gate the new task must pass (added Nov 22)

Sources: Granola notes and transcripts of Nov 10 (Tom and Liane; Karthik), Nov 17 (Tom and Liane) and Nov 19 (Liane); `results/PLAN_NOV19.md` sections 2 and 3. Transcripts carry no speaker names; attributions are from context.

Why we left. HAZARD could not show that thinking is worth anything, so no result on it answered the question.
- A hand-written reflex (walk to the most valuable visible object) scored 0.76 on the paper's version, above every slower agent, and random was close behind. If the fast rule already wins, there is never a reason to run the slow solver.
- CR played it by hand with the clock off: 0.22 on the current version against the reflex at 0.20. Causes she named on Nov 17: the walk covers about 1 m per 100 frames; a failed walk returns the same message whether a wall blocked it or it only ran out of steps; distances are clipped so no consistent map can be built; look-around turns a fixed angle.
- Fixes behind flags lifted the agents above random, not above the reflex.
- Nov 17 decision: set HAZARD aside (return only if a focused week is worth it), no Unity for the restart, build one hazard on the lab's simulated lab room. Liane's reasons for that room: the people who built it are in the lab, the tidybot repo has motion-planning tools, and the room exists for real. Karthik, Nov 10: get the environment right before any method, do not treat a benchmark as fixed, play it yourself.
- CR, Nov 22: the third reason for the lab room over another Unity world is eventual expansion to the real robot (A67).

Gate check result (Nov 22, `firelab/gate.py`, NOTES.md): random passes with 6 or more items; the reflex stays within 0.06 to 0.12 of a planner that knows the future on every rule set tried but one (8 items, walls, 2 HP per tick: +0.19). The reflex is kept as in HAZARD and RRARA (CR, Nov 22). The rules need a new mechanic before level 2 is built; CR to choose (A65 addendum).

The gate. CR's three conditions (Nov 17), all measured with the clock off, on level 2:
1. Random does badly.
2. The reflex does clearly worse than an agent with unlimited time: a very good LLM with the full rules in context, or a human.
3. So the reflex is not near the ceiling.
Level 1 cannot pass (an agent that stands still far from the fire scores 1.0, and "do not step in fire" is the whole task; Liane, Nov 19). It is scaffolding for level 2.

Risk in this plan as written. Nothing in D6 guarantees condition 2. `results/PLAN_NOV19.md` section 4 expected the reflex to fail because furniture makes choke points and the spread cuts off routes; the room as measured in chunk 2 has none (19 of 240 cells blocked, all along the far wall). On an open 3.2 m x 3.0 m floor with three items, nearest-first may be close to the best order.

Proposed changes, each needs CR's yes (A65):
- Move the gate before the clock. After chunk 4 (skills): random, reflex and an exhaustive planner that tries every rescue order and every cross-the-fire choice with the clock off. The planner is an upper bound that costs CPU seconds and no LLM calls. If planner minus reflex is small, change the task (choke points, more items than time allows, value spread, item drain) before anything else is built.
- Condition 2 proper is then run with a strong LLM, clock off, rules in context, and CR playing through a text page; neither is deferred behind "if the first two hold".
- Thresholds are written down before the runs (A65), not chosen after.

## 1. Decisions the plan rests on

Each is also an entry for ASSUMPTIONS.md (A56 onward) when its chunk lands.

D1. Where the code lives: a package `firelab/` inside our repo (thinking-fast-hazard), not a fork of kindergarden and not prpl-tidybot. It subclasses kinder's PrplLab3D env and registers its own gymnasium ids (`firelab/FireLab3D-l1-v0`, `-l2-v0`). Liane installs kindergarden at the pinned commit plus our package. prpl-tidybot needs no change: its KinderSimEnv calls `gymnasium.make(env_id)`, and gymnasium's `module:namespace/Name-v0` id form imports our package first (verified in chunk 7). Alternatives and why not: a kindergarden fork means their CI, their review and a commit pin dance for every change; prpl-tidybot is the real-robot repo and the sim env is not its job.

D2. The tick. kinder's kinematic3d backend has no clock (grep in the Nov 22 reply: only `render_fps`, used as video metadata). We define one env step = one tick = 0.1 s, the real TidyBot's control period (prpl-tidybot `third_party/constants.py:48-49`, POLICY_CONTROL_FREQ = 10). Every rate below is per tick. Robot speed: the base may move at most 0.05 m per tick (0.5 m/s), overriding kinder's 0.4 m default, so a tick means the same thing for the body and the fire.

D3. The clock. RRARA's Time Conversion Mechanism: a decision that took t wall seconds is charged ceil(t / 0.1) ticks during which the env steps with the zero action (fire advances, robot stands). This is our HAZARD TimedAgent with a tick instead of a frame and 10 instead of 30. Clock off = charge 0. HAZARD as shipped has no clock; RRARA added it; so do we.

D4. Fire model, level 1: a grid of 0.2 m cells (HAZARD's FIRE_SPREAD_SIZE) over the floor box [-1.5, 1.7] x [-1.5, 1.5] (tidybot's preview floor_bounds), 16 x 15 = 240 cells. One random cell ignites at reset. Each tick, each burning cell ignites each of its 4 neighbours with probability p. Default p = 0.015, chosen so a neighbour catches after about 67 ticks (6.7 s), the same order as HAZARD's box-to-box time (grow 140 frames, then 2 % per 10-frame check rising to certainty by 250 frames; fire.py:146-148, 183-195). Cells blocked by furniture never burn. The HAZARD-faithful grow-then-spread rule with the CHOUKA ramp is an optional second rule, chunk 8. Note the difference from the call: there CR said p = 0.1 per neighbour per tick, which at a 0.1 s tick would burn a room in seconds; p is a config value and the chunk 1 script measures time-to-50 % so CR can pick it.

D5. Fire cells in the observation. kinder observations are a flat vector over a fixed list of objects, built from per-type feature lists (relational_structs). The 240 cells are objects of a new type FireCell with features x, y, state (0 normal, 1 burning, 2 burnt, 3 blocked). This keeps Liane's object-centric tooling and prpl-tidybot's devectorize working unchanged. Internally the env keeps a numpy grid; the objects are a view of it. Alternative rejected: a hazard array in `info`, invisible to anything that reads the state vector.

D6. Level 2 mechanics. Robot health 100, minus 0.15 per tick while its base is in a burning cell (CR's 1.5 HP per second). N valuables (kinder cuboids, value 5 or 1 as HAZARD's value.json splits them); a valuable in a burning cell loses 2 % of its value per tick; a valuable carried out to the safe zone is saved at its remaining value. Safe zone: the strip y > 0.6 in front of the counter (robot home is (0.3, 0.0)). Episode ends when fire covers 50 % of burnable cells, health reaches 0, 500 ticks pass (HAZARD's 1500 frames / 30 x 10), or the robot declares done in the safe zone. Score = value saved / value present at reset, in [0, 1]. HAZARD scores a saved object at half value if it ever burned (challenge.py:205-209); we drain continuously instead, because that is what gives a thinking agent something to trade.

D7. Pick and drop are magic at first. kinder's grasp needs the end-effector marker to touch exactly one object, which needs arm motion planning (test_prpl3d.py steps 2-5, dozens of joint steps per pick). prpl-tidybot already carries out a SkillCall "by teleporting to the state its option model predicted" (sim_env.py:66-71), so the lab's own pipeline accepts magic skills. Our pick: robot base within 0.5 m of the object, 10 ticks, object attached; drop: 5 ticks. Arm planning is a later level, after the three conditions pass.

D8. Where runs happen. Della CPUs for anything over one episode (Liane's advice, CR's rule). The laptop gets a uv venv on G with Python 3.10.14 (prpl-tidybot's floor; kindergarden allows 3.10 to 3.12) for smoke tests only.

## 2. Chunks, in order

Each chunk names: what it delivers, files, line budget, the test that proves it, and what CR should be able to see.

### Chunk 0: environments on disk (no new code)
- uv venv at `G:\iw-thinking-fast\venv_kinder` with Python 3.10.14; kindergarden installed editable from `G:\iw-thinking-fast\kindergarden` with `requirements/kinematic3d.txt` only (pybullet, pybullet_helpers, relational_structs, prpl_utils); no MuJoCo, no kinder-baselines, no prpl-tidybot yet.
- Prove: `tests/envs/kinematic3d/test_prpl3d.py::test_prpl_lab_3d_env` passes; a 200-step random rollout of `kinder/PrplLab3D-o1-v0` prints steps per second (Liane said "fast to render"; we measure).
- Della: the same in `/home/cr7998/iw-thinking-fast/venv_kinder` (the Della root CR chose Nov 8, NOTES.md line 592), commands pasted into NOTES.md. Needs CR's yes before anything runs there.
- Output: NOTES.md entry with exact commands and the measured step rate.

### Chunk 1: the fire grid, no simulator (about 120 lines + 80 tests)
- `firelab/fire.py`: class FireGrid(bounds, cell_size, p_spread, rng). Methods: reset(ignition_cell or random), step() applies D4 once, cell_of(x, y), is_burning(x, y), coverage(), block(cells) for furniture, state array. Also `firelab/measure_spread.py`: for 100 seeds, ticks until 50 % coverage, printed as a small table for p in {0.005, 0.01, 0.015, 0.03, 0.1}.
- `firelab/test_fire.py`: deterministic under a seed; p = 0 never spreads; p = 1 fills the 4-neighbourhood in one tick; blocked cells never burn; coverage counts only burnable cells.
- CR sees: the table that fixes p, written into ASSUMPTIONS.md as A57.

### Chunk 2: level 1 env (about 200 lines + 50 tests)
- `firelab/types.py`: FireCellType and its features; RobotStatus features (health) as an extra object `status`.
- `firelab/env.py`: ObjectCentricFireLab3DEnv(ObjectCentricPrplLab3DEnv) with FireLabConfig (tick_seconds, p_spread, cell_size, max_ticks, health, speed cap). Overrides: `_reset_objects` (ignite one cell, mark cells under the lab URDF's footprint blocked via pybullet AABB), `step` (parent step, then fire.step(), health update, termination), `_get_obs` (parent dict plus cell and status objects), `goal_reached` (survive to max_ticks for level 1). FireLab3DEnv gym wrapper with the markdown descriptions kinder requires. `firelab/__init__.py` registers `firelab/FireLab3D-l1-v0`.
- `firelab/test_env.py`: reset puts exactly one burning cell; zero action for 50 ticks grows coverage; standing in fire drains health; health 0 terminates; vectorize/devectorize round-trips.
- CR sees: `gymnasium.make("firelab/FireLab3D-l1-v0")` works; one rendered frame with burning cells drawn (pybullet debug boxes, red).

### Chunk 3: level 2 valuables and score (about 180 lines + 60 tests)
**Revised Nov 23 (CR: start the build).** The rule check (DESIGN chunks 3a to 3d) came first and fixed the rules (A65, A70, A71). The build is now split so each part stays under 250 new lines:
- **3e, the room:** partition with one 0.8 m doorway, base collisions on, the fence at the floor edge, the safe strip, the rule check's ignition. `firelab/env.py`, `firelab/test_room.py`.
- **3f, the items:** N items with values as settings (main rule set 6 items, values 1, 2, 3, 5, 8), 2 % per tick burning, magic pick from a neighbouring cell (10 ticks), bag, banking in the safe strip, score, end at 50 % coverage; registers `firelab/FireLab3D-l2-v0`.
- Then chunk 4 (skills), then a small runner with random and the reflex on a few rooms, compared with the rule check room by room.

The bullets below are the original text.
- `firelab/valuables.py`: spawn N cuboids in the floor box outside the ignition cell, values 5 or 1; per-tick drain in fire; saved flag when dropped in the safe zone; score().
- `firelab/env.py` additions: level 2 termination (50 % coverage, done-in-safe-zone), `info` carries score, coverage, health, ticks; `firelab/FireLab3D-l2-v0` registered with `-o3` and `-o5` variants.
- Tests: drain rate, saved value frozen, score bounds, termination at 50 %.
- CR sees: a scripted episode (walk to one item, back to the safe zone) scoring 1/3 or so, by hand.

### Chunk 4: skills, the action layer every agent uses (about 220 lines + 60 tests)
- `firelab/skills.py`: walk_to(x, y) as A* over the fire grid treating burning and blocked cells as walls (a flag lets a route cross fire, for agents that choose to take damage), executed one tick per 0.05 m step through env.step; pick(obj) and drop() as D7; wait(k); done(). Each skill returns (ticks_used, result, reason), the honest-result convention from HAZARD's A32 list.
- `firelab/test_skills.py`: path avoids a burning wall; path through fire only with the flag; pick fails beyond 0.5 m; durations as documented.
- CR sees: the measured duration table of every skill (the HAZARD "frames per action" lesson, measured not assumed).

### Chunk 5: the clock and the runner (about 200 lines + 50 tests)
- `firelab/clock.py`: TimedPolicy(inner, tick_seconds, mode on/off/fixed-k): times `act(obs)`, charges ceil(t / tick_seconds) zero-action ticks, logs one JSON line per decision (wall_s, ticks_charged, tick_before, tick_after, action).
- `firelab/run.py`: `python -m firelab.run --env l2-o3 --agent random --episodes 25 --clock on --seeds splits/test.json --out results/firelab/<name>`; one folder per episode with decisions.jsonl and eval.json; a summary.json with mean and sd. `firelab/splits.json`: 20 train, 5 validation, 25 test seeds (Liane's requirement).
- Tests: a 0.3 s sleeping agent is charged 3 ticks; clock off charges 0; the summary math.
- CR sees: a one-episode run folder she can read.

### Chunk 6: baselines and the three conditions (about 200 lines + 40 tests)
- `firelab/agents/random_agent.py`, `firelab/agents/reflex.py` (walk to the nearest unburnt valuable, pick, return to the safe zone, never enter fire, leave when coverage > 40 %), `firelab/agents/program.py` (Liane's Policy.act over an observation dict built from the state; the dict and the API from scripts/compiled/API.md, re-pointed).
- Runs (Della, after CR's yes): random x 25, reflex x 25, both clock on and off; a human sitting of 5 episodes clock off through a text page is the unlimited-time check (port of scripts/human later if the first two conditions hold).
- CR sees: the three-condition table. If random is high or the reflex is near the ceiling, we change p, N or the room before any LLM is run.

### Chunk 7: README for Liane and the prpl-tidybot hook (README plus about 40 lines)
- `firelab/README.md`: what is implemented, every constant and its source, how to install (two pip lines), how to run one episode, how to write an agent (the `act(obs)` contract), how the clock charges time, what the score means, the seed splits, known gaps.
- `firelab/conf/firelab-l2.yaml`: a prpl-tidybot env yaml whose sim pipeline is their KinderSimEnv with `env_id: firelab:firelab/FireLab3D-l2-o3-v0`; verified by running their `scripts/run_planner.py env=firelab-l2 mode=sim` from a checkout with our package installed. If the module-prefix id does not reach their make call, the fallback is a 20-line KinderSimEnv subclass in our package that imports firelab first.
- Then: ping Liane.

### Chunk 8 (optional, after Liane's first look): HAZARD-faithful spread and level 3
- `spread_rule: hazard` in FireGrid: grow for 14 updates, then the CHOUKA ramp, one child per direction. Extinguisher object: pick it, `extinguish(cell)` skill sets a 3 x 3 patch to burnt, 20 ticks.

## 3. What is deliberately not in this plan
- Arm motion planning for real grasps (D7). kinder has it (pybullet_helpers motion planning, see test_prpl3d.py); it comes after the three conditions pass.
- LLM agents (Jev, Haiku, honest text). Their adapters exist for HAZARD; the port is one chunk each, after chunk 6 says the task rewards thinking.
- Liane's compile loop itself. Her side; our side is the observation dict and `act(obs)` in chunk 6.
- Flood, wind, alarm, moving people. Same room, later tasks; the FireGrid class is written so a flood is a different spread rule and a different drain.

## 4. Open assumptions to write into ASSUMPTIONS.md as the chunks land
- A56 D1 package location and gymnasium id form.
- A57 tick = 0.1 s, speed cap 0.05 m per tick, p_spread from the chunk 1 table.
- A58 240-cell grid over tidybot's floor bounds; cells under the URDF footprint blocked by AABB, not by the real collision mesh.
- A59 level 2 rates: health 0.15 per tick, value 2 % per tick, 50 % coverage end, 500 ticks.
- A60 magic pick 0.5 m and 10 ticks, drop 5 ticks.
- A61 Liane's counter pick-and-place extension is in neither clone; we build on kindergarden 86c55a4 until she names a branch.
- A62 Python 3.10.14 venv on G (uv) and Della; prpl-tidybot itself not installed until chunk 7.

## 5. Order of the next three sessions
1. Chunk 0, then chunk 1 (pure Python, runs on the laptop, no simulator yet).
2. Chunks 2 and 3 (the env), smoke-tested on the laptop, then on Della.
3. Chunks 4 to 6, the Della runs, the three-condition table, then chunk 7 and the ping to Liane.

## 6. After the first run (written Nov 23 for CR's approval; no code yet)

State: level 2 runs in kinder; random 0.041 and the reflex 0.598 over rooms 0 to 49 agree with the rule check (A78). Missing for CR's second condition: an unlimited-time agent in the simulator. Missing for the research question: thinking that costs time.

### Step 0: time one plan on Della (CR runs it; before any Della setup)
`della_planner_timing/` holds copies of `firelab/fire.py` and `firelab/gate.py` with an empty `__init__.py` (so only numpy is needed), CR's `results/slides_nov23/time_planner.py` unchanged, and `time_on_della.sh`, which runs it on one core of a compute node through `srun` and prints the CPU model. About 2 minutes of compute. Output: median seconds per plan at 6, 8 and 10 items, to set beside the laptop's 0.02 s, 1.8 s and 6 to 24 s.

### Chunk 6: the planner in the simulator, under a clock (three parts, each under 250 lines)

**6a. `firelab/planner.py` (about 120 lines + 80 of tests).**
- `believe(body)`: builds the rule check's board from what the robot can see now: which cells burn, each item's value now and at the start, the robot's cell, health, bag and banked value. The future fire is the rule check's forecast (the front advances one cell per 30 ticks, `gate.forecast`).
- `plan(world, state)`: the rule check's exhaustive search, returning the best list of actions and the number of search steps it took. One search step = one state expanded (one call of `search`, gate.py:316).
- Tests: on rooms 0 to 9 at tick 0 the plan and its value equal `gate.plan` on `gate.forecast`; the step count is the same on every machine.
- **Decision 1 (CR):** the step counter needs a line inside the search. Either (a) a copy of the 25-line search lives in `planner.py`, held equal to `gate.plan` by a test, and `gate.py` stays byte for byte what produced the Nov 22 to 23 numbers (my suggestion), or (b) `gate.plan` gets an optional counter.

**6b. `firelab/clock.py` and the three thinking agents (about 130 lines + 70 of tests).**
- Cost of a thought, two ways, both logged for every thought whichever one is charged:
  - measured: real seconds of the whole decision (building the board and searching), ticks = ceil(seconds x 10);
  - declared: search steps x a fixed number of seconds per step, the same on every machine. For a language model later: a fixed number of seconds per call.
- Clock modes: `off` (charge 0: the unlimited-time agent of CR's second condition), `measured`, `declared`, `fixed k` (the rule check's charged number, kept so its table can be reproduced in the simulator).
- One line per thought in `thoughts.jsonl`: room, agent, tick before, measured seconds, search steps, declared seconds, ticks by each rule, ticks charged, the estimate (below), the actions chosen. This is CLAUDE.md rule 5.
- `thinker(body, clock, commit, moving)`: a port of `gate.thinker` (gate.py:375-397) onto the body. always = plan before every action, standing; commit = plan once and follow it, think again when an action cannot be carried out or the plan runs out (A70 b); moving = the reflex acts while the plan is made.
- "The reflex acts during the thought" is done in order, not with threads: the planner runs and is timed, then the world is advanced by that many ticks with the reflex acting. This is exact, because the fire does not depend on the planner and the planner's input was fixed when the thought began.
- Needs one addition to `skills.py`: a deadline tick on `walk`, `get`, `home`, so the reflex can be stopped when the thought ends (it finishes the pick or drop it is in, A70 c).
- **Decision 2 (CR), a problem the real clock creates.** In the rule check the planner knew its charge before it started, so it planned from where the robot would be when the thought ended (gate.py:387). With a real clock, and equally with a per-step charge, nobody knows how long the thought will take until it is over. Proposal: before each thought the agent states an estimate E in ticks (first thought: a constant per number of items from the timing table; afterwards: what its last thought cost). It plans from the state it predicts for now + E. Then the true cost n is charged. If n differs from E the plan starts from a slightly wrong state; it is followed anyway, and the first action that cannot be carried out starts a new thought. E and n are both logged. Alternatives: plan from the present state and ignore the thinking time (simpler, always a little late); or real threads in real time (not repeatable, and 50 s of wall time per episode).
- **Decision 3 (CR):** the seconds per search step for the declared cost is measured once on one named machine and then frozen. My suggestion: the Della compute node of step 0.

**6c. Runner and runs (about 80 lines + tests).** `run_rooms.py` gains the three planner agents and `--clock`. Runs, each announced first (what, where, how long):
1. Clock off, plan before every action, rooms 0 to 49: the unlimited-time number in the simulator, and with it CR's second and third conditions (the reflex is at 0.598; the rule check's planner was at 0.85 for this rule set).
2. The three agents under the declared clock, rooms 0 to 49.
3. The three agents under the measured clock, rooms 0 to 49, on Della only (Decision 4).

Not planned: the full ladder of fixed charges (0 to 100 ticks) in the simulator. One fixed charge can be run if CR wants a direct check against the rule check's table.

### Della
- **Decision 4 (CR):** measured-clock runs only on a Della compute node, one core per task, with the CPU model and a reference plan's time written into every result file. Clock-off and declared-clock runs give the same result on any machine, so they may run on the laptop when under 20 minutes. This replaces D8's "Della for anything over one episode".
- **Decision 5 (CR):** to reach Della the code must be in git. As in the HAZARD weeks, `firelab/`, the new scripts and the four documents are copied into the clone `thinking-fast-hazard/`, I commit there (no co-author line), CR pushes. Then one setup script for CR to run on Della: a Python 3.10 environment with kindergarden at 86c55a4 and pybullet (CPU only, no display), ending in the test suite.

### Chunk 7: the interface every other agent sits on (plan only; no code until CR says)
One file family, used by RRARA, the language-model agents and Liane's compiled programs.
- **7a, the observation dictionary and the commands** (`firelab/interface.py`, about 150 lines). `observe(body)` returns one dictionary: tick and ticks left; health; share of floor burning and the share that ends the episode; the robot (cell, x, y, in the safe strip, in fire); the bag and the banked value; every item (id, cell, value now, value at start, burning or not, where it is, cells and ticks to reach it on a fire-free route or none); the floor as a grid (free, burning, wall) and the cells the base may use; the job in hand; the last command and its Result; the cost table (4 ticks a cell, pick 10, drop 5, 2 % per tick, 2 HP per tick). Shaped after `scripts/compiled/API.md` v1.2 where it can be (frame becomes tick, objects become items, last_action and last_result as there), so Liane's compiler prompt and validator carry over. `COMMANDS`, `legal(obs)` and `do(body, command)`: get an item, go home, go to a cell, pick, drop, wait. An illegal command is logged and costs one tick, as in API.md.
- **7b, the text** (about 100 lines). `describe(obs)`: the honest text of A32 and A41 for this room: every option with its predicted effect and cost in ticks, every item with value, loss per tick and distance, the last result in numbers with its reason, the full history, the room as a small character map.
- **7c, programs** (about 80 lines). Liane's contract (a class `Policy` with `decide(self, obs)` returning a command), the loader with the import allow-list, and the reflex written as a program and tested to decide exactly as `agents.reflex`.
- Before any of it is written, the preconditions table (CLAUDE.md rule 2) is extended for the three users. Known now: this room is fully observed (every item and every burning cell is in the state), where HAZARD gave the agent only what its camera had seen; RRARA and the honest text were built for the second case.

## 7. RRARA version 1 on this task: the spec (Nov 23; plan only, no code)

Source: CR's full reading of the paper's second version on Nov 23 (A87), and our own HAZARD implementation in `scripts/rrara/` for the parts the paper leaves out. None of the three planning agents of chunk 6 is RRARA.

**What the paper gives (CR).** Version 1 is a rule that picks the action, plus a language model that checks that action in the background and answers valid, or invalid with a replacement. The rule's action stands unless the answer is invalid. Its 0.00 s latency is because no model call stands in front of the action and the background model's time is not counted. Version 2 replaces the rule with a kept list of subgoals from a language model, rewritten only when a staleness score reaches 0.6; the weights are not published. Not in the paper: the reflex rule, the prompts, whether an invalid answer interrupts the action under way, how the background running is built, any code.

**The spec here, version 1.** Each piece says where it comes from.

| piece | on this task | from |
|---|---|---|
| the rule | `agents.reflex`: most valuable item first, bank after every pick, never enter fire. One decision = one command (`get k` or `home`) | ours (CR's official reflex, Nov 22); the paper gives no rule |
| the checker | after the rule picks a command at tick t, a language model is given the room as text (chunk 7b), the command picked and the legal commands, and answers VALID, or INVALID plus one legal command as replacement | paper (valid / invalid + replacement); prompt ours |
| what an answer changes | at the next decision: if the answer about the previous command was invalid, its replacement is taken in place of the rule's next choice, provided it is still legal; otherwise the rule's choice stands. An answer is used only if it is about the immediately preceding decision | `scripts/rrara/reconcile.py` (paper eq. 9; our A31) |
| the command under way | is not interrupted by an invalid answer | ours (the paper does not say); open point 1 |
| accounting of the checker, a toggle | **free**: the checker's time is not counted and a finished answer always counts (the paper). **strict**: the answer exists only once its time has passed, ticks = ceil(seconds x 10) after the check began, and counts only if that is no later than the next decision | `scripts/rrara` modes "faithful" and "worldtime" (A30) |
| time of a call, a toggle | **measured**: the real seconds of the call. **declared**: a fixed number of seconds per call, the same for every call and machine | CR, Nov 23 |
| how "in the background" is built | in order, with no threads: the command is carried out, the call is made and timed, and the tick at which its answer exists is worked out from the toggle above. The fire does not depend on the model and the model's input is fixed when the check begins, so this equals running them side by side and can be repeated | ours (as chunk 6b); the paper does not say. `scripts/rrara/background.py` used a real thread because HAZARD ran in real time |
| what is saved | per decision: the room text, the rule's command, the raw answer, valid or invalid, the replacement, measured and declared seconds, tokens in and out, the stop reason, whether the answer was used, late or not legal, the command carried out | CR's saved-games rule (A80) |

**Open points for CR before it is built.**
1. Interrupting. On HAZARD an action was short; here one command can last 20 to 80 ticks, so an invalid answer that waits for the next decision can come very late. A second variant could stop the walk at the tick the answer exists (the skills already take a deadline). Suggestion: build the non-interrupting one first, as the faithful version, and the interrupting one as a named variant.
2. What the rule's own decision costs. It takes a few milliseconds. Suggestion: 0 ticks, which is what "0.00 s" means in the paper; under the measured clock ceil would make every decision 1 tick.
3. A refusal or a cut-off answer from the model counts as no answer: the rule stands. No second model answers in its place, so a run is never a mix of two models. (The Anthropic guidance for new code is to switch on a fallback model; for a benchmark that would blur which model was measured.)
4. A34 says our HAZARD reflex reimplements the paper's eq. (2); CR's reading today is that the reflex rule is not in the paper. To reconcile when the HAZARD notes are next touched; it does not affect this task, whose rule is ours.

Version 2 (the kept list and the staleness score) is not specified here.

## 8. The language-model chunks: CR's additions (Nov 23; plan only, no code)

These come after chunk 7 (the observation dictionary, the text, the command list).

**8a. One way to ask a model** (`firelab/models.py`, about 120 lines with tests). `ask(model, text, commands, key_file)` returns the command chosen (or none), the raw answer, measured seconds, tokens in and out, and the stop reason. Two back ends:
- Jev: a port of `scripts/jev/client.py` (one choice question whose options are the lettered commands; A37).
- Anthropic, through the official `anthropic` package, written from the claude-api skill's Python reference (read at that point, not from memory): model `claude-opus-5-5`, effort low. Thinking cannot be switched off on that model, so the request leaves the thinking setting out and controls depth with effort only; no temperature; the output limit is set with room for the thinking, since thinking counts against it. **The stop reason is checked before the answer is read**: only a normal end of turn is read as an answer; a refusal or a cut-off is recorded as no answer.
- The key is passed as a file path only (`--key-file`); it is read once, goes only into the client, and is never printed, logged or saved in a game.

**8b. The agents on a model** (about 150 lines with tests).
- A model before every command, standing while it answers (the blocking baseline that RRARA's first table penalises).
- RRARA version 1 as specified in section 7.
- Two toggles on every run, both written into every result file: checker accounting **free** or **strict**; time per call **measured** or **declared**. The declared time lets a strong model be run as if it answered in a short fixed time, to see what a fast model as good as it could do.

**8c. Runs, in this order.**
1. Jev, on the laptop.
2. `claude-opus-5-5`, effort low, declared time per call, on della-vis1: compute nodes have no internet and the working key is only on Della. Several rooms side by side as separate processes on that node (it has no job queue), each writing its own rows file, which the runner already supports.
3. Before any model run: 3 rooms first; then what, where, how long, and the cost in dollars worked out from the token counts of those 3 rooms. Prices to use, to be checked on the day: Opus 5.5 at 4 dollars per million input tokens and 20 per million output tokens (thinking is billed as output); Jev at 0.042 dollars per million input tokens (A37).

**Not verified, to check when 8 is built:** that the `pyk310` environment made on a login node works on della-vis1; that the `anthropic` package installs there; where the Jev key file is on the laptop and the Anthropic key file on Della (paths from CR, never contents).

## 9. Two repositories: the HAZARD archive and the benchmark (written Nov 25, 13:46, for CR's approval; nothing moved yet)

Why: Liane should work in one repository that holds only the kinder work, and at a dataset release the research part should come out as one folder. This section is the plan and the file map. No file is moved until CR says go. `NAME` below stands for the name CR picks.

### 9.1 What goes where

| Repository 1, the HAZARD archive (today's repo, frozen after the split) | Repository 2, the benchmark (new, all live work) |
|---|---|
| `scripts/` for HAZARD (110 of the 131 files), `results/` (the HAZARD runs), `logs/`, `refs/` | the Python package `NAME/` with `tasks/fire`, `design`, `research/fire` |
| `HAZARD_CHEATSHEET.md`, `BASELINE.md`, `RRARA_BASELINES.md`, `RRARA_V1_PLAN.md`, `README_HAZARD.md` | `README.md`: the task, how to run it, how to add an agent |
| `NOTES.md` and `ASSUMPTIONS.md` as they stand on the split day, with a last line saying where they continue | `llm_notes/`: `PLAN.md`, `DESIGN.md`, and the live `NOTES.md` and `ASSUMPTIONS.md`, whole |
| `README.md`: a map, with a link to repository 2 | the 21 fire scripts, the fire results, the Della setup |

Reason for the live logs in repository 2: the code cites 24 assumption numbers (A32 to A99), 631 of the 1 698 lines of NOTES.md are already kinder work, and every session writes code and notes together. Two repositories for one session means two commits and two pushes each time, and the two drift.
At a release the dataset is repository 2 without `NAME/research/` and `llm_notes/`.

### 9.2 The file map of repository 2 after step 1 (file names unchanged; only folders change)

```
NAME-repo/
├── README.md                     from firelab/README.md, paths updated
├── pyproject.toml                NEW, about 15 lines, so that one install replaces the PYTHONPATH line
├── llm_notes/                    PLAN.md  DESIGN.md  NOTES.md  ASSUMPTIONS.md
├── scripts/                      della_firelab_setup.sh
└── NAME/                         the one Python package
    ├── tasks/
    │   └── fire/                 __init__.py (registers the room)  fire.py  items.py  types.py  env.py  skills.py
    │       │                     interface.py  text.py  agents.py  planner.py  clock.py  gate.py  thinkers.py
    │       │                     programs.py  validate.py  PROGRAM_API.md
    │       ├── examples/         reflex_program.py
    │       └── tests/            13 test files: fire, items, env, room, level2, skills, interface, text,
    │                             agents, planner, gate, thinkers, programs
    ├── design/                   gate_grid.py  measure_spread.py
    ├── research/
    │   └── fire/                 run_rooms.py  models.py  model_agents.py  compile.py  gate_clock.py  gate_agents.py
    │       ├── tests/            4 test files: run_rooms, models, model_agents, compile
    │       ├── scripts/          the other 20 fire scripts (tables, summaries, probes, slurm, Liane's loop)
    │       ├── slides_nov23/     the chart scripts that read the fire results
    │       └── results/          firelab_sim  firelab_gate  firelab_compiled  firelab_show
    └── test_layers.py            NEW, about 20 lines: fails if tasks or design import from research
```

Where each file goes follows from what it imports today (checked Nov 25 with grep over `firelab/*.py`):
- **tasks/fire** holds every file that imports nothing from the research files. `validate.py` is one of them: it plays a program and the reflex on any rooms and writes to the file named by `--out`. It is already a plain runner for programs.
- **thinkers.py stays in tasks/fire.** The reference planner (the agent `always`, 0.82) is in it, in the same function as `commit` and `moving`. Sending it to research would leave the benchmark unable to run its own reference score.
- **programs.py stays in tasks/fire**, not in a top-level `agents/` folder. It imports the fire task's observation, so it is a fire file today. A folder for agents that work on any task waits for a second task.
- **research/fire** holds what imports models or runs the experiments. `run_rooms.py` goes there whole in step 1; it imports the thinkers, the model agents and the model clients.
- **Known wart, left for later:** `gate_grid.py` (design) writes its table into the results folder under research. It needs its own output folder; that is a code change, so it is not in step 1.

### 9.3 Steps

**Step 0. CR decides** the three things in 9.5, and makes an empty GitHub repository.

**Step 1. Move only.** No logic changes.
1. In the clone: `git subtree split -P firelab` makes a branch that holds only `firelab/` with its 11 commits of history. Repository 2 starts from that branch. The fire scripts and the four logs are copied in with one commit that names the commit of repository 1 they came from; their earlier history stays in repository 1.
2. `git mv` every file to its place in the map.
3. Rewrite the 134 import lines in 34 files, the 4 result-folder constants, the paths in the 21 scripts, and these places that a search for import lines misses:
   - `__init__.py` registers the room with the text `"firelab.env:FireLab3DEnv"`. That text is an import path and must change. The room ids themselves (`firelab/FireLab3D-l2-v0`) stay.
   - `compile.py` finds `PROGRAM_API.md` and the example next to itself; after the move they are in another folder.
   - `gate.py` line 25, `from firelab.fire import ...`. gate.py is closed since Nov 23; this one line needs CR's word (9.5).
4. `PROGRAM_API.md` is not edited in step 1. The program-writer reads it word for word, so its text stays the text the kept program was written from. Its two mentions of old file paths are noted in NOTES.md and fixed when the next compile is planned.
5. Add `pyproject.toml` and `test_layers.py`. These two files are the only new code, about 35 lines.

**Checks for step 1, in this order.** Stop at the first that fails.
- All 157 tests plus the new one.
- `all_agents_table.py`, `gap_table.py`, `liane_summary.py` print the same numbers as on Nov 25 before the move.
- One replay with no model: the kept program on rooms 0 to 49, on the laptop, 6 minutes (measured Nov 23). Every room's score must equal the stored file `rooms_0_49_laptop.json`. This needs CR's yes on the day, as any run does.

**Step 2. Della, once.** A new clone of repository 2, the environment set up inside it by the setup script, the saved games and the Haiku training games copied from the old clone into the new results folder, and one short run with no model to see that it starts. The Della session does this; CR gives it the message. The Nov 25 message about moving folders inside the old clone is not needed if this step happens.

**Step 3. The plain runner, a chunk of its own, with its own plan before any code.** One command that plays random, the reflex, the reference planner or a program on any rooms and writes one row per room into a folder the user names. It grows from `validate.py`. `run_rooms.py` is not cut in two; it stays the research runner. Under 250 new lines.

**Step 4. Repository 1.** After repository 2 is pushed and step 2 has passed: one commit that removes `firelab/` and the fire scripts from repository 1 and points its README to repository 2. History keeps the old files. The Drive map's links change once, here.

### 9.4 Not in this plan

Renaming files (`fire.py` to `world.py` and so on). A written task contract. A top-level `agents/` folder. A `configs/` folder: the main rule set is constants in `env.py` and `items.py` today, and a settings file is a code change. Splitting `model_agents.py` into the weak baseline and RRARA. Splitting `thinkers.py`. Updating kinder to its newer commits.

### 9.5 For CR to decide before step 1

1. The name of repository 2 and of the package. One short word, the same for both.
2. **Decided by CR, Nov 25, 15:30:** the live notes go in repository 2, in a folder of their own named for what they are, the working notes of the Claude sessions. Folder name `llm_notes/`. PLAN.md and DESIGN.md go in the same folder, because they are written in the same sessions and cite the same numbers (my choice, decided without CR, to review).
3. Whether the one import line of `gate.py` may be changed.

Defaults I will use unless CR says otherwise: history by `git subtree split`; I work directly in the clone of repository 2 and the copy script is retired for kinder work; the old folder `G:\iw-thinking-fast\firelab` is left untouched until CR says it can go; `pyproject.toml` is added.

### 9.6 Who can read repository 2 (asked by CR, Nov 25)

- **The chats on the laptop.** A chat reads folders on this disk, not GitHub. This chat and the monitoring chat both work in `G:\iw-thinking-fast` (checked Nov 25). So the clone of repository 2 goes inside that folder, as `G:\iw-thinking-fast\NAME-repo`, and both chats can read and edit it with no new setting. New chats should keep being opened in `G:\iw-thinking-fast`, not inside the clone: the saved memory and `CLAUDE.md` belong to that folder.
- **The chat in the vault folder** cannot read it unless CR adds the folder to that chat. It does not need to.
- **The Della chat** reads only what is on Della. It needs its own clone there (step 2). Not known: whether the login that pulls today's repository on Della also works for a new private repository.
- **Liane, and any Claude she uses,** see exactly what is pushed to GitHub and nothing else: no memory, no `CLAUDE.md` from this laptop. If her Claude should follow the same rules, a short `CLAUDE.md` has to be in repository 2 itself. Not in step 1; for CR to decide.
