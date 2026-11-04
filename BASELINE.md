---
type: plan
status: written by Claude on Oct 16 from Tom's brief (PDF, p. 3), the RRARA note in the vault, the vault session "Non-codex chat access" (Oct 16), and the HAZARD code; nothing here is run yet except where NOTES.md says so
---
# What "running a baseline on HAZARD" means, and what it needs

## 1. The baselines Tom's brief asks for (Thinking Fast Enough.pdf, p. 3, verbatim table)

| Baseline | What it does | Why it is included |
|---|---|---|
| Always deliberate | Calls the slow solver at every step | Fails obviously under time pressure, which motivates the problem |
| Always react | Uses only the fast policy | The opposite extreme |
| Fixed budget | One solver, same thinking time every step | The standard approach before learned budgets |
| Single-solver learned gate | Muppidi et al.'s method | Tests whether having multiple tools helps |
| No preparation (X = 0) | Same gate and tools, off-the-shelf only | Tests whether preparation helps, including whether the coding agent's work matters |

Metrics (same page): performance across a grid of (X, Y); where the floor on Y sits per task family; regret of the gate against an oracle gate; graceful degradation as Y shrinks.

## 2. What the vault session concluded (Oct 16, its words paraphrased)

- The work on HAZARD is exactly Tom's evaluation section: impose the clock, run the five baselines, build the gate, vary X, draw the map. No simulator cloning is needed anywhere.
- Purpose 1, "make waiting cost something": prefer Muppidi's declared k (a known price to plot against); use RRARA's measured latency once, to show the price is real on this machine. A cheap first experiment: check whether HAZARD's deadlines sit in Shperberg's "Goldilocks" zone (tight but not hopeless).
- Purpose 2, fast versus slow: start with two solvers. RRARA's rule is the fastest and dumbest fast solver; a reactive policy with an uncertainty signal is better input for a gate later.
- Purpose 3, the gate: do two, the hand-derived value-of-information rule (Russell and Wefald, Coles) and a learned one (Muppidi's gate with the solver swapped for an LLM or planner call), and report the gap.
- HAZARD's holes (Testbed Checklist): the clock is not in the repo, the task mix is not designed, episodes are slow, no arm. HAZARD stays the testbed; DOMINO is the later robot step.

## 3. What RRARA did, which is the thing to reproduce first (vault note zhengReflexFirstReflect2025)

- Time Conversion Mechanism: F = ceil(t_inference x 30). After each decision the simulator advances F frames with the agent stationary ("blocking inference"); the action then executes on possibly stale information. Inference-latency ratio ILR = t_inf / (t_inf + t_action).
- Agents under TCM (Table 2, value rate at mean latency): rule 0.19 at 0.00 s; GPT-3.5 0.25 at 2.54 s; MCTS 0.14 at 4.30 s; LLM-PrePlanner 0.40 at 0.40 s; RRARA-v1 0.37 at 0.00 s; RRARA-v2 0.44 at 0.38 s. Table 1: Claude 3.7 Sonnet loses 0.41 value rate under TCM, GPT-3.5 loses 0.12.
- Scenarios: fire and flood in four indoor rooms, wind in four outdoor regions, 25 each, one environment held out.
- Not published: the staleness-score weights, the exact latency measurement window, any code. Hardware not stated in the note (question for the vault session, A15).

## 4. What HAZARD gives, from the code (file:line in the clone at commit d35c71e)

- Episode loop: `src/HAZARD/challenge.py:293-420`. Each iteration: observe (`_obs()`), `agent.choose_target(state, processed_input)` (line 318), then execute `walk_to` / `pick_up` / `drop` / `explore` via `policy/env_actions.py`, each of which runs many simulator frames.
- Frames: `env.controller.frame_count` increments once per `communicate()` (`envs/fire/fire.py:219`); `comm_counter` similarly (`fireagent_controller.py:469`). Episode ends when `frame_count >= max_steps` (`challenge.py:398`) or no targets are left; `max_steps` is 1500 for fire and flood, 3000 for wind (`challenge.py:53-59`). At RRARA's 30 FPS that is 50 s and 100 s of world time.
- Per-decision record: `actions.txt` gets `step {comm_counter} action ...` per decision (`challenge.py:323-327`), so frames per high-level action are consecutive differences. This is the measurement CLAUDE.md asks for.
- Agents: `rule`, `greedy`, `mcts`, `mctsv2`, `random`, `llm`, `llmv2`, `rl`, `oracle`, `record` (`run_experiments.py:26-85`). `submit()` also accepts an agent object (`run_experiments.py:168`), which is how `scripts/run_baseline.py` wraps a shipped agent with timing without editing HAZARD.
- Where the clock hooks in: between `choose_target` returning and the action executing (`challenge.py:318-322`). TCM = call `env.controller.communicate([])` F times there. For a declared-k clock, the same hook with F = k.
- Metrics: HAZARD writes `eval_result.json` per episode; `src/HAZARD/utils/calc_value.py <output_dir> <task>` prints Value, Step, Damage (`submit.md:71-84`).
- Known bug hit on Oct 16: `challenge.py:323` dereferences `nearest_object` on any `pick_up` even when no `walk_to` succeeded before, raising `KeyError: -1`. The random agent hits it on its first action; any agent hits it after a failed `walk_to`. One-line guard needed (ASSUMPTIONS A16).

## 5. The order of work, mapped to CLAUDE.md's plan

Step 1 (now): shipped agents, no clock, a few episodes, with latency and frames per action logged.
- `scripts/run_baseline.py --env fire --agent rule|greedy|mcts|mctsv2|random --episodes N`
- Output per run: `results/<env>_<agent>_epN_<stamp>/{decisions.jsonl, summary.json, <scene>/eval_result.json, actions.txt}`
- Reference numbers under `results/`, exact commands in `NOTES.md`.

Step 2 (needs CR's go): the clock.
- 2a. TCM as in RRARA: advance F = ceil(t x 30) frames after each decision, agent stationary. Rerun rule, greedy, mcts. Reproduce the shape of RRARA Table 2 (MCTS falls below rule).
- 2b. Declared k (Muppidi's form, the vault session's preference): same hook with F = k, for k in a small set. This is Tom's "fixed budget" baseline for each solver.
- 2c. RRARA itself: rule reflex + async LLM reflector + cached preplanner; staleness weights are an open assumption.
- "Always react" = rule or greedy with F = 0. "Always deliberate" = mcts (or an LLM) with F charged every step. "Single-solver learned gate" = Muppidi's gate choosing k for one solver. "No preparation" = the same gate with no precomputed A* or cached plan.

Open questions for CR before step 2: which LLM and account for the slow solver (A9); whether the laptop or adroit-vis is the machine of record for timing (A15); whether to apply the A16 guard patch.
