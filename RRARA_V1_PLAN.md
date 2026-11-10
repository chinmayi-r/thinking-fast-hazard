# RRARA-v1 on HAZARD: the files, in the order CR writes them

Written Nov 9 2026. CR writes every file herself; Claude explains each one before it is typed and checks it after.
Source for the method: Zheng et al., arXiv 2506.07223 v2 (vault note `Papers/zhengReflexFirstReflect2025.md`,
summary in `RRARA_BASELINES.md`). Source for the hook: `scripts/run_baseline.py:43-105` (TimedAgent) and
HAZARD `src/HAZARD/challenge.py:289-430` (the decision loop). Open assumptions go to `ASSUMPTIONS.md` A30+.

## What RRARA-v1 is, in one paragraph

HAZARD's loop asks the agent for one high-level action (walk_to X, pick_up, drop, explore), executes it for up to
100 frames, then asks again. RRARA-v1 answers that question instantly with HAZARD's own rule agent (the "reflex",
0.1 ms, so the clock charges 1 frame). While the simulator executes that action, an LLM (the "reflector") looks at
the same observation and the reflex's choice in a background thread and answers "valid" or "invalid, do Y
instead". At the next decision, if a verdict has arrived and it says invalid, Y is executed instead of the
reflex's next choice; otherwise the reflex acts again. The agent never waits for the LLM, so its blocking latency
stays at the reflex's, and the LLM's quality still gets in through the override.

## The files

| # | File | What it is | Lines |
|---|---|---|---|
| 0 | `RRARA_V1_DESIGN.md` | preconditions table (CLAUDE.md rule 2) and the design decisions | prose |
| 1 | `scripts/rrara/__init__.py` | empty; makes the folder a Python package | 0 |
| 2 | `scripts/rrara/reflex.py` | wraps HAZARD's RuleBasedAgent; returns its action and the list of legal actions | ~30 |
| 3 | `scripts/rrara/reflector.py` | builds the check prompt, calls Claude, parses VALID / INVALID + replacement, logs wall time | ~80 |
| 4 | `scripts/rrara/background.py` | runs one function in a thread; lets the caller ask "done yet?" without waiting | ~30 |
| 5 | `scripts/rrara/reconcile.py` | the override rule: reflex proposal + verdict (if any, if in time) -> final action; pure function | ~40 |
| 6 | `scripts/rrara/test_rrara.py` | tests for 3's parser and 5's rule with fake inputs, no simulator | ~50 |
| 7 | `scripts/rrara/agent.py` | `RRARAv1Agent`: HAZARD's agent interface (agent_type, reset, choose_target) composing 2-5 | ~80 |
| 8 | `scripts/run_baseline.py` | edit: `--agent rrara_v1` builds the agent; `--reflector_mode`; summary gets the override counts | ~20 changed |

Order matters: 2, 3, 5 can each be run and checked alone; 4 is tiny; 6 proves 3 and 5 before any simulator time;
7 only wires them; 8 is the last edit. One laptop scene with `--clock tcm` is the first real run.

## Design decisions to settle in file 0 (each becomes an ASSUMPTIONS entry)

- A30. When is a verdict "in time"? RRARA's protocol excludes background compute from latency (their words:
  Response Latency excludes planner and reflector compute), so under their rule the verdict is always usable at
  the next decision. The world-time-consistent rule: the verdict is usable only if ceil(t_llm x 30) <= frames the
  action actually consumed (frame_count difference between the two decisions). Implement both behind
  `--reflector_mode faithful|worldtime`; report both; default worldtime.
- A31. A verdict applies only at the very next decision; an older one is discarded (RRARA-v1 has no staleness
  score; that is the PrePlanner's, v2). The replacement is executed only if it is still in the legal action list.
- A32. Reflector model and prompt. RRARA used GPT-3.5 fine-tuned on 1,000 GPT-5 trajectories; we use
  `claude-haiku-5-5` zero-shot with a short prompt (targets and values, explored objects, held object, nearest
  object, the reflex's proposal, the legal actions; answer VALID or INVALID plus one action). No fine-tuning.
- A33. The override happens between high-level actions (after a walk_to finishes, up to 100 frames), not
  mid-walk. HAZARD's `walk_to_single` (one path step per decision) would allow mid-walk override; v1 does not use it.

## Controlled comparison: what every agent shares, so that RRARA is the only difference (Nov 9, from reading the shipped agents)

What HAZARD already fixes for every agent (so nothing to build):
- The contract: `reset(goal_objects, objects_info)` and `choose_target(state, processed_input)` returning `(name, param)`
  (`documentation/agents/custom_agent.md:5-41`, template `policy/custom.py:11-20`); `submit()` takes an object
  (`run_experiments.py:91`); `challenge.py:257-259` accepts label `custom`.
- The observation: `processed_input` is built once per decision in `challenge.py:136-163`: explored objects as
  `{name, category, id}` (no positions), held objects, nearest object, frame count, last action's result. `state` adds the
  raw rgb/depth/temperature/segmentation and the semantic map (doc lines 20-27).
- The executor and its durations (walk_to <= 100 frames, `challenge.py:345-355`), the 1500-frame budget
  (`challenge.py:421`), the scorer (`challenge.py:get_score`), the 25 test scenes.
What our runner already fixes for every agent: the stopwatch and the clock (`run_baseline.py:71-105`), the per-decision
log and summary, the machine label.

How the shipped agents differ from each other (the things a framework must hold equal or declare):
- Information used. Rule (`policy/rule_based.py:65-74`) reads only `processed_input`. Greedy and MCTS
  (`fire_flood_heuristic.py`, `mcts.py:190-199`) add object positions and a temperature history from `state`. The LLM
  agent (`llm.py:542-587`) computes the agent position from `state['goal_map']`, per-object distance and temperature
  (`progress2text`, `llm.py:382-430`), and keeps an action history.
- Hard-coded reflexes inside the LLM agent: `llm.py:562-577` drops a held target and picks up a nearby target without
  asking the model; the model is consulted only for where to walk or whether to explore. So HAZARD's LLM baseline is
  already "reflex for the obvious step, LLM for target choice", which is worth saying in the write-up.
- Action vocabulary. Rule's legal list (`rule_based.py:38-63`) and the LLM's (`llm.py:487-532`, cap of 10) differ; both
  map onto the same four executor actions.
- Randomness. Rule picks at random among legal actions (`rule_based.py:93`); MCTS uses `random` in selection and
  rollouts (`mcts.py:290,329`); the LLM samples at temperature 0.7 (`llm.py:33`). The simulator itself is not
  reproducible across machines (A22 item 16; Della vs laptop Nov 9). None of this is seeded today.
- Two quirks that hit every LLM-based agent equally and must be documented, not fixed silently: the template's
  `$OBJECT_HISTORY$` placeholder is never substituted (`llm.py:596-598` replaces STATE, ACTION_HISTORY, TARGET_OBJECTS,
  AVAILABLE_ACTIONS only; seen verbatim in `results/sweep_fire_llm_haiku_test25/*/0/0043_info.json`), and the
  "distance ... m" values are semantic-map grid cells, not metres (356.45 "m" for an object in a kitchen).

Framework rules (each becomes a line in RRARA_V1_DESIGN.md):
- F1. One wrapper for all agents: `TimedAgent` unchanged, plus a per-episode seed for Python's `random` (agent-side
  randomness only; the simulator's remains and is reported as sd over 25 scenes).
- F2. The reflex is an untouched `RuleBasedAgent` instance, the same class the rule baseline runs. Zero new policy code.
- F3. The reflector is an untouched instance of HAZARD's `LLM` class with a different prompt template (validate the
  proposed action, answer VALID or the letter of a better option). It therefore shares with the LLM baseline the
  description text, the distance and temperature fields, the available-action list, the parser, the Anthropic adapter
  (A18) and the call log. The LLM baseline is the synchronous control for the reflector.
- F4. Same model, max_tokens and temperature for the LLM baseline and the reflector (A32: `claude-haiku-5-5`).
- F5. Same scenes, machine, commit, clock, aggregator. New columns only: overrides per episode, verdicts in time,
  reflector wall time.
- F6. The ablation ladder that isolates each piece, all through the same wrapper: rule alone (done); LLM alone,
  synchronous, clocked (Haiku run exists unclocked; clocked to do); RRARA-v1 faithful (D1 paper rule); RRARA-v1 honest
  (D1 strict rule); RRARA-v1 with an instant reflector (verdict applied synchronously at zero charge: the upper bound
  on what the reflector can add); RRARA-v1 with the reflector disabled (must reproduce rule alone within noise: the
  wiring check).
