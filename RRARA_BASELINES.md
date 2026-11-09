# RRARA: what each baseline is and why it is there

Draft for Liane, written Nov 8 2026 from the arXiv v2 text of Zheng, Mao, Zhang, Cai, "Reflex First, Reflect Later: Latency-Aware Embodied LLM Agents for Dynamic Response" (arXiv 2506.07223, CVPR 2025 Embodied AI Workshop) and from HAZARD's code at commit d35c71e. The "why" lines are my reading; the paper gives a reason for almost none of its baselines.

## The setup the baselines run in

- HAZARD, three hazards (fire, flood, wind), 25 held-out test scenes per hazard, value rate as the score.
- Time Conversion Mechanism (TCM): each decision's wall-clock time t becomes ceil(t x 30) simulator frames. The world advances that many frames while the agent stands still, then the action runs on the now-stale observation.
- Table 1: ten agents, value rate with and without TCM. Table 2: thirteen agents under TCM, with Response Latency (seconds of blocking delay per decision), average actions per episode, Latency-to-Action Ratio (percent), and value rate averaged over the three hazards.
- No code is released. The paper says only "HAZARD baselines are reproduced using their default hyperparameters." So their baselines are HAZARD's shipped agents, run under their clock.

## The baselines, one by one

**Rule** (HAZARD `policy/rule_based.py`)
- What: hand-written if/then over what the agent holds and what is nearest. Not holding and a target is near: pick it up. Holding: walk to the bag and drop. Otherwise walk to the next target (`rule_based.py:38-63`). No computation, so 0.00 s latency and 26 actions per episode.
- Why: the floor, and the reflex inside RRARA-v1. It shows that an agent that never thinks is untouched by TCM (fire 0.33 to 0.35, average change 0.00).

**Greedy** (HAZARD `policy/fire_flood_heuristic.py`, class `GreedyAgent`)
- What: one-step choice over the object list. Each candidate action gets cost = distance divided by the object's value, with a halved value for objects already burning or flooded (`fire_flood_heuristic.py:41-107`). Picks the cheapest. 0.14 s.
- Why: the cheapest thing that computes anything. It costs a few frames per decision and still gains nothing from them (0.08 base, 0.11 under TCM). I read it as the control for "a little planning is not the problem".

**MCTS** (HAZARD `policy/mcts.py`)
- What: Monte Carlo tree search over the symbolic object list, 2000 simulations and rollout depth 20 per decision (`mcts.py:134-138`). 4.30 s per decision, 8 actions per episode.
- Why: the classic planner. Best non-LLM agent without a clock (fire 0.75) and it collapses under TCM (0.25). This one row is the paper's argument: deciding well but slowly loses to deciding badly and fast.

**GPT-3.5** (HAZARD `policy/llm.py` pipeline, text prompt in, one action out)
- What: the shipped LLM agent with GPT-3.5-Turbo. 2.54 s, 11 actions, value rate 0.25 averaged over hazards.
- Why: the cheapest LLM and the model they later fine-tune. It is the LLM that loses least under TCM (average change -0.12) because it is the fastest.

**GPT-4, GPT-5, Claude 3.7 Sonnet, Claude 4.6 Sonnet, Gemini 3.5 Flash, Llama-2-7b**
- What: the same pipeline with the model swapped. Latencies 4.0 to 14.4 s per decision; 4 to 9 actions per episode; all of them lose 0.26 to 0.41 value rate under TCM.
- Why: to show the ordering flips. Without the clock Claude 3.7 is the best agent in the table (fire 0.77); with it, the worst of the APIs (0.10). Llama-2-7b at 14 s shows a small local model is not automatically fast either.

**Fine-tuned GPT-3.5**
- What: GPT-3.5-Turbo-0125 fine-tuned on 1,000 GPT-5 planning trajectories, two epochs. Value rate 0.25 to 0.35 (Table 3), with 30 examples giving 0.30.
- Why: to make the reflector good enough while staying cheap. This is the part of their system that is preparation done before the episode.

**LLM-PrePlanner**
- What: the LLM writes an ordered list of object-centric subgoals (walk to, pick up, drop) once; the list is cached and executed top-down; it is recomputed only when a staleness score crosses 0.6. The staleness weights are not published. 0.40 s, 20 actions, value rate 0.40.
- Why: to amortise LLM calls. Almost all the LLM's quality at a sixth of the latency. The cache is the second preparation axis in the system.

**RRARA-v1**
- What: rule reflex acts immediately; the fine-tuned GPT-3.5 checks the action asynchronously while it executes and replaces it only if it says invalid. 0.00 s blocking latency, 27 actions, value rate 0.37.
- Why: the ablation that isolates async reflection. The reflex alone scores 0.19; adding a background checker nearly doubles it with no added blocking time.

**RRARA-v2**
- What: PrePlanner as the fast path instead of the rule, same async reflector. 0.38 s, 25 actions, value rate 0.44, their best.
- Why: to show the two ideas stack: cached plan for quality, reflection for correction.

## What this means for my next step

- There is no RRARA code, so "reproduce their baselines on their codebase" means: add TCM to HAZARD's decision loop and rerun HAZARD's shipped agents under it. That is the same as the clocked baseline run already planned.
- My unclocked numbers on the fire test set do not match their base column: rule 0.08 against their 0.33, MCTS 0.15 against 0.75, greedy 0.11 against 0.08, random 0.36 (they do not report random). The HAZARD authors attribute the gap to a ThreeDWorld collision bug in some versions; their reply and my logs are with them now.
- Table 2's rule row (0.19) is the three-hazard average; Table 1's fire-only rule row under TCM is 0.35. Compare per hazard against Table 1.
- Order of work: TCM in the loop; rerun rule, greedy, MCTS and Claude Haiku under TCM on the fire test set and compare the drop with Table 1; then the three RRARA pieces (reflex, async reflector, PrePlanner with a guessed staleness rule, flagged).

## Table 1 and Table 2 as published

Table 1, value rate, base / TCM:

| Agent | Fire | Flood | Wind | Avg change |
|---|---|---|---|---|
| MCTS | 0.75 / 0.25 | 0.40 / 0.15 | 0.15 / 0.02 | -0.29 |
| Rule | 0.33 / 0.35 | 0.25 / 0.23 | 0.00 / 0.00 | 0.00 |
| Greedy | 0.08 / 0.11 | 0.08 / 0.09 | 0.00 / 0.00 | +0.01 |
| GPT-3.5 | 0.65 / 0.44 | 0.36 / 0.24 | 0.09 / 0.06 | -0.12 |
| GPT-4 | 0.71 / 0.22 | 0.44 / 0.21 | 0.20 / 0.04 | -0.29 |
| GPT-5 | 0.73 / 0.35 | 0.47 / 0.22 | 0.24 / 0.07 | -0.27 |
| Llama-2-7b | 0.60 / 0.05 | 0.31 / 0.05 | 0.04 / 0.00 | -0.28 |
| Claude 3.7 Sonnet | 0.77 / 0.10 | 0.48 / 0.10 | 0.21 / 0.02 | -0.41 |
| Claude 4.6 Sonnet | 0.71 / 0.18 | 0.43 / 0.11 | 0.24 / 0.02 | -0.36 |
| Gemini 3.5 Flash | 0.69 / 0.22 | 0.40 / 0.15 | 0.09 / 0.03 | -0.26 |

Table 2, under TCM, averaged over hazards:

| Agent | Response latency (s) | Actions / episode | LAR (%) | Value rate |
|---|---|---|---|---|
| Rule | 0.00 | 26 | 0 | 0.19 |
| Greedy | 0.14 | 20 | 6 | 0.07 |
| GPT-3.5 | 2.54 | 11 | 52 | 0.25 |
| GPT-5 | 4.00 | 9 | 67 | 0.21 |
| MCTS | 4.30 | 8 | 61 | 0.14 |
| GPT-4 | 5.46 | 7 | 72 | 0.16 |
| Gemini 3.5 Flash | 5.94 | 6 | 73 | 0.13 |
| Claude 4.6 Sonnet | 9.83 | 5 | 86 | 0.11 |
| Claude 3.7 Sonnet | 13.05 | 4 | 89 | 0.07 |
| Llama-2-7b | 14.36 | 4 | 89 | 0.03 |
| LLM-PrePlanner | 0.40 | 20 | 17 | 0.40 |
| RRARA-v1 | 0.00 | 27 | 0 | 0.37 |
| RRARA-v2 | 0.38 | 25 | 17 | 0.44 |

Not defined in the paper: the formula for Response Latency and for LAR, and the staleness weights. Defined: ILR = t_inf / (t_inf + t_action), not reported in any table.
