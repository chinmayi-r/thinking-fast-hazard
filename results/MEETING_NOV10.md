# Thinking Fast Enough on HAZARD: status for the Nov 10 meeting

Task: HAZARD fire, 25 test kitchens, 1500-frame budget (50 s at 30 fps). Score = value saved / value available, mean over the 25 kitchens (sd in parentheses where known).

Clock: RRARA's time-conversion rule (arXiv 2506.07223, eq. 1). Each decision's wall-clock time t is charged as ceil(t x 30) frames during which the robot stands still and the fire advances. Implemented as a wrapper around any agent (scripts/run_baseline.py, TimedAgent); HAZARD itself untouched; verified on every record of every clocked run that frames charged = ceil(wall x 30) = frame advance.

Two HAZARD versions (ASSUMPTIONS.md A29). The HAZARD authors confirmed by email that the paper numbers come from commit 5a50c2d + TDW 1.12.5 ("paper's snapshot"). The current snapshot d35c71e + TDW 1.13.0 changed the action executors: walk arrival radius 1.25 m -> 0.5 m, pick-up reach tightened, non-target pick-ups refused. Walks fail far more often on the current snapshot. Everything below was run on both. Decision for Tom: which one to build on (CR's current choice: current, "closer to real life").

## Results (value rate, clock off -> clock on at 30 fps)

| Brain | Current HAZARD (d35c71e) | Paper's snapshot (5a50c2d) | RRARA paper, Table 1 fire |
|---|---|---|---|
| rule (shipped) | 0.141 -> 0.131 | 0.410 -> 0.383 | 0.33 -> 0.35 |
| greedy (shipped) | 0.133 -> 0.145 | 0.270 -> 0.249 | 0.08 -> 0.11 |
| MCTS (shipped) | 0.114 -> 0.102 | 0.750 -> 0.359 | 0.75 -> 0.25 |
| Haiku LLM agent (shipped llm, Tom's weak baseline) | 0.170 -> 0.076 (Della 25/25) | not run | GPT-3.5/4-class rows in paper |
| our reflex alone (paper's eq. 2 rule, reimplemented) | clock on: 0.199 (0.19) | clock on: 0.757 (0.18) | RL agent 0.00 |
| RRARA faithful (reflex + Haiku override, background thinking free) | 0.160 (0.18) Della 25/25; 0.135 (0.13) laptop 25/25 | 0.708 (0.22) Della 25/25 (A35) | RRARA-v1 0.37 (Table 2 avg) |
| RRARA worldtime (override only if the verdict arrived before the checked action ended) | 0.220 (sd 0.227) [Della, 25/25, after the timing fix]; old 0.147 / 0.142 invalid | not yet | not in paper |

Worldtime note (found 10:05 Nov 10): the background runner timed a verdict from its start to the moment the agent read it at the NEXT decision, which includes the whole simulated action (about 40 s of laptop wall time), instead of to the moment the verdict existed (about 2.2 s). So every worldtime verdict was "late" by construction and the laptop 0.147 / Della 0.142 rows measure reflex-alone plus a tax, not worldtime. Fixed (scripts/rrara/background.py, regression test added, 14 tests pass); the old folders are kept with the suffix _failed_walltime_bug; the sweep is being rerun. Faithful rows are unaffected (lateness is ignored by definition there; only the logged wall field was wrong).

\* Della partial, 13-14 of 25 kitchens, interrupted by the 05:30 Della maintenance reboot; resumable. All other rows are 25/25. Machine stated per number (A15): shipped agents and reflex-alone rows on Della compute nodes; RRARA rows on the laptop (RTX 3060) and on della-vis1; the Haiku-unclocked 0.170 is a laptop run (Oct 28).

Cost per decision under the clock (frames charged per episode, out of 1500): rule 1 frame per decision (about 30 per episode); greedy 2-12; MCTS 7-121 (936 per episode on the paper's snapshot: 62 % of the budget, decisions per episode 32.7 -> 14.0); our reflex 0.008-0.02 s, 22-33 per episode; RRARA 2 % of the budget; Haiku LLM agent 3.95 s per decision, 975 per episode (65 % of the budget).

## What the numbers say

1. The paper's result reproduces on the paper's snapshot: MCTS collapses under the clock (0.750 -> 0.359, paper 0.75 -> 0.25) and the rule agent does not move. On the current snapshot nothing collapses because nothing works: every brain sits near 0.13 with or without the clock. Movement failure, not latency, is the bottleneck there.
2. A 10 ms reflex alone beats everything under a clock. On the paper's snapshot it scores 0.757, equal to the unclocked MCTS (0.750), twice the clocked MCTS, and double the paper's own RRARA-v1 (0.37). On the current snapshot it is the best clocked agent (0.199 vs rule 0.131, paired +0.069, se 0.032). The reflex is the paper's eq. (2) policy (nearest target by path distance over value) plus one addition of ours, a penalty on targets whose walk already failed (A34), which the paper does not need because its walks rarely fail.
3. Adding the slow brain does not help on either snapshot, so far. On the paper's snapshot faithful scores 0.708 against the reflex alone at 0.757 (paired -0.049, se 0.031; 21 % overrides, 42 of them illegal by the time they would run), still double the paper's RRARA-v1 0.37. On the current snapshot, faithful mode (the paper's accounting, background Haiku free): Haiku overrides the reflex on 28 % of its checks and the score drops from 0.199 to 0.160 (Della, paired -0.040, se 0.025) or 0.135 (laptop). Worldtime mode (honest accounting, rerun on Della after the timing bug above, 25/25): 0.220, paired +0.020 vs reflex alone (se 0.022, within one se) and +0.060 vs faithful (se 0.027). A verdict takes 1.9 s (57 frames): in time for the 100-frame walks, late for pick-ups and drops, so 100 of 421 verdicts (24 %) were late and discarded and 95 (23 %) overrode the reflex. Counting only verdicts that existed before the checked action ended, Haiku neither helps nor hurts; letting late verdicts override (faithful) costs 0.04. For the slow brain to matter it must be both fast enough to arrive and right more often than the reflex; the faithful numbers say Haiku zero-shot is not the latter here.
4. Tom's weak baseline measured: the shipped LLM agent with Haiku loses 65 % of the episode standing still and scores 0.076 under the clock (Della 25/25; 0.170 without, laptop).

## Caveats and open assumptions (full list in ASSUMPTIONS.md A29-A34)

- Reflector is Haiku zero-shot with a one-word VALID/letter prompt; the paper fine-tuned GPT-3.5 on expert traces. The paper's staleness weights and exact reflex rules are unpublished (code request drafted to the Sydney authors; not sent).
- The paper does not charge for background reflector compute (its Limitations section says so); "worldtime" is our stricter accounting, A30.
- HAZARD describes a capped 100-frame walk as "paused", not failed, in the LLM's text (llm.py:534-540); the reflector therefore rarely sees a failure. Honest-wording fix not yet applied.
- RRARA rows ran on two machines (laptop and della-vis1); numbers within 0.03 of each other; Della's are the ones to keep once complete.

## Still running / next

- Della login node: Haiku LLM agent, RRARA faithful, RRARA worldtime, 13-16 of 25 each, stopped by the 05:30 reboot; restart recipe in Della's results/DELLA_TWO_TRACKS.md.
- Not started: flood and wind; the honest failure wording; Coles 2024 and Muppidi 2026 adaptations (plan step 3).
- Files: NOTES.md (every run, timestamps), ASSUMPTIONS.md (A1-A34), results/DELLA_TWO_TRACKS.md on Della (per-scene tables, costs), scripts/rrara/ (agent, reflex, reflector, referee, tests).
