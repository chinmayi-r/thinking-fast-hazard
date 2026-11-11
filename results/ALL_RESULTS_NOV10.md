# Every number so far (Nov 10, 14:40)

Fire, HAZARD test set, 25 kitchens unless stated, value rate = value saved / value available. Clock = RRARA's rule, ceil(seconds x 30) frames charged per decision. Machine in brackets (A15); HAZARD version per column (A29).

| Agent | Current HAZARD d35c71e, clock off | Current, clock on | Paper's HAZARD 5a50c2d, clock off | Paper's, clock on | What the papers report |
|---|---|---|---|---|---|
| random (shipped) | 0.36 [laptop, Oct 28] | not run | not run | 0.401 [Della, 24/25; one scene hit the 55 min cap, simulator hung mid-action] | HAZARD paper: n/a |
| rule (shipped) | 0.141 [Della] | 0.131 [Della] | 0.410 [Della] | 0.383 [Della] | HAZARD 0.53; RRARA Table 1 fire 0.33 -> 0.35 |
| greedy (shipped) | 0.133 [Della] | 0.145 [Della] | 0.270 [Della] | 0.249 [Della] | RRARA 0.08 -> 0.11 |
| MCTS (shipped) | 0.114 [Della] | 0.102 [Della] | 0.750 [Della] | 0.359 [Della] | HAZARD 0.76; RRARA 0.75 -> 0.25 |
| Claude Haiku LLM agent, text (shipped llm) | 0.170 [laptop, Oct 28] | 0.076 [Della, 25/25] | not run | not run | RRARA: GPT-3.5 0.25, GPT-5 0.21 (3-hazard avg, clock on) |
| Claude Haiku + camera image (ours, Oct 28) | 0.17 [laptop] | not run | not run | not run | - |
| Claude Opus, text, 5 kitchens only (Oct 28) | 0.14 [laptop] | not run | not run | not run | - |
| HAZARD rule inside the RRARA wrapper, checker off (wiring check) | - | 0.099 [laptop] | - | - | should equal rule row; within noise |
| RRARA reflex alone, paper's eq. 2 rule, bounded search (ours) | - | 0.199 [Della] | - | 0.757 [Della] | reflex alone never reported |
| same, unbounded search (earlier build) | - | 0.228 [Della] | - | 0.741 [Della] | - |
| RRARA reflex + Haiku checker, "faithful" (paper's accounting: background thinking free) | - | 0.160 [Della, 25/25]; 0.135 [laptop, 25/25] | - | 0.708 [Della, 25/25, A35 loader]; paired vs reflex alone -0.049 (se 0.031); 21 % overrides, 42 replacements illegal at execution | RRARA-v1 0.37, RRARA-v2 0.44 (3-hazard avg) |
| RRARA reflex + Haiku checker, "worldtime" (verdict counts only if it arrived in time) | - | 0.220 [Della, 25/25, timing fix c4b967d]; paired vs reflex alone +0.020 (se 0.022), vs faithful +0.060 (se 0.027); 24 % of verdicts late, 23 % overrides. Earlier 0.147 [laptop] / 0.142 [Della] INVALID (timing bug) | - | not run | not in paper |

Cost per decision under the clock (frames charged per 1500-frame episode): rule ~1 per decision, ~30 per episode; greedy 2-12; MCTS 7-121 per decision, 936 per episode on the paper's version (62 % of the budget); our reflex 0.008-0.02 s, 22-33 per episode; RRARA wrapper 2 % of the budget; Haiku LLM agent 3.95 s per decision, 975 per episode (65 %).

RRARA paper, Table 2 (3-hazard average, clock on): Rule 0.19, Greedy 0.07, GPT-3.5 0.25, GPT-5 0.21, MCTS 0.14, LLM-PrePlanner 0.40, RRARA-v1 0.37, RRARA-v2 0.44.

## Worldtime rerun status (laptop, started 10:00)
Kitchen 1: 0.167. Kitchen 10: 0.000. Kitchens 11 and 12: crashed with numpy "Unable to allocate 12 MiB" / "8 MiB" inside HAZARD's segmentation mask (seg_id.py:65-68) at 10:28 and 10:46, i.e. the laptop ran out of RAM during the Zoom meeting. Kitchen 13 stalled at 10:52 with no process left. No python or TDW process is running now. The sweep has to be restarted from kitchen 11; kitchens 1 and 10 can be kept.
