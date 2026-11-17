# Message to Tom, Nov 17 (before the 11:00 meeting; deck v25 at the usual link)

Hi Tom, here is a written update of the last week so we can spend the meeting on the decision at the end.

The question
A robot has a reflex that answers in no time and a slow brain that may answer better. We charge thinking on the same clock as the world (30 frames per second of wall time, RRARA's rule) and ask whether a gate between the two beats either alone.

The table (HAZARD fire, the paper's snapshot 5a50c2d, 25 test kitchens, clock on, all on Della)
* hand reflex 0.757; RRARA (reflex + Haiku checker) 0.708; Jev (TypeSafe's 0.2 s decision model, HAZARD's own text) 0.718 at 194 frames of thinking per episode; Haiku alone 0.410 at 977 frames of thinking; random 0.401; rule 0.383; MCTS 0.359.
* Jev beats Haiku on all 24 shared kitchens (one crashed inside HAZARD), paired gain 0.31. The slow brain spends two thirds of the episode thinking and lands at random's level: "by the time it's done thinking, the world will be destroyed", measured.
* Nothing beats the reflex. RRARA's checker costs a few points on average; its overrides are never wiser than the rule. A fast model can replace the hand reflex (Jev ties it), which is the one positive.

Why the current snapshot (d35c71e) scores 0.1 for everyone
Three body defects, each found in the code and verified in saved map arrays, each with a flagged fix (off by default) and unit tests:
* The walk loop creeps. Each step sends the body to the path's first corner, usually 13 cm away, with a 50 cm arrival radius, so the body turns, walks 5 cm, reports success, and re-plans; "max steps" is a 100-frame budget, so a walk nets about 1 m and arrives only from 1.5 m. The paper's snapshot has the same loop; its 1.25 m arrival forgives one more metre, which is most of the gap between 0.7 there and 0.1 here. Fix: a lookahead waypoint (2 m per 100 frames measured).
* The mapper records the ceiling as walls (it keeps the highest point over each cell, cut at 2.5 m). Fix: ignore points above body height. Rule agent over 25 kitchens, clock on: 0.230 with the fix vs 0.131 shipped, paired +0.10, better on 18 kitchens.
* The mapper clamps every out-of-map point to the border, so the far-plane pixels around an object's silhouette paint phantom copies of it 25 m away; HAZARD's own prompt reports "distance" as the centre of those cells, so a comb 2 m away reads 30 m. Every brain that reads distances (Haiku, Jev, rule, our honest text) picks targets by phantoms; random reads nothing, which is why random holds up on the current snapshot. Fix: drop out-of-map points, as any range-sensor driver does.
Also found: HAZARD's heading function returns the mirror of the true heading; our honest text had left and right swapped until this morning (fixed, tests).

Honest text and the human ceiling
HAZARD's prompt gives distances in cells, hides why a walk failed, and prints a literal placeholder for history. I rewrote it to say what a navigation stack knows (metres, bearings, feasibility, stop reason, full history, cost of each option). With it Jev saves items on 3a-1 where before it only looked around. Playing the same page by hand with unlimited time, CR and I both score 0.222 on 3a-1; with the three body fixes I reach three items but two have already burned. The worth-5 item on that kitchen sits on a fire source and is gone before frame 50. The ceiling there is the fire, not the brain.

Liane's loop
Built end to end (API, trace converter, compiler, validator, audited loader). Round 1 compiled by Opus from Haiku's traces: 0.204 on five training scenes; round 2, with round 1's program and validation in the prompt: 0.294. Both are on the 20 held-out kitchens this afternoon.

What it shows
On HAZARD fire the best policy is "always run the reflex", and that is the task, not our implementation: nearest valuable item first is the right answer, so deliberation has no decision to improve, and a gate has nothing to protect. This is a clean negative with the reason named, and it gives the specification for a testbed that can show the gate: decisions where the greedy choice is wrong often enough that a slow solver, given time, scores well above a reflex, and where that time costs something. Everything built this month (clock, honest text, Jev, compiled loop, body instrumentation) transfers unchanged.

Decisions for today
1. Keep HAZARD as the latency testbed and report the reflex ceiling plus the three body defects; or
2. Design HAZARD scenes where greedy is wrong (a far high-value item that burns first, items behind doors), and run flood and wind under the clock, which the paper shows the reflex losing; or
3. Move to a benchmark that needs lookahead, as Karthik suggested, and port the clock and the gate to it.

Still running today on Della: Jev with the honest text and with the body fixes on both snapshots, the compiled programs on the held-out kitchens, and the phantom-object fix as its own column set. Every number is labelled by machine and snapshot in NOTES.md; the deck has the tables.
