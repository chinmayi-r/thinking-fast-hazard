# Human baseline on HAZARD fire: the protocol (ASSUMPTIONS.md A36, file 4 of 4)

## What it answers
On the current HAZARD snapshot (d35c71e + TDW 1.13.0, laptop A15, clock off) every policy sits below random: Haiku text 0.170,
greedy 0.114, mcts 0.151, rule 0.075, random 0.356 (`results/ALL_RESULTS_NOV10.md`). Karthik's question is whether that is the
brain or the body. A person who gets exactly Haiku's text and drives exactly Haiku's body (the same menu, the same automatic
pick-up and drop, the same 100-step walks) answers it: if the person beats random, a better brain can win with this body and the
LLM's failure is a reasoning failure; if the person lands with the others, the body (the walks that stop short, A29) caps everyone.
The clock stays off (CR, Nov 12): the person's seconds are not charged, but every decision's wall time and the frames it would cost
at 30 fps are logged, so the sitting also says how long a person would last under RRARA's clock.

## Setup (laptop)
- `G:\iw-thinking-fast` with the venv, the TDW build and HAZARD d35c71e as in `NOTES.md` (Oct 16-28 entries). Nothing else runs.
- Git Bash, from `G:\iw-thinking-fast`:
  ```
  CAMERA=off scripts/human/play.sh <initials>_text
  CAMERA=on  scripts/human/play.sh <initials>_cam
  ```
  One HAZARD process per kitchen; the page opens at http://127.0.0.1:8765/ for the first kitchen and the same tab reconnects for
  the rest (it says "lost the connection to HAZARD" between kitchens and, sometimes for a few seconds, while the simulator
  steps; wait, or reload, both are fine). The score stays on the page for 20 s after each kitchen.
- The TDW build window shows the scene. In the text-only sitting keep it minimised or covered; the page is the only view.

## The sittings
1. **Text-only** (`CAMERA=off`): the person sees what Haiku sees and nothing more. This is the comparison that answers the question.
2. **Camera** (`CAMERA=on`): same five kitchens, the robot's camera frame beside the text. Compares against the Haiku vlm sweep
   (0.168) and shows what the text leaves out. The person has now seen the kitchens once; say so when reporting (order effect), or
   use a second person for this sitting.
3. Five kitchens per sitting: `mm_kitchen_3a-1, -10, -11, -12, -13`, the first five of HAZARD's sorted test set, so they are the
   kitchens `--episodes 5` runs for any other agent. About 10-15 min each on the laptop with a few seconds of thinking per decision.
4. One kitchen at a time, no pausing the simulator (it does not step while the page waits, so there is nothing to pause).
5. The person reads the right panel of the page first (how walks end, pick-up and drop are automatic, the lists are only as
   fresh as the last look, the score rule). No other hints; no second attempt at a kitchen (a kitchen with eval_result.json is
   skipped on rerun; delete its folder to replay it, and say so in the notes).

## What is recorded
Per kitchen, under `results/human_<sitting>/<kitchen>/`: `summary.json` (block `human`: score, every click, decisions / asked /
automatic / rejected inputs, thinking time mean and max; plus the runner's `decision_wall_s`, `frames_at_30fps_total`,
`final_frame_count`), `decisions.jsonl` (one line per decision with wall seconds and frames at 30 fps), `0/human_log.json` and
`0/<frame>_human.json` (the prompt text as shown, the menu, the letter chosen, rejected inputs), `0/<frame>_human_frame.png` when the
camera is on, and HAZARD's own `<frame>_info.json`, `actions.txt`, `eval_result.json`.

## What to compare it with
`TABLE_ONLY=1 scripts/human/play.sh <sitting>` prints the sitting beside the laptop sweeps on the same five kitchens. The sweep
values today (`results/sweep_fire_*_test25`, d35c71e, clock off, value rate per kitchen):

| agent | 3a-1 | 3a-10 | 3a-11 | 3a-12 | 3a-13 | mean of 5 | all 25 |
|---|---|---|---|---|---|---|---|
| random | 0.667 | 0.409 | 0.406 | 0.565 | 0.818 | 0.573 | 0.356 |
| rule | 0.111 | 0.091 | 0.000 | 0.261 | 0.455 | 0.184 | 0.075 |
| greedy | 0.111 | 0.318 | 0.000 | 0.152 | 0.182 | 0.153 | 0.114 |
| mcts | 0.111 | 0.682 | 0.312 | 0.000 | 0.091 | 0.239 | 0.151 |
| Haiku llm, text | 0.000 | 0.091 | 0.312 | 0.000 | 0.636 | 0.208 | 0.170 |
| Haiku vlm, camera | 0.056 | 0.000 | 0.000 | 0.022 | 0.000 | 0.016 | 0.168 |
| auto-clicker smoke (always A, Nov 13) | 0.111 | | | | | | |

Random is high on these five kitchens in particular (0.573 against 0.356 over 25), so a person has to clear a high bar here; quote
both the five-kitchen and the 25-kitchen rows. The reflex (RRARA-v1 off) has no unclocked laptop sweep; its clocked value on
d35c71e is 0.199 (Della).

## Reading the result
- Person well above random on the five: the brain is the gap. The Haiku agent's choices, not the walks, lose the value.
- Person at random or below: the body is the gap on this snapshot; the question moves to the authors' snapshot (5a50c2d), where
  the walks arrive (reflex 0.757 clocked) and random does not help (0.401).
- Either way, `frames_at_30fps_total` over `final_frame_count` says how much of the episode the person's thinking would have
  eaten under the clock (the Nov 13 smoke run: 3297 frames of thinking for a 1593-frame episode, that is, the clock would have
  ended the sitting after about seven decisions).

## Open (A36)
- Five kitchens and one person per sitting are proposals, not measurements of variance; a second person doubles the cost.
- Only d35c71e is installed on the laptop; a sitting on 5a50c2d needs that checkout here with the A35 loader (Della has no display
  for a person). Not done.
- The person sees HAZARD's text as shipped, including the literal `$OBJECT_HISTORY$` line (Nov 11 finding) and distances in
  HAZARD's own units ("356.45 m" at the first decision of the smoke run; `llm.py:432-452`). Both are what Haiku gets; both are on
  the A32 honest-wording list, undecided.
- Clock off is the chosen setting; `CLOCK=tcm` runs the same sitting charged, if ever wanted.
