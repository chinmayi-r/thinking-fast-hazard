# thinking-fast-hazard

Fall 2026 independent work, Chinmayi Ramasubramanian (advisor Tom Silver, co-advisor Karthik Narasimhan).
Baselines on HAZARD (Zhou et al., ICLR 2024) with per-decision latency logged, as the first step of "Thinking Fast Enough".

## Layout

| Path | Contents |
|---|---|
| `NOTES.md` | Lab notebook, append-only: every command, number and crash, Oct 6 to Nov 3 2026. |
| `ASSUMPTIONS.md` | Assumptions A1 to A25, the preconditions of each lit-review method, and whether it ports to HAZARD. |
| `BASELINE.md` | How the baselines are run; the clock convention frames = ceil(seconds x 30). |
| `HAZARD_CHEATSHEET.md` | How HAZARD works, with file and line citations. |
| `scripts/` | `run_baseline.py` (runner, timing wrapper, clock), sweep drivers, `vlm_agent.py`, `aggregate_sweeps.py`, `make_videos.py`, and the diffs applied to HAZARD (`A16_*`, `A19_*`, `A18_*`). |
| `results/sweep_*/<scene>/` | `actions.txt`, `eval_result.json`, `output.log` (HAZARD's own output); `decisions.jsonl`, `summary.json` (per-decision wall-clock from the wrapper). PNGs and per-frame TDW logs omitted for size. |
| `results/aggregate_laptop.md` | Aggregated table. |
| `results/videos/` | Recorded episodes on `mm_kitchen_3a-1`: MCTS, Claude Haiku with camera image. |
| `results/hazard_fire_test25_repro_logs_2026-11-03.zip` | Log bundle sent to the HAZARD authors Nov 3: full logs for rule, greedy, MCTS, random on the fire test set, plus the two patches. |
| `logs/` | Driver and per-run stdout. |

Not included: the HAZARD checkout (github.com/zhouqqhh/HAZARD, commit `d35c71e`, with `scripts/A16_*.diff` and `scripts/A19_*.diff` applied), ThreeDWorld 1.13.0, Python environments, API key.

## Fire test set, 25 scenes, one episode each, no clock, laptop (RTX 3060)

Value rate = avg_score / avg_max_score from `eval_result.json`, averaged over scenes. Decision time = wall-clock per `choose_target` call.

| Agent | Value rate | Decisions / ep | Mean decision s | Max decision s |
|---|---|---|---|---|
| rule | 0.075 | 18.4 | 0.0001 | 0.0001 |
| greedy | 0.114 | 19.5 | 0.33 | 0.86 |
| MCTS | 0.151 | 19.6 | 4.27 | 17.0 |
| random | 0.356 | 53.8 | 0 | 0 |
| Claude Haiku 5.5, text | 0.170 | 45.2 | 1.67 | 14.4 |
| Claude Haiku 5.5, text + image | 0.168 | 32.8 | 2.50 | 37.3 |
| Claude Opus 5.5, text (5 scenes) | 0.138 | 18.4 | 6.48 | 11.9 |

Paper (fire): rule 53%, MCTS 76%. Correspondence with the authors about the gap: `NOTES.md`, Nov 3.

## Run

From the HAZARD checkout with the venv active and a TDW build:

```
python ../scripts/run_baseline.py --env fire --agent rule --episodes 1 --run_on_test
python ../scripts/run_baseline.py --env fire --agent mcts --episodes 1 --run_on_test --clock tcm --fps 30
python ../scripts/run_baseline.py --env fire --agent llm --lm_source anthropic --lm_id claude-haiku-5-5 --api_key_file <path> --episodes 1 --run_on_test
```

Every run's exact command is in `NOTES.md`.

## Next (Nov 3 meeting, Tom Silver and Liane Galanti)

1. RRARA baseline on HAZARD (reflex rules, asynchronous LLM planner with staleness-aware cache, preplanner), under the clock.
2. Variation: the background process synthesises a Python policy instead of a plan, with a switch rule for when to use it.
