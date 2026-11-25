# The fire task (firelab)

**What this is.** A robot in a room where a fire spreads. It picks up items and carries them to a safe strip before
the room burns. The fire keeps spreading while the robot thinks, so every second of thinking has a price.
It runs on the lab's kinder simulator (kindergarden at commit `86c55a4`, pybullet).

**What it is for.** To study when thinking is worth its time. A simple rule that never thinks and a strong model
that thinks slowly should both fall short, so that there is something to gain from choosing between them.

**Where things are.**
- The task and the agents: this folder
- Every run: [results/](results/)
- What a program-writer reads: [PROGRAM_API.md](PROGRAM_API.md)
- Plan, design and log, one level up: [PLAN.md](../PLAN.md), [DESIGN.md](../DESIGN.md), [NOTES.md](../NOTES.md), [ASSUMPTIONS.md](../ASSUMPTIONS.md)

## The room

- A grid of 16 columns by 15 rows; one cell is 0.2 m; one tick is 0.1 s.
- One partition across row 5 with one doorway. Rows 11 and above are the safe strip.
- 8 items with values 1, 2, 3, 5 or 8. An item on a burning cell loses 2 % of its value per tick until it is picked.
- The robot has health 100 and loses 2 per tick while its cell burns.
- An episode ends when half the floor burns, after 500 ticks, or at health 0.
- Score = value banked on the safe strip / value on the floor at the start.
- A room is named by a number (its seed). The same number is the same room every time.

## The two versions of the room

| name | file | what it is |
|---|---|---|
| the rule check | [gate.py](gate.py) | a small board model of the rules, fast, used to choose the rule set |
| the simulator | [env.py](env.py) | the actual task on kinder and pybullet |

## The agents

| name on the command line | what it does |
|---|---|
| `random` | random commands |
| `reflex` | a simple rule: get an item, bring it home, never enter fire |
| `always` | a planner that plans before every command |
| `commit` | a planner that plans once and follows the plan |
| `moving` | the reflex acts while the planner plans |
| `model` | a language model is asked before every command; the robot waits for the answer |
| `rrara1` | the rule acts and a language model checks its action in the background (RRARA, version 1) |
| a program | a Python file with no model call, see [PROGRAM_API.md](PROGRAM_API.md) and [examples/](examples/) |

The clock: thinking time is charged in ticks, ticks = ceil(seconds x 10). `--clock off` charges nothing,
`declared` charges a stated cost, `measured` charges the real time.

## How to run it

Set up the environment once with [scripts/della_firelab_setup.sh](../scripts/della_firelab_setup.sh). Then, from the
top of the repository with `PYTHONPATH` set to it:

```
python -m pytest -q firelab
python -m firelab.run_rooms --tag my_run --agents random reflex --first 0 --rooms 5
python -m firelab.run_rooms --tag my_planner --agents always --clock declared --first 0 --rooms 5
python -m firelab.validate --program firelab/examples/reflex_program.py --first 50 --rooms 5 --out my_program.json
```

A run named `my_run` writes into `firelab/results/firelab_sim/my_run/`. A model agent also needs `--model` and
`--key-file`; `python -m firelab.run_rooms --help` lists every setting.

## Where the results are

| folder | what is in it |
|---|---|
| [results/firelab_sim/](results/firelab_sim/) | the simulator runs. One folder per run, with one row per room and agent in `rows_*.csv`. The saved games are not in git. |
| [results/firelab_gate/](results/firelab_gate/) | the rule check's files |
| [results/firelab_compiled/](results/firelab_compiled/) | programs written before the game by a model, and their test files |

Scripts that read these files and play nothing:
- [scripts/all_agents_table.py](../scripts/all_agents_table.py): every agent on rooms 0 to 49 in one table
- [scripts/gap_table.py](../scripts/gap_table.py): every run against the reflex
- [scripts/liane_summary.py](../scripts/liane_summary.py): a program on rooms 50 to 99 beside the other agents
- [scripts/run_summary.py](../scripts/run_summary.py), [scripts/model_run_summary.py](../scripts/model_run_summary.py): one run in detail
