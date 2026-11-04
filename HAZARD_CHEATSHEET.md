# Driving HAZARD by hand (laptop), the five-minute live demo

Everything below is what I run; file:line references are into `HAZARD/` at commit d35c71e. Open Git Bash in `G:\iw-thinking-fast`.

## 0. One-time environment (already done)
`venv/` (Python 3.10.11, torch 2.1.2+cu118, mmcv 2.1.0, mmdet 3.3.0, tdw 1.13.0.0, gym 0.26.2), `tdw_build/TDW/TDW.exe` (ThreeDWorld 1.13.0), `HAZARD/` (editable install). Set once per shell:
```bash
export PYTHONPATH='G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD' TMP='G:\iw-thinking-fast\.cache\tmp' TEMP='G:\iw-thinking-fast\.cache\tmp'
```

## 1. Hello world: Python starts the simulator, talks to it, shuts it down
Use Git Bash (in the app's PowerShell tab, type `& "C:\Program Files\Git\bin\bash.exe"` first; the prompt becomes `$`).
```bash
./venv/Scripts/python.exe scripts/tdw_hello.py --port 1071   # starts TDW.exe itself, connects, steps 3 empty frames, prints Hello world, kills the build
```
Do NOT also start `TDW.exe` by hand before this; the script does it, and two builds on one port fight. If you only want to see the window idle: `./tdw_build/TDW/TDW.exe "-port 1071" &` (press Enter if the `$` prompt seems missing afterwards; close the window to stop it).
Why by hand: HAZARD constructs its controller with `check_version=False` (`src/HAZARD/challenge.py:65-74`), and under that flag TDW never downloads or launches a build (`tdw/controller.py:60-70`); it just waits on the port.

## 2. What a scene is
```bash
ls HAZARD/src/HAZARD/data/room_setup_fire/ | head          # 100 fire/flood scenes: mm_craftroom_2a-1 ... mm_kitchen_2a-25, plus test_set/
ls HAZARD/src/HAZARD/data/room_setup_wind/ | head          # 80 wind scenes: suburb_scene_2023-*
ls HAZARD/src/HAZARD/data/room_setup_fire/mm_craftroom_2a-1/
head -c 600 HAZARD/src/HAZARD/data/room_setup_fire/mm_craftroom_2a-1/log.txt   # the TDW command log that rebuilds the room, one JSON list per frame
cat HAZARD/src/HAZARD/data/room_setup_fire/info.txt
cat HAZARD/src/HAZARD/data/meta_data/value.json            # object category -> value (the score)
head -c 400 HAZARD/src/HAZARD/data/meta_data/temperature.json   # per-category ignition/burn parameters
```
A scene = a replayed command log (room, furniture, 76 asset bundles streamed from S3 on first load) + target objects + where the hazard starts.

## 3. What an episode is (the loop, in the code)
- `HAZARD.submit(...)` -> `run_experiments.py:89-200` builds the agent and a `Challenge`, iterates scenes.
- `challenge.py:236-279`: reset, load scene (70 to 90 s on the laptop), list targets.
- `challenge.py:293-420`: the loop. `state = controller._obs()`; `current_action = agent.choose_target(state, processed_input)` (line 318); then the action runs for many frames via `policy/env_actions.py` (`agent_walk_to`, `agent_pickup`, `agent_drop`, `agent_explore`).
- One `controller.communicate()` = one frame (`envs/fire/fire.py:219`). Episode ends at `frame_count >= max_steps` (1500 fire/flood, 3000 wind, `challenge.py:53-59`) or when all targets are rescued.
- Fire spreads on its own every frame (`envs/fire/manager.py`); nothing waits for the agent, but nothing charges the agent's thinking time either: the loop blocks on `choose_target`.

## 4. Run an agent (the command I use for everything)
```bash
cd HAZARD
../venv/Scripts/python.exe -X utf8 ../scripts/run_baseline.py --env fire --agent rule --episodes 1 --port 10826
```
- `--agent rule | greedy | mcts | mctsv2 | random` (shipped: `run_experiments.py:26-85`), `--env fire | flood | wind`.
- The runner starts the build itself, wraps the agent to time every decision, and writes `results/<env>_<agent>_ep1_<stamp>/`:
  - `decisions.jsonl`: one line per decision: wall seconds, frames = ceil(s x 30), frame_count before the decision, the action.
  - `summary.json`: wall time, decisions, latency mean/max, eval results, final frame.
  - `<scene>/eval_result.json` (HAZARD's own score: `avg_score`, `avg_max_score`, `target_status`), `<scene>/actions.txt`, `<scene>/0/` observation dumps for llm/mcts.
- Add `--clock tcm` to charge thinking time RRARA-style (world advances ceil(t x 30) frames after each decision), or `--clock k --k 60` for a declared budget.
- LLM agent (needs a key file): `--agent llm --lm_source anthropic --lm_id claude-opus-5-5 --api_key_file "G:\iw-thinking-fast\.secrets\anthropic_key.txt"`.

## 5. Read the results
```bash
cat results/fire_rule_ep1_*/summary.json
cat results/fire_rule_ep1_*/mm_craftroom_2a-1/eval_result.json
cat results/fire_rule_ep1_*/mm_craftroom_2a-1/actions.txt
python -c "import json;[print(json.loads(l)) for l in open(r'results/fire_rule_ep1_20261016_144416/decisions.jsonl')]"
```
Value rate = avg_score / avg_max_score. HAZARD's own metric script for a whole output dir: `python HAZARD/src/HAZARD/utils/calc_value.py <output_dir> fire` (`submit.md:71-84`).

## 6. Other knobs of `submit()` (`documentation/get_started/submit.md:13-44`)
`env_name`, `agent` (string or your own object with `agent_type`, `reset`, `choose_target`), `data_dir` (one specific scene), `max_test_episode`, `run_on_test` (held-out set), `perceptional` (Mask R-CNN instead of ground-truth segmentation; needs the 341 MB checkpoint), `effect_on_agents` (hazard hurts the agent), `use_cached_assets` (local S3 cache), `record_with_agents` (frames for a video; `ffmpeg -i img_%04d.jpg ...`, `common_utils.md:27`).

## 7. Things I changed in HAZARD (both as diffs in `scripts/`)
- A16: `challenge.py:322-331` guard so `pick_up` with no prior successful walk logs instead of crashing (`KeyError: -1`).
- A18: `policy/llm.py` Anthropic source + OpenAI SDK 1.x fix + per-call latency log.

## 8. Numbers so far (laptop, fire, one scene, one episode each; `NOTES.md` has the tables)
Frames per action: explore 49, walk_to 40 (success) / 106 (fail), pick_up 18, drop 18. Decision latency: rule 0.1 ms, greedy 0.23 s, mcts 2.8 s mean (5.1 s max), so mcts would be charged 1926 frames in a 1500-frame episode.
