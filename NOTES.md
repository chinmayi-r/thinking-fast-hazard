# NOTES

Exact commands that worked, what broke, and measured facts (frames per high-level action, decision latencies). Newest at the bottom.

## Oct 16: step 1a, requirements survey (read only, nothing installed)

Sources: HAZARD GitHub `main` (HEAD pushed 2026-07-23; `UMass-Foundation-Model/HAZARD`, `zhouqqhh/HAZARD` and `UMass-Embodied-AGI/HAZARD` all resolve to the same repo) and TDW GitHub `master`. Raw copies with line numbers were read in the session scratchpad, not kept.

| Requirement | Value | Source |
|---|---|---|
| OS | Windows 10 or 11 supported (also Ubuntu 18/20/22, macOS) | `tdw/Documentation/lessons/setup/pc.md:11` |
| Python (TDW) | 3.8+ | `pc.md:12` |
| Python (HAZARD) | `>=3.7` declared; effective floor 3.10 from torch, ceiling 3.11 from mmcv Windows wheels | `HAZARD/pyproject.toml:13`; PyPI torch `requires_python>=3.10`; openmmlab wheel index cu118/torch2.1 lists cp38 to cp311 only |
| GPU | "A GPU, the faster the better"; NVIDIA drivers work better; possible without GPU but slower and less realistic | `pc.md:13`, `pc.md:16-18` |
| Display | The build "is a windowed application that requires a display and a GPU" | `tdw/Documentation/lessons/setup/install.md:7` |
| TDW build | Auto-downloaded on first `Controller()` to `~/tdw_build`; Windows zip 111.4 MB (v1.13.0, 2024-05-31, LTS) | `tdw/Python/tdw/backend/paths.py:19-20`; `core_concepts/launch_build.md:35`; GitHub release assets; `tdw/README.md:7` |
| HAZARD build launch | `check_version=False` means TDW never downloads or launches the build; it waits for one on the port | `HAZARD/src/HAZARD/challenge.py:65-74`; `tdw/Python/tdw/controller.py:60-76` |
| Python deps | numpy, tdw, opencv-python, mmcv>=2.0.0rc1, torch, mmdet, gym, pyastar2d, tiktoken, openai, backoff, inflect, streamlit; all unpinned except mmcv | `HAZARD/requirements.txt:1-13` |
| Hard import chain | torch, mmcv, mmdet imported at module level by the fire controller regardless of perception flag | `src/HAZARD/envs/fire/fireagent_controller.py:21-22,30`; `src/HAZARD/utils/vision.py:4-11` |
| Perception checkpoint | `tdw_rcnn.pth`, 341 MB, Google Drive, to `src/HAZARD/data`; loaded only when `use_gt=False` | `install.md:26`; `fireagent_controller.py:57-59`; `utils/vision.py:25` |
| Scene/model assets | Streamed from AWS at run time; optional local cache via `use_cached_assets` into `os.getcwd()/data/assets`; size undocumented | `common_utils.md:31-35`; `submit.md:27`; `fireagent_controller.py:89-92` |
| Repo data | 631 files, 10.6 MB under `src/HAZARD/data` (room setups for fire/flood/wind, test sets) | GitHub tree API |
| Episode counts | fire 25, flood 25, wind 20 test episodes | `submit.md:116` |
| Render size | 512x512 | `challenge.py:47` |
| Port | any free port (`1071` in docs, `10826` in `starter_code.py:4`) | `submit.md:19` |
| LLM agents | OpenAI API key, `gpt-3.5-turbo`/`gpt-4` | `submit.md:34-38`; `policy/llm.py:3` |
| Optional | ffmpeg for video; openmim if mmcv pip build fails; OpenAI baselines for RL training | `common_utils.md:27`; `install.md:20-24,38` |

Machine match (facts from `CLAUDE.md`): OS matches. GPU present and NVIDIA, matches; VRAM 4 GB is a watch item for perception and rules out local LLMs. Display present, matches. Python 3.14 on PATH does not match; a 3.10 or 3.11 interpreter is needed. Disk: C (1.1 GB) conflicts with TDW's hard-coded `~/tdw_build` and the default pip/torch/HF caches; G (100 GB) is fine for everything if caches are redirected. Network to GitHub, PyPI and AWS is required. Open assumptions A1 to A13 are in `ASSUMPTIONS.md`.

## Oct 16: step 1b, laptop install on G (A1 to A13 approved by CR at 13:45)

Machine correction: `nvidia-smi` reports the RTX 3060 Laptop GPU with 6144 MiB, driver 591.86, not the 4 GB in `CLAUDE.md`.

What ran, in order (Git Bash from `G:\iw-thinking-fast`):

```bash
# Python 3.10.11, official NuGet package, extracted to G (the Store 3.10 fails to launch from a non-interactive shell, 0x80070005)
curl -sSL -o python310_dl/python.3.10.11.nupkg "https://www.nuget.org/api/v2/package/python/3.10.11"
unzip -q -o python310_dl/python.3.10.11.nupkg -d python310
./python310/tools/python.exe -m venv "G:\iw-thinking-fast\venv"
PIP_CACHE_DIR='G:\iw-thinking-fast\.cache\pip' ./venv/Scripts/python.exe -m pip install --upgrade pip wheel setuptools

# HAZARD at commit d35c71ebd24c08783552bab246df8ac82190e1e2 (2026-07-23), 18 MB without .git
git clone https://github.com/UMass-Foundation-Model/HAZARD.git HAZARD

# TDW build v1.13.0 (zip 111,380,671 bytes; extracted 238 MB; TDW/version.txt = 1.13.0.0), kept off C per A3
curl -sSL -o tdw_build/TDW_Windows.zip https://github.com/threedworld-mit/tdw/releases/download/v1.13.0/TDW_Windows.zip
unzip -q -o tdw_build/TDW_Windows.zip -d tdw_build

# Python packages: scripts/install_laptop.sh, log in logs/install_laptop.log
bash scripts/install_laptop.sh > logs/install_laptop.log 2>&1

# Smoke test (build started by the script, check_version=False, launch_build=False as in HAZARD)
./venv/Scripts/python.exe scripts/tdw_hello.py --port 1071
```

Layout: `venv/`, `python310/`, `HAZARD/`, `tdw_build/TDW/`, `.cache/{pip,torch,hf}`, `logs/`, `scripts/`.

## Oct 16: step 1c, Adroit probe (CR typed, Claude read the tab)

Login node `adroit5`: no NVIDIA driver, `DISPLAY` unset, no `vglrun`/`xpra`, has `Xvfb` and `glxinfo`; anaconda3 modules 2020.2 to 2025.6; internet to github.com and pypi.org OK.

Vis node `adroit-vis` (same probe line):
- 2x NVIDIA A100 80GB PCIe, driver 610.57.04
- `DISPLAY` unset over plain SSH; `/usr/bin/vglrun`, `/usr/bin/xpra`, `/usr/bin/Xvfb`, `/usr/bin/glxinfo` all present (TDW's remote recipe, `tdw/Documentation/lessons/remote/xpra.md:5-20`, needs exactly xpra + VirtualGL)
- quota: home 15.7 of 19.5 GiB (nearly full, keep everything on scratch); `/scratch/network/cr7998` 102.6 of 293 GiB; filesystem 12 TB free
- internet: github.com and pypi.org return HTTP 200

Next on Adroit: `scripts/adroit_tdw_hello.sh` (conda 3.10 env on scratch, tdw 1.13.0.0, Linux build on scratch, Xvfb :99 plus `vglrun -d :0` if a GPU X server exists, hello-world controller on port 10826).

## Oct 16: step 1c, Adroit hello-world attempt 1 (CR pasted `scripts/adroit_tdw_hello.sh`, output pasted back)

- `adroit-vis` is RHEL 9.8. Conda env `python=3.10.22` created at `/scratch/network/cr7998/iw-thinking-fast/py310`; `tdw 1.13.0.0` installed; TDW Linux build v1.13.0 extracted to `.../tdw_build/TDW/`, `version.txt` = 1.13.0.0.
- X sockets present: X1 to X15 plus per-user ones (X1640, X1953, ...). No `:0`, so `vglrun -d :0` is not an option; `glxinfo` on Xvfb :99 reports `llvmpipe` (software). Next try: VirtualGL's EGL back end, `vglrun -d egl`.
- **Failure:** `from tdw.controller import Controller` raised `ModuleNotFoundError: No module named 'pkg_resources'` at `tdw/librarian.py:3`. Cause: conda's pip pulled setuptools 84.0.0, which no longer ships `pkg_resources`. Fix: `pip install "setuptools<81"`. The laptop venv also got setuptools 84.0.0 (see `logs/install_laptop.log`), so the same pin went into `scripts/install_laptop.sh`.
- Side effect: the build process started before the import failed and stayed alive, holding `tee`'s pipe, so the shell looked hung. Recovery: Ctrl-C, `pkill -u cr7998 -f TDW.x86_64; pkill -u cr7998 Xvfb`.

## Oct 16: step 1c, Adroit hello-world attempt 2: SUCCESS, GPU rendering on adroit-vis without sudo

Commands (CR pasted on `adroit-vis`, after Ctrl-C and killing the orphaned build):

```bash
pkill -u cr7998 -f TDW.x86_64; pkill -u cr7998 Xvfb
cd /scratch/network/cr7998/iw-thinking-fast
./py310/bin/python -m pip install -q "setuptools<81"
Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 &
sleep 2; export DISPLAY=:99
vglrun -d egl glxinfo -B | grep renderer          # -> OpenGL renderer string: NVIDIA A100 80GB PCIe/PCIe/SSE2
timeout 300 ./py310/bin/python tdw_hello_adroit.py ./tdw_build/TDW/TDW.x86_64 vglrun -d egl
pkill -u cr7998 -f TDW.x86_64; pkill -u cr7998 Xvfb
```

Result:
- `connected after 6.42s; build 1.13.0 unity 2020.3.24f1`
- empty `communicate([])`: 2.4 ms, 1.1 ms, 2.6 ms
- `Hello world!`, exit 0
- Unity `Player.log`: `Renderer: NVIDIA A100 80GB PCIe/PCIe/SSE2`, `Vendor: NVIDIA Corporation`, `OPENGL LOG: Creating OpenGL 4.5 graphics device`

So the recipe is Xvfb for the window plus VirtualGL's EGL back end for the GPU (`vglrun -d egl`), which needs no `:0` X server, no MyAdroit desktop, and no sudo. `scripts/adroit_tdw_hello.sh` now does this automatically. Still unmeasured on Adroit: asset streaming from AWS (compute nodes have no internet; the vis node does), scene load time, frames per HAZARD action.

## Oct 16: step 1b result, laptop install done and TDW + HAZARD smoke tests pass

`scripts/install_laptop.sh` finished with exit 0 (`logs/install_laptop.log`; first attempt `logs/install_laptop_attempt1_tempOnC.log` failed because pip's temp dir was on C). Final versions: numpy 1.26.4, torch 2.1.2+cu118 (CUDA available, 11.8), torchvision 0.16.2, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0, tdw 1.13.0.0, gym 0.26.2, opencv-python 4.11.0.86, pyastar2d 1.1.4, openai 3.31.0, HAZARD 0.0.1 (editable), setuptools 80.10.2 (pinned <81). `pip check` reported nothing.

TDW hello-world (`scripts/tdw_hello.py --port 1071`, log `logs/tdw_hello_laptop.log`): build started from `G:\iw-thinking-fast\tdw_build\TDW\TDW.exe`; TDW printed "You need to launch your own build." as expected with `check_version=False`; controller connected after 20.44 s (build cold start); build 1.13.0, Unity 2020.3.24f1; empty `communicate([])` 2.8 / 2.2 / 2.4 ms; Hello world; exit 0. `Player.log`: Direct3D 11.0, Renderer NVIDIA GeForce RTX 3060 Laptop GPU.

HAZARD import (`from HAZARD import submit`, cwd `HAZARD/`, `PYTHONPATH=G:\iw-thinking-fast\HAZARD;G:\iw-thinking-fast\HAZARD\src\HAZARD`): OK in 27.2 s (torch + mmdet import cost). Warnings only: pkg_resources deprecation (tdw), gym unmaintained notice, "Overriding environment fire-v0 / flood-v0 already in registry" (the registration issue named in `install.md:32`; harmless).

Compare: Adroit vis node connected in 6.42 s with the same empty-step cost (1 to 3 ms). Per A15, each number above is laptop-only.

## Oct 16: A16 applied (CR approved 15:0x): guard in HAZARD `src/HAZARD/challenge.py:322-331`

Diff saved as `scripts/A16_challenge_pickup_guard.diff` (apply with `git -C HAZARD apply ../scripts/A16_challenge_pickup_guard.diff` on a fresh clone; Adroit's clone does not have it yet). Behaviour is unchanged when `nearest_object` is valid; when it is None the action log says `none None` instead of crashing, and HAZARD's own execution branch (`challenge.py:380-383`) then reports "You need to walk to an object first."

## Oct 16: first full HAZARD episode with measurements (laptop, fire, scene mm_craftroom_2a-1, rule agent, no clock)

Command (Git Bash, cwd `HAZARD/`, PYTHONPATH as in A10): `../venv/Scripts/python.exe -X utf8 ../scripts/run_baseline.py --env fire --agent rule --episodes 1 --port 10826`
Output: `results/fire_rule_ep1_20261016_144416/{summary.json, decisions.jsonl, mm_craftroom_2a-1/eval_result.json, mm_craftroom_2a-1/actions.txt}`, log `logs/baseline_fire_rule_ep1.log`.

- Wall clock 449.4 s for one episode: scene reset 70.6 s, then 18 decisions over 1536 frames (`max_steps` 1500). About 0.25 s of wall time per simulated frame on the laptop, i.e. the simulator runs at roughly 4 FPS wall while the clock convention calls a frame 1/30 s (A17).
- Decision latency of the rule agent: mean 87 microseconds, max 175 microseconds. At 30 FPS that is 1 frame by the ceiling rule (RRARA reports it as 0.00 s).
- Outcome: 1 of 11 targets rescued (`avg_score` 1.0 of `avg_max_score` 19, so value rate 0.053 on this scene); the agent spent 12 consecutive failed `walk_to` calls on object 83. RRARA's rule-agent average over 25 scenarios is 0.19; one scene is not comparable.
- `actions.txt` prints `step 0` for every action because `comm_counter` never increments on this path (`fireagent_controller.py:469` is not reached); `frame_count` is the usable counter.

**Frames per high-level action, measured (the number CLAUDE.md said is undocumented):**

| action | result | n | frames mean | min | max |
|---|---|---|---|---|---|
| explore | success | 2 | 49.0 | 49 | 49 |
| walk_to | success | 2 | 40.0 | 39 | 41 |
| walk_to | fail (max steps reached) | 12 | 106.6 | 102 | 120 |
| pick_up | success | 1 | 18.0 | 18 | 18 |
| drop | success | 1 | 18.0 | 18 | 18 |

Reading: a successful short walk is about 40 frames (1.3 s of world time at 30 FPS); a failed walk burns about 106 frames (3.5 s), which is the walk's 100-step cap (`challenge.py:340`, `max_steps=100`) plus overhead; pick up and drop are 18 frames each. For scale, one 2.5 s LLM call under RRARA's clock is 75 frames, about two successful walks or 5 percent of the episode.

## Oct 16: step 1 reference numbers, laptop, fire, scene mm_craftroom_2a-1, one episode each, no clock

Runs: `scripts/laptop_baseline_queue.sh "greedy mcts random" fire 1` (same runner as the rule run); results in `results/fire_<agent>_ep1_<stamp>/`.

| agent | wall s | decisions | final frame | rescued | value rate (score/max 19) | decision latency mean / max (s) | frames charged if clocked at 30 FPS |
|---|---|---|---|---|---|---|---|
| rule | 449 | 18 | 1536 | 1/11 | 0.053 | 0.0001 / 0.0002 | 18 |
| greedy | 502 | 18 | 1538 | 2/11 | 0.105 | 0.229 / 0.366 | 132 |
| mcts | 612 | 23 | 1505 | 3/11 | 0.368 | 2.771 / 5.119 | 1926 |
| random | TBD | | | | | | |

MCTS decision time grows over the episode (0.88, 1.19, ... 4.59, 5.12 s) as its internal model accumulates state. Its 1926 charged frames exceed the 1500-frame episode: under RRARA's clock MCTS would spend the entire episode thinking, which is the mechanism behind RRARA's Table 2 (MCTS 0.14 below rule 0.19 under TCM) and Tom's "by the time it's done thinking, the world will be destroyed." Unclocked, the ranking is the paper's (MCTS above greedy above rule).

Frames per action across the three agents (success cases): walk_to 11 to 55 (mean 33 to 41), pick_up 14 to 42, drop 17 to 64, explore 49 every time; a failed walk_to is 90 to 120 (mean 103 to 107). All three agents burned most of the episode on repeated failed walks to one unreachable object (rule: 12 of 18 decisions; greedy: 12 of 18; mcts: 9 of 23), which is where a smarter agent gains.

Caveats: one scene, one seed each; machine is the laptop (A15); random pending.

## Oct 16: step 1, random agent row (laptop, fire, mm_craftroom_2a-1, A16 guard applied)

random: wall 641 s, 50 decisions, final frame 1517, rescued 5/11, value rate 0.474 (score 9 of 19), decision latency 0.00 s, 37 actions succeeded and 13 failed (the guard turned the old `KeyError` crash into "You need to walk to an object first"), `results/fire_random_ep1_20261016_151148/`.

Reading: on this one scene the random agent beats mcts (0.368), greedy (0.105) and rule (0.053) because the three deterministic agents lock onto one unreachable target and repeat a failed 100-step walk until the episode ends (12 of 18, 12 of 18, 9 of 23 decisions), while random keeps moving. That is a property of the shipped agents on this scene, not evidence about thinking time; it also means single-scene numbers are noisy and the 25-scenario averages (RRARA, the HAZARD paper) are the only comparable ones.

Step-1 table, laptop, fire, mm_craftroom_2a-1, one episode each, no clock:

| agent | wall s | decisions | rescued | value rate | decision latency mean / max (s) | frames charged if clocked at 30 FPS |
|---|---|---|---|---|---|---|
| rule | 449 | 18 | 1/11 | 0.053 | 0.0001 / 0.0002 | 18 |
| greedy | 502 | 18 | 2/11 | 0.105 | 0.229 / 0.366 | 132 |
| mcts | 612 | 23 | 3/11 | 0.368 | 2.771 / 5.119 | 1926 |
| random | 641 | 50 | 5/11 | 0.474 | 0.0000 / 0.0001 | 50 |

## Oct 16: A18 applied (CR: "fix up the llm thing, I'll give you key"), diff in `scripts/A18_llm_anthropic_and_openai_v1.diff`

Patch script `scripts/patch_A18_llm.py` (idempotent) edits `HAZARD/src/HAZARD/policy/llm.py`:
- new `source='anthropic'` using the official Anthropic SDK 1.15.0 (`anthropic.Anthropic(api_key=...)`, `client.messages.create(model, max_tokens, messages, output_config={'effort': ...})`). Claude 5.5 models reject `temperature`/`top_p`, so none are sent; thinking is always on, so `max_tokens` is raised to at least 4096 and effort defaults to `low` (env `HAZARD_ANTHROPIC_EFFORT` overrides); a `refusal` stop reason yields an empty answer so HAZARD's `parse_answer` falls back instead of crashing; prompt-cut guard uses cl100k_base as an approximation with a 100k limit.
- OpenAI branch moved from `openai.ChatCompletion.create` (SDK < 1.0) to `openai.chat.completions.create` / `openai.completions.create` with attribute access; unknown model ids fall back to the cl100k_base tokenizer.
- every API call appends `{wall_s, input_tokens, output_tokens, stop_reason, model}` to `agent.call_log`; `scripts/run_baseline.py` writes these to `llm_calls.json` and summarises them in `summary.json["llm"]`.
- `backoff` retries on both SDKs' errors, max 6 tries; key rotation (`sleep()`) rebuilds the Anthropic client.
- Dry construction (no API call, dummy key) succeeds for both sources from cwd `src/HAZARD`; the runner now passes the prompt template by absolute path so cwd does not matter.

Key handling: CR creates `G:\iw-thinking-fast\.secrets\anthropic_key.txt` (one line); the runner passes the path via HAZARD's own `api_key_file` parameter (`run_experiments.py:28-33` reads it). Claude never reads or echoes the file. Cost: Opus 5.5 list price $4 / $20 per 1M tokens; HAZARD makes 2 calls per decision (CoT + answer); estimate per episode is recorded in `summary.json`.

Run (laptop, Git Bash, from `HAZARD/`): `../venv/Scripts/python.exe -X utf8 ../scripts/run_baseline.py --env fire --agent llm --lm_source anthropic --lm_id claude-opus-5-5 --api_key_file "G:\iw-thinking-fast\.secrets\anthropic_key.txt" --episodes 1 --port 10826` (add `--clock tcm` for the charged version).

## Oct 16: step 2a, first clocked results and two incidents (laptop, fire, mm_craftroom_2a-1, `--clock tcm --fps 30`)

- rule under the clock: wall 666 s, 14 decisions, 14 frames charged (1 per decision, latency 0.1 ms), final frame 1533, rescued 0/11, value rate 0.000 (`results/fire_rule_tcm30_ep1_20261016_152303/`). Unclocked it rescued 1/11; the difference is 14 frames of charge plus run-to-run variation, not a clock effect. RRARA Table 2 has rule 0.19 at 0.00 s on 25 scenarios.
- greedy under the clock: Python segmentation fault after decision 7 (frame 429, 78 frames charged so far, right after a successful drop while reading the next 1024x1024 observation), `logs/baseline_fire_greedy_tcm30_ep1.log`. Cause unknown; it coincided with a second simulator being started by hand for the hello-world demo on the same GPU. Rerun queued.
- mcts under the clock: its simulator window disappeared about 40 s after the scene load began (15:39), most likely closed by hand while two TDW windows were open; HAZARD's controller then waited on the dead socket forever and was killed at 15:43. Rerun queued.
- Correction: CR's hello-world at 15:39 (PowerShell) succeeded (connected in 3.28 s, Hello world, prompt returned); the process I killed at 15:43 was the spare simulator window from a manual `TDW.exe` line PowerShell had started in the background. Two windows were open at once, which is the likely reason the queue's mcts window got closed. Lesson recorded in `HAZARD_CHEATSHEET.md`: one simulator at a time on the laptop, never close a TDW window during a run.

## Oct 16 16:03: LLM baseline attempt 2, blocked by key scope (no spend)

Reboot at 15:57 was a Windows Update restart (System log: shutdown.exe 15:54:53, TrustedInstaller 15:57:13), not a simulator crash. Free RAM during runs was 0.5 to 0.8 GB of 15.7 GB (browser, app, Defender), the likely cause of the slow loads, the greedy segfault and the hangs; close heavy apps before runs.

Claude Opus 5.5 episode, attempt 2 (`logs/baseline_fire_llm_opus_ep1_attempt2.log`, `results/fire_llm_ep1_20261016_160047/`): scene reset 69 s, first decision reached, API returned 400 "This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header". Six retries, exit 1, zero decisions, zero cost. Fix A18b (`scripts/patch_A18b_workspace.py`, diff v2): the Anthropic client sends `anthropic-workspace-id` from env `ANTHROPIC_WORKSPACE_ID` when set. CR can instead create a workspace-scoped key in the Anthropic Console.

## Oct 16 16:19: step 2a complete, shipped agents under RRARA's clock (laptop, fire, mm_craftroom_2a-1, one episode each, `--clock tcm --fps 30`)

| agent | unclocked value rate | clocked value rate | decisions (clocked) | latency mean / max (s) | frames charged | final frame |
|---|---|---|---|---|---|---|
| rule | 0.053 (1/11) | 0.000 (0/11) | 14 | 0.000 / 0.001 | 14 | 1533 |
| greedy | 0.105 (2/11) | 0.105 (2/11) | 17 | 0.383 / 0.841 | 203 | 1505 |
| mcts | 0.368 (3/11) | 0.053 (1/11) | 12 | 2.686 / 8.346 | 973 | 1803 |

Runs: `results/fire_rule_tcm30_ep1_20261016_152303`, `results/fire_greedy_tcm30_ep1_20261016_160350` (rerun; first attempt segfaulted), `results/fire_mcts_tcm30_ep1_20261016_161200` (rerun; first attempt lost its window).

Reading: charging thinking time flips the ranking. Unclocked, MCTS is the best shipped agent (0.368); under the clock it loses 973 of its 1500 frames to deliberation (12 decisions, 2.7 s each, one of 8.3 s), gets through only 12 decisions instead of 23, and falls to 0.053, below greedy (0.105). Greedy's 0.2 to 0.8 s decisions cost it 203 frames and no rescues; the rule agent's cost is 14 frames. This is the shape of RRARA Table 2 (under TCM: MCTS 0.14 below rule 0.19, 25-scenario averages) reproduced on one scene with the laptop's own latencies. The final frame of 1803 shows the clock overshooting `max_steps` because charged frames are stepped before HAZARD's own check (`challenge.py:398`); harmless for the score (the episode simply ends after the next action) but worth clamping in step 2b.

Caveats: one scene, one seed; latencies measured on the laptop with low free RAM (greedy's mean rose from 0.23 s unclocked to 0.38 s clocked for no algorithmic reason); machine of record still to be decided (A15).

## Oct 16 16:30: Tom's step 3, the LLM baseline, Claude Opus 5.5 unclocked (laptop, fire, mm_craftroom_2a-1, one episode)

`results/fire_llm_ep1_20261016_161935/` (`summary.json`, `llm_calls.json`, per-decision prompts and chain-of-thought under `mm_craftroom_2a-1/0/`), log `logs/baseline_fire_llm_opus_ep1_attempt3.log`. Model `claude-opus-5-5`, effort low, max_tokens 4096, org key with `anthropic-workspace-id` header (A18b).

- 19 decisions, 30 API calls (HAZARD's chain-of-thought makes two calls per LLM decision; some decisions are rule shortcuts in `llm.py:choose_target` with no call), 24,480 input and 6,988 output tokens, cost estimate $0.24.
- Decision latency: mean 6.42 s, max 11.56 s (API call mean 3.96 s, max 9.90 s). If charged at 30 FPS: 3670 frames, 2.4x the 1500-frame episode.
- Outcome: rescued 2/11, score 6 of 19, value rate 0.316 (it went for high-value items: the first reasoning trace weighs value, distance and temperature explicitly). 6 actions succeeded, 13 failed (the same unreachable-walk problem as the other agents).
- Compare RRARA Table 2 without clock: GPT-3.5 0.25 at 2.54 s; here Opus 5.5 0.32 at 6.4 s on one scene.

## Oct 16 16:39: Claude Opus 5.5 under RRARA's clock (laptop, fire, mm_craftroom_2a-1, one episode, `--clock tcm --fps 30`)

`results/fire_llm_tcm30_ep1_20261016_163048/`: 7 decisions (10 API calls, $0.07), decision latency mean 5.54 s, max 9.28 s, 1167 frames charged, final frame 1656, rescued 1/11, value rate 0.053. Unclocked (previous section): 19 decisions, 0.316.

Fire, one scene, unclocked -> clocked value rate: rule 0.053 -> 0.000; greedy 0.105 -> 0.105; mcts 0.368 -> 0.053; Claude Opus 5.5 0.316 -> 0.053. The two deliberators (mcts, Claude) both collapse to the rule agent's level once their seconds are paid in frames; greedy, at 0.4 s per decision, is unaffected. RRARA Table 1 reports the same for Claude 3.7 Sonnet (minus 0.41) and GPT-3.5 (minus 0.12) on 25 scenes. This is Tom's step-3 prediction ("by the time it's done thinking, the world will be destroyed") measured end to end on the laptop, with every decision's wall-clock in `decisions.jsonl` and every API call's in `llm_calls.json`.

Total Anthropic spend today: about $0.31 of the $4 credit (estimates from list prices).

## Oct 16 17:11: step 1, flood (laptop, scene mm_craftroom_2a-1 flood variant, one episode each, unclocked)

| agent | wall s | decisions | rescued | value rate | decision latency mean / max (s) | frames if clocked |
|---|---|---|---|---|---|---|
| rule | 434 | 15 | crashed at the end | n/a | 0.000 / 0.000 | 15 |
| greedy | 424 | 16 | 1/11 | 0.026 | 0.158 / 0.375 | 83 |
| mcts | 484 | 19 | 1/11 | 0.132 | 1.883 / 2.216 | 1083 |
| random | 422 | 47 | 5/11 | 0.132 | 0.000 / 0.000 | 47 |

rule crashed with `RecursionError: maximum recursion depth exceeded` inside `copy.deepcopy` (HAZARD code path, see `logs/baseline_flood_rule_ep1.log`); the 15 decisions before it are logged. A HAZARD bug to look at before the flood sweep (candidate A19). Same pattern as fire otherwise: mcts thinks 1.9 s per decision and would be charged 1083 of 1500 frames.
Flood rule crash location: `envs/flood/flood.py:123` (`communicate`) -> `floodagent_controller.py:400` -> `next_key_frame` (`:316`) during `agent_walk_to` (`policy/env_actions.py:128`) at decision 15; a `copy.deepcopy` inside the flood controller's per-frame bookkeeping recursed past Python's limit. Only the rule agent hit it; greedy, mcts and random completed full episodes in the same scene.

## Oct 16 17:49: step 1, wind (laptop, scene suburb_scene_2023-0-0, one episode each, unclocked, max_steps 3000)

| agent | wall s | decisions | final frame | rescued (any value) | value rate | decision latency mean / max (s) | frames if clocked |
|---|---|---|---|---|---|---|---|
| rule | 589 | 39 | 3056 | 0/15 | 0.000 | 0.000 / 0.000 | 39 |
| greedy | 428 | 26 | 3018 | 0/15 | 0.000 | 0.084 / 0.100 | 78 |
| mcts | 680 | 29 | 3090 | 0/15 | 0.000 | 6.648 / 10.498 | 5797 |
| random | 452 | 76 | 3002 | 3/15 | 0.000 | 0.000 / 0.000 | 76 |

Wind is the hazard where the HAZARD paper itself reports every agent near 0.3 value rate on average; on this one scene all four shipped agents score 0 (random's three deliveries were zero-value objects). MCTS's per-decision time doubles here (6.6 s, max 10.5 s) because the wind scene has 15 targets and 3 containers to plan over; charged at 30 FPS it would think for nearly two episodes. No crashes.

### Unclocked value rate by hazard (laptop, one scene each)

| agent | fire | flood | wind |
|---|---|---|---|
| rule | 0.053 | crashed (A19 candidate) | 0.000 |
| greedy | 0.105 | 0.026 | 0.000 |
| mcts | 0.368 | 0.132 | 0.000 |
| random | 0.474 | 0.132 | 0.000 |
| Claude Opus 5.5 | 0.316 | not run | not run |

Paper reference (HAZARD, 25-scene averages, no clock): LLM pipeline rescues 77.8% of value in fire, MCTS 75.9%; everything about 31% in wind. Our single-scene numbers are far below those averages, which says more about scene mm_craftroom_2a-1 (an unreachable target traps the deterministic agents) and single seeds than about the agents; the 25-scene sweep is the next step and needs the machine-of-record decision (A15).

## Oct 16 18:19: lane complete; clocked flood and wind (laptop, one episode each, `--clock tcm --fps 30`)

| hazard | agent | unclocked | clocked | decisions (clocked) | latency mean / max (s) | frames charged |
|---|---|---|---|---|---|---|
| flood | greedy | 0.026 | 0.026 | 15 | 0.219 / 0.452 | 106 |
| flood | mcts | 0.132 | 0.026 | 12 | 1.764 / 2.417 | 641 |
| wind | greedy | 0.000 | crashed at frame 2827 (RecursionError, same as flood rule) | 24 | 0.082 / 0.096 | 72 |
| wind | mcts | 0.000 | 0.000 | 8 (was 29) | 4.813 / 6.644 | 1160 |

Runs: `results/flood_greedy_tcm30_ep1_*`, `results/flood_mcts_tcm30_ep1_*`, `results/wind_greedy_tcm30_ep1_*`, `results/wind_mcts_tcm30_ep1_*`; lane log `logs/all_baselines.log`, per-run logs `logs/baseline_<hazard>_<agent>[_tcm30]_ep1.log`.

The recursion crash (flood rule at decision 15, wind greedy at frame 2827) is one HAZARD bug: a `copy.deepcopy` in the controller's per-frame bookkeeping (`envs/flood/flood.py:123` path) whose structure deepens over the episode until Python's recursion limit. It shows up late in long episodes regardless of agent; candidate fix A19 (raise the recursion limit or replace the deepcopy) to be checked before any 25-scene sweep.

### Final picture of the day (laptop, one scene per hazard, value rate; frames charged in parentheses)

| agent | fire | fire clocked | flood | flood clocked | wind | wind clocked |
|---|---|---|---|---|---|---|
| rule | 0.053 | 0.000 (14) | crash | not run | 0.000 | not run |
| greedy | 0.105 | 0.105 (203) | 0.026 | 0.026 (106) | 0.000 | crash (72) |
| mcts | 0.368 | 0.053 (973) | 0.132 | 0.026 (641) | 0.000 | 0.000 (1160) |
| random | 0.474 | not run | 0.132 | not run | 0.000 | not run |
| Claude Opus 5.5 | 0.316 | 0.053 (1167) | not run | not run | not run | not run |

Pattern across hazards: charging thinking time leaves the sub-second agents where they were and pulls every deliberator (mcts, Claude) down to the reflex level or below. That is the premise of the project confirmed on the benchmark, and the baseline the gate has to beat.

## Oct 16 18:3x: A19 applied (CR approved), root cause is in ThreeDWorld 1.13, not HAZARD

Traceback (both crashes, `logs/baseline_flood_rule_ep1.log`, `logs/baseline_wind_greedy_tcm30_ep1.log`): HAZARD controller `communicate` -> `tdw/controller.py:171` -> `tdw/add_ons/replicant_base.py:181` `self._previous_action = deepcopy(self.action)` -> `copy.deepcopy` recursion. Mechanism: TDW's built-in move/turn actions are created with `previous=self._previous_action` (`tdw/add_ons/replicant.py:100,142,197,210,245`) and store it (`tdw/replicant/actions/move_to.py:105`); each finished action is deep-copied including that link, so the stored chain grows by one action per high-level action and the copy overflows Python's recursion limit after roughly 40 actions in one episode. HAZARD's own actions pass `previous=None` everywhere (`envs/*/agent.py`), so the authors' 2024 TDW presumably did not deep-copy (A2 risk realized).

Fix (`scripts/patch_A19_previous_action_chain.py`, diff `scripts/A19_previous_action_chain.diff`): after `super().communicate(commands)` in `envs/fire/fireagent_controller.py`, `envs/flood/floodagent_controller.py`, `envs/wind/windagent_controller.py`, set `agent._previous_action._previous_action = None` for each agent, so the chain never exceeds one link. TDW only reads the immediate predecessor (walk-animation continuity), so behaviour is unchanged. Verification: `scripts/laptop_fire_sweep.sh` first reruns the failing case (flood, rule, one episode) before the sweep.

25-scene fire sweep started right after: `results/sweep_fire_<agent>_ep25/` for rule, greedy, mcts, random, unclocked, resumable (`--resume_dir`), lane log `logs/fire_sweep.log`. Expected duration about 3.5 h per agent on the laptop.
A19 verified 20:55: flood, rule agent, one episode (the crash case): exit 0, 25 decisions, final frame 1602, rescued 2/11, value rate 0.158, no recursion error (`results/flood_rule_ep1_<20:47 stamp>/`). The 25-scene fire sweep then started at 20:55 (`logs/fire_sweep.log`). The lane sat idle from about 18:35 to 20:47 (laptop asleep, see below).

## Oct 26: resumed after 10 days; Tom's Oct 16 action items; A20 VLM baseline built

State found: the Oct 16 fire sweep died on scene 1 (laptop slept that night, rebooted Oct 23); all 25-scene results must be produced now. Sweep restarted 10:57 (`scripts/laptop_fire_sweep.sh`, resumable, `logs/fire_sweep.log`).

Tom's asks (Granola summary of the Oct 16 meeting): simple baselines on all scenes (random, greedy, LLM) with success and latency (mean and max); a VLM baseline that uses the images; send results plus a couple of demo videos; then meet to choose the direction.

A20 (applied, diff `scripts/A18_A20_llm_anthropic_openai_vlm_v3.diff`): `policy/llm.py` gets `self.next_images`; when set, the Anthropic branch sends the first user turn as [image..., text] and clears the list after the call. `scripts/vlm_agent.py` subclasses HAZARD's `LLM`: per decision it takes `state['raw']['rgb']` (3x1024x1024 floats), resizes to 512 px, PNG-encodes, attaches, then runs HAZARD's unchanged prompt/chain-of-thought/parsing. Runner: `--agent vlm --lm_source anthropic --lm_id <model> --api_key_file <file>`. Dry test without API: constructs, encodes, message shape ['image', 'text'].

## Oct 26 11:0x: lanes queued for Tom's deliverable (CR approved)

1. `scripts/laptop_fire_sweep.sh` (running since 10:57): rule, greedy, mcts, random on the first 25 fire training scenes, unclocked, resumable into `results/sweep_fire_<agent>_ep25/`.
2. `scripts/laptop_llm_lane.sh` (waits for "FIRE SWEEP DONE"): Claude Haiku 5.5 as `llm` on 25 scenes, as `vlm` on 25 scenes, then Claude Opus 5.5 as `llm` on 5 scenes (`results/sweep_fire_{llm_haiku,vlm_haiku,llm_opus}_ep*`); then two recorded episodes on mm_craftroom_2a-1 (mcts, Haiku vlm) with `--record`, and `scripts/make_videos.py` -> `results/videos/*.mp4` via ffmpeg at 30 fps.
3. `scripts/aggregate_sweeps.py` builds the table (value rate mean and sd, targets rescued, decisions per episode, decision latency mean and max, frames if clocked, API latency, cost).

Budget: Haiku about 1 cent per episode, Opus about 24 cents; planned spend about $2 of the remaining $3.7. Machine: laptop only (A15), must stay awake overnight.

## Oct 26 15:10: the sweep stalled at scene 3; cause and the v2 lane

The multi-scene run (one HAZARD `submit` iterating scenes in one simulator process) completed mm_craftroom_2a-10 at 11:08:55, started resetting mm_craftroom_2a-11, and the build process died silently at 11:08 (no crash dump; `Player-prev.log` ends in ordinary asset warnings; no sleep event in the power log; S3 reachable). Python then waited on the dead socket for four hours. Loading a second floorplan into one Unity process on this memory-starved laptop is the suspect. Sweep v2 (`scripts/laptop_sweep_v2.sh`, log `logs/sweep_v2.log`): one process per scene, `timeout 30m` per scene, resumable by scene (`--resume_dir` + `--data_dir` -> `results/sweep_<tag>/<scene>/`), stay-awake helper (`scripts/stay_awake.py`, SetThreadExecutionState, no settings changed) for the whole lane. Windows sleep timeout on this laptop is 3 minutes (AC and DC), `powercfg` query.

## Oct 26 15:3x: Adroit plan, from the CBM project's Slurm patterns

CBM ran everything as Slurm batch jobs on Adroit GPU nodes: `--time=12:00:00` typical, one `--time=2-00:00:00` training job that survived by `#SBATCH --requeue` + `--signal=B:USR1@180` + a `trap` calling `scontrol requeue`, resuming from checkpoints; sweeps as `--array=0-4`; MIG slices `gpu:3g.20gb:1`. For HAZARD the per-scene runner is already resumable, so the port is a job array over scenes: `scripts/adroit_scene_array.slurm` (one scene per task, own Xvfb display `:200+i`, own TDW port `20000+i`, `vglrun -d egl`, 45-minute tasks, `%4` concurrent to respect the gpu-short QOS). UNTESTED: compute nodes have no internet, so HAZARD's asset cache (A7, `use_cached_assets` + `download_assets.sh` on the vis node, which has internet) must be filled first; the S3 reachability test on adroit-vis is still outstanding.

## Oct 26 16:xx: Oct 16 Adroit hang diagnosed (adroit-vis session, relayed; A15: adroit numbers, not laptop)

Cause, verified on adroit-vis by the remote session: `scripts/adroit_hazard_first_run.sh:31` launched the build as `TDW.x86_64 -port $PORT`, two argv tokens. The Linux build ignores the split form and listens on its default port 1071 (observed: a listener on 1071 received the build's connection 5.5 s after a launch with `-port 20103`). Python waited forever in `tdw/controller.py:76` (`self.socket.recv()` in `Controller.__init__`), matching the faulthandler stack in `logs/diag_fire_rule.log` on the node. Both `-port=PORT` and the single token `"-port PORT"` (what tdw's own `launch_build` passes, `tdw/controller.py:575`) complete the handshake in 5.5 s; `send_version` reports TDW 1.13.0, Unity 2020.3.24f1. `scripts/run_baseline.py:153` already passes the single-token form, so the laptop runner was never affected; the shell script is fixed to `-port=$PORT` and the bundle rebuilt. Side finding: the Oct 16 build process was still alive on adroit-vis at 100% CPU ten days later with no sockets and no X server; the remote session killed it (SIGTERM ignored, SIGKILL). The bundle was not on the node (the Oct 16 copy did not land); copied today by scp.

## Oct 26 17:2x: Adroit re-test, 4 OpenMP threads (adroit-vis session, relayed; A15: adroit numbers, never mixed with the laptop's)

Fire, rule, `test_set/mm_kitchen_3a-1`, one episode via `scripts/run_baseline.py` with `OMP_NUM_THREADS=4`: exit 0, 19 decisions, final frame 1577, rescued 2/5 targets (frames 412 and 774), avg_score 1.5 of 9.0 = value rate 0.167, decision wall-clock mean and max 0.0001 s, wall 303.6 s of which reset 58.6 s; simulator 0.16 s per frame (1577 frames in about 252 s). Same episode at the default 64 threads (HAZARD's `submit()`, 15-minute cap): exit 124, 5 decisions, 404 frames in 718 s = 1.78 s per frame, Python at 556% CPU inside HAZARD's CPU semantic-map update (`utils/model.py:311`, py-spy). The thread cap is an 11x speedup; at 4 threads one Adroit lane is faster per frame than the laptop (0.25 to 0.29 s per frame, `logs/sweep_v2.log`). For the rule agent decision latency is unaffected (no torch in the agent); for greedy, mcts and LLM agents per-decision time on Adroit reflects that node's CPU and API path, the A15 question. Results `results/task2_fire_rule_omp4/mm_kitchen_3a-1/` on the node. Second hang route found: a build launched on a display that is not up ("Desktop is 0 x 0") never connects and never exits; the Adroit scripts now check the X socket before launching. Full assumption list A22 (1-13) in `ASSUMPTIONS.md`, pending CR. Sweeps held pending CR's go.

Laptop comparison for the whole job (280 episodes: 70 scenes x 4 agents): laptop serial at 6 to 8 min per rule episode is about 30 to 35 h, more for mcts; Adroit at 4 lanes x 0.16 s per frame is about 5 min per episode per lane, so roughly 6 h plus mcts overhead.

## Oct 26 17:3x: A22 approved, Adroit sweeps released (CR)

CR approved A22 (1-13) as written and released the Adroit sweeps: fire, flood, wind for rule, greedy, mcts, random, 4 lanes, 4 threads per process, 60-minute cap per scene, resumable, results under `results/sweep_<env>_<agent>_test25/` (wind: `_test20`) on the node. Laptop keeps running the fire rule test set in parallel; tables stay separate (A15).

## Oct 26 17:4x: laptop fire rule test set, scene mm_kitchen_3a-22 failed on memory (laptop)

`results/sweep_fire_rule_test25/mm_kitchen_3a-22/summary.json`: exit 1 after 6 decisions at frame 550, `numpy ArrayMemoryError: Unable to allocate 4.00 MiB` for the 1024x1024 uint32 segmentation image in `utils/seg_id.py:66` (called from `fireagent_controller.py:503` `_obs`). Not a HAZARD bug: the laptop (16 GB RAM, about 2.7 GB free physical and 5.5 GB free commit at 17:40, with a game client and several editor processes resident) hit its commit limit. The scene has no `eval_result.json`, so rerunning `scripts/laptop_sweep_v2.sh` for the rule agent after the lane finishes redoes exactly this scene (`run_baseline.py:137-141` skips evaluated scenes). The Adroit sweep covers the same scene independently (A15). Keep heavy applications closed while the laptop lane runs.

## Oct 26 17:15: likely cause of the Adroit hang, and a GPU utilisation note

`scripts/adroit_hazard_first_run.sh` launched the Linux build with `-port $PORT` as two argv words; every run that worked (laptop shell script `"-port $PORT"`, `tdw_hello.py`/`run_baseline.py` Popen `f"-port {port}"`, TDW's own `Controller.launch_build`, `tdw/controller.py:575`) passes one argument containing the space. S3 was confirmed fast from adroit-vis (13 MB in 0.4 s). Hypothesis: the build fell back to port 1071 and HAZARD waited on 10826. To be confirmed by the first run from the Adroit session using the runner. The script was also edited on disk at 16:47 (not by Claude) to `-port=$PORT`; reset to the quoted form.

GPU utilisation on the laptop during the sweep is about 25% (3.5 GB VRAM used), CPU about 56% overall: the loop is lock-step (build renders, sends 1024x1024 images over TCP, Python builds maps, decides, sends commands), so the GPU idles while Python works. The per-frame cost is CPU-bound single-thread Python plus image transfer, which is why the A100 will not speed an episode up much; parallel scenes are the gain (A15 caveat applies to any mixed-machine table).

## Oct 26 17:20: correction (this session) and division of labour between the two laptop sessions

Correction: my 17:15 hypothesis that the Oct 16 Adroit hang came from the port-argument form is retracted. The Adroit session measured the real cause (A22 item 8, NOTES 17:2x): HAZARD's per-frame semantic-map update (`utils/model.py:311`) run by torch over all 64 cores takes 1.78 s per frame versus 0.16 s with `OMP_NUM_THREADS=4`; the "hang" was an episode running eleven times slower than expected. A second route to the same symptom is a build launched on a display that is not up (A22 item 11). The port-form note stays in the brief only as history.

Division of labour from now on: the session "SSH connection to adroit-vis.princeton.edu" (cwd this folder) coordinates the Adroit node and relays its reports; this session owns the laptop lane (`logs/sweep_v2.log`), the Claude runs and the table for Tom. Both append to NOTES.md and ASSUMPTIONS.md; neither rewrites them whole (this session's earlier whole-file edits to ASSUMPTIONS.md are retired in favour of appends).

## Oct 26 18:40: laptop fire test set, rule agent, complete (24/25 scenes)

Lane `scripts/laptop_sweep_v2.sh`, one process per scene, no clock. Results `results/sweep_fire_rule_test25/<scene>/`.

| | value |
|---|---|
| scenes evaluated | 24 of 25 (3a-22 failed at frame 550: numpy 4 MiB allocation error, memory; redone in pass 2) |
| value rate mean (sd) | 0.071 (0.11) |
| targets rescued with any value | 19 of 241 |
| decisions per episode | 17.9 |
| decision latency | 0.1 ms (rule lookup; frames if clocked: 1 per decision, 18 per episode) |
| final frame count | 1502 to 1592 (max_steps 1500 plus the last action's frames) |
| wall per scene | mean 466 s, min 208 s (the failure), max 731 s; 3.24 h for the sweep |

Reading: 18 of 24 scenes score 0. The rule agent rescues at most 3 objects on the richest scenes (3a-5, 3a-8, 3a-9). HAZARD's paper reports rule-agent fire value rates near 0.2 on its own test split (Table 3), and RRARA Table 2 lists 0.19; our split is the shipped `test_set/mm_kitchen_3a-*` folders (A21), so the gap is a scene-split question, not a bug, until the Adroit run of the same scenes (same split, A15) confirms the 0.07.

Fix made: `scripts/aggregate_sweeps.py` read `decisions.jsonl` and `summary.json` only at the sweep-folder level (the Oct 16 layout); it now also merges the per-scene files the v2 lane writes, so the latency and decisions/ep columns are filled (that is why the 18:39 watcher table showed n/a). Per-scene LLM call stats are summed across scenes.

Projection for the remaining lane at this pace: greedy about 3.5 h (started 18:37), mcts about 4 h, random about 3 h, then Haiku llm 25 and vlm 25 (each about 4 h plus API time), Opus 5, videos: roughly 24 h total if nothing fails. Pass 2 is chained to redo any scene without eval_result.json; the stay-awake helpers hold the laptop through both passes.

## Oct 26 20:50: second memory failure, greedy 3a-2

`sweep_fire_greedy_test25/mm_kitchen_3a-2` died at frame 360 in `policy/astar.py:57` (numpy could not allocate 1.99 MiB for a 511x511 float64), the same cause as rule 3a-22: free RAM was about 1 GB while a game ran beside TDW and Python. CR closed the game at 20:5x (free RAM now 4.3 GB). Greedy is 11 of 25 done with one failure; pass 2 redoes both failed scenes. Everything else in the lane is unchanged.

## Oct 26 21:39: Adroit fire test set complete (adroit-vis session, relayed; A15: adroit numbers only, `machine` field in every summary.json is adroit-vis.princeton.edu)

100/100 episodes (25 test scenes x rule, greedy, mcts, random), zero non-zero exits, 17:14:52 to 21:39:09 (4 h 24 m on 4 lanes). Table from `scripts/aggregate_sweeps.py` on the node (`results/aggregate_adroit.md` there):

| run | scenes | value rate mean (sd) | targets rescued (any value) | decisions/ep | decision latency mean / max (s) | frames if clocked, mean/ep (of 1500) | status |
|---|---|---|---|---|---|---|---|
| sweep_fire_rule_test25 | 25 | 0.092 (0.10) | 0.109 | 19.8 | 0.000 / 0.000 | 20 | done (25 ep) |
| sweep_fire_greedy_test25 | 25 | 0.176 (0.16) | 0.206 | 21.6 | 0.180 / 0.373 | 128 | done (25 ep) |
| sweep_fire_mcts_test25 | 25 | 0.128 (0.18) | 0.154 | 19.4 | 2.624 / 7.215 | 1533 | done (25 ep) |
| sweep_fire_random_test25 | 25 | 0.371 (0.18) | 0.397 | 55.1 | 0.000 / 0.044 | 55 | done (25 ep) |

Wall per episode (worker start to runner exit, including build handshake and scene reset): mean 625 s, min 484, max 836; rule 622, greedy 608, mcts 663, random 607. Simulator rate with 4 lanes: 0.39 to 0.42 s per frame per lane (final frame about 1550) versus 0.16 single-lane; the GPUs were 60 to 100% busy with another user's jobs throughout. Decision wall-clock is unaffected by lane count (rule 0.0001 s; mcts max 7.2 s), so the latency columns stand; s/frame is not comparable with the laptop. Reading: unclocked, mcts (2.6 s per decision) would be charged 1533 frames per episode, more than the 1500-frame episode, the RRARA Table 1 effect again at test-set scale. Random scoring highest (0.371) repeats the laptop's single-scene finding (random 0.474, Oct 16) and is now 25 scenes; RRARA Table 2 reports rule 0.19 on 25 scenarios, our rule is 0.092. The accidental 17:11 instance produced nine non-zero exits (five port-in-use, four killed); all nine scenes were redone. Flood started 21:39:09, 100 episodes, ETA about 02:10; wind after.

Laptop `scripts/aggregate_sweeps.py` already reads the per-scene layout (`load()`, lines 23-39, Oct 26 15:06), so the node's fallback edit (A22 item 15) needs no mirroring.

## Oct 27 01:37: Adroit flood test set complete, 99/100 (adroit-vis session, relayed; A15: adroit numbers only)

Flood block 21:39:09 to 01:37:03 (3 h 58 m, 4 lanes), same 25 `room_setup_fire` test scenes with `--env flood`. Table from `scripts/aggregate_sweeps.py` on the node:

| run | scenes | value rate mean (sd) | targets rescued (any value) | decisions/ep | decision latency mean / max (s) | frames if clocked, mean/ep (of 1500) | status |
|---|---|---|---|---|---|---|---|
| sweep_flood_rule_test25 | 25 | 0.094 (0.09) | 0.170 | 21.5 | 0.000 / 0.000 | 21 | done (25 ep) |
| sweep_flood_greedy_test25 | 25 | 0.124 (0.11) | 0.182 | 41.2 | 0.143 / 0.390 | 199 | done (25 ep) |
| sweep_flood_mcts_test25 | 25 | 0.110 (0.14) | 0.154 | 19.5 | 2.138 / 6.084 | 1262 | done (25 ep) |
| sweep_flood_random_test25 | 24 | 0.228 (0.14) | 0.367 | 53.0 | 0.000 / 0.001 | 53 | 24 ep done, 1 retrying |

Wall per episode: mean 561 s (min 390, max 936); rule 553, greedy 568, mcts 582, random 539. Simulator 0.35 to 0.37 s per frame per lane, GPUs still 95 to 100% busy with another user's jobs.

One HAZARD crash, flood, random, mm_kitchen_3a-5: exit 1 at frame 1324 after 59 decisions. Traceback: the random agent returned `('walk_to_single', 0)`; `challenge.py:372` maps the renumbered id through `id_reverse_renumbering`, which returns -1 when no object carries that id (`challenge.py:128-132`); `env_actions.py:161` then indexes `manager.objects[-1]`, `KeyError: -1`. Same unguarded-id family as A16 (`pick_up` with no nearest object), on the walk path this time. The remote's unverified guess: the target (object 6564923) had fallen out of the scene (y = -3873). Failed attempt preserved at `results/failed_runs/sweep_flood_random_test25_mm_kitchen_3a-5_attempt1/` on the node; the scene is being retried once through the resumable driver (a fresh random draw, not a reproduction). No HAZARD patch without CR's go. Wind started 01:37:03, 80 episodes (20 `suburb_scene_2023-3-*` scenes x 4 agents), ETA about 05:00.

## Oct 27 05:32: Adroit sweeps complete, 299/300 (adroit-vis session, relayed; A15: adroit numbers only)

Total 17:14:52 Oct 26 to 05:32:03 Oct 27, 12 h 17 m for 300 episodes on 4 lanes. Final 12-row table on the node at `results/aggregate_adroit.md`; per-scene files under `results/sweep_<env>_<agent>_test<N>/<scene>/`. Node left idle.

Flood retry: random, mm_kitchen_3a-5, 05:26 to 05:32 (363 s), exit 0, 59 decisions, score 2.0/7. Flood random is 25/25: value rate 0.230 (0.14), targets rescued 0.372. No HAZARD change (A23 not triggered); the crashed first attempt stays in `results/failed_runs/`.

Wind block 01:37:03 to 05:31:03 (3 h 54 m), 20 `suburb_scene_2023-3-*` test scenes; step limit 3000 (`challenge.py:59`), so the aggregate header "of 1500" is wrong for these rows:

| run | scenes | value rate mean (sd) | targets rescued (any value) | decisions/ep | decision latency mean / max (s) | frames if clocked, mean/ep (of 3000) | status |
|---|---|---|---|---|---|---|---|
| sweep_wind_rule_test20 | 20 | 0.000 (0.00) | 0.000 | 120.7 | 0.000 / 0.000 | 121 | done (20 ep) |
| sweep_wind_greedy_test20 | 20 | 0.000 (0.00) | 0.000 | 388.6 | 0.104 / 0.432 | 1408 | done (20 ep) |
| sweep_wind_mcts_test20 | 19 | 0.000 (0.00) | 0.000 | 38.8 | 11.180 / 56.821 | 13030 | done (19 ep) |
| sweep_wind_random_test20 | 20 | 0.000 (0.00) | 0.017 | 79.7 | 0.000 / 0.000 | 80 | done (20 ep) |

Every wind episode scored 0. Each wind test scene has exactly one target (`avg_max_score` 1.0 in all 79 `eval_result.json`); no agent delivered it; random's 0.017 is one pickup worth 0 at the end. Episodes ran to the 3000-frame limit (mean final frame 3020 to 3054). Wall per episode mean 689 s (min 351, max 3600 = the timeout); rule 541, greedy 759, mcts 1016, random 440. Simulator 0.14 to 0.29 s per frame per lane. MCTS on wind is 11 to 13 s per decision, 4 to 5x fire and flood (2.1 to 2.6 s); rule and random stay at 0.0001 s everywhere, so this is the wind scene's planning size, not the node.

Non-zero exit: wind, mcts, suburb_scene_2023-3-8, exit 124 (60-minute cap) at frame 2932 of 3000 after 97 decisions; that episode's decisions averaged 31.4 s, max 97.5 s. Partial folder moved to `results/failed_runs/sweep_wind_mcts_test20_suburb_scene_2023-3-8_attempt1_timeout/` so its 97 decisions do not enter the latency row (with them: 41.7 decisions/ep, 13.5 s mean, 97.5 s max). Pending CR: rerun once under a 120-minute cap (A22 item 19) or leave wind mcts at 19/20.

Against the paper (vault note `Papers/zhouHAZARDChallengeEmbodied2024.md`, 25-scene averages, no clock, perception on): fire MCTS 75.9%, rule 53.1%, GPT-4 pipeline 77.8%; wind "about 31%" for everyone. Ours on the held-out test sets with ground-truth segmentation (A6): fire mcts 0.128, rule 0.092, random 0.371; flood similar; wind 0.000 for all four. The gap is large on every hazard and is now measured over 25 and 20 scenes on a second machine, so it is not a single-scene or single-seed artefact (NOTES Oct 16 18:19 guessed that). Candidates, none verified: the paper's numbers may be on the training scenes or an earlier scene set; TDW 1.13 physics versus the authors' 2024 build (A2); the shipped agents' code may differ from what produced the paper's table; `max_steps` or scoring differences. This is the first question for Tom and for the vault session.

Laptop cross-check (laptop numbers, `scripts/aggregate_sweeps.py` at 05:30, lane still running mcts): fire test set rule 24 scenes 0.071 (0.11), greedy 24 scenes 0.118 (0.11), mcts 18 scenes 0.140 (0.19). Same range and ranking family as Adroit (rule 0.092, greedy 0.176, mcts 0.128), so the two machines agree on success. Latency differs as A15 predicted: laptop mcts 5.07 s mean, 17.0 s max, versus Adroit 2.62 / 7.2; laptop greedy 0.338 s versus 0.180. Clocked baselines must therefore name the machine.

## Oct 27: wind mcts suburb_scene_2023-3-8 rerun approved (CR), 120-minute cap, one lane on adroit-vis

CR approved A22 item 19. One attempt only; if it times out again wind mcts stays 19/20. The Adroit session was also asked to correct the aggregate header ("of 1500") for the wind rows, whose limit is 3000 (`challenge.py:59`).

## Oct 27 12:12: Adroit complete, 300/300; HAZARD does not score wind at this commit (adroit-vis session, verified on the laptop copy)

Wind mcts suburb_scene_2023-3-8 rerun (A22 item 19, 120-minute cap, one lane, 12:00:56 to 12:12:12): exit 0, 30 decisions, final frame 3066, score 0.0 / 1.0, decision latency mean 11.04 s, max 25.36 s, min 2.59 s, wall 665 s. The timed-out first attempt had 97 decisions and a 97 s decision, so MCTS varies 3x in decision count between draws on this scene. Wind mcts row, 20 scenes: value rate 0.000, 38.4 decisions/ep, latency 11.175 / 56.821 s, 12876 frames if clocked (of 3000). Final Adroit totals: fire 100/100, flood 100/100, wind 80/80; `results/aggregate_adroit.md` on the node, regenerated with the step limit per row. The laptop `scripts/aggregate_sweeps.py` now mirrors that (step_limit 3000 for wind runs, header "of step limit", cell "N (of L)").

Correction to the wind reading, verified here: `challenge.py:186-229` `get_score()` computes value only `if self.env_name in ["fire", "flood"]` (line 191); the else branch (lines 226-228) sets `total_score = 0`, `max_score = 1` unconditionally. So every wind value rate from commit d35c71e is 0 by construction, whatever the agent does. Each wind test scene has 15 targets (not one, as the wind report first said; `avg_max_score` 1.0 is the else branch). `eval_result.json` `target_status` still records pickups (random: 5 targets with a step number across 20 scenes; rule, greedy, mcts: 0), unscored. This also corrects the Oct 16 18:19 reading ("on this one scene all four shipped agents score 0"): those laptop wind zeros were the evaluator, not the agents. Wind results at this commit are usable for decision latency and the pickup record only. A wind scoring rule would be ours, proposed as A24.

## Oct 27 12:25: laptop lane reordered (CR: "yes, any way to make it shorter?"); Adroit offered the vlm/Opus sweeps

Laptop shipped-agent sweeps are complete (rule 24, greedy 24, mcts 25, random 25 of 25 scenes; two memory failures to redo). Haiku 5.5 text started 12:14 (scene 1: 3.8 s per decision at decision 9, no API errors). `scripts/laptop_lane_v3.sh` (started 12:22, background) waits for that sweep's done marker, then stops the v2 driver and its pass-2 watcher (they would otherwise start the vlm and Opus sweeps), redoes the two failed shipped-agent scenes, records the two demo episodes on `mm_kitchen_3a-1` (mcts, Haiku vlm) and encodes them, and only then runs the Haiku vlm 25 and Opus 5 sweeps, unless `logs/SKIP_LAPTOP_VLM` exists. That file is the switch for moving those two sweeps to adroit-vis, idle since 12:12: `ADROIT_LLM_BRIEF.md` has the hand-off (CR copies the key file to the node's `.secrets/` by scp; the node session never reads it; cost caps 1.5 USD Opus, 2.5 USD node total). Not sent until CR approves: the brief for the node said LLM runs stay on the laptop unless CR gives the key path.

Why the Adroit session sat idle 05:32 to 12:00 (CR's question): it followed the "ask CR before acting" rule literally for a one-scene rerun (wind mcts timeout, A22 item 19) and waited. The brief above adds a standing rule for the node: retry once or skip, log the question, keep going; only HAZARD source edits and spend beyond the caps wait for CR. The same rule applies to this session's lanes (they already retry via pass 2 and never wait).

## Oct 27: A24 approved; wind scored after the fact with HAZARD's own `calc_value.py` (laptop numbers here; Adroit's requested)

Finding: HAZARD ships a post-hoc scorer, `src/HAZARD/utils/calc_value.py`, separate from the in-run `get_score()`. Its `get_values()` (lines 22-62) maps target id to model name through `SceneSetup.target_id2name` (`utils/scene_setup.py:55-70`, from the scene's `log.txt` `add_object` commands), values a target 5 if `value.json[name] == 1` else 1, applies the fire/flood state discount, and for wind a discount of 1 (line 56); a target counts as picked when `target_status` is not False (line 97). That is the rule CR approved for A24, so it is used as is and labelled as the authors' scorer. Invocation, from the HAZARD repo root (relative data paths), `PYTHONPATH` set to both HAZARD paths (A10): `python src/HAZARD/utils/calc_value.py <dir with <scene>/eval_result.json> <fire|flood|wind>`. Cosmetic bug: `ZeroDivisionError` at line 119 after the averages print when nothing was picked in any scene. Note `value.json` keys are short names (`backpack`, `small_purse`) while wind objects carry model names (`b04_1106_backpack`), so most wind targets fall to the low value 1 under this rule; that is the authors' rule as shipped.

Cross-check, laptop: fire rule test set, 24 scenes, calc_value.py value rate 0.0710, in-run evaluator (`scripts/aggregate_sweeps.py`) 0.071; agree. Average rate 0.088, average step 503, damage rate 0.368.

Laptop wind, Oct 16 single episodes (`suburb_scene_2023-0-0`, training set, unclocked), calc_value.py: random value rate 0.130 (rate 0.2, 3 of 15 picked, average step 562, damage 0); rule, greedy, mcts 0.0 (nothing picked). The clocked greedy wind run has no `eval_result.json` (Oct 16 crash, A19) and scores nothing. This replaces the Oct 16 18:19 reading that "all four shipped agents score 0" on wind: random did pick up 3 targets there. Adroit's 80 wind episodes are being scored the same way on the node; the paper's "about 31%" on wind remains far above anything seen so far.

## Oct 27: Adroit scored with calc_value.py; in-run flood evaluator is biased low (adroit-vis session found it, mechanism verified on the laptop copy)

Adroit, all 12 sweep folders, no reruns, `results/calc_value_adroit.{txt,json}` on the node (A15: adroit numbers):

| run | valid | Average rate | Average value rate (calc_value.py) | Average step | Damage rate | in-run value rate |
|---|---|---|---|---|---|---|
| fire_rule | 25 | 0.135 | 0.092 | 666.0 | 0.593 | 0.092 |
| fire_greedy | 25 | 0.241 | 0.176 | 373.8 | 0.353 | 0.176 |
| fire_mcts | 25 | 0.178 | 0.128 | 462.2 | 0.421 | 0.128 |
| fire_random | 25 | 0.455 | 0.371 | 316.8 | 0.429 | 0.371 |
| flood_rule | 25 | 0.203 | 0.095 | 509.6 | 0.905 | 0.094 |
| flood_greedy | 25 | 0.215 | 0.138 | 359.8 | 0.778 | 0.124 |
| flood_mcts | 25 | 0.198 | 0.118 | 399.2 | 0.816 | 0.110 |
| flood_random | 25 | 0.433 | 0.256 | 355.2 | 0.761 | 0.230 |
| wind_rule | 20 | 0.000 | 0.000 | n/a | n/a | 0 (hardcoded) |
| wind_greedy | 20 | 0.000 | 0.000 | n/a | n/a | 0 (hardcoded) |
| wind_mcts | 20 | 0.000 | 0.000 | n/a | n/a | 0 (hardcoded) |
| wind_random | 20 | 0.017 | 0.019 | 1178.4 | 0.000 | 0 (hardcoded) |

Fire: calc_value.py equals the in-run evaluator in all four rows. Wind: only random scores (5 pickups over 20 scenes x 15 targets). Flood: calc_value.py reads 0.001 to 0.026 higher than in-run. Cause, traced by the Adroit session on `mm_kitchen_3a-10` (in-run 3.0/11, calc 4.0/11; two picked targets ended NORMAL and got 0.5 instead of 1) and verified in the laptop's HAZARD copy: `utils/scene_setup.py:85` imports `ObjectStatus` by the absolute path `envs.flood.object` and builds `setup.objects` from it; `envs/flood/floodagent_controller.py:115` adds those objects to the manager, so their `.state` is `envs.flood.utils.ObjectState` (via `envs/flood/object.py:3`). `challenge.py:9` compares against `src.HAZARD.envs.flood.utils.ObjectState`, a different class object for the same file, so `== FloodObjectState.NORMAL` (and `FLOATING`) at `challenge.py:216-218` is always False; only the waterproof branch gives full value, and `.state.value` at `:219` still records 0 correctly in `final_states`. Fire is unaffected because `fireagent_controller.py:171-173` builds its objects from its own relative `ObjectStatus` (`:18`), the same class `challenge.py:10` compares against. Nothing in HAZARD changed. Consequence: for flood, the calc_value.py column is the authors' intended rule and the in-run column is low by 0.5 x value per picked target that stayed NORMAL and is not waterproof; the laptop's Oct 16 flood single episodes (in-run) carry the same bias. The `aggregate_adroit.md` on the node now has an 11th column, "value rate, calc_value.py (authors' post-hoc scorer)"; the laptop's `aggregate_sweeps.py` does not yet (no laptop flood sweep exists; add when one does). Shell note from the node: set ROOT and PYTHONPATH in separate `export` lines, or PYTHONPATH ends up empty.

## Oct 27 16:40: laptop Haiku 5.5 text sweep complete, 25/25; v3 lane took over cleanly

`sweep_fire_llm_haiku_test25`, `--agent llm --lm_source anthropic --lm_id claude-haiku-5-5`, HAZARD's prompt_v2 with chain of thought, max 512 new tokens, no clock, 12:14 to 16:36 (4 h 22 m, one process per scene). Zero non-zero exits, every API stop reason `end_turn`.

| | value |
|---|---|
| value rate mean (sd) | 0.170 (0.22) |
| targets rescued with any value | 0.154 |
| decisions per episode | 45.2 |
| decision latency mean / max | 1.67 s / 14.4 s (decisions that skip the API are near 0) |
| API call mean / max | 2.42 s / 9.83 s |
| frames if clocked, mean per episode | 2272 of 1500 |
| tokens | 695k input, 227k output over 25 episodes |
| cost | 0.18 USD for the sweep (about 0.007 USD per episode); the 3.7 USD credit was never the constraint |
| wall per scene | mean 598 s, min 487, max 891 |

Action mix over 1129 decisions: pick_up 726, walk_to 338, drop 38, explore 27. Haiku asks for pick_up two thirds of the time, most of them before it has walked to the object; a pick_up costs 18 frames whether or not it succeeds (NOTES Oct 16, frames per action), which is why it fits 45 decisions in 1500 frames where the planners fit 18 to 20. Its success (0.170) sits with mcts (0.151) and above greedy and rule, below random (0.356). Under the clock it would be charged 2272 frames per episode, more than the episode: Tom's step-3 prediction at test-set scale, with the cheapest model.

v3 takeover at 16:36:17: the v2 driver had already launched the first vlm scene (16:36:09, 8 s earlier); v3 killed that process before it made any API call (the scene folder was empty; removed) and killed the v2 driver, the pass-2 watcher, and, by matching its marker argument, the Oct 26 stay-awake helper (the v3 helper watching `LANE V3 DONE` holds the laptop). Now running: rule 3a-22 redo, then greedy 3a-2, then the two recorded demos and encoding (about 18:00), then Haiku vlm 25 and Opus 5 on the laptop unless Adroit takes them (CR's decision pending; `ADROIT_LLM_BRIEF.md`). Laptop table so far: `results/aggregate_laptop.md`.

## Oct 27 16:5x: CR approved moving the Haiku vlm 25 and Opus 5 sweeps to adroit-vis

CR's words: "Approved: I will scp the key to the node now. Send the Adroit session the brief and set SKIP_LAPTOP_VLM." `logs/SKIP_LAPTOP_VLM` created, so the laptop v3 lane stops after the two redos and the demo videos. The node session receives `ADROIT_LLM_BRIEF.md` verbatim (key copied by CR, never read by any session; cost caps 1.5 USD Opus lane, 2.5 USD node total, both far above the measured 0.18 USD for 25 Haiku text episodes). The node's numbers will be labelled adroit-vis (A15); the laptop keeps the Haiku text row.

## Oct 27 18:30: the split, after the laptop had already started the vlm sweep

CR's approval arrived at 18:20; the v3 lane had checked `logs/SKIP_LAPTOP_VLM` once, at 17:13, and started the vlm sweep (videos done 17:13: `results/videos/fire_mcts_ep1_20261027_165402.mp4`, `fire_vlm_ep1_20261027_170240.mp4`). Laptop vlm at 18:17: 6 of 25 scenes (3a-1, 3a-10 to 3a-14; value rate 0.027 so far, 5.2 s per decision, 2.66 s per API call). Split: the laptop continues through 3a-19 and `scripts/laptop_stop_after_3a19.sh` (background) then kills the lane and releases the stay-awake helpers; adroit-vis takes the other 14 vlm scenes (3a-9 down to 3a-3, 3a-25 down to 3a-20, 3a-2) on 4 lanes plus Opus 5.5 text on 3a-1, 3a-10 to 3a-13 on a fifth. Brief sent to the node session (Remote Control route, delivery unconfirmed) and to the relay session (held by that session's permission mode, undelivered). The vlm row for Tom will combine success over both machines (success was machine-independent for the shipped agents) with latency per machine (A15).

## Oct 27 18:35: A25 decided (no HAZARD patch); laptop aggregator now carries the calc_value.py column

CR's words: "No HAZARD patch. Report flood and wind value rates from calc_value.py and keep the in-run column for the record." Recorded in ASSUMPTIONS.md A25 (CR approved as stated). HAZARD unchanged on both machines.

Laptop `scripts/aggregate_sweeps.py` (backup `.cache/bak/aggregate_sweeps.py.bak_a25`): new column "value rate, calc_value.py (scenes scored)" placed before the in-run column, which is now labelled "value rate in-run mean (sd)". `--calc` runs HAZARD's `src/HAZARD/utils/calc_value.py <sweep dir> <fire|flood|wind>` for every `results/sweep_*` folder from the HAZARD repo root with the venv python (both HAZARD paths on PYTHONPATH, A10), parses the four averages it prints before its cosmetic ZeroDivisionError, and caches them in `results/calc_value_laptop.json` (laptop numbers; the node keeps `results/calc_value_adroit.json`, A15). Without `--calc` the cached file is read. Took 2 m 7 s for seven sweep folders while the vlm lane was running; the lane was not touched.

Laptop table, `results/aggregate_laptop.md`, 18:30 (laptop numbers; the fire test set, A21):

| run | scenes | value rate, calc_value.py | value rate in-run (sd) | rescued any | decisions/ep | latency mean / max (s) | frames if clocked (of 1500) | API mean / max (s) | cost USD | status |
|---|---|---|---|---|---|---|---|---|---|---|
| fire_rule | 25 | 0.075 | 0.075 (0.11) | 0.081 | 18.4 | 0.000 / 0.000 | 18 | n/a | 0 | 0 |
| fire_greedy | 25 | 0.114 | 0.114 (0.11) | 0.146 | 19.5 | 0.334 / 0.862 | 205 | n/a | 0 | 0 |
| fire_mcts | 25 | 0.151 | 0.151 (0.22) | 0.150 | 19.6 | 4.269 / 16.997 | 2515 | n/a | 0 | 0 |
| fire_random | 25 | 0.356 | 0.356 (0.22) | 0.364 | 53.8 | 0.000 / 0.000 | 54 | n/a | 0 | 0 |
| fire_llm_haiku | 25 | 0.170 | 0.170 (0.22) | 0.154 | 45.2 | 1.666 / 14.392 | 2272 | 2.42 / 9.83 | 0.18 | 0 |
| fire_vlm_haiku | 7 so far | 0.063 | 0.063 (0.09) | 0.077 | 15.5 | 5.193 / 9.245 | 2423 | 2.72 / 7.45 | 0.06 | running (laptop stops after 3a-19) |

The rule and greedy redos are in: both rows are now 25 of 25 (rule 0.071 on 24 scenes became 0.075; greedy 0.118 on 24 became 0.114; `sweep_fire_rule_ep25` is the abandoned 2-scene Oct 26 folder, not a row for Tom). Fire: calc_value.py equals the in-run evaluator in every row, as on Adroit. The laptop has no flood or wind sweep folders, so its calc_value.py column has nothing to add there yet; the Oct 16 single-episode flood numbers (`results/flood_*_ep1_*`) are in-run and biased low (A25), and the Oct 16 wind single episodes were scored with calc_value.py on Oct 27 (random 0.130, others 0). For Tom's table: fire from either column, flood and wind from calc_value.py (Adroit's 12-row table above), in-run kept beside them for the record.

## Oct 27 18:5x: adroit-vis is down; the split is cancelled, the laptop runs the whole vlm and Opus sweeps

From the laptop: `adroit-vis.princeton.edu` (128.112.128.83) does not answer on port 22 or ping; the login node `adroit.princeton.edu` (128.112.128.94) answers on port 22. The node session shows "Reconnecting to adroit-vis" with its last turn six hours ago, so the brief sent at 18:25 has not been read. CR's scp put the (already present) bundle, not the key, on the node, so the key is not there either. `scripts/laptop_stop_after_3a19.sh` killed before it fired: the v3 lane continues the vlm sweep through all 25 scenes and then Opus 5.5 on 5 scenes (it read `SKIP_LAPTOP_VLM` before the file existed), ETA about 22:40. If the node comes back, the split can be revived by stopping the laptop at the meeting point by hand; duplicated scenes would cost about 0.01 USD each.

## Oct 27 19:10: session restart, one API blip, one hung reset; pass 2 queued

The app restarted this Claude session around 19:00 and marked its background tasks "stopped", but the processes were alive (checked by pid): the v3 lane, the run_baseline for vlm 3a-17, and the Oct 26 stay-awake helper (marker `SWEEP V2 ALL DONE`, which the lane's last line prints). The v3 stay-awake helper did die; the Oct 26 one covers the rest. The stop-after-3a-19 watcher had survived its TaskStop (the wrapper died, the bash child did not); killed by pid at 19:03 so it cannot stop the lane.

Two vlm failures: 3a-16 exit 1 at 18:34, `anthropic.APIConnectionError: Connection error` (network blip; api.anthropic.com and TDW's S3 reachable again at 19:03); 3a-17 started 18:35 and sat at "Resetting Environment" with zero decisions for 28 minutes, so the 30-minute cap ends it at about 19:05 (probably the build waiting on an asset download during the same blip). `scripts/laptop_lane_v3_pass2.sh` (background) waits for `LANE V3 DONE`, then redoes any vlm or Opus scene without eval_result.json once, and prints `LANE V3 PASS2 DONE`. Granola transcript of the Oct 16 meeting re-read for the Tom message: Tom asked "do you have a regular VLM? Is it using the images?" and "once you have the baselines done, send me the results ... and maybe some videos ... then we can meet again".

## Oct 28 02:45: v3 lane finished 22:51; pass 2 was asleep for four hours (Modern Standby)

v3 lane: vlm 19 of 25 (six failures), Opus 5 of 5, `LANE V3 DONE 22:50:57`. Pass 2 started 22:51 on vlm 3a-16 and the laptop entered Modern Standby at about 22:56 (Windows System log: Kernel-Power 506 at 19:57 and 20:21, 507 wake at 20:14 and 02:42); the scene's decisions resumed at 02:43 when CR woke it, 8 decisions in. The Oct 26 stay-awake helper had exited at 22:50:57 (its marker printed) and the v3 one had died in the 19:00 session restart, so pass 2 ran with no hold; a new helper (`LANE V3 PASS2 DONE`) started 02:44. The 19:57 and 20:21 standbys happened with a helper running, so `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` does not stop Modern Standby on this laptop (lid or power button; A: unknown which).

vlm failures: 3a-17 exit 124 (hung at reset, 30-min cap, 19:05); 3a-24, 3a-25, 3a-3, 3a-6 exit 1 at 20:45 to 21:23, all `anthropic.APITimeoutError` (network throttled around the standby transitions); 3a-16 the 18:34 connection error. All six are in pass 2, about 11 min each, so about 03:50 if the laptop stays awake.

Opus 5.5 text, 5 scenes (3a-1, 3a-10 to 3a-13), complete: value rate 0.138 (0.09), rescued 0.184, 18.4 decisions/ep, decision 6.48 s mean / 11.9 s max, API 3.73 / 9.49 s, 3586 frames if clocked. vlm so far (19 scenes): value rate 0.201 (0.26), rescued 0.196, 35.1 decisions/ep, decision 2.27 s mean / 37.3 s max, API 2.55 / 7.45 s, 0.06 USD.

## Oct 28 03:30: all laptop sweeps complete; pass 2 recovered every failed scene

Pass 2 (02:51 to 03:30 after the wake): vlm 3a-16, 3a-17, 3a-24, 3a-25, 3a-3, 3a-6 all exit 0. Final laptop coverage, fire test set: rule 25, greedy 25, mcts 25, random 25, Haiku text 25, Haiku with images 25, Opus text 5. Table regenerated: `results/aggregate_laptop.md` (the "calc_value.py" column, added by the relay session from the Adroit post-hoc scorer, is stale for the vlm and Opus rows; for fire it equals the in-run value rate, which is the column used below).

| agent | scenes | value rate (sd) | rescued | decisions/ep | decision s mean / max | frames if clocked / 1500 | API s mean / max | cost USD |
|---|---|---|---|---|---|---|---|---|
| rule | 25 | 0.075 (0.11) | 0.081 | 18.4 | 0.0001 / 0.0001 | 18 | | |
| greedy | 25 | 0.114 (0.11) | 0.146 | 19.5 | 0.334 / 0.862 | 205 | | |
| mcts | 25 | 0.151 (0.22) | 0.150 | 19.6 | 4.27 / 17.0 | 2515 | | |
| random | 25 | 0.356 (0.22) | 0.364 | 53.8 | 0 / 0 | 54 | | |
| Haiku 5.5 text | 25 | 0.170 (0.22) | 0.154 | 45.2 | 1.67 / 14.4 | 2272 | 2.42 / 9.83 | 0.18 |
| Haiku 5.5 + image (A20) | 25 | 0.168 (0.24) | 0.166 | 32.8 | 2.50 / 37.3 | 2475 | 2.53 / 7.45 | 0.20 |
| Opus 5.5 text, 5 scenes | 5 | 0.138 (0.09) | 0.184 | 18.4 | 6.48 / 11.9 | 3586 | 3.73 / 9.49 | 1.37 |

Reading: the image does not change Haiku's success (0.168 vs 0.170) but changes its behaviour: 32.8 decisions per episode instead of 45.2 (fewer blind pick_up attempts) and a 37 s worst-case decision (one call; the API max is 7.45 s, so the extra is image encoding plus HAZARD's own processing on that step; to be checked in decisions.jsonl before quoting). Opus on 5 scenes is 0.138 with half the decisions of Haiku and 2.7x the per-decision time; 5 scenes is too few to rank it against Haiku (sd 0.09 on 5). Opus cost 1.37 USD for 5 episodes (0.27 per episode, 25 episodes would be about 7 USD, affordable with the 200 USD credit CR now has). Total Claude spend on the laptop: 1.75 USD.

Action mix (Oct 28 03:35): Haiku + image over 819 decisions: pick_up 412, walk_to 344, drop 41, explore 22 (text-only was pick_up 726 of 1129). Opus over 92 decisions: walk_to 69, explore 9, pick_up 7, drop 7, the planner-like pattern of rule/greedy/mcts, and it scores like them. Wall per scene: Haiku + image 1168 s mean (text 598 s): the image path doubles the episode's wall time, which is the 37 s worst-case decision's family, not the API. Scenes with any value rescued: Haiku + image 17 of 25, Opus 4 of 5. All helpers and lanes have exited; nothing is running on the laptop.

### Oct 28, 12:30: demo videos were frame-shuffled; re-encoded (Tom's "firefighter appears in the fire")

Tom (Slack DM, Oct 28 12:1x) asked why the MCTS video shows the agent momentarily inside the fire. Cause: HAZARD saves the
record camera's frames as `img_<comm_counter>.jpg` with no zero padding (`fireagent_controller.py:473-474`,
`filename=str(self.comm_counter)`), and `scripts/make_videos.py` sorted the paths as text, so the sequence ran
0, 1, 10, 100, 1000..1009, 101, 1010, ... A 1 fps contact sheet of the old mp4 alternates between early frames (agent at the
fire origin, two small fire patches) and late frames (fire grown over that same corner, agent at the bottom), which is the
"agent in the fire" Tom saw. Fix: `make_videos.py` now sorts by the integer in the file name; both videos re-encoded
(`venv/Scripts/python.exe scripts/make_videos.py`, 3 m 17 s for 1503 + 1548 frames); the old files kept in
`results/videos/old_lexsort/`. Verified with `ffmpeg -i <mp4> -vf "fps=1,scale=640:-1,tile=4x3"` contact sheets: continuous.

Two real behaviours visible in the corrected MCTS video (`decisions.jsonl` frames -> seconds at 30 fps):
- 1.4 s to 9.7 s (decisions 1-6): the agent walks to the fire origin and picks up and drops two targets (hair comb 8657745,
  comb 15599511) while standing beside, at times inside, the two small fire patches. Nothing in HAZARD stops or penalises
  this: the fire is a visual effect with no collider (`fire.py:79`), the A* weights in `policy/astar.py:36-58` use only
  explored/unexplored and height (obstacles), not temperature, and the "agent dies if too hot" check is commented out
  (`fireagent_controller.py:447-451`; `AgentStatus` endurance 60 at `envs/fire/object.py:61-67` is tracked but unused).
- 12.1 s to 28.4 s (decisions 7-10, four consecutive `walk_to 56` for bag_3 at the bottom of the room): the agent leaves
  through the top-left doorway, is outside the record camera's view for about 12 s (raw frames 400 and 600 show it beyond
  the walls), and re-enters through the bottom doorway at about 26 s. Each walk_to is one A* step on the semantic map
  (`env_actions.py:117-124`) with unexplored cells at weight 30; it took four calls to arrive. The VLM video shows the same
  excursion at about 13-16 s. Not a recording artefact; worth raising with Tom as a HAZARD navigation quirk.

## Nov 3: the paper-gap is a known open issue on the HAZARD repo (found while drafting the email to the authors)

GitHub issue #10 "Issues with Baseline Reproduction" (yangqingz, Apr 2024, still open): greedy about 31% value rate, MCTS and LLM about 20% after the suggested fix; maintainer zhouqqhh (Qinhong Zhou) replied that (a) the paper's experiments used a TDW build "from around 2023.10", (b) `move_forward` still moves slightly on a detected collision so repeated attempts walk through walls, which they "addressed by adjusting the weights in the A* algorithm" (large penalty for height above 1.6), and (c) the README links the authors' own log files for reproduction. Another commenter (liuxiao916) suspected points recorded around the room in the 2D map construction, sending the agent to targets outside the room. Issue #16 "Questions on reproducing the result" (namwoam, May 2025, no reply): GPT-4o with llm-v2 gives fire value 0.23, flood 0.14, and asks which of llm-v1/v2/v3 is the paper's "LLM pipeline". So the gap we measured (rule 0.08, mcts 0.15, Haiku 0.17) is consistent with what others get from the public code; the authors' numbers came from an older TDW build (we run 1.13.0, A2). Open: compare the authors' shared logs (README) with ours for actions per episode; check whether our A* copy has the height penalty. Contact: Qinhong Zhou zhouqqhh@gmail.com (the repo's committer address), Chuang Gan ganchuang@csail.mit.edu (homepage).

## Nov 3 16:10: HAZARD authors replied; repro bundle packed
Qinhong Zhou replied within 40 min to Liane's email: "collision detection bugs in some versions of TDW", asked for trajectories and logs. Checked the HAZARD checkout before answering: agents (`policy/rule_based.py`, `fire_flood_heuristic.py`, `mcts.py`, `rand.py`) unmodified at d35c71e; local edits are A16 (challenge.py logging guard), A19 (controllers, `_previous_action` chain) and A18/A20 (llm.py, unused by the shipped-agent runs). Correction to the email: random is HAZARD's own `rand.py`, we added only the row. Bundle: `results/for_hazard_authors/` (README, patches/, per-scene actions.txt, log.txt, eval_result.json, decisions.jsonl, summary.json, output.log for 100 scenes, PNGs excluded), zipped to `results/hazard_fire_test25_repro_logs_2026-11-03.zip`. Laptop runs only; Adroit copies are on the node (down). Not sent yet.
