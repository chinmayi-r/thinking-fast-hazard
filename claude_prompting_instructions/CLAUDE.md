# iw-thinking-fast: HAZARD benchmark work for CR's fall 2026 IW

Advisor Tom Silver (co-advisor Karthik Narasimhan). Project brief: "Thinking Fast Enough": a robot with a preparation budget X and an online budget Y, a toolbox of solvers of different speeds, and a learned gate choosing among them while the world keeps moving.

## Read these first, read-only
- Lit review vault (Obsidian, git repo, do NOT write into it): `G:\Obsidian\Princeton-Obsidian\Princeton\PRPL Research\Lit Review\`
  - `Two Lineages.md`: the planning line, the RL line, and the third strand (RRARA), with what each paper does and leaves out.
  - `Port Preconditions.md`: for each method, what it assumes versus what HAZARD provides, and the verdict on whether a port is direct, adapted, or impossible. Copied here as `ASSUMPTIONS.md`.
  - `Papers/zhouHAZARDChallengeEmbodied2024.md` and `Papers/zhengReflexFirstReflect2025.md`: the benchmark and the one latency-aware agent on it.
  - `Map.md`: every paper scored on preparation / several solvers / world moves.
- Tom's brief: `G:\Obsidian\Princeton-Obsidian\Princeton\PRPL Research\Thinking Fast Enough.pdf`.

## Rules CR set
1. **No silent assumptions.** Every assumption (Python version, GPU, simulator build, which paper, any unpublished constant such as RRARA's staleness weights) is written down in `ASSUMPTIONS.md` under "Open assumptions" and raised with CR before acting on it.
2. **Preconditions before code.** Before implementing any method on HAZARD, list the conditions its results rest on (full observability, perfect simulator, known deadlines, deterministic outcomes, fixed timestep, symbolic model, etc.), compare each to HAZARD, and state what breaks. The table in `ASSUMPTIONS.md` is the starting point; extend it, do not skip it.
3. **Read code and docs, not blog posts.** Cite file paths and line numbers for every claim about how HAZARD works, especially how many frames each high-level action consumes, which is undocumented and must be measured.
4. **Keep everything in this folder.** Code, venvs, ThreeDWorld build, results. Nothing in the vault.
5. **Latency instrumented from day one.** Log wall-clock decision time per step and the frames it would cost at 30 FPS (RRARA's convention: frames = ceil(seconds × 30)).
6. **If context is missing**, stop and ask CR. CR has a separate Claude session open in the vault folder that holds the full lit-review context and can answer questions about any paper's assumptions; CR will relay.

## Tom's order (Oct 1 meeting, his words)
1. "First of all, see if the benchmark runs."
2. "Can we take a baseline from one of these prior works" (the lit-review papers).
3. His weak baseline: "simply ask Claude or Codex to try and solve these environments," expected to fail because "by the time it's done thinking, the world will be destroyed."
4. The research idea to try first: a fast LLM acts at step n while a slow LLM reasons about step n+1, reconciled continuously (Karthik's speculative-decoding analogy). Tom: "more direct" than true anytime. RRARA (arXiv 2506.07223) is this idea with a rule-based reflex; reimplementing it is the obvious step 2.

## Machine facts (checked Oct 16)
- Windows 11, PowerShell 5.1. GPU: NVIDIA RTX 3060 Laptop, 4 GB VRAM. Also Intel Iris Xe.
- Disk: C has 1.1 GB free (do not install anything on C); G has 100 GB free (use G); H has 157 GB free.
- Python on PATH is 3.14 from the Microsoft Store. HAZARD's dependencies (mmcv, torch, gym) will very likely need Python 3.9 or 3.10 in a dedicated venv or conda env on G. Confirm before installing.
- ThreeDWorld "requires a display and a GPU"; Windows is listed as supported; the detailed PC install page has not been read yet.
- git at `C:\Program Files\Git\cmd\git.exe`.

## The plan, in CR's order (Oct 16)
1. **HAZARD runs.** ThreeDWorld install, HAZARD install, smallest demo for fire, flood, wind; the shipped agents (random, greedy, rule, MCTS, LLM) on a few episodes; reference numbers under `results/`; exact commands in `NOTES.md`. Measure frames per high-level action.
2. **Baselines under a clock.** Add the time-conversion rule (frames = ceil(seconds × 30)) and rerun the shipped agents; reproduce RRARA's Table 1 degradation. Then RRARA itself (reflex + async reflector + cached preplanner), from the paper; flag the unpublished staleness weights as an open assumption.
3. **Most recent paper from each chain, adapted.** Planning line: Coles 2024. RL line: Muppidi 2026. For each, go through `ASSUMPTIONS.md` first: list the paper's preconditions, compare to HAZARD, write what breaks and what adapter is needed, show CR, then implement the adapted version. Report what survived and what did not, side by side with the paper's own numbers on its own benchmark.
4. **The papers Tom's brief cites, same process.** Shperberg 2021 (deadline-aware subgoal scoring), Sleep-time Compute (a preparation axis X), Russell & Wefald (the hand-derived gate rule), Moerland (only if a trained gate exists). Verdicts are in `ASSUMPTIONS.md`.
Each step ends with a report: what ran, what did not, every open assumption. Do not start the next step without CR's go.
