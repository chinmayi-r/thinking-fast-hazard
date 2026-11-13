"""Run HAZARD's shipped agents with per-decision latency instrumentation (CLAUDE.md rule 5), without editing HAZARD.

Usage (Git Bash, from G:\iw-thinking-fast, venv active or via ./venv/Scripts/python.exe):
    python scripts/run_baseline.py --env fire --agent rule --episodes 1 --port 10826 \
        --build "G:\iw-thinking-fast\tdw_build\TDW\TDW.exe" --out results

What it does
- starts the TDW build by hand (A3) with its output in logs/, waits, then calls HAZARD.submit(...)
- wraps the agent so every choose_target() call is timed: wall-clock seconds, frames = ceil(seconds * 30)
  (RRARA's convention, A13), plus the controller's frame_count before the decision, so frames per
  high-level action fall out as consecutive differences
- the wrapper is handed to submit() as a custom agent object (submit.md:59-67), so HAZARD's own loop is untouched
- writes <out>/<env>_<agent>_<stamp>/decisions.jsonl and summary.json

Linux (adroit-vis): pass --build .../TDW/TDW.x86_64 --launch "vglrun -d egl" and run under Xvfb (DISPLAY set).
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HOLDER = {}


def install_challenge_hook():
    """Remember the Challenge instance HAZARD builds inside submit(), to read env.controller.frame_count."""
    import src.HAZARD.challenge as chmod  # the module name HAZARD itself uses (run_experiments.py:1)
    orig = chmod.Challenge.__init__

    def patched(self, *a, **k):
        orig(self, *a, **k)
        HOLDER["challenge"] = self

    chmod.Challenge.__init__ = patched


class TimedAgent:
    """Delegates everything to the shipped agent; times choose_target and logs frame counts."""

    def __init__(self, inner, log_path: Path, clock: str = "none", k: int = 0, fps: int = 30):
        self.inner = inner
        self.clock, self.k, self.fps = clock, k, fps
        self.frames_charged_total = 0
        self.agent_type = inner.agent_type  # challenge.py dispatches on this
        self.log = open(log_path, "w", encoding="utf-8")
        self.n = 0
        self.episode = 0
        self.latencies = []

    def __getattr__(self, name):  # only called for attributes not found on the wrapper
        return getattr(self.inner, name)

    def reset(self, *a, **k):
        self.episode += 1
        self.n = 0
        return self.inner.reset(*a, **k)

    def _frame(self):
        ch = HOLDER.get("challenge")
        try:
            return int(ch.env.controller.frame_count), int(ch.env.controller.comm_counter)
        except Exception:
            return -1, -1

    def choose_target(self, state, processed_input):
        f0, c0 = self._frame()
        t = time.perf_counter()
        action = self.inner.choose_target(state, processed_input)
        dt = time.perf_counter() - t
        self.n += 1
        self.latencies.append(dt)
        # The clock (step 2 of the plan; BASELINE.md section 4): charge the decision's duration in world frames by
        # stepping the simulator with the agent stationary, at the same point HAZARD's loop would execute the action.
        charged = 0
        if self.clock == "tcm":
            charged = math.ceil(dt * self.fps)
        elif self.clock == "k":
            charged = self.k
        if charged > 0:
            ctrl = HOLDER["challenge"].env.controller
            for _ in range(charged):
                ctrl.communicate([])
            self.frames_charged_total += charged
        f1, _ = self._frame()
        rec = {
            "frames_charged": charged,
            "frame_count_after_charge": f1,
            "episode": self.episode,
            "decision": self.n,
            "wall_s": round(dt, 6),
            "frames_at_30fps": math.ceil(dt * 30),
            "frame_count_before": f0,
            "comm_counter_before": c0,
            "action": str(action),
            "t_unix": round(time.time(), 3),
        }
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        return action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="fire", choices=["fire", "flood", "wind"])
    ap.add_argument("--agent", default="rule", help="rule | greedy | mcts | mctsv2 | random")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--port", type=int, default=10826)
    ap.add_argument("--build", default=r"G:\iw-thinking-fast\tdw_build\TDW\TDW.exe")
    ap.add_argument("--launch", default="", help='prefix for the build command, e.g. "vglrun -d egl"')
    ap.add_argument("--build_args", default="", help='extra args after -port for the build, e.g. "-logFile logs/unity_5.log" '
                                                      '(Unity standalone flag; one Player.log per concurrent task on a cluster)')
    ap.add_argument("--out", default=r"G:\iw-thinking-fast\results")
    ap.add_argument("--run_on_test", action="store_true")
    ap.add_argument("--no_build", action="store_true", help="assume a build is already listening on --port")
    ap.add_argument("--clock", default="none", choices=["none", "tcm", "k"],
                    help="none: HAZARD as shipped. tcm: RRARA's rule, advance ceil(t*fps) frames after each decision. "
                         "k: declared budget, advance exactly --k frames after each decision (Muppidi's form).")
    ap.add_argument("--k", type=int, default=0, help="frames charged per decision when --clock k")
    ap.add_argument("--fps", type=int, default=30, help="frames per second for --clock tcm (RRARA: 30)")
    # LLM agents (A18): HAZARD's llm / llmv2 with source openai or anthropic. The key is read by HAZARD from a file
    # you create yourself (one key per line); this runner never prints or stores it.
    ap.add_argument("--lm_source", default="anthropic", choices=["anthropic", "openai", "huggingface"])
    ap.add_argument("--lm_id", default="claude-opus-5-5")
    ap.add_argument("--api_key_file", default="", help="path to a file holding the API key (never echoed)")
    ap.add_argument("--max_tokens", type=int, default=512, help="HAZARD's max new tokens per LLM call")
    ap.add_argument("--resume_dir", default="", help="existing results folder to continue a sweep in (scenes with eval_result.json are skipped)")
    ap.add_argument("--record", action="store_true", help="HAZARD record_with_agents: capture every frame under <out>/images for a video (challenge.py:73)")
    ap.add_argument("--data_dir", default="", help="one specific scene folder (HAZARD data_dir); default: the training set in order")
    ap.add_argument("--reflector_mode", default="faithful", choices=["off", "faithful", "worldtime"],
                    help="rrara_v1 only (A30): off = reflex alone; faithful = RRARA's accounting, a finished background verdict "
                         "always counts; worldtime = it counts only if ceil(wall x fps) <= frames the checked action took")
    ap.add_argument("--reflex", default="paper", choices=["paper", "hazard_rule"],
                    help="rrara_v1 only (A34): paper = RRARA eq. (2) reimplemented (value over A* path distance); "
                         "hazard_rule = HAZARD's shipped RuleBasedAgent untouched")
    # Human baseline (A36, scripts/human): a person answers the LLM agent's menu on a local page. Clock off unless --clock.
    ap.add_argument("--human_port", type=int, default=8765, help="human only: port of the local page (0 = any free port)")
    ap.add_argument("--no_camera", action="store_true", help="human only: do not save or show the robot's camera frame (text-only run)")
    ap.add_argument("--no_browser", action="store_true", help="human only: do not open the page in the default browser")
    ap.add_argument("--human_hold", type=float, default=20.0, help="human only: seconds the final score stays on the page before exit")
    ap.add_argument("--cached_assets", action="store_true",
                    help="HAZARD use_cached_assets: load assets from <cwd>/data/assets instead of S3 (A28, cluster compute nodes "
                         "have no internet). Missing files make HAZARD write data/assets/download_assets.sh and exit 0.")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clock_tag = "" if args.clock == "none" else (f"_tcm{args.fps}" if args.clock == "tcm" else f"_k{args.k}")
    if args.resume_dir and args.data_dir:  # one scene per process (Oct 26): <sweep>/<scene>, skipped if already evaluated
        out_dir = Path(args.resume_dir) / Path(args.data_dir).name
        if (out_dir / "eval_result.json").exists():
            print(f"already evaluated: {out_dir}")
            return 0
    elif args.resume_dir:  # reuse a sweep folder: HAZARD skips scenes that already have eval_result.json (run_experiments.py:184-187)
        out_dir = Path(args.resume_dir)
    else:
        out_dir = Path(args.out) / f"{args.env}_{args.agent}{clock_tag}_ep{args.episodes}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.out).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    build = None
    if not args.no_build:
        build_log = open(logs_dir / f"build_{stamp}.log", "w")
        cmd = shlex.split(args.launch) + [args.build, f"-port {args.port}"] + shlex.split(args.build_args)
        build = subprocess.Popen(cmd, stdout=build_log, stderr=subprocess.STDOUT)
        print(f"build pid={build.pid}: {' '.join(cmd)}")
        time.sleep(5)

    install_challenge_hook()
    from src.HAZARD.run_experiments import get_examplar_agent, submit  # same module path HAZARD uses

    # absolute prompt path so LLM agents work from any cwd (HAZARD's own default is relative to src/HAZARD)
    prompt_path = str(Path(__file__).resolve().parents[1] / "HAZARD" / "src" / "HAZARD" / "policy" / "llm_configs" / "prompt_v2.csv")
    page = None  # the human baseline's page (A36); stays None for every other agent
    if args.agent == "rrara_v1":  # RRARA-v1 (scripts/rrara/agent.py): rule reflex + async LLM reflector; HAZARD's "custom" slot
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rrara.agent import RRARAv1Agent
        inner = RRARAv1Agent(task=args.env, mode=args.reflector_mode, lm_id=args.lm_id, api_key_file=args.api_key_file,
                             fps=args.fps, reflex=args.reflex)
    elif args.agent == "vlm":  # A20: HAZARD's llm agent + the camera frame per decision (scripts/vlm_agent.py)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vlm_agent import VLMAgent
        from policy.llm import SamplingParameters
        keys = [l.strip() for l in open(args.api_key_file) if l.strip()] if args.api_key_file else ""  # as run_experiments.py:28-33
        inner = VLMAgent(source=args.lm_source, lm_id=args.lm_id, prompt_template_path=prompt_path, cot=True,
                         sampling_parameters=SamplingParameters(max_tokens=args.max_tokens), task=args.env, api_key=keys)
    elif args.agent == "human":  # A36: HAZARD's LLM agent with a person as the generator (scripts/human/human_agent.py)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from human.runner import make_human_agent, LiveCounters
        inner, page = make_human_agent(task=args.env, prompt_path=prompt_path, get_challenge=lambda: HOLDER.get("challenge"),
                                       port=args.human_port, camera=not args.no_camera, open_browser=not args.no_browser)
        print(f"human page: {page.url}  (the first decision appears once the scene has loaded)")
    else:
        inner = get_examplar_agent(agent_name=args.agent, env_name=args.env, prompt_path=prompt_path,
                                   lm_source=args.lm_source, lm_id=args.lm_id, api_key_file=args.api_key_file,
                                   max_tokens=args.max_tokens)
    agent = TimedAgent(inner, out_dir / "decisions.jsonl", clock=args.clock, k=args.k, fps=args.fps)

    t0 = time.perf_counter()
    rc = 0
    err = ""
    try:
        submit(output_dir=str(out_dir), env_name=args.env, agent=agent, port=args.port,
               max_test_episode=args.episodes, run_on_test=args.run_on_test,
               record_with_agents=args.record, data_dir=args.data_dir, use_cached_assets=args.cached_assets)
    except Exception as e:  # keep the summary even when HAZARD raises (e.g. challenge.py:323 KeyError)
        import traceback
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        rc = 1
    wall = time.perf_counter() - t0

    # collect HAZARD's own per-episode eval_result.json files
    evals = {}
    for p in out_dir.rglob("eval_result.json"):
        try:
            evals[str(p.relative_to(out_dir))] = json.load(open(p))
        except Exception as e:
            evals[str(p.relative_to(out_dir))] = f"unreadable: {e}"
    lat = agent.latencies
    summary = {
        "env": args.env, "agent": args.agent, "episodes_requested": args.episodes,
        "clock": args.clock, "k": args.k, "fps": args.fps, "frames_charged_total": agent.frames_charged_total,
        "machine": os.environ.get("COMPUTERNAME") or os.uname().nodename,
        "wall_s_total": round(wall, 1), "decisions": len(lat),
        "decision_wall_s": {"mean": round(sum(lat) / len(lat), 4) if lat else None,
                             "max": round(max(lat), 4) if lat else None,
                             "min": round(min(lat), 4) if lat else None},
        "frames_at_30fps_total": sum(math.ceil(x * 30) for x in lat),
        "eval_results": evals, "exit": rc, "error": err[-2000:] if err else "",
        "final_frame_count": HOLDER.get("challenge").env.controller.frame_count if HOLDER.get("challenge") else None,
    }
    if hasattr(inner, "rrara_summary"):  # RRARA-v1: override counts, verdict timing, reflector calls
        summary["rrara"] = inner.rrara_summary()
        json.dump(inner.decision_log, open(out_dir / "rrara_decisions.json", "w"), indent=1)
        if inner.reflector is not None:
            json.dump(inner.reflector.verdicts, open(out_dir / "rrara_verdicts.json", "w"), indent=1)
            json.dump(inner.reflector.call_log, open(out_dir / "llm_calls.json", "w"), indent=1)
    if page is not None:  # A36: final score to the page, then the person's decisions in the summary
        from human.runner import finish_human_run
        summary["human"] = finish_human_run(page, inner, evals, counters=LiveCounters(lambda: HOLDER.get("challenge")),
                                            hold_s=args.human_hold)
    call_log = getattr(inner, "call_log", None)  # A18: per-API-call latency for llm agents
    if call_log:
        calls = [c["wall_s"] for c in call_log]
        summary["llm"] = {"source": args.lm_source, "model": args.lm_id, "calls": len(calls),
                          "call_wall_s": {"mean": round(sum(calls) / len(calls), 3), "max": round(max(calls), 3),
                                          "min": round(min(calls), 3), "total": round(sum(calls), 1)},
                          "input_tokens": sum(c["input_tokens"] for c in call_log),
                          "output_tokens": sum(c["output_tokens"] for c in call_log),
                          "stop_reasons": sorted(set(str(c["stop_reason"]) for c in call_log)),
                          "cost_usd_estimate": round(getattr(inner, "total_cost", 0.0), 4)}
        json.dump(call_log, open(out_dir / "llm_calls.json", "w"), indent=1)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    if build is not None:
        try:
            build.terminate()
            build.wait(timeout=10)
        except Exception:
            build.kill()
    return rc


if __name__ == "__main__":
    sys.exit(main())
