"""A39 step (iv): score a compiled program on a fixed list of training scenes (never the 25 test kitchens) and write
validation.json for the next compile round. One HAZARD process per scene through scripts/run_baseline.py --agent program,
resumable (a scene with eval_result.json is skipped), clock on by default. Runs the simulator: start it only with CR's go.

Usage (from G:\\iw-thinking-fast; on Della add --build/--launch/--cached_assets as the Slurm scripts do):
  python scripts/compiled/validate.py --program results/compiled/round1/policy_round1.py --out results/compiled/round1/validation \\
      --scenes mm_craftroom_2a-1 mm_craftroom_2a-2 mm_kitchen_2a-1 mm_kitchen_2a-2 mm_craftroom_3a-1 [--clock tcm] [--dry_run]
Writes <out>/validation.json: per scene value rate, mean, sd, decisions, illegal decisions, frames charged; the same file
compile.py --validation reads. Tests: scripts/compiled/test_validate.py.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(ROOT, "scripts", "run_baseline.py")
DEFAULT_SCENES = ["mm_craftroom_2a-1", "mm_craftroom_2a-2", "mm_kitchen_2a-1", "mm_kitchen_2a-2", "mm_craftroom_3a-1"]
TRAIN = os.path.join(ROOT, "HAZARD", "src", "HAZARD", "data", "room_setup_fire")


def scene_dir(name, train_root=TRAIN):
    """Training scenes live directly under room_setup_fire/<name>; the 25 test kitchens under test_set/ and are refused."""
    if name.startswith("mm_kitchen_3a-"):
        raise ValueError(f"{name} is a test kitchen (test_set); validation uses training scenes only (A39 point 4)")
    return os.path.join(train_root, name)


def run_command(program, scene, out_dir, clock="tcm", fps=30, port=10826, build=None, launch="", extra=()):
    cmd = [sys.executable, "-X", "utf8", RUNNER, "--env", "fire", "--agent", "program", "--program", program, "--episodes", "1",
           "--port", str(port), "--out", os.path.join(ROOT, "results"), "--resume_dir", out_dir, "--data_dir", scene_dir(scene),
           "--clock", clock, "--fps", str(fps)]
    if build:
        cmd += ["--build", build]
    if launch:
        cmd += ["--launch", launch]
    cmd += list(extra)
    return cmd


def collect(out_dir, scenes):
    """validation.json content from the per-scene folders that exist."""
    per = {}
    for s in scenes:
        ev = os.path.join(out_dir, s, "eval_result.json")
        summ = os.path.join(out_dir, s, "summary.json")
        if not os.path.exists(ev):
            per[s] = None
            continue
        e = json.load(open(ev))
        rec = {"value_rate": round(e["avg_score"] / e["avg_max_score"], 3) if e.get("avg_max_score") else None}
        if os.path.exists(summ):
            sm = json.load(open(summ))
            rec["decisions"] = sm.get("decisions")
            rec["frames_charged"] = sm.get("frames_charged_total")
            prog = sm.get("program") or {}
            rec["illegal_or_crashed"] = prog.get("illegal_or_crashed")
            rec["wall_s_mean"] = prog.get("wall_s_mean")
            rec["actions"] = prog.get("actions")
        per[s] = rec
    rates = [r["value_rate"] for r in per.values() if r and r["value_rate"] is not None]
    return {"scenes": per, "n": len(rates), "mean": round(statistics.mean(rates), 3) if rates else None,
            "sd": round(statistics.stdev(rates), 3) if len(rates) > 1 else None,
            "illegal_total": sum((r.get("illegal_or_crashed") or 0) for r in per.values() if r)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--clock", default="tcm", choices=["none", "tcm"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=10826)
    ap.add_argument("--build", default=None)
    ap.add_argument("--launch", default="")
    ap.add_argument("--extra", nargs="*", default=[], help="extra run_baseline flags, e.g. --cached_assets")
    ap.add_argument("--timeout", type=int, default=1500, help="seconds per scene")
    ap.add_argument("--dry_run", action="store_true", help="print the commands and the collected results, run nothing")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = []
    for s in args.scenes:
        if os.path.exists(os.path.join(args.out, s, "eval_result.json")):
            log.append({"scene": s, "skipped": True})
            continue
        cmd = run_command(args.program, s, args.out, clock=args.clock, fps=args.fps, port=args.port, build=args.build,
                          launch=args.launch, extra=args.extra)
        if args.dry_run:
            log.append({"scene": s, "cmd": cmd})
            continue
        t = time.perf_counter()
        r = subprocess.run(cmd, cwd=os.path.join(ROOT, "HAZARD"), timeout=args.timeout)
        log.append({"scene": s, "exit": r.returncode, "wall_s": round(time.perf_counter() - t, 1)})
    result = collect(args.out, args.scenes)
    result["program"] = args.program
    result["clock"] = args.clock
    result["log"] = log
    with open(os.path.join(args.out, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
