"""Aggregate results/sweep_* folders into one markdown table (the numbers Tom asked for: success and latency,
mean and max, per agent over all scenes). Usage:  ./venv/Scripts/python.exe scripts/aggregate_sweeps.py [--md out.md]
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import subprocess

ROOT = r"G:\iw-thinking-fast"
# A24/A25 (CR, Oct 27): flood and wind value rates are reported from HAZARD's own post-hoc scorer
# src/HAZARD/utils/calc_value.py; the in-run column stays for the record (wind in-run is hardcoded 0 at
# challenge.py:191,226-228; flood in-run is biased low by the enum mismatch at challenge.py:216-218, A25).
# --calc runs the scorer over every sweep dir and caches it here (laptop numbers; the node keeps
# results/calc_value_adroit.json, A15). Without --calc the cached file is read if present.
CALC_JSON = os.path.join(ROOT, "results", "calc_value_laptop.json")
CALC = {}


def run_calc_value(dirs):
    """calc_value.py <sweep dir> <fire|flood|wind>, from the HAZARD repo root (relative data paths) with the venv
    python and both HAZARD paths on PYTHONPATH (A10). Its averages print before the cosmetic ZeroDivisionError at
    line 119 (nothing picked), so they still parse; a sweep with no eval_result.json yields nan."""
    py = os.path.join(ROOT, "venv", "Scripts", "python.exe")
    hz = os.path.join(ROOT, "HAZARD")
    env = dict(os.environ, PYTHONPATH=hz + os.pathsep + os.path.join(hz, "src", "HAZARD"))
    out = {}
    for d in dirs:
        name = os.path.basename(d)
        task = name.split("_")[1] if "_" in name else ""
        if task not in ("fire", "flood", "wind"):
            continue
        p = subprocess.run([py, "-X", "utf8", os.path.join("src", "HAZARD", "utils", "calc_value.py"), d, task],
                           cwd=hz, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        rec = {"task": task, "exit": p.returncode, "valid": 0, "rate": float("nan"), "value_rate": float("nan"),
               "step": float("nan"), "damage": float("nan")}
        lines = p.stdout.splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("task name ") and i + 1 < len(lines) and lines[i + 1].strip().isdigit():
                rec["valid"] = int(lines[i + 1].strip())
            for key, label in (("rate", "Average rate "), ("value_rate", "Average value rate "),
                               ("step", "Average step "), ("damage", "Damage rate ")):
                if ln.startswith(label):
                    try:
                        rec[key] = float(ln[len(label):].strip())
                    except ValueError:
                        pass
        out[name] = rec
    json.dump(out, open(CALC_JSON, "w", encoding="utf-8"), indent=1)
    return out


def load(d):
    evs = {}
    for p in glob.glob(os.path.join(d, "*", "eval_result.json")):
        scene = os.path.basename(os.path.dirname(p))
        try:
            ev = json.load(open(p))
            evs[scene] = ev
        except Exception:
            pass
    # Oct 16 lanes wrote one decisions.jsonl/summary.json per sweep folder; the Oct 26 v2 lane (one process per
    # scene) writes them under <sweep>/<scene>/. Read both; per-scene records get episode = scene name.
    decs = []
    for dp in [os.path.join(d, "decisions.jsonl")] + sorted(glob.glob(os.path.join(d, "*", "decisions.jsonl"))):
        if os.path.exists(dp):
            scene = os.path.basename(os.path.dirname(dp)) if os.path.dirname(dp) != d else None
            for l in open(dp, encoding="utf-8"):
                if l.strip():
                    x = json.loads(l)
                    if scene: x["episode"] = scene
                    decs.append(x)
    sm = {}
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        sm = json.load(open(sp))
    else:  # merge per-scene summaries: llm call stats summed, exit = 0 only if every scene finished
        sms = [json.load(open(q)) for q in sorted(glob.glob(os.path.join(d, "*", "summary.json")))]
        if sms:
            sm = {"exit": max(x.get("exit", 1) for x in sms), "scenes_with_summary": len(sms)}
            ll = [x["llm"] for x in sms if x.get("llm")]
            if ll:
                calls = sum(x["calls"] for x in ll)
                sm["llm"] = {"calls": calls,
                             "call_wall_s": {"mean": sum(x["call_wall_s"]["mean"] * x["calls"] for x in ll) / max(calls, 1),
                                             "max": max(x["call_wall_s"]["max"] for x in ll)},
                             "cost_usd_estimate": sum(x.get("cost_usd_estimate", 0.0) for x in ll),
                             "input_tokens": sum(x.get("input_tokens", 0) for x in ll),
                             "output_tokens": sum(x.get("output_tokens", 0) for x in ll)}
    return evs, decs, sm


def row(d):
    name = os.path.basename(d)
    evs, decs, sm = load(d)
    n = len(evs)
    vr = [ev.get("avg_score", 0) / max(ev.get("avg_max_score", 1), 1) for ev in evs.values()]
    rescued_any = [sum(1 for v in ev.get("target_status", {}).values() if v is not False) for ev in evs.values()]
    targets = [len(ev.get("target_status", {})) for ev in evs.values()]
    lat = [x["wall_s"] for x in decs]
    per_ep = {}
    for x in decs:
        per_ep.setdefault(x.get("episode", 1), []).append(x["wall_s"])
    frames_if_clocked = [sum(math.ceil(v * 30) for v in vals) for vals in per_ep.values()]
    llm = sm.get("llm", {})
    calc = CALC.get(name, {})
    step_limit = 3000 if "wind" in name else 1500  # challenge.py:53-59 (max_steps); mirrors the adroit copy (A22 item 19 follow-up)
    return {
        "run": name, "scenes": n, "step_limit": step_limit,
        "value_rate_mean": st.mean(vr) if vr else float("nan"), "value_rate_sd": st.pstdev(vr) if len(vr) > 1 else 0.0,
        "calc_value_rate": calc.get("value_rate", float("nan")), "calc_valid": calc.get("valid", 0),
        "success_any_mean": (sum(rescued_any) / sum(targets)) if sum(targets) else float("nan"),
        "decisions_per_ep": (len(decs) / len(per_ep)) if per_ep else float("nan"),
        "lat_mean": st.mean(lat) if lat else float("nan"), "lat_max": max(lat) if lat else float("nan"),
        "frames_if_clocked_mean": st.mean(frames_if_clocked) if frames_if_clocked else float("nan"),
        "api_call_mean": llm.get("call_wall_s", {}).get("mean", float("nan")), "api_call_max": llm.get("call_wall_s", {}).get("max", float("nan")),
        "cost_usd": llm.get("cost_usd_estimate", 0.0), "exit": sm.get("exit", "running"),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--md", default="")
    ap.add_argument("--calc", action="store_true", help="rerun HAZARD's calc_value.py over every sweep dir (A24/A25)")
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(ROOT, "results", "sweep_*")))
    global CALC
    if args.calc:
        CALC = run_calc_value(dirs)
    elif os.path.exists(CALC_JSON):
        CALC = json.load(open(CALC_JSON, encoding="utf-8"))
    rows = [row(d) for d in dirs]
    hdr = "| run | scenes | value rate, calc_value.py (scenes scored) | value rate in-run mean (sd) | targets rescued (any value) | decisions/ep | decision latency mean / max (s) | frames if clocked, mean/ep (of step limit) | API call mean / max (s) | cost USD | status |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        f = lambda v, p=3: ("n/a" if isinstance(v, float) and math.isnan(v) else (f"{v:.{p}f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {r['run']} | {r['scenes']} | {f(r['calc_value_rate'])} ({r['calc_valid']}) | {f(r['value_rate_mean'])} ({f(r['value_rate_sd'], 2)}) | {f(r['success_any_mean'])} | {f(r['decisions_per_ep'], 1)} | {f(r['lat_mean'])} / {f(r['lat_max'])} | {f(r['frames_if_clocked_mean'], 0)} (of {r['step_limit']}) | {f(r['api_call_mean'], 2)} / {f(r['api_call_max'], 2)} | {f(r['cost_usd'], 2)} | {r['exit']} |")
    md = hdr + "\n".join(lines) + "\n"
    print(md)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(md)


if __name__ == "__main__":
    main()
