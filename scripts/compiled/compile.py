"""A39 step (iii): the compiler. Given the program API document, the worked example, recorded traces (step ii) and, from
round 2 on, the previous round's program and its validation results, ask a slow model (Claude Opus, CR's key, read only
from --api_key_file) for a new `Policy` and write it to policy_round<N>.py, audited by compiled.api.load_program before it
is accepted. Offline: no simulator; the validation runner (step iv) scores the file later. `--dry_run` writes the prompt
and stops, so the loop can be inspected without a model call. Tests: scripts/compiled/test_compile.py.

Usage (from G:\\iw-thinking-fast, PYTHONPATH as for the tests):
  python scripts/compiled/compile.py --traces results/sweep_fire_llm_haiku_test25 --kitchens mm_kitchen_3a-1 mm_kitchen_3a-10 \\
      --out results/compiled/round1 --round 1 --api_key_file G:\\iw-thinking-fast\\.secrets\\anthropic_key.txt [--dry_run]
  ... --round 2 --previous results/compiled/round1/policy_round1.py --validation results/compiled/round1/validation.json
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
API_DOC = os.path.join(HERE, "API.md")
EXAMPLE = os.path.join(HERE, "examples", "reflex_program.py")
DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS = 8000

SYSTEM = ("You write deterministic Python policies for a simulated rescue robot. You get an API document, a working example "
          "program, recorded episodes of a slow language-model agent (its decisions and what happened), and sometimes the "
          "previous program with its scores. Return ONE complete Python module in a single ```python block, defining "
          "`class Policy` exactly as the API requires, importing only what the API allows, with no model calls and no "
          "randomness. Before the code, write at most ten lines on what the recorded episodes show and what the program does "
          "differently. Prefer simple rules that cannot crash: every branch must return a legal action.")


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_prompt(api_doc, example_src, episodes_text, previous_src=None, validation=None, notes=None):
    parts = ["# The API a program plays through\n\n" + api_doc,
             "# A worked example that compiles and plays (the reflex)\n\n```python\n" + example_src.rstrip() + "\n```",
             "# Recorded episodes of the slow agent on training kitchens (what it was told, what it chose, what happened)\n\n" + episodes_text]
    if previous_src:
        parts.append("# The previous round's program\n\n```python\n" + previous_src.rstrip() + "\n```")
    if validation:
        parts.append("# How the previous program scored on the validation kitchens\n\n" + json.dumps(validation, indent=1))
    if notes:
        parts.append("# Notes from the person running this\n\n" + notes)
    parts.append("# Task\n\nWrite the next program. Keep what worked, fix what failed in the recorded outcomes (walks that "
                 "never arrive, repeated walks to one object, items shown as worth 5 that the scorer pays 1 for, time spent "
                 "standing still), and return the whole module in one ```python block.")
    return "\n\n".join(parts)


def extract_module(text):
    """The last ```python ... ``` block of a reply, or the whole reply if there is none."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip() + "\n"


def call_model(prompt, api_key_file, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS):
    """One Anthropic call; the key is read from the file and never printed. Returns (text, usage dict, wall seconds)."""
    import anthropic  # imported here so the dry run and the tests need no SDK
    key = [l.strip() for l in open(api_key_file, encoding="utf-8") if l.strip()][0]
    client = anthropic.Anthropic(api_key=key)
    t = time.perf_counter()
    msg = client.messages.create(model=model, max_tokens=max_tokens, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}])
    wall = time.perf_counter() - t
    text = "".join(getattr(b, "text", "") for b in msg.content)
    usage = {"input_tokens": getattr(msg.usage, "input_tokens", None), "output_tokens": getattr(msg.usage, "output_tokens", None),
             "model": getattr(msg, "model", model), "stop_reason": getattr(msg, "stop_reason", None)}
    return text, usage, wall


def compile_round(out_dir, round_no, episodes_text, api_key_file=None, previous_src=None, validation=None, notes=None,
                  model=DEFAULT_MODEL, dry_run=False, caller=None):
    """Write <out_dir>/prompt_round<N>.md, then (unless dry_run) <out_dir>/reply_round<N>.md and policy_round<N>.py plus
    compile_round<N>.json (usage, wall, audit). `caller(prompt) -> text` replaces the model in tests."""
    from compiled.api import audit_program
    os.makedirs(out_dir, exist_ok=True)
    prompt = build_prompt(read_text(API_DOC), read_text(EXAMPLE), episodes_text, previous_src, validation, notes)
    with open(os.path.join(out_dir, f"prompt_round{round_no}.md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    rec = {"round": round_no, "model": model, "prompt_chars": len(prompt), "dry_run": dry_run}
    if dry_run:
        with open(os.path.join(out_dir, f"compile_round{round_no}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        return rec
    if caller is not None:
        text, usage, wall = caller(prompt), {}, 0.0
    else:
        if not api_key_file:
            raise ValueError("--api_key_file is required unless --dry_run")
        text, usage, wall = call_model(prompt, api_key_file, model=model)
    with open(os.path.join(out_dir, f"reply_round{round_no}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    src = extract_module(text)
    policy_path = os.path.join(out_dir, f"policy_round{round_no}.py")
    with open(policy_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    problems = audit_program(policy_path)
    rec.update({"usage": usage, "wall_s": round(wall, 2), "policy": policy_path, "audit": problems, "accepted": not problems})
    with open(os.path.join(out_dir, f"compile_round{round_no}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="sweep folder of recorded episodes (one subfolder per kitchen)")
    ap.add_argument("--kitchens", nargs="*", default=None, help="which kitchens to show the compiler (default: all in the folder)")
    ap.add_argument("--max_decisions", type=int, default=20, help="decisions shown per episode")
    ap.add_argument("--with_reasoning", action="store_true", help="include the slow agent's chain of thought")
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--previous", default=None, help="policy_round<N-1>.py")
    ap.add_argument("--validation", default=None, help="validation.json of the previous round")
    ap.add_argument("--notes", default=None, help="a text file of notes for the compiler")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api_key_file", default="", help="path to CR's key file (never printed)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(HERE))
    from compiled.traces import load_sweep, render_episode
    from honest.honest_text import name_value_table
    eps = load_sweep(args.traces, kitchens=args.kitchens, name_values=name_value_table())
    if not eps:
        print("no episodes found", file=sys.stderr)
        return 2
    text = "\n\n".join(render_episode(e, max_decisions=args.max_decisions, with_reasoning=args.with_reasoning) for e in eps)
    rec = compile_round(args.out, args.round, text, api_key_file=args.api_key_file or None,
                        previous_src=read_text(args.previous) if args.previous else None,
                        validation=json.load(open(args.validation)) if args.validation else None,
                        notes=read_text(args.notes) if args.notes else None, model=args.model, dry_run=args.dry_run)
    print(json.dumps(rec, indent=1))
    return 0 if rec.get("accepted", True) else 1


if __name__ == "__main__":
    sys.exit(main())
