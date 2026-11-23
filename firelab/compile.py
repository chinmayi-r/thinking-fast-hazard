"""The writing step of Liane's method on the fire task (A99; a port of scripts/compiled/compile.py).

Given the program API (firelab/PROGRAM_API.md), the worked example (firelab/examples/reflex_program.py), the saved
games of a slow model (run_rooms.py, agent `model`) and, from round 2 on, the previous round's program with its score
in every validation room, a strong model is asked for a new program. The reply's program is written to
policy_round<N>.py and must pass firelab.programs.audit_program before it may play. Offline: no simulator here;
firelab/validate.py scores the file afterwards. `--dry-run` writes the prompt and stops, with no model call.

Run:  python -m firelab.compile --games results/firelab_sim/haiku_train/games --out results/firelab_compiled/v1 \\
          --round 1 --key-file <path> [--dry-run]
      ... --round 2 --previous results/firelab_compiled/v1/policy_round1.py --validation results/firelab_compiled/v1/validation_round1.json
The key is given as a file path only; it is read once, handed to the client, and never printed or written anywhere.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from firelab.models import cost_usd, read_key
from firelab.programs import audit_program

HERE = Path(__file__).parent
API_DOC, EXAMPLE = HERE / "PROGRAM_API.md", HERE / "examples" / "reflex_program.py"
WRITER, EFFORT, MAX_TOKENS = "claude-opus-5-5", "medium", 16000  # medium and 16 000 keep three rounds under the 2-dollar limit (A99)
SYSTEM = ("You write deterministic Python programs that control a rescue robot in a simulated room with a spreading fire. "
          "You get the API document, a working example program, recorded games of a slow language-model agent (what it saw, "
          "what it chose, what happened), and sometimes the previous program with its scores. Return ONE complete Python "
          "module in a single ```python block, defining `class Policy` exactly as the API requires, importing only what the "
          "API allows, with no model calls and no randomness. Before the code, write at most ten lines on what the recorded "
          "games show and what the program does differently. Prefer simple rules that cannot crash: every branch must "
          "return a legal command or None.")
KEEP = ("Time:", "You:", "Bag:", "Items on the floor:", "  item ", "Safe strip")  # the state lines shown per decision


def render_game(path, max_decisions: int = 25) -> str:
    """One saved game as text for the writer: per decision the state the slow agent saw, its choice, and the outcome."""
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    head, end = records[0], records[-1]
    calls = [r for r in records if r["type"] == "thought"]
    commands = [r for r in records if r["type"] == "command"]
    out = [f"## Room {head['room']}: final score {end['score']:.3f}, ended at tick {end['ticks']} with health {end['health']:.0f}; "
           f"{len(calls)} decisions"]
    for n, call in enumerate(calls[:max_decisions]):
        state = [line for line in call["text"].split("\n") if line.startswith(KEEP)]
        chose = call["options"][call["choice"]] if call["choice"] is not None else "(no answer)"
        took = f", after {call['measured_s']:.1f} s of thinking ({call['charged']} ticks standing)"
        result = commands[n] if n < len(commands) else None
        outcome = "(the episode ended first)" if result is None else f"{'ok' if result['done'] else result['reason']}, {result['ticks']} ticks"
        out.append(f"Decision {n + 1}\n" + "\n".join(state) + f"\nChose: {chose}{took}\nResult: {outcome}")
    if len(calls) > max_decisions:
        out.append(f"({len(calls) - max_decisions} later decisions not shown)")
    return "\n\n".join(out)


def build_prompt(api_doc: str, example_src: str, games_text: str, previous_src: str | None = None, validation: dict | None = None,
                 full_text: str | None = None) -> str:
    parts = ["# The API a program plays through\n\n" + api_doc,
             "# A worked example that loads and plays (the reflex)\n\n```python\n" + example_src.rstrip() + "\n```"]
    if full_text:
        parts.append("# One complete text as the slow agent saw it (for the layout of the room)\n\n" + full_text)
    parts.append("# Recorded games of the slow agent on training rooms (what it saw, what it chose, what happened)\n\n" + games_text)
    if previous_src:
        parts.append("# The previous round's program\n\n```python\n" + previous_src.rstrip() + "\n```")
    if validation:
        parts.append("# How the previous program scored in each validation room (the reflex's score in the same room beside it)\n\n"
                     + json.dumps(validation, indent=1))
    parts.append("# Task\n\nWrite the next program. Keep what worked and fix what the recorded outcomes show going wrong "
                 "(value lost to fire, a bag lost because the way home closed or the episode ended, health lost, time spent "
                 "standing or walking for little). Return the whole module in one ```python block.")
    return "\n\n".join(parts)


def extract_module(text: str) -> str:
    """The last ```python ... ``` block of a reply, or the whole reply if there is none."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip() + "\n"


def call_writer(prompt: str, key_file, model: str = WRITER) -> tuple:
    """One streamed call through the official SDK. Returns (text or None, usage dict, seconds).

    Thinking cannot be disabled on claude-opus-5-5, so only the effort is set. The stop reason is checked before
    the text is read: anything but a normal end of turn gives no text.
    """
    import anthropic  # here, so the dry run and the tests need no SDK

    client = anthropic.Anthropic(api_key=read_key(key_file))
    began = time.perf_counter()
    with client.messages.stream(model=model, max_tokens=MAX_TOKENS, system=SYSTEM, output_config={"effort": EFFORT},
                                messages=[{"role": "user", "content": prompt}]) as stream:
        message = stream.get_final_message()
    seconds = time.perf_counter() - began
    usage = {"input_tokens": int(message.usage.input_tokens), "output_tokens": int(message.usage.output_tokens),
             "model": model, "stop_reason": str(message.stop_reason)}
    usage["dollars"] = round(cost_usd(model, usage["input_tokens"], usage["output_tokens"]), 4)
    if message.stop_reason != "end_turn":
        return None, usage, seconds
    return "".join(block.text for block in message.content if block.type == "text"), usage, seconds


def compile_round(out_dir, round_no: int, games_text: str, key_file=None, previous_src: str | None = None,
                  validation: dict | None = None, full_text: str | None = None, model: str = WRITER, dry_run: bool = False,
                  caller=None) -> dict:
    """Write prompt_round<N>.md, then (unless dry_run) reply_round<N>.md, policy_round<N>.py and compile_round<N>.json.

    `caller(prompt) -> text` stands in for the model in tests.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(API_DOC.read_text(encoding="utf-8"), EXAMPLE.read_text(encoding="utf-8"), games_text, previous_src, validation, full_text)
    (out_dir / f"prompt_round{round_no}.md").write_text(prompt, encoding="utf-8")
    record = {"round": round_no, "model": model, "prompt_chars": len(prompt), "dry_run": dry_run}
    if not dry_run:
        text, usage, seconds = (caller(prompt), {}, 0.0) if caller is not None else call_writer(prompt, key_file, model)
        record.update(usage=usage, seconds=round(seconds, 2), accepted=False)
        if text is None:
            record["audit"] = ["no program: the reply did not end normally"]
        else:
            (out_dir / f"reply_round{round_no}.md").write_text(text, encoding="utf-8")
            policy = out_dir / f"policy_round{round_no}.py"
            policy.write_text(extract_module(text), encoding="utf-8", newline="\n")
            record.update(policy=str(policy), audit=audit_program(policy))
            record["accepted"] = not record["audit"]
    (out_dir / f"compile_round{round_no}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="folder of saved games (room_NNN_model.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--max-decisions", type=int, default=25, help="decisions shown per game")
    ap.add_argument("--previous", default=None, help="policy_round<N-1>.py")
    ap.add_argument("--validation", default=None, help="the validation file of the previous round")
    ap.add_argument("--model", default=WRITER)
    ap.add_argument("--key-file", default=None, help="path of the key file (never printed)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    games = sorted(Path(args.games).glob("room_*_model.jsonl"))
    if not games:
        print("no saved games found in", args.games, file=sys.stderr)
        return 2
    first_call = next(json.loads(line) for line in games[0].read_text(encoding="utf-8").splitlines() if '"type": "thought"' in line)
    record = compile_round(
        args.out, args.round, "\n\n".join(render_game(g, args.max_decisions) for g in games), key_file=args.key_file,
        previous_src=Path(args.previous).read_text(encoding="utf-8") if args.previous else None,
        validation=json.loads(Path(args.validation).read_text(encoding="utf-8")) if args.validation else None,
        full_text=first_call["text"], model=args.model, dry_run=args.dry_run)
    print(json.dumps(record, indent=1))
    return 0 if args.dry_run or record.get("accepted") else 1


if __name__ == "__main__":
    sys.exit(main())
