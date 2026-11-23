"""Programs (chunk 7c): Liane's contract on this task. A port of the loader in scripts/compiled/api.py.

A program is a Python file that defines

    class Policy:
        def __init__(self, rules): ...      # rules: the parts of the observation that never change in an episode
        def decide(self, obs): ...          # obs: firelab.interface.Session.observe(); returns a command, or None to stop

It plays alone: no model call, no file, no network, no clock, no randomness. The loader refuses a file that imports
anything outside ALLOWED_IMPORTS or has no Policy with __init__ and decide. `decide` may keep memory in `self`; a new
Policy is made for every episode. A command that is illegal, or a `decide` that raises, costs a tick and is recorded;
the episode goes on. A legal command that cannot be done (an item out of reach, a drop outside the strip) costs what
the skill says, often nothing; ten such in a row cost a tick. The time `decide` takes is measured for every decision.
"""

from __future__ import annotations

import ast
import importlib.util
import time
from pathlib import Path

from firelab.interface import Session

ALLOWED_IMPORTS = ("math", "collections", "heapq", "itertools", "functools", "typing", "dataclasses", "numpy")
RULE_KEYS = ("api_version", "costs", "end_share", "max_health", "total_value")
MAX_IDLE = 10


def audit_program(path) -> list:
    """Reasons the file is unfit to play; an empty list means fit."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as e:
        return [f"cannot parse: {e}"]
    problems = []
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        problems += [f"import of {n!r} is not allowed" for n in names if n.split(".")[0] not in ALLOWED_IMPORTS]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Policy"]
    if not classes:
        return problems + ["no `class Policy` at module level"]
    methods = {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    return problems + [f"Policy has no `{m}` method" for m in ("__init__", "decide") if m not in methods]


def load_program(path):
    """The program's module, imported from its file after audit_program passes."""
    problems = audit_program(path)
    if problems:
        raise ValueError("program refused: " + "; ".join(problems))
    spec = importlib.util.spec_from_file_location("firelab_program_" + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_program(session: Session, module, log: list | None = None) -> list:
    """Play one episode with a fresh Policy. One record per decision goes to `log`."""
    log = [] if log is None else log
    first = session.observe()
    policy = module.Policy({k: first[k] for k in RULE_KEYS})
    idle = 0  # decisions in a row that used no time
    while not session.skills.over:
        obs = session.observe()
        began, error = time.perf_counter(), ""
        try:
            command = policy.decide(obs)
        except Exception as e:  # a mistake in the program must not end the episode
            command, error = ("wait", 1), f"{type(e).__name__}: {e}"
        seconds = time.perf_counter() - began
        if command is None:
            break  # the program says it is done
        result = session.do(command)
        idle = idle + 1 if session.skills.ticks == obs["tick"] else 0
        if idle >= MAX_IDLE:  # a refused pick or an unreachable item is free, as for our reflex; an endless run of them is not
            session.skills.wait(1)
            idle = 0
        log.append({"tick": obs["tick"], "command": list(command) if isinstance(command, (list, tuple)) else str(command),
                    "seconds": round(seconds, 5), "ticks": result.ticks, "done": result.done, "reason": result.reason, "error": error})
    return log
