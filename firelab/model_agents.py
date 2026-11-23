"""The agents that use a language model (chunk 8b; spec in PLAN.md sections 7 and 8).

  model_every_command   the model is asked before every command and the robot stands while it answers. The
                        blocking baseline: the kind of agent RRARA's first table shows losing to the clock.
  rrara_v1              RRARA version 1 on this task. Our rule (firelab.agents.reflex_choice) picks the command
                        and it starts at once; the model is asked, in the background, whether that command is
                        valid, and if not, for a replacement. At the next decision an invalid answer's replacement
                        is taken in place of the rule's choice if it is still on the menu; otherwise the rule
                        stands. The command under way is never interrupted (A89). The rule's own decision costs
                        0 ticks; a refusal, cut-off or error is no answer and the rule stands (CR, Nov 24).
Two toggles, both written into every record:
  accounting   "free": the checker's time is not counted and a finished answer always counts (the paper).
               "strict": the answer exists only `charged` ticks after the check began and counts only if that
               is no later than the next decision (scripts/rrara "worldtime").
  call_time    "measured": the real seconds of the call. "declared": a fixed number of seconds for every call.
"In the background" is done in order: the model's input is fixed when the check begins and the fire does not
depend on the model, so making the call first and working out when its answer exists gives the same episode.
"""

from __future__ import annotations

from firelab.agents import reflex_choice
from firelab.clock import ticks
from firelab.interface import Session
from firelab.text import describe, say

ASK = "Choose the one option that will bank the most value before the episode ends."
CHECK = ("A fast rule has already started a command. Judge it: answer A if it is the best option now; otherwise choose "
         "the option the robot should take next instead.")


def call_cost(answer, call_time: str, declared_seconds: float) -> dict:
    """Both times of one call and the ticks charged for it under the toggle."""
    both = {"measured": ticks(answer.seconds), "declared": ticks(declared_seconds)}
    return {"measured_s": round(answer.seconds, 4), "declared_s": declared_seconds, "ticks_measured": both["measured"],
            "ticks_declared": both["declared"], "charged": both[call_time]}


def _record(obs, text, options, answer, cost, **more) -> dict:
    return {"tick": obs["tick"], "text": text, "options": options, "raw": answer.raw, "choice": answer.choice,
            "stop_reason": answer.stop_reason, "error": answer.error, "tokens_in": answer.tokens_in,
            "tokens_out": answer.tokens_out, **cost, "steps": 0, "estimate": cost["charged"], **more}


def _entry(command, session: Session, result) -> tuple:
    """A command in the log's shape: (name, item or None, tick it ended, Result)."""
    item = command[1] if len(command) > 1 and isinstance(command[1], int) and not isinstance(command[1], bool) else None
    return (command[0], item, session.skills.ticks, result)


def model_every_command(session: Session, model, call_time: str = "measured", declared_seconds: float = 0.3,
                        records: list | None = None, log: list | None = None, **_) -> list:
    """Ask, stand for the time the answer took, do what was answered; repeat. No answer: wait 10 ticks."""
    records, log = ([] if records is None else records), ([] if log is None else log)
    skills = session.skills
    while not skills.over:
        obs = session.observe()
        menu = session.legal(obs)
        options = [say(c, obs) for c in menu]
        text = describe(obs, options, session.history)
        answer = model.ask(text, options, ASK)
        cost = call_cost(answer, call_time, declared_seconds)
        command = menu[answer.choice] if answer.choice is not None else ("wait", 10)
        records.append(_record(obs, text, options, answer, cost, command=list(command)))
        skills.wait(cost["charged"])  # the robot stands while the model answers
        if skills.over:
            break
        log.append(_entry(command, session, session.do(command)))
        if skills.ticks == obs["tick"]:
            skills.wait(1)  # an answer that changed nothing still costs a tick
    return log


def rrara_v1(session: Session, model, accounting: str = "free", call_time: str = "measured", declared_seconds: float = 0.3,
             records: list | None = None, log: list | None = None, **_) -> list:
    """The rule acts; the model checks each command in the background. See the top of this file."""
    records, log = ([] if records is None else records), ([] if log is None else log)
    skills, skip, pending, idle = session.skills, set(), None, 0
    while not skills.over:
        obs = session.observe()
        now, menu = obs["tick"], session.legal(obs)
        rule = reflex_choice(skills, skip)
        action, status = rule, "none"
        if pending is not None:  # the answer about the previous command
            if pending["choice"] is None:
                status = "no answer"
            elif accounting == "strict" and pending["ready_tick"] > now:
                status = "late"
            elif pending["choice"] == 0:
                status = "valid"
            elif pending["replacement"] in menu:
                action, status = pending["replacement"], "override"
            else:
                status = "replacement not on the menu now"
        if action is None:
            break  # the rule has nothing left to do and nothing replaced it
        others = [c for c in menu if c != action]
        options = [f"VALID: keep the rule's command, {say(action, obs)}"] + [say(c, obs) for c in others]
        text = describe(obs, options, session.history, note=f"A fast rule has ALREADY STARTED this command: {say(action, obs)}.")
        answer = model.ask(text, options, CHECK)
        cost = call_cost(answer, call_time, declared_seconds)
        pending = {"choice": answer.choice, "ready_tick": now + cost["charged"],
                   "replacement": others[answer.choice - 1] if answer.choice else None}
        records.append(_record(obs, text, options, answer, cost, rule=list(rule or []), previous_check=status,
                               command=list(action), accounting=accounting, call_time=call_time, ready_tick=pending["ready_tick"]))
        result = session.do(action)  # the command starts at once: the rule's decision costs 0 ticks
        log.append(_entry(action, session, result))
        if action[0] == "get" and not result.done and not skills.over:
            skip.add(action[1])  # as the reflex: an item it could not reach is not tried again
        if action[0] == "home" and not result.done:
            break  # as the reflex: trapped with the load, dead, or out of time
        idle = idle + 1 if skills.ticks == now else 0
        if idle >= 10:
            skills.wait(1)
            idle = 0
    return log


AGENTS = {"model": model_every_command, "rrara1": rrara_v1}
