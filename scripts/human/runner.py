"""Human baseline for HAZARD, file 3 of 4: what run_baseline.py needs for --agent human (ASSUMPTIONS.md A36).

run_baseline.py builds the agent with make_human_agent(), hands it to TimedAgent + HAZARD's submit() exactly like the
other agents (clock off unless --clock is given), and calls finish_human_run() once submit() returns. Everything here
is HAZARD-free and testable with a fake challenge object: the only HAZARD facts used are the Challenge attributes the
runner's hook already exposes (run_baseline.HOLDER["challenge"]):
    challenge.target_status   {target_id: False | frame it was dropped in the bag}   challenge.py:253, 180
    challenge.data_dir        the scene folder                                       run_experiments.py:193
"""
import os
import time

from human.page import PageChooser


class LiveCounters:
    """What the page's right panel shows beside the agent's own counters: targets saved so far and the kitchen."""

    def __init__(self, get_challenge):
        self.get_challenge = get_challenge  # callable returning the Challenge instance, or None before the first episode

    def __call__(self):
        ch = self.get_challenge()
        out = {}
        if ch is None:
            return out
        status = getattr(ch, "target_status", None)
        if isinstance(status, dict):
            out["saved"] = f"{sum(1 for v in status.values() if v)} of {len(status)} targets"
        data_dir = getattr(ch, "data_dir", None)
        if data_dir:
            out["kitchen"] = os.path.basename(os.path.normpath(str(data_dir)))
        return out


class CountedChooser:
    """The page chooser with the live counters pushed right before each question, so the panel is current when the
    person reads it. Called in HAZARD's thread by HumanAgent._ask_person; blocks like PageChooser does."""

    def __init__(self, page, counters):
        self.page = page
        self.counters = counters

    def __call__(self, decision):
        self.page.set_info(**self.counters())
        return self.page(decision)


def make_human_agent(task, prompt_path, get_challenge, host="127.0.0.1", port=8765, camera=True, open_browser=True):
    """HumanAgent wired to a PageChooser. Imported lazily: human_agent pulls in HAZARD's policy.llm (~40 s cold)."""
    from human.human_agent import HumanAgent
    page = PageChooser(host=host, port=port, open_browser=open_browser)
    agent = HumanAgent(task=task, prompt_template_path=prompt_path, chooser=CountedChooser(page, LiveCounters(get_challenge)),
                       camera=camera)
    return agent, page


def score_from_evals(evals):
    """Mean value rate over HAZARD's eval_result.json files (avg_score / avg_max_score each), None if there are none."""
    rates = []
    for v in evals.values():
        if isinstance(v, dict) and v.get("avg_max_score"):
            rates.append(v["avg_score"] / v["avg_max_score"])
    return round(sum(rates) / len(rates), 4) if rates else None


def finish_human_run(page, agent, evals, counters=None, hold_s=20.0):
    """Show the result on the page, hold it there for hold_s seconds (the process ends right after), then close the
    server. Returns the block run_baseline.py puts under summary["human"]."""
    score = score_from_evals(evals)
    extra = counters() if counters else {}
    page.finish(score=score, **extra)
    out = {"url": page.url, "score": score, "camera": bool(getattr(agent, "camera", False)),
           "clicks": [{"seq": s, "letter": l} for s, l in page.choices]}
    out.update(agent.summary() if hasattr(agent, "summary") else {})
    if hold_s > 0:
        print(f"human page: score {score} shown at {page.url} for {hold_s:g} s")
        time.sleep(hold_s)
    page.close()
    return out
