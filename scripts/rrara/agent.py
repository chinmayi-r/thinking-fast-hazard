"""RRARA-v1 agent for HAZARD (Zheng et al. 2025, arXiv 2506.07223, sec. 3.2 and 3.4).

HAZARD's agent contract (documentation/agents/custom_agent.md:5-41, template policy/custom.py:11-20): an object with a
label `agent_type`, `reset(goal_objects, objects_info)` once per episode, and `choose_target(state, processed_input)`
once per decision returning (action_name, param). challenge.py:257-258 accepts the label "custom".

Each decision (RRARA_V1_PLAN.md, framework rules F1-F6):
  1. tell the reflector how the previous action went (so its history text matches the LLM baseline's);
  2. collect the background verdict, if one finished; it counts only if it is about the previous decision (A31);
  3. ask the reflex (HAZARD's RuleBasedAgent, untouched) for this turn's action;
  4. let the referee (reconcile.decide) pick reflex or replacement under the chosen mode (A30);
  5. start the reflector on the chosen action in the background, unless one check is still running;
  6. log one record and return. Steps 1-6 take well under a millisecond, so the clock charges one frame (paper: RL 0.00 s).
Modes: "off" = reflex only (the wiring check, must match the rule baseline); "faithful" = the paper's accounting;
"worldtime" = a verdict counts only if ceil(wall x 30) <= frames the checked action took.
"""
from policy.rule_based import RuleBasedAgent  # HAZARD's rule agent, untouched: the reflex is the rule baseline's brain (F2)

from rrara.background import BackgroundJob
from rrara.reconcile import decide, legal_actions


class RRARAv1Agent:
    def __init__(self, task, mode="faithful", lm_id="claude-haiku-5-5", api_key_file="", fps=30, reflex="paper"):
        assert mode in ("off", "faithful", "worldtime")
        assert reflex in ("paper", "hazard_rule")
        self.task = task
        self.mode = mode
        self.fps = fps
        self.agent_type = "custom"
        self.reflex_kind = reflex
        if reflex == "paper":  # A34: the paper's eq. (2) policy, reimplemented (scripts/rrara/reflex.py)
            from rrara.reflex import PaperReflex
            self.reflex = PaperReflex(task=task)
        else:  # HAZARD's shipped rule agent, untouched (the Nov 9 skeleton runs)
            self.reflex = RuleBasedAgent(task=task)
        self.reflector = None
        self.job = None
        if mode != "off":
            from rrara.reflector import Reflector  # imported here so mode "off" needs no API key or anthropic package
            self.reflector = Reflector(task=task, lm_id=lm_id, api_key_file=api_key_file)
            self.job = BackgroundJob()
        self.decision_log = []
        self._episode_start()

    def _episode_start(self):
        self.n = 0
        self.last_action = None
        self.last_frame = None
        if self.job is not None:
            self.job.discard()

    def reset(self, goal_objects, objects_info):
        self.goal_objects = goal_objects
        self.objects_info = objects_info
        self.reflex.reset(goal_objects, objects_info)
        if self.reflector is not None:
            self.reflector.reset(goal_objects, objects_info)
        self._episode_start()

    def choose_target(self, state, processed_input):
        frame = processed_input["step"]
        rec = {"decision": self.n, "frame": frame, "mode": self.mode, "verdict_status": "none"}

        # 1. how the previous action went
        if self.last_action is not None:
            if self.reflector is not None:
                self.reflector.note_executed(self.last_action, processed_input["action_result"], processed_input["action_info"])
            if hasattr(self.reflex, "note_result"):
                self.reflex.note_result(self.last_action, processed_input["action_result"])

        # 2. a finished background verdict, usable only if it checked the previous decision (A31)
        verdict = None
        if self.job is not None:
            got = self.job.collect()
            if got is not None:
                result, wall, tag = got
                if isinstance(result, dict) and "error" not in result:
                    result = dict(result, wall_s=wall)  # the time the agent waited, thread included
                if tag == self.n - 1:
                    verdict, rec["verdict_status"] = result, "used"
                else:
                    rec["verdict_status"] = f"stale (about decision {tag})"
            elif self.job.pending():
                rec["verdict_status"] = "still running"

        # 3. reflex
        proposed = self.reflex.choose_target(state, processed_input)

        # 4. referee
        action_frames = 0 if self.last_frame is None else frame - self.last_frame
        legal = legal_actions(processed_input, self.goal_objects)
        action, why = decide(proposed, verdict, legal, mode=self.mode, action_frames=action_frames, fps=self.fps)
        if why["override"]:
            self.reflex.last_action = action  # keep the rule's "persist last action" logic coherent with what runs
        rec.update(proposed=[proposed[0], str(proposed[1])], action=[action[0], str(action[1])], **why)
        if hasattr(self.reflex, "last_costs"):
            rec["reflex_costs"] = self.reflex.last_costs
        if verdict is not None:
            rec["verdict"] = {k: verdict.get(k) for k in ("valid", "replacement", "wall_s", "raw")}

        # 5. start the next background check on the action that will execute
        if self.job is not None:
            if self.job.pending():
                rec["check"] = "skipped: reflector still busy"
            else:
                self.job.start(self.reflector.validate, state, processed_input, action, tag=self.n)
                rec["check"] = "started"

        # 6. log and return
        self.decision_log.append(rec)
        self.last_action, self.last_frame = action, frame
        self.n += 1
        return action

    def rrara_summary(self):
        paths = {}
        for r in self.decision_log:
            paths[r["path"]] = paths.get(r["path"], 0) + 1
        out = {"mode": self.mode, "reflex": self.reflex_kind, "decisions": len(self.decision_log),
               "overrides": sum(1 for r in self.decision_log if r["override"]),
               "verdicts_used": sum(1 for r in self.decision_log if r["verdict_status"] == "used"),
               "verdicts_stale": sum(1 for r in self.decision_log if r["verdict_status"].startswith("stale")),
               "checks_skipped_busy": sum(1 for r in self.decision_log if r.get("check", "").startswith("skipped")),
               "paths": paths}
        if self.reflector is not None:
            w = [v["wall_s"] for v in self.reflector.verdicts]
            out["reflector"] = {"model": self.reflector.lm_id, "calls": len(w),
                                "wall_s": {"mean": round(sum(w) / len(w), 3), "max": round(max(w), 3)} if w else None,
                                "invalid": sum(1 for v in self.reflector.verdicts if not v["valid"]),
                                "cost_usd_estimate": round(self.reflector.total_cost, 5)}
        return out
