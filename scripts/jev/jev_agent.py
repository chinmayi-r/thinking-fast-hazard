"""HAZARD's LLM agent with Jev as the generator (ASSUMPTIONS.md A37). Same pattern as the human baseline
(scripts/human/human_agent.py) and the VLM agent: a subclass of policy.llm.LLM whose generator is not an API chat
call. LLM.run (llm.py:589-655) builds the prompt and the menu and parses the answer exactly as for Haiku; the only
difference is cot=False (one call, one letter) and that Jev cannot produce free text.

PYTHONPATH must hold HAZARD/src/HAZARD (as run_experiments.py:2 does) so `policy.llm` imports.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ for `jev.client`
from policy.llm import LLM, SamplingParameters  # HAZARD module
from jev.client import JevClient, DEFAULT_MODEL


class JevAgent(LLM):
    def __init__(self, task, prompt_template_path, api_key_file="", model=DEFAULT_MODEL, client=None, examples_path=None):
        # source "anthropic" only selects the chat-style prompt (llm.py:77); the Anthropic client is dropped below,
        # as HumanAgent does (human_agent.py:61-65). No Anthropic key is needed or used.
        super().__init__(source="anthropic", lm_id=model, prompt_template_path=prompt_template_path, cot=False,
                         sampling_parameters=SamplingParameters(max_tokens=8), task=task, api_key="")
        self.anthropic_client = None
        self.client = client or JevClient(api_key_file=api_key_file, model=model)
        self.call_log = self.client.call_log  # run_baseline.py reads inner.call_log (A18 format)
        self.examples = None
        if examples_path:
            with open(examples_path, encoding="utf-8") as f:
                self.examples = f.read()
        self.decisions = []  # one per asked decision: letter, confidence, probabilities, wall_s, menu
        self.generator = self._decide  # LLM.run calls self.generator(messages, sampling_params) (llm.py:640)

    def reset(self, goal_objects, objects_info):
        super().reset(goal_objects, objects_info)
        self.decisions = []

    def _decide(self, messages, sampling_params):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        # the same menu run() just built from the same state (get_available_plans is pure; llm.py:487-520)
        _, _, plan_list, actions = self.get_available_plans()
        menu = [(chr(ord("A") + i), plan) for i, plan in enumerate(plan_list)]
        res = self.client.decide(prompt, menu=menu, examples=self.examples)
        self.decisions.append({"n": len(self.decisions) + 1, "letter": res["letter"], "plan": dict(menu)[res["letter"]],
                               "confidence": res["confidence"], "probabilities": res["probabilities"],
                               "wall_s": res["wall_s"], "model": res["model"], "menu": [t for _, t in menu],
                               "prompt": prompt,  # the state text Jev saw (HAZARD's or the honest text); saved as jev_decisions.json
                               "t_unix": round(time.time(), 3)})
        return [res["letter"]], res["cost_usd"]  # parse_answer's one-letter branch (llm.py:342-380); usage in USD

    def summary(self):
        conf = [d["confidence"] for d in self.decisions if d["confidence"] is not None]
        walls = [d["wall_s"] for d in self.decisions]
        return {"decisions": len(self.decisions),
                "confidence_mean": round(sum(conf) / len(conf), 3) if conf else None,
                "wall_s_mean": round(sum(walls) / len(walls), 4) if walls else None,
                "wall_s_max": round(max(walls), 4) if walls else None,
                "models": sorted(set(str(d["model"]) for d in self.decisions)),
                "cost_usd": round(self.client.total_cost, 8)}
