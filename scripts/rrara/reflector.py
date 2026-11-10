"""RRARA-v1 Async-Reflector (Zheng et al. 2025, sec. 3.4, eq. 8-9): given the reflex's action and the latest observation,
answer valid / invalid, with a replacement action only when invalid.

Framework rule F3 (RRARA_V1_PLAN.md): this is HAZARD's own LLM agent class with a different prompt, so the scene text,
distances, temperatures, available-action list, answer parser, Anthropic adapter (A18) and call log are the ones the LLM
baseline uses. The only differences from the baseline are the template (reflector_prompt.csv, which adds $PROPOSED$ and
asks for VALID or a letter) and that this class never chooses on its own: validate() is called by the agent, in a thread.
This file knows nothing about threads or the simulator clock.
"""
import os
import time

from policy.llm import LLM, SamplingParameters  # HAZARD module (src/HAZARD on PYTHONPATH, as run_experiments.py:2 does)

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reflector_prompt.csv")


def _same_action(a, b):
    """('walk_to', '64') and ('walk_to', 64) are the same action; HAZARD mixes str and int ids."""
    return a[0] == b[0] and str(a[1]) == str(b[1])


class Reflector(LLM):
    def __init__(self, task, lm_id="claude-haiku-5-5", source="anthropic", api_key_file="", max_tokens=64):
        # key list read from a file, as run_experiments.py:30-33 does; the key is never printed or stored elsewhere
        keys = [l.strip() for l in open(api_key_file) if l.strip()] if api_key_file else ""
        super().__init__(source=source, lm_id=lm_id, prompt_template_path=TEMPLATE, cot=False,
                         sampling_parameters=SamplingParameters(max_tokens=max_tokens), task=task, api_key=keys)
        self.agent_type = "reflector"  # never handed to HAZARD; the RRARA agent owns this object
        self.verdicts = []  # one record per validate() call, for the summary

    # --- what the LLM baseline does at the top of choose_target (llm.py:543-560), without choosing ---
    def describe(self, state, processed_input):
        agent_map = state["goal_map"]
        pts = (agent_map == -2).nonzero()
        self.agent_pos = (pts[0].astype(float).mean(), pts[1].astype(float).mean())
        self.current_step = processed_input["step"]
        self.holding_objects = processed_input["holding_objects"]
        self.object_list = processed_input["explored_object_name_list"]
        self.nearest_object = processed_input["nearest_object"]
        self.current_state = state
        self.current_seen_objects_id = [str(x) for x in set(state["raw"]["seg_mask"].flatten())]
        self.update_object_status()
        progress = self.progress2text()
        plans_text, n, plan_list, actions = self.get_available_plans()
        return progress, plans_text, plan_list, actions

    def note_executed(self, action, result, info):
        """The agent tells the reflector what the reflex did and how it went, so $ACTION_HISTORY$ matches the baseline's."""
        self.update_history(self._action_text(action, [], []))
        self.update_history_action_result(result, info)

    def _action_text(self, action, plan_list, actions):
        for text, a in zip(plan_list, actions):
            if _same_action(a, action):
                return text
        name, param = action
        return {"pick_up": f"pick up the object I just walked to ({param})", "drop": "put the holding object in my bag",
                "explore": "look around"}.get(name, f"go to object ({param})")

    # --- the validation function V of eq. (8) ---
    def validate(self, state, processed_input, proposed):
        progress, plans_text, plan_list, actions = self.describe(state, processed_input)
        history = [f"{a} ({self.action_result_to_description(r, i)})" for a, r, i in
                   zip(self.action_history, self.action_history_result, self.action_info_history)]
        prompt = (self.prompt_template.replace("$TARGET_OBJECTS$", self.objects_list2text() + "\n")
                  .replace("$STATE$", progress)
                  .replace("$ACTION_HISTORY$", ", ".join(history[-10:]) + "\n")
                  .replace("$PROPOSED$", self._action_text(proposed, plan_list, actions))
                  .replace("$AVAILABLE_ACTIONS$", plans_text))
        t0 = time.perf_counter()
        outputs, usage = self.generator([{"role": "user", "content": prompt}], self.sampling_params)
        wall = time.perf_counter() - t0
        self.total_cost += usage
        text = outputs[0].strip()
        valid = "VALID" in text.upper() and "INVALID" not in text.upper()
        replacement = None
        if not valid:
            plan = self.parse_answer(plan_list, text)
            replacement = actions[plan_list.index(plan)]
            if _same_action(replacement, proposed):  # it named the started action: that is a VALID
                valid, replacement = True, None
        rec = {"step": self.current_step, "proposed": [proposed[0], str(proposed[1])], "valid": valid,
               "replacement": None if replacement is None else [replacement[0], str(replacement[1])],
               "raw": text[:200], "wall_s": round(wall, 4), "n_options": len(actions)}
        self.verdicts.append(rec)
        return rec
