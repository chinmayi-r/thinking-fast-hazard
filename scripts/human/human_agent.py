"""Human baseline for HAZARD, file 1 of 4: the agent half (ASSUMPTIONS.md A36).

A person plays HAZARD through the code path the Haiku text agent uses. HumanAgent is HAZARD's own LLM agent class
(policy/llm.py) with the language model replaced by a person: LLM.choose_target and LLM.run are untouched, so the
scene text, the action history wording, the lettered menu and the answer parser are the ones Haiku gets. The only
new code is the "generator": instead of an API call it writes the decision (prompt text, menu, optional camera frame,
counters) to the episode folder, hands it to a chooser callable, and returns the letter the person picked.

What this file does NOT do (later files): no web page (file 2), no --agent human branch in run_baseline.py (file 3),
no play protocol (file 4). The default chooser reads from stdin so the agent can be exercised on its own.

Per decision it writes <save_dir>/<step>_human.json (the decision record), <save_dir>/<step>_human_frame.png when
camera=True, and rewrites <save_dir>/human_log.json (all records of the episode so far). HAZARD itself writes
<step>_info.json beside them (llm.py:584-586).
"""
import json
import os
import time

import numpy as np

from policy.llm import LLM, SamplingParameters  # HAZARD module (src/HAZARD on PYTHONPATH, as run_experiments.py:2 does)


def stdin_chooser(decision):
    """File-1 chooser: print what the person would see on the page and read one line from stdin."""
    print(f"\n===== decision {decision['n']} | frame {decision['frame']} of {decision['frames_total']} =====")
    print(decision["prompt"])  # ends with the lettered menu, as it does for Haiku
    if decision["rejected"]:
        print(f"  (not an option: {decision['rejected'][-1]!r}; type one of {decision['letters']})")
    return input("Your choice (letter): ")


def _find_rgb(state):
    """Same lookup as scripts/vlm_agent.py:_find_rgb."""
    raw = state.get("raw", {}) if isinstance(state, dict) else {}
    for key in ("rgb", "img", "image"):
        if key in raw:
            return raw[key]
    return None


def _to_png(rgb, path, image_side=512):
    """Same conversion as scripts/vlm_agent.py:_encode: HAZARD gives (3, H, W) floats in [0, 1]."""
    from PIL import Image
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = arr.transpose(1, 2, 0)[:, :, :3]
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if image_side and max(im.size) > image_side:
        im = im.resize((image_side, image_side))
    im.save(path, format="PNG")


class HumanAgent(LLM):
    FRAMES_TOTAL = 1500  # HAZARD's episode length (challenge.py:53-56, 289); shown to the person as "frame x of 1500"

    def __init__(self, task, prompt_template_path, chooser=None, camera=True, image_side=512):
        # source "anthropic" only picks the chat-style prompt and sampling dict (llm.py:77, 86-98); with an empty key
        # the client object is built but never used, and we drop it below so no call can ever go out.
        super().__init__(source="anthropic", lm_id="human", prompt_template_path=prompt_template_path, cot=False,
                         sampling_parameters=SamplingParameters(max_tokens=8), task=task, api_key="")
        self.anthropic_client = None
        self.generator = self._ask_person  # LLM.run calls self.generator(messages, sampling_params) (llm.py:640)
        self.chooser = chooser or stdin_chooser
        self.camera = camera
        self.image_side = image_side
        self.records = []  # one per decision of the current episode, automatic ones included
        self.n = 0  # decisions this episode
        self._pending = None  # what choose_target knows and run() does not (frame, save_dir, png path)

    # --- episode boundary (challenge.py calls reset per episode) ---
    def reset(self, goal_objects, objects_info):
        super().reset(goal_objects, objects_info)
        self.records = []
        self.n = 0

    # --- HAZARD's entry point; the parent does everything, we only note what it does not tell run() ---
    def choose_target(self, state, processed_input):
        self.n += 1
        frame = int(processed_input["step"])  # challenge.py:161: the simulator frame count
        save_dir = processed_input["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        png = None
        if self.camera:
            rgb = _find_rgb(state)
            if rgb is not None:
                png = os.path.join(save_dir, f"{frame:04d}_human_frame.png")
                try:
                    _to_png(rgb, png, self.image_side)
                except Exception as e:  # the picture must never stop the game
                    print(f"HumanAgent: could not save frame: {e}")
                    png = None
        self._pending = {"n": self.n, "frame": frame, "save_dir": save_dir, "frame_png": png,
                         "holding": [o.get("category", "?") for o in processed_input["holding_objects"]],
                         "asked": False}
        t0 = time.perf_counter()
        action = super().choose_target(state, processed_input)  # may return pick_up / drop without asking (llm.py:560-567)
        if not self._pending["asked"]:
            rec = {"n": self.n, "frame": frame, "automatic": True, "action": [action[0], str(action[1])],
                   "wall_s": round(time.perf_counter() - t0, 4), "frame_png": png, "holding": self._pending["holding"]}
            self._write(rec)
        self._pending = None
        return action

    # --- the "generator": a person instead of a model ---
    def _ask_person(self, messages, sampling_params):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        # the same menu run() just built from the same state (get_available_plans is pure; llm.py:487-520)
        _, _, plan_list, actions = self.get_available_plans()
        menu = [{"letter": chr(ord("A") + i), "text": plan, "action": [a[0], str(a[1])]}
                for i, (plan, a) in enumerate(zip(plan_list, actions))]
        p = self._pending or {"n": self.n, "frame": -1, "save_dir": self.save_dir, "frame_png": None, "holding": []}
        p["asked"] = True
        decision = {"n": p["n"], "frame": p["frame"], "frames_total": self.FRAMES_TOTAL, "automatic": False,
                    "prompt": prompt, "menu": menu, "letters": [m["letter"] for m in menu],
                    "frame_png": p["frame_png"], "holding": p["holding"], "rejected": []}
        t0 = time.perf_counter()
        while True:
            raw = self.chooser(decision)
            letter = (raw or "").strip().upper()
            if letter in decision["letters"]:
                break
            decision["rejected"].append(raw)
        decision.update(choice=letter, action=menu[decision["letters"].index(letter)]["action"],
                        wall_s=round(time.perf_counter() - t0, 4))
        self._write(decision)
        # parse_answer (llm.py:342-380) accepts a bare letter: len(text) <= 2 and option in text
        return [letter], 0

    def _write(self, rec):
        self.records.append(rec)
        save_dir = rec.get("save_dir") or (self._pending or {}).get("save_dir") or self.save_dir
        if not save_dir:
            return
        with open(os.path.join(save_dir, f"{rec['frame']:04d}_human.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        with open(os.path.join(save_dir, "human_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def summary(self):
        asked = [r for r in self.records if not r.get("automatic")]
        return {"decisions": len(self.records), "asked": len(asked),
                "automatic": len(self.records) - len(asked),
                "rejected_inputs": sum(len(r.get("rejected", [])) for r in asked),
                "wall_s_mean": round(sum(r["wall_s"] for r in asked) / len(asked), 3) if asked else None,
                "wall_s_max": round(max(r["wall_s"] for r in asked), 3) if asked else None}
