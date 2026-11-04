"""VLM baseline for HAZARD (Tom, Oct 16: "a VLM baseline that actually uses the images").

VLMAgent is HAZARD's own LLM agent (policy/llm.py, with the A18 Anthropic source and the A20 image hook) plus one
change: before each decision it attaches the agent's current camera frame (state['raw']['rgb']) to the prompt, so
Claude sees the room as well as the text description. Everything else (prompt template, chain of thought, action
parsing, latency logging) is HAZARD's. Lives outside HAZARD; the runner constructs it for --agent vlm.
"""
import base64
import io

import numpy as np
from PIL import Image

from policy.llm import LLM  # HAZARD module (src/HAZARD on PYTHONPATH, as run_experiments.py:2 does)


class VLMAgent(LLM):
    def __init__(self, *a, image_side: int = 512, **k):
        super().__init__(*a, **k)
        self.image_side = image_side
        self.images_attached = 0

    @staticmethod
    def _find_rgb(state):
        raw = state.get("raw", {}) if isinstance(state, dict) else {}
        for key in ("rgb", "img", "image"):
            if key in raw:
                return raw[key]
        return None

    def _encode(self, rgb) -> str:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):  # HAZARD gives (3, H, W) floats in [0, 1]
            arr = arr.transpose(1, 2, 0)[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
        if self.image_side and max(im.size) > self.image_side:
            im = im.resize((self.image_side, self.image_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("ascii")

    def choose_target(self, state, processed_input):
        rgb = self._find_rgb(state)
        if rgb is not None and self.source == "anthropic":
            try:
                self.next_images = [self._encode(rgb)]
                self.images_attached += 1
            except Exception as e:  # never let the image path break the decision
                print(f"VLMAgent: could not encode frame: {e}")
                self.next_images = []
        return super().choose_target(state, processed_input)
