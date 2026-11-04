"""A20: let the Anthropic branch of HAZARD's LLM agent attach images to the next call.

Adds `self.next_images` (list of base64 PNG strings) to the LLM class; when non-empty, the first user message's text is
sent as [image blocks..., text block] and the list is cleared after the call. Text-only behaviour is unchanged.
Idempotent. Run from G:\iw-thinking-fast:  ./venv/Scripts/python.exe scripts/patch_A20_vlm_hook.py
"""
import io
import sys

p = r"G:\iw-thinking-fast\HAZARD\src\HAZARD\policy\llm.py"
s = io.open(p, encoding="utf-8").read()
if "A20 (iw-thinking-fast" in s:
    print("already patched")
    sys.exit(0)

old_attr = "        self.call_log = []  # A18: per-call wall_s / tokens / stop_reason, for latency accounting\n"
new_attr = old_attr + "        self.next_images = []  # A20 (iw-thinking-fast, 2026-10-26): base64 PNGs to attach to the next Anthropic call (VLM baseline)\n"
assert s.count(old_attr) == 1
s = s.replace(old_attr, new_attr)

old_call = (
    "                                response = self.anthropic_client.messages.create(\n"
    "                                    model=lm_id, max_tokens=sampling_params['max_tokens'], messages=prompt,\n"
    "                                    output_config={'effort': sampling_params['effort']})\n"
)
new_call = (
    "                                _msgs = prompt\n"
    "                                if getattr(self, 'next_images', None):  # A20: VLM baseline, image + text in the first user turn\n"
    "                                    _first = dict(prompt[0])\n"
    "                                    _first['content'] = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': _b64}}\n"
    "                                                         for _b64 in self.next_images] + [{'type': 'text', 'text': prompt[0]['content']}]\n"
    "                                    _msgs = [_first] + list(prompt[1:])\n"
    "                                response = self.anthropic_client.messages.create(\n"
    "                                    model=lm_id, max_tokens=sampling_params['max_tokens'], messages=_msgs,\n"
    "                                    output_config={'effort': sampling_params['effort']})\n"
)
assert s.count(old_call) == 1
s = s.replace(old_call, new_call)

# clear the images after the (possibly two-call) decision: at the end of _generate's anthropic branch
old_ret = "                        usage = in_tok * pin / 1e6 + out_tok * pout / 1e6  # USD estimate from the cached price table\n                        return generated_samples, usage\n"
new_ret = "                        usage = in_tok * pin / 1e6 + out_tok * pout / 1e6  # USD estimate from the cached price table\n                        self.next_images = []  # A20: images apply to one API call only\n                        return generated_samples, usage\n"
assert s.count(old_ret) == 1
s = s.replace(old_ret, new_ret)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
import ast
ast.parse(s)
print("A20 hook applied, syntax OK")
