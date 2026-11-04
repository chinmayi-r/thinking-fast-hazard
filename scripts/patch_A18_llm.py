"""A18 patch for HAZARD src/HAZARD/policy/llm.py: add an 'anthropic' source (official Anthropic SDK), repair the
OpenAI branch for SDK >= 1.0, and log per-call latency. Idempotent: refuses to run twice. Run from G:\iw-thinking-fast:
    ./venv/Scripts/python.exe scripts/patch_A18_llm.py
"""
import io
import sys

p = r"G:\iw-thinking-fast\HAZARD\src\HAZARD\policy\llm.py"
s = io.open(p, encoding="utf-8").read()
if "A18 (iw-thinking-fast" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, f"anchor count {s.count(old)} != {count}: {old[:70]!r}"
    s = s.replace(old, new)


NL = "\\n"  # the two-character sequence backslash-n inside the patched source

# --- imports
rep("from openai import OpenAIError\nimport backoff\nimport time\n",
    "from openai import OpenAIError\n"
    "import backoff\n"
    "import time\n"
    "try:  # A18 (iw-thinking-fast, 2026-10-16): Claude via the official Anthropic SDK, as a third `source`\n"
    "    import anthropic\n"
    "    _ANTHROPIC_ERRORS = (anthropic.APIError,)\n"
    "except ImportError:  # the openai / huggingface paths must keep working without the package\n"
    "    anthropic = None\n"
    "    _ANTHROPIC_ERRORS = ()\n"
    "\n"
    "# Anthropic list prices, USD per 1M tokens (input, output), cached 2026-10-06; only for the running cost estimate\n"
    "_ANTHROPIC_PRICES = {'claude-opus-5-5': (4.0, 20.0), 'claude-sonnet-5-5': (2.0, 10.0), 'claude-haiku-5-5': (0.10, 0.50),\n"
    "                     'claude-opus-5': (5.0, 25.0), 'claude-sonnet-5': (2.0, 10.0)}\n")

# --- constructor
rep("                 source,  # 'huggingface' or 'openai'\n",
    "                 source,  # 'huggingface', 'openai' or 'anthropic' (A18)\n")
rep("        self.apikey_idx = 0\n"
    "        openai.api_key = self.apikey_list[self.apikey_idx]\n"
    "        self.agent_type = \"llm\"\n",
    "        self.apikey_idx = 0\n"
    "        if source == 'openai':\n"
    "            openai.api_key = self.apikey_list[self.apikey_idx]\n"
    "        self.agent_type = \"llm\"\n"
    "        self.call_log = []  # A18: per-call wall_s / tokens / stop_reason, for latency accounting\n")
rep("        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id\n",
    "        if source == 'anthropic':\n"
    "            self.chat = True\n"
    "        elif source == 'openai':\n"
    "            self.chat = not lm_id.startswith('text-')  # every current OpenAI model is a chat model\n"
    "        else:\n"
    "            self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id\n")
rep("        if self.source == 'openai':\n"
    "            tiktoken.get_encoding(\"cl100k_base\")\n"
    "            self.tokenizer = tiktoken.encoding_for_model(self.lm_id)\n"
    "            if self.chat:\n",
    "        if self.source == 'anthropic':\n"
    "            assert anthropic is not None, 'pip install anthropic'\n"
    "            # HAZARD's prompt-cutting loop needs a token counter; cl100k_base is only an approximation for Claude\n"
    "            # and only guards against absurd prompts (Claude's context is 1M tokens, so the limit is set high)\n"
    "            self.tokenizer = tiktoken.get_encoding('cl100k_base')\n"
    "            self.total_max_tokens = 100000\n"
    "            key = self.apikey_list[self.apikey_idx]\n"
    "            self.anthropic_client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()  # else env / ant auth\n"
    "            self.sampling_params = {\n"
    "                # Claude 5.5 models reject temperature/top_p; thinking is always on, so give the answer room and keep\n"
    "                # the effort low by default (this is the 'fast LLM' baseline). Override with HAZARD_ANTHROPIC_EFFORT.\n"
    "                'max_tokens': max(sampling_parameters.max_tokens, 4096),\n"
    "                'n': sampling_parameters.n,\n"
    "                'effort': os.environ.get('HAZARD_ANTHROPIC_EFFORT', 'low'),\n"
    "            }\n"
    "        elif self.source == 'openai':\n"
    "            tiktoken.get_encoding(\"cl100k_base\")\n"
    "            try:\n"
    "                self.tokenizer = tiktoken.encoding_for_model(self.lm_id)\n"
    "            except KeyError:\n"
    "                self.tokenizer = tiktoken.get_encoding('cl100k_base')\n"
    "            if self.chat:\n")

# --- engine
rep("            @backoff.on_exception(backoff.expo, OpenAIError)\n",
    "            @backoff.on_exception(backoff.expo, (OpenAIError,) + _ANTHROPIC_ERRORS, max_tries=6)\n")
rep("                if source == 'openai':\n"
    "                    base_prompt = prompt[0][\"content\"]\n",
    "                if source in ('openai', 'anthropic'):\n"
    "                    base_prompt = prompt[0][\"content\"]\n")

old_openai_block = (
    "                    try:\n"
    "                        if self.chat:\n"
    "                            response = openai.ChatCompletion.create(\n"
    "                                model=lm_id, messages=prompt, **sampling_params\n"
    "                            )\n"
    "                            # print(json.dumps(response, indent=4))\n"
    "                            if self.debug:\n"
    "                                with open(f\"llm_configs/chat_raw.json\", 'a') as f:\n"
    "                                    f.write(json.dumps(response, indent=4))\n"
    "                                    f.write('" + NL + "')\n"
    "                            generated_samples = [response['choices'][i]['message']['content'] for i in\n"
    "                                                 range(sampling_params['n'])]\n"
    "                            if 'gpt-4' in self.lm_id:\n"
    "                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage'][\n"
    "                                    'completion_tokens'] * 0.06 / 1000\n"
    "                            elif 'gpt-3.5' in self.lm_id:\n"
    "                                usage = response['usage']['total_tokens'] * 0.002 / 1000\n"
    "                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in\n"
    "                        # \t\t\t\t  range(sampling_params['n'])]\n"
    "                        elif \"text-\" in lm_id:\n"
    "                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)\n"
    "                            # print(json.dumps(response, indent=4))\n"
    "                            if self.debug:\n"
    "                                with open(f\"output/raw.json\", 'a') as f:\n"
    "                                    f.write(json.dumps(response, indent=4))\n"
    "                                    f.write('" + NL + "')\n"
    "                            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]\n"
    "                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in\n"
    "                        # \t\t\t  range(sampling_params['n'])]\n"
    "                        else:\n"
    "                            raise ValueError(f\"{lm_id} not available!\")\n"
    "                    except OpenAIError as e:\n"
    "                        print(e)\n"
    "                        print(\"Switch key and retry...\")\n"
    "                        self.sleep()\n"
    "                        raise e\n")

new_block = (
    "                    if source == 'anthropic':\n"
    "                        # A18: one Messages API call per sample; HAZARD's prompts are [user] or [user, assistant, user]\n"
    "                        generated_samples = []\n"
    "                        in_tok = out_tok = 0\n"
    "                        for _ in range(sampling_params['n']):\n"
    "                            t_call = time.perf_counter()\n"
    "                            try:\n"
    "                                response = self.anthropic_client.messages.create(\n"
    "                                    model=lm_id, max_tokens=sampling_params['max_tokens'], messages=prompt,\n"
    "                                    output_config={'effort': sampling_params['effort']})\n"
    "                            except _ANTHROPIC_ERRORS as e:\n"
    "                                print(e)\n"
    "                                print('Anthropic error, retrying...')\n"
    "                                self.sleep()\n"
    "                                raise e\n"
    "                            dt = time.perf_counter() - t_call\n"
    "                            text = ''.join(b.text for b in response.content if b.type == 'text')\n"
    "                            if response.stop_reason == 'refusal':  # safety classifier declined; parse_answer falls back\n"
    "                                print(f'Anthropic refusal: {getattr(response, \"stop_details\", None)}')\n"
    "                                text = ''\n"
    "                            generated_samples.append(text)\n"
    "                            in_tok += response.usage.input_tokens\n"
    "                            out_tok += response.usage.output_tokens\n"
    "                            self.call_log.append({'wall_s': round(dt, 4), 'input_tokens': response.usage.input_tokens,\n"
    "                                                  'output_tokens': response.usage.output_tokens,\n"
    "                                                  'stop_reason': response.stop_reason, 'model': response.model})\n"
    "                            if self.debug:\n"
    "                                os.makedirs('llm_configs', exist_ok=True)\n"
    "                                with open('llm_configs/chat_raw.json', 'a') as f:\n"
    "                                    f.write(response.model_dump_json(indent=2))\n"
    "                                    f.write('" + NL + "')\n"
    "                        pin, pout = _ANTHROPIC_PRICES.get(lm_id, (0.0, 0.0))\n"
    "                        usage = in_tok * pin / 1e6 + out_tok * pout / 1e6  # USD estimate from the cached price table\n"
    "                        return generated_samples, usage\n"
    "                    try:\n"
    "                        if self.chat:\n"
    "                            # OpenAI SDK >= 1.0 (A18): module-level client, attribute access on the response\n"
    "                            t_call = time.perf_counter()\n"
    "                            response = openai.chat.completions.create(\n"
    "                                model=lm_id, messages=prompt, **sampling_params\n"
    "                            )\n"
    "                            dt = time.perf_counter() - t_call\n"
    "                            if self.debug:\n"
    "                                os.makedirs('llm_configs', exist_ok=True)\n"
    "                                with open('llm_configs/chat_raw.json', 'a') as f:\n"
    "                                    f.write(response.model_dump_json(indent=2))\n"
    "                                    f.write('" + NL + "')\n"
    "                            generated_samples = [response.choices[i].message.content or '' for i in\n"
    "                                                 range(sampling_params['n'])]\n"
    "                            if response.usage is not None:\n"
    "                                self.call_log.append({'wall_s': round(dt, 4), 'input_tokens': response.usage.prompt_tokens,\n"
    "                                                      'output_tokens': response.usage.completion_tokens,\n"
    "                                                      'stop_reason': response.choices[0].finish_reason, 'model': response.model})\n"
    "                                if 'gpt-4' in self.lm_id:\n"
    "                                    usage = response.usage.prompt_tokens * 0.03 / 1000 + response.usage.completion_tokens * 0.06 / 1000\n"
    "                                elif 'gpt-3.5' in self.lm_id:\n"
    "                                    usage = response.usage.total_tokens * 0.002 / 1000\n"
    "                        elif 'text-' in lm_id:\n"
    "                            response = openai.completions.create(model=lm_id, prompt=prompt, **sampling_params)\n"
    "                            if self.debug:\n"
    "                                os.makedirs('output', exist_ok=True)\n"
    "                                with open('output/raw.json', 'a') as f:\n"
    "                                    f.write(response.model_dump_json(indent=2))\n"
    "                                    f.write('" + NL + "')\n"
    "                            generated_samples = [response.choices[i].text for i in range(sampling_params['n'])]\n"
    "                        else:\n"
    "                            raise ValueError(f'{lm_id} not available!')\n"
    "                    except OpenAIError as e:\n"
    "                        print(e)\n"
    "                        print('Switch key and retry...')\n"
    "                        self.sleep()\n"
    "                        raise e\n")
rep(old_openai_block, new_block)

# --- key rotation
rep("        self.apikey_idx += 1\n"
    "        if self.apikey_idx >= len(self.apikey_list):\n"
    "            self.apikey_idx = 0\n"
    "        openai.api_key = self.apikey_list[self.apikey_idx]\n"
    "        time.sleep(sleep_time)\n",
    "        self.apikey_idx += 1\n"
    "        if self.apikey_idx >= len(self.apikey_list):\n"
    "            self.apikey_idx = 0\n"
    "        if self.source == 'openai':\n"
    "            openai.api_key = self.apikey_list[self.apikey_idx]\n"
    "        elif self.source == 'anthropic' and anthropic is not None:\n"
    "            key = self.apikey_list[self.apikey_idx]\n"
    "            self.anthropic_client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()\n"
    "        time.sleep(sleep_time)\n")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
import ast
ast.parse(s)
print("llm.py patched, syntax OK")
