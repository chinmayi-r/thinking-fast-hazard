"""A18b: let the Anthropic client send the anthropic-workspace-id header when ANTHROPIC_WORKSPACE_ID is set
(needed for organization-level keys that are not scoped to a workspace). Idempotent."""
import io
import sys

p = r"G:\iw-thinking-fast\HAZARD\src\HAZARD\policy\llm.py"
s = io.open(p, encoding="utf-8").read()
if "_anthropic_client_for" in s:
    print("already patched")
    sys.exit(0)

helper = (
    "\n"
    "def _anthropic_client_for(key):\n"
    "    \"\"\"A18b: build the Anthropic client; an org-level key needs the workspace header (env ANTHROPIC_WORKSPACE_ID).\"\"\"\n"
    "    ws = os.environ.get('ANTHROPIC_WORKSPACE_ID', '').strip()\n"
    "    kwargs = {'default_headers': {'anthropic-workspace-id': ws}} if ws else {}\n"
    "    return anthropic.Anthropic(api_key=key, **kwargs) if key else anthropic.Anthropic(**kwargs)\n"
    "\n"
)
anchor = "class SamplingParameters:\n"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + "\n" + anchor)

old1 = "            self.anthropic_client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()  # else env / ant auth\n"
new1 = "            self.anthropic_client = _anthropic_client_for(key)  # A18b: key from file, else env / ant auth\n"
assert s.count(old1) == 1
s = s.replace(old1, new1)

old2 = "            self.anthropic_client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()\n"
new2 = "            self.anthropic_client = _anthropic_client_for(key)\n"
assert s.count(old2) == 1
s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
import ast
ast.parse(s)
print("A18b applied, syntax OK")
