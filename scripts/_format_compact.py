"""Format duckbot-ask output as compact (one block per result).

Reads `python -m src.cli query` output from stdin and reshapes the
multi-section text output into clean per-result blocks suitable for
Telegram or markdown.

Usage:
  python _format_compact.py <query> <n> <max_chars>
"""
import sys, json, re

query = sys.argv[1]
n = int(sys.argv[2])
max_chars = int(sys.argv[3])
raw = sys.stdin.read()

header_line = raw.split("\n")[0]
try:
    header = json.loads(header_line)
except json.JSONDecodeError:
    print(raw)
    sys.exit(0)

blocks = re.split(r"\n\[(\d+)\] ", "\n" + raw)
for i in range(1, len(blocks) - 1, 2):
    idx = blocks[i]
    body = blocks[i + 1]
    first_line = body.split("\n", 1)[0]
    rest = body[len(first_line) + 1:] if "\n" in body else ""
    source_match = re.search(r"Source: (.+)", rest)
    section_match = re.search(r"Section: (.+?)(?:\n---|\Z)", rest, re.DOTALL)
    source = source_match.group(1) if source_match else ""
    section = section_match.group(1).strip() if section_match else ""
    text = ""
    if "---" in rest:
        text = rest.split("---", 1)[1].strip()
    text = text.replace("\n", " ")[:max_chars]
    print(f"[{idx}] {first_line}")
    if section:
        print(f"    {section}")
    if source:
        if len(source) > 80:
            source = "..." + source[-77:]
        print(f"    @ {source}")
    if text:
        print(f"    > {text}")
    print()

print(f"# {header.get('fused_results', '?')} results for: {query} ({header.get('duration_seconds', '?')}s)")
