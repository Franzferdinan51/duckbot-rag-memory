"""Format duckbot-ask output as snippet (just first result text).

Reads output from `python -m src.cli query` from stdin and prints the
first result's body text, trimmed to --max-chars chars.

Usage:
  python _format_snippet.py <query> <max_chars>

Input format (compact, what the CLI actually produces):
  {"query": "...", ...stats...}\n
  [1] (tier=episodic, rrf=0.05)\n
  Source: /path.md\n
  Section: # Header\n
  The body text.\n
  ---\n
  [2] (tier=semantic, ...)
  ...
"""
import sys
import json
import re

query = sys.argv[1]
max_chars = int(sys.argv[2])
raw = sys.stdin.read()

if not raw.strip():
    print(f"[no results for: {query}]")
    sys.exit(0)

# CLI outputs: JSON header on first line of stderr, compact text blocks on
# stdout. The formatters read stdout, which starts with a JSON line that
# looks like {"query": "...", "fused_results": N, ...}.
first_line, _, after = raw.partition("\n")
try:
    json.loads(first_line)
    compact_text = after
except json.JSONDecodeError:
    compact_text = raw

# Split blocks by "---" on its own line (the CLI's block separator).
blocks = compact_text.split("\n---\n")
if not blocks or not blocks[0].strip():
    fallback = re.sub(r"\s+", " ", raw).strip()
    print(fallback[:max_chars] or f"[no results for: {query}]")
    sys.exit(0)

first_block = blocks[0]

# The body text is everything AFTER the Section line.
# Section format: "Section: <header>\n<text>\n---\n"  (--- may follow the block).
# We strip trailing --- from the block first so \Z only matches true EOF,
# then find the Section line by its \n prefix and take everything after it.
first_block = first_block.rstrip("\n").rstrip("---").strip()
section_line_idx = first_block.find("\nSection: ")
if section_line_idx >= 0:
    after_section = first_block[section_line_idx + 1:]  # skip the \n
    # after_section now starts with "Section: ..." — skip that line
    nl = after_section.find("\n")
    text = after_section[nl + 1:].strip() if nl >= 0 else ""
else:
    # No Section: line — body is everything after the [N] metadata line
    first_nl = first_block.find("\n")
    text = first_block[first_nl + 1:].lstrip("\n") if first_nl >= 0 else first_block

text = text.replace("\n", " ")[:max_chars]
if text:
    print(text)
else:
    # Fallback: show what we got
    fallback = re.sub(r"\s+", " ", raw).strip()
    print(fallback[:max_chars] or f"[no results for: {query}]")
