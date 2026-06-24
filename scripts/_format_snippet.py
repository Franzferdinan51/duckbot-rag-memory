"""Format duckbot-ask output as snippet (just first result text).

Reads `python -m src.cli query` output from stdin and prints the first
result's body text, trimmed to --max-chars chars.

Usage:
  python _format_snippet.py <query> <max_chars>
"""
import sys, re

query = sys.argv[1]
max_chars = int(sys.argv[2])
raw = sys.stdin.read()
m = re.search(r"\[1\].*?---\n(.*?)(?=\n\[\d+\]|\Z)", raw, re.DOTALL)
if m and m.group(1).strip():
    snippet = m.group(1).strip().replace("\n", " ")
    print(snippet[:max_chars])
else:
    # Body was empty (CLI truncated to max-chars). Fall back to the section
    # header — at least that tells the user what the chunk is about.
    sec = re.search(r"Section: (.+)", raw)
    if sec:
        print(f"[snippet truncated to {max_chars} chars] {sec.group(1).strip()[:max_chars]}")
    else:
        print(raw[:max_chars])
