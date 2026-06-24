"""
connectors/learn.py — Hermes /learn shim.

Hermes added `/learn "<what>"` as a CLI command that creates a reusable
skill from any input. We integrate it as a bridge:

  learn(text, ...)
    1. Ingest `text` into the brain as `procedural` tier (a rule /
       best-practice, which is what /learn is for).
    2. Also write `text` to ~/.openclaw/workspace/memory/learning/<date>.md
       so OpenClaw's dreamer + memory system can pick it up.
    3. If Hermes is on PATH, optionally shell out to `hermes learn` to
       trigger its skill-creation path too.

The brain does NOT block on Hermes. If hermes isn't installed, we just
skip step 3.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..memory import Memory, Tier


DEFAULT_LEARNING_DIR = Path.home() / ".openclaw" / "workspace" / "memory" / "learning"


@dataclass
class LearnResult:
    chunk_id: Optional[str] = None
    written_to: Optional[str] = None
    hermes_invoked: bool = False
    hermes_output: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "written_to": self.written_to,
            "hermes_invoked": self.hermes_invoked,
            "hermes_output": self.hermes_output[:400] if self.hermes_output else "",
            "error": self.error,
        }


class LearnBridge:
    def __init__(
        self,
        memory: Memory,
        learning_dir: Path = DEFAULT_LEARNING_DIR,
        invoke_hermes: bool = True,
    ):
        self.memory = memory
        self.learning_dir = Path(learning_dir)
        self.invoke_hermes = invoke_hermes

    async def learn(
        self,
        text: str,
        force_tier: str = "procedural",
        source: str = "<hermes-/learn>",
        metadata: Optional[dict] = None,
    ) -> LearnResult:
        result = LearnResult()

        if not text or not text.strip():
            result.error = "empty text"
            return result

        # Step 1: ingest into the brain.
        try:
            r = await self.memory.remember(
                text,
                source_path=source,
                force_tier=force_tier,
                metadata=metadata or {},
            )
            result.chunk_id = r.chunk_id
        except Exception as e:
            result.error = f"brain remember failed: {e}"
            return result

        # Step 2: write to OpenClaw's learning dir (so dreamer / memory
        # system can pick it up).
        try:
            self.learning_dir.mkdir(parents=True, exist_ok=True)
            date_str = time.strftime("%Y-%m-%d", time.localtime())
            out = self.learning_dir / f"{date_str}.md"
            with out.open("a", encoding="utf-8") as f:
                f.write(f"\n## /learn @ {time.strftime('%H:%M:%S')}\n\n{text}\n")
            result.written_to = str(out)
        except OSError as e:
            result.error = f"learn dir write failed: {e}"

        # Step 3: optionally invoke `hermes learn "<text>"`.
        if self.invoke_hermes and shutil.which("hermes"):
            try:
                proc = subprocess.run(
                    ["hermes", "learn", text],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                result.hermes_invoked = True
                result.hermes_output = (proc.stdout or "") + (proc.stderr or "")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                result.hermes_output = f"hermes invocation failed: {e}"

        return result


# -----------------------------------------------------------------------------
# Sync wrapper
# -----------------------------------------------------------------------------

def _run_async(coro):
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def learn(
    memory: Memory,
    text: str,
    force_tier: str = "procedural",
    source: str = "<hermes-/learn>",
    metadata: Optional[dict] = None,
    invoke_hermes: bool = True,
) -> dict:
    """Sync wrapper for the `/learn` integration."""
    bridge = LearnBridge(memory, invoke_hermes=invoke_hermes)
    return _run_async(bridge.learn(text, force_tier, source, metadata)).to_dict()
