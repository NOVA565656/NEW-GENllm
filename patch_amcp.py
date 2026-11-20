#!/usr/bin/env python3
"""Apply the local provider-chain hook to the pinned AMCP installation."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

EXPECTED_AMCP_VERSION = "0.13.0"
MARKER = "# NEW-GEN11m provider-chain hook"

OLD_BLOCK = '''        # Use new LLM client abstraction
        from .llm import create_llm_client

        llm_client = create_llm_client(cfg.chat)
        self._agent._attach_context_overflow_observer(llm_client)
'''

NEW_BLOCK = '''        # NEW-GEN11m provider-chain hook
        import os

        if os.environ.get("AMCP_CHAIN_ENABLED") == "1":
            from amcp_chain import create_amcp_chain_client

            llm_client = create_amcp_chain_client(
                cfg.chat,
                emit_event=self._agent._emit_event,
            )
        else:
            from .llm import create_llm_client

            llm_client = create_llm_client(cfg.chat)

        self._agent._attach_context_overflow_observer(llm_client)
'''


def main() -> int:
    try:
        version = importlib.metadata.version("amcp-agent")
    except importlib.metadata.PackageNotFoundError:
        print("ERROR: amcp-agent is not installed in this Python environment.", file=sys.stderr)
        return 1

    if version != EXPECTED_AMCP_VERSION:
        print(
            f"ERROR: expected amcp-agent {EXPECTED_AMCP_VERSION}, found {version}. "
            "Review this patch before using a different AMCP version.",
            file=sys.stderr,
        )
        return 1

    spec = importlib.util.find_spec("amcp.tool_loop")
    if spec is None or spec.origin is None:
        print("ERROR: cannot locate amcp.tool_loop.", file=sys.stderr)
        return 1

    path = Path(spec.origin)
    source = path.read_text(encoding="utf-8")

    if MARKER in source:
        print(f"AMCP chain hook already present in {path}")
        return 0

    if OLD_BLOCK not in source:
        print(
            f"ERROR: expected AMCP 0.13.0 ToolLoop.run block not found in {path}. "
            "No file was modified.",
            file=sys.stderr,
        )
        return 1

    backup = path.with_suffix(path.suffix + ".new-gen11m-backup")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")

    patched = source.replace(OLD_BLOCK, NEW_BLOCK, 1)
    path.write_text(patched, encoding="utf-8")

    print(f"Applied provider-chain hook to {path}")
    print(f"Backup saved at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())