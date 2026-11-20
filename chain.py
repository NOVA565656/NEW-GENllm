#!/usr/bin/env python3
from __future__ import annotations

import sys

from dotenv import load_dotenv

from chain_runtime import ProviderChain, format_status


def main() -> int:
    load_dotenv()

    task = sys.stdin.read().strip()
    if not task:
        print("No input provided via stdin.", file=sys.stderr)
        return 1

    chain = ProviderChain()

    if "--venice" in sys.argv:
        stages, final = chain.run_venice(task)
    else:
        stages, final = chain.run_chain(task)

    print(format_status(stages))
    print()

    if not final:
        return 1

    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())