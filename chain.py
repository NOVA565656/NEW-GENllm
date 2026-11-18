#!/usr/bin/env python3
"""Sequential 4-stage coding chain + on-demand Venice uncensored pass."""
import os
import sys
from openai import OpenAI

# ── Pool client (stages 1–4) ──────────────────────────────
pool = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="anything",  # freellmpool ignores this
)

STAGES = [
    (
        "Drafter",
        "You are a senior engineer. Write the first working version of the solution.",
    ),
    (
        "Security",
        "You are a hostile security reviewer. Find injection risks, hardcoded "
        "secrets, unsafe imports, and auth flaws in the code below. Be specific.",
    ),
    (
        "Debugger",
        "You are a debugger. Find logical bugs, concurrency issues, edge cases, "
        "and resource leaks. Suggest precise fixes.",
    ),
    (
        "Polisher",
        "You are a code quality editor. Rewrite the code to be clean, secure, and "
        "idiomatic, incorporating every critique above.",
    ),
]


def run_chain(task: str) -> str:
    """Run the 4-stage sequential verification chain through the free pool."""
    current = task
    for name, system_prompt in STAGES:
        response = pool.chat.completions.create(
            model="auto",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current},
            ],
            temperature=0.3,
        )
        current = response.choices[0].message.content
        print(f"[{name}] done", file=sys.stderr)
    return current


def run_venice(task: str) -> str:
    """On-demand uncensored pass. Calls Venice directly, bypassing the pool."""
    venice = OpenAI(
        base_url="https://api.venice.ai/api/v1",
        api_key=os.environ["VENICE_API_KEY"],
    )
    response = venice.chat.completions.create(
        model="cognitivecomputations/dolphin-mistral-24b-venice-edition",
        messages=[{"role": "user", "content": task}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    task = sys.stdin.read().strip()
    if not task:
        print("No input provided via stdin.", file=sys.stderr)
        sys.exit(1)

    if "--venice" in sys.argv:
        print(run_venice(task))
    else:
        print(run_chain(task))