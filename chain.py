#!/usr/bin/env python3
"""Sequential 4-stage coding chain with Gemini as the security reviewer,
plus an on-demand Venice uncensored pass."""

import os
import sys
from openai import OpenAI

# ── Pool client (Stages 1, 3, 4) ──────────────────────────
pool = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="anything",  # freellmpool ignores this
)

# ── Gemini direct client (Stage 2: Security) ─────────────
gemini = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
)

GEMINI_MODEL = "gemini-2.5-flash"


def run_chain(task: str) -> str:
    """Draft -> Gemini security audit -> Debug -> Polish."""

    # ── Stage 1: Draft via free pool ──────────────────────
    draft_response = pool.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior engineer. Write the first working "
                    "version of the solution."
                ),
            },
            {"role": "user", "content": task},
        ],
        temperature=0.3,
    )
    draft = draft_response.choices[0].message.content
    print("[Drafter] done", file=sys.stderr)

    # ── Stage 2: Security review via Gemini direct ────────
    security_response = gemini.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a hostile security reviewer. Find injection "
                    "risks, hardcoded secrets, unsafe imports, and auth "
                    "flaws in the code below. Be specific."
                ),
            },
            {"role": "user", "content": draft},
        ],
        temperature=0.2,
    )
    security_critique = security_response.choices[0].message.content
    print("[Security/Gemini] done", file=sys.stderr)

    # ── Stages 3 & 4: Debug + Polish via free pool ────────
    context = (
        f"Original task:\n{task}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Security critique:\n{security_critique}"
    )

    debug_response = pool.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a debugger. Find logical bugs, concurrency "
                    "issues, edge cases, and resource leaks. Suggest precise "
                    "fixes."
                ),
            },
            {"role": "user", "content": context},
        ],
        temperature=0.3,
    )
    debugged = debug_response.choices[0].message.content
    print("[Debugger] done", file=sys.stderr)

    polish_response = pool.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code quality editor. Rewrite the code to be "
                    "clean, secure, and idiomatic, incorporating every "
                    "critique above."
                ),
            },
            {"role": "user", "content": debugged},
        ],
        temperature=0.3,
    )
    final = polish_response.choices[0].message.content
    print("[Polisher] done", file=sys.stderr)

    return final


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