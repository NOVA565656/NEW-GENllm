#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

from openai import OpenAI


def provider(name: str, base_url: str, api_key: str, model: str):
    return OpenAI(base_url=base_url, api_key=api_key), model


def run_stage(client: OpenAI, model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def run_chain(task: str) -> str:
    groq, groq_model = provider(
        "groq",
        "https://api.groq.com/openai/v1",
        os.environ["GROQ_API_KEY"],
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    gemini, gemini_model = provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        os.environ["GEMINI_API_KEY"],
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )
    nvidia, nvidia_model = provider(
        "nvidia",
        "https://integrate.api.nvidia.com/v1",
        os.environ["NVIDIA_API_KEY"],
        os.environ["NVIDIA_MODEL"],
    )
    cloudflare, cloudflare_model = provider(
        "cloudflare",
        f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/v1",
        os.environ["CLOUDFLARE_API_TOKEN"],
        os.environ["CLOUDFLARE_MODEL"],
    )

    draft = run_stage(
        groq,
        groq_model,
        "You are a senior engineer. Produce the best first working implementation.",
        task,
    )
    print("[Stage 1 Groq] done", file=sys.stderr)

    security = run_stage(
        gemini,
        gemini_model,
        "You are a hostile security reviewer. Find concrete injection, secret, authentication, and unsafe-code risks.",
        f"Original task:\n{task}\n\nDraft:\n{draft}",
    )
    print("[Stage 2 Gemini] done", file=sys.stderr)

    bug_report = run_stage(
        nvidia,
        nvidia_model,
        "You are a debugger. Find logic bugs, concurrency faults, edge cases, and resource leaks. Return precise fixes.",
        f"Original task:\n{task}\n\nDraft:\n{draft}\n\nSecurity review:\n{security}",
    )
    print("[Stage 3 NVIDIA] done", file=sys.stderr)

    final = run_stage(
        cloudflare,
        cloudflare_model,
        "Rewrite the implementation to be secure, idiomatic, complete, and testable. Incorporate every valid critique.",
        f"Original task:\n{task}\n\nDraft:\n{draft}\n\nSecurity review:\n{security}\n\nBug report:\n{bug_report}",
    )
    print("[Stage 4 Cloudflare] done", file=sys.stderr)
    return final


def run_venice(task: str) -> str:
    venice = OpenAI(
        base_url="https://api.venice.ai/api/v1",
        api_key=os.environ["VENICE_API_KEY"],
    )
    model = os.getenv(
        "VENICE_MODEL",
        "cognitivecomputations/dolphin-mistral-24b-venice-edition",
    )
    return run_stage(
        venice,
        model,
        "Complete the requested work directly and accurately.",
        task,
    )


if __name__ == "__main__":
    task = sys.stdin.read().strip()
    if not task:
        print("No input provided via stdin.", file=sys.stderr)
        raise SystemExit(1)

    if "--venice" in sys.argv:
        print(run_venice(task))
    else:
        print(run_chain(task))