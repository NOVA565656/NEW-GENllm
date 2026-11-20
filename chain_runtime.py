#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai import OpenAI, OpenAIError


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    name: str
    provider: str
    model: str
    status: StageStatus
    output: str = ""
    error: str = ""


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def openai_provider(
    *,
    base_url: str,
    api_key: str,
) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=float(os.getenv("CHAIN_REQUEST_TIMEOUT", "120")),
        max_retries=int(os.getenv("CHAIN_MAX_RETRIES", "2")),
    )


class ProviderChain:
    def __init__(self) -> None:
        self.groq_model = os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        )
        self.gemini_model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )
        self.nvidia_model = env_required("NVIDIA_MODEL")
        self.cloudflare_model = env_required("CLOUDFLARE_MODEL")
        self.venice_model = os.getenv(
            "VENICE_MODEL",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition",
        )

        self.groq = openai_provider(
            base_url="https://api.groq.com/openai/v1",
            api_key=env_required("GROQ_API_KEY"),
        )
        self.gemini = openai_provider(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=env_required("GEMINI_API_KEY"),
        )
        self.nvidia = openai_provider(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=env_required("NVIDIA_API_KEY"),
        )
        self.cloudflare = openai_provider(
            base_url=(
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{env_required('CLOUDFLARE_ACCOUNT_ID')}/ai/v1"
            ),
            api_key=env_required("CLOUDFLARE_API_TOKEN"),
        )
        self.venice = openai_provider(
            base_url="https://api.venice.ai/api/v1",
            api_key=os.getenv("VENICE_API_KEY", ""),
        )

    def complete(
        self,
        client: OpenAI,
        model: str,
        system: str,
        user: str,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            temperature=0.2,
        )
        message = response.choices[0].message
        return message.content or ""

    def run_chain(self, task: str) -> tuple[list[StageResult], str]:
        stages: list[StageResult] = [
            StageResult("Draft", "Groq", self.groq_model, StageStatus.PENDING),
            StageResult("Security Review", "Gemini", self.gemini_model, StageStatus.PENDING),
            StageResult("Bug Hunt", "NVIDIA", self.nvidia_model, StageStatus.PENDING),
            StageResult("Rewrite", "Cloudflare", self.cloudflare_model, StageStatus.PENDING),
        ]

        try:
            stages[0].status = StageStatus.RUNNING
            stages[0].output = self.complete(
                self.groq,
                self.groq_model,
                "You are a senior engineer. Produce the best working first implementation.",
                task,
            )
            stages[0].status = StageStatus.DONE
        except Exception as exc:
            stages[0].status = StageStatus.FAILED
            stages[0].error = self.safe_error(exc)
            return stages, ""

        try:
            stages[1].status = StageStatus.RUNNING
            stages[1].output = self.complete(
                self.gemini,
                self.gemini_model,
                (
                    "You are a hostile security reviewer. Find concrete injection, "
                    "secret, authentication, unsafe import, and authorization issues."
                ),
                f"Original task:\n{task}\n\nDraft:\n{stages[0].output}",
            )
            stages[1].status = StageStatus.DONE
        except Exception as exc:
            stages[1].status = StageStatus.FAILED
            stages[1].error = self.safe_error(exc)
            stages[2].status = StageStatus.SKIPPED
            stages[3].status = StageStatus.SKIPPED
            return stages, ""

        try:
            stages[2].status = StageStatus.RUNNING
            stages[2].output = self.complete(
                self.nvidia,
                self.nvidia_model,
                (
                    "You are a debugger. Find logical bugs, concurrency issues, "
                    "edge cases, resource leaks, and API misuse. Return precise fixes."
                ),
                (
                    f"Original task:\n{task}\n\n"
                    f"Draft:\n{stages[0].output}\n\n"
                    f"Security review:\n{stages[1].output}"
                ),
            )
            stages[2].status = StageStatus.DONE
        except Exception as exc:
            stages[2].status = StageStatus.FAILED
            stages[2].error = self.safe_error(exc)
            stages[3].status = StageStatus.SKIPPED
            return stages, ""

        try:
            stages[3].status = StageStatus.RUNNING
            stages[3].output = self.complete(
                self.cloudflare,
                self.cloudflare_model,
                (
                    "Rewrite the implementation to be complete, secure, idiomatic, "
                    "testable, and consistent. Incorporate every valid critique."
                ),
                (
                    f"Original task:\n{task}\n\n"
                    f"Draft:\n{stages[0].output}\n\n"
                    f"Security review:\n{stages[1].output}\n\n"
                    f"Bug report:\n{stages[2].output}"
                ),
            )
            stages[3].status = StageStatus.DONE
            return stages, stages[3].output
        except Exception as exc:
            stages[3].status = StageStatus.FAILED
            stages[3].error = self.safe_error(exc)
            return stages, ""

    def run_venice(self, task: str) -> tuple[list[StageResult], str]:
        stages = [
            StageResult("Draft", "Venice", self.venice_model, StageStatus.PENDING),
            StageResult("Bug Hunt & Rewrite", "NVIDIA", self.nvidia_model, StageStatus.PENDING),
        ]

        if not os.getenv("VENICE_API_KEY", "").strip():
            stages[0].status = StageStatus.FAILED
            stages[0].error = "Missing environment variable: VENICE_API_KEY"
            stages[1].status = StageStatus.SKIPPED
            return stages, ""

        try:
            stages[0].status = StageStatus.RUNNING
            stages[0].output = self.complete(
                self.venice,
                self.venice_model,
                "Complete the requested work directly and accurately.",
                task,
            )
            stages[0].status = StageStatus.DONE
        except Exception as exc:
            stages[0].status = StageStatus.FAILED
            stages[0].error = self.safe_error(exc)
            stages[1].status = StageStatus.SKIPPED
            return stages, ""

        try:
            stages[1].status = StageStatus.RUNNING
            stages[1].output = self.complete(
                self.nvidia,
                self.nvidia_model,
                (
                    "Act as a security-conscious bug hunter and code editor. "
                    "Fix the draft, test the result, and provide only the final implementation."
                ),
                f"Original task:\n{task}\n\nVenice draft:\n{stages[0].output}",
            )
            stages[1].status = StageStatus.DONE
            return stages, stages[1].output
        except Exception as exc:
            stages[1].status = StageStatus.FAILED
            stages[1].error = self.safe_error(exc)
            return stages, ""

    @staticmethod
    def safe_error(exc: Exception) -> str:
        if isinstance(exc, OpenAIError):
            return exc.__class__.__name__
        return exc.__class__.__name__


def format_status(stages: list[StageResult]) -> str:
    icons = {
        StageStatus.PENDING: "⏳",
        StageStatus.RUNNING: "🔄",
        StageStatus.DONE: "✅",
        StageStatus.FAILED: "❌",
        StageStatus.SKIPPED: "⏭",
    }
    lines = ["Chain status"]
    for stage in stages:
        line = f"{icons[stage.status]} {stage.name} — {stage.provider}"
        if stage.status == StageStatus.FAILED:
            line += f"\n  {stage.error or 'Provider failed'}"
        lines.append(line)
    return "\n".join(lines)