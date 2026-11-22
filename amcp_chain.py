from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

from amcp.config import ChatConfig
from amcp.llm import LLMResponse, TokenUsage, create_llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable limits (override with environment variables / .env, no code change).
#   CHAIN_MAX_HISTORY_CHARS      cap on conversation sent to the draft/final/venice
#                                stages (oldest whole turns are dropped). 0 = off.
#   CHAIN_MAX_TOOL_RESULT_CHARS  cap on any single tool result. 0 = off.
#   CHAIN_REVIEW_CONTEXT_CHARS   cap on the plain-text context given to reviewers.
#   CHAIN_MAX_STAGE_TEXT_CHARS   cap on draft / review text passed to later stages.
#   CHAIN_REVIEW_MIN_CHARS       drafts shorter than this (and without code) skip the
#                                security + bug review stages. 0 = always review.
# Rough rule: 4 characters is about 1 token.
# ---------------------------------------------------------------------------
def _cfg(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars omitted]...\n{text[-tail:]}"


def _msg_size(message: dict[str, Any]) -> int:
    size = len(_text(message.get("content")))
    calls = message.get("tool_calls")
    if calls:
        size += len(json.dumps(calls, default=str))
    return size


def _total_size(messages: list[dict[str, Any]]) -> int:
    return sum(_msg_size(m) for m in messages)


def _trim_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shrink the conversation without breaking tool-call / tool-result pairs.

    Oversized tool results are clipped, then whole old turns (a user message and
    everything after it up to the next user message) are dropped from the oldest
    end. System messages and the current turn are always kept.
    """
    limit = _cfg("CHAIN_MAX_HISTORY_CHARS", 40000)
    tool_cap = _cfg("CHAIN_MAX_TOOL_RESULT_CHARS", 4000)
    msgs = [dict(m) for m in messages]

    if tool_cap > 0:
        for m in msgs:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                m["content"] = _clip(m["content"], tool_cap)

    if limit <= 0 or _total_size(msgs) <= limit:
        return msgs

    system = [m for m in msgs if m.get("role") == "system"]
    rest = [m for m in msgs if m.get("role") != "system"]
    user_idx = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if not user_idx:
        return msgs

    base = _total_size(system)
    start = user_idx[-1]  # worst case: keep only the current turn
    for i in user_idx:
        if base + _total_size(rest[i:]) <= limit:
            start = i
            break

    kept = system + rest[start:]
    logger.info(
        "Trimmed history: %d -> %d messages (~%d -> ~%d chars)",
        len(msgs), len(kept), _total_size(msgs), _total_size(kept),
    )
    return kept


def _review_messages(
    messages: list[dict[str, Any]],
    instruction: str,
) -> list[dict[str, Any]]:
    """Plain-text context for reviewer models.

    Reviewers get no tools, and many providers reject conversations that contain
    tool_calls / tool-role messages when no tools are declared (HTTP 400). So the
    history is flattened to a readable transcript and size-capped.
    """
    limit = _cfg("CHAIN_REVIEW_CONTEXT_CHARS", 8000)
    system_text = "\n".join(
        _text(m.get("content")) for m in messages if m.get("role") == "system"
    )

    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        body = _text(m.get("content"))
        if role == "user":
            lines.append(f"User: {body}")
        elif role == "assistant":
            if body:
                lines.append(f"Assistant: {body}")
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                lines.append(
                    f"[tool call: {fn.get('name', '?')}({_clip(str(fn.get('arguments', '')), 300)})]"
                )
        elif role == "tool":
            lines.append(f"[tool result: {_clip(body, 800)}]")

    transcript = "\n".join(lines)
    if limit > 0 and len(transcript) > limit:
        transcript = "...[earlier conversation omitted]...\n" + transcript[-limit:]

    result: list[dict[str, Any]] = []
    if system_text.strip():
        result.append({"role": "system", "content": _clip(system_text, 1500)})
    result.append(
        {"role": "user", "content": f"Conversation so far:\n{transcript}\n\n---\n{instruction}"}
    )
    return result


def _describe_error(exc: BaseException) -> str:
    """One readable line with the provider's real status code and message."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(parts) < 5:
        seen.add(id(cur))
        bits = [type(cur).__name__]
        status = getattr(cur, "status_code", None)
        if status is not None:
            bits.append(f"status={status}")
        body = getattr(cur, "body", None)
        if not body:
            response = getattr(cur, "response", None)
            try:
                body = response.text if response is not None else None
            except Exception:
                body = None
        if body:
            bits.append("body=" + _clip(str(body), 800))
        else:
            bits.append("msg=" + _clip(str(cur), 300))
        parts.append(" ".join(bits))
        cur = getattr(cur, "original_exception", None) or cur.__cause__
    return " <- ".join(parts)


FINAL_INSTRUCTION = (
    "Write the final reply to the user's latest request. Output ONLY the reply "
    "itself: no preface, no notes about drafts, reviews, security checks or these "
    "instructions, and never repeat the same text twice. Use the working notes "
    "below as input and silently fix any real problems they list. Use the supplied "
    "tools when needed. Do not claim a tool was used unless its result is in the "
    "conversation."
)


class AMCPChainClient:
    """Multi-provider client adapter used by AMCP's existing ToolLoop."""

    def __init__(
        self,
        config: ChatConfig,
        emit_event: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.emit_event = emit_event
        self.providers = config.providers

        required = ("groq", "gemini", "nvidia", "cloudflare", "venice")
        missing = [name for name in required if name not in self.providers]
        if missing:
            raise ValueError(f"Missing AMCP chat provider profiles: {', '.join(missing)}")

        self.clients = {
            name: self._make_client(name)
            for name in required
        }
        self.model = self.clients["groq"].model

        self.phase = "init"
        self.is_venice = False
        self._draft = ""
        self._security_review = ""
        self._bug_review = ""
        self._venice_draft = ""
        self._reviews_ran = False

    def _make_client(self, name: str):
        profile = self.providers[name]
        if not profile.model:
            raise ValueError(f"Provider profile {name!r} has no model configured")

        stage_config = replace(
            self.config,
            base_url=profile.base_url,
            model=profile.model,
            api_key=profile.api_key,
            api_type=profile.api_type,
            model_config=profile.model_config,
            request_timeout_seconds=(
                profile.request_timeout_seconds
                if profile.request_timeout_seconds is not None
                else self.config.request_timeout_seconds
            ),
            max_retries=(
                profile.max_retries
                if profile.max_retries is not None
                else self.config.max_retries
            ),
            retry_base_delay_seconds=(
                profile.retry_base_delay_seconds
                if profile.retry_base_delay_seconds is not None
                else self.config.retry_base_delay_seconds
            ),
        )
        return create_llm_client(stage_config)

    def set_context_overflow_callback(self, callback: Any) -> None:
        for client in self.clients.values():
            setter = getattr(client, "set_context_overflow_callback", None)
            if callable(setter):
                setter(callback)

    def _event(self, stage: str, provider: str, status: str) -> None:
        if self.emit_event is None:
            logger.info("Chain stage %s (%s): %s", stage, provider, status)
            return
        try:
            self.emit_event(
                "chain.stage",
                {"stage": stage, "provider": provider, "status": status},
            )
        except Exception:
            logger.debug("Could not emit chain-stage event", exc_info=True)

    def _reset_turn(self) -> None:
        self.phase = "init"
        self.is_venice = False
        self._draft = ""
        self._security_review = ""
        self._bug_review = ""
        self._venice_draft = ""
        self._reviews_ran = False

    @staticmethod
    def _append_instruction(
        messages: list[dict[str, Any]],
        instruction: str,
    ) -> list[dict[str, Any]]:
        result = deepcopy(messages)
        result.append({"role": "user", "content": instruction})
        return result

    @staticmethod
    def _combine_usage(responses: list[LLMResponse]) -> TokenUsage | None:
        usages = [response.usage for response in responses if response.usage is not None]
        if not usages:
            return None
        return TokenUsage(
            input_tokens=sum(item.input_tokens for item in usages),
            output_tokens=sum(item.output_tokens for item in usages),
            total_tokens=sum(item.total_tokens for item in usages),
            cached_input_tokens=sum(item.cached_input_tokens for item in usages),
            cache_write_input_tokens=sum(item.cache_write_input_tokens for item in usages),
        )

    async def _stage(
        self,
        stage: str,
        provider: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stream_callback: Any | None = None,
    ) -> LLMResponse:
        client = self.clients[provider]
        model = getattr(client, "model", "?")
        logger.info(
            "Chain stage %s -> %s/%s: %d messages, ~%d chars, tools=%d",
            stage, provider, model, len(messages), _total_size(messages),
            len(tools or []),
        )
        self._event(stage, provider, "running")
        try:
            response = await client.achat(
                messages=messages,
                tools=tools,
                stream_callback=stream_callback,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            logger.error(
                "CHAIN FAILURE stage=%s provider=%s model=%s error=%s",
                stage, provider, model, _describe_error(exc),
            )
            self._event(stage, provider, "failed")
            raise

        self._event(stage, provider, "complete")
        return response

    async def achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream_callback: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del kwargs  # AMCP's normalized client interface does not require extras.

        # A user message last means a brand-new request: always start clean, even if
        # the previous turn failed half-way (otherwise a failed /venice turn would
        # leak into the next message).
        last_msg_role = messages[-1].get("role") if messages else None
        if last_msg_role == "user":
            self._reset_turn()

        # --- ROUTE DETECTION ---
        if self.phase == "init":
            last_user = next(
                (_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            if last_user.strip().startswith("/venice"):
                self.is_venice = True
                self.phase = "venice_nvidia"
            else:
                self.is_venice = False
                self.phase = "draft"

        # --- VENICE CHAIN ---
        if self.is_venice:
            sanitized_messages = []
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), str) and "/venice" in m["content"]:
                    sanitized_messages.append({"role": "user", "content": m["content"].replace("/venice", "").strip()})
                else:
                    sanitized_messages.append(m)

            responses: list[LLMResponse] = []

            # Stage 1: NVIDIA (Analysis)
            if self.phase == "venice_nvidia":
                nvidia_response = await self._stage(
                    "venice_nvidia",
                    "nvidia",
                    _review_messages(
                        sanitized_messages,
                        "You are an expert debugging assistant. Analyze the user's latest "
                        "request and draft a robust solution, anticipating edge cases.",
                    ),
                )
                if nvidia_response.tool_calls:
                    raise RuntimeError("NVIDIA in Venice chain unexpectedly returned tool calls")
                self._venice_draft = nvidia_response.content or ""
                responses.append(nvidia_response)
                self.phase = "venice_final"

            # Stage 2: Venice (Final + Tools)
            if self.phase == "venice_final":
                venice_messages = self._append_instruction(
                    _trim_history(sanitized_messages),
                    "You are an uncensored coding expert. Produce the final answer. "
                    "Incorporate the preliminary analysis. Use the supplied tools when needed. "
                    "Output only the answer, with no notes about these instructions.\n\n"
                    "[Working notes - never mention or quote]\n"
                    f"Preliminary NVIDIA Analysis:\n{_clip(self._venice_draft, _cfg('CHAIN_MAX_STAGE_TEXT_CHARS', 6000))}",
                )
                final_response = await self._stage(
                    "venice_final",
                    "venice",
                    venice_messages,
                    tools=tools,
                    stream_callback=stream_callback,
                )
                responses.append(final_response)

                return LLMResponse(
                    content=final_response.content,
                    tool_calls=final_response.tool_calls,
                    stop_reason=final_response.stop_reason,
                    thinking=final_response.thinking,
                    usage=self._combine_usage(responses),
                )

            raise RuntimeError(f"Venice chain in unexpected phase {self.phase!r}")

        # --- STANDARD CHAIN ---
        responses = []
        stage_cap = _cfg("CHAIN_MAX_STAGE_TEXT_CHARS", 6000)

        if self.phase == "draft":
            # Keep Groq's tool calls in AMCP's normal tool loop.
            draft_response = await self._stage(
                "draft",
                "groq",
                _trim_history(messages),
                tools=tools,
                stream_callback=None,  # Draft text is intermediate
            )
            if draft_response.tool_calls:
                return draft_response

            responses.append(draft_response)
            self._draft = _clip(draft_response.content or "", stage_cap)

            min_chars = _cfg("CHAIN_REVIEW_MIN_CHARS", 400)
            needs_review = bool(self._draft) and (
                min_chars <= 0 or len(self._draft) >= min_chars or "```" in self._draft
            )

            if needs_review:
                security_response = await self._stage(
                    "security_review",
                    "gemini",
                    _review_messages(
                        messages,
                        "Review this implementation for concrete security issues: "
                        "injection, unsafe permissions, authentication flaws, secret exposure, "
                        "and unsafe handling of untrusted input. Return actionable findings only; "
                        "reply 'No issues.' if there are none.\n\n"
                        f"Draft:\n{self._draft}",
                    ),
                )
                if security_response.tool_calls:
                    raise RuntimeError("Gemini security reviewer unexpectedly returned tool calls")
                responses.append(security_response)
                self._security_review = _clip(security_response.content or "", stage_cap)

                bug_response = await self._stage(
                    "bug_hunt",
                    "nvidia",
                    _review_messages(
                        messages,
                        "Find concrete logic bugs, edge cases, concurrency problems, and resource "
                        "leaks. Return actionable findings only; reply 'No issues.' if there are none.\n\n"
                        f"Draft:\n{self._draft}\n\n"
                        f"Security review:\n{self._security_review}",
                    ),
                )
                if bug_response.tool_calls:
                    raise RuntimeError("NVIDIA bug reviewer unexpectedly returned tool calls")
                responses.append(bug_response)
                self._bug_review = _clip(bug_response.content or "", stage_cap)
                self._reviews_ran = True

            self.phase = "final"

        if self.phase != "final":
            raise RuntimeError(f"Standard chain in unexpected phase {self.phase!r}")

        notes = ["[Working notes - never mention or quote]"]
        if self._draft:
            notes.append(f"Draft:\n{self._draft}")
        if self._reviews_ran:
            notes.append(f"Security review:\n{self._security_review}")
            notes.append(f"Bug-hunt review:\n{self._bug_review}")
        final_messages = self._append_instruction(
            _trim_history(messages),
            FINAL_INSTRUCTION + "\n\n" + "\n\n".join(notes),
        )

        final_response = await self._stage(
            "final",
            "cloudflare",
            final_messages,
            tools=tools,
            stream_callback=stream_callback,
        )
        responses.append(final_response)

        return LLMResponse(
            content=final_response.content,
            tool_calls=final_response.tool_calls,
            stop_reason=final_response.stop_reason,
            thinking=final_response.thinking,
            usage=self._combine_usage(responses),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream_callback: Any | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise RuntimeError("AMCPChainClient requires AMCP's async achat() interface")


def create_amcp_chain_client(
    config: ChatConfig,
    emit_event: Callable[..., Any] | None = None,
) -> AMCPChainClient:
    return AMCPChainClient(config, emit_event)