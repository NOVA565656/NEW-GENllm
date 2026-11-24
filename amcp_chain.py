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
#   CHAIN_DRAFT_PROVIDER         who writes the draft (and may use tools). Default gemini.
#   CHAIN_SECURITY_PROVIDER      security reviewer. Default nvidia.
#   CHAIN_BUG_PROVIDER           bug-hunt reviewer. Default groq.
#   CHAIN_FINAL_PROVIDER         final answer (and tools). Default cloudflare.
#   Each must be one of: groq, gemini, nvidia, cloudflare, venice.
#   CHAIN_DRAFT_FALLBACK         used once if the draft provider fails with a temporary
#                                error (429/5xx/413/timeout, or Gemini's thought_signature 400). Default
#                                cloudflare. Empty = off.
#   CHAIN_FINAL_FALLBACK         same, for the final stage. Default gemini. Empty = off.
#   CHAIN_SKIP_FINAL_WHEN_NO_REVIEW  1 (default): when the draft is short and has no code, the
#                                draft IS the answer (1 model call instead of 2). 0 = always
#                                run the final stage.
#   CHAIN_LOG_STAGES             1 = print one "CHAIN CALL ..." line per model call (size of the
#                                messages and of the tool definitions). Default 0.
#   Reviewers (security, bug) are best-effort: if one fails, the answer is still
#   produced without that review instead of the whole message failing.
# Rough rule: 4 characters is about 1 token.
# ---------------------------------------------------------------------------
def _cfg(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _provider(env_name: str, default: str) -> str:
    raw = os.environ.get(env_name)
    return (default if raw is None else raw).strip().lower()


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

    result: list[dict[str, Any]] = [{"role": "system", "content": NO_TOOLS_NOTE}]
    if system_text.strip():
        result.append({"role": "system", "content": _clip(system_text, 1500)})
    result.append(
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{transcript}\n\n---\n{instruction}"
            ),
        }
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


# Gemini 3 models return a "thought signature" with every tool call and answer HTTP 400
# ("Function call is missing a thought_signature") if the history sent back lacks it.
# Other providers never produce one, and AMCP may not keep it, so before a request goes to
# Gemini we add Google's documented skip value to any tool call that has none.
_GEMINI_SKIP_SIGNATURE = "skip_thought_signature_validator"


def _with_gemini_signatures(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages):
        return messages
    patched: list[dict[str, Any]] = []
    for m in messages:
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not calls:
            patched.append(m)
            continue
        new_calls = []
        for call in calls:
            if not isinstance(call, dict):
                new_calls.append(call)
                continue
            extra = call.get("extra_content")
            google = extra.get("google") if isinstance(extra, dict) else None
            if isinstance(google, dict) and google.get("thought_signature"):
                new_calls.append(call)
                continue
            call = dict(call)
            call["extra_content"] = {"google": {"thought_signature": _GEMINI_SKIP_SIGNATURE}}
            new_calls.append(call)
        m = dict(m)
        m["tool_calls"] = new_calls
        patched.append(m)
    return patched


def _is_signature_error(exc: BaseException) -> bool:
    return "thought_signature" in _describe_error(exc).lower()


_TRANSIENT_STATUS = {408, 409, 413, 425, 429, 500, 502, 503, 504, 529}
_TRANSIENT_NAMES = ("timeout", "connection", "ratelimit", "internalserver", "serviceunavailable", "overloaded")


def _is_transient(exc: BaseException) -> bool:
    """True for errors where trying a different provider can reasonably help."""
    if _is_signature_error(exc):
        return True
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(cur, "status_code", None)
        if isinstance(status, int):
            return status in _TRANSIENT_STATUS
        name = type(cur).__name__.lower()
        if any(tag in name for tag in _TRANSIENT_NAMES):
            return True
        cur = getattr(cur, "original_exception", None) or cur.__cause__
    return False


NO_TOOLS_NOTE = (
    "You have NO tools in this conversation. Never call a tool or function and never "
    "emit tool-call syntax. Answer in plain text only."
)

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
        self.stage_providers = {
            "draft": _provider("CHAIN_DRAFT_PROVIDER", "gemini"),
            "security": _provider("CHAIN_SECURITY_PROVIDER", "nvidia"),
            "bug": _provider("CHAIN_BUG_PROVIDER", "groq"),
            "final": _provider("CHAIN_FINAL_PROVIDER", "cloudflare"),
        }
        for role, name in self.stage_providers.items():
            if name not in self.clients:
                raise ValueError(
                    f"CHAIN_{role.upper()}_PROVIDER={name!r} is not one of {', '.join(required)}"
                )
        self.fallbacks = {
            "draft": _provider("CHAIN_DRAFT_FALLBACK", "cloudflare"),
            "final": _provider("CHAIN_FINAL_FALLBACK", "gemini"),
        }
        for role, name in list(self.fallbacks.items()):
            if name and name not in self.clients:
                raise ValueError(
                    f"CHAIN_{role.upper()}_FALLBACK={name!r} is not one of {', '.join(required)}"
                )
            if name == self.stage_providers[role]:
                self.fallbacks[role] = ""
        self.model = self.clients[self.stage_providers["draft"]].model
        logger.info("Chain stage providers: %s", self.stage_providers)

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
        if provider == "gemini":
            messages = _with_gemini_signatures(messages)
        tool_chars = len(json.dumps(tools, default=str)) if tools else 0
        logger.log(
            logging.WARNING if _cfg("CHAIN_LOG_STAGES", 0) else logging.INFO,
            "CHAIN CALL stage=%s provider=%s model=%s messages=%d chars=%d tools=%d tool_chars=%d",
            stage, provider, model, len(messages), _total_size(messages),
            len(tools or []), tool_chars,
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

    async def _stage_with_fallback(
        self,
        stage: str,
        role: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stream_callback: Any | None = None,
    ) -> LLMResponse:
        primary = self.stage_providers[role]
        try:
            return await self._stage(
                stage, primary, messages, tools=tools, stream_callback=stream_callback
            )
        except Exception as exc:  # CancelledError is BaseException and is not caught
            backup = self.fallbacks.get(role, "")
            if not backup or not _is_transient(exc):
                raise
            logger.warning(
                "Stage %s: %s failed with a temporary error, retrying once on %s",
                stage, primary, backup,
            )
            return await self._stage(
                f"{stage}_fallback", backup, messages,
                tools=tools, stream_callback=stream_callback,
            )

    async def _review_stage(
        self,
        stage: str,
        role: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Best-effort reviewer: never lets a review failure kill the user's answer."""
        try:
            response = await self._stage(stage, self.stage_providers[role], messages)
        except Exception as exc:
            logger.warning("Review stage %s skipped after failure: %s", stage, type(exc).__name__)
            return ""
        if response.tool_calls:
            logger.warning("Review stage %s returned tool calls; ignoring them", stage)
            return ""
        return response.content or ""

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
            draft_response = await self._stage_with_fallback(
                "draft", "draft", _trim_history(messages), tools=tools
            )
            if draft_response.tool_calls:
                return draft_response

            responses.append(draft_response)
            self._draft = _clip(draft_response.content or "", stage_cap)

            min_chars = _cfg("CHAIN_REVIEW_MIN_CHARS", 400)
            needs_review = bool(self._draft) and (
                min_chars <= 0 or len(self._draft) >= min_chars or "```" in self._draft
            )

            if (
                not needs_review
                and self._draft
                and _cfg("CHAIN_SKIP_FINAL_WHEN_NO_REVIEW", 1)
            ):
                # Simple reply: no reviews were needed, so a second "final" call would only
                # rewrite the same text. Return the draft and save a call (and quota).
                return LLMResponse(
                    content=draft_response.content,
                    tool_calls=None,
                    stop_reason=draft_response.stop_reason,
                    thinking=draft_response.thinking,
                    usage=self._combine_usage(responses),
                )

            if needs_review:
                self._security_review = _clip(await self._review_stage(
                    "security_review",
                    "security",
                    _review_messages(
                        messages,
                        "Review this implementation for concrete security issues: "
                        "injection, unsafe permissions, authentication flaws, secret exposure, "
                        "and unsafe handling of untrusted input. Return actionable findings only; "
                        "reply 'No issues.' if there are none.\n\n"
                        f"Draft:\n{self._draft}",
                    ),
                ), stage_cap)

                security_note = (
                    f"\n\nSecurity review:\n{self._security_review}" if self._security_review else ""
                )
                self._bug_review = _clip(await self._review_stage(
                    "bug_hunt",
                    "bug",
                    _review_messages(
                        messages,
                        "Find concrete logic bugs, edge cases, concurrency problems, and resource "
                        "leaks. Return actionable findings only; reply 'No issues.' if there are none.\n\n"
                        f"Draft:\n{self._draft}{security_note}",
                    ),
                ), stage_cap)
                self._reviews_ran = bool(self._security_review or self._bug_review)

            self.phase = "final"

        if self.phase != "final":
            raise RuntimeError(f"Standard chain in unexpected phase {self.phase!r}")

        notes = ["[Working notes - never mention or quote]"]
        if self._draft:
            notes.append(f"Draft:\n{self._draft}")
        if self._security_review:
            notes.append(f"Security review:\n{self._security_review}")
        if self._bug_review:
            notes.append(f"Bug-hunt review:\n{self._bug_review}")
        final_messages = self._append_instruction(
            _trim_history(messages),
            FINAL_INSTRUCTION + "\n\n" + "\n\n".join(notes),
        )

        final_response = await self._stage_with_fallback(
            "final", "final", final_messages,
            tools=tools, stream_callback=stream_callback,
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