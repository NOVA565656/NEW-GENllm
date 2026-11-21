from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable

from amcp.config import ChatConfig
from amcp.llm import LLMResponse, TokenUsage, create_llm_client

logger = logging.getLogger(__name__)


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

        # Added "venice" to the required list
        required = ("groq", "gemini", "nvidia", "cloudflare", "venice")
        missing = [name for name in required if name not in self.providers]
        if missing:
            raise ValueError(f"Missing AMCP chat provider profiles: {', '.join(missing)}")

        self.clients = {
            name: self._make_client(name)
            for name in required
        }
        self.model = self.clients["groq"].model

        # Added phase initialization and venice tracking
        self.phase = "init"
        self.is_venice = False
        self._draft = ""
        self._security_review = ""
        self._bug_review = ""
        self._venice_draft = ""

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
        self._event(stage, provider, "running")
        try:
            response = await self.clients[provider].achat(
                messages=messages,
                tools=tools,
                stream_callback=stream_callback,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
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

        # --- NEW TURN RESET LOGIC ---
        # If the last message is from the user, it means we are starting a new request.
        # This ensures the chain resets properly for follow-up messages in the same session.
        last_msg_role = messages[-1].get("role") if messages else None
        if last_msg_role == "user" and self.phase not in ("init", "draft", "venice_nvidia"):
            self.phase = "init"
            self.is_venice = False
            self._draft = ""
            self._security_review = ""
            self._bug_review = ""
            self._venice_draft = ""

        # --- ROUTE DETECTION ---
        if self.phase == "init":
            # Look for the most recent user message to check for /venice
            last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            if isinstance(last_user, str) and last_user.strip().startswith("/venice"):
                self.is_venice = True
                self.phase = "venice_nvidia"
            else:
                self.is_venice = False
                self.phase = "draft"

        # --- VENICE CHAIN ---
        if self.is_venice:
            # Sanitize the /venice command so the AI doesn't get confused
            sanitized_messages = []
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), str) and "/venice" in m["content"]:
                    sanitized_messages.append({"role": "user", "content": m["content"].replace("/venice", "").strip()})
                else:
                    sanitized_messages.append(m)

            responses = []

            # Stage 1: NVIDIA (Analysis)
            if self.phase == "venice_nvidia":
                nvidia_messages = self._append_instruction(
                    sanitized_messages,
                    "You are an expert debugging assistant. Analyze this request and draft a robust solution, anticipating edge cases.",
                )
                nvidia_response = await self._stage(
                    "venice_nvidia",
                    "nvidia",
                    nvidia_messages,
                )
                if nvidia_response.tool_calls:
                    raise RuntimeError("NVIDIA in Venice chain unexpectedly returned tool calls")
                self._venice_draft = nvidia_response.content or ""
                responses.append(nvidia_response)
                self.phase = "venice_final"

            # Stage 2: Venice (Final + Tools)
            if self.phase == "venice_final":
                venice_messages = self._append_instruction(
                    sanitized_messages,
                    "You are an uncensored coding expert. Produce the final answer. Incorporate the preliminary analysis. Use the supplied tools when needed.\n\n"
                    f"Preliminary NVIDIA Analysis:\n{self._venice_draft}",
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

        # --- STANDARD CHAIN ---
        else:
            if self.phase == "draft":
                # Keep Groq's tool calls in AMCP's normal tool loop.
                draft_response = await self._stage(
                    "draft",
                    "groq",
                    messages,
                    tools=tools,
                    stream_callback=None, # Draft text is intermediate
                )
                if draft_response.tool_calls:
                    return draft_response

                self._draft = draft_response.content or ""

                security_messages = self._append_instruction(
                    messages,
                    "Review this implementation for concrete security issues: "
                    "injection, unsafe permissions, authentication flaws, secret exposure, "
                    "and unsafe handling of untrusted input. Return actionable findings only.\n\n"
                    f"Draft:\n{self._draft}",
                )
                security_response = await self._stage(
                    "security_review",
                    "gemini",
                    security_messages,
                )
                if security_response.tool_calls:
                    raise RuntimeError("Gemini security reviewer unexpectedly returned tool calls")
                self._security_review = security_response.content or ""

                bug_messages = self._append_instruction(
                    messages,
                    "Find concrete logic bugs, edge cases, concurrency problems, and resource "
                    "leaks. Return actionable findings only.\n\n"
                    f"Draft:\n{self._draft}\n\n"
                    f"Security review:\n{self._security_review}",
                )
                bug_response = await self._stage(
                    "bug_hunt",
                    "nvidia",
                    bug_messages,
                )
                if bug_response.tool_calls:
                    raise RuntimeError("NVIDIA bug reviewer unexpectedly returned tool calls")
                self._bug_review = bug_response.content or ""

                self.phase = "final"

            final_messages = self._append_instruction(
                messages,
                "Produce the final answer for the user's original request. Incorporate "
                "valid security and bug-hunt findings. Use the supplied tools when needed. "
                "Do not claim a tool was used unless AMCP returned its result.\n\n"
                f"Draft:\n{self._draft}\n\n"
                f"Security review:\n{self._security_review}\n\n"
                f"Bug-hunt review:\n{self._bug_review}",
            )

            final_response = await self._stage(
                "final",
                "cloudflare",
                final_messages,
                tools=tools,
                stream_callback=stream_callback,
            )

            responses = [final_response]
            if self.phase == "final" and self._draft:
                responses.extend([])

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