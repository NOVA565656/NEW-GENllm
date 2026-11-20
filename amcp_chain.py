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

        required = ("groq", "gemini", "nvidia", "cloudflare")
        missing = [name for name in required if name not in self.providers]
        if missing:
            raise ValueError(f"Missing AMCP chat provider profiles: {', '.join(missing)}")

        self.clients = {
            name: self._make_client(name)
            for name in required
        }
        self.model = self.clients["groq"].model

        self.phase = "draft"
        self._draft = ""
        self._security_review = ""
        self._bug_review = ""

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

        if self.phase == "draft":
            # Keep Groq's tool calls in AMCP's normal tool loop. AMCP will execute
            # them and call this adapter again with the paired tool results.
            draft_response = await self._stage(
                "draft",
                "groq",
                messages,
                tools=tools,
                # Draft text is intermediate; do not stream it to Telegram.
                stream_callback=None,
            )
            if draft_response.tool_calls:
                return draft_response

            self._draft = draft_response.content or ""

            # Reviewer stages are analysis-only. Tool execution remains with
            # AMCP and is not delegated to these providers.
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

        # Rebuild from AMCP's current canonical messages each time. If Cloudflare
        # requests a tool, AMCP appends the assistant tool call and tool result;
        # the next call sees those messages and this stage continues normally.
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

        # Include usage for the stages performed during this call. Usage from
        # earlier Groq tool-call rounds was already recorded by AMCP separately.
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