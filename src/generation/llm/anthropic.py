"""
Enterprise RAG System - Anthropic (Claude) LLM Client
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from anthropic import AsyncAnthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.config import settings
from src.core.exceptions import LLMError, LLMRateLimitError
from src.generation.llm.base import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    StreamChunk,
)


class AnthropicClient(LLMClient):
    """Anthropic Claude LLM client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._default_model = default_model or settings.DEFAULT_LLM_MODEL
        self._client: Optional[AsyncAnthropic] = None

    def _get_client(self) -> AsyncAnthropic:
        """Get or create the Anthropic client."""
        if self._client is None:
            if not self._api_key:
                raise LLMError("anthropic", "API key not configured")
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return self._default_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((Exception,)),
    )
    async def generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response using Claude."""
        client = self._get_client()
        model = model or self._default_model

        # Extract system message if present
        system = None
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        try:
            response = await client.messages.create(
                model=model,
                messages=api_messages,
                system=system or "",
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences or [],
            )

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            self.logger.info(
                "LLM generation complete",
                provider="anthropic",
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return LLMResponse(
                content=content,
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason or "end_turn",
                metadata={
                    "id": response.id,
                    "type": response.type,
                },
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str and "limit" in error_str:
                raise LLMRateLimitError("anthropic")
            self.logger.error("Anthropic generation failed", error=str(e))
            raise LLMError("anthropic", str(e))

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using Claude."""
        client = self._get_client()
        model = model or self._default_model

        # Extract system message
        system = None
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        try:
            async with client.messages.stream(
                model=model,
                messages=api_messages,
                system=system or "",
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences or [],
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(content=text)

                # Get final message for metadata
                message = await stream.get_final_message()
                yield StreamChunk(
                    content="",
                    is_final=True,
                    metadata={
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                        "stop_reason": message.stop_reason,
                    },
                )

        except Exception as e:
            self.logger.error("Anthropic streaming failed", error=str(e))
            raise LLMError("anthropic", str(e))


# Singleton instance
_anthropic_client: Optional[AnthropicClient] = None


def get_anthropic_client() -> AnthropicClient:
    """Get the global Anthropic client instance."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AnthropicClient()
    return _anthropic_client
