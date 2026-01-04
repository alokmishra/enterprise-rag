"""
Enterprise RAG System - OpenAI LLM Client
"""

from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI
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


class OpenAIClient(LLMClient):
    """OpenAI GPT LLM client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
    ):
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._default_model = default_model
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise LLMError("openai", "API key not configured")
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def provider(self) -> str:
        return "openai"

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
        """Generate a response using GPT."""
        client = self._get_client()
        model = model or self._default_model

        # Convert messages to OpenAI format
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
            )

            choice = response.choices[0]
            content = choice.message.content or ""

            self.logger.info(
                "LLM generation complete",
                provider="openai",
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            return LLMResponse(
                content=content,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                finish_reason=choice.finish_reason or "stop",
                metadata={
                    "id": response.id,
                },
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str and "limit" in error_str:
                raise LLMRateLimitError("openai")
            self.logger.error("OpenAI generation failed", error=str(e))
            raise LLMError("openai", str(e))

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using GPT."""
        client = self._get_client()
        model = model or self._default_model

        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(content=chunk.choices[0].delta.content)

                # Check for final chunk
                if chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        metadata={
                            "finish_reason": chunk.choices[0].finish_reason,
                        },
                    )

        except Exception as e:
            self.logger.error("OpenAI streaming failed", error=str(e))
            raise LLMError("openai", str(e))


# Singleton instance
_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get the global OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client
