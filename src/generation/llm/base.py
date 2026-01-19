"""
Enterprise RAG System - LLM Client Base Classes
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from src.core.logging import LoggerMixin


@dataclass
class LLMMessage:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    finish_reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk of streamed response."""
    content: str
    is_final: bool = False
    metadata: dict = field(default_factory=dict)


class LLMClient(ABC, LoggerMixin):
    """Abstract base class for LLM clients."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of conversation messages
            model: Model to use (defaults to provider default)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate a streaming response from the LLM.

        Args:
            messages: List of conversation messages
            model: Model to use (defaults to provider default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with content
        """
        pass

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple completion interface.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text content
        """
        messages = []
        if system:
            messages.append(LLMMessage(role="system", content=system))
        messages.append(LLMMessage(role="user", content=prompt))

        response = await self.generate(messages, **kwargs)
        return response.content
