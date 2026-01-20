"""
Enterprise RAG System - Local LLM Client

Client for local LLMs (Llama, Mistral, etc.) for air-gapped deployments.
Supports both llama-cpp-python and Hugging Face transformers backends.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

from src.core.config import settings
from src.core.exceptions import ConfigurationError, LLMError
from src.generation.llm.base import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    StreamChunk,
)


class LocalLLMClient(LLMClient):
    """
    Client for local LLM inference.

    Supports multiple backends:
    - llama-cpp-python: For GGUF/GGML models (default)
    - transformers: For Hugging Face models

    Configuration:
    - LOCAL_LLM_MODEL_PATH: Path to model file or directory
    - LOCAL_LLM_MODEL_TYPE: "llama", "mistral", "phi", etc.
    - LOCAL_LLM_BACKEND: "llama-cpp" or "transformers"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: Optional[str] = None,
        backend: str = "llama-cpp",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        **kwargs: Any,
    ):
        """
        Initialize local LLM client.

        Args:
            model_path: Path to model file/directory
            model_type: Model type (llama, mistral, phi)
            backend: Inference backend (llama-cpp, transformers)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            **kwargs: Additional backend-specific parameters
        """
        self._model_path = model_path or settings.LOCAL_LLM_MODEL_PATH
        self._model_type = model_type or settings.LOCAL_LLM_MODEL_TYPE
        self._backend = backend
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._extra_params = kwargs

        # Lazy-loaded model
        self._model: Any = None

        if not self._model_path:
            raise ConfigurationError(
                "LOCAL_LLM_MODEL_PATH must be set for local LLM"
            )

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "local"

    @property
    def default_model(self) -> str:
        """Get the default model name."""
        return self._model_type or "local"

    def _load_model(self) -> Any:
        """Load the model (lazy initialization)."""
        if self._model is not None:
            return self._model

        if self._backend == "llama-cpp":
            self._model = self._load_llama_cpp()
        elif self._backend == "transformers":
            self._model = self._load_transformers()
        else:
            raise ConfigurationError(f"Unsupported backend: {self._backend}")

        return self._model

    def _load_llama_cpp(self) -> Any:
        """Load model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ConfigurationError(
                "llama-cpp-python is required for local LLM. "
                "Install with: pip install llama-cpp-python"
            )

        self.logger.info(
            "Loading local LLM with llama-cpp",
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
        )

        return Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
            **self._extra_params,
        )

    def _load_transformers(self) -> Any:
        """Load model using Hugging Face transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ConfigurationError(
                "transformers and torch are required for transformers backend. "
                "Install with: pip install transformers torch"
            )

        self.logger.info(
            "Loading local LLM with transformers",
            model_path=self._model_path,
        )

        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        return {"model": model, "tokenizer": tokenizer}

    def _format_messages(self, messages: list[LLMMessage]) -> str:
        """Format messages into a prompt string."""
        # Simple chat format - can be customized per model type
        formatted_parts = []

        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role == "user":
                formatted_parts.append(f"<|user|>\n{msg.content}</s>")
            elif msg.role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{msg.content}</s>")

        # Add assistant prompt for generation
        formatted_parts.append("<|assistant|>\n")

        return "\n".join(formatted_parts)

    def _format_messages_llama(self, messages: list[LLMMessage]) -> str:
        """Format messages for Llama-style models."""
        formatted_parts = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                if system_prompt:
                    formatted_parts.append(
                        f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{msg.content} [/INST]"
                    )
                    system_prompt = None
                else:
                    formatted_parts.append(f"[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                formatted_parts.append(f" {msg.content} ")

        return "".join(formatted_parts)

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
        Generate a response using the local LLM.

        Args:
            messages: List of conversation messages
            model: Ignored for local LLM (uses configured model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        # Load model lazily
        llm = self._load_model()

        # Format messages into prompt
        if self._model_type in ("llama", "llama2", "llama3"):
            prompt = self._format_messages_llama(messages)
        else:
            prompt = self._format_messages(messages)

        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        if self._backend == "llama-cpp":
            result = await loop.run_in_executor(
                None,
                lambda: llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_sequences or [],
                    echo=False,
                ),
            )

            content = result["choices"][0]["text"]
            input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = result.get("usage", {}).get("completion_tokens", 0)
            finish_reason = result["choices"][0].get("finish_reason", "stop")

        else:  # transformers
            model_obj = llm["model"]
            tokenizer = llm["tokenizer"]

            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model_obj, "device"):
                inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}

            result = await loop.run_in_executor(
                None,
                lambda: model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                ),
            )

            output = tokenizer.decode(result[0], skip_special_tokens=True)
            # Remove the prompt from output
            content = output[len(prompt):].strip()
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = result.shape[1] - input_tokens
            finish_reason = "stop"

        self.logger.info(
            "Local LLM generation completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_type=self._model_type,
        )

        return LLMResponse(
            content=content,
            model=self._model_type or "local",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            finish_reason=finish_reason,
            metadata={"backend": self._backend},
        )

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
        Generate a streaming response using the local LLM.

        Args:
            messages: List of conversation messages
            model: Ignored for local LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **kwargs: Additional parameters

        Yields:
            StreamChunk objects with content
        """
        # Load model lazily
        llm = self._load_model()

        # Format messages into prompt
        if self._model_type in ("llama", "llama2", "llama3"):
            prompt = self._format_messages_llama(messages)
        else:
            prompt = self._format_messages(messages)

        if self._backend == "llama-cpp":
            # llama-cpp supports streaming natively
            loop = asyncio.get_event_loop()

            def stream_generator():
                return llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_sequences or [],
                    echo=False,
                    stream=True,
                )

            # Get the generator in executor
            stream = await loop.run_in_executor(None, stream_generator)

            # Iterate over stream chunks
            for chunk in stream:
                text = chunk["choices"][0].get("text", "")
                is_final = chunk["choices"][0].get("finish_reason") is not None

                yield StreamChunk(
                    content=text,
                    is_final=is_final,
                    metadata={"backend": self._backend},
                )

        else:
            # For transformers, generate full response and yield in chunks
            # (true streaming with transformers requires TextIteratorStreamer)
            response = await self.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                **kwargs,
            )

            # Yield content in chunks
            chunk_size = 50
            content = response.content
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i : i + chunk_size]
                is_final = i + chunk_size >= len(content)
                yield StreamChunk(
                    content=chunk_text,
                    is_final=is_final,
                    metadata={"backend": self._backend},
                )


# Singleton instance
_local_client: Optional[LocalLLMClient] = None


def get_local_llm_client() -> LocalLLMClient:
    """Get the global local LLM client instance."""
    global _local_client
    if _local_client is None:
        _local_client = LocalLLMClient()
    return _local_client


def reset_local_llm_client() -> None:
    """Reset the local LLM client. Used in testing."""
    global _local_client
    _local_client = None
