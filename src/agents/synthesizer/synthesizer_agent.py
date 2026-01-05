"""
Enterprise RAG System - Synthesizer Agent

The Synthesizer Agent generates coherent responses from retrieved context.
It handles the core response generation using an LLM.
"""

import time
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    ContextItem,
    GeneratedResponse,
)
from src.generation.llm import get_default_llm_client, LLMMessage
from src.generation.prompts import build_rag_prompt


class SynthesizerAgent(BaseAgent):
    """
    Agent responsible for generating responses from context.

    The synthesizer:
    1. Builds prompts from retrieved context
    2. Generates responses using LLM
    3. Incorporates feedback from critic for revisions
    4. Maintains citation references
    """

    REVISION_PROMPT = """You previously generated a response that needs revision based on the following feedback:

Original Response:
{original_response}

Feedback:
{feedback}

Suggestions:
{suggestions}

Context (same as before):
{context}

Original Question: {question}

Please generate an improved response that addresses the feedback. Maintain citations [1], [2], etc."""

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="synthesizer",
            agent_type=AgentType.SYNTHESIZER,
            temperature=0.3,
            max_tokens=4096,
        )
        super().__init__(config)

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Generate a response from the retrieved context.

        Args:
            state: Current agent state with context
            **kwargs: Additional parameters (revision, feedback)

        Returns:
            AgentResult containing generated response
        """
        start_time = time.time()

        try:
            is_revision = kwargs.get("revision", False)

            if is_revision and state.critic_feedback:
                response = await self._generate_revision(state)
            else:
                response = await self._generate_initial(state)

            # Store draft response
            state.draft_responses.append(response.content)

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Response generated",
                is_revision=is_revision,
                content_length=len(response.content),
                tokens_used=response.tokens_used,
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output=response,
                tokens_used=response.tokens_used,
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Synthesis failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            return AgentResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _generate_initial(self, state: AgentState) -> GeneratedResponse:
        """Generate initial response."""
        # Build RAG prompt
        system_prompt, user_prompt = build_rag_prompt(
            question=state.original_query,
            context_items=state.retrieved_context,
            history=state.conversation_history,
        )

        llm = get_default_llm_client()
        response = await llm.generate(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return GeneratedResponse(
            content=response.content,
            citations=[],  # Will be populated by Citation agent
            confidence=None,
            model=response.model,
            tokens_used=response.total_tokens,
        )

    async def _generate_revision(self, state: AgentState) -> GeneratedResponse:
        """Generate revised response based on feedback."""
        # Get latest feedback
        feedback = state.critic_feedback[-1] if state.critic_feedback else {}
        original = state.draft_responses[-1] if state.draft_responses else ""

        # Format context
        context_parts = []
        for i, item in enumerate(state.retrieved_context, 1):
            context_parts.append(f"[{i}] {item.source}: {item.content[:500]}")
        context = "\n\n".join(context_parts)

        # Build revision prompt
        prompt = self.REVISION_PROMPT.format(
            original_response=original,
            feedback=feedback.get("feedback", ""),
            suggestions="\n".join(feedback.get("suggestions", [])),
            context=context,
            question=state.original_query,
        )

        llm = get_default_llm_client()
        response = await llm.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return GeneratedResponse(
            content=response.content,
            citations=[],
            confidence=None,
            model=response.model,
            tokens_used=response.total_tokens,
        )

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have context to synthesize from."""
        return bool(state.retrieved_context)
