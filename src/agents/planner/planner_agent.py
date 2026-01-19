"""
Enterprise RAG System - Planner Agent

The Planner Agent analyzes incoming queries and creates execution plans.
It determines query complexity, required retrieval strategies, and
orchestrates the overall response generation approach.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    QueryComplexity,
    RetrievalStrategy,
)
from src.generation.llm import get_default_llm_client, LLMMessage


class ExecutionPlan:
    """Execution plan for processing a query."""

    def __init__(
        self,
        complexity: QueryComplexity,
        retrieval_strategy: RetrievalStrategy,
        requires_decomposition: bool = False,
        sub_queries: Optional[list[str]] = None,
        requires_verification: bool = True,
        requires_critic: bool = True,
        estimated_steps: int = 1,
        reasoning: str = "",
    ):
        self.complexity = complexity
        self.retrieval_strategy = retrieval_strategy
        self.requires_decomposition = requires_decomposition
        self.sub_queries = sub_queries or []
        self.requires_verification = requires_verification
        self.requires_critic = requires_critic
        self.estimated_steps = estimated_steps
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "retrieval_strategy": self.retrieval_strategy.value,
            "requires_decomposition": self.requires_decomposition,
            "sub_queries": self.sub_queries,
            "requires_verification": self.requires_verification,
            "requires_critic": self.requires_critic,
            "estimated_steps": self.estimated_steps,
            "reasoning": self.reasoning,
        }


class PlannerAgent(BaseAgent):
    """
    Agent responsible for analyzing queries and creating execution plans.

    The planner:
    1. Classifies query complexity (simple, standard, complex)
    2. Determines the best retrieval strategy
    3. Decomposes complex queries into sub-queries
    4. Decides which agents need to be involved
    """

    PLANNING_PROMPT = """You are a query planning assistant for a RAG system. Analyze the following query and create an execution plan.

Query: {query}

Conversation History:
{history}

Analyze the query and provide a plan in JSON format:
{{
    "complexity": "simple" | "standard" | "complex",
    "retrieval_strategy": "vector" | "hybrid" | "multi_query" | "hyde",
    "requires_decomposition": true | false,
    "sub_queries": ["sub-query 1", "sub-query 2"] (if decomposition needed),
    "requires_verification": true | false,
    "requires_critic": true | false,
    "estimated_steps": 1-5,
    "reasoning": "Brief explanation of your analysis"
}}

Guidelines:
- "simple": Direct factual questions with single answers (e.g., "What is X?")
- "standard": Questions requiring synthesis from multiple sources
- "complex": Multi-hop reasoning, comparisons, or analysis questions

- "vector": Best for semantic similarity searches
- "hybrid": Combines keyword and semantic search, good for specific terms
- "multi_query": Generate multiple query variations
- "hyde": Generate hypothetical answer for better retrieval

- Set requires_decomposition=true for complex multi-part questions
- Set requires_verification=true for factual claims that should be checked
- Set requires_critic=true for responses that need quality evaluation

Return ONLY the JSON object, no other text."""

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="planner",
            agent_type=AgentType.PLANNER,
            temperature=0.0,
            max_tokens=1000,
        )
        super().__init__(config)

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Analyze the query and create an execution plan.

        Args:
            state: Current agent state with query

        Returns:
            AgentResult containing the ExecutionPlan
        """
        import time
        start_time = time.time()

        try:
            plan = await self._create_plan(state)

            # Update state with execution plan
            state.execution_plan = plan.to_dict()

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Created execution plan",
                complexity=plan.complexity.value,
                strategy=plan.retrieval_strategy.value,
                decomposition=plan.requires_decomposition,
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output=plan,
                tokens_used=0,  # Will be updated by LLM call
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Planning failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            # Return default plan on failure
            default_plan = ExecutionPlan(
                complexity=QueryComplexity.STANDARD,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                reasoning="Default plan due to planning error",
            )
            return AgentResult(
                success=False,
                output=default_plan,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _create_plan(self, state: AgentState) -> ExecutionPlan:
        """Create execution plan using LLM."""
        # Format conversation history
        history = ""
        if state.conversation_history:
            history_parts = []
            for msg in state.conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]  # Truncate
                history_parts.append(f"{role}: {content}")
            history = "\n".join(history_parts)
        else:
            history = "No previous conversation"

        prompt = self.PLANNING_PROMPT.format(
            query=state.original_query,
            history=history,
        )

        llm = get_default_llm_client()
        response = await llm.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Parse JSON response
        plan_data = self._parse_plan(response.content)

        return ExecutionPlan(
            complexity=QueryComplexity(plan_data.get("complexity", "standard")),
            retrieval_strategy=RetrievalStrategy(
                plan_data.get("retrieval_strategy", "hybrid")
            ),
            requires_decomposition=plan_data.get("requires_decomposition", False),
            sub_queries=plan_data.get("sub_queries", []),
            requires_verification=plan_data.get("requires_verification", True),
            requires_critic=plan_data.get("requires_critic", True),
            estimated_steps=plan_data.get("estimated_steps", 1),
            reasoning=plan_data.get("reasoning", ""),
        )

    def _parse_plan(self, response: str) -> dict[str, Any]:
        """Parse plan JSON from LLM response."""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Return defaults if parsing fails
        return {
            "complexity": "standard",
            "retrieval_strategy": "hybrid",
            "requires_decomposition": False,
            "requires_verification": True,
            "requires_critic": True,
            "estimated_steps": 1,
            "reasoning": "Failed to parse plan",
        }

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have a query to plan for."""
        return bool(state.original_query and len(state.original_query) > 0)
