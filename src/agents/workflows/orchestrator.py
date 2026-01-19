"""
Enterprise RAG System - Orchestrator

The Orchestrator coordinates all agents and manages the
execution flow for RAG queries.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.verifier import VerifierAgent
from src.agents.critic import CriticAgent, CriticDecision
from src.agents.citation import CitationAgent
from src.agents.formatter import FormatterAgent, OutputFormat
from src.core.logging import get_logger
from src.core.types import AgentState, AgentType
from src.core.exceptions import QueryTimeoutError


logger = get_logger(__name__)


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""
    max_iterations: int = 3
    enable_verification: bool = True
    enable_critic: bool = True
    output_format: OutputFormat = OutputFormat.MARKDOWN
    timeout_seconds: int = 120


class ExecutionTrace(BaseModel):
    """Trace of agent execution."""
    trace_id: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    final_status: str = "pending"


class Orchestrator:
    """
    Central orchestrator for the multi-agent RAG system.

    The orchestrator:
    1. Creates and manages agent instances
    2. Coordinates execution flow
    3. Handles iteration and refinement loops
    4. Manages state transitions
    5. Produces execution traces
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.logger = logger

        # Initialize agents
        self._init_agents()

    def _init_agents(self) -> None:
        """Initialize all agent instances."""
        self.planner = PlannerAgent()
        self.retriever = RetrieverAgent()
        self.synthesizer = SynthesizerAgent()
        self.verifier = VerifierAgent()
        self.critic = CriticAgent()
        self.citation = CitationAgent()
        self.formatter = FormatterAgent()

    async def execute(
        self,
        query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute a query through the multi-agent system.

        Args:
            query: The user's query
            conversation_history: Previous conversation turns
            **kwargs: Additional execution parameters

        Returns:
            Dictionary containing response and execution trace
        """
        start_time = time.time()
        trace_id = str(uuid4())

        # Initialize state
        state = AgentState(
            trace_id=trace_id,
            original_query=query,
            conversation_history=conversation_history or [],
            token_budget_remaining=kwargs.get("token_budget", 8000),
        )

        # Initialize trace
        trace = ExecutionTrace(trace_id=trace_id)

        try:
            # Execute pipeline with timeout
            result = await asyncio.wait_for(
                self._execute_pipeline(state, trace, start_time),
                timeout=self.config.timeout_seconds
            )
            return result

        except asyncio.TimeoutError:
            trace.final_status = "timeout"
            trace.total_latency_ms = (time.time() - start_time) * 1000

            self.logger.error(
                "Query execution timed out",
                trace_id=trace_id,
                timeout_seconds=self.config.timeout_seconds,
            )

            raise QueryTimeoutError(
                timeout_seconds=self.config.timeout_seconds,
                trace_id=trace_id
            )

        except Exception as e:
            trace.final_status = "error"
            trace.total_latency_ms = (time.time() - start_time) * 1000

            self.logger.error(
                "Query execution failed",
                trace_id=trace_id,
                error=str(e),
            )

            return {
                "response": "I apologize, but I encountered an error processing your request.",
                "citations": [],
                "trace": trace.model_dump(),
                "error": str(e),
                "metadata": {
                    "trace_id": trace_id,
                    "latency_ms": trace.total_latency_ms,
                },
            }

    async def _execute_pipeline(
        self,
        state: AgentState,
        trace: ExecutionTrace,
        start_time: float,
    ) -> dict[str, Any]:
        """Execute the main query pipeline."""
        # Step 1: Planning
        state, trace = await self._execute_planner(state, trace)

        # Step 2: Retrieval
        state, trace = await self._execute_retriever(state, trace)

        # Iteration loop
        iteration = 0
        while iteration < self.config.max_iterations:
            state.iteration_count = iteration

            # Step 3: Synthesis
            state, trace = await self._execute_synthesizer(state, trace)

            # Step 4: Verification (optional)
            if self.config.enable_verification:
                state, trace = await self._execute_verifier(state, trace)

            # Step 5: Critic evaluation (optional)
            if self.config.enable_critic:
                state, trace = await self._execute_critic(state, trace)

                # Check critic decision
                if state.critic_feedback:
                    decision = state.critic_feedback[-1].get("decision", CriticDecision.PASS)

                    if decision == CriticDecision.PASS:
                        break
                    elif decision == CriticDecision.RETRIEVAL_NEEDED:
                        # Re-retrieve with different strategy
                        state, trace = await self._execute_retriever(
                            state, trace, expand_search=True
                        )
                    elif decision == CriticDecision.REJECT:
                        # Cannot provide good response
                        break
                    # MINOR_REVISION and MAJOR_REVISION continue to next iteration
            else:
                # No critic, break after first synthesis
                break

            iteration += 1

        # Step 6: Citations
        state, trace = await self._execute_citation(state, trace)

        # Step 7: Formatting
        state, trace = await self._execute_formatter(
            state, trace, format=self.config.output_format
        )

        # Prepare result
        total_latency = (time.time() - start_time) * 1000
        trace.total_latency_ms = total_latency
        trace.final_status = "success"

        self.logger.info(
            "Query execution complete",
            trace_id=state.trace_id,
            iterations=iteration + 1,
            latency_ms=total_latency,
        )

        return {
            "response": self._get_final_response(state),
            "citations": self._get_citations(state),
            "trace": trace.model_dump(),
            "metadata": {
                "trace_id": state.trace_id,
                "iterations": iteration + 1,
                "latency_ms": total_latency,
            },
        }

    async def _execute_planner(
        self,
        state: AgentState,
        trace: ExecutionTrace,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the planner agent."""
        step_start = time.time()

        result = await self.planner.execute(state)

        if result.success and result.output:
            state.execution_plan = result.output.model_dump() if hasattr(result.output, 'model_dump') else result.output

        trace.steps.append({
            "agent": "planner",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": self._summarize_output(result.output) if result.success else None,
            "error": result.error,
        })

        return state, trace

    async def _execute_retriever(
        self,
        state: AgentState,
        trace: ExecutionTrace,
        expand_search: bool = False,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the retriever agent."""
        result = await self.retriever.execute(
            state,
            expand_search=expand_search,
        )

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict) and "context" in output:
                state.retrieved_context = output["context"]

        trace.steps.append({
            "agent": "retriever",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": f"{len(state.retrieved_context)} context items" if result.success else None,
            "error": result.error,
        })

        return state, trace

    async def _execute_synthesizer(
        self,
        state: AgentState,
        trace: ExecutionTrace,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the synthesizer agent."""
        result = await self.synthesizer.execute(state)

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict) and "response" in output:
                state.draft_responses.append(output["response"])
            elif isinstance(output, str):
                state.draft_responses.append(output)

        trace.steps.append({
            "agent": "synthesizer",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": f"Response generated ({len(state.draft_responses[-1])} chars)" if result.success and state.draft_responses else None,
            "error": result.error,
        })

        return state, trace

    async def _execute_verifier(
        self,
        state: AgentState,
        trace: ExecutionTrace,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the verifier agent."""
        result = await self.verifier.execute(state)

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict):
                state.verification_results = output.get("results", [])

        summary = None
        if result.success and result.output:
            output = result.output
            if isinstance(output, dict) and "summary" in output:
                s = output["summary"]
                summary = f"Verified: {s.get('supported', 0)} supported, {s.get('contradicted', 0)} contradicted"

        trace.steps.append({
            "agent": "verifier",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": summary,
            "error": result.error,
        })

        return state, trace

    async def _execute_critic(
        self,
        state: AgentState,
        trace: ExecutionTrace,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the critic agent."""
        result = await self.critic.execute(state)

        summary = None
        if result.success and result.output:
            feedback = result.output
            if hasattr(feedback, 'overall_score'):
                summary = f"Score: {feedback.overall_score:.2f}, Decision: {feedback.decision}"
            elif isinstance(feedback, dict):
                summary = f"Score: {feedback.get('overall_score', 0):.2f}, Decision: {feedback.get('decision', 'unknown')}"

        trace.steps.append({
            "agent": "critic",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": summary,
            "error": result.error,
        })

        return state, trace

    async def _execute_citation(
        self,
        state: AgentState,
        trace: ExecutionTrace,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the citation agent."""
        result = await self.citation.execute(state)

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict):
                state.metadata = state.metadata if hasattr(state, 'metadata') else {}
                if not hasattr(state, 'metadata') or state.metadata is None:
                    # Add citations to state - need to store in a different way
                    pass

        trace.steps.append({
            "agent": "citation",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": f"{result.output.get('citation_count', 0)} citations" if result.success and isinstance(result.output, dict) else None,
            "error": result.error,
        })

        return state, trace

    async def _execute_formatter(
        self,
        state: AgentState,
        trace: ExecutionTrace,
        **kwargs,
    ) -> tuple[AgentState, ExecutionTrace]:
        """Execute the formatter agent."""
        result = await self.formatter.execute(state, **kwargs)

        if result.success and result.output:
            output = result.output
            if isinstance(output, dict) and "formatted_response" in output:
                state.draft_responses.append(output["formatted_response"])

        trace.steps.append({
            "agent": "formatter",
            "success": result.success,
            "latency_ms": result.latency_ms,
            "output_summary": f"Formatted as {kwargs.get('format', 'markdown')}" if result.success else None,
            "error": result.error,
        })

        return state, trace

    def _get_final_response(self, state: AgentState) -> str:
        """Get the final response from state."""
        if state.draft_responses:
            return state.draft_responses[-1]
        return "Unable to generate a response."

    def _get_citations(self, state: AgentState) -> list[dict]:
        """Get citations from state."""
        # Extract from context items
        citations = []
        seen_sources = set()

        for item in state.retrieved_context:
            if item.source not in seen_sources:
                citations.append({
                    "source": item.source,
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                })
                seen_sources.add(item.source)

        return citations

    def _summarize_output(self, output: Any) -> str:
        """Create a brief summary of agent output."""
        if output is None:
            return "No output"

        if hasattr(output, 'model_dump'):
            output = output.model_dump()

        if isinstance(output, dict):
            keys = list(output.keys())[:3]
            return f"Dict with keys: {keys}"
        elif isinstance(output, list):
            return f"List with {len(output)} items"
        elif isinstance(output, str):
            return f"String ({len(output)} chars)"
        else:
            return str(type(output).__name__)


class SimpleOrchestrator(Orchestrator):
    """
    Simplified orchestrator for basic queries.

    Skips verification and critic steps for faster execution.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        config = config or OrchestratorConfig(
            max_iterations=1,
            enable_verification=False,
            enable_critic=False,
        )
        super().__init__(config)


class StreamingOrchestrator(Orchestrator):
    """
    Orchestrator with streaming support.

    Yields response chunks as they are generated.
    """

    async def execute_streaming(
        self,
        query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ):
        """
        Execute query with streaming response.

        Yields:
            Response chunks as they are generated
        """
        trace_id = str(uuid4())

        # Initialize state
        state = AgentState(
            trace_id=trace_id,
            original_query=query,
            conversation_history=conversation_history or [],
        )

        # Planning
        yield {"type": "status", "message": "Planning query..."}
        await self.planner.execute(state)

        # Retrieval
        yield {"type": "status", "message": "Retrieving context..."}
        retrieval_result = await self.retriever.execute(state)
        if retrieval_result.success and retrieval_result.output:
            if isinstance(retrieval_result.output, dict):
                state.retrieved_context = retrieval_result.output.get("context", [])

        # Synthesis with streaming
        yield {"type": "status", "message": "Generating response..."}

        # Use streaming synthesis
        async for chunk in self.synthesizer.execute_streaming(state):
            yield {"type": "content", "chunk": chunk}

        # Final status
        yield {"type": "done", "trace_id": trace_id}
