"""
Enterprise RAG System - Retriever Agent

The Retriever Agent is responsible for fetching relevant context
from the knowledge base based on the query and execution plan.
"""

import time
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentConfig, AgentResult
from src.core.types import (
    AgentState,
    AgentType,
    ContextItem,
    RetrievalStrategy,
)
from src.retrieval import (
    get_vector_searcher,
    get_hybrid_searcher,
    QueryExpander,
    HyDEGenerator,
    get_context_assembler,
)


class RetrieverAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant context.

    The retriever:
    1. Executes the retrieval strategy from the plan
    2. Handles query expansion and HyDE if needed
    3. Deduplicates and assembles context
    4. Updates the agent state with retrieved context
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="retriever",
            agent_type=AgentType.RETRIEVER,
            timeout_seconds=30,
        )
        super().__init__(config)

        self._vector_searcher = get_vector_searcher()
        self._hybrid_searcher = get_hybrid_searcher()
        self._query_expander = QueryExpander()
        self._hyde_generator = HyDEGenerator()
        self._context_assembler = get_context_assembler()

    async def execute(self, state: AgentState, **kwargs) -> AgentResult:
        """
        Retrieve relevant context based on the execution plan.

        Args:
            state: Current agent state with query and plan
            **kwargs: Additional parameters (top_k, filters)

        Returns:
            AgentResult containing retrieved context items
        """
        start_time = time.time()

        try:
            # Get retrieval parameters from plan
            plan = state.execution_plan or {}
            strategy = RetrievalStrategy(
                plan.get("retrieval_strategy", "hybrid")
            )
            sub_queries = plan.get("sub_queries", [])
            top_k = kwargs.get("top_k", 10)
            filters = kwargs.get("filters")

            # Determine queries to execute
            queries = [state.original_query]
            if sub_queries:
                queries.extend(sub_queries)

            # Execute retrieval
            all_results = []

            for query in queries:
                results = await self._retrieve(
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    filters=filters,
                )
                all_results.extend(results)

            # Deduplicate and assemble context
            assembled = self._context_assembler.assemble(
                results=all_results,
                strategy="relevance",
            )

            # Update state
            state.retrieved_context = assembled.items

            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "Retrieval complete",
                strategy=strategy.value,
                queries_count=len(queries),
                results_count=len(assembled.items),
                trace_id=state.trace_id,
            )

            return AgentResult(
                success=True,
                output={
                    "context_items": assembled.items,
                    "total_retrieved": len(all_results),
                    "final_count": len(assembled.items),
                    "truncated": assembled.truncated,
                },
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.logger.error(
                "Retrieval failed",
                error=str(e),
                trace_id=state.trace_id,
            )
            return AgentResult(
                success=False,
                output={"context_items": [], "error": str(e)},
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list:
        """Execute retrieval with the specified strategy."""
        if strategy == RetrievalStrategy.VECTOR:
            result = await self._vector_searcher.search(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            return result.results

        elif strategy == RetrievalStrategy.HYBRID:
            result = await self._hybrid_searcher.search(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            return result.results

        elif strategy == RetrievalStrategy.MULTI_QUERY:
            # Expand query and search with variations
            variations = await self._query_expander.expand(query, num_variations=3)
            all_results = []

            for var in variations:
                result = await self._hybrid_searcher.search(
                    query=var,
                    top_k=top_k,
                    filters=filters,
                )
                all_results.extend(result.results)

            return all_results

        elif strategy == RetrievalStrategy.HYDE:
            # Generate hypothetical document embedding
            hyde_embedding = await self._hyde_generator.generate_hyde_embedding(query)
            results = await self._vector_searcher.search_by_embedding(
                embedding=hyde_embedding,
                top_k=top_k,
                filters=filters,
            )
            return results

        else:
            # Default to hybrid
            result = await self._hybrid_searcher.search(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            return result.results

    async def validate_input(self, state: AgentState) -> bool:
        """Validate that we have a query to retrieve for."""
        return bool(state.original_query)
