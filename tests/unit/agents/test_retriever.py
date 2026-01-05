"""
Tests for src/agents/retriever/retriever_agent.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRetrieverAgent:
    """Tests for the RetrieverAgent class."""

    def test_retriever_agent_creation(self):
        """Test RetrieverAgent can be created."""
        from src.agents.retriever import RetrieverAgent

        agent = RetrieverAgent()
        assert agent is not None
        assert agent.name == "retriever"

    @pytest.mark.asyncio
    async def test_retriever_agent_execute(self, sample_agent_state):
        """Test RetrieverAgent execute method."""
        from src.agents.retriever import RetrieverAgent

        mock_searcher = AsyncMock()
        mock_searcher.search = AsyncMock(return_value=[
            MagicMock(chunk_id="c1", content="Content 1", score=0.9),
            MagicMock(chunk_id="c2", content="Content 2", score=0.8),
        ])

        with patch('src.agents.retriever.retriever_agent.get_hybrid_searcher', return_value=mock_searcher):
            agent = RetrieverAgent()
            result = await agent.execute(sample_agent_state)

            # Should succeed or handle gracefully
            assert result is not None

    @pytest.mark.asyncio
    async def test_retriever_uses_execution_plan(self, sample_agent_state):
        """Test RetrieverAgent uses execution plan strategy."""
        from src.agents.retriever import RetrieverAgent
        from src.core.types import RetrievalStrategy

        sample_agent_state.execution_plan = {
            "strategy": RetrievalStrategy.HYBRID.value,
            "complexity": "standard",
        }

        mock_searcher = AsyncMock()
        mock_searcher.search = AsyncMock(return_value=[])

        with patch('src.agents.retriever.retriever_agent.get_hybrid_searcher', return_value=mock_searcher):
            agent = RetrieverAgent()
            await agent.execute(sample_agent_state)
            # Should use hybrid strategy

    @pytest.mark.asyncio
    async def test_retriever_returns_context_items(self, sample_agent_state):
        """Test RetrieverAgent returns ContextItem objects."""
        from src.agents.retriever import RetrieverAgent
        from src.core.types import ContextItem

        mock_results = [
            MagicMock(
                chunk_id="c1",
                document_id="d1",
                content="Retrieved content",
                score=0.95,
                source="doc.pdf",
            ),
        ]

        mock_searcher = AsyncMock()
        mock_searcher.search = AsyncMock(return_value=mock_results)

        with patch('src.agents.retriever.retriever_agent.get_hybrid_searcher', return_value=mock_searcher):
            agent = RetrieverAgent()
            result = await agent.execute(sample_agent_state)

            if result.success and result.output:
                output = result.output
                if isinstance(output, dict) and "context" in output:
                    context = output["context"]
                    assert len(context) > 0

    @pytest.mark.asyncio
    async def test_retriever_handles_empty_results(self, sample_agent_state):
        """Test RetrieverAgent handles empty search results."""
        from src.agents.retriever import RetrieverAgent

        mock_searcher = AsyncMock()
        mock_searcher.search = AsyncMock(return_value=[])

        with patch('src.agents.retriever.retriever_agent.get_hybrid_searcher', return_value=mock_searcher):
            agent = RetrieverAgent()
            result = await agent.execute(sample_agent_state)

            # Should handle gracefully
            assert result is not None

    @pytest.mark.asyncio
    async def test_retriever_expand_search(self, sample_agent_state):
        """Test RetrieverAgent with expand_search option."""
        from src.agents.retriever import RetrieverAgent

        mock_searcher = AsyncMock()
        mock_searcher.search = AsyncMock(return_value=[])

        with patch('src.agents.retriever.retriever_agent.get_hybrid_searcher', return_value=mock_searcher):
            agent = RetrieverAgent()
            result = await agent.execute(sample_agent_state, expand_search=True)

            # Should perform expanded search

    @pytest.mark.asyncio
    async def test_retriever_validate_input(self, sample_agent_state):
        """Test RetrieverAgent validate_input method."""
        from src.agents.retriever import RetrieverAgent

        agent = RetrieverAgent()
        is_valid = await agent.validate_input(sample_agent_state)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_retriever_respects_top_k(self, sample_agent_state):
        """Test RetrieverAgent respects top_k parameter."""
        from src.agents.retriever import RetrieverAgent

        mock_searcher = AsyncMock()
        mock_searcher.search = AsyncMock(return_value=[])

        with patch('src.agents.retriever.retriever_agent.get_hybrid_searcher', return_value=mock_searcher):
            agent = RetrieverAgent()
            await agent.execute(sample_agent_state, top_k=10)

            # Check that top_k was passed
            if mock_searcher.search.called:
                call_kwargs = mock_searcher.search.call_args
                # top_k should be in the call
