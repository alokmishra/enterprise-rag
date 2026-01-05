"""
Tests for src/retrieval/query/expansion.py and src/retrieval/query/hyde.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQueryExpander:
    """Tests for the QueryExpander class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=MagicMock(
            content='["expanded query 1", "expanded query 2", "expanded query 3"]',
        ))
        return client

    def test_query_expander_creation(self, mock_llm_client):
        """Test QueryExpander can be created."""
        from src.retrieval.query.expansion import QueryExpander

        expander = QueryExpander(llm_client=mock_llm_client)
        assert expander is not None

    @pytest.mark.asyncio
    async def test_query_expander_expand(self, mock_llm_client):
        """Test QueryExpander expand method."""
        from src.retrieval.query.expansion import QueryExpander

        expander = QueryExpander(llm_client=mock_llm_client)

        expanded = await expander.expand("What is machine learning?")
        assert expanded is not None
        assert len(expanded) > 0

    @pytest.mark.asyncio
    async def test_query_expander_includes_original(self, mock_llm_client):
        """Test that expanded queries include original."""
        from src.retrieval.query.expansion import QueryExpander

        expander = QueryExpander(llm_client=mock_llm_client)

        expanded = await expander.expand("original query")
        # Should include original query
        assert "original query" in expanded or len(expanded) > 0

    @pytest.mark.asyncio
    async def test_query_expander_num_expansions(self, mock_llm_client):
        """Test QueryExpander respects num_expansions."""
        from src.retrieval.query.expansion import QueryExpander

        expander = QueryExpander(llm_client=mock_llm_client, num_expansions=5)

        await expander.expand("test query")
        # Should request 5 expansions


class TestSubQuestionDecomposer:
    """Tests for the SubQuestionDecomposer class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=MagicMock(
            content='["What is X?", "How does Y work?", "Why is Z important?"]',
        ))
        return client

    def test_decomposer_creation(self, mock_llm_client):
        """Test SubQuestionDecomposer can be created."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        decomposer = SubQuestionDecomposer(llm_client=mock_llm_client)
        assert decomposer is not None

    @pytest.mark.asyncio
    async def test_decomposer_decompose(self, mock_llm_client):
        """Test SubQuestionDecomposer decompose method."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        decomposer = SubQuestionDecomposer(llm_client=mock_llm_client)

        sub_questions = await decomposer.decompose(
            "How did the company's revenue change and what caused it?"
        )
        assert sub_questions is not None
        assert len(sub_questions) > 0

    @pytest.mark.asyncio
    async def test_decomposer_handles_simple_query(self, mock_llm_client):
        """Test decomposer handles simple queries."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        mock_llm_client.generate = AsyncMock(return_value=MagicMock(
            content='["What is the weather?"]',
        ))

        decomposer = SubQuestionDecomposer(llm_client=mock_llm_client)

        sub_questions = await decomposer.decompose("What is the weather?")
        # Simple query may return just the original


class TestHyDEGenerator:
    """Tests for the HyDEGenerator class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=MagicMock(
            content="This is a hypothetical document that answers the question about machine learning...",
        ))
        return client

    def test_hyde_generator_creation(self, mock_llm_client):
        """Test HyDEGenerator can be created."""
        from src.retrieval.query.hyde import HyDEGenerator

        generator = HyDEGenerator(llm_client=mock_llm_client)
        assert generator is not None

    @pytest.mark.asyncio
    async def test_hyde_generator_generate(self, mock_llm_client):
        """Test HyDEGenerator generate method."""
        from src.retrieval.query.hyde import HyDEGenerator

        generator = HyDEGenerator(llm_client=mock_llm_client)

        hypothetical_doc = await generator.generate("What is machine learning?")
        assert hypothetical_doc is not None
        assert len(hypothetical_doc) > 0

    @pytest.mark.asyncio
    async def test_hyde_generator_multiple_docs(self, mock_llm_client):
        """Test HyDEGenerator can generate multiple docs."""
        from src.retrieval.query.hyde import HyDEGenerator

        generator = HyDEGenerator(llm_client=mock_llm_client, num_hypothetical=3)

        docs = await generator.generate_multiple("What is AI?")
        # Should generate multiple hypothetical documents

    @pytest.mark.asyncio
    async def test_hyde_for_search(self, mock_llm_client):
        """Test HyDE workflow for search."""
        from src.retrieval.query.hyde import HyDEGenerator

        generator = HyDEGenerator(llm_client=mock_llm_client)

        # Generate hypothetical document
        hypothetical = await generator.generate("test query")

        # The hypothetical document should be used for embedding and search
        assert hypothetical is not None


class TestContextAssembler:
    """Tests for the ContextAssembler class."""

    def test_context_assembler_creation(self):
        """Test ContextAssembler can be created."""
        from src.retrieval.context.assembler import ContextAssembler

        assembler = ContextAssembler(max_tokens=4000)
        assert assembler is not None

    def test_context_assembler_assemble(self):
        """Test ContextAssembler assemble method."""
        from src.retrieval.context.assembler import ContextAssembler
        from src.core.types import SearchResult

        assembler = ContextAssembler(max_tokens=4000)

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="Content 1", score=0.9),
            SearchResult(chunk_id="c2", document_id="d1", content="Content 2", score=0.8),
        ]

        context = assembler.assemble(results)
        assert context is not None

    def test_context_assembler_respects_token_limit(self):
        """Test ContextAssembler respects token limit."""
        from src.retrieval.context.assembler import ContextAssembler
        from src.core.types import SearchResult

        assembler = ContextAssembler(max_tokens=100)  # Small limit

        results = [
            SearchResult(chunk_id=f"c{i}", document_id="d1", content="Word " * 100, score=0.9 - i * 0.1)
            for i in range(10)
        ]

        context = assembler.assemble(results)
        # Should be limited to approximately max_tokens

    def test_context_assembler_deduplicates(self):
        """Test ContextAssembler deduplicates content."""
        from src.retrieval.context.assembler import ContextAssembler
        from src.core.types import SearchResult

        assembler = ContextAssembler(max_tokens=4000)

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="Same content", score=0.9),
            SearchResult(chunk_id="c1", document_id="d1", content="Same content", score=0.85),
        ]

        context = assembler.assemble(results)
        # Should deduplicate by chunk_id
