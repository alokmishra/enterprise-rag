"""
Tests for src/retrieval/query/expansion.py and src/retrieval/query/hyde.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import SearchResult


class TestQueryExpander:
    """Tests for the QueryExpander class."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return MagicMock(
            content="expanded query 1\nexpanded query 2\nexpanded query 3",
        )

    def test_query_expander_creation(self):
        """Test QueryExpander can be created."""
        from src.retrieval.query.expansion import QueryExpander

        expander = QueryExpander()
        assert expander is not None
        assert expander.num_variations == 3

    def test_query_expander_creation_with_variations(self):
        """Test QueryExpander with custom num_variations."""
        from src.retrieval.query.expansion import QueryExpander

        expander = QueryExpander(num_variations=5)
        assert expander.num_variations == 5

    @pytest.mark.asyncio
    async def test_query_expander_expand(self, mock_llm_response):
        """Test QueryExpander expand method."""
        from src.retrieval.query.expansion import QueryExpander

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            expander = QueryExpander()
            expanded = await expander.expand("What is machine learning?")

            assert expanded is not None
            assert len(expanded) > 0
            assert "What is machine learning?" in expanded

    @pytest.mark.asyncio
    async def test_query_expander_includes_original(self, mock_llm_response):
        """Test that expanded queries include original."""
        from src.retrieval.query.expansion import QueryExpander

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            expander = QueryExpander()
            expanded = await expander.expand("original query")

            assert expanded[0] == "original query"

    @pytest.mark.asyncio
    async def test_query_expander_num_variations(self, mock_llm_response):
        """Test QueryExpander respects num_variations."""
        from src.retrieval.query.expansion import QueryExpander

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            expander = QueryExpander(num_variations=5)
            await expander.expand("test query")

            call_args = mock_llm.generate.call_args
            assert "5" in str(call_args)

    @pytest.mark.asyncio
    async def test_query_expander_fallback_on_error(self):
        """Test QueryExpander returns original on error."""
        from src.retrieval.query.expansion import QueryExpander

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))
            mock_get_llm.return_value = mock_llm

            expander = QueryExpander()
            expanded = await expander.expand("test query")

            assert expanded == ["test query"]


class TestSubQuestionDecomposer:
    """Tests for the SubQuestionDecomposer class."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return MagicMock(
            content="What is X?\nHow does Y work?\nWhy is Z important?",
        )

    def test_decomposer_creation(self):
        """Test SubQuestionDecomposer can be created."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        decomposer = SubQuestionDecomposer()
        assert decomposer is not None

    @pytest.mark.asyncio
    async def test_decomposer_decompose(self, mock_llm_response):
        """Test SubQuestionDecomposer decompose method."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            decomposer = SubQuestionDecomposer()
            sub_questions = await decomposer.decompose(
                "How did the company's revenue change and what caused it?"
            )

            assert sub_questions is not None
            assert len(sub_questions) > 0

    @pytest.mark.asyncio
    async def test_decomposer_handles_simple_query(self):
        """Test decomposer handles simple queries."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        mock_response = MagicMock(content="What is the weather?")

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            decomposer = SubQuestionDecomposer()
            sub_questions = await decomposer.decompose("What is the weather?")

            assert len(sub_questions) >= 1

    @pytest.mark.asyncio
    async def test_decomposer_fallback_on_error(self):
        """Test decomposer returns original on error."""
        from src.retrieval.query.expansion import SubQuestionDecomposer

        with patch('src.retrieval.query.expansion.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))
            mock_get_llm.return_value = mock_llm

            decomposer = SubQuestionDecomposer()
            sub_questions = await decomposer.decompose("test query")

            assert sub_questions == ["test query"]


class TestHyDEGenerator:
    """Tests for the HyDEGenerator class."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return MagicMock(
            content="This is a hypothetical document that answers the question about machine learning...",
        )

    def test_hyde_generator_creation(self):
        """Test HyDEGenerator can be created."""
        from src.retrieval.query.hyde import HyDEGenerator

        generator = HyDEGenerator()
        assert generator is not None

    @pytest.mark.asyncio
    async def test_hyde_generator_generate_hypothetical_document(self, mock_llm_response):
        """Test HyDEGenerator generate_hypothetical_document method."""
        from src.retrieval.query.hyde import HyDEGenerator

        with patch('src.retrieval.query.hyde.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            generator = HyDEGenerator()
            hypothetical_doc = await generator.generate_hypothetical_document("What is machine learning?")

            assert hypothetical_doc is not None
            assert len(hypothetical_doc) > 0

    @pytest.mark.asyncio
    async def test_hyde_generator_with_domain_context(self, mock_llm_response):
        """Test HyDEGenerator with domain context."""
        from src.retrieval.query.hyde import HyDEGenerator

        with patch('src.retrieval.query.hyde.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            generator = HyDEGenerator()
            hypothetical_doc = await generator.generate_hypothetical_document(
                "What is machine learning?",
                domain_context="AI and Data Science"
            )

            assert hypothetical_doc is not None

    @pytest.mark.asyncio
    async def test_hyde_generator_generate_embedding(self, mock_llm_response):
        """Test HyDEGenerator generate_hyde_embedding method."""
        from src.retrieval.query.hyde import HyDEGenerator

        mock_embedding = [0.1] * 1536

        with patch('src.retrieval.query.hyde.get_default_llm_client') as mock_get_llm, \
             patch('src.retrieval.query.hyde.get_embedding_provider') as mock_get_embed:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            mock_embed = AsyncMock()
            mock_embed.embed_text = AsyncMock(return_value=mock_embedding)
            mock_get_embed.return_value = mock_embed

            generator = HyDEGenerator()
            embedding = await generator.generate_hyde_embedding("What is AI?")

            assert embedding == mock_embedding

    @pytest.mark.asyncio
    async def test_hyde_generator_fallback_on_error(self):
        """Test HyDE falls back to query on error."""
        from src.retrieval.query.hyde import HyDEGenerator

        with patch('src.retrieval.query.hyde.get_default_llm_client') as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))
            mock_get_llm.return_value = mock_llm

            generator = HyDEGenerator()
            hypothetical_doc = await generator.generate_hypothetical_document("test query")

            assert hypothetical_doc == "test query"


class TestContextAssembler:
    """Tests for the ContextAssembler class."""

    def test_context_assembler_creation(self):
        """Test ContextAssembler can be created."""
        from src.retrieval.context.assembler import ContextAssembler

        assembler = ContextAssembler(max_tokens=4000)
        assert assembler is not None
        assert assembler.max_tokens == 4000

    def test_context_assembler_default_values(self):
        """Test ContextAssembler default values."""
        from src.retrieval.context.assembler import ContextAssembler

        assembler = ContextAssembler()
        assert assembler.max_tokens == 8000
        assert assembler.max_items == 20
        assert assembler.deduplicate is True

    def test_context_assembler_assemble(self):
        """Test ContextAssembler assemble method."""
        from src.retrieval.context.assembler import ContextAssembler, AssembledContext

        assembler = ContextAssembler(max_tokens=4000)

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="Content 1", score=0.9, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="Content 2", score=0.8, metadata={}),
        ]

        context = assembler.assemble(results)
        assert isinstance(context, AssembledContext)
        assert len(context.items) > 0

    def test_context_assembler_respects_token_limit(self):
        """Test ContextAssembler respects token limit."""
        from src.retrieval.context.assembler import ContextAssembler

        assembler = ContextAssembler(max_tokens=100)

        results = [
            SearchResult(chunk_id=f"c{i}", document_id="d1", content="Word " * 100, score=0.9 - i * 0.1, metadata={})
            for i in range(10)
        ]

        context = assembler.assemble(results)
        assert context.truncated or context.total_tokens <= 100

    def test_context_assembler_deduplicates(self):
        """Test ContextAssembler deduplicates content."""
        from src.retrieval.context.assembler import ContextAssembler

        assembler = ContextAssembler(max_tokens=4000, deduplicate=True)

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="Same content here", score=0.9, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="Same content here", score=0.85, metadata={}),
        ]

        context = assembler.assemble(results)
        assert len(context.items) == 1

    def test_context_assembler_empty_results(self):
        """Test ContextAssembler with empty results."""
        from src.retrieval.context.assembler import ContextAssembler

        assembler = ContextAssembler()
        context = assembler.assemble([])

        assert len(context.items) == 0
        assert context.total_tokens == 0
        assert context.truncated is False

    def test_context_assembler_group_by_source(self):
        """Test ContextAssembler group_by_source method."""
        from src.retrieval.context.assembler import ContextAssembler
        from src.core.types import ContextItem

        assembler = ContextAssembler()

        items = [
            ContextItem(content="Content 1", source="doc1.pdf", chunk_id="c1", document_id="d1", relevance_score=0.9),
            ContextItem(content="Content 2", source="doc1.pdf", chunk_id="c2", document_id="d1", relevance_score=0.8),
            ContextItem(content="Content 3", source="doc2.pdf", chunk_id="c3", document_id="d2", relevance_score=0.7),
        ]

        groups = assembler.group_by_source(items)
        assert len(groups) == 2
        assert len(groups["doc1.pdf"]) == 2
        assert len(groups["doc2.pdf"]) == 1
