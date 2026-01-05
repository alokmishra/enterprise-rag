"""Tests for knowledge graph RAG integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.knowledge_graph.rag import (
    KnowledgeGraphRAG,
    GraphEnhancedRetriever,
    GraphContext,
)
from src.knowledge_graph.extraction.entities import Entity, EntityType
from src.knowledge_graph.extraction.pipeline import ExtractionResult
from src.knowledge_graph.storage.base import GraphNode


class TestGraphContext:
    """Tests for GraphContext class."""

    def test_create_context(self):
        """Test creating graph context."""
        context = GraphContext(
            entities=[
                Entity(id="e1", name="Apple", type=EntityType.ORGANIZATION),
            ],
            context_text="Apple is a technology company.",
            graph_triples=[("Apple", "PRODUCES", "iPhone")],
            relevance_score=0.85,
        )

        assert len(context.entities) == 1
        assert context.relevance_score == 0.85
        assert len(context.graph_triples) == 1


class TestKnowledgeGraphRAG:
    """Tests for KnowledgeGraphRAG."""

    @pytest.fixture
    def mock_store(self):
        """Create mock Neo4j store."""
        store = AsyncMock()
        store.initialize = AsyncMock()
        store.store_extraction_result = AsyncMock(return_value={"nodes_created": 5})
        store.get_neighbors = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock extraction pipeline."""
        pipeline = AsyncMock()
        pipeline.initialize = AsyncMock()
        pipeline.extract = AsyncMock(return_value=ExtractionResult(
            entities=[
                Entity(id="e1", name="Test Entity", type=EntityType.ORGANIZATION),
            ],
            relationships=[],
        ))
        return pipeline

    @pytest.fixture
    def kg_rag(self, mock_store, mock_pipeline):
        """Create KnowledgeGraphRAG instance."""
        rag = KnowledgeGraphRAG(
            store=mock_store,
            extraction_pipeline=mock_pipeline,
        )
        return rag

    @pytest.mark.asyncio
    async def test_initialize(self, kg_rag, mock_store, mock_pipeline):
        """Test initialization."""
        await kg_rag.initialize()

        assert kg_rag._initialized
        mock_store.initialize.assert_called_once()
        mock_pipeline.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_document(self, kg_rag, mock_store, mock_pipeline):
        """Test document ingestion."""
        kg_rag._initialized = True

        result = await kg_rag.ingest_document(
            text="Apple Inc. is a technology company.",
            document_id="doc-1",
            tenant_id="tenant-1",
        )

        assert isinstance(result, ExtractionResult)
        mock_pipeline.extract.assert_called_once()
        mock_store.store_extraction_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_for_query(self, kg_rag, mock_store):
        """Test getting context for a query."""
        kg_rag._initialized = True

        with patch.object(kg_rag.searcher, "search") as mock_search:
            mock_search.return_value = MagicMock(
                nodes=[
                    GraphNode(
                        id="n1",
                        labels=["Entity", "ORGANIZATION"],
                        properties={"name": "Apple"},
                    ),
                ],
                edges=[],
            )

            with patch.object(kg_rag.context_builder, "build_context") as mock_build:
                mock_build.return_value = "Apple is a company"

                context = await kg_rag.get_context_for_query(
                    query="What is Apple?",
                    tenant_id="tenant-1",
                )

                assert isinstance(context, GraphContext)
                assert len(context.entities) >= 0

    @pytest.mark.asyncio
    async def test_enhance_retrieval(self, kg_rag, mock_store):
        """Test retrieval enhancement."""
        kg_rag._initialized = True

        vector_results = [
            {"id": "doc-1", "text": "Apple makes iPhones", "score": 0.9},
            {"id": "doc-2", "text": "Google is a search company", "score": 0.8},
        ]

        with patch.object(kg_rag, "get_context_for_query") as mock_context:
            mock_context.return_value = GraphContext(
                entities=[Entity(id="e1", name="Apple", type=EntityType.ORGANIZATION)],
                context_text="Apple is a company",
            )

            enhanced = await kg_rag.enhance_retrieval(
                query="What does Apple make?",
                vector_results=vector_results,
            )

            # Results should have graph context
            assert "graph_context" in enhanced[0]
            assert "related_entities" in enhanced[0]

    @pytest.mark.asyncio
    async def test_answer_with_graph(self, kg_rag, mock_store):
        """Test answering with graph."""
        kg_rag._initialized = True

        with patch.object(kg_rag, "get_context_for_query") as mock_context:
            mock_context.return_value = GraphContext(
                entities=[
                    Entity(id="e1", name="Apple", type=EntityType.ORGANIZATION),
                ],
                context_text="Apple is a company",
                graph_triples=[("Apple", "PRODUCES", "iPhone")],
            )

            result = await kg_rag.answer_with_graph("What is Apple?")

            assert "entities" in result
            assert "relationships" in result
            assert "confidence" in result


class TestGraphEnhancedRetriever:
    """Tests for GraphEnhancedRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = AsyncMock()
        store.search.return_value = [
            MagicMock(id="doc-1", score=0.9, payload={"text": "Result 1"}),
            MagicMock(id="doc-2", score=0.8, payload={"text": "Result 2"}),
        ]
        return store

    @pytest.fixture
    def mock_kg_rag(self):
        """Create mock KnowledgeGraphRAG."""
        kg = AsyncMock()
        kg.get_context_for_query = AsyncMock(return_value=GraphContext(
            entities=[Entity(id="e1", name="Test", type=EntityType.CONCEPT)],
        ))
        return kg

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_kg_rag):
        """Create retriever."""
        return GraphEnhancedRetriever(
            vector_store=mock_vector_store,
            knowledge_graph=mock_kg_rag,
        )

    @pytest.mark.asyncio
    async def test_retrieve(self, retriever, mock_vector_store, mock_kg_rag):
        """Test retrieval."""
        results = await retriever.retrieve(
            query="test query",
            top_k=5,
        )

        assert isinstance(results, list)
        mock_vector_store.search.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_vector_only(self, mock_vector_store):
        """Test retrieval with vector store only."""
        retriever = GraphEnhancedRetriever(
            vector_store=mock_vector_store,
            knowledge_graph=None,
        )

        results = await retriever.retrieve("test", top_k=5)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_retrieve_graph_only(self, mock_kg_rag):
        """Test retrieval with graph only."""
        retriever = GraphEnhancedRetriever(
            vector_store=None,
            knowledge_graph=mock_kg_rag,
        )

        results = await retriever.retrieve("test", top_k=5)

        assert isinstance(results, list)

    def test_merge_results(self, retriever):
        """Test merging results from different sources."""
        vector_results = [
            {"id": "doc-1", "text": "Result 1", "score": 0.9},
        ]
        graph_results = [
            {"id": "doc-1", "text": "Result 1 graph", "score": 0.8},
            {"id": "doc-2", "text": "Result 2", "score": 0.7},
        ]

        merged = retriever._merge_results(vector_results, graph_results)

        # doc-1 should have both scores
        doc1 = next(r for r in merged if r["id"] == "doc-1")
        assert "vector_score" in doc1
        assert "graph_score" in doc1
        assert "combined_score" in doc1

    @pytest.mark.asyncio
    async def test_get_context(self, retriever, mock_kg_rag):
        """Test building context."""
        results = [
            {"id": "doc-1", "text": "Result text here"},
        ]

        context = await retriever.get_context("test query", results)

        assert isinstance(context, str)
        assert "Result text" in context
