"""Tests for knowledge graph queries."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.knowledge_graph.query.traversal import (
    GraphTraverser,
    TraversalConfig,
    TraversalResult,
    TraversalStrategy,
)
from src.knowledge_graph.query.search import (
    GraphSearcher,
    GraphSearchQuery,
    GraphSearchResult,
    SearchType,
)
from src.knowledge_graph.query.cypher import CypherQueryBuilder
from src.knowledge_graph.storage.base import GraphNode, GraphEdge


class TestTraversalConfig:
    """Tests for TraversalConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TraversalConfig()

        assert config.strategy == TraversalStrategy.BFS
        assert config.max_depth == 3
        assert config.max_nodes == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = TraversalConfig(
            strategy=TraversalStrategy.DFS,
            max_depth=5,
            max_nodes=50,
            edge_types=["WORKS_FOR", "LOCATED_IN"],
        )

        assert config.strategy == TraversalStrategy.DFS
        assert config.max_depth == 5
        assert len(config.edge_types) == 2


class TestTraversalResult:
    """Tests for TraversalResult."""

    def test_create_result(self):
        """Test creating a traversal result."""
        result = TraversalResult(
            nodes=[
                GraphNode(id="n1", labels=["Entity"]),
                GraphNode(id="n2", labels=["Entity"]),
            ],
            edges=[
                GraphEdge(id="e1", source_id="n1", target_id="n2", type="R"),
            ],
            nodes_visited=5,
            max_depth_reached=2,
        )

        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.nodes_visited == 5

    def test_to_dict(self):
        """Test converting result to dict."""
        result = TraversalResult(
            nodes=[
                GraphNode(id="n1", labels=["Entity"], properties={"name": "Test"}),
            ],
            edges=[],
        )

        data = result.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "statistics" in data

    def test_get_subgraph(self):
        """Test getting subgraph."""
        result = TraversalResult(
            nodes=[GraphNode(id="n1", labels=[])],
            edges=[],
        )

        nodes, edges = result.get_subgraph()

        assert len(nodes) == 1
        assert len(edges) == 0


class TestGraphTraverser:
    """Tests for GraphTraverser."""

    @pytest.fixture
    def mock_store(self):
        """Create mock graph store."""
        store = AsyncMock()
        store.get_node.return_value = GraphNode(id="n1", labels=["Entity"])
        store.get_neighbors.return_value = []
        return store

    @pytest.fixture
    def traverser(self, mock_store):
        """Create graph traverser."""
        return GraphTraverser(mock_store)

    @pytest.mark.asyncio
    async def test_traverse_bfs(self, traverser, mock_store):
        """Test BFS traversal."""
        mock_store.get_neighbors.return_value = [
            (GraphNode(id="n2", labels=["Entity"]), GraphEdge(id="e1", source_id="n1", target_id="n2", type="R")),
        ]

        config = TraversalConfig(
            strategy=TraversalStrategy.BFS,
            max_depth=2,
            max_nodes=10,
        )

        result = await traverser.traverse("n1", config)

        assert result.nodes_visited >= 1
        assert len(result.nodes) >= 1

    @pytest.mark.asyncio
    async def test_traverse_dfs(self, traverser, mock_store):
        """Test DFS traversal."""
        mock_store.get_neighbors.return_value = []

        config = TraversalConfig(
            strategy=TraversalStrategy.DFS,
            max_depth=3,
        )

        result = await traverser.traverse("n1", config)

        assert result.nodes_visited >= 1

    @pytest.mark.asyncio
    async def test_get_neighborhood(self, traverser):
        """Test getting neighborhood."""
        result = await traverser.get_neighborhood("n1", depth=1)

        assert isinstance(result, TraversalResult)


class TestGraphSearchQuery:
    """Tests for GraphSearchQuery."""

    def test_entity_search_query(self):
        """Test creating entity search query."""
        query = GraphSearchQuery(
            text="Apple Inc",
            entity_types=None,
            search_type=SearchType.ENTITY,
            limit=10,
        )

        assert query.text == "Apple Inc"
        assert query.search_type == SearchType.ENTITY

    def test_relationship_search_query(self):
        """Test creating relationship search query."""
        query = GraphSearchQuery(
            source_entity="Tim Cook",
            relationship_types=["WORKS_FOR"],
            search_type=SearchType.RELATIONSHIP,
        )

        assert query.source_entity == "Tim Cook"
        assert "WORKS_FOR" in query.relationship_types


class TestGraphSearchResult:
    """Tests for GraphSearchResult."""

    def test_to_context(self):
        """Test converting to context string."""
        result = GraphSearchResult(
            nodes=[
                GraphNode(
                    id="n1",
                    labels=["PERSON"],
                    properties={"name": "John Doe"},
                ),
            ],
            edges=[],
        )

        context = result.to_context()

        assert "PERSON" in context
        assert "John Doe" in context


class TestGraphSearcher:
    """Tests for GraphSearcher."""

    @pytest.fixture
    def mock_store(self):
        """Create mock graph store."""
        store = AsyncMock()
        store.find_nodes.return_value = [
            GraphNode(id="n1", labels=["Entity"], properties={"name": "Test"}),
        ]
        store.find_edges.return_value = []
        store.search_entities.return_value = []
        return store

    @pytest.fixture
    def searcher(self, mock_store):
        """Create graph searcher."""
        return GraphSearcher(mock_store)

    @pytest.mark.asyncio
    async def test_search_entities(self, searcher):
        """Test entity search."""
        query = GraphSearchQuery(
            entity_names=["Test"],
            search_type=SearchType.ENTITY,
        )

        result = await searcher.search(query)

        assert isinstance(result, GraphSearchResult)

    @pytest.mark.asyncio
    async def test_search_relationships(self, searcher, mock_store):
        """Test relationship search."""
        mock_store.find_edges.return_value = [
            GraphEdge(id="e1", source_id="n1", target_id="n2", type="R"),
        ]
        mock_store.get_node.return_value = GraphNode(id="n1", labels=[])

        query = GraphSearchQuery(
            source_entity="n1",
            search_type=SearchType.RELATIONSHIP,
        )

        result = await searcher.search(query)

        assert isinstance(result, GraphSearchResult)

    @pytest.mark.asyncio
    async def test_find_related_entities(self, searcher, mock_store):
        """Test finding related entities."""
        mock_store.get_neighbors.return_value = [
            (GraphNode(id="n2", labels=[]), GraphEdge(id="e1", source_id="n1", target_id="n2", type="R")),
        ]

        result = await searcher.find_related_entities("n1", max_depth=2)

        assert isinstance(result, GraphSearchResult)


class TestCypherQueryBuilder:
    """Tests for CypherQueryBuilder."""

    def test_simple_match(self):
        """Test simple MATCH query."""
        query, params = (
            CypherQueryBuilder()
            .match("(n:Entity)")
            .return_("n")
            .build()
        )

        assert "MATCH (n:Entity)" in query
        assert "RETURN n" in query

    def test_match_with_where(self):
        """Test MATCH with WHERE clause."""
        query, params = (
            CypherQueryBuilder()
            .match("(n:Entity)")
            .where("n.name = $name", name="Test")
            .return_("n")
            .build()
        )

        assert "WHERE n.name = $name" in query
        assert params["name"] == "Test"

    def test_match_with_limit(self):
        """Test MATCH with LIMIT."""
        query, params = (
            CypherQueryBuilder()
            .match("(n)")
            .return_("n")
            .limit(10)
            .build()
        )

        assert "LIMIT" in query

    def test_create_node(self):
        """Test CREATE query."""
        query, params = (
            CypherQueryBuilder()
            .create("(n:Entity {name: $name})", name="Test")
            .return_("n")
            .build()
        )

        assert "CREATE" in query
        assert params["name"] == "Test"

    def test_merge_node(self):
        """Test MERGE query."""
        query, params = (
            CypherQueryBuilder()
            .merge("(n:Entity {id: $id})", id="123")
            .on_create_set("n.created = datetime()")
            .return_("n")
            .build()
        )

        assert "MERGE" in query
        assert "ON CREATE SET" in query

    def test_find_entity_by_name(self):
        """Test convenience method for finding entity."""
        builder = CypherQueryBuilder.find_entity_by_name("Apple", "ORGANIZATION")
        query, params = builder.build()

        assert "MATCH" in query
        assert "Entity" in query
        assert "ORGANIZATION" in query
        assert params["name"] == "Apple"

    def test_find_path(self):
        """Test convenience method for finding path."""
        builder = CypherQueryBuilder.find_path("Apple", "Tim Cook", max_depth=3)
        query, params = builder.build()

        assert "shortestPath" in query
        assert params["source"] == "Apple"
        assert params["target"] == "Tim Cook"

    def test_search_entities(self):
        """Test convenience method for search."""
        builder = CypherQueryBuilder.search_entities("machine learning", limit=5)
        query, params = builder.build()

        assert "fulltext" in query
        assert params["term"] == "machine learning"

    def test_chained_operations(self):
        """Test chaining multiple operations."""
        query, params = (
            CypherQueryBuilder()
            .match("(n:Entity)")
            .where("n.type = $type", type="PERSON")
            .with_("n")
            .match("(n)-[r]->(m)")
            .return_("n", "r", "m")
            .order_by("n.name")
            .limit(10)
            .build()
        )

        assert "MATCH" in query
        assert "WITH" in query
        assert "ORDER BY" in query
        assert "LIMIT" in query
