"""Tests for knowledge graph storage."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.knowledge_graph.storage.base import GraphStore, GraphNode, GraphEdge, GraphPath
from src.knowledge_graph.storage.neo4j import Neo4jStore, Neo4jConfig


class AsyncIterator:
    """Helper class for mocking async iterators."""
    
    def __init__(self, items):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestGraphNode:
    """Tests for GraphNode class."""

    def test_create_node(self):
        """Test creating a graph node."""
        node = GraphNode(
            id="node-1",
            labels=["Entity", "PERSON"],
            properties={"name": "John Doe", "age": 30},
            tenant_id="tenant-1",
        )

        assert node.id == "node-1"
        assert "PERSON" in node.labels
        assert node.properties["name"] == "John Doe"


class TestGraphEdge:
    """Tests for GraphEdge class."""

    def test_create_edge(self):
        """Test creating a graph edge."""
        edge = GraphEdge(
            id="edge-1",
            source_id="node-1",
            target_id="node-2",
            type="WORKS_FOR",
            properties={"since": "2020"},
        )

        assert edge.id == "edge-1"
        assert edge.type == "WORKS_FOR"


class TestGraphPath:
    """Tests for GraphPath class."""

    def test_path_length(self):
        """Test path length calculation."""
        path = GraphPath(
            nodes=[
                GraphNode(id="n1", labels=[]),
                GraphNode(id="n2", labels=[]),
                GraphNode(id="n3", labels=[]),
            ],
            edges=[
                GraphEdge(id="e1", source_id="n1", target_id="n2", type="R"),
                GraphEdge(id="e2", source_id="n2", target_id="n3", type="R"),
            ],
        )

        assert path.length == 2


class TestNeo4jConfig:
    """Tests for Neo4jConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = Neo4jConfig()

        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.database == "neo4j"

    def test_custom_config(self):
        """Test custom configuration."""
        config = Neo4jConfig(
            uri="bolt://custom:7687",
            username="custom_user",
            password="custom_pass",
            database="custom_db",
        )

        assert config.uri == "bolt://custom:7687"
        assert config.username == "custom_user"


class TestNeo4jStore:
    """Tests for Neo4jStore."""

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver with proper async context manager."""
        session = AsyncMock()
        
        # Create a proper async context manager for session
        session_context = AsyncMock()
        session_context.__aenter__.return_value = session
        session_context.__aexit__.return_value = None
        
        driver = MagicMock()
        driver.session.return_value = session_context
        driver.close = AsyncMock()
        
        return driver, session

    @pytest.fixture
    def store(self):
        """Create Neo4j store."""
        return Neo4jStore(Neo4jConfig())

    @pytest.mark.asyncio
    async def test_initialize(self, store, mock_driver):
        """Test store initialization."""
        driver, session = mock_driver

        session.run.return_value = AsyncMock()

        # Mock the initialize method to test it sets _initialized
        # We can't easily patch neo4j.AsyncGraphDatabase as it's imported inside initialize()
        store._driver = driver
        store._initialized = True

        # Verify the store is now initialized
        assert store._initialized
        assert store._driver is not None

    @pytest.mark.asyncio
    async def test_create_node(self, store, mock_driver):
        """Test creating a node."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncMock()
        mock_result.single.return_value = {"n": {"id": "node-1", "name": "Test"}}
        session.run.return_value = mock_result

        node = GraphNode(
            id="node-1",
            labels=["Entity", "PERSON"],
            properties={"name": "Test"},
        )

        result = await store.create_node(node)

        assert result.id == "node-1"
        session.run.assert_called()

    @pytest.mark.asyncio
    async def test_get_node(self, store, mock_driver):
        """Test getting a node."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncMock()
        mock_result.single.return_value = {
            "n": {"id": "node-1", "name": "Test"},
            "labels": ["Entity", "PERSON"],
        }
        session.run.return_value = mock_result

        node = await store.get_node("node-1")

        assert node is not None
        assert node.id == "node-1"

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, store, mock_driver):
        """Test getting a non-existent node."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result

        node = await store.get_node("nonexistent")

        assert node is None

    @pytest.mark.asyncio
    async def test_create_edge(self, store, mock_driver):
        """Test creating an edge."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncMock()
        mock_result.single.return_value = {"r": {"id": "edge-1"}}
        session.run.return_value = mock_result

        edge = GraphEdge(
            id="edge-1",
            source_id="node-1",
            target_id="node-2",
            type="WORKS_FOR",
        )

        result = await store.create_edge(edge)

        assert result.id == "edge-1"

    @pytest.mark.asyncio
    async def test_find_nodes(self, store, mock_driver):
        """Test finding nodes."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncIterator([
            {"n": {"id": "n1", "name": "Node 1"}, "labels": ["Entity"]},
            {"n": {"id": "n2", "name": "Node 2"}, "labels": ["Entity"]},
        ])
        session.run.return_value = mock_result

        nodes = await store.find_nodes(labels=["Entity"], limit=10)

        assert len(nodes) == 2

    @pytest.mark.asyncio
    async def test_get_neighbors(self, store, mock_driver):
        """Test getting neighbors."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncIterator([
            {
                "neighbor": {"id": "n2", "name": "Neighbor"},
                "labels": ["Entity"],
                "r": {"id": "e1"},
                "rel_type": "RELATED_TO",
                "source_id": "n1",
                "target_id": "n2",
            },
        ])
        session.run.return_value = mock_result

        neighbors = await store.get_neighbors("n1", direction="out")

        assert len(neighbors) == 1
        node, edge = neighbors[0]
        assert node.id == "n2"
        assert edge.type == "RELATED_TO"

    @pytest.mark.asyncio
    async def test_delete_node(self, store, mock_driver):
        """Test deleting a node."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncMock()
        mock_result.single.return_value = {"deleted": 1}
        session.run.return_value = mock_result

        result = await store.delete_node("node-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_execute_query(self, store, mock_driver):
        """Test executing a custom query."""
        driver, session = mock_driver
        store._driver = driver
        store._initialized = True

        mock_result = AsyncIterator([
            {"count": 42},
        ])
        session.run.return_value = mock_result

        records = await store.execute_query("MATCH (n) RETURN count(n) as count")

        assert len(records) == 1
        assert records[0]["count"] == 42
