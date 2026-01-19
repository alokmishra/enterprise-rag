"""Base classes for graph storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    id: str
    labels: list[str]  # Node labels (types)
    properties: dict[str, Any] = field(default_factory=dict)

    # Metadata
    tenant_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class GraphEdge:
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: str  # Relationship type
    properties: dict[str, Any] = field(default_factory=dict)

    # Metadata
    tenant_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class GraphPath:
    """A path in the graph."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    @property
    def length(self) -> int:
        return len(self.edges)


class GraphStore(ABC):
    """Abstract base class for graph storage."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the graph store."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        pass

    # Node operations
    @abstractmethod
    async def create_node(
        self,
        node: GraphNode,
        **kwargs,
    ) -> GraphNode:
        """Create a node."""
        pass

    @abstractmethod
    async def get_node(
        self,
        node_id: str,
        **kwargs,
    ) -> Optional[GraphNode]:
        """Get a node by ID."""
        pass

    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        **kwargs,
    ) -> Optional[GraphNode]:
        """Update node properties."""
        pass

    @abstractmethod
    async def delete_node(
        self,
        node_id: str,
        **kwargs,
    ) -> bool:
        """Delete a node."""
        pass

    @abstractmethod
    async def find_nodes(
        self,
        labels: Optional[list[str]] = None,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
        **kwargs,
    ) -> list[GraphNode]:
        """Find nodes by labels and properties."""
        pass

    # Edge operations
    @abstractmethod
    async def create_edge(
        self,
        edge: GraphEdge,
        **kwargs,
    ) -> GraphEdge:
        """Create an edge."""
        pass

    @abstractmethod
    async def get_edge(
        self,
        edge_id: str,
        **kwargs,
    ) -> Optional[GraphEdge]:
        """Get an edge by ID."""
        pass

    @abstractmethod
    async def delete_edge(
        self,
        edge_id: str,
        **kwargs,
    ) -> bool:
        """Delete an edge."""
        pass

    @abstractmethod
    async def find_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> list[GraphEdge]:
        """Find edges by criteria."""
        pass

    # Graph operations
    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        direction: str = "both",  # in, out, both
        limit: int = 100,
        **kwargs,
    ) -> list[tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes."""
        pass

    @abstractmethod
    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        edge_types: Optional[list[str]] = None,
        **kwargs,
    ) -> Optional[GraphPath]:
        """Find shortest path between nodes."""
        pass

    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        **kwargs,
    ) -> list[dict]:
        """Execute a native query."""
        pass
