"""Graph traversal algorithms for knowledge graph."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Set
from collections import defaultdict

from src.knowledge_graph.storage.base import GraphStore, GraphNode, GraphEdge, GraphPath

logger = logging.getLogger(__name__)


class TraversalStrategy(str, Enum):
    """Traversal strategies."""
    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    DIJKSTRA = "dijkstra"  # Shortest path
    RANDOM_WALK = "random_walk"


@dataclass
class TraversalConfig:
    """Configuration for graph traversal."""
    strategy: TraversalStrategy = TraversalStrategy.BFS
    max_depth: int = 3
    max_nodes: int = 100
    edge_types: Optional[list[str]] = None
    node_labels: Optional[list[str]] = None
    direction: str = "both"  # in, out, both

    # Filtering
    node_filter: Optional[Callable[[GraphNode], bool]] = None
    edge_filter: Optional[Callable[[GraphEdge], bool]] = None

    # Scoring
    edge_weight_property: Optional[str] = None  # Property to use as edge weight
    default_weight: float = 1.0


@dataclass
class TraversalResult:
    """Result from graph traversal."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)

    # Statistics
    nodes_visited: int = 0
    edges_traversed: int = 0
    max_depth_reached: int = 0

    # Node scores (for algorithms that compute them)
    node_scores: dict[str, float] = field(default_factory=dict)

    def get_subgraph(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Get the traversed subgraph."""
        return self.nodes, self.edges

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "nodes": [
                {"id": n.id, "labels": n.labels, **n.properties}
                for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.type,
                    **e.properties,
                }
                for e in self.edges
            ],
            "statistics": {
                "nodes_visited": self.nodes_visited,
                "edges_traversed": self.edges_traversed,
                "max_depth_reached": self.max_depth_reached,
            },
        }


class GraphTraverser:
    """Traverse knowledge graph using various algorithms."""

    def __init__(
        self,
        store: GraphStore,
        config: Optional[TraversalConfig] = None,
    ):
        self.store = store
        self.config = config or TraversalConfig()

    async def traverse(
        self,
        start_node_id: str,
        config: Optional[TraversalConfig] = None,
    ) -> TraversalResult:
        """Traverse the graph from a starting node."""
        cfg = config or self.config

        if cfg.strategy == TraversalStrategy.BFS:
            return await self._bfs(start_node_id, cfg)
        elif cfg.strategy == TraversalStrategy.DFS:
            return await self._dfs(start_node_id, cfg)
        elif cfg.strategy == TraversalStrategy.DIJKSTRA:
            return await self._dijkstra(start_node_id, cfg)
        elif cfg.strategy == TraversalStrategy.RANDOM_WALK:
            return await self._random_walk(start_node_id, cfg)
        else:
            return await self._bfs(start_node_id, cfg)

    async def _bfs(
        self,
        start_node_id: str,
        config: TraversalConfig,
    ) -> TraversalResult:
        """Breadth-first search traversal."""
        result = TraversalResult()
        visited: Set[str] = set()
        queue: list[tuple[str, int]] = [(start_node_id, 0)]  # (node_id, depth)

        while queue and len(result.nodes) < config.max_nodes:
            node_id, depth = queue.pop(0)

            if node_id in visited:
                continue

            if depth > config.max_depth:
                continue

            visited.add(node_id)
            result.nodes_visited += 1
            result.max_depth_reached = max(result.max_depth_reached, depth)

            # Get node
            node = await self.store.get_node(node_id)
            if not node:
                continue

            # Apply node filter
            if config.node_filter and not config.node_filter(node):
                continue

            # Check label filter
            if config.node_labels:
                if not any(label in node.labels for label in config.node_labels):
                    continue

            result.nodes.append(node)

            # Get neighbors
            neighbors = await self.store.get_neighbors(
                node_id,
                edge_types=config.edge_types,
                direction=config.direction,
                limit=config.max_nodes,
            )

            for neighbor_node, edge in neighbors:
                result.edges_traversed += 1

                # Apply edge filter
                if config.edge_filter and not config.edge_filter(edge):
                    continue

                result.edges.append(edge)

                if neighbor_node.id not in visited:
                    queue.append((neighbor_node.id, depth + 1))

        return result

    async def _dfs(
        self,
        start_node_id: str,
        config: TraversalConfig,
    ) -> TraversalResult:
        """Depth-first search traversal."""
        result = TraversalResult()
        visited: Set[str] = set()

        async def dfs_recursive(node_id: str, depth: int, path: list[str]):
            if node_id in visited:
                return
            if depth > config.max_depth:
                return
            if len(result.nodes) >= config.max_nodes:
                return

            visited.add(node_id)
            result.nodes_visited += 1
            result.max_depth_reached = max(result.max_depth_reached, depth)

            node = await self.store.get_node(node_id)
            if not node:
                return

            if config.node_filter and not config.node_filter(node):
                return

            result.nodes.append(node)

            neighbors = await self.store.get_neighbors(
                node_id,
                edge_types=config.edge_types,
                direction=config.direction,
            )

            for neighbor_node, edge in neighbors:
                result.edges_traversed += 1

                if config.edge_filter and not config.edge_filter(edge):
                    continue

                result.edges.append(edge)

                if neighbor_node.id not in visited:
                    await dfs_recursive(
                        neighbor_node.id,
                        depth + 1,
                        path + [neighbor_node.id],
                    )

        await dfs_recursive(start_node_id, 0, [start_node_id])
        return result

    async def _dijkstra(
        self,
        start_node_id: str,
        config: TraversalConfig,
    ) -> TraversalResult:
        """Dijkstra's algorithm for weighted shortest paths."""
        import heapq

        result = TraversalResult()
        distances: dict[str, float] = {start_node_id: 0}
        previous: dict[str, Optional[str]] = {start_node_id: None}
        visited: Set[str] = set()
        pq = [(0, start_node_id)]  # (distance, node_id)

        while pq and len(result.nodes) < config.max_nodes:
            current_dist, node_id = heapq.heappop(pq)

            if node_id in visited:
                continue

            visited.add(node_id)
            result.nodes_visited += 1

            node = await self.store.get_node(node_id)
            if node:
                result.nodes.append(node)
                result.node_scores[node_id] = current_dist

            neighbors = await self.store.get_neighbors(
                node_id,
                edge_types=config.edge_types,
                direction=config.direction,
            )

            for neighbor_node, edge in neighbors:
                result.edges_traversed += 1

                if neighbor_node.id in visited:
                    continue

                # Get edge weight
                weight = config.default_weight
                if config.edge_weight_property:
                    weight = edge.properties.get(
                        config.edge_weight_property,
                        config.default_weight,
                    )

                new_dist = current_dist + weight

                if neighbor_node.id not in distances or new_dist < distances[neighbor_node.id]:
                    distances[neighbor_node.id] = new_dist
                    previous[neighbor_node.id] = node_id
                    heapq.heappush(pq, (new_dist, neighbor_node.id))

                result.edges.append(edge)

        return result

    async def _random_walk(
        self,
        start_node_id: str,
        config: TraversalConfig,
    ) -> TraversalResult:
        """Random walk traversal."""
        import random

        result = TraversalResult()
        current_id = start_node_id
        path = [current_id]

        for step in range(config.max_nodes):
            result.nodes_visited += 1

            node = await self.store.get_node(current_id)
            if node:
                result.nodes.append(node)

            neighbors = await self.store.get_neighbors(
                current_id,
                edge_types=config.edge_types,
                direction=config.direction,
            )

            if not neighbors:
                break

            # Random selection
            neighbor_node, edge = random.choice(neighbors)
            result.edges.append(edge)
            result.edges_traversed += 1

            current_id = neighbor_node.id
            path.append(current_id)
            result.max_depth_reached = len(path) - 1

        return result

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_paths: int = 5,
        config: Optional[TraversalConfig] = None,
    ) -> list[GraphPath]:
        """Find multiple paths between two nodes."""
        cfg = config or self.config
        paths = []

        # Use store's built-in path finding
        path = await self.store.find_path(
            source_id,
            target_id,
            max_depth=cfg.max_depth,
            edge_types=cfg.edge_types,
        )

        if path:
            paths.append(path)

        return paths

    async def get_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        config: Optional[TraversalConfig] = None,
    ) -> TraversalResult:
        """Get the neighborhood of a node up to specified depth."""
        cfg = config or TraversalConfig(
            strategy=TraversalStrategy.BFS,
            max_depth=depth,
            max_nodes=1000,
        )
        return await self.traverse(node_id, cfg)

    async def find_connected_components(
        self,
        config: Optional[TraversalConfig] = None,
    ) -> list[TraversalResult]:
        """Find all connected components in the graph."""
        cfg = config or self.config
        components = []
        visited: Set[str] = set()

        # Get all nodes
        all_nodes = await self.store.find_nodes(limit=10000)

        for node in all_nodes:
            if node.id not in visited:
                # Traverse from this node
                component = await self.traverse(node.id, cfg)
                components.append(component)

                # Mark all nodes in component as visited
                for n in component.nodes:
                    visited.add(n.id)

        return components

    async def compute_pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 20,
        config: Optional[TraversalConfig] = None,
    ) -> dict[str, float]:
        """Compute PageRank scores for nodes."""
        # Get all nodes and edges
        all_nodes = await self.store.find_nodes(limit=10000)
        node_ids = [n.id for n in all_nodes]
        n = len(node_ids)

        if n == 0:
            return {}

        # Initialize scores
        scores = {node_id: 1.0 / n for node_id in node_ids}
        outgoing: dict[str, list[str]] = defaultdict(list)

        # Build adjacency
        for node in all_nodes:
            neighbors = await self.store.get_neighbors(
                node.id,
                direction="out",
                limit=1000,
            )
            for neighbor, edge in neighbors:
                outgoing[node.id].append(neighbor.id)

        # Iterate
        for _ in range(iterations):
            new_scores = {}
            for node_id in node_ids:
                # Get incoming links
                incoming = await self.store.get_neighbors(
                    node_id,
                    direction="in",
                    limit=1000,
                )

                rank_sum = 0.0
                for source_node, edge in incoming:
                    out_degree = len(outgoing.get(source_node.id, []))
                    if out_degree > 0:
                        rank_sum += scores[source_node.id] / out_degree

                new_scores[node_id] = (1 - damping) / n + damping * rank_sum

            scores = new_scores

        return scores
