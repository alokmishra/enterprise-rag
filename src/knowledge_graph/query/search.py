"""Graph search for knowledge graph queries."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.knowledge_graph.storage.base import GraphStore, GraphNode, GraphEdge
from src.knowledge_graph.extraction.entities import EntityType

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    """Types of graph searches."""
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    PATTERN = "pattern"
    SEMANTIC = "semantic"


@dataclass
class GraphSearchQuery:
    """A query for graph search."""
    # Text query
    text: Optional[str] = None

    # Entity filters
    entity_types: Optional[list[EntityType]] = None
    entity_names: Optional[list[str]] = None

    # Relationship filters
    relationship_types: Optional[list[str]] = None
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None

    # Pattern matching (Cypher-like)
    pattern: Optional[str] = None

    # Pagination
    limit: int = 10
    offset: int = 0

    # Tenant isolation
    tenant_id: Optional[str] = None

    # Search type
    search_type: SearchType = SearchType.ENTITY


@dataclass
class GraphSearchResult:
    """Result from graph search."""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0

    # For semantic search
    scores: dict[str, float] = field(default_factory=dict)

    def to_context(self) -> str:
        """Convert search results to context string for RAG."""
        context_parts = []

        for node in self.nodes:
            node_type = node.labels[0] if node.labels else "Entity"
            name = node.properties.get("name", node.id)
            description = node.properties.get("description", "")

            context_parts.append(f"{node_type}: {name}")
            if description:
                context_parts.append(f"  Description: {description}")

            # Add related entities
            for edge in self.edges:
                if edge.source_id == node.id:
                    context_parts.append(f"  -> {edge.type} -> {edge.target_id}")
                elif edge.target_id == node.id:
                    context_parts.append(f"  <- {edge.type} <- {edge.source_id}")

        return "\n".join(context_parts)


class GraphSearcher:
    """Search the knowledge graph."""

    def __init__(
        self,
        store: GraphStore,
        embedder=None,  # For semantic search
    ):
        self.store = store
        self.embedder = embedder

    async def search(
        self,
        query: GraphSearchQuery,
    ) -> GraphSearchResult:
        """Execute a graph search."""
        import time
        start_time = time.time()

        if query.search_type == SearchType.ENTITY:
            result = await self._search_entities(query)
        elif query.search_type == SearchType.RELATIONSHIP:
            result = await self._search_relationships(query)
        elif query.search_type == SearchType.PATTERN:
            result = await self._search_pattern(query)
        elif query.search_type == SearchType.SEMANTIC:
            result = await self._search_semantic(query)
        else:
            result = await self._search_entities(query)

        result.query_time_ms = (time.time() - start_time) * 1000
        return result

    async def _search_entities(
        self,
        query: GraphSearchQuery,
    ) -> GraphSearchResult:
        """Search for entities."""
        result = GraphSearchResult()

        # Build labels filter
        labels = None
        if query.entity_types:
            labels = ["Entity"] + [et.value for et in query.entity_types]

        # Build properties filter
        properties = {}
        if query.tenant_id:
            properties["tenant_id"] = query.tenant_id

        if query.entity_names:
            # Search each name
            for name in query.entity_names:
                nodes = await self.store.find_nodes(
                    labels=labels,
                    properties={"name": name, **properties},
                    limit=query.limit,
                )
                result.nodes.extend(nodes)
        elif query.text:
            # Full-text search using Neo4j
            nodes = await self.store.search_entities(
                query.text,
                entity_type=query.entity_types[0] if query.entity_types else None,
                tenant_id=query.tenant_id,
                limit=query.limit,
            )
            result.nodes = nodes
        else:
            # Return all entities matching filters
            nodes = await self.store.find_nodes(
                labels=labels,
                properties=properties if properties else None,
                limit=query.limit,
            )
            result.nodes = nodes

        result.total_count = len(result.nodes)

        # Get relationships for found entities
        for node in result.nodes[:10]:  # Limit relationship fetch
            edges = await self.store.find_edges(
                source_id=node.id,
                limit=5,
            )
            result.edges.extend(edges)

            edges = await self.store.find_edges(
                target_id=node.id,
                limit=5,
            )
            result.edges.extend(edges)

        return result

    async def _search_relationships(
        self,
        query: GraphSearchQuery,
    ) -> GraphSearchResult:
        """Search for relationships."""
        result = GraphSearchResult()

        edges = await self.store.find_edges(
            source_id=query.source_entity,
            target_id=query.target_entity,
            edge_type=query.relationship_types[0] if query.relationship_types else None,
            limit=query.limit,
        )

        result.edges = edges
        result.total_count = len(edges)

        # Get connected nodes
        node_ids = set()
        for edge in edges:
            node_ids.add(edge.source_id)
            node_ids.add(edge.target_id)

        for node_id in list(node_ids)[:20]:
            node = await self.store.get_node(node_id)
            if node:
                result.nodes.append(node)

        return result

    async def _search_pattern(
        self,
        query: GraphSearchQuery,
    ) -> GraphSearchResult:
        """Search using a Cypher pattern."""
        result = GraphSearchResult()

        if not query.pattern:
            return result

        # Execute pattern query
        records = await self.store.execute_query(
            query.pattern,
            {"tenant_id": query.tenant_id} if query.tenant_id else None,
        )

        # Parse results
        for record in records:
            for key, value in record.items():
                if hasattr(value, "labels"):
                    # Node
                    node_data = dict(value)
                    result.nodes.append(GraphNode(
                        id=node_data.pop("id", ""),
                        labels=list(value.labels),
                        properties=node_data,
                    ))
                elif hasattr(value, "type"):
                    # Relationship
                    edge_data = dict(value)
                    result.edges.append(GraphEdge(
                        id=edge_data.pop("id", ""),
                        source_id=value.start_node["id"],
                        target_id=value.end_node["id"],
                        type=value.type,
                        properties=edge_data,
                    ))

        result.total_count = len(result.nodes) + len(result.edges)
        return result

    async def _search_semantic(
        self,
        query: GraphSearchQuery,
    ) -> GraphSearchResult:
        """Semantic search using embeddings."""
        result = GraphSearchResult()

        if not query.text or not self.embedder:
            return await self._search_entities(query)

        # Generate query embedding
        query_embedding = await self.embedder.embed(query.text)

        # For semantic search, we need vector similarity
        # This would typically use a hybrid approach with vector DB
        # For now, fall back to text search
        return await self._search_entities(query)

    async def find_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[list[str]] = None,
        max_depth: int = 2,
        limit: int = 20,
    ) -> GraphSearchResult:
        """Find entities related to a given entity."""
        result = GraphSearchResult()
        visited = set()
        to_visit = [(entity_id, 0)]

        while to_visit and len(result.nodes) < limit:
            current_id, depth = to_visit.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            node = await self.store.get_node(current_id)
            if node and current_id != entity_id:
                result.nodes.append(node)

            neighbors = await self.store.get_neighbors(
                current_id,
                edge_types=relationship_types,
                limit=10,
            )

            for neighbor, edge in neighbors:
                if edge not in result.edges:
                    result.edges.append(edge)

                if neighbor.id not in visited:
                    to_visit.append((neighbor.id, depth + 1))

        result.total_count = len(result.nodes)
        return result

    async def answer_graph_question(
        self,
        question: str,
        tenant_id: Optional[str] = None,
    ) -> GraphSearchResult:
        """Answer a natural language question using the graph."""
        # Extract entities from question
        from src.knowledge_graph.extraction.entities import EntityExtractor

        extractor = EntityExtractor(provider="spacy")
        await extractor.initialize()
        entities = await extractor.extract(question)

        result = GraphSearchResult()

        if entities:
            # Search for mentioned entities
            entity_names = [e.name for e in entities]
            query = GraphSearchQuery(
                entity_names=entity_names,
                tenant_id=tenant_id,
                limit=10,
            )
            result = await self._search_entities(query)

            # Find relationships between found entities
            if len(result.nodes) >= 2:
                for i, node1 in enumerate(result.nodes):
                    for node2 in result.nodes[i + 1:]:
                        path = await self.store.find_path(
                            node1.id,
                            node2.id,
                            max_depth=3,
                        )
                        if path:
                            for edge in path.edges:
                                if edge not in result.edges:
                                    result.edges.append(edge)

        return result


class GraphContextBuilder:
    """Build context from graph for RAG."""

    def __init__(
        self,
        searcher: GraphSearcher,
    ):
        self.searcher = searcher

    async def build_context(
        self,
        query: str,
        max_entities: int = 5,
        max_relationships: int = 10,
        tenant_id: Optional[str] = None,
    ) -> str:
        """Build context string from graph for RAG."""
        # Search for relevant entities
        search_query = GraphSearchQuery(
            text=query,
            search_type=SearchType.ENTITY,
            limit=max_entities,
            tenant_id=tenant_id,
        )
        result = await self.searcher.search(search_query)

        if not result.nodes:
            return ""

        # Build context
        context_parts = ["Knowledge Graph Context:"]

        for node in result.nodes:
            name = node.properties.get("name", node.id)
            node_type = node.labels[-1] if node.labels else "Entity"
            canonical = node.properties.get("canonical_name", name)

            context_parts.append(f"\n{node_type}: {name}")

            if canonical != name:
                context_parts.append(f"  Also known as: {canonical}")

            # Add properties
            for key, value in node.properties.items():
                if key not in ["id", "name", "canonical_name", "tenant_id", "created_at"]:
                    if isinstance(value, str) and len(value) < 200:
                        context_parts.append(f"  {key}: {value}")

            # Add relationships
            related = await self.searcher.find_related_entities(
                node.id,
                max_depth=1,
                limit=5,
            )

            for edge in related.edges[:max_relationships]:
                if edge.source_id == node.id:
                    target = next(
                        (n for n in related.nodes if n.id == edge.target_id),
                        None,
                    )
                    if target:
                        target_name = target.properties.get("name", target.id)
                        context_parts.append(f"  - {edge.type} -> {target_name}")
                else:
                    source = next(
                        (n for n in related.nodes if n.id == edge.source_id),
                        None,
                    )
                    if source:
                        source_name = source.properties.get("name", source.id)
                        context_parts.append(f"  - {source_name} {edge.type} -> this")

        return "\n".join(context_parts)
