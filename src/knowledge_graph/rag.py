"""Knowledge graph integration with RAG pipeline."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.knowledge_graph.extraction import (
    ExtractionPipeline,
    ExtractionResult,
    Entity,
)
from src.knowledge_graph.storage import Neo4jStore, Neo4jConfig, GraphNode
from src.knowledge_graph.query import (
    GraphSearcher,
    GraphSearchQuery,
    GraphSearchResult,
    GraphContextBuilder,
    SearchType,
)
from src.knowledge_graph.query.traversal import GraphTraverser, TraversalConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """Context retrieved from knowledge graph."""
    entities: list[Entity] = field(default_factory=list)
    context_text: str = ""
    graph_triples: list[tuple[str, str, str]] = field(default_factory=list)
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphRAG:
    """Integrate knowledge graph with RAG pipeline."""

    def __init__(
        self,
        store: Optional[Neo4jStore] = None,
        extraction_pipeline: Optional[ExtractionPipeline] = None,
        config: Optional[dict] = None,
    ):
        self.store = store or Neo4jStore()
        self.extraction_pipeline = extraction_pipeline or ExtractionPipeline()
        self.config = config or {}

        self.searcher = GraphSearcher(self.store)
        self.context_builder = GraphContextBuilder(self.searcher)
        self.traverser = GraphTraverser(self.store)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        await asyncio.gather(
            self.store.initialize(),
            self.extraction_pipeline.initialize(),
        )
        self._initialized = True

    async def close(self) -> None:
        """Close connections."""
        await self.store.close()

    async def ingest_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ExtractionResult:
        """Extract knowledge from document and store in graph."""
        if not self._initialized:
            await self.initialize()

        # Extract entities and relationships
        result = await self.extraction_pipeline.extract(
            text,
            document_id=document_id,
        )

        # Store in graph
        await self.store.store_extraction_result(result, tenant_id)

        logger.info(
            f"Ingested document {document_id}: "
            f"{len(result.entities)} entities, "
            f"{len(result.relationships)} relationships"
        )

        return result

    async def get_context_for_query(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        max_entities: int = 5,
        max_depth: int = 2,
    ) -> GraphContext:
        """Get relevant graph context for a query."""
        if not self._initialized:
            await self.initialize()

        context = GraphContext()

        # Search for relevant entities
        search_result = await self.searcher.search(GraphSearchQuery(
            text=query,
            search_type=SearchType.ENTITY,
            tenant_id=tenant_id,
            limit=max_entities,
        ))

        if not search_result.nodes:
            return context

        # Extract entities from nodes
        for node in search_result.nodes:
            entity = self._node_to_entity(node)
            context.entities.append(entity)

        # Build context text
        context.context_text = await self.context_builder.build_context(
            query,
            max_entities=max_entities,
            tenant_id=tenant_id,
        )

        # Get graph triples
        for node in search_result.nodes[:3]:
            neighbors = await self.store.get_neighbors(
                node.id,
                limit=5,
            )
            for neighbor, edge in neighbors:
                source_name = node.properties.get("name", node.id)
                target_name = neighbor.properties.get("name", neighbor.id)
                context.graph_triples.append(
                    (source_name, edge.type, target_name)
                )

        return context

    async def enhance_retrieval(
        self,
        query: str,
        vector_results: list[dict],
        tenant_id: Optional[str] = None,
    ) -> list[dict]:
        """Enhance vector search results with graph context."""
        if not self._initialized:
            await self.initialize()

        # Get graph context
        graph_context = await self.get_context_for_query(
            query,
            tenant_id=tenant_id,
        )

        # Add graph context to results
        for result in vector_results:
            result["graph_context"] = graph_context.context_text
            result["related_entities"] = [
                {"name": e.name, "type": e.type.value}
                for e in graph_context.entities
            ]

        # Boost results that match graph entities
        entity_names = {e.name.lower() for e in graph_context.entities}

        for result in vector_results:
            text = result.get("text", "").lower()
            matches = sum(1 for name in entity_names if name in text)
            if matches > 0:
                result["score"] = result.get("score", 0) * (1 + 0.1 * matches)

        # Re-sort by score
        vector_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return vector_results

    async def answer_with_graph(
        self,
        question: str,
        tenant_id: Optional[str] = None,
    ) -> dict:
        """Answer a question using knowledge graph."""
        if not self._initialized:
            await self.initialize()

        result = {
            "answer": "",
            "entities": [],
            "relationships": [],
            "context": "",
            "confidence": 0.0,
        }

        # Get graph context
        graph_context = await self.get_context_for_query(
            question,
            tenant_id=tenant_id,
            max_entities=10,
        )

        if not graph_context.entities:
            result["answer"] = "No relevant information found in knowledge graph."
            return result

        result["entities"] = [
            {"name": e.name, "type": e.type.value}
            for e in graph_context.entities
        ]
        result["relationships"] = [
            {"source": s, "type": t, "target": o}
            for s, t, o in graph_context.graph_triples
        ]
        result["context"] = graph_context.context_text
        result["confidence"] = 0.8 if graph_context.entities else 0.0

        # Generate answer from context
        # In a real system, this would use an LLM
        answer_parts = []
        for entity in graph_context.entities[:3]:
            answer_parts.append(f"{entity.name} ({entity.type.value})")

        if graph_context.graph_triples:
            answer_parts.append("\nRelationships:")
            for s, t, o in graph_context.graph_triples[:5]:
                answer_parts.append(f"  {s} --{t}--> {o}")

        result["answer"] = "\n".join(answer_parts)

        return result

    def _node_to_entity(self, node: GraphNode) -> Entity:
        """Convert graph node to entity."""
        from src.knowledge_graph.extraction.entities import EntityType

        entity_type = EntityType.CONCEPT
        for label in node.labels:
            try:
                entity_type = EntityType(label)
                break
            except ValueError:
                continue

        return Entity(
            id=node.id,
            name=node.properties.get("name", node.id),
            type=entity_type,
            aliases=node.properties.get("aliases", []),
            properties={
                k: v for k, v in node.properties.items()
                if k not in ["id", "name", "aliases", "type"]
            },
        )


class GraphEnhancedRetriever:
    """Retriever that combines vector search with knowledge graph."""

    def __init__(
        self,
        vector_store=None,
        knowledge_graph: Optional[KnowledgeGraphRAG] = None,
        config: Optional[dict] = None,
    ):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.config = config or {}

        # Weights for combining scores
        self.vector_weight = self.config.get("vector_weight", 0.7)
        self.graph_weight = self.config.get("graph_weight", 0.3)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        **kwargs,
    ) -> list[dict]:
        """Retrieve relevant documents using both vector and graph search."""
        results = []

        # Vector search
        if self.vector_store:
            vector_results = await self._vector_search(query, top_k * 2, tenant_id)
            results.extend(vector_results)

        # Graph search
        if self.knowledge_graph:
            graph_results = await self._graph_search(query, top_k, tenant_id)

            # Merge results
            results = self._merge_results(results, graph_results)

        # Sort by combined score
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return results[:top_k]

    async def _vector_search(
        self,
        query: str,
        limit: int,
        tenant_id: Optional[str],
    ) -> list[dict]:
        """Perform vector search."""
        if not self.vector_store:
            return []

        # Implementation depends on vector store interface
        results = await self.vector_store.search(
            query=query,
            limit=limit,
            filter={"tenant_id": tenant_id} if tenant_id else None,
        )

        return [
            {
                "id": r.id,
                "text": r.payload.get("text", ""),
                "score": r.score,
                "source": "vector",
                **r.payload,
            }
            for r in results
        ]

    async def _graph_search(
        self,
        query: str,
        limit: int,
        tenant_id: Optional[str],
    ) -> list[dict]:
        """Perform graph-based search."""
        if not self.knowledge_graph:
            return []

        context = await self.knowledge_graph.get_context_for_query(
            query,
            tenant_id=tenant_id,
            max_entities=limit,
        )

        results = []
        for entity in context.entities:
            results.append({
                "id": entity.id,
                "text": f"{entity.name}: {entity.properties.get('description', '')}",
                "score": entity.confidence,
                "source": "graph",
                "entity_type": entity.type.value,
            })

        return results

    def _merge_results(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> list[dict]:
        """Merge and re-score results from both sources."""
        # Create lookup by ID
        result_map = {}

        for r in vector_results:
            r["vector_score"] = r.get("score", 0)
            r["graph_score"] = 0
            result_map[r["id"]] = r

        for r in graph_results:
            if r["id"] in result_map:
                result_map[r["id"]]["graph_score"] = r.get("score", 0)
                result_map[r["id"]]["entity_type"] = r.get("entity_type")
            else:
                r["vector_score"] = 0
                r["graph_score"] = r.get("score", 0)
                result_map[r["id"]] = r

        # Compute combined scores
        for result in result_map.values():
            result["combined_score"] = (
                self.vector_weight * result.get("vector_score", 0) +
                self.graph_weight * result.get("graph_score", 0)
            )

        return list(result_map.values())

    async def get_context(
        self,
        query: str,
        results: list[dict],
        tenant_id: Optional[str] = None,
    ) -> str:
        """Build context string from retrieval results."""
        context_parts = []

        # Add vector search results
        for i, r in enumerate(results[:5]):
            context_parts.append(f"[{i + 1}] {r.get('text', '')[:500]}")

        # Add graph context
        if self.knowledge_graph:
            graph_context = await self.knowledge_graph.get_context_for_query(
                query,
                tenant_id=tenant_id,
            )
            if graph_context.context_text:
                context_parts.append("\n" + graph_context.context_text)

        return "\n\n".join(context_parts)
