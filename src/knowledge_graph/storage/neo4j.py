"""Neo4j graph storage implementation."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

from src.knowledge_graph.storage.base import (
    GraphStore,
    GraphNode,
    GraphEdge,
    GraphPath,
)
from src.knowledge_graph.extraction.entities import Entity, EntityType
from src.knowledge_graph.extraction.relationships import Relationship, RelationshipType

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0


class Neo4jStore(GraphStore):
    """Neo4j implementation of graph storage."""

    def __init__(self, config: Optional[Neo4jConfig] = None):
        self.config = config or Neo4jConfig()
        self._driver = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Neo4j connection."""
        if self._initialized:
            return

        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
            )

            # Verify connection
            async with self._driver.session(database=self.config.database) as session:
                await session.run("RETURN 1")

            # Create indexes
            await self._create_indexes()

            self._initialized = True
            logger.info("Neo4j connection established")

        except ImportError:
            logger.error("neo4j driver not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def _create_indexes(self) -> None:
        """Create necessary indexes."""
        indexes = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.canonical_name]",
        ]

        async with self._driver.session(database=self.config.database) as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index creation note: {e}")

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._initialized = False

    async def create_node(
        self,
        node: GraphNode,
        **kwargs,
    ) -> GraphNode:
        """Create a node in Neo4j."""
        labels = ":".join(node.labels) if node.labels else "Node"
        properties = {
            "id": node.id,
            **node.properties,
            "created_at": datetime.utcnow().isoformat(),
        }

        if node.tenant_id:
            properties["tenant_id"] = node.tenant_id

        query = f"""
        CREATE (n:{labels} $properties)
        RETURN n
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, properties=properties)
            record = await result.single()

            if record:
                node.created_at = properties["created_at"]
                return node

        return node

    async def get_node(
        self,
        node_id: str,
        **kwargs,
    ) -> Optional[GraphNode]:
        """Get a node by ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()

            if record:
                node_data = dict(record["n"])
                return GraphNode(
                    id=node_data.pop("id"),
                    labels=record["labels"],
                    properties=node_data,
                    tenant_id=node_data.get("tenant_id"),
                    created_at=node_data.get("created_at"),
                )

        return None

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
        **kwargs,
    ) -> Optional[GraphNode]:
        """Update node properties."""
        properties["updated_at"] = datetime.utcnow().isoformat()

        query = """
        MATCH (n {id: $node_id})
        SET n += $properties
        RETURN n, labels(n) as labels
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                node_id=node_id,
                properties=properties,
            )
            record = await result.single()

            if record:
                node_data = dict(record["n"])
                return GraphNode(
                    id=node_data.pop("id"),
                    labels=record["labels"],
                    properties=node_data,
                )

        return None

    async def delete_node(
        self,
        node_id: str,
        **kwargs,
    ) -> bool:
        """Delete a node and its relationships."""
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            return record and record["deleted"] > 0

    async def find_nodes(
        self,
        labels: Optional[list[str]] = None,
        properties: Optional[dict[str, Any]] = None,
        limit: int = 100,
        **kwargs,
    ) -> list[GraphNode]:
        """Find nodes by labels and properties."""
        label_str = ":".join(labels) if labels else ""
        match_clause = f"MATCH (n:{label_str})" if label_str else "MATCH (n)"

        where_clauses = []
        params = {"limit": limit}

        if properties:
            for key, value in properties.items():
                param_name = f"prop_{key}"
                where_clauses.append(f"n.{key} = ${param_name}")
                params[param_name] = value

        where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        {match_clause}
        {where_str}
        RETURN n, labels(n) as labels
        LIMIT $limit
        """

        nodes = []
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            async for record in result:
                node_data = dict(record["n"])
                nodes.append(GraphNode(
                    id=node_data.pop("id", ""),
                    labels=record["labels"],
                    properties=node_data,
                ))

        return nodes

    async def create_edge(
        self,
        edge: GraphEdge,
        **kwargs,
    ) -> GraphEdge:
        """Create an edge between nodes."""
        properties = {
            "id": edge.id,
            **edge.properties,
            "created_at": datetime.utcnow().isoformat(),
        }

        if edge.tenant_id:
            properties["tenant_id"] = edge.tenant_id

        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        CREATE (source)-[r:{edge.type} $properties]->(target)
        RETURN r
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                properties=properties,
            )
            record = await result.single()

            if record:
                edge.created_at = properties["created_at"]

        return edge

    async def get_edge(
        self,
        edge_id: str,
        **kwargs,
    ) -> Optional[GraphEdge]:
        """Get an edge by ID."""
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        RETURN r, type(r) as type, startNode(r).id as source_id, endNode(r).id as target_id
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, edge_id=edge_id)
            record = await result.single()

            if record:
                edge_data = dict(record["r"])
                return GraphEdge(
                    id=edge_data.pop("id"),
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    type=record["type"],
                    properties=edge_data,
                )

        return None

    async def delete_edge(
        self,
        edge_id: str,
        **kwargs,
    ) -> bool:
        """Delete an edge."""
        query = """
        MATCH ()-[r {id: $edge_id}]->()
        DELETE r
        RETURN count(r) as deleted
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, edge_id=edge_id)
            record = await result.single()
            return record and record["deleted"] > 0

    async def find_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> list[GraphEdge]:
        """Find edges by criteria."""
        type_str = f":{edge_type}" if edge_type else ""

        where_clauses = []
        params = {"limit": limit}

        if source_id:
            where_clauses.append("source.id = $source_id")
            params["source_id"] = source_id

        if target_id:
            where_clauses.append("target.id = $target_id")
            params["target_id"] = target_id

        where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (source)-[r{type_str}]->(target)
        {where_str}
        RETURN r, type(r) as type, source.id as source_id, target.id as target_id
        LIMIT $limit
        """

        edges = []
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, **params)
            async for record in result:
                edge_data = dict(record["r"])
                edges.append(GraphEdge(
                    id=edge_data.pop("id", ""),
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    type=record["type"],
                    properties=edge_data,
                ))

        return edges

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        direction: str = "both",
        limit: int = 100,
        **kwargs,
    ) -> list[tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes."""
        type_filter = "|".join(edge_types) if edge_types else ""
        type_str = f":{type_filter}" if type_filter else ""

        if direction == "out":
            pattern = f"(n)-[r{type_str}]->(neighbor)"
        elif direction == "in":
            pattern = f"(n)<-[r{type_str}]-(neighbor)"
        else:
            pattern = f"(n)-[r{type_str}]-(neighbor)"

        query = f"""
        MATCH {pattern}
        WHERE n.id = $node_id
        RETURN neighbor, labels(neighbor) as labels, r, type(r) as rel_type,
               startNode(r).id as source_id, endNode(r).id as target_id
        LIMIT $limit
        """

        results = []
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, node_id=node_id, limit=limit)
            async for record in result:
                neighbor_data = dict(record["neighbor"])
                node = GraphNode(
                    id=neighbor_data.pop("id", ""),
                    labels=record["labels"],
                    properties=neighbor_data,
                )

                edge_data = dict(record["r"])
                edge = GraphEdge(
                    id=edge_data.pop("id", ""),
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    type=record["rel_type"],
                    properties=edge_data,
                )

                results.append((node, edge))

        return results

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        edge_types: Optional[list[str]] = None,
        **kwargs,
    ) -> Optional[GraphPath]:
        """Find shortest path between nodes."""
        type_filter = "|".join(edge_types) if edge_types else ""
        type_str = f":{type_filter}" if type_filter else ""

        query = f"""
        MATCH path = shortestPath((source)-[{type_str}*1..{max_depth}]-(target))
        WHERE source.id = $source_id AND target.id = $target_id
        RETURN path, nodes(path) as nodes, relationships(path) as edges
        """

        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                source_id=source_id,
                target_id=target_id,
            )
            record = await result.single()

            if record:
                nodes = []
                for node in record["nodes"]:
                    node_data = dict(node)
                    nodes.append(GraphNode(
                        id=node_data.pop("id", ""),
                        labels=list(node.labels),
                        properties=node_data,
                    ))

                edges = []
                for rel in record["edges"]:
                    edge_data = dict(rel)
                    edges.append(GraphEdge(
                        id=edge_data.pop("id", ""),
                        source_id=rel.start_node["id"],
                        target_id=rel.end_node["id"],
                        type=rel.type,
                        properties=edge_data,
                    ))

                return GraphPath(nodes=nodes, edges=edges)

        return None

    async def execute_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        **kwargs,
    ) -> list[dict]:
        """Execute a Cypher query."""
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, parameters or {})
            records = []
            async for record in result:
                records.append(dict(record))
            return records

    # High-level methods for entities and relationships

    async def store_entity(
        self,
        entity: Entity,
        tenant_id: Optional[str] = None,
    ) -> GraphNode:
        """Store an entity as a graph node."""
        node = GraphNode(
            id=entity.id,
            labels=["Entity", entity.type.value],
            properties={
                "name": entity.name,
                "type": entity.type.value,
                "canonical_name": entity.canonical_name or entity.name,
                "aliases": entity.aliases,
                "confidence": entity.confidence,
                "external_ids": entity.external_ids,
                **entity.properties,
            },
            tenant_id=tenant_id,
        )
        return await self.create_node(node)

    async def store_relationship(
        self,
        relationship: Relationship,
        tenant_id: Optional[str] = None,
    ) -> GraphEdge:
        """Store a relationship as a graph edge."""
        edge = GraphEdge(
            id=relationship.id,
            source_id=relationship.source_entity_id,
            target_id=relationship.target_entity_id,
            type=relationship.type.value,
            properties={
                "label": relationship.label,
                "confidence": relationship.confidence,
                "source_text": relationship.source_text,
                "bidirectional": relationship.bidirectional,
                **relationship.properties,
            },
            tenant_id=tenant_id,
        )
        return await self.create_edge(edge)

    async def store_extraction_result(
        self,
        result,  # ExtractionResult
        tenant_id: Optional[str] = None,
    ) -> dict:
        """Store an extraction result."""
        stats = {"nodes_created": 0, "edges_created": 0}

        # Store entities
        for entity in result.entities:
            await self.store_entity(entity, tenant_id)
            stats["nodes_created"] += 1

        # Store relationships
        for rel in result.relationships:
            await self.store_relationship(rel, tenant_id)
            stats["edges_created"] += 1

        return stats

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[GraphNode]:
        """Full-text search for entities."""
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_search', $query)
        YIELD node, score
        WHERE ($type IS NULL OR node.type = $type)
          AND ($tenant_id IS NULL OR node.tenant_id = $tenant_id)
        RETURN node, labels(node) as labels, score
        ORDER BY score DESC
        LIMIT $limit
        """

        params = {
            "query": query,
            "type": entity_type.value if entity_type else None,
            "tenant_id": tenant_id,
            "limit": limit,
        }

        nodes = []
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(cypher, params)
            async for record in result:
                node_data = dict(record["node"])
                nodes.append(GraphNode(
                    id=node_data.pop("id", ""),
                    labels=record["labels"],
                    properties={**node_data, "search_score": record["score"]},
                ))

        return nodes
