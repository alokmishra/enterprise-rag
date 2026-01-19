"""Cypher query builder for Neo4j."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CypherClause:
    """A clause in a Cypher query."""
    type: str  # MATCH, WHERE, RETURN, etc.
    content: str
    parameters: dict[str, Any] = field(default_factory=dict)


class CypherQueryBuilder:
    """Build Cypher queries programmatically."""

    def __init__(self):
        self._clauses: list[CypherClause] = []
        self._parameters: dict[str, Any] = {}
        self._param_counter: int = 0

    def match(
        self,
        pattern: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add MATCH clause."""
        self._clauses.append(CypherClause("MATCH", pattern, params))
        self._parameters.update(params)
        return self

    def optional_match(
        self,
        pattern: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add OPTIONAL MATCH clause."""
        self._clauses.append(CypherClause("OPTIONAL MATCH", pattern, params))
        self._parameters.update(params)
        return self

    def where(
        self,
        condition: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add WHERE clause."""
        self._clauses.append(CypherClause("WHERE", condition, params))
        self._parameters.update(params)
        return self

    def and_where(
        self,
        condition: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add AND condition to WHERE clause."""
        self._clauses.append(CypherClause("AND", condition, params))
        self._parameters.update(params)
        return self

    def or_where(
        self,
        condition: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add OR condition to WHERE clause."""
        self._clauses.append(CypherClause("OR", condition, params))
        self._parameters.update(params)
        return self

    def return_(
        self,
        *expressions: str,
    ) -> "CypherQueryBuilder":
        """Add RETURN clause."""
        self._clauses.append(CypherClause("RETURN", ", ".join(expressions)))
        return self

    def return_distinct(
        self,
        *expressions: str,
    ) -> "CypherQueryBuilder":
        """Add RETURN DISTINCT clause."""
        self._clauses.append(CypherClause("RETURN DISTINCT", ", ".join(expressions)))
        return self

    def order_by(
        self,
        *expressions: str,
        desc: bool = False,
    ) -> "CypherQueryBuilder":
        """Add ORDER BY clause."""
        order = " DESC" if desc else ""
        content = ", ".join(f"{e}{order}" for e in expressions)
        self._clauses.append(CypherClause("ORDER BY", content))
        return self

    def limit(
        self,
        count: int,
    ) -> "CypherQueryBuilder":
        """Add LIMIT clause."""
        param_name = self._next_param("limit")
        self._clauses.append(CypherClause("LIMIT", f"${param_name}"))
        self._parameters[param_name] = count
        return self

    def skip(
        self,
        count: int,
    ) -> "CypherQueryBuilder":
        """Add SKIP clause."""
        param_name = self._next_param("skip")
        self._clauses.append(CypherClause("SKIP", f"${param_name}"))
        self._parameters[param_name] = count
        return self

    def create(
        self,
        pattern: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add CREATE clause."""
        self._clauses.append(CypherClause("CREATE", pattern, params))
        self._parameters.update(params)
        return self

    def merge(
        self,
        pattern: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add MERGE clause."""
        self._clauses.append(CypherClause("MERGE", pattern, params))
        self._parameters.update(params)
        return self

    def on_create_set(
        self,
        *assignments: str,
    ) -> "CypherQueryBuilder":
        """Add ON CREATE SET clause."""
        self._clauses.append(CypherClause("ON CREATE SET", ", ".join(assignments)))
        return self

    def on_match_set(
        self,
        *assignments: str,
    ) -> "CypherQueryBuilder":
        """Add ON MATCH SET clause."""
        self._clauses.append(CypherClause("ON MATCH SET", ", ".join(assignments)))
        return self

    def set(
        self,
        *assignments: str,
        **params,
    ) -> "CypherQueryBuilder":
        """Add SET clause."""
        self._clauses.append(CypherClause("SET", ", ".join(assignments), params))
        self._parameters.update(params)
        return self

    def delete(
        self,
        *variables: str,
    ) -> "CypherQueryBuilder":
        """Add DELETE clause."""
        self._clauses.append(CypherClause("DELETE", ", ".join(variables)))
        return self

    def detach_delete(
        self,
        *variables: str,
    ) -> "CypherQueryBuilder":
        """Add DETACH DELETE clause."""
        self._clauses.append(CypherClause("DETACH DELETE", ", ".join(variables)))
        return self

    def with_(
        self,
        *expressions: str,
    ) -> "CypherQueryBuilder":
        """Add WITH clause."""
        self._clauses.append(CypherClause("WITH", ", ".join(expressions)))
        return self

    def unwind(
        self,
        expression: str,
        alias: str,
    ) -> "CypherQueryBuilder":
        """Add UNWIND clause."""
        self._clauses.append(CypherClause("UNWIND", f"{expression} AS {alias}"))
        return self

    def call(
        self,
        procedure: str,
        *args: str,
    ) -> "CypherQueryBuilder":
        """Add CALL clause for procedure."""
        args_str = ", ".join(args) if args else ""
        self._clauses.append(CypherClause("CALL", f"{procedure}({args_str})"))
        return self

    def yield_(
        self,
        *variables: str,
    ) -> "CypherQueryBuilder":
        """Add YIELD clause."""
        self._clauses.append(CypherClause("YIELD", ", ".join(variables)))
        return self

    def param(
        self,
        name: str,
        value: Any,
    ) -> str:
        """Add a parameter and return its reference."""
        param_name = self._next_param(name)
        self._parameters[param_name] = value
        return f"${param_name}"

    def _next_param(self, prefix: str = "p") -> str:
        """Generate next parameter name."""
        self._param_counter += 1
        return f"{prefix}_{self._param_counter}"

    def build(self) -> tuple[str, dict[str, Any]]:
        """Build the Cypher query."""
        parts = []
        for clause in self._clauses:
            parts.append(f"{clause.type} {clause.content}")

        query = "\n".join(parts)
        return query, self._parameters

    def __str__(self) -> str:
        """Return the query string."""
        query, _ = self.build()
        return query

    # Convenience methods for common patterns

    @classmethod
    def find_entity_by_name(
        cls,
        name: str,
        entity_type: Optional[str] = None,
    ) -> "CypherQueryBuilder":
        """Create query to find entity by name."""
        builder = cls()
        label = f":Entity:{entity_type}" if entity_type else ":Entity"

        builder.match(f"(e{label})")
        builder.where("e.name = $name", name=name)
        builder.return_("e")

        return builder

    @classmethod
    def find_relationships(
        cls,
        source_name: str,
        target_name: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> "CypherQueryBuilder":
        """Create query to find relationships."""
        builder = cls()

        rel_pattern = f"[r:{relationship_type}]" if relationship_type else "[r]"
        target_pattern = "(t:Entity)" if target_name else "(t)"

        builder.match(f"(s:Entity)-{rel_pattern}->{target_pattern}")
        builder.where("s.name = $source", source=source_name)

        if target_name:
            builder.and_where("t.name = $target", target=target_name)

        builder.return_("s", "r", "t")

        return builder

    @classmethod
    def find_path(
        cls,
        source_name: str,
        target_name: str,
        max_depth: int = 5,
    ) -> "CypherQueryBuilder":
        """Create query to find shortest path."""
        builder = cls()

        builder.match(f"path = shortestPath((s:Entity)-[*1..{max_depth}]-(t:Entity))")
        builder.where("s.name = $source AND t.name = $target",
                      source=source_name, target=target_name)
        builder.return_("path")

        return builder

    @classmethod
    def get_entity_neighborhood(
        cls,
        entity_name: str,
        depth: int = 1,
    ) -> "CypherQueryBuilder":
        """Create query to get entity neighborhood."""
        builder = cls()

        builder.match(f"(e:Entity)-[r*1..{depth}]-(n)")
        builder.where("e.name = $name", name=entity_name)
        builder.return_distinct("e", "r", "n")

        return builder

    @classmethod
    def search_entities(
        cls,
        search_term: str,
        limit: int = 10,
    ) -> "CypherQueryBuilder":
        """Create full-text search query."""
        builder = cls()

        builder.call("db.index.fulltext.queryNodes", "'entity_search'", "$term")
        builder.yield_("node", "score")
        builder.return_("node", "score")
        builder.order_by("score", desc=True)
        builder.limit(limit)

        builder._parameters["term"] = search_term

        return builder

    @classmethod
    def upsert_entity(
        cls,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict[str, Any],
    ) -> "CypherQueryBuilder":
        """Create upsert query for entity."""
        builder = cls()

        builder.merge(f"(e:Entity:{entity_type} {{id: $id}})",
                      id=entity_id)
        builder.on_create_set(
            "e.name = $name",
            "e.created_at = datetime()",
        )
        builder.on_match_set("e.updated_at = datetime()")
        builder.set("e += $props", props=properties)
        builder._parameters["name"] = name

        builder.return_("e")

        return builder
