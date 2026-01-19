"""Extraction pipeline for knowledge graph construction."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

from src.knowledge_graph.extraction.entities import (
    EntityExtractor,
    EntityLinker,
    Entity,
)
from src.knowledge_graph.extraction.relationships import (
    RelationshipExtractor,
    Relationship,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from knowledge extraction pipeline."""
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    # Processing metadata
    document_id: Optional[str] = None
    source_text_length: int = 0
    extraction_time_ms: float = 0.0
    extracted_at: Optional[datetime] = None

    # Statistics
    entity_counts: dict[str, int] = field(default_factory=dict)
    relationship_counts: dict[str, int] = field(default_factory=dict)

    def compute_statistics(self) -> None:
        """Compute extraction statistics."""
        # Count entities by type
        self.entity_counts = {}
        for entity in self.entities:
            type_name = entity.type.value
            self.entity_counts[type_name] = self.entity_counts.get(type_name, 0) + 1

        # Count relationships by type
        self.relationship_counts = {}
        for rel in self.relationships:
            type_name = rel.type.value
            self.relationship_counts[type_name] = self.relationship_counts.get(type_name, 0) + 1

    def to_triples(self) -> list[tuple[str, str, str]]:
        """Convert to RDF-like triples."""
        entity_lookup = {e.id: e for e in self.entities}
        triples = []

        for rel in self.relationships:
            source = entity_lookup.get(rel.source_entity_id)
            target = entity_lookup.get(rel.target_entity_id)

            if source and target:
                triples.append((source.name, rel.type.value, target.name))

        return triples

    def merge_with(self, other: "ExtractionResult") -> "ExtractionResult":
        """Merge another extraction result into this one."""
        # Merge entities (deduplicate by name and type)
        existing_entities = {(e.name.lower(), e.type): e for e in self.entities}

        for entity in other.entities:
            key = (entity.name.lower(), entity.type)
            if key in existing_entities:
                existing_entities[key].merge_with(entity)
            else:
                existing_entities[key] = entity
                self.entities.append(entity)

        # Merge relationships (deduplicate)
        existing_rels = {
            (r.source_entity_id, r.target_entity_id, r.type)
            for r in self.relationships
        }

        for rel in other.relationships:
            key = (rel.source_entity_id, rel.target_entity_id, rel.type)
            if key not in existing_rels:
                self.relationships.append(rel)
                existing_rels.add(key)

        self.compute_statistics()
        return self


class ExtractionPipeline:
    """Pipeline for extracting knowledge from text."""

    def __init__(
        self,
        entity_provider: str = "spacy",
        relationship_provider: str = "llm",
        enable_linking: bool = True,
        config: Optional[dict] = None,
    ):
        self.entity_extractor = EntityExtractor(provider=entity_provider)
        self.relationship_extractor = RelationshipExtractor(provider=relationship_provider)
        self.entity_linker = EntityLinker() if enable_linking else None
        self.config = config or {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all extractors."""
        await asyncio.gather(
            self.entity_extractor.initialize(),
            self.relationship_extractor.initialize(),
        )
        self._initialized = True

    async def extract(
        self,
        text: str,
        document_id: Optional[str] = None,
        extract_relationships: bool = True,
        link_entities: bool = True,
        **kwargs,
    ) -> ExtractionResult:
        """Extract entities and relationships from text."""
        start_time = datetime.now()

        if not self._initialized:
            await self.initialize()

        result = ExtractionResult(
            document_id=document_id,
            source_text_length=len(text),
        )

        # Extract entities
        entities = await self.entity_extractor.extract(text, document_id)
        result.entities = entities

        # Link entities to knowledge bases
        if link_entities and self.entity_linker and entities:
            result.entities = await self.entity_linker.link_batch(entities)

        # Extract relationships
        if extract_relationships and len(entities) >= 2:
            relationships = await self.relationship_extractor.extract(
                text, entities, document_id
            )
            result.relationships = relationships

        # Compute statistics
        result.compute_statistics()
        result.extracted_at = datetime.now()
        result.extraction_time_ms = (
            result.extracted_at - start_time
        ).total_seconds() * 1000

        return result

    async def extract_batch(
        self,
        texts: list[tuple[str, Optional[str]]],  # (text, document_id) pairs
        **kwargs,
    ) -> list[ExtractionResult]:
        """Extract from multiple texts in parallel."""
        tasks = [
            self.extract(text, doc_id, **kwargs)
            for text, doc_id in texts
        ]
        return await asyncio.gather(*tasks)

    async def extract_and_merge(
        self,
        texts: list[tuple[str, Optional[str]]],
        **kwargs,
    ) -> ExtractionResult:
        """Extract from multiple texts and merge results."""
        results = await self.extract_batch(texts, **kwargs)

        if not results:
            return ExtractionResult()

        merged = results[0]
        for result in results[1:]:
            merged.merge_with(result)

        return merged


class IncrementalExtractor:
    """Extract knowledge incrementally from streaming text."""

    def __init__(
        self,
        pipeline: Optional[ExtractionPipeline] = None,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):
        self.pipeline = pipeline or ExtractionPipeline()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._buffer = ""
        self._accumulated_result = ExtractionResult()

    async def process_chunk(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> ExtractionResult:
        """Process a chunk of text."""
        self._buffer += text

        # Extract when buffer is large enough
        if len(self._buffer) >= self.chunk_size:
            # Extract from current buffer
            result = await self.pipeline.extract(
                self._buffer,
                document_id,
            )

            # Merge with accumulated results
            self._accumulated_result.merge_with(result)

            # Keep overlap for context
            self._buffer = self._buffer[-self.overlap:]

            return result

        return ExtractionResult()

    async def flush(
        self,
        document_id: Optional[str] = None,
    ) -> ExtractionResult:
        """Process remaining buffer and return final results."""
        if self._buffer:
            result = await self.pipeline.extract(
                self._buffer,
                document_id,
            )
            self._accumulated_result.merge_with(result)
            self._buffer = ""

        return self._accumulated_result

    def reset(self) -> None:
        """Reset extractor state."""
        self._buffer = ""
        self._accumulated_result = ExtractionResult()
