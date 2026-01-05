"""Relationship extraction for knowledge graph construction."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from src.knowledge_graph.extraction.entities import Entity, EntityType
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Common relationship types."""
    # Organizational
    WORKS_FOR = "WORKS_FOR"
    EMPLOYED_BY = "EMPLOYED_BY"
    FOUNDED = "FOUNDED"
    CEO_OF = "CEO_OF"
    MEMBER_OF = "MEMBER_OF"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PARTNER_OF = "PARTNER_OF"
    ACQUIRED = "ACQUIRED"

    # Location
    LOCATED_IN = "LOCATED_IN"
    HEADQUARTERS_IN = "HEADQUARTERS_IN"
    BORN_IN = "BORN_IN"
    LIVES_IN = "LIVES_IN"

    # Temporal
    OCCURRED_ON = "OCCURRED_ON"
    STARTED_ON = "STARTED_ON"
    ENDED_ON = "ENDED_ON"

    # Creation/Authorship
    CREATED = "CREATED"
    AUTHORED = "AUTHORED"
    DEVELOPED = "DEVELOPED"
    INVENTED = "INVENTED"

    # Association
    RELATED_TO = "RELATED_TO"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    PART_OF = "PART_OF"
    HAS_PART = "HAS_PART"
    INSTANCE_OF = "INSTANCE_OF"
    SUBCLASS_OF = "SUBCLASS_OF"

    # Actions
    USES = "USES"
    PRODUCES = "PRODUCES"
    PROVIDES = "PROVIDES"
    REQUIRES = "REQUIRES"

    # Custom
    CUSTOM = "CUSTOM"


@dataclass
class Relationship:
    """A relationship between two entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    type: RelationshipType
    label: str  # Human-readable label

    # Relationship details
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    # Provenance
    source_text: Optional[str] = None  # The text that indicates this relationship
    source_document_id: Optional[str] = None

    # Direction
    bidirectional: bool = False

    def reverse(self) -> "Relationship":
        """Create reversed relationship."""
        reverse_types = {
            RelationshipType.WORKS_FOR: RelationshipType.EMPLOYED_BY,
            RelationshipType.EMPLOYED_BY: RelationshipType.WORKS_FOR,
            RelationshipType.PART_OF: RelationshipType.HAS_PART,
            RelationshipType.HAS_PART: RelationshipType.PART_OF,
            RelationshipType.SUBSIDIARY_OF: RelationshipType.ACQUIRED,
        }

        return Relationship(
            id=str(uuid4()),
            source_entity_id=self.target_entity_id,
            target_entity_id=self.source_entity_id,
            type=reverse_types.get(self.type, self.type),
            label=f"reverse of {self.label}",
            properties=self.properties.copy(),
            confidence=self.confidence,
            source_text=self.source_text,
            source_document_id=self.source_document_id,
        )


class RelationshipExtractor:
    """Extract relationships between entities."""

    def __init__(
        self,
        provider: str = "llm",  # llm, pattern, dependency
        model: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.model = model
        self.config = config or {}
        self._initialized = False
        self._nlp = None

    async def initialize(self) -> None:
        """Initialize the relationship extractor."""
        if self._initialized:
            return

        if self.provider == "dependency":
            await self._init_spacy()
        else:
            self._initialized = True

    async def _init_spacy(self) -> None:
        """Initialize spaCy for dependency parsing."""
        try:
            import spacy
            model_name = self.model or "en_core_web_sm"

            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                await asyncio.to_thread(spacy.cli.download, model_name)
                self._nlp = spacy.load(model_name)

            self._initialized = True
        except ImportError:
            logger.warning("spaCy not installed")

    async def extract(
        self,
        text: str,
        entities: list[Entity],
        document_id: Optional[str] = None,
        **kwargs,
    ) -> list[Relationship]:
        """Extract relationships from text given entities."""
        if not self._initialized:
            await self.initialize()

        if len(entities) < 2:
            return []

        if self.provider == "llm":
            return await self._extract_llm(text, entities, document_id)
        elif self.provider == "pattern":
            return await self._extract_pattern(text, entities, document_id)
        elif self.provider == "dependency":
            return await self._extract_dependency(text, entities, document_id)
        else:
            return await self._extract_pattern(text, entities, document_id)

    async def _extract_llm(
        self,
        text: str,
        entities: list[Entity],
        document_id: Optional[str] = None,
    ) -> list[Relationship]:
        """Extract relationships using LLM."""
        try:
            from openai import AsyncOpenAI

            settings = get_settings()
            client = AsyncOpenAI(api_key=settings.openai_api_key)

            entity_list = "\n".join([
                f"- {e.name} ({e.type.value})" for e in entities[:20]  # Limit entities
            ])

            relationship_types = ", ".join([rt.value for rt in RelationshipType if rt != RelationshipType.CUSTOM])

            prompt = f"""Given the following text and entities, extract relationships between the entities.

Text:
{text[:3000]}

Entities:
{entity_list}

Relationship types: {relationship_types}

For each relationship found, provide:
- source: The source entity name
- target: The target entity name
- type: The relationship type
- evidence: The text that supports this relationship

Return as JSON array. Only include relationships you are confident about.

Relationships:"""

            response = await client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=2000,
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)

            # Build entity lookup
            entity_lookup = {e.name.lower(): e for e in entities}
            for e in entities:
                for alias in e.aliases:
                    entity_lookup[alias.lower()] = e

            relationships = []
            for item in data.get("relationships", data if isinstance(data, list) else []):
                source_name = item.get("source", "").lower()
                target_name = item.get("target", "").lower()

                source_entity = entity_lookup.get(source_name)
                target_entity = entity_lookup.get(target_name)

                if source_entity and target_entity:
                    try:
                        rel_type = RelationshipType(item.get("type", "RELATED_TO").upper())
                    except ValueError:
                        rel_type = RelationshipType.RELATED_TO

                    relationships.append(Relationship(
                        id=str(uuid4()),
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        type=rel_type,
                        label=f"{source_entity.name} {rel_type.value} {target_entity.name}",
                        source_text=item.get("evidence", ""),
                        source_document_id=document_id,
                    ))

            return relationships

        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {e}")
            return await self._extract_pattern(text, entities, document_id)

    async def _extract_pattern(
        self,
        text: str,
        entities: list[Entity],
        document_id: Optional[str] = None,
    ) -> list[Relationship]:
        """Extract relationships using pattern matching."""
        relationships = []

        # Build entity lookup by position
        entity_positions = []
        for entity in entities:
            for mention in entity.mentions:
                entity_positions.append({
                    "entity": entity,
                    "start": mention.start_char,
                    "end": mention.end_char,
                    "text": mention.text,
                })

        # Sort by position
        entity_positions.sort(key=lambda x: x["start"])

        # Relationship patterns
        patterns = [
            # Organization relationships
            (r"(?P<person>\w+)\s+(?:works?\s+(?:for|at)|is\s+employed\s+by)\s+(?P<org>\w+)",
             RelationshipType.WORKS_FOR, EntityType.PERSON, EntityType.ORGANIZATION),

            (r"(?P<org1>\w+)\s+(?:acquired|bought|purchased)\s+(?P<org2>\w+)",
             RelationshipType.ACQUIRED, EntityType.ORGANIZATION, EntityType.ORGANIZATION),

            (r"(?P<org>\w+)\s+(?:is\s+)?(?:headquartered|based|located)\s+in\s+(?P<loc>\w+)",
             RelationshipType.LOCATED_IN, EntityType.ORGANIZATION, EntityType.LOCATION),

            # Person relationships
            (r"(?P<person>\w+)\s+(?:founded|started|created)\s+(?P<org>\w+)",
             RelationshipType.FOUNDED, EntityType.PERSON, EntityType.ORGANIZATION),

            (r"(?P<person>\w+)\s+is\s+(?:the\s+)?CEO\s+of\s+(?P<org>\w+)",
             RelationshipType.CEO_OF, EntityType.PERSON, EntityType.ORGANIZATION),

            # Product relationships
            (r"(?P<org>\w+)\s+(?:makes?|produces?|manufactures?)\s+(?P<product>\w+)",
             RelationshipType.PRODUCES, EntityType.ORGANIZATION, EntityType.PRODUCT),

            # Generic relationships
            (r"(?P<entity1>\w+)\s+(?:is\s+)?(?:part\s+of|belongs?\s+to)\s+(?P<entity2>\w+)",
             RelationshipType.PART_OF, None, None),
        ]

        text_lower = text.lower()

        for pattern, rel_type, source_type, target_type in patterns:
            for match in re.finditer(pattern, text_lower):
                # Find matching entities
                groups = match.groupdict()
                source_text = list(groups.values())[0]
                target_text = list(groups.values())[1]

                source_entity = self._find_entity(source_text, entities, source_type)
                target_entity = self._find_entity(target_text, entities, target_type)

                if source_entity and target_entity and source_entity.id != target_entity.id:
                    relationships.append(Relationship(
                        id=str(uuid4()),
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        type=rel_type,
                        label=f"{source_entity.name} {rel_type.value} {target_entity.name}",
                        source_text=match.group(),
                        source_document_id=document_id,
                        confidence=0.7,
                    ))

        # Co-occurrence based relationships (entities mentioned close together)
        for i, pos1 in enumerate(entity_positions):
            for pos2 in entity_positions[i + 1:]:
                # Check if entities are within 100 characters
                distance = pos2["start"] - pos1["end"]
                if 0 < distance < 100:
                    # Different entities of different types
                    if (pos1["entity"].id != pos2["entity"].id and
                            pos1["entity"].type != pos2["entity"].type):
                        relationships.append(Relationship(
                            id=str(uuid4()),
                            source_entity_id=pos1["entity"].id,
                            target_entity_id=pos2["entity"].id,
                            type=RelationshipType.RELATED_TO,
                            label=f"{pos1['entity'].name} RELATED_TO {pos2['entity'].name}",
                            source_text=text[pos1["start"]:pos2["end"]],
                            source_document_id=document_id,
                            confidence=0.5,  # Lower confidence for co-occurrence
                        ))

        return relationships

    async def _extract_dependency(
        self,
        text: str,
        entities: list[Entity],
        document_id: Optional[str] = None,
    ) -> list[Relationship]:
        """Extract relationships using dependency parsing."""
        if not self._nlp:
            return await self._extract_pattern(text, entities, document_id)

        doc = await asyncio.to_thread(self._nlp, text)
        relationships = []

        # Build entity lookup by text span
        entity_lookup = {}
        for entity in entities:
            entity_lookup[entity.name.lower()] = entity
            for alias in entity.aliases:
                entity_lookup[alias.lower()] = entity

        # Find verbs connecting entities
        for token in doc:
            if token.pos_ == "VERB":
                # Find subject and object
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        # Get the full noun phrase
                        subject_text = self._get_noun_phrase(child)
                        subject = entity_lookup.get(subject_text.lower())

                    elif child.dep_ in ("dobj", "pobj", "attr"):
                        obj_text = self._get_noun_phrase(child)
                        obj = entity_lookup.get(obj_text.lower())

                if subject and obj and subject.id != obj.id:
                    rel_type = self._verb_to_relationship(token.lemma_)
                    relationships.append(Relationship(
                        id=str(uuid4()),
                        source_entity_id=subject.id,
                        target_entity_id=obj.id,
                        type=rel_type,
                        label=f"{subject.name} {token.text} {obj.name}",
                        source_text=token.sent.text,
                        source_document_id=document_id,
                        confidence=0.8,
                    ))

        return relationships

    def _find_entity(
        self,
        text: str,
        entities: list[Entity],
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """Find entity matching text."""
        text_lower = text.lower().strip()

        for entity in entities:
            if entity_type and entity.type != entity_type:
                continue

            if entity.name.lower() == text_lower:
                return entity

            for alias in entity.aliases:
                if alias.lower() == text_lower:
                    return entity

        return None

    def _get_noun_phrase(self, token) -> str:
        """Get full noun phrase from token."""
        phrase_tokens = [token]

        for child in token.children:
            if child.dep_ in ("compound", "amod", "det"):
                phrase_tokens.append(child)

        phrase_tokens.sort(key=lambda t: t.i)
        return " ".join(t.text for t in phrase_tokens)

    def _verb_to_relationship(self, verb: str) -> RelationshipType:
        """Map verb to relationship type."""
        verb_mapping = {
            "work": RelationshipType.WORKS_FOR,
            "employ": RelationshipType.EMPLOYED_BY,
            "found": RelationshipType.FOUNDED,
            "create": RelationshipType.CREATED,
            "develop": RelationshipType.DEVELOPED,
            "invent": RelationshipType.INVENTED,
            "acquire": RelationshipType.ACQUIRED,
            "buy": RelationshipType.ACQUIRED,
            "purchase": RelationshipType.ACQUIRED,
            "use": RelationshipType.USES,
            "produce": RelationshipType.PRODUCES,
            "provide": RelationshipType.PROVIDES,
            "require": RelationshipType.REQUIRES,
            "locate": RelationshipType.LOCATED_IN,
            "base": RelationshipType.LOCATED_IN,
        }
        return verb_mapping.get(verb.lower(), RelationshipType.RELATED_TO)
