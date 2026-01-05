"""Entity extraction for knowledge graph construction."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Standard entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    METRIC = "METRIC"
    CUSTOM = "CUSTOM"


@dataclass
class EntityMention:
    """A mention of an entity in text."""
    text: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    context: Optional[str] = None  # Surrounding text


@dataclass
class Entity:
    """An extracted entity."""
    id: str
    name: str
    type: EntityType
    mentions: list[EntityMention] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    # Linking information
    canonical_name: Optional[str] = None
    external_ids: dict[str, str] = field(default_factory=dict)  # e.g., {"wikidata": "Q123"}

    # Confidence and provenance
    confidence: float = 1.0
    source_document_id: Optional[str] = None

    def merge_with(self, other: "Entity") -> "Entity":
        """Merge another entity into this one."""
        self.mentions.extend(other.mentions)
        self.aliases.extend(other.aliases)
        self.aliases = list(set(self.aliases))
        self.properties.update(other.properties)
        self.external_ids.update(other.external_ids)
        self.confidence = max(self.confidence, other.confidence)
        return self


class EntityExtractor:
    """Extract entities from text using various methods."""

    def __init__(
        self,
        provider: str = "spacy",  # spacy, transformers, openai, anthropic
        model: Optional[str] = None,
        custom_entity_types: Optional[list[str]] = None,
        config: Optional[dict] = None,
    ):
        self.provider = provider
        self.model = model
        self.custom_entity_types = custom_entity_types or []
        self.config = config or {}
        self._initialized = False
        self._nlp = None

    async def initialize(self) -> None:
        """Initialize the entity extractor."""
        if self._initialized:
            return

        if self.provider == "spacy":
            await self._init_spacy()
        elif self.provider == "transformers":
            await self._init_transformers()
        elif self.provider in ["openai", "anthropic"]:
            self._initialized = True
        else:
            self._initialized = True

    async def _init_spacy(self) -> None:
        """Initialize spaCy NER."""
        try:
            import spacy
            model_name = self.model or "en_core_web_sm"

            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                # Model not installed, try downloading
                logger.info(f"Downloading spaCy model: {model_name}")
                await asyncio.to_thread(spacy.cli.download, model_name)
                self._nlp = spacy.load(model_name)

            self._initialized = True
        except ImportError:
            logger.warning("spaCy not installed")

    async def _init_transformers(self) -> None:
        """Initialize transformers NER pipeline."""
        try:
            from transformers import pipeline

            model_name = self.model or "dslim/bert-base-NER"
            self._nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
            self._initialized = True
        except ImportError:
            logger.warning("transformers not installed")

    async def extract(
        self,
        text: str,
        document_id: Optional[str] = None,
        **kwargs,
    ) -> list[Entity]:
        """Extract entities from text."""
        if not self._initialized:
            await self.initialize()

        if self.provider == "spacy":
            return await self._extract_spacy(text, document_id)
        elif self.provider == "transformers":
            return await self._extract_transformers(text, document_id)
        elif self.provider == "openai":
            return await self._extract_openai(text, document_id)
        elif self.provider == "anthropic":
            return await self._extract_anthropic(text, document_id)
        else:
            return await self._extract_regex(text, document_id)

    async def _extract_spacy(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> list[Entity]:
        """Extract entities using spaCy."""
        if not self._nlp:
            return []

        doc = await asyncio.to_thread(self._nlp, text)

        entities = {}
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            key = (ent.text.lower(), entity_type)

            mention = EntityMention(
                text=ent.text,
                start_char=ent.start_char,
                end_char=ent.end_char,
                context=text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)],
            )

            if key not in entities:
                entities[key] = Entity(
                    id=str(uuid4()),
                    name=ent.text,
                    type=entity_type,
                    mentions=[mention],
                    source_document_id=document_id,
                )
            else:
                entities[key].mentions.append(mention)

        return list(entities.values())

    async def _extract_transformers(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> list[Entity]:
        """Extract entities using transformers NER."""
        if not self._nlp:
            return []

        results = await asyncio.to_thread(self._nlp, text)

        entities = {}
        for item in results:
            entity_type = self._map_ner_label(item["entity_group"])
            key = (item["word"].lower(), entity_type)

            mention = EntityMention(
                text=item["word"],
                start_char=item["start"],
                end_char=item["end"],
                confidence=item["score"],
            )

            if key not in entities:
                entities[key] = Entity(
                    id=str(uuid4()),
                    name=item["word"],
                    type=entity_type,
                    mentions=[mention],
                    confidence=item["score"],
                    source_document_id=document_id,
                )
            else:
                entities[key].mentions.append(mention)
                entities[key].confidence = max(entities[key].confidence, item["score"])

        return list(entities.values())

    async def _extract_openai(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> list[Entity]:
        """Extract entities using OpenAI."""
        try:
            from openai import AsyncOpenAI

            settings = get_settings()
            client = AsyncOpenAI(api_key=settings.openai_api_key)

            prompt = f"""Extract all named entities from the following text.
For each entity, provide:
- name: The entity text
- type: One of PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, EVENT, CONCEPT, TECHNOLOGY
- aliases: Any alternative names or abbreviations

Return as JSON array.

Text:
{text[:4000]}

Entities:"""

            response = await client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=2000,
            )

            import json
            content = response.choices[0].message.content
            data = json.loads(content)

            entities = []
            for item in data.get("entities", data if isinstance(data, list) else []):
                entity_type = EntityType(item.get("type", "CONCEPT").upper())
                entities.append(Entity(
                    id=str(uuid4()),
                    name=item.get("name", ""),
                    type=entity_type,
                    aliases=item.get("aliases", []),
                    source_document_id=document_id,
                ))

            return entities

        except Exception as e:
            logger.error(f"OpenAI entity extraction failed: {e}")
            return []

    async def _extract_anthropic(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> list[Entity]:
        """Extract entities using Anthropic Claude."""
        try:
            from anthropic import AsyncAnthropic

            settings = get_settings()
            client = AsyncAnthropic(api_key=settings.anthropic_api_key)

            prompt = f"""Extract all named entities from the following text.
For each entity, provide:
- name: The entity text
- type: One of PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, EVENT, CONCEPT, TECHNOLOGY
- aliases: Any alternative names or abbreviations

Return ONLY a valid JSON array, no other text.

Text:
{text[:4000]}"""

            response = await client.messages.create(
                model=self.model or "claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            import json
            content = response.content[0].text

            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            entities = []
            for item in data:
                try:
                    entity_type = EntityType(item.get("type", "CONCEPT").upper())
                except ValueError:
                    entity_type = EntityType.CONCEPT

                entities.append(Entity(
                    id=str(uuid4()),
                    name=item.get("name", ""),
                    type=entity_type,
                    aliases=item.get("aliases", []),
                    source_document_id=document_id,
                ))

            return entities

        except Exception as e:
            logger.error(f"Anthropic entity extraction failed: {e}")
            return []

    async def _extract_regex(
        self,
        text: str,
        document_id: Optional[str] = None,
    ) -> list[Entity]:
        """Basic regex-based entity extraction."""
        entities = []

        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append(Entity(
                id=str(uuid4()),
                name=match.group(),
                type=EntityType.PERSON,
                mentions=[EntityMention(
                    text=match.group(),
                    start_char=match.start(),
                    end_char=match.end(),
                )],
                source_document_id=document_id,
            ))

        # URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        for match in re.finditer(url_pattern, text):
            entities.append(Entity(
                id=str(uuid4()),
                name=match.group(),
                type=EntityType.TECHNOLOGY,
                source_document_id=document_id,
            ))

        # Dates (basic patterns)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                id=str(uuid4()),
                name=match.group(),
                type=EntityType.DATE,
                source_document_id=document_id,
            ))

        # Money amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?)\b'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                id=str(uuid4()),
                name=match.group(),
                type=EntityType.MONEY,
                source_document_id=document_id,
            ))

        return entities

    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy NER label to EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.WORK_OF_ART,
            "LAW": EntityType.LAW,
            "LANGUAGE": EntityType.LANGUAGE,
            "NORP": EntityType.ORGANIZATION,  # Nationalities, religious, political groups
            "FAC": EntityType.LOCATION,  # Facilities
            "CARDINAL": EntityType.METRIC,
            "ORDINAL": EntityType.METRIC,
            "QUANTITY": EntityType.METRIC,
        }
        return mapping.get(label.upper(), EntityType.CONCEPT)

    def _map_ner_label(self, label: str) -> EntityType:
        """Map transformers NER label to EntityType."""
        mapping = {
            "PER": EntityType.PERSON,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "MISC": EntityType.CONCEPT,
        }
        return mapping.get(label.upper(), EntityType.CONCEPT)


class EntityLinker:
    """Link entities to external knowledge bases."""

    def __init__(
        self,
        knowledge_base: str = "wikidata",  # wikidata, dbpedia, custom
        config: Optional[dict] = None,
    ):
        self.knowledge_base = knowledge_base
        self.config = config or {}
        self._cache: dict[str, dict] = {}

    async def link(
        self,
        entity: Entity,
        **kwargs,
    ) -> Entity:
        """Link entity to external knowledge base."""
        cache_key = f"{entity.name}:{entity.type.value}"

        if cache_key in self._cache:
            entity.external_ids = self._cache[cache_key]
            return entity

        if self.knowledge_base == "wikidata":
            entity = await self._link_wikidata(entity)

        self._cache[cache_key] = entity.external_ids
        return entity

    async def _link_wikidata(self, entity: Entity) -> Entity:
        """Link entity to Wikidata."""
        try:
            import aiohttp

            search_url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "search": entity.name,
                "language": "en",
                "format": "json",
                "limit": 1,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("search", [])

                        if results:
                            result = results[0]
                            entity.external_ids["wikidata"] = result["id"]
                            entity.canonical_name = result.get("label", entity.name)

                            if "description" in result:
                                entity.properties["description"] = result["description"]

        except Exception as e:
            logger.warning(f"Wikidata linking failed for {entity.name}: {e}")

        return entity

    async def link_batch(
        self,
        entities: list[Entity],
        **kwargs,
    ) -> list[Entity]:
        """Link multiple entities."""
        tasks = [self.link(entity, **kwargs) for entity in entities]
        return await asyncio.gather(*tasks)
