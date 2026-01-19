"""Tests for relationship extraction."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.knowledge_graph.extraction.relationships import (
    RelationshipExtractor,
    Relationship,
    RelationshipType,
)
from src.knowledge_graph.extraction.entities import Entity, EntityType


class TestRelationshipType:
    """Tests for RelationshipType enum."""

    def test_relationship_types_exist(self):
        """Test that all relationship types exist."""
        assert RelationshipType.WORKS_FOR
        assert RelationshipType.FOUNDED
        assert RelationshipType.LOCATED_IN
        assert RelationshipType.RELATED_TO
        assert RelationshipType.ACQUIRED

    def test_relationship_type_values(self):
        """Test relationship type string values."""
        assert RelationshipType.WORKS_FOR.value == "WORKS_FOR"
        assert RelationshipType.FOUNDED.value == "FOUNDED"


class TestRelationship:
    """Tests for Relationship class."""

    def test_create_relationship(self):
        """Test creating a relationship."""
        rel = Relationship(
            id=str(uuid4()),
            source_entity_id="e1",
            target_entity_id="e2",
            type=RelationshipType.WORKS_FOR,
            label="John works for Apple",
            confidence=0.9,
        )

        assert rel.source_entity_id == "e1"
        assert rel.target_entity_id == "e2"
        assert rel.type == RelationshipType.WORKS_FOR

    def test_reverse_relationship(self):
        """Test reversing a relationship."""
        rel = Relationship(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            type=RelationshipType.WORKS_FOR,
            label="John works for Apple",
        )

        reversed_rel = rel.reverse()

        assert reversed_rel.source_entity_id == "e2"
        assert reversed_rel.target_entity_id == "e1"
        assert reversed_rel.type == RelationshipType.EMPLOYED_BY


class TestRelationshipExtractor:
    """Tests for RelationshipExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create relationship extractor."""
        return RelationshipExtractor(provider="pattern")

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities."""
        return [
            Entity(
                id="e1",
                name="Tim Cook",
                type=EntityType.PERSON,
                mentions=[],
            ),
            Entity(
                id="e2",
                name="Apple",
                type=EntityType.ORGANIZATION,
                mentions=[],
            ),
            Entity(
                id="e3",
                name="Cupertino",
                type=EntityType.LOCATION,
                mentions=[],
            ),
        ]

    @pytest.fixture
    def sample_text(self):
        """Sample text for extraction."""
        return "Tim Cook works for Apple. Apple is headquartered in Cupertino."

    @pytest.mark.asyncio
    async def test_initialize(self, extractor):
        """Test extractor initialization."""
        await extractor.initialize()
        assert extractor._initialized

    @pytest.mark.asyncio
    async def test_extract_pattern(self, extractor, sample_text, sample_entities):
        """Test pattern-based extraction."""
        extractor._initialized = True

        relationships = await extractor._extract_pattern(
            sample_text,
            sample_entities,
            "doc-1",
        )

        # Should find some relationships
        assert isinstance(relationships, list)

    @pytest.mark.asyncio
    async def test_extract_llm(self, sample_text, sample_entities):
        """Test LLM-based extraction."""
        extractor = RelationshipExtractor(provider="llm")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(content='{"relationships": [{"source": "Tim Cook", "target": "Apple", "type": "WORKS_FOR"}]}')
        )]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            extractor._initialized = True
            relationships = await extractor.extract(
                sample_text,
                sample_entities,
                "doc-1",
            )

            assert isinstance(relationships, list)

    @pytest.mark.asyncio
    async def test_extract_with_few_entities(self, extractor):
        """Test extraction with insufficient entities."""
        extractor._initialized = True

        relationships = await extractor.extract(
            "Some text",
            [Entity(id="e1", name="Single", type=EntityType.CONCEPT)],
            "doc-1",
        )

        # Should return empty list with < 2 entities
        assert relationships == []

    def test_verb_to_relationship(self, extractor):
        """Test verb to relationship type mapping."""
        assert extractor._verb_to_relationship("work") == RelationshipType.WORKS_FOR
        assert extractor._verb_to_relationship("found") == RelationshipType.FOUNDED
        assert extractor._verb_to_relationship("acquire") == RelationshipType.ACQUIRED
        assert extractor._verb_to_relationship("unknown") == RelationshipType.RELATED_TO

    def test_find_entity(self, extractor, sample_entities):
        """Test finding entity by text."""
        entity = extractor._find_entity(
            "Tim Cook",
            sample_entities,
            EntityType.PERSON,
        )

        assert entity is not None
        assert entity.name == "Tim Cook"

    def test_find_entity_not_found(self, extractor, sample_entities):
        """Test finding entity that doesn't exist."""
        entity = extractor._find_entity(
            "Unknown Person",
            sample_entities,
            EntityType.PERSON,
        )

        assert entity is None
