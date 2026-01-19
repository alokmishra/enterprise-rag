"""
Tests for src/generation/prompts/
"""

import pytest

from src.core.types import ContextItem
from src.generation.prompts.templates import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    PromptTemplate,
    build_rag_prompt,
    format_context,
    format_conversation_history,
)


class TestRAGPromptTemplates:
    """Tests for RAG prompt templates."""

    def test_rag_system_prompt_exists(self):
        """Test that RAG system prompt exists."""
        assert RAG_SYSTEM_PROMPT is not None
        assert isinstance(RAG_SYSTEM_PROMPT, PromptTemplate)
        assert len(RAG_SYSTEM_PROMPT.template) > 0

    def test_rag_user_prompt_exists(self):
        """Test that RAG user prompt template exists."""
        assert RAG_USER_PROMPT is not None
        assert isinstance(RAG_USER_PROMPT, PromptTemplate)
        assert "{question}" in RAG_USER_PROMPT.template

    def test_rag_prompt_includes_context_placeholder(self):
        """Test that RAG prompt includes context placeholder."""
        assert "{context}" in RAG_USER_PROMPT.template


class TestBuildRAGPrompt:
    """Tests for the build_rag_prompt function."""

    def _make_context_items(self, content: str, source: str = "test.pdf") -> list[ContextItem]:
        """Helper to create context items."""
        return [
            ContextItem(
                content=content,
                source=source,
                chunk_id="chunk_1",
                document_id="doc_1",
                relevance_score=0.9,
            )
        ]

    def test_build_rag_prompt(self):
        """Test build_rag_prompt function."""
        context_items = self._make_context_items("The policy states that...")

        system_prompt, user_prompt = build_rag_prompt(
            question="What is the policy?",
            context_items=context_items,
        )
        assert "policy" in user_prompt.lower()
        assert "The policy states" in user_prompt

    def test_build_rag_prompt_includes_question(self):
        """Test that built prompt includes the question."""
        context_items = self._make_context_items("To reset your password, go to settings.")

        system_prompt, user_prompt = build_rag_prompt(
            question="How do I reset my password?",
            context_items=context_items,
        )
        assert "password" in user_prompt.lower()

    def test_build_rag_prompt_includes_context(self):
        """Test that built prompt includes the context."""
        context_items = self._make_context_items("Specific context content here.")

        system_prompt, user_prompt = build_rag_prompt(
            question="Test question",
            context_items=context_items,
        )
        assert "Specific context content here" in user_prompt

    def test_build_rag_prompt_returns_tuple(self):
        """Test build_rag_prompt returns tuple of system and user prompts."""
        context_items = self._make_context_items("Some context.")

        result = build_rag_prompt(
            question="Test question",
            context_items=context_items,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        system_prompt, user_prompt = result
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    def test_build_rag_prompt_with_history(self):
        """Test build_rag_prompt with conversation history."""
        context_items = self._make_context_items("Context content")

        system_prompt, user_prompt = build_rag_prompt(
            question="Follow up question",
            context_items=context_items,
            history=[
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
        )
        assert user_prompt is not None
        assert "First question" in user_prompt or "Previous conversation" in user_prompt


class TestFormatContext:
    """Tests for format_context function."""

    def test_format_context_empty(self):
        """Test format_context with empty list."""
        result = format_context([])
        assert result == "No relevant context found."

    def test_format_context_with_items(self):
        """Test format_context with items."""
        items = [
            ContextItem(
                content="First chunk content",
                source="doc1.pdf",
                chunk_id="chunk_1",
                document_id="doc_1",
                relevance_score=0.9,
            ),
            ContextItem(
                content="Second chunk content",
                source="doc2.pdf",
                chunk_id="chunk_2",
                document_id="doc_2",
                relevance_score=0.8,
            ),
        ]
        result = format_context(items)
        assert "[1]" in result
        assert "[2]" in result
        assert "doc1.pdf" in result
        assert "First chunk content" in result

    def test_format_context_with_max_length(self):
        """Test format_context respects max_length."""
        items = [
            ContextItem(
                content="A" * 1000,
                source="doc.pdf",
                chunk_id="chunk_1",
                document_id="doc_1",
                relevance_score=0.9,
            ),
            ContextItem(
                content="B" * 1000,
                source="doc2.pdf",
                chunk_id="chunk_2",
                document_id="doc_2",
                relevance_score=0.8,
            ),
        ]
        result = format_context(items, max_length=500)
        assert len(result) <= 600  # Some buffer for formatting


class TestFormatConversationHistory:
    """Tests for format_conversation_history function."""

    def test_format_empty_history(self):
        """Test formatting empty history."""
        result = format_conversation_history([])
        assert result == ""

    def test_format_history_with_messages(self):
        """Test formatting history with messages."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_conversation_history(history)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result


class TestPromptTemplate:
    """Tests for PromptTemplate class."""

    def test_prompt_template_format(self):
        """Test PromptTemplate format method."""
        template = PromptTemplate(
            name="test",
            description="Test template",
            template="Hello {name}, welcome to {place}!",
        )
        result = template.format(name="Alice", place="Wonderland")
        assert result == "Hello Alice, welcome to Wonderland!"
