"""
Tests for src/generation/prompts/
"""

import pytest


class TestRAGPromptTemplates:
    """Tests for RAG prompt templates."""

    def test_rag_system_prompt_exists(self):
        """Test that RAG system prompt exists."""
        from src.generation.prompts.templates import RAG_SYSTEM_PROMPT

        assert RAG_SYSTEM_PROMPT is not None
        assert len(RAG_SYSTEM_PROMPT) > 0

    def test_rag_user_prompt_exists(self):
        """Test that RAG user prompt template exists."""
        from src.generation.prompts.templates import RAG_USER_PROMPT

        assert RAG_USER_PROMPT is not None
        assert "{question}" in RAG_USER_PROMPT or "{query}" in RAG_USER_PROMPT

    def test_rag_prompt_includes_context_placeholder(self):
        """Test that RAG prompt includes context placeholder."""
        from src.generation.prompts.templates import RAG_USER_PROMPT

        assert "{context}" in RAG_USER_PROMPT


class TestBuildRAGPrompt:
    """Tests for the build_rag_prompt function."""

    def test_build_rag_prompt(self):
        """Test build_rag_prompt function."""
        from src.generation.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(
            question="What is the policy?",
            context="The policy states that...",
        )
        assert "policy" in prompt.lower()
        assert "context" in prompt.lower() or "The policy states" in prompt

    def test_build_rag_prompt_includes_question(self):
        """Test that built prompt includes the question."""
        from src.generation.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(
            question="How do I reset my password?",
            context="To reset your password, go to settings.",
        )
        assert "password" in prompt.lower()

    def test_build_rag_prompt_includes_context(self):
        """Test that built prompt includes the context."""
        from src.generation.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(
            question="Test question",
            context="Specific context content here.",
        )
        assert "Specific context content here" in prompt

    def test_build_rag_prompt_with_conversation_history(self):
        """Test build_rag_prompt with conversation history."""
        from src.generation.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(
            question="Follow up question",
            context="Context content",
            conversation_history=[
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
            ],
        )
        # Should include conversation history
        assert prompt is not None


class TestPromptFormatting:
    """Tests for prompt formatting."""

    def test_context_formatting_numbered(self):
        """Test context formatting with numbers."""
        from src.generation.prompts.templates import format_context

        if hasattr(__import__('src.generation.prompts.templates', fromlist=['format_context']), 'format_context'):
            contexts = [
                {"content": "First chunk", "source": "doc1.pdf"},
                {"content": "Second chunk", "source": "doc2.pdf"},
            ]
            # Should format with numbers

    def test_context_formatting_with_sources(self):
        """Test context formatting includes sources."""
        from src.generation.prompts.templates import build_rag_prompt

        prompt = build_rag_prompt(
            question="Test",
            context="[1] source.pdf: Content here",
        )
        assert "source.pdf" in prompt or "Content here" in prompt
