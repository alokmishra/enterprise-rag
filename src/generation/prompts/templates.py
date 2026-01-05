"""
Enterprise RAG System - Prompt Templates
"""

from dataclasses import dataclass
from typing import Optional

from src.core.types import ContextItem


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    template: str
    name: str
    description: str = ""

    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)


# =============================================================================
# RAG Prompts
# =============================================================================

RAG_SYSTEM_PROMPT = PromptTemplate(
    name="rag_system",
    description="System prompt for RAG responses",
    template="""You are a helpful assistant that answers questions based on the provided context.

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise and direct in your responses
- Cite sources using [1], [2], etc. format when referencing specific information
- If you're uncertain, express your level of confidence
- Do not make up information that isn't in the context"""
)

RAG_USER_PROMPT = PromptTemplate(
    name="rag_user",
    description="User prompt template for RAG queries",
    template="""Context:
{context}

Question: {question}

Please answer the question based on the context provided above. Include citations [1], [2], etc. when referencing specific sources."""
)

RAG_USER_PROMPT_WITH_HISTORY = PromptTemplate(
    name="rag_user_with_history",
    description="User prompt with conversation history",
    template="""Previous conversation:
{history}

Context:
{context}

Question: {question}

Please answer the question based on the context provided above, taking into account the conversation history. Include citations [1], [2], etc. when referencing specific sources."""
)


# =============================================================================
# Query Analysis Prompts
# =============================================================================

QUERY_ANALYSIS_PROMPT = PromptTemplate(
    name="query_analysis",
    description="Analyze query complexity and intent",
    template="""Analyze the following query and determine:
1. Query complexity (simple, standard, complex)
2. Query type (factual, analytical, comparative, procedural)
3. Key concepts to search for
4. Whether conversation history is needed

Query: {query}

Respond in JSON format:
{{
    "complexity": "simple|standard|complex",
    "query_type": "factual|analytical|comparative|procedural",
    "key_concepts": ["concept1", "concept2"],
    "needs_history": true|false,
    "reasoning": "brief explanation"
}}"""
)


# =============================================================================
# Fact Verification Prompts
# =============================================================================

VERIFICATION_PROMPT = PromptTemplate(
    name="verification",
    description="Verify claims against sources",
    template="""Verify the following claim against the provided sources.

Claim: {claim}

Sources:
{sources}

Determine if this claim is:
- SUPPORTED: Direct evidence supports the claim
- PARTIALLY_SUPPORTED: Some evidence, but not complete
- NOT_FOUND: No relevant evidence found
- CONTRADICTED: Evidence contradicts the claim

Respond in JSON format:
{{
    "status": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_FOUND|CONTRADICTED",
    "evidence": "relevant quote or summary from sources",
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}}"""
)


# =============================================================================
# Helper Functions
# =============================================================================

def format_context(
    context_items: list[ContextItem],
    max_length: Optional[int] = None,
) -> str:
    """
    Format context items for inclusion in prompts.

    Args:
        context_items: List of context items to format
        max_length: Optional maximum character length

    Returns:
        Formatted context string with source numbers
    """
    if not context_items:
        return "No relevant context found."

    parts = []
    total_length = 0

    for i, item in enumerate(context_items, 1):
        source_label = item.source or f"Document {item.document_id[:8]}"
        formatted = f"[{i}] Source: {source_label}\n{item.content}\n"

        if max_length and total_length + len(formatted) > max_length:
            break

        parts.append(formatted)
        total_length += len(formatted)

    return "\n".join(parts)


def format_conversation_history(
    history: list[dict[str, str]],
    max_turns: int = 5,
) -> str:
    """
    Format conversation history for inclusion in prompts.

    Args:
        history: List of message dicts with 'role' and 'content'
        max_turns: Maximum number of turns to include

    Returns:
        Formatted history string
    """
    if not history:
        return ""

    # Take last N turns
    recent = history[-max_turns * 2:]  # Each turn has user + assistant

    parts = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")

    return "\n".join(parts)


def build_rag_prompt(
    question: str,
    context_items: list[ContextItem],
    history: Optional[list[dict[str, str]]] = None,
    max_context_length: int = 8000,
) -> tuple[str, str]:
    """
    Build a complete RAG prompt with system and user messages.

    Args:
        question: The user's question
        context_items: Retrieved context items
        history: Optional conversation history
        max_context_length: Maximum context character length

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system = RAG_SYSTEM_PROMPT.template

    context = format_context(context_items, max_context_length)

    if history:
        formatted_history = format_conversation_history(history)
        user = RAG_USER_PROMPT_WITH_HISTORY.format(
            history=formatted_history,
            context=context,
            question=question,
        )
    else:
        user = RAG_USER_PROMPT.format(
            context=context,
            question=question,
        )

    return system, user
