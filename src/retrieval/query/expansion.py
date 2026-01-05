"""
Enterprise RAG System - Query Expansion
"""

from typing import Optional

from src.core.logging import LoggerMixin
from src.generation.llm import get_default_llm_client, LLMMessage


class QueryExpander(LoggerMixin):
    """
    Expands queries into multiple variations for better retrieval.

    Techniques:
    - Synonym expansion
    - Question rephrasing
    - Sub-question decomposition
    """

    EXPANSION_PROMPT = """Generate {num_variations} alternative versions of the following query.
Each version should:
- Preserve the original meaning
- Use different words or phrasing
- Be a complete, standalone question

Original query: {query}

Return ONLY the alternative queries, one per line, without numbering or prefixes."""

    def __init__(self, num_variations: int = 3):
        self.num_variations = num_variations

    async def expand(
        self,
        query: str,
        num_variations: Optional[int] = None,
    ) -> list[str]:
        """
        Generate query variations.

        Args:
            query: Original query
            num_variations: Number of variations to generate

        Returns:
            List of query variations (including original)
        """
        num_variations = num_variations or self.num_variations

        try:
            llm = get_default_llm_client()

            response = await llm.generate(
                messages=[
                    LLMMessage(
                        role="user",
                        content=self.EXPANSION_PROMPT.format(
                            num_variations=num_variations,
                            query=query,
                        ),
                    )
                ],
                temperature=0.7,  # Some creativity for variations
                max_tokens=500,
            )

            # Parse variations from response
            variations = [
                line.strip()
                for line in response.content.strip().split("\n")
                if line.strip() and not line.strip().startswith(("-", "*", "•"))
            ]

            # Always include original query first
            all_queries = [query] + variations[:num_variations]

            self.logger.debug(
                "Expanded query",
                original=query,
                num_variations=len(variations),
            )

            return all_queries

        except Exception as e:
            self.logger.warning("Query expansion failed", error=str(e))
            # Return original query on failure
            return [query]


class SubQuestionDecomposer(LoggerMixin):
    """
    Decomposes complex queries into simpler sub-questions.

    Useful for multi-hop reasoning queries.
    """

    DECOMPOSITION_PROMPT = """Break down the following complex question into simpler sub-questions that, when answered, would help answer the main question.

Main question: {query}

Rules:
- Generate 2-4 sub-questions
- Each sub-question should be self-contained
- Sub-questions should be in logical order
- Return ONLY the sub-questions, one per line

Sub-questions:"""

    async def decompose(self, query: str) -> list[str]:
        """
        Decompose a complex query into sub-questions.

        Args:
            query: Complex query to decompose

        Returns:
            List of sub-questions
        """
        try:
            llm = get_default_llm_client()

            response = await llm.generate(
                messages=[
                    LLMMessage(
                        role="user",
                        content=self.DECOMPOSITION_PROMPT.format(query=query),
                    )
                ],
                temperature=0.3,
                max_tokens=500,
            )

            # Parse sub-questions
            sub_questions = []
            for line in response.content.strip().split("\n"):
                line = line.strip()
                # Remove common prefixes
                for prefix in ["- ", "* ", "• ", "1.", "2.", "3.", "4.", "5."]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line and len(line) > 10:  # Filter out very short lines
                    sub_questions.append(line)

            self.logger.debug(
                "Decomposed query",
                original=query,
                num_sub_questions=len(sub_questions),
            )

            return sub_questions if sub_questions else [query]

        except Exception as e:
            self.logger.warning("Query decomposition failed", error=str(e))
            return [query]
