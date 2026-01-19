"""
Enterprise RAG System - HyDE (Hypothetical Document Embeddings)

HyDE generates a hypothetical document that would answer the query,
then uses that document's embedding for retrieval. This often improves
retrieval quality because the hypothetical document is more similar
to actual documents than the short query.

Reference: https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

from typing import Optional

from src.core.logging import LoggerMixin
from src.generation.llm import get_default_llm_client, LLMMessage
from src.ingestion.embeddings import get_embedding_provider


class HyDEGenerator(LoggerMixin):
    """
    Generates hypothetical documents for improved retrieval.
    """

    HYDE_PROMPT = """Write a detailed passage that would answer the following question.
The passage should be informative, factual, and approximately 100-200 words.
Write as if this passage exists in a knowledge base.

Question: {query}

Passage:"""

    HYDE_PROMPT_WITH_CONTEXT = """Based on the domain context provided, write a detailed passage that would answer the following question.
The passage should be informative, factual, and approximately 100-200 words.
Write as if this passage exists in a knowledge base.

Domain context: {context}

Question: {query}

Passage:"""

    async def generate_hypothetical_document(
        self,
        query: str,
        domain_context: Optional[str] = None,
    ) -> str:
        """
        Generate a hypothetical document for the query.

        Args:
            query: The search query
            domain_context: Optional context about the domain

        Returns:
            Hypothetical document text
        """
        llm = get_default_llm_client()

        if domain_context:
            prompt = self.HYDE_PROMPT_WITH_CONTEXT.format(
                context=domain_context,
                query=query,
            )
        else:
            prompt = self.HYDE_PROMPT.format(query=query)

        try:
            response = await llm.generate(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.5,  # Some creativity but mostly factual
                max_tokens=300,
            )

            hypothetical_doc = response.content.strip()

            self.logger.debug(
                "Generated hypothetical document",
                query_length=len(query),
                doc_length=len(hypothetical_doc),
            )

            return hypothetical_doc

        except Exception as e:
            self.logger.warning("HyDE generation failed", error=str(e))
            # Return query as fallback
            return query

    async def generate_hyde_embedding(
        self,
        query: str,
        domain_context: Optional[str] = None,
    ) -> list[float]:
        """
        Generate embedding from hypothetical document.

        Args:
            query: The search query
            domain_context: Optional domain context

        Returns:
            Embedding vector for the hypothetical document
        """
        # Generate hypothetical document
        hypothetical_doc = await self.generate_hypothetical_document(
            query, domain_context
        )

        # Generate embedding
        embedding_provider = get_embedding_provider()
        embedding = await embedding_provider.embed_text(hypothetical_doc)

        self.logger.info(
            "Generated HyDE embedding",
            query=query[:50],
            doc_preview=hypothetical_doc[:100],
        )

        return embedding

    async def generate_multiple_hyde_embeddings(
        self,
        query: str,
        num_hypotheticals: int = 3,
        domain_context: Optional[str] = None,
    ) -> list[list[float]]:
        """
        Generate multiple hypothetical documents and their embeddings.

        Useful for ensemble retrieval.

        Args:
            query: The search query
            num_hypotheticals: Number of hypothetical documents
            domain_context: Optional domain context

        Returns:
            List of embedding vectors
        """
        import asyncio

        # Generate multiple hypothetical documents
        tasks = [
            self.generate_hypothetical_document(query, domain_context)
            for _ in range(num_hypotheticals)
        ]
        hypothetical_docs = await asyncio.gather(*tasks)

        # Generate embeddings
        embedding_provider = get_embedding_provider()
        result = await embedding_provider.embed_texts(list(hypothetical_docs))

        return result.embeddings
