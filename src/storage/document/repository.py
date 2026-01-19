"""
Enterprise RAG System - Document Repository
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import LoggerMixin
from src.core.types import Document, DocumentMetadata, Chunk, DocumentStatus, ContentType
from src.storage.document.models import DocumentModel, ChunkModel, QueryLogModel


class DocumentRepository(LoggerMixin):
    """Repository for document operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, document: Document) -> str:
        """Create a new document."""
        doc_model = DocumentModel(
            id=document.id,
            content=document.content,
            status=document.status,
            source=document.metadata.source,
            source_id=document.metadata.source_id,
            title=document.metadata.title,
            author=document.metadata.author,
            file_type=document.metadata.file_type,
            file_size=document.metadata.file_size,
            language=document.metadata.language,
            custom_metadata=document.metadata.custom,
            document_created_at=document.metadata.created_at,
            document_modified_at=document.metadata.modified_at,
        )

        self._session.add(doc_model)
        await self._session.flush()

        self.logger.debug("Created document", document_id=document.id)
        return document.id

    async def get(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        result = await self._session.execute(
            select(DocumentModel)
            .where(DocumentModel.id == document_id)
            .options(selectinload(DocumentModel.chunks))
        )
        doc_model = result.scalar_one_or_none()

        if doc_model is None:
            return None

        return self._to_domain(doc_model)

    async def get_by_source(self, source: str, source_id: str) -> Optional[Document]:
        """Get a document by source and source_id."""
        result = await self._session.execute(
            select(DocumentModel)
            .where(
                DocumentModel.source == source,
                DocumentModel.source_id == source_id,
            )
            .options(selectinload(DocumentModel.chunks))
        )
        doc_model = result.scalar_one_or_none()

        if doc_model is None:
            return None

        return self._to_domain(doc_model)

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[DocumentStatus] = None,
        source: Optional[str] = None,
    ) -> list[Document]:
        """List documents with optional filtering."""
        query = select(DocumentModel).order_by(DocumentModel.created_at.desc())

        if status:
            query = query.where(DocumentModel.status == status)
        if source:
            query = query.where(DocumentModel.source == source)

        query = query.offset(offset).limit(limit)
        result = await self._session.execute(query)
        doc_models = result.scalars().all()

        return [self._to_domain(m, include_chunks=False) for m in doc_models]

    async def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
    ) -> bool:
        """Update document status."""
        result = await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(status=status)
        )
        return result.rowcount > 0

    async def update(
        self,
        document_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update document fields."""
        result = await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(**updates)
        )
        return result.rowcount > 0

    async def delete(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        result = await self._session.execute(
            delete(DocumentModel).where(DocumentModel.id == document_id)
        )
        return result.rowcount > 0

    async def count(
        self,
        status: Optional[DocumentStatus] = None,
        source: Optional[str] = None,
    ) -> int:
        """Count documents with optional filtering."""
        query = select(func.count(DocumentModel.id))

        if status:
            query = query.where(DocumentModel.status == status)
        if source:
            query = query.where(DocumentModel.source == source)

        result = await self._session.execute(query)
        return result.scalar() or 0

    def _to_domain(
        self,
        model: DocumentModel,
        include_chunks: bool = True,
    ) -> Document:
        """Convert database model to domain object."""
        chunks = []
        if include_chunks and model.chunks:
            chunks = [
                Chunk(
                    id=c.id,
                    document_id=c.document_id,
                    content=c.content,
                    content_type=c.content_type,
                    position=c.position,
                    metadata=c.metadata,
                    parent_id=c.parent_id,
                )
                for c in sorted(model.chunks, key=lambda x: x.position)
            ]

        return Document(
            id=model.id,
            content=model.content,
            status=model.status,
            metadata=DocumentMetadata(
                source=model.source,
                source_id=model.source_id,
                title=model.title,
                author=model.author,
                file_type=model.file_type,
                file_size=model.file_size,
                language=model.language,
                created_at=model.document_created_at,
                modified_at=model.document_modified_at,
                custom=model.custom_metadata or {},
            ),
            chunks=chunks,
        )


class ChunkRepository(LoggerMixin):
    """Repository for chunk operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_many(self, chunks: list[Chunk]) -> list[str]:
        """Create multiple chunks."""
        chunk_models = [
            ChunkModel(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                content_type=chunk.content_type,
                position=chunk.position,
                parent_id=chunk.parent_id,
                chunk_metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        self._session.add_all(chunk_models)
        await self._session.flush()

        return [c.id for c in chunks]

    async def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        result = await self._session.execute(
            select(ChunkModel).where(ChunkModel.id == chunk_id)
        )
        chunk_model = result.scalar_one_or_none()

        if chunk_model is None:
            return None

        return self._to_domain(chunk_model)

    async def get_by_document(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.position)
        )
        chunk_models = result.scalars().all()

        return [self._to_domain(m) for m in chunk_models]

    async def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get multiple chunks by ID."""
        result = await self._session.execute(
            select(ChunkModel).where(ChunkModel.id.in_(chunk_ids))
        )
        chunk_models = result.scalars().all()

        return [self._to_domain(m) for m in chunk_models]

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        result = await self._session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document_id)
        )
        return result.rowcount

    def _to_domain(self, model: ChunkModel) -> Chunk:
        """Convert database model to domain object."""
        return Chunk(
            id=model.id,
            document_id=model.document_id,
            content=model.content,
            content_type=model.content_type,
            position=model.position,
            metadata=model.chunk_metadata or {},
            parent_id=model.parent_id,
        )


class QueryLogRepository(LoggerMixin):
    """Repository for query log operations."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        query: str,
        answer: Optional[str] = None,
        conversation_id: Optional[str] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        confidence: Optional[float] = None,
        trace_data: Optional[dict] = None,
        sources_used: Optional[list] = None,
    ) -> str:
        """Create a query log entry."""
        query_id = str(uuid4())

        log_model = QueryLogModel(
            id=query_id,
            query=query,
            answer=answer,
            conversation_id=conversation_id,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            confidence=confidence,
            trace_data=trace_data or {},
            sources_used=sources_used or [],
        )

        self._session.add(log_model)
        await self._session.flush()

        return query_id

    async def get(self, query_id: str) -> Optional[QueryLogModel]:
        """Get a query log by ID."""
        result = await self._session.execute(
            select(QueryLogModel).where(QueryLogModel.id == query_id)
        )
        return result.scalar_one_or_none()

    async def update_feedback(
        self,
        query_id: str,
        rating: int,
        feedback: Optional[str] = None,
    ) -> bool:
        """Update feedback for a query."""
        result = await self._session.execute(
            update(QueryLogModel)
            .where(QueryLogModel.id == query_id)
            .values(rating=rating, feedback=feedback)
        )
        return result.rowcount > 0
