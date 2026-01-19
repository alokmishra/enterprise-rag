"""
Enterprise RAG System - Qdrant Vector Store Implementation
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.storage.base import VectorStore


class QdrantSearchResult:
    """Search result from Qdrant."""

    def __init__(
        self,
        id: str,
        score: float,
        payload: dict[str, Any],
        vector: Optional[list[float]] = None,
    ):
        self.id = id
        self.score = score
        self.payload = payload
        self.vector = vector

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "payload": self.payload,
            "vector": self.vector,
        }


class QdrantVectorStore(VectorStore[QdrantSearchResult]):
    """Qdrant implementation of vector store."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._url = url or settings.QDRANT_URL
        self._api_key = api_key or settings.QDRANT_API_KEY
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        """Connect to Qdrant server."""
        if self._client is not None:
            return

        try:
            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=30,
            )
            # Verify connection
            await self._client.get_collections()
            self.logger.info("Connected to Qdrant", url=self._url)
        except Exception as e:
            self._client = None
            raise VectorStoreError(f"Failed to connect to Qdrant: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Qdrant server."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self.logger.info("Disconnected from Qdrant")

    async def health_check(self) -> dict[str, Any]:
        """Check Qdrant health."""
        if not self._client:
            return {"status": "disconnected", "latency_ms": 0}

        try:
            start = asyncio.get_event_loop().time()
            await self._client.get_collections()
            latency = (asyncio.get_event_loop().time() - start) * 1000
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "latency_ms": 0}

    def _ensure_connected(self) -> AsyncQdrantClient:
        """Ensure client is connected."""
        if self._client is None:
            raise VectorStoreError("Not connected to Qdrant")
        return self._client

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: str = "cosine",
    ) -> None:
        """Create a new vector collection."""
        client = self._ensure_connected()

        distance_map = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
        }
        distance = distance_map.get(distance_metric.lower(), models.Distance.COSINE)

        try:
            await client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=distance,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )
            self.logger.info(
                "Created Qdrant collection",
                collection=name,
                dimension=dimension,
                distance=distance_metric,
            )
        except UnexpectedResponse as e:
            if "already exists" in str(e):
                self.logger.debug("Collection already exists", collection=name)
            else:
                raise VectorStoreError(f"Failed to create collection: {e}")

    async def delete_collection(self, name: str) -> None:
        """Delete a vector collection."""
        client = self._ensure_connected()

        try:
            await client.delete_collection(collection_name=name)
            self.logger.info("Deleted Qdrant collection", collection=name)
        except UnexpectedResponse as e:
            raise VectorStoreError(f"Failed to delete collection: {e}")

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        client = self._ensure_connected()

        try:
            collections = await client.get_collections()
            return any(c.name == name for c in collections.collections)
        except Exception:
            return False

    async def insert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Insert vectors into a collection."""
        client = self._ensure_connected()

        if len(ids) != len(vectors):
            raise VectorStoreError("IDs and vectors must have same length")

        payloads = payloads or [{} for _ in ids]
        if len(payloads) != len(ids):
            raise VectorStoreError("Payloads must have same length as IDs")

        points = [
            models.PointStruct(
                id=id_,
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]

        try:
            await client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )
            self.logger.debug(
                "Inserted vectors",
                collection=collection,
                count=len(points),
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to insert vectors: {e}")

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[QdrantSearchResult]:
        """Search for similar vectors."""
        client = self._ensure_connected()

        # Build filter if provided
        query_filter = None
        if filters:
            query_filter = self._build_filter(filters)

        try:
            results = await client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            return [
                QdrantSearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete vectors by ID."""
        client = self._ensure_connected()

        try:
            await client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )
            self.logger.debug(
                "Deleted vectors",
                collection=collection,
                count=len(ids),
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to delete vectors: {e}")

    async def get(
        self,
        collection: str,
        ids: list[str],
    ) -> list[QdrantSearchResult]:
        """Get vectors by ID."""
        client = self._ensure_connected()

        try:
            results = await client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=True,
                with_vectors=True,
            )

            return [
                QdrantSearchResult(
                    id=str(r.id),
                    score=1.0,
                    payload=r.payload or {},
                    vector=r.vector if isinstance(r.vector, list) else None,
                )
                for r in results
            ]
        except Exception as e:
            raise VectorStoreError(f"Failed to get vectors: {e}")

    def _build_filter(self, filters: dict[str, Any]) -> models.Filter:
        """Build Qdrant filter from dict."""
        conditions = []

        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            elif isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                            ),
                        )
                    )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions)


# Singleton instance
_vector_store: Optional[QdrantVectorStore] = None


def get_vector_store() -> QdrantVectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store


async def init_vector_store() -> QdrantVectorStore:
    """Initialize and connect the vector store."""
    store = get_vector_store()
    await store.connect()

    # Ensure default collection exists
    if not await store.collection_exists(settings.QDRANT_COLLECTION_NAME):
        await store.create_collection(
            name=settings.QDRANT_COLLECTION_NAME,
            dimension=settings.EMBEDDING_DIMENSIONS,
        )

    return store


async def close_vector_store() -> None:
    """Close the vector store connection."""
    global _vector_store
    if _vector_store is not None:
        await _vector_store.disconnect()
        _vector_store = None
