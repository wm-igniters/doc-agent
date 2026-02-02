"""
Retriever module for hybrid search in Qdrant.

Performs:
- Dense search (vector similarity)
- Sparse search (BM25-style keyword matching)
- Reciprocal Rank Fusion (RRF) to combine results
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qdrant_client import AsyncQdrantClient, models

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store."""

    id: str
    content: str
    title: str
    url: str
    section: Optional[str] = None
    source: str = "docs"
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    metadata: Optional[dict[str, Any]] = None


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse search with RRF fusion.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[AsyncQdrantClient] = None
        self._rrf_k = 60  # RRF constant

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=60,  # Increased timeout for large batch upserts
            )
        return self._client

    async def health_check(self) -> bool:
        """Check Qdrant connection."""
        client = await self._get_client()
        collections = await client.get_collections()
        return True

    async def search(
        self,
        dense_vector: np.ndarray,
        sparse_vector: dict[str, float],
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedDocument]:
        """
        Perform hybrid search combining dense and sparse methods.

        Args:
            dense_vector: Query embedding (768d for BGE)
            sparse_vector: Sparse vector {token: weight}
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of retrieved documents sorted by RRF score
        """
        top_k = top_k or self.settings.retrieval_top_k
        client = await self._get_client()
        collection = self.settings.qdrant_collection_name

        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
            if conditions:
                qdrant_filter = models.Filter(must=conditions)

        # Execute hybrid search (dense + sparse)
        dense_results = []
        sparse_results = []

        try:
            # Dense search (semantic similarity)
            dense_response = await client.query_points(
                collection_name=collection,
                query=dense_vector.tolist(),
                using="dense",
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            dense_results = dense_response.points
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")

        try:
            # Sparse search (keyword/lexical matching)
            if sparse_vector:
                # Convert sparse_vector dict to Qdrant sparse format
                # sparse_vector is {token_id: weight} from embedder
                indices = list(sparse_vector.keys())
                values = list(sparse_vector.values())
                
                sparse_response = await client.query_points(
                    collection_name=collection,
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                sparse_results = sparse_response.points
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")

        # Combine with RRF
        combined = self._rrf_fusion(dense_results, sparse_results)

        # Convert to RetrievedDocument objects
        documents = []
        for item in combined[:top_k]:
            payload = item.get("payload", {})
            documents.append(
                RetrievedDocument(
                    id=str(item.get("id", "")),
                    content=payload.get("content", ""),
                    title=payload.get("title", ""),
                    url=payload.get("url", ""),
                    section=payload.get("section"),
                    source=payload.get("source", "docs"),
                    dense_score=item.get("dense_score", 0.0),
                    sparse_score=item.get("sparse_score", 0.0),
                    rrf_score=item.get("rrf_score", 0.0),
                    metadata=payload,
                )
            )

        logger.info(
            f"Retrieved {len(documents)} documents "
            f"(dense: {len(dense_results)}, sparse: {len(sparse_results)})"
        )
        return documents

    def _rrf_fusion(
        self,
        dense_results: list,
        sparse_results: list,
    ) -> list[dict[str, Any]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.

        RRF Score = Σ 1 / (k + rank)
        """
        scores: dict[str, dict[str, Any]] = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = str(result.id)
            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "payload": result.payload or {},
                    "dense_score": result.score,
                    "sparse_score": 0.0,
                    "rrf_score": 0.0,
                }
            scores[doc_id]["rrf_score"] += 1 / (self._rrf_k + rank)
            scores[doc_id]["dense_score"] = result.score

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = str(result.id)
            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "payload": result.payload or {},
                    "dense_score": 0.0,
                    "sparse_score": result.score,
                    "rrf_score": 0.0,
                }
            scores[doc_id]["rrf_score"] += 1 / (self._rrf_k + rank)
            scores[doc_id]["sparse_score"] = result.score

        # Sort by RRF score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        return sorted_results

    async def ensure_collection(self, embedding_dim: int = 768) -> None:
        """
        Create the collection if it doesn't exist.
        Called during indexing.
        """
        client = await self._get_client()
        collection = self.settings.qdrant_collection_name

        # Check if collection exists
        collections = await client.get_collections()
        exists = any(c.name == collection for c in collections.collections)

        if not exists:
            logger.info(f"Creating collection: {collection}")
            await client.create_collection(
                collection_name=collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        ),
                    ),
                },
            )

            # Create payload indexes for filtering
            await client.create_payload_index(
                collection_name=collection,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=collection,
                field_name="doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            logger.info(f"Collection {collection} created successfully")
        else:
            logger.info(f"Collection {collection} already exists")

    async def upsert_batch(
        self,
        points: list[dict[str, Any]],
    ) -> None:
        """
        Upsert a batch of points to the collection.

        Each point should have:
        - id: str
        - dense_vector: list[float]
        - sparse_vector: dict[int, float] (index -> value)
        - payload: dict
        """
        client = await self._get_client()
        collection = self.settings.qdrant_collection_name

        qdrant_points = []
        for point in points:
            sparse = point.get("sparse_vector", {})
            qdrant_points.append(
                models.PointStruct(
                    id=point["id"],
                    vector={
                        "dense": point["dense_vector"],
                        "sparse": models.SparseVector(
                            indices=list(sparse.keys()) if sparse else [],
                            values=list(sparse.values()) if sparse else [],
                        ),
                    },
                    payload=point.get("payload", {}),
                )
            )

        await client.upsert(
            collection_name=collection,
            points=qdrant_points,
        )
        logger.debug(f"Upserted {len(qdrant_points)} points")

    async def delete_collection(self) -> None:
        """Delete the collection (for full reindex)."""
        client = await self._get_client()
        collection = self.settings.qdrant_collection_name

        try:
            await client.delete_collection(collection_name=collection)
            logger.info(f"Deleted collection: {collection}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

    async def close(self) -> None:
        """Close Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
