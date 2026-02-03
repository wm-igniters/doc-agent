"""
3-tier caching system for WaveMaker Docs Agent.

Tier 1: Exact match cache (hash of normalized query)
Tier 2: Semantic cache (vector similarity)
Tier 3: Embedding cache (reuse computed embeddings)
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional

import numpy as np
import redis.asyncio as redis

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Multi-tier caching system using Redis.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[redis.Redis] = None
        self._ttl_seconds = self.settings.cache_ttl_hours * 3600
        self._semantic_index_key = "semantic:index"
        self._semantic_max_candidates = self.settings.semantic_cache_max_candidates

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self.settings.redis_url,
                decode_responses=False,  # We handle encoding ourselves
            )
        return self._client

    async def ping(self) -> bool:
        """Check Redis connection."""
        client = await self._get_client()
        return await client.ping()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
        # Lowercase, strip, collapse whitespace
        normalized = " ".join(query.lower().strip().split())
        return normalized

    def _hash_query(self, query: str) -> str:
        """Generate hash of normalized query."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    # === Tier 1: Exact Match Cache ===

    async def get_exact(self, query: str) -> Optional[dict[str, Any]]:
        """
        Check for exact match in cache.
        Returns cached response or None.
        """
        try:
            client = await self._get_client()
            key = f"exact:{self._hash_query(query)}"
            cached = await client.get(key)
            if cached:
                logger.info(f"Cache hit (exact): {query[:50]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set_exact(self, query: str, response: dict[str, Any]) -> None:
        """Store response in exact match cache."""
        try:
            client = await self._get_client()
            key = f"exact:{self._hash_query(query)}"
            await client.setex(
                key,
                self._ttl_seconds,
                json.dumps(response),
            )
            logger.debug(f"Cached (exact): {query[:50]}...")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    # === Tier 2: Semantic Cache ===

    async def get_semantic(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> Optional[dict[str, Any]]:
        """
        Check for semantically similar cached query.
        Uses vector similarity comparison.
        """
        try:
            client = await self._get_client()

            # Fetch most recent semantic cache entries
            candidate_keys = await client.zrevrange(
                self._semantic_index_key,
                0,
                self._semantic_max_candidates - 1,
            )
            if not candidate_keys:
                return None

            # Bulk fetch cached payloads
            cached_results = await client.mget(candidate_keys)

            threshold = self.settings.semantic_cache_threshold
            for key, cached_data in zip(candidate_keys, cached_results):
                if not cached_data:
                    # Remove stale entry from index
                    await client.zrem(self._semantic_index_key, key)
                    continue

                cached = json.loads(cached_data)
                cached_embedding = np.array(cached.get("embedding", []))

                if cached_embedding.size == 0:
                    continue

                # Compute cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity >= threshold:
                    logger.info(
                        f"Cache hit (semantic, sim={similarity:.3f}): {query[:50]}..."
                    )
                    return cached.get("response")
            return None
        except Exception as e:
            logger.warning(f"Semantic cache error: {e}")
            return None

    async def set_semantic(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: dict[str, Any],
    ) -> None:
        """Store response with embedding for semantic matching."""
        try:
            client = await self._get_client()
            key = f"semantic:{self._hash_query(query)}"

            data = {
                "query": query,
                "embedding": query_embedding.tolist(),
                "response": response,
            }

            await client.setex(
                key,
                self._ttl_seconds,
                json.dumps(data),
            )
            # Track entry in sorted index to limit scan range
            await client.zadd(self._semantic_index_key, {key: time.time()})
            # Trim index to the most recent N entries
            index_size = await client.zcard(self._semantic_index_key)
            if index_size and index_size > self._semantic_max_candidates * 2:
                excess = index_size - (self._semantic_max_candidates * 2)
                await client.zremrangebyrank(
                    self._semantic_index_key,
                    0,
                    excess - 1,
                )
            logger.debug(f"Cached (semantic): {query[:50]}...")
        except Exception as e:
            logger.warning(f"Semantic cache set error: {e}")

    # === Tier 3: Embedding Cache ===

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        try:
            client = await self._get_client()
            key = f"embed:{self._hash_query(text)}"
            cached = await client.get(key)
            if cached:
                return np.array(json.loads(cached))
            return None
        except Exception as e:
            logger.warning(f"Embedding cache error: {e}")
            return None

    async def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding for text."""
        try:
            client = await self._get_client()
            key = f"embed:{self._hash_query(text)}"
            # Embeddings cached longer (less likely to change)
            await client.setex(
                key,
                self._ttl_seconds * 24,  # 24x longer TTL
                json.dumps(embedding.tolist()),
            )
        except Exception as e:
            logger.warning(f"Embedding cache set error: {e}")

    # === Utilities ===

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def invalidate_all(self) -> int:
        """Clear all cache entries. Returns number of keys deleted."""
        try:
            client = await self._get_client()
            keys = await client.keys("exact:*")
            keys.extend(await client.keys("semantic:*"))
            keys.extend(await client.keys("embed:*"))

            if keys:
                deleted = await client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
            return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
