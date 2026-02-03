"""
Main RAG pipeline orchestrator.

Ties together all layers:
1. Cache check
2. Query embedding
3. Hybrid retrieval
4. Reranking
5. Response generation
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Optional

import numpy as np

from src.config.settings import get_settings
from src.core.academy import AcademyClient
from src.core.cache import SemanticCache
from src.core.embedder import Embedder
from src.core.generator import ResponseGenerator
from src.core.reranker import Reranker
from src.core.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Main RAG pipeline for answering documentation queries.

    Flow:
    1. Check cache (exact match, then semantic)
    2. Generate query embeddings (dense + sparse)
    3. Retrieve documents from Qdrant + videos from Academy (parallel)
    4. Rerank documents with cross-encoder (videos not reranked)
    5. Generate response with Claude
    6. Cache the response
    """

    def __init__(self):
        self.settings = get_settings()
        self.cache = SemanticCache()
        self.embedder = Embedder()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.academy = AcademyClient()

    async def query(
        self,
        query: str,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """
        Process a query and return complete response.

        Args:
            query: User's question
            include_sources: Whether to include source citations

        Returns:
            Dict with answer, sources, videos, cached flag
        """
        timings: dict[str, float] = {}
        start_total = time.perf_counter()

        # Layer 1: Check exact cache
        start = time.perf_counter()
        cached = await self.cache.get_exact(query)
        timings["cache_exact"] = time.perf_counter() - start
        if cached:
            cached["cached"] = True
            timings["total"] = time.perf_counter() - start_total
            logger.info(f"Timing(query='{query[:40]}...'): {timings}")
            return cached

        # Layer 2: Generate embeddings
        start = time.perf_counter()
        dense_vector = await self.embedder.embed_query(query)
        sparse_vector = self.embedder.generate_sparse_vector(query)
        timings["embedding"] = time.perf_counter() - start

        # Check semantic cache
        start = time.perf_counter()
        semantic_cached = await self.cache.get_semantic(query, dense_vector)
        timings["cache_semantic"] = time.perf_counter() - start
        if semantic_cached:
            semantic_cached["cached"] = True
            timings["total"] = time.perf_counter() - start_total
            logger.info(f"Timing(query='{query[:40]}...'): {timings}")
            return semantic_cached

        # Layer 3: Retrieve documents and videos in parallel
        start = time.perf_counter()
        documents_task = self.retriever.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        videos_task = self.academy.search_videos(query, limit=3)

        # Execute both searches concurrently
        results = await asyncio.gather(
            documents_task,
            videos_task,
            return_exceptions=True,
        )

        # Handle results and exceptions (including CancelledError which inherits from BaseException)
        documents = results[0] if isinstance(results[0], list) else []
        videos = results[1] if isinstance(results[1], list) else []

        # Log any failures
        if not isinstance(results[0], list):
            logger.error(f"Document retrieval failed ({type(results[0]).__name__}): {results[0]}")
        if not isinstance(results[1], list):
            logger.warning(f"Video search failed ({type(results[1]).__name__}): {results[1]}")
        timings["retrieval"] = time.perf_counter() - start

        # Layer 4: Rerank documents
        start = time.perf_counter()
        if documents and self.reranker.should_rerank(documents):
            # Trim candidate list to reduce reranker payload
            candidate_limit = max(self.settings.reranker_candidates, self.settings.rerank_top_k)
            documents = documents[:candidate_limit]
            documents = await self.reranker.rerank(query, documents)
        elif documents:
            # Just take top_k without reranking
            documents = documents[:self.settings.rerank_top_k]
        if documents:
            documents = documents[:self.settings.rerank_top_k]
        timings["rerank"] = time.perf_counter() - start

        # Layer 5: Generate response
        start = time.perf_counter()
        if not documents:
            response = {
                "answer": "I don't have information about this topic in the documentation. Please try rephrasing your question or check the WaveMaker documentation directly.",
                "sources": [],
                "videos": [],
                "cached": False,
            }
        else:
            result = await self.generator.generate(
                query=query,
                documents=documents if include_sources else documents[:3],
                videos=videos,
            )
            response = {
                "answer": result["answer"],
                "sources": [s.model_dump() for s in result["sources"]] if include_sources else [],
                "videos": [v.model_dump() for v in result["videos"]],
                "cached": False,
            }
        timings["generation"] = time.perf_counter() - start

        # Cache the response
        start = time.perf_counter()
        await self.cache.set_exact(query, response)
        await self.cache.set_semantic(query, dense_vector, response)
        timings["cache_write"] = time.perf_counter() - start
        timings["total"] = time.perf_counter() - start_total
        logger.info(f"Timing(query='{query[:40]}...'): {timings}")

        return response

    async def query_stream(
        self,
        query: str,
        include_sources: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process a query and stream the response.

        Yields:
            Dict chunks with type and content
        """
        # Layer 1: Check exact cache
        cached = await self.cache.get_exact(query)
        if cached:
            # Stream cached response
            yield {"type": "text", "content": cached.get("answer", "")}
            if include_sources and cached.get("sources"):
                yield {"type": "sources", "sources": cached["sources"]}
            if cached.get("videos"):
                yield {"type": "videos", "videos": cached["videos"]}
            yield {"type": "done", "cached": True}
            return

        # Layer 2: Generate embeddings
        dense_vector = await self.embedder.embed_query(query)
        sparse_vector = self.embedder.generate_sparse_vector(query)

        # Check semantic cache
        semantic_cached = await self.cache.get_semantic(query, dense_vector)
        if semantic_cached:
            yield {"type": "text", "content": semantic_cached.get("answer", "")}
            if include_sources and semantic_cached.get("sources"):
                yield {"type": "sources", "sources": semantic_cached["sources"]}
            if semantic_cached.get("videos"):
                yield {"type": "videos", "videos": semantic_cached["videos"]}
            yield {"type": "done", "cached": True}
            return

        # Layer 3: Retrieve documents and videos in parallel
        documents_task = self.retriever.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        videos_task = self.academy.search_videos(query, limit=3)

        # Execute both searches concurrently
        results = await asyncio.gather(
            documents_task,
            videos_task,
            return_exceptions=True,
        )

        # Handle results and exceptions (including CancelledError which inherits from BaseException)
        documents = results[0] if isinstance(results[0], list) else []
        videos = results[1] if isinstance(results[1], list) else []

        # Log any failures
        if not isinstance(results[0], list):
            logger.error(f"Document retrieval failed ({type(results[0]).__name__}): {results[0]}")
        if not isinstance(results[1], list):
            logger.warning(f"Video search failed ({type(results[1]).__name__}): {results[1]}")

        # Layer 4: Rerank documents
        if documents and self.reranker.should_rerank(documents):
            documents = await self.reranker.rerank(query, documents)
        elif documents:
            # Just take top_k without reranking
            documents = documents[:self.settings.rerank_top_k]

        # Layer 5: Stream response
        if not documents:
            yield {
                "type": "text",
                "content": "I don't have information about this topic in the documentation. Please try rephrasing your question or check the WaveMaker documentation directly.",
            }
            yield {"type": "sources", "sources": []}
            yield {"type": "videos", "videos": []}
            yield {"type": "done", "cached": False}
            return

        # Collect full response for caching
        full_answer = ""
        sources_data = []
        videos_data = []

        async for chunk in self.generator.generate_stream(
            query=query,
            documents=documents if include_sources else documents[:3],
            videos=videos,
        ):
            yield chunk

            # Collect for caching
            if chunk.get("type") == "text":
                full_answer += chunk.get("content", "")
            elif chunk.get("type") == "sources":
                sources_data = chunk.get("sources", [])
            elif chunk.get("type") == "videos":
                videos_data = chunk.get("videos", [])

        # Cache the complete response
        response = {
            "answer": full_answer,
            "sources": sources_data,
            "videos": videos_data,
            "cached": False,
        }
        await self.cache.set_exact(query, response)
        await self.cache.set_semantic(query, dense_vector, response)

    async def close(self) -> None:
        """Clean up resources."""
        await self.cache.close()
        await self.retriever.close()
        await self.academy.close()
