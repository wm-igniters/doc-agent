"""
Reranker module for precise relevance scoring.

Supports two providers:
- Jina AI API (recommended for Apple Silicon)
- Local cross-encoder (may have MLX issues on Apple Silicon)
"""

import logging
from typing import Optional

import httpx

from src.config.settings import get_settings
from src.core.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker with support for multiple providers.
    """

    def __init__(self):
        self.settings = get_settings()
        self._local_model = None
        self._local_tokenizer = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for API calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: Optional[int] = None,
    ) -> list[RetrievedDocument]:
        """
        Rerank documents based on relevance to query.
        Dispatches to configured provider (jina or local).
        """
        if not documents:
            return []

        top_k = top_k or self.settings.rerank_top_k
        provider = self.settings.reranker_provider.lower()

        if provider == "jina":
            return await self._rerank_jina(query, documents, top_k)
        elif provider == "local":
            return self._rerank_local(query, documents, top_k)
        else:
            logger.warning(f"Unknown reranker provider: {provider}, using jina")
            return await self._rerank_jina(query, documents, top_k)

    async def _rerank_jina(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Rerank using Jina AI API."""
        if not self.settings.jina_api_key:
            logger.warning("Jina API key not set, skipping reranking")
            return documents[:top_k]

        try:
            client = self._get_http_client()
            
            # Prepare documents for Jina API
            doc_texts = [doc.content[:2000] for doc in documents]  # Truncate for API
            
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.settings.jina_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "jina-reranker-v3",
                    "query": query,
                    "top_n": top_k,
                    "documents": doc_texts,
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Jina API error: {response.status_code} - {response.text}")
                return documents[:top_k]

            result = response.json()
            
            # Build reranked list based on Jina's response
            reranked = []
            for item in result.get("results", []):
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                
                if idx < len(documents):
                    doc = documents[idx]
                    reranked.append(
                        RetrievedDocument(
                            id=doc.id,
                            content=doc.content,
                            title=doc.title,
                            url=doc.url,
                            section=doc.section,
                            source=doc.source,
                            dense_score=doc.dense_score,
                            sparse_score=doc.sparse_score,
                            rrf_score=float(score),  # Use Jina score
                            metadata=doc.metadata,
                        )
                    )

            logger.info(
                f"Jina reranked {len(documents)} -> {len(reranked)} documents "
                f"(top score: {reranked[0].rrf_score:.3f})"
            )
            return reranked

        except Exception as e:
            logger.exception(f"Jina reranking error: {e}")
            return documents[:top_k]

    async def _rerank_local(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        """
        Rerank using local cross-encoder model.
        Runs in a thread pool to avoid blocking.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        def _compute():
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Lazy load model
            if self._local_model is None:
                logger.info(f"Loading local reranker model: {self.settings.reranker_model}")
                self._local_tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.reranker_model
                )
                self._local_model = AutoModelForSequenceClassification.from_pretrained(
                    self.settings.reranker_model
                )
                self._local_model = self._local_model.to("cpu")
                self._local_model.eval()

            # Filter empty documents
            valid_docs = [doc for doc in documents if doc.content and doc.content.strip()]
            if not valid_docs:
                return []

            # Prepare pairs
            pairs = [(query, doc.content[:2000]) for doc in valid_docs]
            queries = [p[0] for p in pairs]
            doc_texts = [p[1] for p in pairs]

            # Tokenize and score
            inputs = self._local_tokenizer(
                queries,
                doc_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._local_model(**inputs)
                scores = outputs.logits.squeeze(-1).tolist()

            if isinstance(scores, float):
                scores = [scores]

            # Log scores
            logger.info(f"Local rerank scores: {[f'{s:.3f}' for s in scores[:3]]}")

            # Sort and return top_k
            scored_docs = list(zip(valid_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in scored_docs[:top_k]:
                reranked.append(
                    RetrievedDocument(
                        id=doc.id,
                        content=doc.content,
                        title=doc.title,
                        url=doc.url,
                        section=doc.section,
                        source=doc.source,
                        dense_score=doc.dense_score,
                        sparse_score=doc.sparse_score,
                        rrf_score=float(score),
                        metadata=doc.metadata,
                    )
                )

            top_score = reranked[0].rrf_score if reranked else 0.0
            logger.info(
                f"Local reranked {len(documents)} -> {len(reranked)} documents "
                f"(top score: {top_score:.3f})"
            )
            return reranked

        return await loop.run_in_executor(None, _compute)

    def should_rerank(self, documents: list[RetrievedDocument]) -> bool:
        """Decide if reranking is needed."""
        if not documents:
            return False

        # Skip if reranking is disabled
        if not self.settings.reranker_enabled:
            return False

        top_score = documents[0].rrf_score if documents else 0

        # Skip reranking if top result is very confident
        if top_score > 0.04:
            logger.debug(f"Skipping rerank: top RRF score {top_score:.4f} is high")
            return False

        return True

    async def close(self) -> None:
        """Cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
