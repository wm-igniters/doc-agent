# WaveMaker Docs Agent - Technical Architecture

A comprehensive technical deep-dive into the RAG system architecture, implementation details, and design decisions.

**Last Updated:** 2026-02-02

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Document Processing Pipeline](#document-processing-pipeline)
3. [Embedding Strategy](#embedding-strategy)
4. [Vector Database Design](#vector-database-design)
5. [Retrieval Strategy](#retrieval-strategy)
6. [Academy MCP Integration](#academy-mcp-integration)
7. [Parallel Execution Architecture](#parallel-execution-architecture)
8. [Reranking Layer](#reranking-layer)
9. [Caching Architecture](#caching-architecture)
10. [LLM Generation](#llm-generation)
11. [Performance Optimizations](#performance-optimizations)
12. [Error Handling & Resilience](#error-handling--resilience)
13. [Evaluation & Metrics](#evaluation--metrics)

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING PIPELINE                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Clone   │→ │  Parse   │→ │  Chunk   │→ │  Embed   │→ │  Store   │  │
│  │  Repo    │   │  Markdown│   │  Docs    │   │  Chunks  │   │  Qdrant  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                               QUERY PIPELINE                                  │
│                                                                               │
│   Query → Cache → Embed → ┌─ Documents (Qdrant) ─┐ → Rerank → Generate      │
│             ↓       ↓      │                       │     ↓          ↓         │
│          Redis   BGE-base └─ Videos (Academy MCP) ┘  Jina AI    Claude       │
│                   (768-dim)    (Parallel Fetch)     API        Sonnet 4.5     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```python
# Simplified architecture flow
class RAGPipeline:
    async def query(self, query: str) -> AsyncGenerator[dict, None]:
        # L1: Cache
        if cached := await self.cache.get_semantic(query):
            yield cached; return

        # L2: Embed
        query_vector = self.embedder.embed(query)  # BGE → 768-dim

        # L3: Parallel Retrieval ⚡
        docs_task = self.retriever.search(query_vector)
        videos_task = self.academy.search_videos(query)

        docs, videos = await asyncio.gather(
            docs_task,
            videos_task,
            return_exceptions=True  # Isolated failures
        )

        # L4: Rerank (documents only)
        docs = await self.reranker.rerank(query, docs)

        # L5: Generate (with docs + videos)
        async for chunk in self.generator.stream(query, docs, videos):
            yield chunk
```

---

## Document Processing Pipeline

### Chunking Strategy

**Challenge:** Documents vary from 100 to 10,000+ tokens. LLMs have context limits. How do we split effectively?

**Our approach: Semantic-aware hierarchical chunking**

```python
# Configuration
CHUNK_MIN_TOKENS = 100   # Avoid tiny, context-less chunks
CHUNK_MAX_TOKENS = 512   # Stay within model context windows
CHUNK_TARGET_TOKENS = 350  # Optimal for retrieval quality

# Algorithm
def chunk_document(content: str, metadata: dict) -> list[Chunk]:
    chunks = []

    # 1. Split by headers (preserve semantic units)
    sections = split_by_headers(content)  # H1 > H2 > H3

    # 2. For each section, apply token-aware splitting
    for section in sections:
        if token_count(section) <= CHUNK_MAX_TOKENS:
            chunks.append(section)
        else:
            # Split on paragraph boundaries, respecting min/max
            chunks.extend(split_paragraphs(section))

    # 3. Sliding window overlap for context continuity
    chunks = add_overlap(chunks, overlap_tokens=50)

    return chunks
```

**Why these parameters?**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_tokens` | 100 | Smaller chunks lack context, hurt retrieval quality |
| `max_tokens` | 512 | BERT-based models (like BGE) trained on 512 tokens max |
| `target_tokens` | 350 | Sweet spot: enough context, room for overlap |
| `overlap` | 50 | Prevents losing context at chunk boundaries |

### Metadata Extraction

Each chunk carries rich metadata for filtering and citation:

```python
@dataclass
class ChunkMetadata:
    # Document identification
    doc_id: str           # Unique document hash
    file_path: str        # learn/app-development/widgets/button.md

    # Content hierarchy
    title: str            # "Button Widget"
    section: Optional[str]  # "Properties > Styling"
    parent_sections: list[str]  # ["Button Widget", "Properties"]

    # URL construction
    url_slug: str         # button-widget
    url_hash: Optional[str]  # #styling

    # Chunk position
    chunk_index: int      # 0, 1, 2, ... for ordering
    total_chunks: int     # Total chunks in document
```

---

## Embedding Strategy

### Model Selection: BAAI/bge-base-en-v1.5

**Why this model?**

| Criterion | BGE-base | OpenAI ada-002 | Cohere embed-v3 |
|-----------|----------|----------------|-----------------|
| **Quality (MTEB)** | 63.55 | 61.0 | 64.5 |
| **Dimensions** | 768 | 1536 | 1024 |
| **Cost** | Free (local) | $0.0001/1K tokens | $0.0001/1K tokens |
| **Latency** | ~50ms (local) | ~200ms (API) | ~150ms (API) |
| **Privacy** | Full control | Data sent to OpenAI | Data sent to Cohere |

**Decision:** BGE-base offers best quality/cost tradeoff for our use case (technical English docs).

### Embedding Configuration

```python
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device="cpu",  # MPS/CUDA for GPU acceleration
        )
        self.instruction = "Represent this sentence for searching relevant passages:"

    def embed_query(self, query: str) -> list[float]:
        # BGE requires instruction prefix for queries (not documents)
        prefixed = f"{self.instruction} {query}"
        return self.model.encode(prefixed).tolist()

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        # Documents don't need instruction prefix
        return self.model.encode(docs).tolist()
```

### Dimensionality Trade-offs

```
768 dimensions (BGE-base):
├── Memory: 768 * 4 bytes = 3KB per vector
├── 10,000 docs = ~30MB vector storage
├── HNSW index overhead: ~3x = ~90MB total
└── Search latency: O(log n) with HNSW

1536 dimensions (OpenAI):
├── Memory: 1536 * 4 bytes = 6KB per vector
├── 10,000 docs = ~60MB vector storage
├── More nuanced semantic space
└── Diminishing returns for our corpus size
```

---

## Vector Database Design

### Qdrant Configuration

```python
# Collection schema
collection_config = {
    "vectors": {
        "dense": {
            "size": 768,
            "distance": "Cosine",  # Normalized similarity
        },
        "sparse": {
            "modifier": "idf",  # BM25-style sparse vectors
        }
    },
    "optimizers_config": {
        "default_segment_number": 2,  # For small collections
        "indexing_threshold": 10000,  # Start indexing at 10k docs
    },
    "hnsw_config": {
        "m": 16,              # Connections per node
        "ef_construct": 100,  # Build-time accuracy
        "full_scan_threshold": 1000,  # Below this, brute force
    }
}
```

### HNSW Index Parameters

**m (connections per layer):**
```
m=4:  Faster insert, less memory, lower recall
m=16: Good balance (our choice)
m=64: Higher recall, slower builds, more memory
```

**ef_construct (build-time beam width):**
```
ef_construct=50:  Faster indexing, lower quality
ef_construct=100: Good quality (our choice)
ef_construct=500: High quality, slow builds
```

**ef (search-time beam width):**
```python
# Higher ef = slower but more accurate search
search_params = {
    "exact": False,
    "hnsw_ef": 128,  # Search-time quality
}
```

### Payload Design

```python
# What we store alongside vectors
payload = {
    # For filtering
    "source": "docs",  # docs | academy (future)
    "category": "widgets",

    # For reconstruction
    "content": "The Button widget allows...",
    "title": "Button Widget",
    "url": "https://docs.wavemaker.com/...",
    "section": "Properties",

    # For ranking
    "chunk_index": 0,
    "doc_importance": 0.8,  # Based on page views (future)
}
```

---

## Retrieval Strategy

### Hybrid Search (Dense + Sparse)

**Why hybrid?**

| Query Type | Dense (Semantic) | Sparse (Lexical) | Best For |
|------------|-----------------|------------------|----------|
| "How to create API" | ✅ Understands intent | ❌ Misses synonyms | Dense |
| "AIRA" (abbreviation) | ❌ No semantic meaning | ✅ Exact match | Sparse |
| "Configure DB_HOST" | ❌ Looks like noise | ✅ Token match | Sparse |

**Implementation:**

```python
async def hybrid_search(
    query: str,
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    top_k: int = 30,
) -> list[Document]:
    # Parallel search
    dense_results = await qdrant.search(
        vector=dense_vector,
        limit=top_k,
        with_payload=True,
    )

    sparse_results = await qdrant.search(
        vector=SparseVector(sparse_vector),
        limit=top_k,
        with_payload=True,
    )

    # Reciprocal Rank Fusion
    return reciprocal_rank_fusion(dense_results, sparse_results, k=60)
```

### Reciprocal Rank Fusion (RRF)

**Formula:**
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where:
- `k` = 60 (constant, prevents high ranks from dominating)
- `rank_i(d)` = rank of document `d` in result list `i`

**Example:**
```python
def reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    scores = defaultdict(float)

    for results in results_lists:
        for rank, doc in enumerate(results, 1):
            scores[doc.id] += 1.0 / (k + rank)

    # Sort by RRF score
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [get_doc(doc_id) for doc_id, score in fused]

# Example calculation:
# Doc A: Rank 1 in dense, Rank 5 in sparse
# RRF(A) = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

# Doc B: Rank 3 in dense, Rank 2 in sparse
# RRF(B) = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

# Result: Doc B > Doc A (despite A ranking higher in dense)
```

---

## Academy MCP Integration

### Model Context Protocol (MCP) Overview

**What is MCP?**
- Standardized protocol for AI systems to connect to external tools
- JSON-RPC over HTTP with stateful sessions
- Think "USB-C for AI" - universal connector for data sources

**Why MCP over REST?**

| Feature | REST API | MCP |
|---------|----------|-----|
| **Protocol** | Custom per service | Standardized JSON-RPC |
| **Discovery** | Hardcoded endpoints | Dynamic tool discovery (`list_tools()`) |
| **Session** | Stateless | Stateful with persistent connections |
| **Reconnection** | Manual implementation | Built-in auto-reconnect |
| **Timeouts** | Custom handling | Protocol-level timeout management |

### Streamable HTTP Transport

**Architecture:**

```
Client (Docs Agent)                Server (Academy)
       │                                   │
       │  POST /mcp (session init)         │
       ├──────────────────────────────────>│
       │<──────────────────────────────────┤
       │  200 OK {session_id, protocol}    │
       │                                   │
       │  GET /mcp (long-lived connection) │
       ├──────────────────────────────────>│
       │<──── SSE stream (server→client) ──┤
       │                                   │
       │  POST /mcp (tool calls)           │
       ├──────────────────────────────────>│
       │<──────────────────────────────────┤
       │  202 Accepted                     │
       │                                   │
       │<──── response via SSE stream ─────┤
```

### Academy Client Implementation

```python
import asyncio
import json
import time
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Connection configuration
MCP_CONNECT_TIMEOUT = 10.0  # seconds - initial connection
MCP_SESSION_MAX_AGE = 300    # seconds - 5 minutes, auto-refresh
MCP_CALL_TIMEOUT = 15.0      # seconds - tool call timeout

class AcademyClient:
    """
    MCP client for WaveMaker Academy video search.

    Features:
    - Persistent session management
    - Auto-reconnection on timeout
    - Session age tracking (recreate after 5min)
    - Configurable timeouts
    - Graceful error handling
    """

    def __init__(self):
        self.settings = get_settings()
        self._session: Optional[ClientSession] = None
        self._context = None
        self._session_created_at: Optional[float] = None

    def _is_session_stale(self) -> bool:
        """Check if session needs recreation."""
        if not self._session or not self._session_created_at:
            return True

        age = time.time() - self._session_created_at
        if age > MCP_SESSION_MAX_AGE:
            logger.debug(f"Session stale: {age:.1f}s > {MCP_SESSION_MAX_AGE}s")
            return True

        return False

    async def _ensure_session(self) -> ClientSession:
        """Get or create MCP session with timeout."""
        if self._is_session_stale():
            await self._close_session()

        if not self._session:
            # Create streamable HTTP client with timeout
            self._context = streamablehttp_client(self.settings.academy_mcp_url)

            try:
                self._read, self._write, _ = await asyncio.wait_for(
                    self._context.__aenter__(),
                    timeout=MCP_CONNECT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"MCP connect timeout: {MCP_CONNECT_TIMEOUT}s")

            # Initialize MCP session
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
            await self._session.initialize()

            self._session_created_at = time.time()
            logger.info(f"MCP session initialized (max_age={MCP_SESSION_MAX_AGE}s)")

        return self._session

    async def search_videos(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search Academy videos via MCP tool.

        Retry logic: 2 attempts with fresh session on timeout/connection errors.
        """
        max_retries = 2

        for attempt in range(max_retries):
            try:
                session = await self._ensure_session()

                # Call MCP tool with timeout
                result = await asyncio.wait_for(
                    session.call_tool(
                        name="wm-academy-semantic-search",
                        arguments={"query": query, "limit": limit},
                    ),
                    timeout=MCP_CALL_TIMEOUT,
                )

                # Parse response
                data = json.loads(result.content[0].text)
                videos = data.get("body", [])

                logger.info(f"Found {len(videos)} videos: {query[:50]}")
                return videos

            except (TimeoutError, ConnectionError, asyncio.TimeoutError) as e:
                logger.warning(f"MCP error (attempt {attempt+1}): {e}")
                await self._close_session()  # Force recreation

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # Brief backoff
                else:
                    logger.error(f"MCP failed after {max_retries} attempts")
                    return []

            except Exception as e:
                logger.error(f"MCP unexpected error: {e}", exc_info=True)
                return []

        return []
```

### MCP Tool Schema

**Tool:** `wm-academy-semantic-search`

**Input:**
```json
{
  "query": "REST API",
  "limit": 3
}
```

**Output:**
```json
{
  "headers": {},
  "body": [
    {
      "id": 66,
      "title": "Build REST API",
      "description": "Learn to create REST APIs in WaveMaker",
      "moduleName": "Deep Dive",
      "code": "CHAP_66",
      "link": "https://dev-next-academy.wavemaker.com/Watch?wm=CHAP_66"
    }
  ],
  "statusCodeValue": 200,
  "statusCode": "OK"
}
```

### Connection Lifecycle & Resilience

**Session Management:**

```
t=0s:    Query 1 → Create session → Call tool → Success
t=30s:   Query 2 → Reuse session → Call tool → Success
t=300s:  Query 3 → Session stale → Recreate → Success
t=301s:  Query 4 → Reuse new session → Success
```

**Timeout Handling:**

```
Query arrives
    ↓
Ensure session (10s timeout)
    ├─ Success → Continue
    └─ Timeout → Raise error → Retry with new session
    ↓
Call tool (15s timeout)
    ├─ Success → Parse response
    ├─ Timeout → Close session → Retry (attempt 2)
    └─ After 2 failures → Return empty list (graceful degradation)
```

**Gateway Timeout (504) Handling:**

The MCP SDK maintains a long-lived GET stream for server→client communication. If idle >60s, gateways may close this connection with 504. The SDK auto-reconnects:

```
10:43:10 - Session created ✅
10:44:11 - Gateway timeout 504 (idle >60s)
10:44:11 - SDK: "GET stream disconnected, reconnecting in 1000ms..."
10:44:12 - SDK reconnects automatically
```

**Impact:** Background noise only. User queries trigger new session if needed via `_is_session_stale()`.

---

## Parallel Execution Architecture

### Why Parallel Retrieval?

**Problem:** Sequential execution compounds latency

```
Sequential:
  Document search: 2.5s
        ↓
  Video search: 1.5s
        ↓
  Total: 4.0s

Parallel:
  Document search: 2.5s  ←┐
                           ├→ Total: 2.5s (max of two)
  Video search: 1.5s      ←┘
```

**Additional Benefits:**
- Failures in one don't block the other
- Better user experience (faster responses)
- Cost-effective (don't pay for serial waiting time)

### Implementation

```python
async def query(self, query: str) -> dict:
    # Layer 2: Generate embeddings
    dense_vector = await self.embedder.embed_query(query)
    sparse_vector = self.embedder.generate_sparse_vector(query)

    # Layer 3: Parallel retrieval ⚡
    documents_task = self.retriever.search(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
    )
    videos_task = self.academy.search_videos(query, limit=3)

    # Execute concurrently with isolated error handling
    results = await asyncio.gather(
        documents_task,
        videos_task,
        return_exceptions=True,  # Don't let one failure crash both
    )

    # Handle results
    documents = results[0] if isinstance(results[0], list) else []
    videos = results[1] if isinstance(results[1], list) else []

    # Log failures without blocking
    if not isinstance(results[0], list):
        logger.error(f"Document retrieval failed: {type(results[0]).__name__}")
    if not isinstance(results[1], list):
        logger.warning(f"Video search failed: {type(results[1]).__name__}")

    # Continue with whatever succeeded
    return documents, videos
```

### Error Isolation

**Key insight:** `return_exceptions=True` in `asyncio.gather()`

```python
# Without return_exceptions (default):
results = await asyncio.gather(task1, task2)
# If task2 raises → entire gather() raises → task1 result lost

# With return_exceptions=True:
results = await asyncio.gather(task1, task2, return_exceptions=True)
# If task2 raises → results[1] = Exception object
#                 → results[0] = task1 result (preserved)
```

**Handling mixed results:**

```python
# Type-safe result extraction
documents = results[0] if isinstance(results[0], list) else []
videos = results[1] if isinstance(results[1], list) else []

# Catches all exception types (including BaseException subclasses like CancelledError)
if not isinstance(results[1], list):
    logger.warning(f"Video failure ({type(results[1]).__name__}): {results[1]}")
    # Continue with documents only - graceful degradation
```

---

## Reranking Layer

### Cross-Encoder vs Bi-Encoder

```
 BI-ENCODER (Embedding Search):
┌─────────┐     ┌─────────┐
│  Query  │     │   Doc   │
└────┬────┘     └────┬────┘
     │               │
     ▼               ▼
┌─────────┐     ┌─────────┐
│ Encoder │     │ Encoder │   ← Same model, separate encoding
└────┬────┘     └────┬────┘
     │               │
     ▼               ▼
  [0.2, 0.8]    [0.3, 0.7]   ← Compare with cosine similarity
     │               │
     └───────────────┘
           ↓
     similarity = 0.95

Pros: Fast (encode once, compare many)
Cons: Approximate (no cross-attention)


CROSS-ENCODER (Reranking):
┌─────────────────────────┐
│   [CLS] Query [SEP] Doc │   ← Concatenated input
└───────────┬─────────────┘
            │
            ▼
      ┌───────────┐
      │ Encoder   │   ← Full attention between Q and D
      └─────┬─────┘
            │
            ▼
    relevance_score = 0.82

Pros: Precise (full cross-attention)
Cons: Slow (O(n) for n docs)
```

### Jina Reranker API

**Why Jina over local cross-encoder?**

| Factor | Local (ms-marco-MiniLM) | Jina API |
|--------|------------------------|----------|
| **Apple Silicon** | MLX causes NaN scores | ✅ Works |
| **Quality** | Good | Excellent (multilingual) |
| **Latency** | ~500ms (30 docs) | ~500ms (30 docs) |
| **Cost** | Free | ~$0.0001/call |

**Implementation:**

```python
async def rerank_jina(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[tuple[int, float]]:
    response = await httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": [d[:2000] for d in documents],  # Truncate
        },
    )

    return [
        (r["index"], r["relevance_score"])
        for r in response.json()["results"]
    ]
```

### Selective Reranking

**Optimization:** Skip reranking if RRF confidence is already high

```python
def should_rerank(documents: list[Document]) -> bool:
    """Skip reranking if top result is very confident."""
    if not documents:
        return False

    top_rrf_score = documents[0].rrf_score

    # High RRF score = dense and sparse agree strongly
    if top_rrf_score > 0.04:  # ~top 25 in both lists
        logger.debug(f"Skip rerank: confident (RRF={top_rrf_score:.4f})")
        return False

    return True
```

**Why 0.04 threshold?**

```
RRF = 1/(60+1) + 1/(60+1) = 0.0328  (rank 1 in both)
RRF = 1/(60+1) + 1/(60+25) = 0.0282 (rank 1, rank 25)
RRF = 1/(60+25) + 1/(60+25) = 0.0235 (rank 25 in both)

Threshold 0.04: Rerank when methods disagree significantly
```

**Reranking Decision Logic:**

```python
# Only documents are reranked, not videos
if documents and self.reranker.should_rerank(documents):
    documents = await self.reranker.rerank(query, documents)
elif documents:
    # High confidence, just take top_k
    documents = documents[:self.settings.rerank_top_k]

# Videos skip reranking (already ranked by Academy MCP)
```

---

## Caching Architecture

### Three-Tier Cache Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CACHE LAYER                                      │
│                                                                               │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │  TIER 1: EXACT    │  │  TIER 2: SEMANTIC │  │  TIER 3: EMBEDDING│        │
│  │                   │  │                   │  │                   │        │
│  │  hash(query)→resp │  │  vector:resp pairs│  │  text→embedding   │        │
│  │                   │  │  sim(q,cached)>θ  │  │                   │        │
│  │  O(1) lookup      │  │  Linear scan      │  │  Skip re-compute  │        │
│  │  100% match only  │  │  95%+ similarity  │  │  Reuse embeddings │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
│                                                                               │
│  Key: exact:{sha256}     Key: semantic:{sha256}   Key: embed:{sha256}        │
│  TTL: 1 hour             TTL: 1 hour              TTL: 24 hours              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Exact Match Cache

```python
async def get_exact(self, query: str) -> Optional[dict]:
    """O(1) lookup using query hash."""
    normalized = " ".join(query.lower().strip().split())
    key = f"exact:{sha256(normalized)[:32]}"
    return await redis.get(key)
```

### Tier 2: Semantic Cache

```python
async def get_semantic(
    self,
    query: str,
    query_embedding: np.ndarray,
) -> Optional[dict]:
    """Find semantically similar cached query."""
    threshold = 0.95

    # Scan cached embeddings (limited to 100 for performance)
    for cached in await redis.keys("semantic:*")[:100]:
        cached_embedding = cached["embedding"]
        similarity = cosine_similarity(query_embedding, cached_embedding)

        if similarity >= threshold:
            return cached["response"]

    return None
```

### Tier 3: Embedding Cache

```python
async def get_embedding(self, text: str) -> Optional[np.ndarray]:
    """Reuse previously computed embeddings."""
    key = f"embed:{sha256(text)[:32]}"
    cached = await redis.get(key)
    if cached:
        return np.array(json.loads(cached))
    return None

async def set_embedding(self, text: str, embedding: np.ndarray) -> None:
    """Cache embedding with 24x longer TTL (embeddings rarely change)."""
    await redis.setex(
        key=f"embed:{sha256(text)[:32]}",
        ttl=self.ttl * 24,  # 24 hours instead of 1 hour
        value=embedding.tolist(),
    )
```

### Cache Priority Flow

```
Query: "How to create REST API?"
         │
         ▼
┌─────────────────────────┐
│  Tier 1: Exact Match?   │ ← Hash lookup, O(1)
│  exact:{hash} exists?   │
└───────────┬─────────────┘
            │ Miss
            ▼
┌─────────────────────────┐
│  Tier 2: Semantic?      │ ← Vector similarity scan
│  Any cache sim > 0.95?  │
└───────────┬─────────────┘
            │ Miss
            ▼
┌─────────────────────────┐
│  Tier 3: Embedding?     │ ← Avoid re-computing embedding
│  embed:{hash} exists?   │
└───────────┬─────────────┘
            │ Miss/Hit
            ▼
      Continue to retrieval
      (with cached or new embedding)
```

---

## LLM Generation

### System Prompt Design

```python
SYSTEM_PROMPT = """You are the **WaveMaker Documentation Assistant**, an expert on the WaveMaker low-code development platform.

## 🎯 Your Mission

Provide clear, accurate, and actionable answers based ONLY on the provided documentation context.

## 📋 Response Structure

ALWAYS structure your responses in this order:

1. **Direct Answer** (1-2 sentences) - Immediately answer the question
2. **Details** - Expand with relevant information, organized logically
3. **Steps** (if applicable) - Number each step clearly
4. **Code Examples** (if in context) - Use proper code blocks with language tags
5. **Related Information** (optional) - Brief mention of related features

## ✅ Citation Rules (CRITICAL)

- ALWAYS cite sources using numbered references: [1], [2], etc.
- Place citations INLINE, immediately after the relevant information
- Each number corresponds to a document in the context
- Multiple sources supporting the same point: [1][2]
- NEVER make up information not in the provided context

## 📺 Video Recommendations

When videos are provided in the context:
- Include them as clickable markdown links: `[Video Title](URL)`
- Format: "For visual guides, see: [Video Title](URL)"
- Place at the end of your response in a "**Related Videos**" section
- Only recommend videos that clearly match the user's question

## ⚠️ When You Don't Know

If the context doesn't contain the answer:
- State clearly: "I don't have specific information about this in the documentation."
- Suggest related topics if applicable
- DO NOT guess or hallucinate
"""
```

### Context Formatting with Videos

```python
def _format_context(
    documents: list[Document],
    videos: Optional[list[dict]] = None,
) -> str:
    """Format documents and videos for LLM consumption."""
    parts = ["## 📚 Documentation Context\n"]

    for i, doc in enumerate(documents, 1):
        parts.append(f"""
### [{i}] {doc.title}
**Source:** {doc.url}
{f"**Section:** {doc.section}" if doc.section else ""}

{doc.content}

---
""")

    if videos:
        parts.append("\n## 🎬 Related Videos (supplementary)\n")
        for video in videos:
            title = video.get('title', 'Video')
            link = video.get('link', '')
            # Format as markdown link for LLM to include in response
            parts.append(f"📺 [{title}]({link})")
        parts.append("")
        parts.append("*(Note: Include these video links in your response as clickable markdown links.)*")

    return "\n".join(parts)
```

### Streaming Implementation

```python
async def generate_stream(
    query: str,
    documents: list[Document],
    videos: Optional[list[dict]] = None,
) -> AsyncGenerator[dict, None]:
    client = anthropic.AsyncAnthropic()

    user_prompt = self._build_user_prompt(query, documents, videos)

    async with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=0.2,  # Low for factual answers
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "text", "content": text}

    # After streaming, yield structured data
    yield {
        "type": "sources",
        "sources": [doc.to_source() for doc in documents]
    }

    if videos:
        yield {
            "type": "videos",
            "videos": [{"title": v["title"], "url": v["link"]} for v in videos]
        }

    yield {"type": "done", "cached": False}
```

### Temperature Selection

```
temperature = 0.0: Deterministic, may miss nuance
temperature = 0.2: Slight variation, mostly factual (our choice)
temperature = 0.7: Creative, may hallucinate
temperature = 1.0: Very creative, unreliable for docs
```

---

## Performance Optimizations

### Lazy Model Loading

```python
class Embedder:
    """Lazy-load model on first use, not app startup."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info("Loading embedding model...")
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model
```

### Connection Pooling

```python
# Reuse HTTP connections
class RerankerClient:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
        return self._client
```

### Batch Processing (Indexing)

```python
async def index_documents(chunks: list[Chunk], batch_size: int = 100):
    """Batch embed and upsert for efficiency."""

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Batch embed (much faster than one-by-one)
        vectors = embedder.encode([c.content for c in batch])

        # Batch upsert
        points = [
            PointStruct(id=c.id, vector=v, payload=c.metadata)
            for c, v in zip(batch, vectors)
        ]
        await qdrant.upsert(collection, points)
```

---

## Error Handling & Resilience

### Graceful Degradation Strategy

**Core Principle:** System should remain useful even with partial failures

```
Component Failure → Impact → Mitigation
─────────────────────────────────────────
Cache down       → Slower    → Direct retrieval
Embedder fails   → Critical  → Error 500 (can't search)
Qdrant down      → Critical  → Error 503 (primary data)
Academy MCP down → Degraded  → Return docs only (no videos)
Reranker fails   → Degraded  → Use RRF scores, skip reranking
LLM fails        → Critical  → Error 503 (can't generate)
```

### MCP-Specific Error Handling

```python
# Timeout errors → Retry with fresh session
except (TimeoutError, ConnectionError, asyncio.TimeoutError) as e:
    logger.warning(f"MCP connection error: {e}")
    await self._close_session()  # Force recreation
    if attempt < max_retries - 1:
        await asyncio.sleep(0.5)
    else:
        return []  # Graceful degradation

# Parse errors → Log and return empty (corrupted response)
except json.JSONDecodeError as e:
    logger.error(f"Invalid MCP response: {e}", exc_info=True)
    return []

# All other errors → Log and return empty
except Exception as e:
    logger.error(f"Unexpected MCP error: {e}", exc_info=True)
    return []
```

### Parallel Execution Error Isolation

```python
results = await asyncio.gather(
    documents_task,
    videos_task,
    return_exceptions=True,  # Key: isolate failures
)

# Type-safe extraction (handles any exception type)
documents = results[0] if isinstance(results[0], list) else []
videos = results[1] if isinstance(results[1], list) else []

# User gets documents even if videos fail
```

### Retry Configuration

```python
# MCP Client
MCP_CONNECT_TIMEOUT = 10.0   # Tunable: increase if network slow
MCP_SESSION_MAX_AGE = 300     # Tunable: decrease for aggressive refresh
MCP_CALL_TIMEOUT = 15.0       # Tunable: increase if searches slow
max_retries = 2               # Tunable: balance reliability vs latency

# Backoff strategy
await asyncio.sleep(0.5)      # Brief delay between retries
```

---

## Evaluation & Metrics

### Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Recall@K** | % of relevant docs in top K results | > 90% at K=30 |
| **MRR** | Mean Reciprocal Rank of first relevant | > 0.7 |
| **NDCG@5** | Normalized DCG for ranked results | > 0.8 |
| **Latency P95** | 95th percentile response time | < 5s |
| **Cache Hit Rate** | % queries served from cache | > 40% |
| **Video Integration Rate** | % responses with videos | > 60% |

### Evaluation Dataset

```python
# eval/test_queries.json
{
    "queries": [
        {
            "query": "How to create a new page in WaveMaker?",
            "relevant_docs": ["pages-overview.md", "create-page.md"],
            "ideal_answer_keywords": ["Pages", "New", "Markup"],
            "expected_videos": true  # Should include video results
        }
    ]
}
```

### Monitoring Dashboard

```python
# Key observability metrics
class MetricsCollector:
    def collect(self):
        return {
            "latency": {
                "cache_check": histogram(cache_timings),
                "embedding": histogram(embed_timings),
                "retrieval": histogram(retrieval_timings),
                "mcp_video": histogram(mcp_timings),
                "reranking": histogram(rerank_timings),
                "generation": histogram(generation_timings),
            },
            "cache": {
                "hit_rate_exact": exact_hits / total_queries,
                "hit_rate_semantic": semantic_hits / total_queries,
            },
            "mcp": {
                "success_rate": mcp_successes / mcp_attempts,
                "avg_videos_returned": mean(video_counts),
                "timeout_rate": mcp_timeouts / mcp_attempts,
            },
            "quality": {
                "avg_sources_cited": mean(source_counts),
                "avg_response_length": mean(response_lengths),
            }
        }
```

---

## Configuration Parameters

### Complete Configuration Matrix

```python
# Embedding
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# Chunking
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 512
CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

# Retrieval
RETRIEVAL_TOP_K = 30        # Documents to retrieve
DENSE_WEIGHT = 0.5          # RRF weight for dense search
SPARSE_WEIGHT = 0.5         # RRF weight for sparse search
RRF_K = 60                  # RRF constant

# Reranking
RERANK_TOP_K = 5            # Final documents for LLM
RERANK_PROVIDER = "jina"    # jina | local
RERANK_SKIP_THRESHOLD = 0.04  # Skip if RRF > this

# MCP
MCP_CONNECT_TIMEOUT = 10.0  # seconds
MCP_SESSION_MAX_AGE = 300   # seconds
MCP_CALL_TIMEOUT = 15.0     # seconds
MCP_MAX_RETRIES = 2
VIDEO_LIMIT = 3             # Videos per query

# Caching
CACHE_TTL_HOURS = 1
SEMANTIC_CACHE_THRESHOLD = 0.95
EMBEDDING_CACHE_MULTIPLIER = 24

# LLM
LLM_PROVIDER = "anthropic"  # anthropic | openai | ollama
LLM_MODEL = "claude-sonnet-4-5-20250929"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2048
```

---

## References

- [BGE Embeddings Paper](https://arxiv.org/abs/2309.07597)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Model Context Protocol Spec](https://spec.modelcontextprotocol.io/specification/2025-06-18/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [FastMCP Python SDK](https://github.com/jlowin/fastmcp)

---

*Last Updated: 2026-02-02*
*Architecture Version: 2.0 (with MCP integration)*
