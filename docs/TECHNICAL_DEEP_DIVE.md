# WaveMaker Docs Agent - Technical Deep Dive

A comprehensive technical explanation of the RAG pipeline architecture, clarifying implementation details and design decisions.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Layer 1: Three-Tier Caching](#layer-1-three-tier-caching)
3. [Layer 2: Hybrid Embeddings](#layer-2-hybrid-embeddings)
4. [Layer 3: Retrieval & Fusion](#layer-3-retrieval--fusion)
5. [Layer 4: Reranking](#layer-4-reranking)
6. [Layer 5: Generation](#layer-5-generation)
7. [Performance Characteristics](#performance-characteristics)

---

## System Overview

### Pipeline Flow

```
Query → Cache (3 tiers) → Embeddings (dense + sparse)
     → Retrieval (docs ∥ videos) → Rerank (docs only)
     → Generate → Cache → Response
```

### Key Design Principles

1. **Performance**: Multi-tier caching, parallel execution
2. **Accuracy**: Hybrid search (semantic + keyword), cross-encoder reranking
3. **Reliability**: Graceful degradation, auto-reconnection
4. **Cost-efficiency**: Local models where possible, smart cache reuse

---

## Layer 1: Three-Tier Caching

### Architecture

```
┌─────────────┐   Miss   ┌──────────────┐   Miss   ┌──────────────┐
│  Tier 1:    │─────────→│  Tier 2:     │─────────→│  Tier 3:     │
│  Exact      │          │  Semantic    │          │  Embedding   │
│  Match      │          │  Similarity  │          │  Reuse       │
└─────────────┘          └──────────────┘          └──────────────┘
     ↓ Hit                    ↓ Hit                     ↓ Hit
  Response              Response                  Embeddings
```

### Tier 1: Exact Match Cache

**Purpose:** Instant lookup for identical queries

**Storage:**
```python
key: "exact:{sha256(normalized_query)[:32]}"
value: {
    "answer": "...",
    "sources": [...],
    "videos": [...],
    "query_id": "..."
}
TTL: cache_ttl_hours * 3600  # e.g., 24 hours
```

**Normalization:**
```python
# Before: "What   IS  AIRA?"
normalized = " ".join(query.lower().strip().split())
# After: "what is aira?"
hash = sha256(normalized.encode()).hexdigest()[:32]
```

**Lookup time:** O(1), ~5ms

### Tier 2: Semantic Cache

**Purpose:** Match similar queries with different wording

**Storage:**
```python
key: "semantic:{sha256(normalized_query)[:32]}"
value: {
    "query": "What is AIRA?",
    "embedding": [0.23, -0.45, ...],  # 768-dim vector
    "response": {
        "answer": "...",
        "sources": [...],
        "videos": [...]
    }
}
TTL: cache_ttl_hours * 3600
```

**Matching algorithm:**
```python
# 1. Get all semantic cache entries (limit 100 for speed)
# 2. For each cached entry:
#    - Load cached_embedding
#    - Compute cosine_similarity(query_embedding, cached_embedding)
#    - If similarity >= 0.95 (configurable threshold):
#        Return cached response

# Cosine similarity formula:
similarity = dot(a, b) / (norm(a) * norm(b))
```

**Lookup time:** O(n) where n ≤ 100, ~20-50ms

**Example:**
```
Query 1: "What is AIRA?"          → embedding: [0.23, -0.45, 0.89, ...]
Query 2: "Tell me about AIRA"     → embedding: [0.24, -0.44, 0.88, ...]
Cosine similarity: 0.96 → Cache hit! Return Query 1's answer
```

### Tier 3: Embedding Cache

**Purpose:** Reuse computed embeddings (optimization only)

**Storage:**
```python
key: "embed:{sha256(normalized_text)[:32]}"
value: [0.23, -0.45, 0.89, ...]  # 768-dim vector
TTL: cache_ttl_hours * 3600 * 24  # 24x longer (e.g., 24 days)
```

**Why longer TTL?**
- Embeddings for same text don't change
- Model stays the same
- Much cheaper to cache longer

**Savings:**
- Embedding computation: ~50-100ms per query
- Tier 3 hit: ~5ms lookup

### Cache Flow Decision Tree

```
User query arrives
    │
    ▼
Tier 1: Check exact match
    ├─ Hit → Return cached response (5ms)
    └─ Miss
        │
        ▼
    Tier 3: Check embedding cache
        ├─ Hit → Use cached embedding
        └─ Miss → Compute embedding (100ms)
        │
        ▼
    Tier 2: Check semantic similarity
        ├─ Hit (≥0.95) → Return cached response (50ms)
        └─ Miss → Continue to retrieval
```

### Cache Storage Details

**What each tier stores:**

| Tier | Key | Value Type | Purpose |
|------|-----|-----------|---------|
| 1 | Hash of query | Complete response (JSON) | Instant exact match |
| 2 | Hash of query | Query + embedding + response | Similar meaning match |
| 3 | Hash of text | Embedding only (vector) | Avoid re-computing embeddings |

**Key insight:** Tier 3 is NOT a cache layer in the retrieval flow - it's an **optimization cache** that speeds up embedding computation.

---

## Layer 2: Hybrid Embeddings

### Why Two Types of Vectors?

**Dense vectors** (semantic) catch conceptual similarity.
**Sparse vectors** (keyword) catch exact term matches.

```
Query: "How to create prefab in WaveMaker?"

Dense (BGE model):
  ├─ Captures: creation, components, reusable elements
  └─ Finds: docs about widgets, partial pages, components

Sparse (BM25-style):
  ├─ Captures: exact terms "prefab", "wavemaker"
  └─ Finds: docs with those specific keywords

Hybrid:
  └─ Combines both → finds docs that are semantically relevant
                     AND contain the specific technical terms
```

### Dense Vector Generation

**Model:** BAAI/bge-base-en-v1.5

**Why BGE?**
1. **Trained for retrieval** - Optimized specifically for search tasks
2. **Asymmetric design** - Short query → long document matching
3. **MTEB benchmark** - Top performer on retrieval benchmarks
4. **768 dimensions** - Sweet spot (quality vs. speed)
5. **Free & local** - No API costs, runs on CPU

**Process:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
dense_vector = model.encode(query, normalize_embeddings=True)
# Returns: numpy array of shape (768,), all values filled
```

**Characteristics:**
- All 768 dimensions have non-zero values
- L2 normalized (length = 1)
- Similar meanings → close vectors (cosine similarity)

**Example:**
```
"REST API"           → [0.23, -0.45, 0.89, 0.12, ...]
"web service"        → [0.21, -0.43, 0.91, 0.14, ...]  # Similar!
"database schema"    → [0.67, 0.12, -0.34, 0.89, ...]  # Different
```

### Sparse Vector Generation

**Algorithm:** Custom BM25-style TF weighting

**Why NOT use a package?**
- rank-bm25 requires document corpus (we don't have it at query time)
- Simple TF weighting is sufficient for query representation
- Custom implementation = full control

**Process:**

```python
# Step 1: Tokenization
text = "How to create prefab in WaveMaker?"
tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
# Result: ["how", "to", "create", "prefab", "in", "wavemaker"]

# Step 2: Remove stopwords
stopwords = {"how", "to", "in", ...}
filtered = [t for t in tokens if t not in stopwords]
# Result: ["create", "prefab", "wavemaker"]

# Step 3: Calculate term frequencies
term_counts = Counter(filtered)
total_terms = len(filtered)
# {"create": 1, "prefab": 1, "wavemaker": 1}, total: 3

# Step 4: TF weighting with saturation
sparse_vector = {}
for term, count in term_counts.items():
    tf = count / total_terms        # create: 1/3 = 0.333
    weight = min(1.0, tf * 5)       # 0.333 * 5 = 1.0 (capped)

    if weight > 0.1:  # Filter low weights
        # Hash term to index (deterministic)
        token_hash = hashlib.md5(term.encode()).hexdigest()
        index = int(token_hash[:8], 16) % 100000
        sparse_vector[index] = round(weight, 3)

# Result: {42351: 1.0, 78234: 1.0, 12845: 1.0}
```

**Characteristics:**
- Most indices are zero (sparse)
- Only actual terms present have weights
- Weights range 0.1 to 1.0
- Domain terms preserved automatically

### Domain-Specific Terms

**How custom terminologies are handled:**

```python
Query: "How to create prefab in WM RN?"

Tokenization:
  ["how", "to", "create", "prefab", "in", "wm", "rn"]

Stopword filter:
  ["create", "prefab", "wm", "rn"]  # Domain terms preserved!

Why it works:
  ✓ "prefab" not in English stopwords → kept
  ✓ "wm" not in English stopwords → kept
  ✓ "rn" not in English stopwords → kept
  ✓ Short terms get same weight as long terms
```

**Key insight:** Domain terminology naturally stands out because:
1. Not in common English stopwords
2. Often unique/rare → high discriminative value
3. TF weighting gives proper importance

### Sparse vs Dense Comparison

| Property | Dense (BGE) | Sparse (BM25) |
|----------|-------------|---------------|
| **Dimensions** | 768 (all filled) | ~100k (mostly zero) |
| **Captures** | Semantic meaning | Exact keywords |
| **Model** | Neural network | Statistical |
| **Package** | sentence-transformers | Custom (stdlib only) |
| **Example** | "REST API" ≈ "web service" | "REST" ≠ "web" |
| **Best for** | Conceptual similarity | Technical terms |
| **Storage** | 768 floats (3KB) | ~5-20 indices (100 bytes) |

---

## Layer 3: Retrieval & Fusion

### Parallel Execution

**Architecture:**
```python
# Both searches start simultaneously
documents_task = retriever.search(dense_vector, sparse_vector)
videos_task = academy.search_videos(query, limit=3)

# Wait for both (or timeout)
results = await asyncio.gather(
    documents_task,
    videos_task,
    return_exceptions=True  # Don't fail if one fails
)

documents = results[0] if isinstance(results[0], list) else []
videos = results[1] if isinstance(results[1], list) else []
```

**Timing:**
```
Sequential:  docs (2s) → videos (1.5s) = 3.5s total
Parallel:    max(docs (2s), videos (1.5s)) = 2s total
Savings:     1.5s (43% faster)
```

### 3A: Qdrant Hybrid Search

**Process:**

```python
# Execute TWO searches in Qdrant
dense_results = qdrant.query_points(
    collection="wavemaker_docs",
    query=dense_vector.tolist(),  # [0.23, -0.45, ...]
    using="dense",
    limit=20
)
# Returns: 20 docs sorted by cosine similarity

sparse_results = qdrant.query_points(
    collection="wavemaker_docs",
    query=SparseVector(indices=[42351, ...], values=[1.0, ...]),
    using="sparse",
    limit=20
)
# Returns: 20 docs sorted by sparse dot product
```

**Storage in Qdrant:**

Each document chunk stored with:
```json
{
    "id": "doc_123_chunk_5",
    "vector": {
        "dense": [0.12, -0.34, ...],    // 768 floats
        "sparse": {
            "indices": [123, 456, ...],  // Term indices
            "values": [0.8, 0.6, ...]    // Term weights
        }
    },
    "payload": {
        "content": "To create a REST API...",
        "title": "REST API Development",
        "url": "https://docs.wavemaker.com/learn/apis/rest",
        "section": "Creating APIs"
    }
}
```

**Search mechanisms:**

**Dense search:**
```
Cosine similarity = dot(query_dense, doc_dense) / (||query|| * ||doc||)

Since vectors are normalized (||v|| = 1):
Cosine similarity = dot(query_dense, doc_dense)

Higher score = more similar
```

**Sparse search:**
```
Dot product = Σ(query_sparse[i] * doc_sparse[i])

Only non-zero indices contribute:
  If query has "prefab" and doc has "prefab" → both indices match → score++
  If query has "prefab" but doc doesn't → no match → no contribution

Higher score = more keyword overlap
```

### Reciprocal Rank Fusion (RRF)

**Purpose:** Merge two ranked lists (dense results + sparse results)

**Why not weighted average?**
```
Problem: Different score scales
  Dense scores: 0.5 - 0.95 (cosine similarity)
  Sparse scores: 2.5 - 15.8 (dot product)

Can't do: 0.5 * dense + 0.5 * sparse  ← Wrong scales!
```

**RRF Solution:**

**Formula:**
```python
RRF_score(doc) = Σ 1/(k + rank_i)

Where:
  k = 60 (smoothing constant)
  rank_i = position in result list i (1-indexed)
  Σ = sum over all lists where doc appears
```

**Example:**

```
Doc A:
  Dense: rank 1 → 1/(60+1) = 0.0164
  Sparse: rank 5 → 1/(60+5) = 0.0154
  RRF: 0.0164 + 0.0154 = 0.0318

Doc B:
  Dense: rank 3 → 1/(60+3) = 0.0159
  Sparse: not found → 0
  RRF: 0.0159

Doc C:
  Dense: not found → 0
  Sparse: rank 1 → 1/(60+1) = 0.0164
  RRF: 0.0164

Ranking: A (0.0318) > C (0.0164) > B (0.0159)
```

**Why RRF works:**
1. **Scale-independent** - Only ranks matter, not scores
2. **Rewards consensus** - Docs in multiple lists score higher
3. **Graceful handling** - Works even if doc only in one list
4. **No tuning needed** - k=60 is standard, works well

**Implementation:**
```python
def _rrf_fusion(dense_results, sparse_results, k=60):
    scores = {}

    # Process dense results
    for rank, result in enumerate(dense_results, start=1):
        doc_id = result.id
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank)

    # Process sparse results
    for rank, result in enumerate(sparse_results, start=1):
        doc_id = result.id
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank)

    # Sort by RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 3B: Academy MCP Video Search

**Model Context Protocol (MCP)** - Standard for AI tool integration

**Connection flow:**
```
1. Create streamable HTTP client
2. Connect with timeout (10s)
3. Initialize MCP session
4. Call tool: wm-academy-semantic-search
5. Parse JSON response
6. Extract video metadata
```

**Session management:**
```python
Session lifecycle:
  ├─ Create: On first query
  ├─ Reuse: For queries within 5 minutes
  ├─ Refresh: Auto-recreate after 5 minutes
  └─ Timeout handling: 2 retries with fresh session
```

**Tool call:**
```python
result = await session.call_tool(
    name="wm-academy-semantic-search",
    arguments={
        "query": "REST API",
        "limit": 3
    }
)

# Response format:
{
    "headers": {},
    "body": [
        {
            "id": "12345",
            "title": "Build REST API",
            "description": "Learn how to create REST APIs",
            "moduleName": "API Development",
            "code": "CHAP_45",
            "link": "https://academy.wavemaker.com/Watch?wm=CHAP_45"
        },
        ...
    ],
    "statusCodeValue": 200
}
```

**Graceful degradation:**
```python
# If MCP fails:
try:
    videos = await academy.search_videos(query)
except Exception as e:
    logger.warning(f"Video search failed: {e}")
    videos = []  # Continue without videos

# User still gets documentation, just no video recommendations
```

---

## Layer 4: Reranking

### Bi-Encoder vs Cross-Encoder

**Bi-Encoder (BGE - used in Layer 2):**
```
Query:    "How to create API" → Encoder → [0.23, -0.45, ...]
Document: "API guide..."      → Encoder → [0.21, -0.43, ...]
                                              ↓
                                    Cosine Similarity
                                              ↓
                                          Score: 0.94
```
- Encode query and document **separately**
- Compare vectors in embedding space
- **Fast** - pre-compute all docs once
- **Good** accuracy

**Cross-Encoder (Reranker - used in Layer 4):**
```
Input: "[CLS] How to create API [SEP] API guide... [SEP]"
         ↓
    Single Encoder (sees both together)
         ↓
    Classification Head
         ↓
    Score: 0.82 (relevance probability)
```
- Encode query + document **together**
- Model sees interaction between them
- **Slow** - must re-encode for every query-doc pair
- **Excellent** accuracy

### When to Rerank

**Decision logic:**
```python
def should_rerank(documents):
    if not documents:
        return False

    # Skip if disabled
    if not settings.reranker_enabled:
        return False

    # Skip if top result is very confident
    top_rrf_score = documents[0].rrf_score
    if top_rrf_score > 0.04:
        logger.debug(f"Top score {top_rrf_score} high, skip rerank")
        return False

    return True
```

**Insight:** If RRF already found a strong match, don't waste time/money reranking.

### Jina Reranker

**Model:** jina-reranker-v2-base-multilingual

**API call:**
```python
response = await httpx.post(
    "https://api.jina.ai/v1/rerank",
    headers={"Authorization": f"Bearer {jina_api_key}"},
    json={
        "model": "jina-reranker-v2-base-multilingual",
        "query": "How to create REST API",
        "top_n": 5,
        "documents": [
            "To create REST API, navigate to...",  # doc 1 (truncated to 2000 chars)
            "REST API endpoints define...",         # doc 2
            ...                                      # 20 total docs
        ]
    }
)

# Response:
{
    "results": [
        {"index": 3, "relevance_score": 0.92},  # Doc 4 is most relevant
        {"index": 0, "relevance_score": 0.81},  # Doc 1 second
        {"index": 7, "relevance_score": 0.75},  # Doc 8 third
        {"index": 1, "relevance_score": 0.68},  # Doc 2 fourth
        {"index": 15, "relevance_score": 0.62}  # Doc 16 fifth
    ]
}
```

**Reordering:**
```
Before reranking (RRF order):
  [Doc1, Doc2, Doc3, Doc4, ...]

After reranking (Jina cross-encoder scores):
  [Doc4, Doc1, Doc8, Doc2, Doc16]  # Top 5 most relevant
```

### Why Jina API vs Local

**Jina API:**
- ✓ No local GPU needed
- ✓ No MLX/PyTorch issues on Apple Silicon
- ✓ Fast inference (optimized servers)
- ✓ Multilingual support
- ✗ Small API cost (~$0.0001 per query)
- ✗ Network latency

**Local cross-encoder:**
- ✓ Free (no API costs)
- ✓ No network dependency
- ✗ Requires PyTorch
- ✗ MLX backend issues on M1/M2/M3 Macs
- ✗ Slower on CPU

**Decision:** Use Jina as default for reliability.

---

## Layer 5: Generation

### Context Construction

**Prompt structure:**
```python
system_prompt = """
You are a WaveMaker documentation expert.
Answer questions using ONLY the provided context.
Cite sources using [1], [2], etc.
Include video links as markdown at the end.
"""

context = ""
# Add documents
for i, doc in enumerate(documents[:5], 1):
    context += f"[{i}] {doc.title}\n{doc.content}\n\n"

# Add videos
context += "\n**Related Videos:**\n"
for video in videos:
    context += f"📺 [{video.title}]({video.url})\n"

user_message = f"{context}\n\nQuestion: {query}"
```

**Claude API call:**
```python
response = await anthropic.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ],
    stream=True,  # For streaming responses
    max_tokens=2048
)
```

### Streaming

**Server-Sent Events (SSE) format:**
```
data: {"type": "text", "content": "To create"}
data: {"type": "text", "content": " a REST API"}
data: {"type": "text", "content": ", navigate to..."}
data: {"type": "sources", "sources": [{...}, {...}]}
data: {"type": "videos", "videos": [{...}]}
data: {"type": "done", "cached": false}
```

**Benefits:**
- User sees response immediately (better UX)
- Lower perceived latency
- Can cancel long responses

---

## Performance Characteristics

### Latency Breakdown

```
Cold query (no cache):
├─ Layer 1: Cache check         5ms     ⚡
├─ Layer 2: Embeddings          100ms   ⏱️
│   ├─ Dense (BGE)              80ms
│   └─ Sparse (custom)          20ms
├─ Layer 3: Parallel retrieval  2000ms  🐌
│   ├─ Qdrant search           200ms    (runs in parallel)
│   └─ Academy MCP             1500ms   (runs in parallel)
│   └─ Wait for slowest        2000ms
├─ Layer 4: Reranking          500ms    ⏱️
│   └─ Jina API (20→5 docs)
└─ Layer 5: Generation         2500ms   🐌
    └─ Claude streaming
────────────────────────────────────────
Total:                          5.1s

Warm query (cache hit):
└─ Layer 1: Exact cache         5ms     ⚡
────────────────────────────────────────
Total:                          5ms     (1000x faster!)
```

### Cost Breakdown

```
Per query (cache miss):
├─ Embeddings (BGE)           $0      (local)
├─ Qdrant search              $0      (free tier)
├─ Academy MCP                $0      (internal service)
├─ Reranking (Jina)           $0.0001
└─ Generation (Claude)        $0.005
──────────────────────────────────────
Total:                         $0.0051 per query

Per query (cache hit):
└─ Redis lookup               $0      (local/managed)
──────────────────────────────────────
Total:                         $0

With 50% cache hit rate:
Average cost:                  $0.0025 per query
```

### Scalability

**Bottlenecks:**

1. **Generation (Claude)** - Most expensive, slowest
   - Solution: Aggressive caching (Tier 1 + 2)
   - Result: 50-70% cache hit rate in production

2. **MCP Video Search** - External service, can be slow
   - Solution: Parallel execution + graceful degradation
   - Result: Doesn't block document retrieval

3. **Reranking** - API call overhead
   - Solution: Skip when top RRF score is confident
   - Result: ~30% of queries skip reranking

**Capacity:**
```
Assumed limits:
├─ Claude: 1000 req/min  → 60K queries/hour
├─ Jina: 500 req/min     → 30K queries/hour
├─ Qdrant: 10K req/min   → 600K queries/hour
└─ Academy: 100 req/min  → 6K queries/hour

System limit: 6K queries/hour (limited by Academy MCP)
With cache (50% hit): 12K queries/hour
```

---

## Key Design Decisions

### 1. Why 3-tier cache instead of 2?

**Tier 3 (embedding cache) is an optimization:**
- Embeddings are expensive to compute (100ms)
- Same text → same embedding (deterministic)
- Reuse across different cache entries
- Longer TTL makes sense (embeddings don't change)

### 2. Why hybrid search (dense + sparse)?

**Different failure modes:**
```
Query: "How to use WaveMaker prefabs?"

Dense only:
  ✓ Finds: component architecture docs
  ✗ Misses: docs that specifically say "prefab"

Sparse only:
  ✓ Finds: docs with exact word "prefab"
  ✗ Misses: conceptually similar docs using "reusable component"

Hybrid (RRF fusion):
  ✓ Finds: docs with "prefab" AND conceptually similar docs
  ✓ Best of both worlds
```

### 3. Why rerank after retrieval?

**Two-stage is optimal:**
```
Stage 1 (Retrieval): Fast approximate search
  ├─ Bi-encoder: O(1) vector lookup
  └─ Returns: 20 candidates in 200ms

Stage 2 (Reranking): Slow precise scoring
  ├─ Cross-encoder: O(n) per query-doc pair
  └─ Process: 20 docs in 500ms

If we used cross-encoder for all 10K docs:
  └─ Time: 10K * 25ms = 250 seconds (too slow!)
```

### 4. Why parallel retrieval?

**Independent operations should not block each other:**
```
Documents: Core functionality
Videos: Bonus content

If videos fail → user still gets docs (acceptable)
If docs fail → videos alone aren't useful (problem)

Parallel execution:
  ✓ Total time = max(docs, videos) not sum
  ✓ Video failure doesn't block docs
  ✓ Better user experience
```

### 5. Why MCP instead of REST API?

**Standards-based integration:**
- Tool discovery (list_tools)
- Session management (persistent connections)
- Auto-reconnect (built-in retry logic)
- Timeout handling (protocol-level)
- Future-proof (Academy can add tools without breaking client)

---

## Glossary

**Bi-encoder:** Embeds query and documents separately, compares in vector space
**Cosine similarity:** Measure of angle between vectors (0=opposite, 1=identical)
**Cross-encoder:** Encodes query+document together for precise relevance
**Dense vector:** All dimensions filled (768 floats), captures semantics
**Embedding:** Fixed-size numerical representation of text
**MCP:** Model Context Protocol - standard for AI tool integration
**RRF:** Reciprocal Rank Fusion - algorithm to merge ranked lists
**Sparse vector:** Mostly zeros, only keyword indices filled
**TF:** Term Frequency - how often a term appears
**TTL:** Time To Live - how long to keep cached data

---

*Last updated: 2026-02-02*
