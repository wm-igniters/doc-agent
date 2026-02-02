# Quick Reference Guide

Fast lookup for key concepts, commands, and configurations.

---

## System Architecture

```
Query → Cache (Tier 1→2→3) → Embeddings (Dense+Sparse)
      → Retrieval (Docs∥Videos) → Rerank → Generate → Response
```

---

## Layer Summary

| Layer | Component | Latency | Purpose |
|-------|-----------|---------|---------|
| 1 | Redis Cache (3 tiers) | 5-50ms | Avoid reprocessing |
| 2 | BGE + BM25 Embeddings | 100ms | Convert text to vectors |
| 3 | Qdrant + Academy MCP | 2s | Find relevant docs + videos |
| 4 | Jina Reranker | 500ms | Precise relevance scoring |
| 5 | Claude LLM | 2-3s | Generate cited answer |

---

## Three-Tier Cache

| Tier | Type | Lookup | What it stores |
|------|------|--------|----------------|
| 1 | Exact | Hash → Response | Complete answer JSON |
| 2 | Semantic | Vector similarity (≥0.95) | Query + embedding + response |
| 3 | Embedding | Hash → Vector | Just the 768-dim embedding |

**Flow:** Tier 1 (exact) → Tier 3 (get embedding) → Tier 2 (semantic) → Miss → Continue

---

## Embeddings

### Dense Vectors (BGE)

```python
Model: BAAI/bge-base-en-v1.5
Dimensions: 768 (all filled)
Purpose: Semantic similarity
Example: "REST API" ≈ "web service"
Package: sentence-transformers
```

### Sparse Vectors (BM25)

```python
Algorithm: Custom TF weighting
Dimensions: ~100k (mostly zero)
Purpose: Keyword matching
Example: "REST" ≠ "web" (exact term needed)
Package: None (stdlib: Counter, re, hashlib)
```

### Why Both?

**Hybrid search** = Best of both worlds
- Dense finds conceptually similar docs
- Sparse finds exact technical terms
- RRF merges both rankings

---

## Reciprocal Rank Fusion (RRF)

**Formula:**
```
RRF_score = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
```

**Why?**
- Scale-independent (works with different score ranges)
- Rewards consensus (docs in both lists rank higher)
- No tuning needed (k=60 is standard)

**Example:**
```
Doc appears in both:     0.0164 + 0.0154 = 0.0318 ← High
Doc in dense only:       0.0159 + 0      = 0.0159 ← Low
```

---

## Bi-Encoder vs Cross-Encoder

### Bi-Encoder (BGE - Layer 2)

```
Query   → Encoder → Vector1  ─┐
Document → Encoder → Vector2  ─┤→ Cosine Similarity → Score
                               ─┘

Speed: Fast (vectors pre-computed)
Accuracy: Good
Usage: Initial retrieval (10K docs)
```

### Cross-Encoder (Jina - Layer 4)

```
[Query + Document] → Encoder → Relevance Score

Speed: Slow (must encode each pair)
Accuracy: Excellent
Usage: Reranking (20 docs → 5 best)
```

---

## Model Context Protocol (MCP)

**What:** Standard protocol for AI tool integration

**Why use it?**
- ✓ Tool discovery (`list_tools()`)
- ✓ Session management (persistent connection)
- ✓ Auto-reconnect (2 retries)
- ✓ Graceful failure (docs continue if videos fail)

**Our usage:**
```python
Tool: wm-academy-semantic-search
Input: {"query": "REST API", "limit": 3}
Output: [{"title": "...", "link": "...", "code": "CHAP_45"}, ...]
```

---

## Configuration

### Environment Variables

```bash
# === Models ===
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_PROVIDER=jina  # or "local"

# === Qdrant ===
QDRANT_URL=https://xxx.cloud.qdrant.io
QDRANT_API_KEY=xxx
QDRANT_COLLECTION_NAME=wavemaker_docs

# === Redis Cache ===
REDIS_URL=redis://localhost:6379
CACHE_TTL_HOURS=24
SEMANTIC_CACHE_THRESHOLD=0.95

# === Academy MCP ===
ACADEMY_MCP_URL=https://dev-academyservices.wavemaker.com/mcp

# === Anthropic ===
ANTHROPIC_API_KEY=xxx

# === Jina (if using jina reranker) ===
JINA_API_KEY=xxx

# === Retrieval ===
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
RERANK_ENABLED=true
```

### Tuneable Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CACHE_TTL_HOURS` | 24 | How long to cache responses |
| `SEMANTIC_CACHE_THRESHOLD` | 0.95 | Min similarity for semantic cache hit |
| `RETRIEVAL_TOP_K` | 20 | Docs retrieved before reranking |
| `RERANK_TOP_K` | 5 | Final docs after reranking |
| `RERANK_ENABLED` | true | Whether to rerank (skip if confident) |

---

## Performance

### Latency

```
Cache hit (Tier 1):     ~5ms      ⚡⚡⚡
Cache hit (Tier 2):     ~50ms     ⚡⚡
Full pipeline:          ~5s       ⏱️
  ├─ Embeddings:        100ms
  ├─ Retrieval:         2s (parallel)
  ├─ Reranking:         500ms
  └─ Generation:        2.5s
```

### Cost

```
Per query (no cache):   $0.005
Per query (cached):     $0
Cache hit rate:         50-70%
Average cost:           ~$0.0025 per query
```

### Scalability

```
Bottleneck: Academy MCP (100 req/min)
Capacity:   6K queries/hour (no cache)
           12K queries/hour (50% cache)
```

---

## API Endpoints

### POST /v1/chat

**Request:**
```json
{
  "query": "How to create REST API?",
  "stream": true,
  "include_sources": true
}
```

**Response (streaming):**
```
data: {"type": "text", "content": "To create..."}
data: {"type": "sources", "sources": [...]}
data: {"type": "videos", "videos": [...]}
data: {"type": "done", "cached": false}
```

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "redis": "ok",
    "qdrant": "ok",
    "academy_mcp": "ok"
  }
}
```

### POST /admin/index

**Request:**
```json
{
  "force_reindex": false,
  "branch": "release-12"
}
```

**Response:**
```json
{
  "status": "completed",
  "documents_processed": 523,
  "chunks_created": 2841,
  "duration_seconds": 245.3
}
```

---

## Common Operations

### Test the pipeline

```bash
# Start services
docker compose up -d redis qdrant

# Run agent
python -m src.main

# Test query
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How to create a REST API?", "stream": false}'
```

### Clear cache

```python
from src.core.cache import SemanticCache

cache = SemanticCache()
deleted = await cache.invalidate_all()
print(f"Deleted {deleted} cache entries")
```

### Reindex documents

```bash
# Via API
curl -X POST http://localhost:8000/admin/index \
  -H "Content-Type: application/json" \
  -d '{"force_reindex": true}'

# Via CLI
python -m src.indexer --force-reindex
```

### Check MCP connection

```python
from src.core.academy import AcademyClient

academy = AcademyClient()
videos = await academy.search_videos("REST API", limit=3)
print(f"Found {len(videos)} videos")
```

---

## Troubleshooting

### Cache not working

```bash
# Check Redis connection
redis-cli ping
# Should return: PONG

# Check cache entries
redis-cli keys "exact:*"
redis-cli keys "semantic:*"
```

### Embeddings slow

```python
# Check model loading
from src.core.embedder import Embedder

embedder = Embedder()
import time
start = time.time()
embedding = embedder.generate_dense_embedding("test")
print(f"Dense: {time.time() - start:.3f}s")  # Should be <100ms after warmup
```

### Qdrant connection issues

```bash
# Check collection exists
curl https://xxx.cloud.qdrant.io/collections \
  -H "api-key: xxx"

# Check collection points
curl https://xxx.cloud.qdrant.io/collections/wavemaker_docs \
  -H "api-key: xxx"
```

### MCP timeout

```python
# Increase timeouts in src/core/academy.py
MCP_CONNECT_TIMEOUT = 20.0  # Default: 10.0
MCP_CALL_TIMEOUT = 30.0     # Default: 15.0
```

### Reranking errors

```bash
# Switch to local reranker if Jina fails
export RERANKER_PROVIDER=local

# Or disable reranking temporarily
export RERANK_ENABLED=false
```

---

## Key Files

```
src/
├── core/
│   ├── cache.py          # 3-tier caching (Redis)
│   ├── embedder.py       # BGE + BM25 embeddings
│   ├── retriever.py      # Qdrant hybrid search + RRF
│   ├── academy.py        # MCP client for videos
│   ├── reranker.py       # Jina/local cross-encoder
│   ├── generator.py      # Claude LLM
│   └── pipeline.py       # Main orchestration
├── api/
│   ├── routes.py         # FastAPI endpoints
│   └── models.py         # Request/response schemas
├── indexer.py            # Document indexing
└── main.py               # Entry point

docs/
├── ARCHITECTURE.md       # High-level overview
├── TECHNICAL_DEEP_DIVE.md  # Detailed explanations
└── QUICK_REFERENCE.md    # This file
```

---

## Glossary Quick Reference

| Term | Definition |
|------|------------|
| **BGE** | BAAI General Embeddings - dense vector model |
| **BM25** | Best Matching 25 - keyword weighting algorithm |
| **Cosine similarity** | Angle between vectors (0-1, higher = more similar) |
| **Cross-encoder** | Encodes query+doc together (accurate, slow) |
| **Bi-encoder** | Encodes separately (fast, good) |
| **Dense vector** | All dims filled (semantic) |
| **Sparse vector** | Mostly zeros (keywords) |
| **MCP** | Model Context Protocol |
| **RAG** | Retrieval-Augmented Generation |
| **RRF** | Reciprocal Rank Fusion |
| **TTL** | Time To Live (cache expiry) |

---

*Last updated: 2026-02-02*
