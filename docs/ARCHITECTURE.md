# WaveMaker Docs Agent - Architecture Guide

A simple explanation of how our AI documentation assistant works, layer by layer.

> **📘 For detailed technical explanations**, see [TECHNICAL_DEEP_DIVE.md](./TECHNICAL_DEEP_DIVE.md) which covers:
> - 3-tier cache implementation details
> - Dense vs sparse embeddings (BGE + BM25)
> - RRF fusion algorithm
> - Bi-encoder vs cross-encoder comparison
> - Performance characteristics and cost breakdown

---

## 🎯 The Problem We're Solving

**User asks:** "How do I create a REST API in WaveMaker?"

**Challenge:** We have 500+ documentation pages. How do we:
1. Find the 3-5 most relevant pages out of 500+?
2. Get relevant video tutorials from Academy?
3. Give the AI only those resources (LLMs have token limits)?
4. Generate an accurate, cited answer with video links?

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                   │
│                 "How to create a REST API?"                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: CACHE (3 Tiers)                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Tier 1:     │→ │ Tier 2:      │→ │ Tier 3:      │               │
│  │ Exact Match │  │ Semantic     │  │ Embedding    │               │
│  │ (instant)   │  │ (similarity) │  │ (reuse)      │               │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
│  "Same question asked before? Return cached!"                       │
└─────────────────────────────────────────────────────────────────────┘
                                │ Cache Miss
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: EMBEDDINGS                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  "How to create REST API" → [0.23, -0.45, 0.89, ...]         │   │
│  │                              (768-dimensional vector)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  Model: BAAI/bge-base-en-v1.5 (runs locally, free)                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: PARALLEL RETRIEVAL ⚡                                      │
│  ┌─────────────────┐         ┌─────────────────┐                   │
│  │  Qdrant Cloud   │         │  Academy MCP    │                   │
│  │  (Vector DB)    │         │  (Video Server) │                   │
│  │  ↓ Returns 30   │         │  ↓ Returns 3-5  │                   │
│  │  docs           │         │  videos         │                   │
│  └─────────────────┘         └─────────────────┘                   │
│         ↓                             ↓                              │
│    Documents                      Videos                             │
│    (both retrieved in parallel - no blocking!)                      │
└─────────────────────────────────────────────────────────────────────┘
                                │ 30 docs + 3-5 videos
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RERANKING (Documents Only)                                 │
│  ┌─────────────────┐                                                │
│  │   Jina Rerank   │  ← Precise relevance scoring                   │
│  │   (API)         │  ← 30 docs → 5 best docs                       │
│  └─────────────────┘                                                │
│  Note: Videos are pre-ranked by Academy MCP, skip reranking         │
└─────────────────────────────────────────────────────────────────────┘
                                │ 5 documents + videos
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5: GENERATION                                                 │
│  ┌─────────────────┐                                                │
│  │  Claude LLM     │  ← Reads 5 docs + videos + question            │
│  │  (Anthropic)    │  ← Generates answer with citations [1][2]      │
│  │                 │  ← Includes video links in response            │
│  └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RESPONSE                                     │
│  "To create a REST API in WaveMaker, navigate to APIs → REST →     │
│   New [1]. Define your endpoints and methods [2]...                 │
│                                                                      │
│   **Related Videos:**                                                │
│   - [Build REST API](https://academy.wavemaker.com/Watch?wm=CHAP_45)│
│   - [REST Variables](https://academy.wavemaker.com/Watch?wm=CHAP_67)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Layer-by-Layer Explanation

### Layer 1: Cache (Redis)

**What it does:** Remembers previous answers to avoid re-processing.

**Why we need it:**
- LLM calls cost money ($0.003-0.01 per query)
- Same questions get asked repeatedly
- Faster response time (~50ms vs ~3s)

**Three-tier caching system:**

| Tier | Type | How it Works | Example |
|------|------|-------------|---------|
| **1** | **Exact Cache** | Hash of question → answer | "What is AIRA?" matches "What is AIRA?" |
| **2** | **Semantic Cache** | Similar meaning → answer | "What is AIRA?" matches "Tell me about AIRA" |
| **3** | **Embedding Cache** | Reuses computed vectors | Same text → Skip re-embedding (24x longer TTL) |

**Example flow:**
```
User 1: "What is AIRA?" → Processed, answer + embedding cached
User 2: "Tell me about AIRA" → Semantic match (95% similar) → Return cached answer
User 3: "What is AIRA?" → Exact match → Instant return (~10ms)
                        → Embedding reused from cache (saves ~50ms)
```

**Technology:** Redis (fast in-memory database)

---

### Layer 2: Embeddings (BAAI/bge-base-en-v1.5)

**What it does:** Converts text into numbers that capture meaning.

**Why we need it:**
- Computers can't understand "meaning" directly
- Numbers allow mathematical similarity comparison
- Similar meanings → similar numbers

**How it works:**
```
"How to create REST API"     → [0.23, -0.45, 0.89, ... 768 numbers]
"Creating RESTful services"  → [0.21, -0.43, 0.91, ... 768 numbers]
                                     ↑
                              Very similar vectors! (cosine similarity ~0.95)

"What is database security?" → [0.67, 0.12, -0.34, ... 768 numbers]
                                     ↑
                              Very different vector (cosine similarity ~0.3)
```

**Why bge-base-en-v1.5?**
- High quality for technical docs
- Runs locally (no API cost)
- Fast (768-dim is good balance of quality/speed)

---

### Layer 3: Parallel Retrieval (Documents + Videos) ⚡

**What it does:** Simultaneously searches two sources:
1. **Qdrant** - Finds 30 most similar documentation pages
2. **Academy MCP** - Finds 3-5 relevant video tutorials

**Why parallel execution?**
```
❌ Sequential (old way):
Document search: 2.5s
   ↓
Video search: 1.5s
   ↓
Total: 4.0s

✅ Parallel (current):
Document search: 2.5s  ←┐
                         ├→ Total: 2.5s (wait for slowest)
Video search: 1.5s      ←┘
```

**Key benefit:** Video failure doesn't block document retrieval!

#### 3A: Document Retrieval (Qdrant Vector Database)

**How it works:**
```
Query Vector: [0.23, -0.45, 0.89, ...]
                    ↓
              Qdrant Search
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Results (sorted by similarity):                             │
│  1. "REST API Development" (score: 0.94)                    │
│  2. "API Endpoints Guide" (score: 0.91)                     │
│  3. "Creating Services" (score: 0.88)                       │
│  ... 27 more docs ...                                       │
└─────────────────────────────────────────────────────────────┘
```

**Why Qdrant Cloud?**
- Managed service (no server maintenance)
- Fast vector search at scale
- Free tier available for development

**Why 30 documents?**
- More candidates = better chance of finding best matches
- Will be filtered down by reranking next

#### 3B: Video Search (Academy MCP)

**What is MCP?** Model Context Protocol - a standard for AI tools to communicate.

**How it works:**
```
Query: "REST API"
    ↓
MCP Client connects to Academy server
    ↓
Calls tool: wm-academy-semantic-search
    ↓
Returns: [
  {title: "Build REST API", link: "...", code: "CHAP_45"},
  {title: "REST Variables", link: "...", code: "CHAP_67"},
  {title: "API Testing", link: "...", code: "CHAP_89"}
]
```

**Why MCP instead of REST API?**
- **Standard protocol** - Works with any MCP-compliant server
- **Tool discovery** - Can ask "what tools are available?"
- **Session management** - Persistent connection, auto-reconnect
- **Timeout handling** - Graceful failure if video service is slow/down

**Smart features:**
- **Auto-reconnect:** If connection times out, automatically retry
- **Session refresh:** Recreates connection every 5 minutes
- **Graceful degradation:** If videos fail, still return documents

---

### Layer 4: Reranking (Jina AI)

**What it does:** Precisely re-scores 30 docs to find the 5 best.

**Why only documents, not videos?**
- Videos are already pre-ranked by Academy MCP's semantic search
- Videos are "bonus" content, not primary sources
- Saves API costs and reduces latency

**The difference:**

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Embedding Search** | ~10ms for 10K docs | Good | Initial filtering |
| **Cross-Encoder Rerank** | ~500ms for 30 docs | Excellent | Final selection |

**Example:**
```
Before Reranking (30 docs, embedding scores):
  1. "REST API Development" (0.94) ← Looks best
  2. "API Endpoints Guide" (0.91)
  3. "Creating Services" (0.88)
  4. "Database REST API" (0.85)  ← Actually most relevant!
  ...

After Reranking (Jina scores):
  1. "Database REST API" (0.82)  ← Jina found this is best match
  2. "REST API Development" (0.76)
  3. "API Endpoints Guide" (0.71)
  4. "Creating Services" (0.45)
  5. "Import APIs" (0.42)
```

**Why Jina AI?**
- High quality multilingual reranker
- API-based (no local GPU needed)
- Works around MLX issues on Apple Silicon

---

### Layer 5: Generation (Claude LLM)

**What it does:** Reads the 5 documents + videos and generates a helpful answer.

**Why we need it:**
- Humans don't want to read 5 documents + 3 videos
- LLM synthesizes information into clear answer
- Adds structure, examples, citations, and video links

**The prompt structure:**
```
SYSTEM: You are a WaveMaker documentation expert. Answer using
        the provided context and cite sources using [1], [2], etc.
        Include video links as clickable markdown.

CONTEXT:
[1] REST API Development - "To create a REST API, go to..."
[2] API Endpoints Guide - "You can define GET, POST methods..."
[3] Creating Services - "Services in WaveMaker include..."
[4] Database REST API - "Database APIs auto-generate CRUD..."
[5] Import APIs - "External APIs can be imported..."

VIDEOS:
📺 [Build REST API](https://academy.wavemaker.com/Watch?wm=CHAP_45)
📺 [REST Variables](https://academy.wavemaker.com/Watch?wm=CHAP_67)

QUESTION: How do I create a REST API in WaveMaker?
```

**Response:**
```
To create a REST API in WaveMaker, follow these steps:

1. Navigate to **APIs → REST → New** [1]
2. Define your endpoint path and HTTP method (GET, POST, etc.) [2]
3. For database entities, WaveMaker auto-generates CRUD APIs [4]

The API will be available at `/rest/your-endpoint` [1][2].

**Related Videos:**
- [Build REST API](https://academy.wavemaker.com/Watch?wm=CHAP_45)
- [REST Variables](https://academy.wavemaker.com/Watch?wm=CHAP_67)
```

**Why Claude?**
- Excellent at following instructions
- High quality, accurate responses
- Good at citing sources and formatting markdown links

---

## 🔄 Complete Example Flow

**User Query:** "How do I add authentication to my WaveMaker app?"

```
1. CACHE CHECK
   ├─ Exact match? No
   └─ Semantic match? No (new question)

2. EMBEDDING
   └─ Query → [0.12, 0.89, -0.34, ...] (768 dims)

3. PARALLEL RETRIEVAL ⚡
   ├─ Qdrant: Returns 30 docs about security, authentication, users...
   └─ Academy MCP: Returns 3 videos about authentication setup
   (Both complete in ~2s total, not 4s sequential!)

4. RERANKING (Jina) - Documents only
   ├─ Input: 30 documents
   └─ Output: Top 5 most relevant
      1. "WaveMaker Security Configuration" (0.89)
      2. "OpenID Authentication" (0.82)
      3. "SAML SSO Setup" (0.78)
      4. "Role-Based Access Control" (0.71)
      5. "Login Page Customization" (0.65)

5. GENERATION (Claude)
   └─ Reads 5 docs + 3 videos, generates:

      "To add authentication to your WaveMaker app:

      1. Go to **Security** settings [1]
      2. Choose a provider:
         - OpenID/OAuth for Google, Auth0 [2]
         - SAML for enterprise SSO [3]
      3. Configure roles for access control [4]

      The login page is auto-generated but customizable [5].

      **Related Videos:**
      - [Security Setup](https://academy.wavemaker.com/Watch?wm=CHAP_12)
      - [OAuth Configuration](https://academy.wavemaker.com/Watch?wm=CHAP_34)"

6. CACHE STORE
   └─ Save response for future similar questions
```

---

## 📊 Cost & Performance Summary

| Layer | Cost | Latency | Notes |
|-------|------|---------|-------|
| Cache | Free | ~5ms | Redis (local or managed) |
| Embedding | Free | ~100ms | Runs locally |
| Documents (Qdrant) | ~$0 | ~200ms | Free tier, runs in parallel |
| Videos (Academy MCP) | ~$0 | ~1.5s | Runs in parallel with docs |
| **Parallel Layer 3** | **~$0** | **~2s** | **Waits for slowest only** |
| Reranking | ~$0.0001 | ~500ms | Jina API |
| Generation | ~$0.005 | ~2-3s | Claude API |
| **Total** | **~$0.005/query** | **~4-5s** | (cached: free, ~50ms) |

---

## 🛠️ Why This Architecture?

### The RAG Pattern (Retrieval-Augmented Generation)

Our architecture follows the **RAG pattern**:

```
Traditional LLM:  Question → LLM → Answer (might hallucinate)

RAG:              Question → Retrieve Docs + Videos → LLM + Context → Grounded Answer
```

**Benefits:**
1. **Accurate** - LLM only answers from real documentation
2. **Up-to-date** - Re-index docs when content changes
3. **Verifiable** - Citations point to source documents
4. **Cost-effective** - Only process relevant docs, not entire corpus
5. **Multimedia** - Combines text docs with video tutorials

### Why Parallel Execution?

```
Sequential Problems:
❌ Video service slow? Entire query slow.
❌ Video service down? Entire query fails.
❌ Wasted time waiting for both sequentially.

Parallel Benefits:
✅ Total latency = max(docs, videos), not sum
✅ Video failure doesn't block documents
✅ Better user experience (faster responses)
```

---

## 🧩 Component Choices Summary

| Component | Choice | Why |
|-----------|--------|-----|
| **Cache** | Redis | Fast, simple, semantic search support |
| **Embedding** | bge-base-en-v1.5 | Free, local, high quality |
| **Vector DB** | Qdrant Cloud | Managed, fast, free tier |
| **Video Search** | Academy MCP | Standards-compliant, auto-reconnect, graceful failure |
| **Reranker** | Jina AI | API-based (avoids local GPU issues) |
| **LLM** | Claude Sonnet 4.5 | High quality, follows instructions well, good markdown |
| **API** | FastAPI | Async, fast, streaming support |

---

## 🎓 Key Takeaways for Developers

1. **Embeddings are dimensionality reduction**
   - Text → Fixed-size numbers that capture meaning

2. **Two-stage retrieval is optimal**
   - Fast approximate search first (vectors)
   - Slow precise rerank second (cross-encoder)

3. **Parallel execution reduces latency**
   - Independent operations run concurrently
   - Failures in one don't block the other

4. **Caching saves money and time**
   - Most questions are repeats or similar

5. **LLMs need context**
   - Don't ask LLM to know everything
   - Give it relevant documents AND videos to reference

6. **Citations build trust**
   - Users can verify answers against sources
   - Video links provide visual learning paths

7. **Graceful degradation is critical**
   - If videos fail, still return documentation
   - System remains useful even with partial failures

---

## 🔍 MCP Integration Details

### What is Model Context Protocol?

**MCP** is like "USB for AI" - a standard protocol that lets AI systems connect to external tools and data sources.

**Our use case:**
```
Docs Agent ←──MCP──→ Academy Server
            (Tool: wm-academy-semantic-search)
```

### Why MCP over REST API?

| Feature | REST API | MCP |
|---------|----------|-----|
| **Discovery** | Hardcoded endpoints | `list_tools()` discovers capabilities |
| **Protocol** | Custom per API | Standardized JSON-RPC |
| **Connection** | One-off requests | Persistent session |
| **Reconnection** | Manual | Auto-reconnect built-in |
| **Timeouts** | Custom handling | Built into protocol |

### Connection Lifecycle

```
Query 1 (t=0s):
  ├─ Create MCP session
  ├─ Initialize protocol
  └─ Call tool: wm-academy-semantic-search
  ✅ Videos returned

Query 2 (t=30s):
  └─ Reuse existing session
  ✅ Videos returned (faster, no setup overhead)

Query 3 (t=5min):
  ├─ Session aged > 5min
  ├─ Auto-recreate session
  └─ Call tool
  ✅ Videos returned

Connection timeout:
  ├─ Attempt 1: Timeout after 15s
  ├─ Close stale session
  ├─ Attempt 2: Retry with new session
  └─ If still fails: Return docs without videos
  ✅ User still gets documentation
```

**Smart features:**
- Sessions auto-refresh every 5 minutes
- Connection timeouts: 10s (connect), 15s (tool call)
- 2 retries with fresh sessions
- Graceful fallback: docs-only if videos unavailable

---

## 📈 Future Enhancements

### Planned Improvements

1. **Query Understanding**
   - Intent classification (how-to vs. conceptual vs. troubleshooting)
   - Auto-detect if user needs video tutorials vs. text docs

2. **Enhanced Video Integration**
   - Timestamp-specific links (jump to relevant part of video)
   - Video transcript search for better matching

3. **Feedback Loop**
   - Track which videos users actually watch
   - Use feedback to improve video ranking

4. **Multi-modal Responses**
   - Interactive diagrams from documentation
   - Code playground integration

5. **Conversation History**
   - Remember previous questions in session
   - Provide context-aware follow-ups

---

*Last updated: 2026-02-02*
