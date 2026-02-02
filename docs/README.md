# Documentation Index

Comprehensive documentation for the WaveMaker Docs Agent RAG pipeline.

---

## 📚 Documentation Structure

### [ARCHITECTURE.md](./ARCHITECTURE.md) - Start Here!

**Best for:** Understanding the system at a high level

**Contents:**
- Problem statement and motivation
- Visual architecture diagrams
- Layer-by-layer explanations
- Complete example flow
- Why this architecture?
- MCP integration overview
- Future enhancements

**When to read:** First time learning about the system, preparing demos, onboarding new team members

---

### [TECHNICAL_DEEP_DIVE.md](./TECHNICAL_DEEP_DIVE.md) - Implementation Details

**Best for:** Understanding how things actually work

**Contents:**
- 3-tier cache implementation (what each tier stores, lookup flow)
- Dense embeddings (BGE model, why it was chosen)
- Sparse embeddings (custom BM25 implementation)
- How domain terms like "prefab", "wm", "rn" are handled
- Reciprocal Rank Fusion (RRF) formula and examples
- Bi-encoder vs Cross-encoder detailed comparison
- Qdrant hybrid search mechanics
- MCP session management and retry logic
- Performance characteristics and cost breakdown

**When to read:** Debugging issues, optimizing performance, understanding design decisions, making architectural changes

---

### [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Fast Lookups

**Best for:** Quick answers during development or demos

**Contents:**
- System architecture one-liner
- Layer summary table
- Cache tiers comparison
- RRF formula
- Bi-encoder vs cross-encoder comparison
- Configuration reference
- Performance numbers
- API endpoints
- Common operations
- Troubleshooting guide
- File structure
- Glossary

**When to read:** Need a quick reminder, looking up config values, troubleshooting, answering specific questions

---

## 🎯 Reading Path by Role

### New Developer
1. [ARCHITECTURE.md](./ARCHITECTURE.md) - Understand the big picture
2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Bookmark for later
3. [TECHNICAL_DEEP_DIVE.md](./TECHNICAL_DEEP_DIVE.md) - Read sections as needed

### Demo Preparation
1. [ARCHITECTURE.md](./ARCHITECTURE.md) - Review the flow and examples
2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Have open for Q&A

### Debugging/Optimization
1. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Check troubleshooting section
2. [TECHNICAL_DEEP_DIVE.md](./TECHNICAL_DEEP_DIVE.md) - Deep dive into relevant layer

### Architecture Review
1. [TECHNICAL_DEEP_DIVE.md](./TECHNICAL_DEEP_DIVE.md) - Understand current design
2. [ARCHITECTURE.md](./ARCHITECTURE.md) - See high-level trade-offs

---

## 🔑 Key Concepts Across Docs

### Three-Tier Caching

| Document | Coverage |
|----------|----------|
| ARCHITECTURE.md | High-level explanation with example flow |
| TECHNICAL_DEEP_DIVE.md | Detailed storage format, lookup algorithm, why Tier 3 exists |
| QUICK_REFERENCE.md | Quick comparison table |

### Hybrid Search (Dense + Sparse)

| Document | Coverage |
|----------|----------|
| ARCHITECTURE.md | Why we need both types |
| TECHNICAL_DEEP_DIVE.md | Implementation details, BGE vs BM25, how domain terms work |
| QUICK_REFERENCE.md | Quick comparison and package info |

### RRF (Reciprocal Rank Fusion)

| Document | Coverage |
|----------|----------|
| ARCHITECTURE.md | Brief mention |
| TECHNICAL_DEEP_DIVE.md | Formula, examples, why it works |
| QUICK_REFERENCE.md | Formula quick reference |

### Reranking (Bi-encoder vs Cross-encoder)

| Document | Coverage |
|----------|----------|
| ARCHITECTURE.md | High-level difference with example |
| TECHNICAL_DEEP_DIVE.md | Detailed comparison, encoding patterns, when to use each |
| QUICK_REFERENCE.md | Side-by-side comparison table |

### MCP Integration

| Document | Coverage |
|----------|----------|
| ARCHITECTURE.md | What MCP is, why we use it, connection lifecycle |
| TECHNICAL_DEEP_DIVE.md | Session management, retry logic, graceful degradation |
| QUICK_REFERENCE.md | Configuration and troubleshooting |

---

## 📊 Quick Stats

**System Characteristics:**
- **Layers:** 5 (Cache → Embeddings → Retrieval → Reranking → Generation)
- **Latency:** ~5s (cold), ~5ms (cached)
- **Cost:** ~$0.0025 per query (with 50% cache hit rate)
- **Accuracy:** Hybrid search + cross-encoder reranking
- **Reliability:** Parallel execution, graceful degradation

**Technology Stack:**
- **Cache:** Redis (3 tiers)
- **Embeddings:** BGE (dense) + Custom BM25 (sparse)
- **Vector DB:** Qdrant Cloud
- **Video Search:** Academy MCP
- **Reranker:** Jina AI API
- **LLM:** Claude Sonnet 4.5
- **API:** FastAPI

---

## 🛠️ Quick Links

**Setup:**
- Main README: [../README.md](../README.md)
- Environment setup: [../.env.example](../.env.example)

**Code:**
- Pipeline: [../src/core/pipeline.py](../src/core/pipeline.py)
- Cache: [../src/core/cache.py](../src/core/cache.py)
- Embeddings: [../src/core/embedder.py](../src/core/embedder.py)
- Retrieval: [../src/core/retriever.py](../src/core/retriever.py)
- Reranking: [../src/core/reranker.py](../src/core/reranker.py)
- MCP Client: [../src/core/academy.py](../src/core/academy.py)

**Tests:**
- Test suite: [../tests/](../tests/)

---

## ❓ FAQ

### Q: Where do I start?

**A:** Read [ARCHITECTURE.md](./ARCHITECTURE.md) first for the big picture.

### Q: How does caching work exactly?

**A:** See [TECHNICAL_DEEP_DIVE.md - Layer 1](./TECHNICAL_DEEP_DIVE.md#layer-1-three-tier-caching) for detailed explanation with code examples.

### Q: What's the difference between dense and sparse vectors?

**A:** See [TECHNICAL_DEEP_DIVE.md - Layer 2](./TECHNICAL_DEEP_DIVE.md#layer-2-hybrid-embeddings) for a complete comparison.

### Q: Why do we use RRF instead of weighted average?

**A:** See [TECHNICAL_DEEP_DIVE.md - RRF section](./TECHNICAL_DEEP_DIVE.md#reciprocal-rank-fusion-rrf) for the reasoning and formula.

### Q: How do I configure the system?

**A:** See [QUICK_REFERENCE.md - Configuration](./QUICK_REFERENCE.md#configuration) for all environment variables.

### Q: The system is slow, how do I debug?

**A:** See [QUICK_REFERENCE.md - Troubleshooting](./QUICK_REFERENCE.md#troubleshooting) for common issues and solutions.

### Q: What's MCP and why do we use it?

**A:** See [ARCHITECTURE.md - MCP Integration](./ARCHITECTURE.md#-mcp-integration-details) for an overview, or [TECHNICAL_DEEP_DIVE.md - 3B](./TECHNICAL_DEEP_DIVE.md#3b-academy-mcp-video-search) for technical details.

---

## 🔄 Document Updates

All three documents are kept in sync and updated together when architectural changes occur.

**Last synchronized:** 2026-02-02

**Update checklist when making changes:**
- [ ] Update ARCHITECTURE.md (high-level changes)
- [ ] Update TECHNICAL_DEEP_DIVE.md (implementation details)
- [ ] Update QUICK_REFERENCE.md (config/commands/numbers)
- [ ] Update this README if structure changes
- [ ] Update date stamps in all files

---

## 📞 Contact

For questions or suggestions about this documentation:
- Open an issue in the repository
- Contact the team

---

*Documentation index last updated: 2026-02-02*
