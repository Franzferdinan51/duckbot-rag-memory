# Open-Source Research Log

Every project we cribbed from, why we cribbed it, and what we stole vs rejected.

## Tier-1 frameworks (full review)

### LangChain (langchain-ai/langchain)
- **Stars:** ~140k, ~3,900 contributors
- **Repo:** https://github.com/langchain-ai/langchain
- **What we liked:** Comprehensive ecosystem, LangGraph stateful workflows, LangSmith observability.
- **What we rejected:** Massive surface area. Adds 5+ dependencies. Most "chains" are 20-line Python scripts in our use case. Would slow cron startup by 3-5s.
- **Verdict:** Not using. We cite their `RecursiveCharacterTextSplitter` pattern instead.

### LlamaIndex (run-llama/llama_index)
- **Stars:** ~50k, ~1,880 contributors
- **Repo:** https://github.com/run-llama/llama_index
- **What we liked:** Strong connectors, chunking strategies, query engines.
- **What we rejected:** Overhead again. Their `VectorStoreIndex` is good but we're using ChromaDB directly — Chroma has 80% of the surface we need.
- **Verdict:** Not using. We cite their `SentenceSplitter` separator order.

### Haystack (deepset-ai/haystack)
- **Stars:** ~25k, ~387 contributors
- **Repo:** https://github.com/deepset-ai/haystack
- **What we liked:** Modular DAG pipelines, evaluation tooling, enterprise features.
- **What we rejected:** Python-first but pulls in many transitive deps. Their component model is heavier than we need.
- **Verdict:** Not using. Borrowed the DAG concept for our `consolidate.py` pipeline.

## Tier-2 memory systems (full review)

### mem0 (mem0ai/mem0)
- **Stars:** tens of thousands
- **Repo:** https://github.com/mem0ai/mem0
- **What we liked:** Hybrid vector + KV store with conflict resolution. Hierarchical memory (user/agent/session). `add`/`search`/`update`/`delete` primitives map cleanly to ours.
- **What we stole:**
  - The **conflict resolution** pattern: when adding a memory that contradicts an existing one, update the existing entry instead of duplicating.
  - The **extraction-first** approach: use LLM to pull facts from episodic logs, not raw chunking.
- **What we rejected:** Their cloud platform. Self-hosted only.
- **Verdict:** Direct inspiration for `src/consolidate.py`. Their LoCoMo benchmark informs our `benchmarks/golden.jsonl`.

### Letta / MemGPT (letta-ai/letta)
- **Stars:** thousands (rapidly growing)
- **Repo:** https://github.com/letta-ai/letta
- **What we liked:** OS-inspired tiered/block-based memory. Core memory blocks always in prompt; recall/archival in Postgres/SQLite. Agents actively read/write/manage memory.
- **What we stole:** The **tier separation** principle. We have 4 tiers (working/episodic/semantic/procedural), not their 3 (core/recall/archival), because we add `procedural` for SOUL.md-style rules.
- **What we rejected:** Stateful runtime / perpetual threads / git-based context versioning. We're not building an agent runtime — DuckBot already has OpenClaw.
- **Verdict:** Architecture inspiration only. Cite them in `tier.py` docstring.

### Cognee (topoteretes/cognee)
- **Stars:** ~19k
- **Repo:** https://github.com/topoteretes/cognee
- **What we liked:** Graph-centric semantic memory. Triplet extraction (subject-predicate-object) on top of vector + relational stores. Operations: `remember`, `recall`, `improve`, `forget`.
- **What we stole:** The **operations vocabulary**. Our CLI uses `remember` (add), `recall` (query), `improve` (consolidate), `forget` (delete).
- **What we rejected:** Their Neo4j dependency. We use ChromaDB only.
- **Verdict:** Operations vocabulary only.

### MemGPT (academic precursor)
- **Status:** Historical; practical impl is Letta.
- **What we liked:** The "RAM vs disk" paging metaphor for context limits.
- **Verdict:** Background reading only. Real impl is in Letta now.

## Tier-3 vector stores (full review)

### ChromaDB
- **What we liked:** Embedded mode (no separate server). Simple Python API. Local-first friendly.
- **Where we use it:** PRIMARY vector store. Embedded, persistent at `data/chroma/`.
- **Why not Qdrant/Weaviate:** Both need a separate server process. Overkill for 50k vectors.

### pgvector
- **What we liked:** If you already use Postgres. ACID + joins.
- **Why not:** We don't run Postgres locally. DuckBot uses SQLite for everything else. ChromaDB fits the same "no server" model.

### Qdrant / Weaviate
- **What we liked:** Performance + hybrid search (dense + sparse).
- **Why not:** Server processes. We can swap ChromaDB for Qdrant later if scale demands.

## Hermes Agent specifics (NousResearch/hermes-agent)
- **Repo:** https://github.com/NousResearch/hermes-agent
- **What they do:** FTS5 session search with LLM summarization for cross-session recall. Agent-curated memory with periodic nudges. Autonomous skill creation.
- **What we stole:**
  - The **FTS5 + LLM** hybrid idea — but we use Chroma's BM25 + vector instead of FTS5.
  - The **periodic nudge** pattern — our cron runs every 90 min during low-activity hours.
- **What we rejected:** Their built-in skill creation loop. We already have `skill_workshop` in OpenClaw.
- **Verdict:** Closest peer. Their design choices validate ours.

## Embedding models

| Model | Provider | Dim | Cost/1M tokens | Quality | Used |
|-------|----------|-----|----------------|---------|------|
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 | High | **PRIMARY** |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 | Highest | optional upgrade |
| `bge-small-en-v1.5` | BAAI | 384 | free (local) | Good | future local mode |
| `nomic-embed-text-v1.5` | Nomic | 768 | free | Good | future local mode |

**Choice:** `text-embedding-3-small` for now. Switch to local (`bge-small`) only if we hit API cost concerns.

## Chunking strategies

| Strategy | Source | When to use |
|----------|--------|-------------|
| Fixed-size | LangChain docs | never alone |
| Recursive (512 tok, 15% OL) | LangChain `RecursiveCharacterTextSplitter` | **PRIMARY** |
| Semantic | LlamaIndex, Chroma | complex narrative docs (not for us) |
| Markdown-aware | Custom | **OUR twist**: respect ## / ### / code blocks |
| Late chunking | Recent arXiv | future improvement |

**2026 consensus** (per Firecrawl / ragaboutit): recursive chunking 400-512 tokens with 10-20% overlap is the workhorse. Semantic chunking costs 2-5x more and rarely wins.

## Hybrid retrieval

Pattern from Cognee + LangChain + Haystack:
1. Vector search on dense embeddings
2. BM25 / keyword search on chunk text
3. Reciprocal Rank Fusion (RRF) to combine
4. Optional cross-encoder rerank (Cohere, Jina, or local `bge-reranker`)
5. Top-K

We skip reranker for v0.1 — RRF alone is good enough at our scale.

## Eval methodology

Following the **LoCoMo** benchmark pattern (mem0 uses it):
- Hand-curated golden queries with known-correct chunks
- Metrics: recall@5, recall@10, MRR, latency p50/p95
- Run weekly via cron, alert on regression

Our `benchmarks/golden.jsonl` is 30+ queries drawn from real DuckBot sessions (e.g., "When did we install cua-driver?", "What was Duckets' rule about local models?").

## What we DIDN'T take (and why)

- **DSPy** (prompt optimization): Cool but requires labeled training data we don't have.
- **RAGFlow** (document-centric): Built for enterprise doc parsing (PDFs, OCR). Our source is markdown — overkill.
- **AutoGen / CrewAI** (agent orchestration): Not building an agent framework, just memory.
- **Pinecone** (managed vector DB): Costs money, no local mode.