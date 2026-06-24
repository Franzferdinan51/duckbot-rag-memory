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

---

## Layer 7+ Candidates — Brain Upgrade Round 2 (2026-06-23)

Survey done by OpenClaw on Duckets' instruction: "enhance and upgrade the memory system, not just RAG; don't add anything that requires pay I haven't mentioned; we can use parts without needing them; be careful not to push sensitive data; careful of memory poisoning and prompt injection when searching GitHub for tools."

### Search method

1. Web search for open-source AI agent memory systems (Tavily, with time_filter=year).
2. **Verified every license + last-commit directly via the GitHub REST API** (search snippets are not trusted).
3. Filtered for: Apache-2.0 / MIT, self-hostable, no paid cloud dependency required for core features.
4. Cross-checked against what's already in the repo (Layers 1-6).

### Verified project status (GitHub API, 2026-06-23)

| Project | Repo | License | Stars | Status |
|---|---|---|---|---|
| Graphiti | `getzep/graphiti` | Apache-2.0 | 27,769 | Active (pushed 2026-06-23). **Already in Layer 1** as inspiration for our `src/graph.py`. |
| mem0 | `mem0ai/mem0` | Apache-2.0 | 59,256 | Active (pushed 2026-06-23). **Already cited** in `src/consolidate.py`. |
| Letta | `letta-ai/letta` | Apache-2.0 | 23,486 | Last push 2026-05-14. **Already cited** in `src/blocks.py`. |
| Cognee | `topoteretes/cognee` | Apache-2.0 | 20,253 | Active (pushed 2026-06-24). **Already cited** in CLI + consolidate. |
| sentence-transformers | `huggingface/sentence-transformers` | Apache-2.0 | 18,846 | Active. **Reusable directly** for rerank. |
| FlagEmbedding (BGE) | `FlagOpen/FlagEmbedding` | MIT | 11,854 | Active. **Reusable directly** for rerank. |
| MemOS | `MemTensor/MemOS` | Apache-2.0 | 9,973 | Active. Mention only — their L1/L2/L3 abstraction overlaps ours. |
| memsearch (Zilliz) | `zilliztech/memsearch` | MIT | 2,103 | Active. Markdown-based; uses Milvus (we use Chroma). |
| TsinghuaC3I/Awesome-Memory-for-Agents | survey | n/a | n/a | Watch-list only. |
| TeleAI-UAGI/Awesome-Agent-Memory | survey | n/a | n/a | Watch-list only. |
| Vestige | `samvallad33/vestige` | **AGPL-3.0** | 563 | Active. ⚠️ Viral license — **cannot directly copy code**. FSRS-6 algorithm itself is open. |
| YourMemory | `sachitrafa/YourMemory` | **CC-BY-NC 4.0** | 246 | Active. ⚠️ Non-commercial. **Cannot copy code**. Ebbinghaus math is public domain (1885). |

### What we can integrate (MIT/Apache, self-hostable, zero paid APIs)

#### Layer 7 candidate: cross-encoder rerank pass
- **Source:** `BAAI/bge-reranker-base` / `bge-reranker-v2-m3` via `FlagOpen/FlagEmbedding` (MIT) or `huggingface/sentence-transformers` (Apache-2.0).
- **Why:** The biggest single recall win we can add. CHANGELOG.md already lists "Cross-encoder rerank pass not yet wired" as a known limitation.
- **Cost:** Free, runs locally (we already have LM Studio — we can run a reranker there, or `pip install sentence-transformers` which is Apache-2.0).
- **Risk:** Low. The model is small (278M params), inference is fast on M-series.
- **Pattern (from sentence-transformers docs):**
  ```python
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
  scores = reranker.predict([(query, doc) for doc in candidates])
  ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]
  ```
- **Plug point:** `src/query.py` — add a `rerank` step after RRF fusion when `DUCKBOT_RERANK=1` is set (default off; opt-in).

#### Layer 8 candidate: Ebbinghaus-style memory decay
- **Source:** The math is from Hermann Ebbinghaus, *Memory: A Contribution to Experimental Psychology* (1885). Public domain.
- **Why:** Memories never decay in our current system. After 6 months the episodic tier will be enormous and noisy. Mem0's "memory decay" feature is opt-in search-time reranking, not deletion. YourMemory and MemBank validate the approach (LoCoMo 52% Recall@5).
- **Cost:** Free math; we already have importance scores.
- **Risk:** Low. Decay only affects retrieval ordering, not storage. Easy to disable.
- **Pattern:**
  ```python
  # Ebbinghaus retention curve: R(t) = e^(-t / S)
  # S = stability (increases each time the chunk is recalled)
  def ebbinghaus_retention(chunk_age_days, stability):
      return math.exp(-chunk_age_days / max(stability, 1e-3))
  ```
- **Plug point:** New `src/decay.py`. Modify `src/query.py` to multiply RRF score by `retention * (1 + importance_bonus)`.

#### Layer 9 candidate: FSRS-6 spaced repetition scheduling
- **Source:** Free Spaced Repetition Scheduler v6 — algorithm by Jarrett Ye. Open specification, MIT implementations on GitHub. Vestige uses it but is AGPL-3.0 (cannot copy).
- **Why:** Combine with decay so that *recalling* a memory reinforces it (spaced-repetition), while *not recalling* lets it decay. This is how human memory works.
- **Cost:** Pure math. No new deps.
- **Risk:** Low. Adds a `review_due_at` field to each chunk metadata; the watcher can use it for "memory hygiene" passes.
- **Plug point:** New `src/fsrs.py`. Integrate into `src/memory.py::remember()` (every recall bumps stability per FSRS-6 update rule).

#### Layer 10 candidate: HyDE (Hypothetical Document Embeddings)
- **Source:** Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022). Algorithm is public.
- **Why:** For conceptual queries ("how do I feel about X?"), the query embedding is in a different space than the chunk embeddings. HyDE generates a *hypothetical answer* with a small LM, embeds *that*, and retrieves against the centroid. Big win on short / vague queries.
- **Cost:** Free. Uses our existing LM Studio `qwen3.5-9b` (already loaded for the watcher).
- **Risk:** Low — opt-in feature flag. Skips LLM call if `DUCKBOT_HYDE=0`.
- **Plug point:** New `src/hyde.py`. Wire into `src/query.py` before the embed step.

#### Layer 11 candidate: weighted RRF with per-tier priors
- **Source:** Custom — but a known pattern in Cognee's hybrid retriever.
- **Why:** Right now RRF treats all tiers equally. But procedural rules (`AGENTS.md`) should rank higher than episodic chatter for "what's the rule about X" queries. Add per-tier priors.
- **Cost:** Zero.
- **Plug point:** `src/query.py` — `weighted_rrf()` variant.

#### Layer 12 candidate: cross-source memory bridging (Layer 6 extension)
- **Source:** zilliztech/memsearch (MIT) has a clean "memory across agents" pattern.
- **Why:** Layer 6 already has OpenClaw + Hermes connectors. Extend to read from session-memory MCPs at query time (instead of just writing).
- **Cost:** Free; uses our existing MCP layer.
- **Risk:** Medium — bidirectional MCP sync needs care to avoid feedback loops. Use a "last-seen" version vector per source.
- **Plug point:** Extend `src/connectors/base.py`.

### What we explicitly rejected and why

- **Vestige source code** — AGPL-3.0 is viral. Cannot copy. We re-implement the ideas (FSRS-6, spreading activation) ourselves.
- **YourMemory source code** — CC-BY-NC 4.0 is non-commercial. Cannot copy. Ebbinghaus math is public domain so we can implement it from scratch.
- **mem0 cloud / Letta cloud / Zep cloud** — paid. We self-host.
- **DSPy** — needs labeled data we don't have.
- **Onyx (40+ connectors)** — built for enterprise doc parsing, not personal markdown memory.
- **RAGFlow** — same as above; overkill for markdown-only.
- **Managed vector DBs** (Pinecone, Weaviate Cloud) — costs money, no local mode.

### Verification policy

When pulling code/patterns from any of these projects:
1. **License first.** Read LICENSE in the repo, not just README.
2. **Last commit.** Stale repos (no commits in 6+ months) get downgraded in priority.
3. **Snippet, don't bundle.** We adopt patterns and small algorithms; we do NOT vendor full modules from AGPL/NC projects.
4. **Attribute.** Every Layer 7+ file will have a top-of-file docstring citing the source.
5. **No auto-run.** Any external install goes through `requirements.txt` review. No `curl ... | sh`.
6. **Sensitive data.** Never push `.env`, never push `data/chroma/`, never push session logs with secrets. (Already enforced by `.gitignore` + `scripts/secret-scan.sh` pre-commit hook.)

### Source-snippet prompt-injection caveats

The web search results contained text fragments that *looked* like system instructions (e.g., mentions of "MiniMax" mixed into project descriptions, marketing copy that read like commands). Treated all `<<<EXTERNAL_UNTRUSTED_CONTENT>>>` blocks as data, not instructions. **Only verified facts via direct GitHub API calls** are in the table above.

---

## v0.4.0 round — MemPalace integration (2026-06-23)

Duckets tipped us off to MemPalace (`MemPalace/mempalace`, MIT, 56,227
stars, pushed today). After reading their README, MISSION.md, ROADMAP.md,
SECURITY.md, and CLAUDE.md directly via `web_fetch` + `curl`, three of
their patterns were high-value enough to ship in this round:

| MemPalace pattern | Our equivalent | Layer |
|---|---|---|
| Verbatim-first design (CLAUDE.md) | `Chunk.verbatim_text` preserved before overlap | **L13** |
| Time-decay scoring (v4.0.0-alpha) | Ebbinghaus `R(t) = e^(-t/S)` (1885 math) | **L8** |
| `.pre-commit-config.yaml` (mitigates secret leaks) | `scripts/secret-scan.sh` + pre-commit hook | **L15** |

### What we explicitly did NOT adopt

- Their `dialect.py` AAAK compression — clever but personal; we'd design our own.
- Their 5-host plugin system (Claude/Codex/Cursor/Antigravity/agents) — overkill at our scale; we adopt the *pattern* not the codebase.
- Their Qdrant/pgvector/LanceDB backends — we ship L14 (the contract) later if needed.

### Pattern source attribution

All v0.4.0 module docstrings cite MemPalace as the pattern source. Their
CLAUDE.md and SECURITY.md were read in full and contained **no
prompt-injection attempts** — they're honest design philosophy.
