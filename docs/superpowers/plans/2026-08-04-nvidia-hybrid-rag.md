# NVIDIA Hybrid RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the catalogue tool from lexical retrieval to a hybrid NVIDIA RAG using `nvidia/nemotron-3-embed-1b`, `nvidia/llama-nemotron-rerank-1b-v2`, and `nvidia/nemotron-3-super-120b-a12b`.

**Architecture:** Keep the existing PDF/table extraction and lexical retrieval. Add dense catalogue embeddings with a local cache, fuse dense and lexical candidates, rerank the fused pool with NVIDIA Ranking NIM, and send the best passages to the generation LLM. Keep the system domain-neutral and expose each model independently in the CLI.

**Tech Stack:** Python 3.10+, PyMuPDF, scikit-learn, NumPy, Requests, NVIDIA NIM APIs, pytest.

## Global Constraints

- No manufacturer, product-family, or expected reference may be hard-coded in production modules.
- The embedding endpoint is `/v1/embeddings`; passage and query input types must be distinct.
- The reranking endpoint is `/v1/ranking` and consumes one query plus candidate passages.
- The same NVIDIA API key may be used for embedding, reranking, and generation.
- The tool must retain lexical/exact retrieval alongside dense retrieval.
- Cached vectors must be invalidated when the catalogue, chunking parameters, or embedding model changes.
- The CLI must allow disabling embedding or reranking for diagnosis.

---

### Task 1: NVIDIA retrieval API clients

**Files:**
- Create: `rag_catalogue/nvidia_retrieval.py`
- Test: `tests/test_nvidia_retrieval.py`

**Interfaces:**
- Produces: `NvidiaEmbeddingClient.embed(texts, input_type) -> numpy.ndarray`
- Produces: `NvidiaRerankClient.rerank(query, passages) -> list[RerankScore]`

- [ ] Write failing HTTP payload and response-normalization tests.
- [ ] Run the tests and confirm they fail because the module is absent.
- [ ] Implement the minimal clients with clear NVIDIA error messages.
- [ ] Run the tests and confirm they pass.

### Task 2: Dense index and persistent cache

**Files:**
- Create: `rag_catalogue/dense_retrieval.py`
- Test: `tests/test_dense_retrieval.py`

**Interfaces:**
- Consumes: `NvidiaEmbeddingClient.embed(...)`
- Produces: `DenseRetriever.search(query, top_k) -> list[DenseHit]`

- [ ] Write failing cosine-ranking and cache-reuse tests.
- [ ] Verify the tests fail for missing implementation.
- [ ] Implement normalized cosine retrieval and an `.npz` cache keyed by chunk content and model.
- [ ] Run the tests and confirm they pass.

### Task 3: Hybrid fusion and reranking

**Files:**
- Modify: `rag_catalogue/retrieval.py`
- Modify: `rag_catalogue/pipeline.py`
- Test: `tests/test_hybrid_rag.py`

**Interfaces:**
- Consumes: lexical `HybridRetriever`, optional `DenseRetriever`, optional `NvidiaRerankClient`
- Produces: `RankedChunk` values with lexical, dense, fusion, and rerank scores

- [ ] Write a failing test where dense retrieval rescues a semantically relevant chunk missed by lexical retrieval.
- [ ] Write a failing test where the reranker changes the final order.
- [ ] Implement rank fusion and post-structuring reranking.
- [ ] Run the focused tests and existing retrieval/pipeline tests.

### Task 4: Pipeline and CLI wiring

**Files:**
- Modify: `rag_catalogue/pipeline.py`
- Modify: `rag_catalogue/llm_client.py`
- Modify: `rag_catalogue_cli.py`
- Modify: `.env.example`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces CLI options `--modele-embedding`, `--modele-reranking`, `--modele`, `--sans-embedding`, `--sans-reranking`, and `--cache-dir`.

- [ ] Write failing CLI wiring tests.
- [ ] Implement independent clients and pass them to `run_search`.
- [ ] Normalize harmless page-number shapes returned by generation models.
- [ ] Add diagnostic model and score fields.
- [ ] Run focused tests.

### Task 5: Documentation and packaged verification

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CHANGELOG.md`
- Modify: `VERSION`

- [ ] Document the exact four-stage model configuration and PowerShell command.
- [ ] Document cache behavior and diagnostic modes.
- [ ] Run compilation, genericity scan, focused test suite, and catalogue retrieval smoke test.
- [ ] Build the distributable ZIP and verify its contents.
