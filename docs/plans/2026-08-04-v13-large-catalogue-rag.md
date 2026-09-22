# V13 Large Catalogue RAG Implementation Plan

**Goal:** Build a domain-neutral V13 RAG pipeline that stays accurate on large, heterogeneous industrial catalogues through hierarchy-aware retrieval, chunk-count adaptive depth, structural lanes, near-duplicate suppression, optional OCR fallback, and richer diagnostics.

**Architecture:** Preserve V12.3.5’s lexical+dense+rerank+generation pipeline, then add a catalogue map that assigns chunks to sections and kinds. Retrieval runs global, section-focused, and structural lanes, fuses them, removes near duplicates, diversifies final context, and only then reranks. PDF inspection detects low-text image pages; OCR remains optional and bounded.

**Tech Stack:** Python 3.10+, PyMuPDF, NumPy, scikit-learn, requests, pytest, NVIDIA hosted embedding/reranking/chat endpoints.

## Global Constraints

- No product-family-specific rules in production code.
- Existing CLI remains backward compatible.
- Default behavior must work without OCR or external OCR installation.
- Existing V12 tests must remain green.
- Diagnostics remain optional and are written separately.
- Cache keys must invalidate when extraction or chunk metadata changes.

---

### Task 1: Catalogue Map and PDF Inspection

**Files:**
- Create: `rag_catalogue/catalogue_map.py`
- Modify: `rag_catalogue/pdf_text.py`
- Test: `tests/test_catalogue_map.py`
- Test: `tests/test_pdf_inspection.py`

**Interfaces:**
- Produces `CatalogueSection`, `CatalogueMap`, `build_catalogue_map(...)`.
- Produces `PdfInspection`, `inspect_pdf(...)`, and optional OCR-aware page extraction.

- [ ] Write failing tests for TOC sections, heuristic fallback, chunk metadata, low-text page detection, and bounded OCR fallback.
- [ ] Run tests and verify expected failures.
- [ ] Implement minimal catalogue map and PDF inspection.
- [ ] Run tests and full suite.

### Task 2: Hierarchical and Multi-Lane Retrieval

**Files:**
- Modify: `rag_catalogue/retrieval.py`
- Modify: `rag_catalogue/dense_retrieval.py`
- Modify: `rag_catalogue/pipeline.py`
- Test: `tests/test_hierarchical_retrieval.py`

**Interfaces:**
- Produces chunk-count adaptive retrieval profile.
- Produces section aggregation, weighted multi-ranking fusion, structural lane, and section-focused lane.

- [ ] Write failing tests for adaptive depth by chunks, section selection, structural lane recovery, and weighted rank fusion.
- [ ] Run tests and verify expected failures.
- [ ] Implement minimal hierarchy-aware retrieval.
- [ ] Run tests and full suite.

### Task 3: Context Quality Controls

**Files:**
- Create: `rag_catalogue/context_quality.py`
- Modify: `rag_catalogue/pipeline.py`
- Test: `tests/test_context_quality.py`

**Interfaces:**
- Produces `deduplicate_ranked_chunks(...)` and `diversify_ranked_chunks(...)`.

- [ ] Write failing tests preserving distinct variant columns while dropping duplicate prose.
- [ ] Run tests and verify expected failures.
- [ ] Implement deduplication and diversity packing.
- [ ] Run tests and full suite.

### Task 4: CLI, Pre-indexing, and Diagnostics

**Files:**
- Modify: `rag_catalogue_cli.py`
- Modify: `rag_catalogue/pipeline.py`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_cli_v13.py`

**Interfaces:**
- Adds `inspecter` and `indexer` commands.
- Adds `--ocr`, `--ocr-lang`, `--ocr-max-pages`, `--sans-hierarchie`, and catalogue profile controls.

- [ ] Write failing CLI tests.
- [ ] Run tests and verify expected failures.
- [ ] Implement CLI options and diagnostic fields.
- [ ] Run tests and full suite.

### Task 5: Distribution Verification

**Files:**
- Modify: `VERIFICATION.md`
- Modify: `VERSION`

- [ ] Run complete test suite.
- [ ] Run Python compilation.
- [ ] Run offline VendorA retrieval smoke tests.
- [ ] Scan production code for hard-coded manufacturer/reference fixtures.
- [ ] Create and inspect ZIP archive.
