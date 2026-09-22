# V13 Hierarchical Industrial Catalogue RAG Implementation Plan

**Goal:** Build V13, a scalable hierarchical RAG for large industrial catalogues with section routing, chunk-count adaptive depth, diversified multi-track retrieval, near-duplicate suppression, optional OCR fallback, and full diagnostics while preserving V12.3.5 behavior.

**Architecture:** Extract every PDF page with layout-aware heading and quality metadata, group contiguous pages into generic catalogue sections, then perform a two-level retrieval: coarse section routing followed by fine hybrid chunk retrieval. Run technical, exact-code, selection-table, and ordering-code tracks, fuse and deduplicate them, rerank a diversified shortlist, and send only the best evidence to the generation LLM. OCR remains an optional page-level fallback and never runs on text-rich pages.

**Tech Stack:** Python 3.11+, PyMuPDF, scikit-learn, NumPy, NVIDIA embeddings/reranking/chat APIs, optional Pillow+pytesseract, pytest.

## Global Constraints

- No product-family-specific business rules in production code.
- Preserve page-level provenance for every retrieved passage.
- Do not replace the complete source specification with a minimal JSON.
- Keep `--diagnostic`, `--details`, reasoning profiles, JSON repair, and V12 output compatibility.
- OCR must be optional and page-scoped; native extraction is always attempted first.
- Default behavior must remain usable for small catalogues without extra configuration.

---

### Task 1: Layout metadata and catalogue sections

**Files:**
- Create: `rag_catalogue/catalogue_map.py`
- Modify: `rag_catalogue/pdf_text.py`
- Test: `tests/test_catalogue_map.py`

**Interfaces:**
- Produces `CatalogueSection`, `build_catalogue_sections(pages)`, page heading/quality metadata, and chunk section metadata.

- [ ] Write failing tests for section grouping, structural heading boundaries, fallback windows, and metadata propagation.
- [ ] Run tests and confirm failure.
- [ ] Implement layout heading extraction, text-quality metrics, section grouping, and enriched chunks.
- [ ] Run tests and confirm pass.

### Task 2: Chunk-count adaptive profile and hierarchical section routing

**Files:**
- Create: `rag_catalogue/hierarchical_retrieval.py`
- Modify: `rag_catalogue/dense_retrieval.py`, `rag_catalogue/pipeline.py`
- Test: `tests/test_hierarchical_retrieval.py`

**Interfaces:**
- Produces `RetrievalProfile`, `adaptive_profile`, `route_sections`, and section-aware candidate filtering.

- [ ] Write failing tests for 100/1,000/5,000 chunk profiles, section routing, neighbor expansion, and global exact fallback.
- [ ] Run tests and confirm failure.
- [ ] Implement section-level lexical+dense indexing and adaptive profile.
- [ ] Run tests and confirm pass.

### Task 3: Multi-track retrieval, deduplication, and diversity

**Files:**
- Create: `rag_catalogue/retrieval_quality.py`
- Modify: `rag_catalogue/pipeline.py`, `rag_catalogue/retrieval.py`
- Test: `tests/test_retrieval_quality.py`

**Interfaces:**
- Produces technical/ordering/selection/exact query tracks, RRF track fusion, near-duplicate suppression, and page/section diversity packing.

- [ ] Write failing tests for track generation, duplicate removal, structural preservation, and page diversity.
- [ ] Run tests and confirm failure.
- [ ] Implement multi-track fusion, token-shingle similarity, and diversified context packing.
- [ ] Run tests and confirm pass.

### Task 4: Optional OCR fallback

**Files:**
- Create: `rag_catalogue/ocr.py`, `requirements-ocr.txt`
- Modify: `rag_catalogue/pdf_text.py`, `rag_catalogue/pipeline.py`, `rag_catalogue_cli.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces `OcrConfig`, OCR engine discovery, low-text page detection, and page-level OCR diagnostics.

- [ ] Write failing tests with an injected fake OCR engine for low-text and text-rich pages.
- [ ] Run tests and confirm failure.
- [ ] Implement optional OCR modes `off`, `auto`, `required` without a hard dependency.
- [ ] Run tests and confirm pass.

### Task 5: Large-catalogue reranking and diagnostics

**Files:**
- Modify: `rag_catalogue/retrieval.py`, `rag_catalogue/retrieval_clients.py`, `rag_catalogue/pipeline.py`, `rag_catalogue_cli.py`
- Test: `tests/test_large_catalogue_pipeline.py`

**Interfaces:**
- Produces bounded batched/tournament reranking and comprehensive retrieval diagnostics.

- [ ] Write failing tests for bounded rerank calls, final shortlist ordering, and diagnostic fields.
- [ ] Run tests and confirm failure.
- [ ] Implement reranking strategy and diagnostics.
- [ ] Run tests and confirm pass.

### Task 6: Documentation, compatibility, and verification

**Files:**
- Modify: `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `VERIFICATION.md`, `.env.example`, `VERSION`
- Test: entire `tests/` suite.

**Interfaces:**
- Preserves existing CLI and adds `--ocr`, `--section-k`, `--profil-catalogue`, and diagnostic fields.

- [ ] Run full test suite.
- [ ] Compile all Python files.
- [ ] Run VendorA smoke tests.
- [ ] Build and inspect ZIP archive.
- [ ] Scan production code for test-specific manufacturer/reference literals.
