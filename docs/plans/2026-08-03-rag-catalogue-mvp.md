# RAG Catalogue MVP Implementation Plan

**Goal:** Build a standalone Python CLI that reads a product technical PDF, retrieves relevant passages from a catalogue PDF, and returns one exact catalogue reference through a configurable NVIDIA-hosted LLM.

**Architecture:** PyMuPDF extracts page text. A local hybrid TF-IDF retriever combines word and character similarity while retaining catalogue page provenance. An OpenAI-compatible client performs two bounded LLM calls: product-need extraction and final reference selection.

**Tech Stack:** Python 3.10+, PyMuPDF, scikit-learn, openai, python-dotenv, pytest.

## Global Constraints

- Return one reference only.
- Do not calculate compatibility percentages in this MVP.
- Do not require an embeddings API.
- Keep the LLM model configurable through `--modele`.
- Preserve 1-based PDF page numbers in all retrieved evidence.
- Return JSON `null` rather than inventing a reference when context is insufficient.

---

### Task 1: PDF extraction and catalogue chunking

**Files:**
- Create: `rag_catalogue/pdf_text.py`
- Create: `tests/test_pdf_text.py`

**Interfaces:**
- Produces: `extract_pdf_pages(path: Path) -> list[PdfPage]`
- Produces: `chunk_pages(pages: list[PdfPage], max_words: int, overlap_words: int) -> list[CatalogueChunk]`

- [x] Write tests that generate a two-page PDF and assert 1-based pages and overlapping chunks.
- [x] Run the tests and verify they fail because the module is missing.
- [x] Implement immutable page and chunk dataclasses plus extraction and chunking.
- [x] Run the tests and verify they pass.

### Task 2: Hybrid lexical retrieval

**Files:**
- Create: `rag_catalogue/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `CatalogueChunk`
- Produces: `HybridRetriever(chunks).search(query: str, top_k: int) -> list[RankedChunk]`

- [x] Write tests proving exact industrial codes and nearby technical wording rank the correct chunk first.
- [x] Run the tests and verify failure.
- [x] Implement normalized word TF-IDF, character TF-IDF, and a small exact-token bonus.
- [x] Run the tests and verify success.

### Task 3: LLM JSON boundary

**Files:**
- Create: `rag_catalogue/llm_client.py`
- Create: `tests/test_llm_client.py`

**Interfaces:**
- Produces: `extract_json_object(text: str) -> dict`
- Produces: `NvidiaChatClient.extract_need(product_text: str) -> dict`
- Produces: `NvidiaChatClient.select_reference(need: dict, chunks: list[RankedChunk]) -> dict`

- [x] Write tests for plain JSON, fenced JSON, malformed JSON, and a fake OpenAI-compatible response.
- [x] Run the tests and verify failure.
- [x] Implement robust JSON extraction and the two prompts with temperature zero.
- [x] Run the tests and verify success.

### Task 4: Pipeline and CLI

**Files:**
- Create: `rag_catalogue/pipeline.py`
- Create: `rag_catalogue/__init__.py`
- Create: `rag_catalogue_cli.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: product PDF, catalogue PDF, LLM client.
- Produces: `run_search(product_pdf, catalogue_pdf, llm, top_k=8) -> dict`
- Produces CLI subcommands `chercher` and `retrouver`.

- [x] Write a pipeline test using a fake LLM and generated PDFs.
- [x] Run the test and verify failure.
- [x] Implement the orchestration and CLI validation/output.
- [x] Run all tests and verify success.

### Task 5: Documentation and VendorA retrieval smoke test

**Files:**
- Create: `README.md`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `tests/test_vendora_smoke.py`

**Interfaces:**
- Documents installation, environment, search command, diagnostic command, and model switching.

- [x] Write a smoke test that searches the uploaded VendorA catalogue for `REF-DEMO-03 2P courbe C 16 A 10 kA ordering information` and expects pages 41 or 42 in the top results.
- [x] Run the smoke test and verify failure before the remaining implementation is complete.
- [x] Add documentation and dependency declarations.
- [x] Run the full test suite and the diagnostic CLI against the real catalogue.
