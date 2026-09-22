# V12.1 Ranked Candidates and Optional Diagnostic Implementation Plan

**Goal:** Return and validate a ranked list of relevant catalogue references while preserving the compact default output and opt-in diagnostic file.

**Architecture:** Extend the LLM response contract to a bounded candidate collection, validate each candidate independently, expose the best candidate through the legacy `reference` field, and keep diagnostics in a separate file controlled by CLI flags.

**Tech Stack:** Python 3, pytest, PyMuPDF, scikit-learn, requests, NVIDIA APIs.

## Global Constraints

- Maximum five candidates.
- Diagnostic disabled by default.
- Preserve all V12 CLI options.
- No product-family-specific validation rules.
- Every retained candidate must have catalogue evidence.

---

### Task 1: Candidate response contract

**Files:** `rag_catalogue/llm_client.py`, `tests/test_llm_client.py`

- [x] Add failing tests for multiple ranked candidates.
- [x] Extend selection prompt and response parsing.
- [x] Preserve compatibility with legacy single-reference responses.
- [x] Run focused tests.

### Task 2: Independent candidate validation

**Files:** `rag_catalogue/pipeline.py`, `tests/test_pipeline.py`, `tests/test_pipeline_v12.py`

- [x] Add failing tests for mixed valid and invalid candidates.
- [x] Validate candidates independently and remove unproved references.
- [x] Normalize ranks and cap the result at five.
- [x] Run focused tests.

### Task 3: Compact output and diagnostics

**Files:** `rag_catalogue_cli.py`, `tests/test_cli.py`

- [x] Add tests for `multiple_candidates` compact output.
- [x] Preserve `--diagnostic`, `--sortie-diagnostic`, and `--details` behavior.
- [x] Keep the diagnostic outside the main result by default.
- [x] Run focused tests.

### Task 4: Documentation and verification

**Files:** `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `VERIFICATION.md`

- [x] Document ranked candidates and tie handling.
- [x] Document opt-in diagnostic and existing options.
- [x] Run the complete local suite and external catalogue smoke tests.
- [x] Compile all Python modules and create the archive.
