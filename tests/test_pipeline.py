from pathlib import Path

import fitz

from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.pipeline import build_retrieval_queries, retrieve_catalogue_chunks, run_search, validate_catalogue_result
from rag_catalogue.retrieval import RankedChunk


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox((50, 50, 550, 780), text, fontsize=10)
    doc.save(path)
    doc.close()


class _FakeLLM:
    model = "nvidia/fake-model"

    def __init__(self) -> None:
        self.received_product_text = ""
        self.received_chunks = []

    def extract_need(self, product_text: str) -> dict:
        self.received_product_text = product_text
        return {
            "product_type": "deep groove ball bearing",
            "source_reference": "SOURCE-20X52",
            "attributes": [
                {"name": "bore", "value": "20", "unit": "mm"},
                {"name": "outside diameter", "value": "52", "unit": "mm"},
                {"name": "width", "value": "15", "unit": "mm"},
                {"name": "seal", "value": "double rubber", "unit": None},
            ],
            "search_queries": [
                "deep groove ball bearing bore 20 mm outside diameter 52 mm width 15 mm double rubber seals"
            ],
        }

    def select_reference(self, need: dict, chunks: list) -> dict:
        self.received_chunks = chunks
        return {
            "reference": "6304-2RS",
            "catalogue_pages": [2],
            "evidence": ["20 x 52 x 15 mm and double rubber seals"],
            "explanation": "Reference stated on the matching catalogue line.",
        }


def test_run_search_returns_one_direct_reference_without_target_code_in_query(tmp_path: Path) -> None:
    product = tmp_path / "product.pdf"
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(product, ["Bearing bore 20 mm, outside diameter 52 mm, width 15 mm, sealed both sides"])
    _write_pdf(
        catalogue,
        [
            "Reference 6204-Z. Deep groove ball bearing. Bore 20 mm. Outside diameter 47 mm. Width 14 mm. One shield.",
            "Reference 6304-2RS. Deep groove ball bearing. Bore 20 mm. Outside diameter 52 mm. Width 15 mm. Double rubber seals.",
        ],
    )
    llm = _FakeLLM()

    result = run_search(product, catalogue, llm, top_k=1)

    assert result["reference"] == "6304-2RS"
    assert result["model"] == "nvidia/fake-model"
    assert "PAGE 1" in llm.received_product_text
    assert llm.received_chunks[0].chunk.page_number == 2
    assert "6304-2RS" not in llm.extract_need("")["search_queries"][0]


def test_multiple_queries_retrieve_complementary_catalogue_evidence(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_multi.pdf"
    _write_pdf(
        catalogue,
        [
            "Accessory page unrelated to the requested bearing.",
            "Selection table: deep groove ball bearing, bore 20 mm, outside diameter 52 mm, width 15 mm, double rubber seals.",
            "Ordering table: reference 6304-2RS for dimensions 20 x 52 x 15 mm, suffix 2RS means two rubber seals.",
            "Reference 6304-Z for dimensions 20 x 52 x 15 mm, suffix Z means one metal shield.",
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        [
            "deep groove ball bearing bore 20 mm outside diameter 52 mm width 15 mm double rubber seals",
            "bearing 20 x 52 x 15 two rubber seals reference",
        ],
        top_k=3,
    )
    pages = {item.chunk.page_number for item in results}

    assert 2 in pages
    assert 3 in pages


def test_run_search_accepts_txt_product_file(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text(
        "Disjoncteur miniature, 2 poles, 6 A, courbe C, 10 kA, 230/400 V AC.",
        encoding="utf-8",
    )
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "Reference CANDIDATE-6A. Miniature circuit breaker 2 poles 6 A C curve 10 kA 240/415 V.",
        ],
    )
    class _TxtLLM(_FakeLLM):
        def extract_need(self, product_text: str) -> dict:
            self.received_product_text = product_text
            return {
                "product_type": "miniature circuit breaker",
                "source_reference": None,
                "attributes": [
                    {"name": "number of poles", "value": "2", "unit": None},
                    {"name": "rated current", "value": "6", "unit": "A"},
                    {"name": "curve", "value": "C", "unit": None},
                    {"name": "breaking capacity", "value": "10", "unit": "kA"},
                ],
                "search_queries": [
                    "miniature circuit breaker 2 poles 6 A C curve 10 kA"
                ],
            }

        def select_reference(self, need: dict, chunks: list) -> dict:
            self.received_chunks = chunks
            return {
                "reference": "CANDIDATE-6A",
                "catalogue_pages": [1],
                "evidence": ["Reference CANDIDATE-6A. Miniature circuit breaker 2 poles 6 A C curve 10 kA"],
                "explanation": "The matching catalogue line states the reference.",
            }

    llm = _TxtLLM()

    result = run_search(product, catalogue, llm, top_k=1)

    assert result["reference"] == "CANDIDATE-6A"
    assert "FICHIER TEXTE product.txt" in llm.received_product_text
    assert "2 poles, 6 A" in llm.received_product_text


class _UnsafeLLM:
    model = "nvidia/unsafe-model"

    def __init__(self) -> None:
        self.received_need = None

    def extract_need(self, product_text: str) -> dict:
        return {
            "product_type": "miniature circuit breaker",
            "source_manufacturer": "Schneider Electric",
            "source_family": "Acti9 iC60N",
            "source_reference": "A9F77206",
            "attributes": [
                {"name": "number of poles", "value": "2", "unit": None},
                {"name": "rated current", "value": "6", "unit": "A"},
                {"name": "curve", "value": "C", "unit": None},
                {"name": "breaking capacity", "value": "10", "unit": "kA"},
            ],
            "search_queries": [
                "Schneider Electric Acti9 iC60N A9F77206 miniature circuit breaker 2P 6A C 10kA"
            ],
            "source_excerpt": product_text,
        }

    def select_reference(self, need: dict, chunks: list) -> dict:
        self.received_need = need
        return {
            "reference": "A9F77206",
            "catalogue_pages": [],
            "evidence": ["Référence complète : A9F77206"],
            "explanation": "Copied from the source sheet.",
        }


def test_run_search_defaults_to_technical_normalisation_without_proof_validation(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("Schneider A9F77206, MCB 2P 6A curve C 10 kA.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["Reference CANDIDATE-6A. Miniature circuit breaker."])

    result = run_search(product, catalogue, _UnsafeLLM(), top_k=1)

    assert result["reference"] == "A9F77206"
    assert result["catalogue_pages"] == []
    assert result["validation_errors"] == []


def test_run_search_rejects_source_reference_and_exposes_retrieval_diagnostics(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("Schneider A9F77206, MCB 2P 6A curve C 10 kA.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "Ordering information. Reference CANDIDATE-6A. Miniature circuit breaker, 2 poles, 6 A, C curve, 10 kA.",
        ],
    )
    llm = _UnsafeLLM()

    result = run_search(
        product, catalogue, llm, top_k=1, include_diagnostics=True, validate_result=True
    )

    assert result["reference"] is None
    assert "source_reference_reused" in result["validation_errors"]
    assert result["catalogue_pages"] == []
    assert "source_reference" not in llm.received_need
    assert "source_excerpt" not in llm.received_need
    assert result["diagnostic"]["retrieval"][0]["page"] == 1
    all_queries = " ".join(result["diagnostic"]["queries"]).casefold()
    assert "a9f77206" not in all_queries
    assert "schneider electric" not in all_queries
    assert "acti9 ic60n" not in all_queries


def test_run_search_rejects_reference_without_a_retrieved_catalogue_page(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("Bearing bore 20 mm, OD 52 mm, width 15 mm.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["Reference 6304-2RS. Bore 20 mm, OD 52 mm, width 15 mm."])

    class _WrongPageLLM(_FakeLLM):
        def select_reference(self, need: dict, chunks: list) -> dict:
            return {
                "reference": "6304-2RS",
                "catalogue_pages": [99],
                "evidence": ["Reference 6304-2RS"],
                "explanation": "Wrong page citation.",
            }

    result = run_search(
        product, catalogue, _WrongPageLLM(), top_k=1, validate_result=True
    )

    assert result["reference"] is None
    assert "catalogue_page_not_retrieved" in result["validation_errors"]


def test_run_search_accepts_reference_only_with_retrieved_page_and_evidence(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("Bearing bore 20 mm, OD 52 mm, width 15 mm.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["Reference 6304-2RS. Bore 20 mm, OD 52 mm, width 15 mm."])

    class _ValidLLM(_FakeLLM):
        def select_reference(self, need: dict, chunks: list) -> dict:
            return {
                "reference": "6304-2RS",
                "catalogue_pages": [1],
                "evidence": ["Reference 6304-2RS. Bore 20 mm, OD 52 mm, width 15 mm."],
                "explanation": "Reference is present on the retrieved catalogue page.",
            }

    result = run_search(product, catalogue, _ValidLLM(), top_k=1)

    assert result["reference"] == "6304-2RS"
    assert result["validation_errors"] == []


def test_selection_llm_receives_selectors_and_constraints_but_not_context_attributes(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("MCB 2P 6A C 10kA; depth 44.5 mm.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        ["Reference TARGET-6A. Miniature circuit breaker 2 poles 6 A C curve 10 kA."],
    )

    class _RoleAwareLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.received_need = None

        def extract_need(self, product_text: str) -> dict:
            return {
                "product_type": "miniature circuit breaker",
                "source_reference": "A9F77206",
                "attributes": [
                    {"name": "number of poles", "value": "2", "unit": None, "role": "selector"},
                    {"name": "rated current", "value": "6", "unit": "A", "role": "selector"},
                    {"name": "curve", "value": "C", "unit": None, "role": "selector"},
                    {"name": "breaking capacity", "value": "10", "unit": "kA", "role": "constraint"},
                    {"name": "installation depth", "value": "44.5", "unit": "mm", "role": "context"},
                ],
                "search_queries": ["miniature circuit breaker 2 poles 6 A C curve 10 kA"],
            }

        def select_reference(self, need: dict, chunks: list) -> dict:
            self.received_need = need
            return {
                "reference": "TARGET-6A",
                "catalogue_pages": [1],
                "evidence": ["Reference TARGET-6A. 2 poles, 6 A, C curve, 10 kA."],
                "explanation": "The selector attributes match.",
            }

    llm = _RoleAwareLLM()
    result = run_search(product, catalogue, llm, top_k=1)

    assert result["reference"] == "TARGET-6A"
    names = [item["name"] for item in llm.received_need["attributes"]]
    assert names == ["number of poles", "rated current", "curve", "breaking capacity"]
    assert "installation depth" not in names


def test_retrieval_adds_adjacent_page_for_multi_page_catalogue_tables(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_spread.pdf"
    _write_pdf(
        catalogue,
        [
            "Selection table. Miniature circuit breaker 2 poles 6 A C curve 10 kA.",
            "Ordering continuation. Reference TARGET-6A and complete ordering code.",
            "Unrelated accessories and dimensions.",
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        "miniature circuit breaker 2 poles 6 A C curve 10 kA",
        top_k=1,
    )
    pages = {item.chunk.page_number for item in results}

    assert 1 in pages
    assert 2 in pages


def test_retrieval_keeps_one_continuation_page_for_each_top_region(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_many_sections.pdf"
    pages = []
    for index in range(1, 6):
        pages.append(
            f"Selection section {index}. Industrial actuator torque 50 Nm voltage 24 V stroke 100 mm."
        )
        pages.append(f"Ordering continuation {index}. Reference ACT-{index}-24V.")
    _write_pdf(catalogue, pages)

    results = retrieve_catalogue_chunks(
        catalogue,
        "industrial actuator torque 50 Nm voltage 24 V stroke 100 mm",
        top_k=5,
    )
    retrieved_pages = {item.chunk.page_number for item in results}

    assert {2, 4, 6, 8, 10}.issubset(retrieved_pages)


def test_retrieval_adds_distant_ordering_page_matching_family_codes(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_family_ordering.pdf"
    _write_pdf(
        catalogue,
        [
            "Selection table. Model AXR63P, industrial circuit protector, 2 poles, 6 A, C curve, 10 kA.",
            "Accessories for AXR devices.",
            "Unrelated dimensions.",
            "Ordering Information. AXR 63 P 2P MC S 00 00 C 00006.",
            "Ordering Information. BZT 40 M 1P B 00006.",
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        "industrial circuit protector 2 poles 6 A C curve 10 kA",
        top_k=1,
    )
    pages = {item.chunk.page_number for item in results}

    assert 1 in pages
    assert 4 in pages


def test_run_search_rejects_descriptive_pseudo_reference_not_present_in_catalogue(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("MCB 2P 6A C 10 kA.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "Selection table. Model HGD63N 6 kA. Model HGD63H 10 kA. "
            "Both support 2P, 6 A and C curve."
        ],
    )

    class _PseudoReferenceLLM(_FakeLLM):
        def extract_need(self, product_text: str) -> dict:
            return {
                "product_type": "miniature circuit breaker",
                "source_reference": "SOURCE-6A",
                "attributes": [
                    {"name": "poles", "value": "2P", "unit": None, "role": "selector"},
                    {"name": "current", "value": "6", "unit": "A", "role": "selector"},
                    {"name": "curve", "value": "C", "unit": None, "role": "selector"},
                    {"name": "breaking capacity", "value": "10", "unit": "kA", "role": "selector"},
                ],
                "search_queries": ["miniature circuit breaker 2P 6A C curve 10 kA"],
            }

        def select_reference(self, need: dict, chunks: list) -> dict:
            return {
                "reference": "HGD63N 2P 6A C 400VAC 10kA Icu",
                "catalogue_pages": [1],
                "evidence": ["Model HGD63N, 63 AF, 10 kA"],
                "explanation": "Descriptive pseudo-reference assembled from flattened text.",
            }

    result = run_search(
        product, catalogue, _PseudoReferenceLLM(), top_k=1, validate_result=True
    )

    assert result["reference"] is None
    assert "reference_not_proved" in result["validation_errors"]


def _ranked_chunk(page: int, text: str) -> RankedChunk:
    return RankedChunk(
        chunk=CatalogueChunk(chunk_id=f"p{page}-c1", page_number=page, text=text),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
        code_score=0.0,
    )


def test_validation_rejects_descriptive_phrase_that_is_not_a_catalogue_reference() -> None:
    chunks = [
        _ranked_chunk(
            14,
            "Selection table. Model HGD63N, 63 AF, 6 kA. Model HGD63H, 63 AF, 10 kA. "
            "No. of Poles 2P. Rated Current 6 A. C Curve.",
        )
    ]
    raw = {
        "reference": "HGD63N 2P 6A C 400VAC 10kA Icu",
        "catalogue_pages": [14],
        "evidence": ["Model HGD63N, 63 AF, 10 kA"],
        "explanation": "Combined product description.",
    }

    result = validate_catalogue_result(raw, {"source_reference": "A9F77206"}, chunks)

    assert result["reference"] is None
    assert "reference_not_explicit_or_constructed" in result["validation_errors"]


def test_validation_accepts_constructed_reference_only_when_ordered_parts_are_proved() -> None:
    ordering_text = (
        "MCB Ordering Information. Ordering Guidelines. "
        "Type HGD Miniature circuit breaker. Frame 63 Standard type. "
        "Short-Circuit Breaking Capacity P 10 kA. Number of Poles 2P 2 Pole. "
        "Tripping Characteristic MC C Curve. Mounting S Front connection. "
        "Auxiliary Contact 00 Non-attachment. Shunt Trip 00 Non-attachment. "
        "Frequency C 50/60 Hz. Rated Current 00006 6 A."
    )
    chunks = [_ranked_chunk(42, ordering_text)]
    parts = ["HGD", "63", "P", "2P", "MC", "S", "00", "00", "C", "00006"]
    raw = {
        "reference": "REF-DEMO-032PMCS0000C00006",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": index, "code": code, "page": 42}
            for index, code in enumerate(parts, start=1)
        ],
        "catalogue_pages": [42],
        "evidence": ["Ordering Guidelines", "P 10 kA", "00006 6 A"],
        "explanation": "Reference assembled from the ordering key.",
    }

    result = validate_catalogue_result(raw, {"source_reference": "A9F77206"}, chunks)

    assert result["reference"] == "REF-DEMO-032PMCS0000C00006"
    assert result["validation_errors"] == []


def test_validation_accepts_assembled_alias_and_value_parts() -> None:
    ordering_text = (
        "Ordering Information. ABC product family. 2P two poles. "
        "C curve. 00006 6 A."
    )
    chunks = [_ranked_chunk(7, ordering_text)]
    raw = {
        "reference": "ABC2PC00006",
        "reference_mode": "assembled",
        "reference_parts": [
            {"value": "ABC", "page": 7},
            {"value": "2P", "page": 7},
            {"value": "C", "page": 7},
            {"value": "00006", "page": 7},
        ],
        "catalogue_pages": [7],
        "evidence": ["Ordering Information"],
        "explanation": "Assembled from the visible order key.",
    }

    result = validate_catalogue_result(raw, {"source_reference": None}, chunks)

    assert result["reference"] == "ABC2PC00006"
    assert result["validation_errors"] == []


def test_validation_rejects_constructed_reference_with_unproved_part() -> None:
    chunks = [_ranked_chunk(42, "Ordering Information. A 10 kA. 2P. C curve. 00006 6 A.")]
    raw = {
        "reference": "ABC2PC00006",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": 1, "code": "ABC", "page": 42},
            {"position": 2, "code": "2P", "page": 42},
            {"position": 3, "code": "C", "page": 42},
            {"position": 4, "code": "00006", "page": 42},
        ],
        "catalogue_pages": [42],
        "evidence": ["Ordering Information"],
        "explanation": "ABC is not present.",
    }

    result = validate_catalogue_result(raw, {"source_reference": None}, chunks)

    assert result["reference"] is None
    assert "constructed_part_not_proved" in result["validation_errors"]


def test_repeated_retrieval_reuses_plain_catalogue_index(tmp_path: Path, monkeypatch) -> None:
    import rag_catalogue.pipeline as pipeline_module

    catalogue = tmp_path / "catalogue_cached.pdf"
    _write_pdf(
        catalogue,
        [
            "Selection table. Model AXR63P, 2 poles, 6 A, C curve, 10 kA.",
            "Ordering Information. AXR 63 P 2P C 00006.",
        ],
    )
    original_retriever = pipeline_module.HybridRetriever
    plain_initialisations = 0

    class _CountingRetriever(original_retriever):
        def __init__(self, chunks):
            nonlocal plain_initialisations
            chunk_ids = {chunk.chunk_id for chunk in chunks}
            if chunk_ids == {"p1-c1", "p2-c1"}:
                plain_initialisations += 1
            super().__init__(chunks)

    monkeypatch.setattr(pipeline_module, "HybridRetriever", _CountingRetriever)
    pipeline_module._catalogue_index_cached.cache_clear()

    pipeline_module.retrieve_catalogue_chunks(
        catalogue,
        "industrial circuit protector 2 poles 6 A C curve 10 kA",
        top_k=2,
    )
    pipeline_module.retrieve_catalogue_chunks(
        catalogue,
        "industrial circuit protector 2P 6A C 10kA",
        top_k=2,
    )

    assert plain_initialisations == 1


def test_build_retrieval_queries_keeps_third_international_query() -> None:
    need = {
        "product_type": "disjoncteur modulaire (MCB)",
        "source_reference": "SRC-1",
        "attributes": [],
        "search_queries": [
            "disjoncteur 2P 6A",
            "protection modulaire 10kA",
            "miniature circuit breaker MCB 2 poles 6 A C curve 10 kA",
        ],
    }

    queries = build_retrieval_queries(need)

    assert "miniature circuit breaker MCB 2 poles 6 A C curve 10 kA" in queries


def test_ordering_scope_rank_prefers_base_function_when_composite_function_is_not_requested() -> None:
    from rag_catalogue.pipeline import _ordering_scope_rank

    base = (
        "PAGE CONTEXT: Protector Ordering Information\n"
        "STRUCTURED TABLE CODE MAP\n"
        "ABC: Base circuit protector\n"
    )
    composite = (
        "PAGE CONTEXT: Composite Protector Ordering Information\n"
        "STRUCTURED TABLE CODE MAP\n"
        "XYZ: Base circuit protector with residual monitoring\n"
    )
    queries = ["base circuit protector 2 poles 6 A"]

    assert _ordering_scope_rank(base, queries) < _ordering_scope_rank(composite, queries)


def test_ordering_scope_rank_prefers_composite_function_when_explicitly_requested() -> None:
    from rag_catalogue.pipeline import _ordering_scope_rank

    base = (
        "PAGE CONTEXT: Protector Ordering Information\n"
        "STRUCTURED TABLE CODE MAP\n"
        "ABC: Base circuit protector\n"
    )
    composite = (
        "PAGE CONTEXT: Composite Protector Ordering Information\n"
        "STRUCTURED TABLE CODE MAP\n"
        "XYZ: Base circuit protector with residual monitoring\n"
    )
    queries = ["XYZ base circuit protector with residual monitoring"]

    assert _ordering_scope_rank(composite, queries) < _ordering_scope_rank(base, queries)


def test_validation_rejects_constructed_prefix_when_order_template_requires_more_slots() -> None:
    ordering_text = (
        "PAGE CONTEXT: Ordering Information\n"
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 10\n"
        "EXAMPLE CODES: AXR | 63 | P | 1P | MC | S | 00 | 00 | C | 00001\n"
        "STRUCTURED TABLE CODE MAP\n"
        "AXR: Industrial protector\n63: 63 frame\nP: 10 kA\n"
        "2P: 2 poles\nMC: C curve\nS: Front connection\n"
        "00: Non-attachment\nC: 50/60 Hz\n00006: 6 A"
    )
    chunks = [_ranked_chunk(42, ordering_text)]
    raw = {
        "reference": "AXR63P",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": 1, "code": "AXR", "page": 42},
            {"position": 2, "code": "63", "page": 42},
            {"position": 3, "code": "P", "page": 42},
        ],
        "catalogue_pages": [42],
        "evidence": ["AXR", "63 frame", "P 10 kA"],
        "explanation": "Only the family prefix was assembled.",
    }

    result = validate_catalogue_result(raw, {"source_reference": None}, chunks)

    assert result["reference"] is None
    assert "incomplete_constructed_reference" in result["validation_errors"]


def test_validation_accepts_full_constructed_reference_matching_order_template_slots() -> None:
    ordering_text = (
        "PAGE CONTEXT: Ordering Information\n"
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 4\n"
        "EXAMPLE CODES: AXR | 1P | C | 00001\n"
        "STRUCTURED TABLE CODE MAP\n"
        "AXR: Industrial protector\n2P: 2 poles\nMC: C curve\n00006: 6 A"
    )
    chunks = [_ranked_chunk(7, ordering_text)]
    raw = {
        "reference": "AXR2PMC00006",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": 1, "code": "AXR", "page": 7},
            {"position": 2, "code": "2P", "page": 7},
            {"position": 3, "code": "MC", "page": 7},
            {"position": 4, "code": "00006", "page": 7},
        ],
        "catalogue_pages": [7],
        "evidence": ["Ordering Information"],
        "explanation": "All order-code slots are present.",
    }

    result = validate_catalogue_result(raw, {"source_reference": None}, chunks)

    assert result["reference"] == "AXR2PMC00006"
    assert result["validation_errors"] == []


def test_run_search_repairs_incomplete_constructed_reference_once(tmp_path: Path, monkeypatch) -> None:
    import rag_catalogue.pipeline as pipeline_module

    product = tmp_path / "product.txt"
    product.write_text("Industrial protector, 2 poles, C curve, 6 A.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["placeholder"])
    ordering_text = (
        "PAGE CONTEXT: Ordering Information\n"
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 4\n"
        "EXAMPLE CODES: AXR | 1P | C | 00001\n"
        "STRUCTURED TABLE CODE MAP\n"
        "AXR: Industrial protector\n2P: 2 poles\nMC: C curve\n00006: 6 A"
    )
    chunks = [_ranked_chunk(7, ordering_text)]
    monkeypatch.setattr(
        pipeline_module,
        "retrieve_catalogue_chunks",
        lambda *_args, **_kwargs: chunks,
    )

    class _RepairingLLM:
        model = "nvidia/test"

        def __init__(self) -> None:
            self.repairs = 0

        def extract_need(self, _text: str) -> dict:
            return {
                "product_type": "industrial protector",
                "source_reference": "SOURCE-1",
                "attributes": [
                    {"name": "poles", "value": "2", "unit": None, "role": "selector"},
                    {"name": "curve", "value": "C", "unit": None, "role": "selector"},
                    {"name": "current", "value": "6", "unit": "A", "role": "selector"},
                ],
                "search_queries": ["industrial protector 2 poles C curve 6 A"],
            }

        def select_reference(self, _need: dict, _chunks: list) -> dict:
            return {
                "reference": "AXR",
                "reference_mode": "constructed",
                "reference_parts": [{"position": 1, "code": "AXR", "page": 7}],
                "catalogue_pages": [7],
                "evidence": ["AXR"],
                "explanation": "Incomplete prefix.",
            }

        def repair_reference(
            self,
            _need: dict,
            _chunks: list,
            _previous_result: dict,
            validation_errors: list[str],
        ) -> dict:
            self.repairs += 1
            assert "incomplete_constructed_reference" in validation_errors or "invalid_reference_parts" in validation_errors
            return {
                "reference": "AXR2PMC00006",
                "reference_mode": "constructed",
                "reference_parts": [
                    {"position": 1, "code": "AXR", "page": 7},
                    {"position": 2, "code": "2P", "page": 7},
                    {"position": 3, "code": "MC", "page": 7},
                    {"position": 4, "code": "00006", "page": 7},
                ],
                "catalogue_pages": [7],
                "evidence": ["Ordering Information"],
                "explanation": "Complete order reference.",
            }

    llm = _RepairingLLM()
    result = run_search(product, catalogue, llm, top_k=1, validate_result=True)

    assert result["reference"] == "AXR2PMC00006"
    assert result["validation_errors"] == []
    assert llm.repairs == 1


def test_validation_canonicalizes_constructed_reference_from_proved_parts() -> None:
    ordering_text = (
        "PAGE CONTEXT: Ordering Information\n"
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 4\n"
        "EXAMPLE CODES: AXR | 1P | C | 00001\n"
        "STRUCTURED TABLE CODE MAP\n"
        "AXR: Industrial protector\n2P: 2 poles\nMC: C curve\n00006: 6 A"
    )
    chunks = [_ranked_chunk(7, ordering_text)]
    raw = {
        "reference": "AXR | 2P | MC | 00006",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": 1, "code": "AXR", "page": 7},
            {"position": 2, "code": "2P", "page": 7},
            {"position": 3, "code": "MC", "page": 7},
            {"position": 4, "code": "00006", "page": 7},
        ],
        "catalogue_pages": [7],
        "evidence": ["Ordering Information"],
        "explanation": "All order-code slots are present.",
    }

    result = validate_catalogue_result(raw, {"source_reference": None}, chunks)

    assert result["reference"] == "AXR2PMC00006"
    assert result["validation_errors"] == []


def test_normalise_generation_result_keeps_all_ranked_candidates() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "HGD63H2PMCS0000C00003",
                "variant": "Deluxe",
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [41],
                "evidence": ["Deluxe Type"],
                "reason": "Closest documented variant.",
                "differences": [],
            },
            {
                "rank": 2,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": "Standard",
                "match_level": "alternative",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [42],
                "evidence": ["Standard Type"],
                "reason": "Compatible standard alternative.",
                "differences": ["Standard instead of Deluxe"],
            },
        ],
        "explanation": "Two relevant variants.",
    }

    result = normalise_generation_result(raw)

    assert result["reference"] == "REF-DEMO-032PMCS0000C00003"
    assert len(result["candidates"]) == 2
    assert "version Deluxe" in result["candidates"][1]["reason"]


def test_validate_catalogue_result_collection_keeps_all_proved_candidates() -> None:
    from rag_catalogue.pipeline import validate_catalogue_result_collection

    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p41-c1",
                page_number=41,
                text="Deluxe Type reference HGD63H2PMCS0000C00003",
            ),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        ),
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p42-c1",
                page_number=42,
                text="Standard Type reference REF-DEMO-032PMCS0000C00003",
            ),
            score=0.9,
            word_score=0.9,
            char_score=0.9,
            exact_score=0.9,
        ),
    ]
    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "HGD63H2PMCS0000C00003",
                "variant": "Deluxe",
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [41],
                "evidence": ["Deluxe Type"],
                "reason": "Closest documented variant.",
                "differences": [],
            },
            {
                "rank": 2,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": "Standard",
                "match_level": "alternative",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [42],
                "evidence": ["Standard Type"],
                "reason": "Compatible standard alternative.",
                "differences": ["Standard instead of Deluxe"],
            },
        ],
        "explanation": "Two proved variants.",
    }

    result = validate_catalogue_result_collection(raw, {"source_reference": None}, chunks)

    assert result["reference"] == "REF-DEMO-032PMCS0000C00003"
    assert [candidate["reference"] for candidate in result["candidates"]] == [
        "REF-DEMO-032PMCS0000C00003",
        "HGD63H2PMCS0000C00003",
    ]
    assert result["validation_errors"] == []


def test_normalise_generation_result_accepts_non_string_explanation_and_reason() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "AXR-100",
                "variant": None,
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [7],
                "evidence": ["Reference AXR-100"],
                "reason": {"summary": "Exact documented match."},
                "differences": "No documented difference.",
            }
        ],
        "explanation": ["One candidate was found.", "It is ranked first."],
    }

    result = normalise_generation_result(raw)

    assert result["explanation"] == (
        "Un seul candidat valide a ete conserve : AXR-100. "
        "Exact documented match."
    )
    assert result["candidates"][0]["reason"] == "Exact documented match."
    assert result["candidates"][0]["differences"] == []


def test_run_search_retries_once_when_first_candidate_list_is_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_catalogue.pipeline as pipeline_module

    product = tmp_path / "product.txt"
    product.write_text("Industrial product, model requirement A.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["placeholder"])
    chunks = [_ranked_chunk(7, "Reference AXR-100")]
    monkeypatch.setattr(
        pipeline_module,
        "retrieve_catalogue_chunks",
        lambda *_args, **_kwargs: chunks,
    )

    class _RetryingLLM:
        model = "nvidia/test"

        def __init__(self) -> None:
            self.selection_calls = 0

        def extract_need(self, _text: str) -> dict:
            return {
                "product_type": "industrial product",
                "source_reference": None,
                "attributes": [
                    {
                        "name": "model requirement",
                        "value": "A",
                        "unit": None,
                        "role": "selector",
                    }
                ],
                "search_queries": ["industrial product model requirement A"],
            }

        def select_reference(self, need: dict, _chunks: list) -> dict:
            self.selection_calls += 1
            if self.selection_calls == 1:
                return {
                    "candidates": [],
                    "explanation": "No candidate selected on the first pass.",
                }
            assert need["validation_feedback"] == ["empty_candidate_list"]
            assert need["previous_result"]["candidates"] == []
            return {
                "candidates": [
                    {
                        "rank": 1,
                        "reference": "AXR-100",
                        "variant": None,
                        "match_level": "closest",
                        "reference_mode": "explicit",
                        "reference_parts": [],
                        "catalogue_pages": [7],
                        "evidence": ["Reference AXR-100"],
                        "reason": "Found after re-examining the same catalogue context.",
                        "differences": [],
                    }
                ],
                "explanation": "One candidate was recovered on the automatic retry.",
            }

    llm = _RetryingLLM()
    result = run_search(product, catalogue, llm, top_k=1)

    assert result["reference"] == "AXR-100"
    assert result["candidates"][0]["reference"] == "AXR-100"
    assert llm.selection_calls == 2



def test_structured_slot_code_map_assigns_codes_to_order_slots_without_product_rules() -> None:
    from rag_catalogue.pipeline import _structured_slot_code_map

    template = (
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 10\n"
        "EXAMPLE CODES: HGD | 63 | N | 1P | MC | S | 00 | 00 | C | 00001\n"
        "SLOT FIELDS: 1=Type | 2=Frame | 3=Breaking capacity | 4=Poles | "
        "5=Curve | 6=Mounting | 7=Auxiliary | 8=Trip device | 9=Frequency | 10=Current"
    )
    blocks = [
        "HGD: Miniature circuit breaker",
        "MB: B Curve\nMC: C Curve\nMD: D Curve",
        "C: 50/60 Hz",
        "63: 63 AF\n125: 125 AF",
        "S: Front connection",
        "00001: 1 A\n00003: 3 A\n00016: 16 A",
        "N: 6 kA\nH: 10 kA",
        "00: Non-attachment",
        "1P: 1 Pole\n2P: 2 Pole\n3P: 3 Pole",
    ]

    slot_map = _structured_slot_code_map(template, blocks)

    assert "SLOT 2 ALLOWED CODES: 63 | 125" in slot_map
    assert "SLOT 3 ALLOWED CODES: N | H" in slot_map
    assert "SLOT 4 ALLOWED CODES: 1P | 2P | 3P" in slot_map
    assert "SLOT 10 ALLOWED CODES: 00001 | 00003 | 00016" in slot_map


def test_structured_slot_code_map_resolves_duplicate_example_code_by_claimed_block() -> None:
    from rag_catalogue.pipeline import _structured_slot_code_map

    template = (
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 4\n"
        "EXAMPLE CODES: AXR | M | S | 00001\n"
        "SLOT FIELDS: 1=Type | 2=Breaking capacity | 3=Mounting | 4=Current"
    )
    blocks = [
        "AXR: Industrial protector",
        "E: 3 kA\nS: 4.5 kA\nM: 6 kA\nP: 10 kA",
        "S: Front connection",
        "00001: 1 A\n00006: 6 A",
    ]

    slot_map = _structured_slot_code_map(template, blocks)

    assert "SLOT 2 ALLOWED CODES: E | S | M | P" in slot_map
    assert "SLOT 3 ALLOWED CODES: S" in slot_map


def test_validation_rejects_constructed_code_used_in_the_wrong_slot() -> None:
    ordering_text = (
        "PAGE CONTEXT: Ordering Information\n"
        "STRUCTURED ORDER TEMPLATE\n"
        "SLOT COUNT: 4\n"
        "EXAMPLE CODES: HGD | 63 | N | 00001\n"
        "STRUCTURED SLOT CODE MAP\n"
        "SLOT 1 ALLOWED CODES: HGD\n"
        "SLOT 2 ALLOWED CODES: 63 | 125\n"
        "SLOT 3 ALLOWED CODES: N | H\n"
        "SLOT 4 ALLOWED CODES: 00001 | 00016\n"
        "STRUCTURED TABLE CODE MAP\n"
        "HGD: Miniature circuit breaker\n63: 63 AF\n125: 125 AF\n"
        "N: 6 kA\nH: 10 kA\n00001: 1 A\n00016: 16 A"
    )
    chunks = [_ranked_chunk(41, ordering_text)]
    raw = {
        "reference": "HGDNN00016",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": 1, "code": "HGD", "page": 41},
            {"position": 2, "code": "N", "page": 41},
            {"position": 3, "code": "N", "page": 41},
            {"position": 4, "code": "00016", "page": 41},
        ],
        "catalogue_pages": [41],
        "evidence": ["Ordering Information"],
        "explanation": "The same visible code was mistakenly reused in two slots.",
    }

    result = validate_catalogue_result(raw, {"source_reference": None}, chunks)

    assert result["reference"] is None
    assert "constructed_part_wrong_slot" in result["validation_errors"]


def test_normalise_generation_result_rebuilds_explanation_from_final_candidates() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 2,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": "Standard",
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [42],
                "evidence": ["Standard Type"],
                "reason": "Respecte les exigences confirmees.",
                "differences": [],
            }
        ],
        "explanation": (
            "Deux candidats ont ete trouves. Le Deluxe HGD63H est premier "
            "et le Standard REF-DEMO-03 est second."
        ),
    }

    result = normalise_generation_result(raw)

    assert result["reference"] == "REF-DEMO-032PMCS0000C00003"
    assert result["candidates"][0]["rank"] == 1
    assert result["explanation"] == (
        "Un seul candidat valide a ete conserve : "
        "REF-DEMO-032PMCS0000C00003 (Standard). "
        "Respecte les exigences confirmees."
    )
    assert "HGD63H" not in result["explanation"]


def test_validate_collection_rebuilds_explanation_after_rejecting_candidate() -> None:
    from rag_catalogue.pipeline import validate_catalogue_result_collection

    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p42-c1",
                page_number=42,
                text="Standard Type reference REF-DEMO-032PMCS0000C00003",
            ),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]
    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "HGD63H2PMCS0000C00003",
                "variant": "Deluxe",
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [41],
                "evidence": ["Deluxe Type"],
                "reason": "Deluxe annonce comme premier.",
                "differences": [],
            },
            {
                "rank": 2,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": "Standard",
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [42],
                "evidence": ["Standard Type"],
                "reason": "Seul candidat prouve dans les pages recuperees.",
                "differences": [],
            },
        ],
        "explanation": "Deux candidats valides, Deluxe puis Standard.",
    }

    result = validate_catalogue_result_collection(
        raw,
        {"source_reference": None},
        chunks,
    )

    assert result["reference"] == "REF-DEMO-032PMCS0000C00003"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["rank"] == 1
    assert result["explanation"] == (
        "Un seul candidat valide a ete conserve : "
        "REF-DEMO-032PMCS0000C00003 (Standard). "
        "Seul candidat prouve dans les pages recuperees."
    )
    assert "Deluxe" not in result["explanation"]


def test_normalise_generation_result_rebuilds_multiple_candidate_explanation() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 5,
                "reference": "REF-STANDARD",
                "variant": "Standard",
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [10],
                "evidence": ["REF-STANDARD"],
                "reason": "Meilleure correspondance prouvee.",
                "differences": [],
            },
            {
                "rank": 8,
                "reference": "REF-DELUXE",
                "variant": "Deluxe",
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [11],
                "evidence": ["REF-DELUXE"],
                "reason": "Alternative compatible.",
                "differences": ["Version Deluxe"],
            },
        ],
        "explanation": "Ancien classement incoherent.",
    }

    result = normalise_generation_result(raw)

    assert [item["rank"] for item in result["candidates"]] == [1, 2]
    assert result["reference"] == "REF-STANDARD"
    assert result["explanation"] == (
        "2 candidats valides ont ete conserves. Le candidat classe premier est "
        "REF-STANDARD (Standard). Respecte les critères techniques confirmés. "
        "La version Standard est classée en premier, car aucune exigence ne justifie "
        "une version Deluxe. Alternatives : REF-DELUXE (Deluxe) — Respecte les mêmes "
        "critères techniques confirmés. La version Deluxe est proposée comme alternative, "
        "sans exigence spécifique la rendant prioritaire."
    )


def test_final_single_candidate_removes_stale_rank_and_missing_alternative_text() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": None,
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [42],
                "evidence": ["Standard frame P, 10 kA"],
                "reason": (
                    "Also satisfies all selector attributes using Standard frame (63) "
                    "and capacity code P (10 kA). Technically equivalent to the Deluxe "
                    "variant; ranked second due to no documented preference for Standard "
                    "over Deluxe."
                ),
                "differences": [
                    "Deluxe uses H while Standard uses P."
                ],
            }
        ],
        "explanation": (
            "Two candidates fulfill the requirement. The Deluxe variant is first "
            "and the Standard variant is second."
        ),
    }

    result = normalise_generation_result(raw)
    candidate = result["candidates"][0]

    assert candidate["rank"] == 1
    assert candidate["match_level"] == "equivalent"
    assert candidate["equivalence_status"] == "equivalent_direct"
    assert candidate["reason"] == (
        "Also satisfies all selector attributes using Standard frame (63) "
        "and capacity code P (10 kA)."
    )
    assert candidate["differences"] == []
    assert "ranked second" not in result["explanation"]
    assert "Deluxe" not in result["explanation"]


def test_final_candidates_infer_variants_prefer_standard_and_use_french() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "HGD63H2PMCS0000C00003",
                "variant": None,
                "match_level": "closest",
                "reference_mode": "constructed",
                "reference_parts": [{"position": 1, "code": "HGD63H2PMCS0000C00003", "page": 41}],
                "catalogue_pages": [41],
                "evidence": ["Deluxe Type ordering template"],
                "reason": "Matches all client selectors using the Deluxe Type ordering template.",
                "differences": [],
            },
            {
                "rank": 2,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": None,
                "match_level": "equivalent",
                "reference_mode": "constructed",
                "reference_parts": [{"position": 1, "code": "REF-DEMO-032PMCS0000C00003", "page": 42}],
                "catalogue_pages": [42],
                "evidence": ["Standard Type ordering template"],
                "reason": "Matches all client selectors using the Standard Type ordering template.",
                "differences": ["Deluxe versus Standard"],
            },
        ],
        "explanation": "Two candidates satisfy the selectors.",
    }

    result = normalise_generation_result(raw)

    assert result["reference"] == "REF-DEMO-032PMCS0000C00003"
    assert [item["variant"] for item in result["candidates"]] == ["Standard", "Deluxe"]
    assert [item["rank"] for item in result["candidates"]] == [1, 2]
    assert "aucune exigence" in result["candidates"][0]["reason"].casefold()
    assert "alternative" in result["candidates"][1]["reason"].casefold()
    assert "Two candidates" not in result["explanation"]
    assert "Matches all" not in result["explanation"]


def test_deluxe_remains_first_when_explicitly_required() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "DELUXE-1",
                "variant": "Deluxe",
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [1],
                "evidence": ["Deluxe version explicitly required by the client"],
                "reason": "La version Deluxe est explicitement exigee par le client.",
                "differences": [],
            },
            {
                "rank": 2,
                "reference": "STANDARD-1",
                "variant": "Standard",
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [2],
                "evidence": ["Standard Type"],
                "reason": "Version Standard.",
                "differences": [],
            },
        ],
        "explanation": "Deux candidats.",
    }

    result = normalise_generation_result(raw)

    assert result["reference"] == "DELUXE-1"
    assert result["candidates"][0]["variant"] == "Deluxe"


def test_ordering_expansion_prefers_query_product_identity_over_linked_neighbor_family() -> None:
    from rag_catalogue.pipeline import _ordering_expansions

    core = [
        _ranked_chunk(
            46,
            "Selection Table HRC. Model HRC63. 2P 3 A C curve 10 kA.",
        )
    ]
    chunks = [
        core[0].chunk,
        CatalogueChunk(
            chunk_id="p42-c1",
            page_number=42,
            text=(
                "MCB Ordering Information. Ordering Guidelines. "
                "STRUCTURED TABLE CODE MAP\n"
                "HGD: Miniature circuit breaker\n"
                "2P: 2 Pole\n00003: 3 A\nP: 10 kA"
            ),
            kind="ordering",
        ),
        CatalogueChunk(
            chunk_id="p58-c1",
            page_number=58,
            text=(
                "RCCB Ordering Information. Ordering Guidelines. "
                "STRUCTURED TABLE CODE MAP\n"
                "HRC: Residual current circuit breaker\n"
                "2P: 2 Pole\n00003: 3 A"
            ),
            kind="ordering",
        ),
    ]

    results = _ordering_expansions(
        core,
        chunks,
        {core[0].chunk.chunk_id: core[0]},
        ["miniature circuit breaker MCB 2P 3 A C curve 10 kA"],
        maximum_pages=2,
    )

    assert [item.chunk.page_number for item in results][:2] == [42, 58]


def test_retrieval_expands_sibling_chunks_from_selected_table_page(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_same_page_table.pdf"
    filler = " ".join(f"description{i}" for i in range(90))
    page_text = (
        "FUSIBLES PHOTOVOLTAIQUES gPV HP10M 1000 VDC GAMME DE PRODUIT. "
        "Table de selection pour fusibles cylindriques. "
        + filler
        + " Numero de catalogue REF-DEMO-02. Numero de reference N1018590. "
        "Tension nominale 1000 VDC. Courant nominal 15 A. Pouvoir de coupure maximal 10 kA."
    )
    _write_pdf(catalogue, [page_text])

    results = retrieve_catalogue_chunks(
        catalogue,
        "fusible photovoltaique gPV 15 A 1000 VDC HP10M",
        top_k=1,
        candidate_k=8,
        max_words=45,
        overlap_words=0,
        hierarchy_enabled=False,
        ocr_mode="off",
    )

    assert any("REF-DEMO-02" in item.chunk.text for item in results)
    assert any("GAMME DE PRODUIT" in item.chunk.text for item in results)


def test_retrieval_keeps_both_neighbors_around_selected_catalogue_table(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_three_page_family.pdf"
    _write_pdf(
        catalogue,
        [
            "Fusibles photovoltaïques gPV HP10M 1000 VDC. Reference REF-DEMO-02, 15 A, pouvoir de coupure 10 kA.",
            "FUSIBLES PHOTOVOLTAIQUES. GAMME DE PRODUIT. Table de selection HP10M 1000 VDC 15 A gPV.",
            "Continuation accessoires porte-fusibles HP10M pour 1000 VDC.",
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        "fusible photovoltaique HP10M 15 A 1000 VDC gPV",
        top_k=1,
        candidate_k=8,
        hierarchy_enabled=False,
        ocr_mode="off",
    )
    pages = {item.chunk.page_number for item in results}

    assert 1 in pages
    assert 2 in pages
    assert 3 in pages


def test_retrieval_keeps_previous_page_for_catalogue_table_headers_without_selection_word(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue_table_header_neighbors.pdf"
    _write_pdf(
        catalogue,
        [
            "Reference REF-DEMO-02. Numero de reference N1018590. Courant 15 A.",
            "FUSIBLES PHOTOVOLTAIQUES gPV HP10M 1000 VDC GAMME DE PRODUIT. Numero de catalogue. Numero de reference. Tension nominale. Courant nominal 15 A. Pouvoir de coupure 30 kA. Fusible cylindrique.",
            "Continuation accessoires porte-fusibles HP10M pour applications photovoltaïques 1000 VDC.",
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        "fusible photovoltaique HP10M 15 A 1000 VDC gPV",
        top_k=1,
        candidate_k=8,
        hierarchy_enabled=False,
        ocr_mode="off",
    )
    pages = {item.chunk.page_number for item in results}

    assert 1 in pages
    assert 2 in pages
    assert 3 in pages


def test_structured_table_selection_preserves_best_block_from_priority_neighbor_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_catalogue.pipeline as pipeline_module
    from rag_catalogue.pdf_text import PdfPage

    catalogue = tmp_path / "catalogue_priority_table_page.pdf"
    _write_pdf(catalogue, ["page 1", "page 2", "page 3"])

    plain_pages = [
        PdfPage(1, "Reference REF-DEMO-02. 15 A."),
        PdfPage(
            2,
            "FUSIBLES PHOTOVOLTAIQUES gPV HP10M 1000 VDC GAMME DE PRODUIT. "
            "Numero de catalogue. Numero de reference. Courant nominal 15 A.",
        ),
        PdfPage(3, "Continuation accessoires HP10M 1000 VDC."),
    ]
    target_block = (
        "Numéro de catalogue: Numéro de référence\n"
        "REF-DEMO-02: N1018590\n"
        "Tension nominale: 1000 VDC\nCourant nominal: 15 A"
    )
    distractors = tuple(
        f"Numéro de catalogue: ACCESSORY-{index}\n"
        "Fusible photovoltaïque gPV HP10M 1000 VDC 15 A cylindrique"
        for index in range(20)
    )

    def fake_extract(_path, *, table_pages=None, **_kwargs):
        requested = set(table_pages or set())
        if not requested:
            return plain_pages
        return [
            PdfPage(1, plain_pages[0].text, (target_block,) if 1 in requested else ()),
            PdfPage(2, plain_pages[1].text, () if 2 not in requested else ("GAMME DE PRODUIT",)),
            PdfPage(3, plain_pages[2].text, distractors if 3 in requested else ()),
        ]

    monkeypatch.setattr(pipeline_module, "extract_pdf_pages", fake_extract)
    pipeline_module._catalogue_index_cached.cache_clear()

    results = retrieve_catalogue_chunks(
        catalogue,
        "fusible photovoltaique gPV HP10M 15 A 1000 VDC",
        top_k=1,
        candidate_k=8,
        hierarchy_enabled=False,
        ocr_mode="off",
    )

    assert any("REF-DEMO-02" in item.chunk.text for item in results)


def test_structured_page_keeps_plain_text_support_when_table_blocks_omit_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import rag_catalogue.pipeline as pipeline_module
    from rag_catalogue.pdf_text import PdfPage

    catalogue = tmp_path / "catalogue_plain_support.pdf"
    _write_pdf(catalogue, ["page 1", "page 2"])
    filler = " ".join(f"description{index}" for index in range(80))
    page1_text = (
        "Fusibles photovoltaïques gPV HP6M 600 VDC. GAMME DE PRODUIT. "
        + filler
        + " Fusibles photovoltaïques gPV HP10M 1000 VDC. "
        "Pouvoir de coupure maximal 10 kA. REF-DEMO-02 N1018590 15 A."
    )
    page2_text = (
        "GAMME DE PRODUIT. Numero de catalogue. Numero de reference. "
        "Fusible photovoltaique gPV 15 A 1000 VDC."
    )
    plain_pages = [PdfPage(1, page1_text), PdfPage(2, page2_text)]

    def fake_extract(_path, *, table_pages=None, **_kwargs):
        requested = set(table_pages or set())
        if not requested:
            return plain_pages
        return [
            PdfPage(
                1,
                page1_text,
                ("Numéro de catalogue: US101HEL\nNuméro de référence: D1009979",)
                if 1 in requested else (),
            ),
            PdfPage(2, page2_text, ("GAMME DE PRODUIT",) if 2 in requested else ()),
        ]

    monkeypatch.setattr(pipeline_module, "extract_pdf_pages", fake_extract)
    pipeline_module._catalogue_index_cached.cache_clear()

    results = retrieve_catalogue_chunks(
        catalogue,
        "fusible photovoltaique gPV HP10M 15 A 1000 VDC",
        top_k=1,
        candidate_k=8,
        hierarchy_enabled=False,
        ocr_mode="off",
    )

    assert any("REF-DEMO-02" in item.chunk.text for item in results)


def test_native_table_support_prefers_exact_reference_page_even_when_late() -> None:
    import rag_catalogue.pipeline as pipeline_module

    selected_tables = []
    plain_by_page = {}
    for page in range(1, 11):
        selected_tables.append(
            RankedChunk(
                chunk=CatalogueChunk(
                    chunk_id=f"p{page}-table-1",
                    page_number=page,
                    text=f"STRUCTURED TABLE VARIANT COLUMN page {page}",
                    kind="variant",
                ),
                score=1.0 - page * 0.01,
                word_score=0.1,
                char_score=0.1,
                exact_score=0.0,
                code_score=0.0,
            )
        )
        exact = 0.9 if page == 10 else 0.05
        text = (
            "REF-DEMO-02 N1018590 1000 VDC 15 A gPV"
            if page == 10
            else f"generic product table page {page}"
        )
        plain_by_page[page] = [
            RankedChunk(
                chunk=CatalogueChunk(
                    chunk_id=f"p{page}-c1",
                    page_number=page,
                    text=text,
                ),
                score=0.4 - page * 0.01,
                word_score=0.1,
                char_score=0.1,
                exact_score=exact,
                code_score=0.0,
            )
        ]

    supports = pipeline_module._select_native_table_support(
        selected_tables,
        plain_by_page,
        maximum=3,
    )

    assert 10 in {item.chunk.page_number for item in supports}
    assert any("REF-DEMO-02" in item.chunk.text for item in supports)


def test_collection_keeps_proved_near_alternative_when_only_source_product_differs() -> None:
    import rag_catalogue.pipeline as pipeline_module

    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p214-c2",
                page_number=214,
                text=(
                    "REF-DEMO-02 N1018590. Fusible photovoltaïque gPV, 1000 VDC, "
                    "15 A, pouvoir de coupure maximal 10 kA."
                ),
            ),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
            code_score=1.0,
        )
    ]
    need = {
        "product_type": "fusible photovoltaïque",
        "attributes": [
            {
                "name": "courant nominal",
                "value": "15",
                "unit": "A",
                "role": "selector",
                "provenance": "source_product",
            },
            {
                "name": "tension nominale",
                "value": "1000",
                "unit": "VDC",
                "role": "selector",
                "provenance": "source_product",
            },
            {
                "name": "pouvoir de coupure",
                "value": "30",
                "unit": "kA",
                "role": "selector",
                "provenance": "source_product",
            },
        ],
    }
    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "REF-DEMO-02",
                "variant": "1000 VDC",
                "match_level": "closest",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [214],
                "evidence": ["REF-DEMO-02 N1018590, 1000 VDC, 15 A, 10 kA"],
                "reason": "Alternative proche explicitement référencée.",
                "differences": ["Pouvoir de coupure 10 kA au lieu de 30 kA"],
            }
        ],
        "explanation": "Aucune correspondance exacte; alternative proche conservée.",
    }

    result = pipeline_module.validate_catalogue_result_collection(raw, need, chunks)

    assert result["reference"] == "REF-DEMO-02"
    assert result["candidates"][0]["differences"] == [
        "Pouvoir de coupure 10 kA au lieu de 30 kA"
    ]
    assert result["validation_errors"] == []


def test_normalise_generation_result_keeps_mandatory_difference_and_forbids_equivalent() -> None:
    from rag_catalogue.pipeline import normalise_generation_result

    raw = {
        "candidates": [
            {
                "rank": 1,
                "reference": "TARGET-EXACT",
                "variant": None,
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [10],
                "evidence": ["TARGET-EXACT"],
                "reason": "All mandatory requirements are satisfied.",
                "differences": [],
                "mandatory_differences": [],
            },
            {
                "rank": 2,
                "reference": "TARGET-CLOSE",
                "variant": None,
                "match_level": "equivalent",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [11],
                "evidence": ["TARGET-CLOSE"],
                "reason": "Close documented alternative.",
                "differences": ["Different primary function"],
                "mandatory_differences": ["Function differs from the derived requirement"],
            },
        ],
        "explanation": "Two candidates.",
    }

    result = normalise_generation_result(raw)

    close = next(
        item for item in result["candidates"] if item["reference"] == "TARGET-CLOSE"
    )
    assert close["mandatory_differences"] == [
        "Function differs from the derived requirement"
    ]
    assert close["match_level"] == "alternative_conditionnelle"
    assert close["equivalence_status"] == "alternative_conditionnelle"


def test_validate_collection_preserves_mandatory_difference_without_rejecting_candidate() -> None:
    from rag_catalogue.pipeline import validate_catalogue_result_collection

    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p12-c1",
                page_number=12,
                text="TARGET-CLOSE is a 400 A four-pole simple switch.",
            ),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]
    raw = {
        "candidates": [{
            "rank": 1,
            "reference": "TARGET-CLOSE",
            "variant": None,
            "match_level": "equivalent",
            "reference_mode": "explicit",
            "reference_parts": [],
            "catalogue_pages": [12],
            "evidence": ["TARGET-CLOSE is a 400 A four-pole simple switch"],
            "reason": "Close documented alternative.",
            "differences": ["Simple switch instead of source transfer"],
            "mandatory_differences": ["Function differs from the derived requirement"],
        }],
        "explanation": "One alternative.",
    }

    result = validate_catalogue_result_collection(
        raw,
        {"source_reference": "SOURCE-REF"},
        chunks,
    )

    assert result["reference"] == "TARGET-CLOSE"
    assert result["candidates"][0]["mandatory_differences"] == [
        "Function differs from the derived requirement"
    ]
    assert result["candidates"][0]["match_level"] != "equivalent"
    assert result["validation_errors"] == []
