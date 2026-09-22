import json
from pathlib import Path

import fitz

from rag_catalogue.pipeline import _catalogue_need, build_retrieval_queries, run_search


def _write_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox((50, 50, 550, 780), text, fontsize=10)
    doc.save(path)
    doc.close()


def test_catalogue_need_keeps_context_and_open_questions() -> None:
    need = {
        "product_type": "miniature circuit breaker MCB",
        "attributes": [
            {"name": "rated current", "value": "3", "unit": "A", "role": "selector", "provenance": "client"},
            {"name": "mounting", "value": "DIN rail", "unit": None, "role": "context", "provenance": "source_product"},
            {"name": "voltage", "value": "AC or DC", "unit": None, "role": "context", "provenance": "to_confirm"},
        ],
        "open_questions": ["application AC ou DC"],
    }

    selection = _catalogue_need(need, ["MCB 3 A"])

    assert len(selection["attributes"]) == 3
    assert selection["attributes"][1]["role"] == "context"
    assert selection["open_questions"] == ["application AC ou DC"]


def test_build_queries_use_selectors_but_keep_multiple_ambiguity_branches() -> None:
    need = {
        "product_type": "miniature circuit breaker MCB",
        "source_manufacturer": "Schneider Electric",
        "source_reference": "M9F11203",
        "attributes": [
            {"name": "number of poles", "value": "2P", "unit": None, "role": "selector", "provenance": "client"},
            {"name": "rated current", "value": "3", "unit": "A", "role": "selector", "provenance": "client"},
            {"name": "curve", "value": "C", "unit": None, "role": "selector", "provenance": "client"},
            {"name": "breaking capacity", "value": "10", "unit": "kA", "role": "selector", "provenance": "client"},
            {"name": "voltage", "value": "125", "unit": "V DC", "role": "context", "provenance": "source_product"},
        ],
        "search_queries": [
            "Schneider M9F11203 miniature circuit breaker MCB 2P 3 A C curve 10 kA AC",
            "miniature circuit breaker MCB 2P 3 A C curve 10 kA DC",
        ],
    }

    queries = build_retrieval_queries(need)
    combined = " ".join(queries).casefold()

    assert "schneider" not in combined
    assert "m9f11203" not in combined
    assert any(" ac" in f" {query.casefold()}" for query in queries)
    assert any(" dc" in f" {query.casefold()}" for query in queries)


class _BundleAwareLLM:
    model = "nvidia/fake"

    def __init__(self) -> None:
        self.received_product_text = ""
        self.received_need = None

    def extract_need(self, product_text: str) -> dict:
        self.received_product_text = product_text
        return {
            "product_type": "miniature circuit breaker MCB",
            "source_manufacturer": "Schneider Electric",
            "source_reference": "M9F11203",
            "attributes": [
                {"name": "number of poles", "value": "2P", "unit": None, "role": "constraint", "provenance": "client"},
                {"name": "rated current", "value": "3", "unit": "A", "role": "constraint", "provenance": "client"},
                {"name": "voltage", "value": "125", "unit": "V DC", "role": "context", "provenance": "source_product"},
            ],
            "open_questions": ["AC or DC application"],
            "search_queries": ["miniature circuit breaker MCB 2P 3 A"],
        }

    def select_reference(self, need: dict, chunks: list) -> dict:
        self.received_need = need
        return {
            "reference": "CANDIDATE-3A",
            "reference_mode": "explicit",
            "reference_parts": [],
            "catalogue_pages": [1],
            "evidence": ["Reference CANDIDATE-3A"],
            "explanation": "explicit candidate",
        }


def test_run_search_reads_full_cahier_plus_explicit_provenance(tmp_path: Path) -> None:
    cahier = tmp_path / "cahier_produit_1.txt"
    provenance = tmp_path / "provenance_produit_1.json"
    catalogue = tmp_path / "catalogue.pdf"
    cahier.write_text("CAHIER COMPLET avec dimensions, normes et fonction.", encoding="utf-8")
    provenance.write_text(json.dumps({
        "demande_client": "MCB 2P 3A Schneider M9F11203",
        "contraintes_client": ["2P", "3A"],
        "caracteristiques_produit": ["125 V DC"],
        "informations_a_confirmer": ["AC ou DC"],
    }, ensure_ascii=False), encoding="utf-8")
    _write_pdf(catalogue, ["Reference CANDIDATE-3A. Miniature circuit breaker 2P 3 A."])
    llm = _BundleAwareLLM()

    result = run_search(
        cahier,
        catalogue,
        llm,
        provenance_file=provenance,
        top_k=1,
        validate_result=True,
    )

    assert result["reference"] == "CANDIDATE-3A"
    assert "CAHIER COMPLET" in llm.received_product_text
    assert "PROVENANCE STRUCTUREE" in llm.received_product_text
    assert "125 V DC" in llm.received_product_text
    assert len(llm.received_need["attributes"]) == 3
    assert llm.received_need["open_questions"] == ["AC or DC application"]


def test_retrieval_query_keeps_more_than_six_variant_defining_attributes() -> None:
    attributes = [
        {"name": f"criterion {index}", "value": f"value{index}", "unit": None, "role": "selector"}
        for index in range(1, 9)
    ]
    need = {
        "product_type": "industrial product",
        "attributes": attributes,
        "search_queries": [],
    }

    queries = build_retrieval_queries(need)

    assert any("value8" in query for query in queries)


class _EmptyThenCandidateLLM:
    model = "nvidia/fake"

    def __init__(self, *, always_empty: bool = False) -> None:
        self.selection_calls = 0
        self.always_empty = always_empty

    def extract_need(self, product_text: str) -> dict:
        return {
            "product_type": "industrial product",
            "attributes": [
                {
                    "name": "rating",
                    "value": "3",
                    "unit": "A",
                    "role": "selector",
                    "provenance": "client",
                }
            ],
            "search_queries": ["industrial product 3 A"],
        }

    def select_reference(self, need: dict, chunks: list) -> dict:
        self.selection_calls += 1
        if self.selection_calls == 1 or self.always_empty:
            return {
                "candidates": [],
                "explanation": ["No candidate selected on this pass."],
            }
        return {
            "candidates": [
                {
                    "rank": 1,
                    "reference": "CANDIDATE-3A",
                    "variant": None,
                    "match_level": "closest",
                    "reference_mode": "explicit",
                    "reference_parts": [],
                    "catalogue_pages": [1],
                    "evidence": ["Reference CANDIDATE-3A"],
                    "reason": "Exact catalogue reference.",
                    "differences": [],
                }
            ],
            "explanation": "Candidate found after automatic retry.",
        }


def test_run_search_retries_once_when_first_generation_returns_empty_candidates(
    tmp_path: Path,
) -> None:
    product = tmp_path / "product.txt"
    product.write_text("Industrial product rated 3 A.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["Reference CANDIDATE-3A. Industrial product rated 3 A."])
    llm = _EmptyThenCandidateLLM()

    result = run_search(product, catalogue, llm, top_k=1)

    assert llm.selection_calls == 2
    assert result["reference"] == "CANDIDATE-3A"
    assert len(result["candidates"]) == 1


def test_run_search_stops_after_one_retry_when_both_generations_are_empty(
    tmp_path: Path,
) -> None:
    product = tmp_path / "product.txt"
    product.write_text("Industrial product rated 3 A.", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["Industrial product rated 3 A; no orderable reference shown."])
    llm = _EmptyThenCandidateLLM(always_empty=True)

    result = run_search(product, catalogue, llm, top_k=1)

    assert llm.selection_calls == 2
    assert result["reference"] is None
    assert result["candidates"] == []
