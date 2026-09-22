from __future__ import annotations

from pathlib import Path

import fitz

from rag_catalogue.pipeline import index_catalogue, retrieve_catalogue_chunks, run_search


def _write_pdf(path: Path, pages: list[tuple[str, int]]) -> None:
    document = fitz.open()
    for text, heading_size in pages:
        page = document.new_page()
        first, _, rest = text.partition("\n")
        page.insert_text((72, 72), first, fontsize=heading_size)
        if rest:
            page.insert_textbox((72, 105, 540, 760), rest, fontsize=9)
    document.save(path)
    document.close()


class _LLM:
    model = "nvidia/test-generation"

    def extract_need(self, _text: str) -> dict:
        return {
            "product_type": "pompe centrifuge",
            "source_reference": None,
            "attributes": [
                {"name": "debit", "value": "20", "unit": "m3/h", "role": "selector"},
            ],
            "search_queries": ["pompe centrifuge debit 20 m3/h"],
        }

    def select_reference(self, _need: dict, chunks: list) -> dict:
        page = chunks[0].chunk.page_number
        return {
            "reference": "PUMP-20",
            "reference_mode": "explicit",
            "reference_parts": [],
            "catalogue_pages": [page],
            "evidence": ["PUMP-20"],
            "explanation": "reference explicite",
        }


def test_v13_diagnostic_reports_hierarchy_ocr_and_context_quality(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("pompe centrifuge debit 20 m3/h", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            ("POMPES CENTRIFUGES\nDescription generale des pompes.", 18),
            ("SELECTION TABLE\nPUMP-20 pompe centrifuge debit 20 m3/h", 14),
            ("ORDERING INFORMATION\nPUMP-20 reference de commande", 14),
            ("MOTEURS\nMoteurs industriels 4 kW", 18),
        ],
    )

    result = run_search(
        product,
        catalogue,
        _LLM(),
        top_k=2,
        candidate_k=8,
        include_diagnostics=True,
        hierarchical=True,
        catalogue_profile="large",
        ocr_mode="auto",
    )

    metadata = result["diagnostic"]["retrieval_metadata"]
    assert metadata["hierarchy_enabled"] is True
    assert metadata["catalogue_profile"] == "large"
    assert metadata["catalogue_chunk_count"] >= 4
    assert metadata["catalogue_section_count"] >= 2
    assert isinstance(metadata["selected_sections"], list)
    assert "deduplicated_chunk_ids" in metadata
    assert metadata["ocr_mode"] == "auto"
    retrieval = result["diagnostic"]["retrieval"]
    assert retrieval
    assert {"section_id", "section_title", "kind"} <= set(retrieval[0])


def test_hierarchical_retrieval_keeps_global_exact_code_fallback(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            ("POMPES\nPompes centrifuges pour eau", 18),
            ("MOTEURS\nMoteurs asynchrones", 18),
            ("ANNEXE\nReference exacte ZXQ-991 special replacement", 18),
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        "pompe ZXQ-991",
        top_k=3,
        candidate_k=12,
        hierarchical=True,
    )

    assert any("ZXQ-991" in item.chunk.text for item in results)


def test_index_catalogue_builds_map_and_plain_index_without_embedding(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            ("CAPTEURS\nCapteurs inductifs", 18),
            ("ORDERING INFORMATION\nCode de commande XS-10", 14),
        ],
    )

    summary = index_catalogue(
        catalogue,
        embedding_client=None,
        cache_dir=tmp_path / "cache",
        ocr_mode="off",
    )

    assert summary["page_count"] == 2
    assert summary["chunk_count"] >= 2
    assert summary["section_count"] >= 1
    assert summary["dense_index_created"] is False
