from pathlib import Path
import os

import pytest

from rag_catalogue.pipeline import build_retrieval_queries, retrieve_catalogue_chunks, validate_catalogue_result


CATALOGUE = Path(__file__).resolve().parents[2] / "INTERRUPTORES-MODULARES-HGD-HRC-HRO-2406.pdf"
RUN_EXTERNAL_CATALOGUE_TESTS = (
    os.environ.get("RUN_EXTERNAL_CATALOGUE_TESTS") == "1" and CATALOGUE.is_file()
)


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_blind_vendora_query_retrieves_ordering_page_without_target_reference() -> None:
    results = retrieve_catalogue_chunks(
        CATALOGUE,
        [
            "miniature circuit breaker standard type 63 AF 10 kA two poles C curve 16 A 240 415 V 50 60 Hz front connection no accessories",
            "MCB 2P C curve rated current 16 A short circuit breaking capacity 10 kA",
        ],
        top_k=5,
    )

    pages = [item.chunk.page_number for item in results]
    assert 42 in pages


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_french_schneider_like_query_retrieves_hgd_ordering_page() -> None:
    need = {
        "product_type": "disjoncteur modulaire",
        "source_reference": "A9F77206",
        "attributes": [
            {"name": "courant nominal", "value": "6", "unit": "A", "role": "selector"},
            {"name": "nombre de pôles", "value": "2", "unit": None, "role": "selector"},
            {"name": "courbe de déclenchement", "value": "C", "unit": None, "role": "selector"},
            {"name": "pouvoir de coupure", "value": "10", "unit": "kA", "role": "selector"},
        ],
        "search_queries": ["disjoncteur modulaire 2P 6A courbe C 10kA 400V"],
    }
    results = retrieve_catalogue_chunks(
        CATALOGUE,
        build_retrieval_queries(need),
        top_k=8,
    )

    pages = {item.chunk.page_number for item in results}
    assert 15 in pages
    assert 42 in pages

@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_actual_v7_need_retrieves_standard_mcb_ordering_page() -> None:
    need = {
        "product_type": "disjoncteur modulaire",
        "source_manufacturer": "Schneider Electric",
        "source_family": "Acti9 iC60N",
        "source_reference": "A9F77206",
        "attributes": [
            {"name": "courant nominal", "value": "6", "unit": "A", "role": "selector"},
            {"name": "nombre de pôles", "value": "2", "unit": None, "role": "selector"},
            {"name": "courbe de déclenchement", "value": "C", "unit": None, "role": "selector"},
            {"name": "tension nominale", "value": "400", "unit": "V AC", "role": "selector"},
            {
                "name": "pouvoir de coupure assigné (IEC 60947-2 à 400 V)",
                "value": "10",
                "unit": "kA",
                "role": "selector",
            },
            {"name": "type de tension", "value": "AC/DC", "unit": None, "role": "selector"},
        ],
        "search_queries": [
            "disjoncteur modulaire 2P 6A courbe C 400V 10kA",
            "disjoncteur modulaire 2 pôles 6 ampères déclenchement courbe C",
            "disjoncteur modulaire iC60N 2P 6A C 400VAC 10kA Icu",
        ],
    }

    results = retrieve_catalogue_chunks(
        CATALOGUE,
        build_retrieval_queries(need),
        top_k=8,
    )

    pages = {item.chunk.page_number for item in results}
    assert 15 in pages
    assert 42 in pages


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_table_extraction_keeps_model_columns_separate() -> None:
    from rag_catalogue.pdf_text import extract_pdf_pages

    pages = {page.page_number: page for page in extract_pdf_pages(CATALOGUE, table_pages={14, 15, 42})}
    page14_blocks = "\n".join(pages[14].structured_blocks)
    page15_blocks = "\n".join(pages[15].structured_blocks)

    assert "Model: HGD63N, 63 AF, 6 kA" in page14_blocks
    assert "Rated Short Circuit Current (Icn): 6 kA" in page14_blocks
    assert "Model: HGD63H, 63 AF, 10 kA" in page14_blocks
    assert "Rated Short Circuit Current (Icn): 10 kA" in page14_blocks
    assert "Model: REF-DEMO-03" in page15_blocks
    assert "Rated Short Circuit Current (Icn): 10 kA" in page15_blocks


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_actual_v7_need_retrieves_standard_ordering_page() -> None:
    need = {
        "product_type": "disjoncteur modulaire",
        "source_manufacturer": "Schneider Electric",
        "source_family": "Acti9 iC60N",
        "source_reference": "A9F77206",
        "attributes": [
            {"name": "courant nominal", "value": "6", "unit": "A", "role": "selector"},
            {"name": "nombre de pôles", "value": "2", "unit": None, "role": "selector"},
            {"name": "courbe de déclenchement", "value": "C", "unit": None, "role": "selector"},
            {"name": "tension nominale", "value": "400", "unit": "V AC", "role": "selector"},
            {"name": "pouvoir de coupure assigné (IEC 60947-2 à 400 V)", "value": "10", "unit": "kA", "role": "selector"},
            {"name": "type de tension", "value": "AC/DC", "unit": None, "role": "selector"},
        ],
        "search_queries": [
            "disjoncteur modulaire 2P 6A courbe C 400V 10kA",
            "disjoncteur modulaire 2 pôles 6 ampères déclenchement courbe C",
            "disjoncteur modulaire iC60N 2P 6A C 400VAC 10kA Icu",
        ],
    }

    results = retrieve_catalogue_chunks(
        CATALOGUE,
        build_retrieval_queries(need),
        top_k=8,
    )

    pages = {item.chunk.page_number for item in results}
    assert 15 in pages
    assert 42 in pages


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_v8_context_prefers_column_safe_hgd_records_and_linked_ordering_schema() -> None:
    need = {
        "product_type": "disjoncteur modulaire",
        "source_manufacturer": "Schneider Electric",
        "source_family": "Acti9 iC60N",
        "source_reference": "A9F77206",
        "attributes": [
            {"name": "courant nominal", "value": "6", "unit": "A", "role": "selector"},
            {"name": "nombre de pôles", "value": "2", "unit": None, "role": "selector"},
            {"name": "courbe de déclenchement", "value": "C", "unit": None, "role": "selector"},
            {"name": "tension nominale", "value": "400", "unit": "V AC", "role": "selector"},
            {"name": "pouvoir de coupure", "value": "10", "unit": "kA", "role": "selector"},
        ],
        "search_queries": ["disjoncteur modulaire 2P 6A courbe C 400V 10kA"],
    }

    results = retrieve_catalogue_chunks(
        CATALOGUE,
        build_retrieval_queries(need),
        top_k=8,
    )
    ids = [item.chunk.chunk_id for item in results]

    assert "p42-table-order" in ids[:6]
    assert "p15-table-5" in ids[:10]
    assert ids.index("p15-table-5") < ids.index("p14-table-2")
    if "p70-table-order" in ids:
        assert ids.index("p42-table-order") < ids.index("p70-table-order")
    hro_identity_positions = [
        index for index, chunk_id in enumerate(ids)
        if chunk_id.startswith("p64-table-")
    ]
    if hro_identity_positions:
        assert ids.index("p15-table-5") < min(hro_identity_positions)
    assert "p14-c1" not in ids


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_variant_chunk_page_context_does_not_repeat_flattened_neighbor_columns() -> None:
    results = retrieve_catalogue_chunks(
        CATALOGUE,
        "miniature circuit breaker 2P 6A C curve 10 kA standard type",
        top_k=8,
    )
    target = next(item for item in results if item.chunk.chunk_id == "p15-table-5")

    assert "PAGE CONTEXT: HGD (Standard Type)" in target.chunk.text
    assert "Model: REF-DEMO-03" in target.chunk.text
    context_line = target.chunk.text.splitlines()[0]
    assert "HGD63M" not in context_line
    assert "HGD63U" not in context_line


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_ordering_chunk_exposes_complete_order_template_without_target_reference() -> None:
    results = retrieve_catalogue_chunks(
        CATALOGUE,
        "miniature circuit breaker 2P 6A C curve 10 kA standard type",
        top_k=8,
    )
    ordering = next(item for item in results if item.chunk.chunk_id == "p42-table-order")

    assert "STRUCTURED ORDER TEMPLATE" in ordering.chunk.text
    assert "SLOT COUNT: 10" in ordering.chunk.text
    assert "EXAMPLE CODES: HGD | 63 | M | 1P | MC | S | 00 | 00 | C | 00001" in ordering.chunk.text
    assert "1=Type" in ordering.chunk.text
    assert "10=Rated Current" in ordering.chunk.text


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_v12_full_cahier_need_retrieves_mcb_selection_and_ordering_pages_for_3a() -> None:
    need = {
        "product_type": "miniature circuit breaker MCB",
        "source_manufacturer": "Schneider Electric",
        "source_reference": "M9F11203",
        "attributes": [
            {"name": "number of poles", "value": "2P", "unit": None, "role": "constraint", "provenance": "client"},
            {"name": "rated current", "value": "3", "unit": "A", "role": "constraint", "provenance": "client"},
            {"name": "curve", "value": "C", "unit": None, "role": "constraint", "provenance": "client"},
            {"name": "breaking capacity", "value": "10", "unit": "kA", "role": "constraint", "provenance": "client"},
            {"name": "documented voltage", "value": "125", "unit": "V DC", "role": "context", "provenance": "source_product"},
        ],
        "open_questions": ["application AC or DC"],
        "search_queries": [
            "miniature circuit breaker MCB 2P 3 A C curve 10 kA AC standard",
            "miniature circuit breaker MCB 2P 3 A C curve 10 kA DC",
        ],
    }

    results = retrieve_catalogue_chunks(
        CATALOGUE,
        build_retrieval_queries(need),
        top_k=8,
    )
    pages = {item.chunk.page_number for item in results}

    assert 15 in pages
    assert 42 in pages


@pytest.mark.skipif(
    not RUN_EXTERNAL_CATALOGUE_TESTS,
    reason="test catalogue externe desactive ou PDF indisponible",
)
def test_deluxe_ordering_slot_map_rejects_breaking_code_in_frame_slot() -> None:
    results = retrieve_catalogue_chunks(
        CATALOGUE,
        "miniature circuit breaker deluxe 2P 16A C curve 10 kA",
        top_k=8,
    )
    ordering = next(item for item in results if item.chunk.chunk_id == "p41-table-order")
    assert "SLOT 2 ALLOWED CODES: 63 | 125" in ordering.chunk.text
    assert "SLOT 3 ALLOWED CODES: N | H" in ordering.chunk.text

    bad = {
        "reference": "HGDNN2PMCS0000C00016",
        "reference_mode": "constructed",
        "reference_parts": [
            {"position": 1, "code": "HGD", "page": 41},
            {"position": 2, "code": "N", "page": 41},
            {"position": 3, "code": "N", "page": 41},
            {"position": 4, "code": "2P", "page": 41},
            {"position": 5, "code": "MC", "page": 41},
            {"position": 6, "code": "S", "page": 41},
            {"position": 7, "code": "00", "page": 41},
            {"position": 8, "code": "00", "page": 41},
            {"position": 9, "code": "C", "page": 41},
            {"position": 10, "code": "00016", "page": 41},
        ],
        "catalogue_pages": [41],
        "evidence": ["Ordering Information"],
        "explanation": "",
    }

    validated = validate_catalogue_result(bad, {"source_reference": None}, results)

    assert validated["reference"] is None
    assert "constructed_part_wrong_slot" in validated["validation_errors"]
