from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.pipeline import adaptive_retrieval_profile, select_top_sections
from rag_catalogue.retrieval import RankedChunk, weighted_rank_fusion


def _item(chunk_id: str, page: int, score: float, section_id: str) -> RankedChunk:
    return RankedChunk(
        chunk=CatalogueChunk(
            chunk_id=chunk_id,
            page_number=page,
            text=f"texte {chunk_id}",
            section_id=section_id,
            section_title=section_id,
        ),
        score=score,
        word_score=score,
        char_score=0,
        exact_score=0,
        fused_score=score,
    )


def test_adaptive_retrieval_profile_uses_chunk_count_not_only_pages() -> None:
    compact = adaptive_retrieval_profile(100, 300, 4, top_k=8, candidate_k=32)
    dense = adaptive_retrieval_profile(100, 1800, 12, top_k=8, candidate_k=32)

    assert compact["top_k"] == 8
    assert compact["candidate_k"] == 32
    assert dense["top_k"] > compact["top_k"]
    assert dense["candidate_k"] > compact["candidate_k"]
    assert dense["section_k"] > compact["section_k"]


def test_select_top_sections_aggregates_multiple_supporting_chunks() -> None:
    ranked = [
        _item("a1", 1, 0.50, "A"),
        _item("b1", 10, 0.49, "B"),
        _item("b2", 11, 0.48, "B"),
        _item("b3", 12, 0.47, "B"),
    ]

    sections = select_top_sections(ranked, maximum=1)

    assert sections == ["B"]


def test_weighted_rank_fusion_combines_three_lanes() -> None:
    a = _item("a", 1, 0.8, "A")
    b = _item("b", 2, 0.7, "B")
    c = _item("c", 3, 0.6, "C")

    fused = weighted_rank_fusion(
        [("global", [a, b], 0.8), ("section", [b, c], 1.2), ("structure", [c], 1.0)],
        top_k=3,
    )

    assert fused[0].chunk.chunk_id in {"b", "c"}
    assert {item.chunk.chunk_id for item in fused} == {"a", "b", "c"}


def test_retrieval_metadata_exposes_hierarchy_and_context_quality(tmp_path) -> None:
    import fitz
    from rag_catalogue.pipeline import retrieve_catalogue_chunks

    pdf = tmp_path / "large-map.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "POMPES CENTRIFUGES", fontsize=18)
    page.insert_text((72, 110), "pompe debit 20 m3 h hauteur 30 m reference PUMP20")
    page = document.new_page()
    page.insert_text((72, 72), "pompe debit 20 m3 h hauteur 30 m reference PUMP20")
    page = document.new_page()
    page.insert_text((72, 72), "ORDERING INFORMATION", fontsize=18)
    page.insert_text((72, 110), "PUMP20 code de commande pompe centrifuge")
    page = document.new_page()
    page.insert_text((72, 72), "MOTEURS ELECTRIQUES", fontsize=18)
    page.insert_text((72, 110), "moteur 4 kW 400 V")
    document.save(pdf)
    document.close()

    metadata = {}
    results = retrieve_catalogue_chunks(
        pdf,
        "pompe centrifuge debit 20 m3 h hauteur 30 m",
        top_k=3,
        candidate_k=8,
        hierarchy_enabled=True,
        catalogue_profile="large",
        retrieval_metadata=metadata,
    )

    assert results
    assert metadata["hierarchy_enabled"] is True
    assert metadata["catalogue_chunk_count"] >= 4
    assert metadata["catalogue_section_count"] >= 2
    assert metadata["selected_sections"]
    assert "retrieval_lanes" in metadata
    assert "deduplicated_chunk_ids" in metadata
    assert any(item.chunk.kind == "ordering" for item in results)


def test_hierarchical_retrieval_keeps_global_exact_reference_fallback(tmp_path) -> None:
    import fitz
    from rag_catalogue.pipeline import retrieve_catalogue_chunks

    pdf = tmp_path / "exact-fallback.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "POMPES", fontsize=18)
    first.insert_text((72, 110), "pompe centrifuge industrielle debit 20 m3 h")
    second = document.new_page()
    second.insert_text((72, 72), "ACCESSOIRES", fontsize=18)
    second.insert_text((72, 110), "reference exacte ZXQ-9917 raccord special")
    document.save(pdf)
    document.close()

    results = retrieve_catalogue_chunks(
        pdf,
        "pompe ZXQ-9917",
        top_k=2,
        candidate_k=8,
        hierarchy_enabled=True,
    )

    assert any("ZXQ-9917" in item.chunk.text for item in results)


def test_adaptive_profile_does_not_overexpand_short_heading_dense_catalogue() -> None:
    profile = adaptive_retrieval_profile(70, 104, 52, top_k=8, candidate_k=32)

    assert profile["top_k"] <= 12
    assert profile["candidate_k"] <= 64
    assert profile["section_k"] <= 6


def test_section_expansion_follows_nearby_shared_family_markers(tmp_path) -> None:
    from pathlib import Path
    from rag_catalogue.catalogue_map import CatalogueMap, CatalogueSection
    from rag_catalogue.pipeline import _expand_section_ids

    def section(section_id: str, title: str, page: int, markers: tuple[str, ...]) -> CatalogueSection:
        return CatalogueSection(
            section_id=section_id,
            title=title,
            start_page=page,
            end_page=page,
            page_numbers=(page,),
            kind="text",
            markers=markers,
            summary=title,
        )

    catalogue_map = CatalogueMap(
        path=Path(tmp_path / "catalogue.pdf"),
        page_count=12,
        source="layout",
        sections=(
            section("s001", "Pompes", 1, ("PUMP",)),
            section("s002", "Product feature", 2, ("PUMP",)),
            section("s003", "Selection table", 3, ("PUMP", "PX20")),
            section("s004", "Standard type", 4, ("PUMP", "PX20")),
            section("s005", "Moteurs", 5, ("MOTOR",)),
        ),
    )

    expanded = _expand_section_ids(catalogue_map, ["s001"])

    assert expanded == ["s001", "s002", "s003", "s004"]
