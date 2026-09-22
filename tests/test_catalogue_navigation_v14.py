from pathlib import Path

import fitz

from rag_catalogue.catalogue_map import CatalogueMap, CatalogueSection
from rag_catalogue.pdf_text import CatalogueChunk, PdfPage
from rag_catalogue.retrieval import RankedChunk


def _ranked(node_id: str, rank: int, score: float, title: str = "") -> RankedChunk:
    return RankedChunk(
        chunk=CatalogueChunk(
            chunk_id=f"nav-{node_id}",
            page_number=rank,
            text=title or node_id,
            section_id=node_id,
            section_title=title or node_id,
            kind="navigation",
        ),
        score=score,
        word_score=score,
        char_score=score / 2,
        exact_score=0.0,
        fused_score=score,
    )


def test_textual_toc_builds_parent_child_navigation_map() -> None:
    from rag_catalogue.catalogue_navigation import build_navigation_map

    pages = [
        PdfPage(
            page_number=1,
            heading="SOMMAIRE",
            heading_confidence=0.95,
            text=(
                "SOMMAIRE\n"
                "FUSIBLES PHOTOVOLTAIQUES 10 - 30\n"
                "Fusibles gPV cylindriques 12 - 18\n"
                "Porte-fusibles photovoltaïques 19 - 22\n"
                "PARAFOUDRES 31 - 45\n"
            ),
        ),
        *[
            PdfPage(page_number=number, text=f"contenu page {number}")
            for number in range(2, 46)
        ],
    ]
    sections = (
        CatalogueSection(
            section_id="s001",
            title="Front matter",
            start_page=1,
            end_page=9,
            page_numbers=tuple(range(1, 10)),
            kind="text",
            markers=(),
            summary="front matter",
        ),
        CatalogueSection(
            section_id="s002",
            title="Fusibles photovoltaïques",
            start_page=10,
            end_page=30,
            page_numbers=tuple(range(10, 31)),
            kind="table",
            markers=("GPV",),
            summary="fusibles photovoltaïques gPV",
        ),
        CatalogueSection(
            section_id="s003",
            title="Parafoudres",
            start_page=31,
            end_page=45,
            page_numbers=tuple(range(31, 46)),
            kind="table",
            markers=("SPD",),
            summary="parafoudres",
        ),
    )
    catalogue_map = CatalogueMap(
        path=Path("catalogue.pdf"),
        page_count=45,
        sections=sections,
        source="layout",
    )

    navigation = build_navigation_map(catalogue_map, pages)

    assert navigation.source == "text-toc"
    assert navigation.structure_confidence >= 0.8
    child = next(node for node in navigation.nodes if node.title == "Fusibles gPV cylindriques")
    parent = next(node for node in navigation.nodes if node.title == "FUSIBLES PHOTOVOLTAIQUES")
    assert child.parent_id == parent.node_id
    assert child.page_numbers == tuple(range(12, 19))
    assert navigation.route_path(child.node_id) == (
        "FUSIBLES PHOTOVOLTAIQUES",
        "Fusibles gPV cylindriques",
    )


def test_unstructured_catalogue_navigation_falls_back_to_global() -> None:
    from rag_catalogue.catalogue_navigation import build_navigation_map, plan_navigation

    pages = [PdfPage(page_number=1, text="texte continu sans titres ni sommaire")]
    catalogue_map = CatalogueMap(
        path=Path("plain.pdf"),
        page_count=1,
        source="layout",
        sections=(
            CatalogueSection(
                section_id="s001",
                title="Page 1",
                start_page=1,
                end_page=1,
                page_numbers=(1,),
                kind="text",
                markers=(),
                summary="texte continu sans titres ni sommaire",
            ),
        ),
    )

    navigation = build_navigation_map(catalogue_map, pages)
    plan = plan_navigation(navigation, [], maximum_routes=3)

    assert navigation.structure_confidence < 0.5
    assert plan.strategy == "global"
    assert plan.global_fallback_used is True
    assert plan.fallback_reason == "structure_unreliable"


def test_navigation_plan_prefers_specific_child_and_keeps_distinct_secondary() -> None:
    from rag_catalogue.catalogue_navigation import (
        CatalogueNavigationMap,
        NavigationNode,
        plan_navigation,
    )

    nodes = (
        NavigationNode(
            node_id="n001",
            title="HELIOPROTECTION",
            start_page=212,
            end_page=229,
            page_numbers=tuple(range(212, 230)),
            level=1,
            parent_id=None,
            summary="HelioProtection fusibles photovoltaïques parafoudres",
            source="text-toc",
        ),
        NavigationNode(
            node_id="n002",
            title="Fusibles photovoltaïques",
            start_page=214,
            end_page=221,
            page_numbers=tuple(range(214, 222)),
            level=2,
            parent_id="n001",
            summary="fusibles photovoltaïques gPV 1000 VDC",
            source="text-toc",
        ),
        NavigationNode(
            node_id="n003",
            title="Porte-fusibles photovoltaïques",
            start_page=222,
            end_page=224,
            page_numbers=(222, 223, 224),
            level=2,
            parent_id="n001",
            summary="porte-fusibles photovoltaïques",
            source="text-toc",
        ),
        NavigationNode(
            node_id="n004",
            title="Fusibles ultra-rapides",
            start_page=290,
            end_page=335,
            page_numbers=tuple(range(290, 336)),
            level=1,
            parent_id=None,
            summary="fusibles ultra rapides semi conducteurs",
            source="text-toc",
        ),
    )
    navigation = CatalogueNavigationMap(
        nodes=nodes,
        source="text-toc",
        structure_confidence=0.95,
        reasons=("textual_toc",),
        page_count=348,
    )
    ranked = [
        _ranked("n001", 1, 0.85, "HELIOPROTECTION"),
        _ranked("n002", 2, 0.83, "Fusibles photovoltaïques"),
        _ranked("n003", 3, 0.30, "Porte-fusibles photovoltaïques"),
        _ranked("n004", 4, 0.12, "Fusibles ultra-rapides"),
    ]

    plan = plan_navigation(navigation, ranked, maximum_routes=2)

    assert plan.strategy == "hierarchical"
    assert plan.primary_node_id == "n002"
    assert plan.secondary_node_ids == ("n003",)
    assert plan.route_path == ("HELIOPROTECTION", "Fusibles photovoltaïques")
    assert set(range(214, 222)).issubset(plan.primary_pages)


def test_local_evidence_quality_requests_global_fallback_when_values_are_missing() -> None:
    from rag_catalogue.catalogue_navigation import assess_local_evidence

    weak = [
        _ranked("n002", 1, 0.5, "fusibles photovoltaïques présentation générale"),
    ]
    strong = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p214",
                page_number=214,
                text="Fusible gPV 15 A 1000 VDC référence REF-DEMO-02",
                section_id="n002",
                section_title="Fusibles photovoltaïques",
                kind="table",
            ),
            score=0.8,
            word_score=0.7,
            char_score=0.4,
            exact_score=0.8,
            code_score=0.5,
            fused_score=0.8,
        )
    ]

    weak_quality = assess_local_evidence(
        weak,
        ["fusible gPV 15 A 1000 VDC"],
    )
    strong_quality = assess_local_evidence(
        strong,
        ["fusible gPV 15 A 1000 VDC"],
    )

    assert weak_quality.sufficient is False
    assert weak_quality.coverage < strong_quality.coverage
    assert strong_quality.sufficient is True
    assert strong_quality.has_reference_signal is True


def test_selection_guide_is_context_but_never_the_route_destination() -> None:
    from rag_catalogue.catalogue_navigation import (
        CatalogueNavigationMap,
        NavigationNode,
        plan_navigation,
    )

    nodes = (
        NavigationNode(
            node_id="n001",
            title="Interrupteurs-sectionneurs",
            start_page=15,
            end_page=79,
            page_numbers=tuple(range(15, 80)),
            level=1,
            parent_id=None,
            summary="NAVIGATION: Interrupteurs-sectionneurs.",
            source="toc",
            section_ids=("s001",),
        ),
        NavigationNode(
            node_id="n002",
            title="Guide de choix",
            start_page=18,
            end_page=21,
            page_numbers=tuple(range(18, 22)),
            level=2,
            parent_id="n001",
            summary="NAVIGATION: Guide de choix interrupteurs-sectionneurs.",
            source="toc",
            section_ids=("s002",),
        ),
        NavigationNode(
            node_id="n003",
            title="GAMME-X",
            start_page=40,
            end_page=63,
            page_numbers=tuple(range(40, 64)),
            level=2,
            parent_id="n001",
            summary="NAVIGATION: GAMME-X interrupteurs-sectionneurs pour la distribution.",
            source="toc",
            section_ids=("s003",),
        ),
    )
    navigation = CatalogueNavigationMap(
        nodes=nodes,
        source="toc",
        structure_confidence=0.95,
        reasons=("pdf_outline",),
        page_count=906,
    )
    # The guide outranks the product section, as it restates the whole chapter.
    ranked = [
        _ranked("n002", 1, 0.90, "Guide de choix"),
        _ranked("n003", 2, 0.70, "GAMME-X"),
    ]

    plan = plan_navigation(navigation, ranked, maximum_routes=3)

    assert plan.primary_node_id == "n003"
    assert 55 in plan.primary_pages
    assert 19 in plan.secondary_pages


def test_local_evidence_is_not_diluted_by_a_translated_query_variant() -> None:
    from rag_catalogue.catalogue_navigation import assess_local_evidence

    candidates = [
        _ranked("n002", 214, 0.8, "Fusible gPV 15 A 1000 VDC reference REF-DEMO-02"),
    ]

    matching_only = assess_local_evidence(candidates, ["fusible gPV 15 A 1000 VDC"])
    with_translation = assess_local_evidence(
        candidates,
        ["fusible gPV 15 A 1000 VDC", "photovoltaic string fuse 15 amp panel mounted"],
    )

    assert with_translation.coverage == matching_only.coverage


def test_v14_retrieval_routes_to_textual_toc_section_and_reports_navigation(tmp_path: Path) -> None:
    from rag_catalogue.pipeline import retrieve_catalogue_chunks

    pdf = tmp_path / "hierarchical.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "SOMMAIRE", fontsize=18)
    first.insert_text((72, 110), "POMPES 2 - 4\nPompes centrifuges 2 - 3\nMOTEURS 5 - 6")
    for number in range(2, 7):
        page = document.new_page()
        if number == 2:
            page.insert_text((72, 72), "POMPES CENTRIFUGES", fontsize=18)
            page.insert_text((72, 110), "pompe debit 20 m3 h hauteur 30 m reference PUMP20")
        elif number == 3:
            page.insert_text((72, 72), "TABLEAU DE SELECTION", fontsize=18)
            page.insert_text((72, 110), "PUMP20 debit 20 m3 h hauteur 30 m")
        elif number == 4:
            page.insert_text((72, 72), "ACCESSOIRES POMPES", fontsize=18)
        else:
            page.insert_text((72, 72), "MOTEURS ELECTRIQUES", fontsize=18)
            page.insert_text((72, 110), "moteur 4 kW 400 V")
    document.save(pdf)
    document.close()

    metadata: dict[str, object] = {}
    results = retrieve_catalogue_chunks(
        pdf,
        "pompe centrifuge debit 20 m3 h hauteur 30 m",
        top_k=3,
        candidate_k=12,
        retrieval_metadata=metadata,
        hierarchy_enabled=True,
    )

    assert results
    assert metadata["navigation_strategy"] == "hierarchical"
    assert metadata["navigation_source"] == "text-toc"
    assert metadata["catalogue_route"] == ["POMPES", "Pompes centrifuges"]
    assert metadata["primary_page_range"] == [2, 3]
    assert metadata["global_fallback_used"] is False
    assert {item.chunk.page_number for item in results}.intersection({2, 3})


def test_v14_retrieval_uses_global_rag_when_no_reliable_structure(tmp_path: Path) -> None:
    from rag_catalogue.pipeline import retrieve_catalogue_chunks

    pdf = tmp_path / "flat.pdf"
    document = fitz.open()
    for number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"texte continu page {number} reference ZX-{number}")
    document.save(pdf)
    document.close()

    metadata: dict[str, object] = {}
    results = retrieve_catalogue_chunks(
        pdf,
        "reference ZX-3",
        top_k=2,
        candidate_k=8,
        retrieval_metadata=metadata,
        hierarchy_enabled=True,
    )

    assert results
    assert metadata["navigation_strategy"] == "global"
    assert metadata["global_fallback_used"] is True
    assert metadata["global_fallback_reason"] == "structure_unreliable"
    assert any("ZX-3" in item.chunk.text for item in results)


def test_v14_inspection_exposes_navigation_map(tmp_path: Path) -> None:
    from rag_catalogue.pipeline import inspect_catalogue

    pdf = tmp_path / "inspect-navigation.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "CONTENTS", fontsize=18)
    first.insert_text((72, 110), "VALVES 2 - 3\nPUMPS 4 - 5")
    for title in ("VALVES", "VALVE TABLE", "PUMPS", "PUMP TABLE"):
        page = document.new_page()
        page.insert_text((72, 72), title, fontsize=18)
    document.save(pdf)
    document.close()

    report = inspect_catalogue(pdf, ocr_mode="off")

    assert report["navigation_source"] == "text-toc"
    assert report["navigation_structure_confidence"] >= 0.8
    assert report["navigation_node_count"] == 2
    assert report["navigation_nodes"][0]["title"] == "VALVES"


def test_textual_toc_reads_continuation_page_without_repeated_heading() -> None:
    from rag_catalogue.catalogue_navigation import build_navigation_map

    pages = [
        PdfPage(
            page_number=1,
            heading="CONTENTS",
            heading_confidence=0.9,
            text="CONTENTS\nPUMPS 10 - 20\nVALVES 21 - 30",
        ),
        PdfPage(
            page_number=2,
            text="MOTORS 31 - 40\nSENSORS 41 - 50",
        ),
        *[PdfPage(page_number=number, text=f"page {number}") for number in range(3, 51)],
    ]
    catalogue_map = CatalogueMap(
        path=Path("continuation.pdf"),
        page_count=50,
        source="layout",
        sections=(
            CatalogueSection(
                section_id="s001",
                title="Catalogue",
                start_page=1,
                end_page=50,
                page_numbers=tuple(range(1, 51)),
                kind="text",
                markers=(),
                summary="catalogue",
            ),
        ),
    )

    navigation = build_navigation_map(catalogue_map, pages)

    assert {node.title for node in navigation.nodes} >= {"PUMPS", "VALVES", "MOTORS", "SENSORS"}
