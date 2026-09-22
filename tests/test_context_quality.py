from rag_catalogue.context_quality import deduplicate_ranked_chunks, diversify_ranked_chunks
from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.retrieval import RankedChunk


def _ranked(chunk_id: str, page: int, text: str, score: float, *, section: str = "s1", kind: str = "technical") -> RankedChunk:
    return RankedChunk(
        chunk=CatalogueChunk(
            chunk_id=chunk_id,
            page_number=page,
            text=text,
            section_id=section,
            section_title=section,
            kind=kind,
        ),
        score=score,
        word_score=score,
        char_score=0.0,
        exact_score=0.0,
        fused_score=score,
    )


def test_deduplicate_ranked_chunks_drops_near_duplicate_prose() -> None:
    first = _ranked("p1-c1", 1, "Pompe centrifuge debit 20 m3 h hauteur 30 m moteur 4 kW", 0.9)
    duplicate = _ranked("p2-c1", 2, "Pompe centrifuge debit 20 m3 h hauteur 30 m moteur 4 kW.", 0.8)
    distinct = _ranked("p3-c1", 3, "Pompe immergee debit 15 m3 h hauteur 50 m", 0.7)

    kept, removed = deduplicate_ranked_chunks([first, duplicate, distinct])

    assert [item.chunk.chunk_id for item in kept] == ["p1-c1", "p3-c1"]
    assert removed == ["p2-c1"]


def test_deduplicate_preserves_distinct_variant_columns() -> None:
    first = _ranked(
        "p5-table-1",
        5,
        "STRUCTURED TABLE VARIANT COLUMN\nModel: AX63M\nCapacity: 6 kA\nCurrent: 16 A",
        0.9,
        kind="variant",
    )
    second = _ranked(
        "p5-table-2",
        5,
        "STRUCTURED TABLE VARIANT COLUMN\nModel: AX63P\nCapacity: 10 kA\nCurrent: 16 A",
        0.8,
        kind="variant",
    )

    kept, removed = deduplicate_ranked_chunks([first, second])

    assert len(kept) == 2
    assert removed == []


def test_diversify_reserves_structural_and_limits_same_page() -> None:
    items = [
        _ranked("order", 20, "ORDERING INFORMATION code map", 1.0, section="s2", kind="ordering"),
        _ranked("a", 1, "technical A", 0.9),
        _ranked("b", 1, "technical B", 0.8),
        _ranked("c", 2, "technical C", 0.7, section="s3"),
    ]

    result = diversify_ranked_chunks(items, top_k=3, max_per_page=1)

    assert result[0].chunk.kind == "ordering"
    assert len([item for item in result if item.chunk.page_number == 1]) == 1
    assert len(result) == 3
