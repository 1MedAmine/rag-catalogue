from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.retrieval import HybridRetriever


def _chunks() -> list[CatalogueChunk]:
    return [
        CatalogueChunk(
            chunk_id="bearing-6204",
            page_number=3,
            text="Reference 6204-Z deep groove ball bearing bore 20 mm outside diameter 47 mm width 14 mm metal shield",
        ),
        CatalogueChunk(
            chunk_id="bearing-6304",
            page_number=4,
            text="Reference 6304-2RS deep groove ball bearing bore 20 mm outside diameter 52 mm width 15 mm double rubber seals",
        ),
        CatalogueChunk(
            chunk_id="sensor-pnp",
            page_number=9,
            text="Reference SEN-M18-PNP-8-M12 inductive proximity sensor M18 PNP normally open sensing distance 8 mm 10-30 V DC M12",
        ),
    ]


def test_exact_industrial_code_ranks_matching_chunk_first() -> None:
    retriever = HybridRetriever(_chunks())

    results = retriever.search("6304-2RS bore 20 mm width 15 mm", top_k=2)

    assert results[0].chunk.chunk_id == "bearing-6304"
    assert results[0].score > results[1].score


def test_natural_technical_query_ranks_matching_product_first() -> None:
    retriever = HybridRetriever(_chunks())

    results = retriever.search(
        "deep groove ball bearing bore 20 mm outside diameter 52 mm width 15 mm double rubber seals",
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "bearing-6304"


def test_unrelated_product_families_do_not_override_matching_attributes() -> None:
    retriever = HybridRetriever(_chunks())

    results = retriever.search(
        "inductive proximity sensor M18 PNP normally open 8 mm supply 10-30 V DC connector M12",
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "sensor-pnp"


def test_exact_token_extraction_keeps_single_digit_technical_values() -> None:
    from rag_catalogue.retrieval import _exact_tokens

    tokens = _exact_tokens("rated current 6 A, curve C, 10 kA")

    assert "6" in tokens
    assert "10" in tokens


def test_allowed_chunk_ids_keep_catalogue_order_when_scores_tie() -> None:
    class ReverseSet(set):
        def __iter__(self):
            return iter(sorted(super().__iter__(), reverse=True))

    chunks = [
        CatalogueChunk(chunk_id="c1", page_number=1, text="same technical text"),
        CatalogueChunk(chunk_id="c2", page_number=2, text="same technical text"),
        CatalogueChunk(chunk_id="c3", page_number=3, text="unrelated"),
    ]
    retriever = HybridRetriever(chunks)

    results = retriever.search(
        "same technical text",
        top_k=2,
        allowed_chunk_ids=ReverseSet({"c1", "c2"}),
    )

    assert [item.chunk.chunk_id for item in results] == ["c1", "c2"]
