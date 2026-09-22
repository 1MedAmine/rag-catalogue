from __future__ import annotations

from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.retrieval import RankedChunk, reciprocal_rank_fusion, rerank_ranked_chunks


def _ranked(chunk_id: str, score: float, *, page: int = 1, dense: float = 0.0) -> RankedChunk:
    return RankedChunk(
        chunk=CatalogueChunk(chunk_id, page, f"text {chunk_id}"),
        score=score,
        word_score=score if not dense else 0.0,
        char_score=0.0,
        exact_score=0.0,
        dense_score=dense,
    )


def test_rrf_keeps_dense_only_candidate_and_tracks_stage_ranks() -> None:
    lexical = [_ranked("lexical", 0.9)]
    dense = [_ranked("semantic", 0.95, page=2, dense=0.95)]

    fused = reciprocal_rank_fusion(lexical, dense, top_k=2, rrf_k=10)

    assert {item.chunk.chunk_id for item in fused} == {"lexical", "semantic"}
    by_id = {item.chunk.chunk_id: item for item in fused}
    assert by_id["lexical"].lexical_rank == 1
    assert by_id["lexical"].dense_rank is None
    assert by_id["semantic"].lexical_rank is None
    assert by_id["semantic"].dense_rank == 1


class _Reranker:
    model = "fake/reranker"

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        assert query == "query"
        return [1.0, 9.0, 3.0]


def test_reranking_changes_final_order() -> None:
    candidates = [_ranked("a", 0.9), _ranked("b", 0.8), _ranked("c", 0.7)]

    result = rerank_ranked_chunks("query", candidates, _Reranker(), top_k=2)

    assert [item.chunk.chunk_id for item in result] == ["b", "c"]
    assert result[0].rerank_score == 9.0
