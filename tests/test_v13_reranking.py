from __future__ import annotations

from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.retrieval import RankedChunk, rerank_ranked_chunks_batched


def _item(index: int) -> RankedChunk:
    return RankedChunk(
        chunk=CatalogueChunk(
            chunk_id=f"c{index}",
            page_number=index + 1,
            text=f"passage {index}",
        ),
        score=float(100 - index),
        word_score=0.0,
        char_score=0.0,
        exact_score=0.0,
        fused_score=float(100 - index),
    )


class _Reranker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, _query: str, passages: list[str]) -> list[float]:
        self.calls.append(list(passages))
        return [float(int(text.rsplit(" ", 1)[1])) for text in passages]


def test_batched_reranking_bounds_request_size_and_returns_global_top() -> None:
    client = _Reranker()
    metadata: dict[str, object] = {}

    result = rerank_ranked_chunks_batched(
        "query",
        [_item(index) for index in range(25)],
        client,
        top_k=5,
        batch_size=8,
        metadata=metadata,
    )

    assert all(len(call) <= 8 for call in client.calls)
    assert [item.chunk.chunk_id for item in result] == ["c24", "c23", "c22", "c21", "c20"]
    assert metadata["rerank_batches"] >= 4
