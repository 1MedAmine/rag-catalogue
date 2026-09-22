from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_catalogue.dense_retrieval import DenseIndex, load_or_create_dense_index
from rag_catalogue.pdf_text import CatalogueChunk


class _Embedder:
    model = "fake/embedder"

    def __init__(self) -> None:
        self.calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        mapping = {
            "hydraulic liquid mover": [1.0, 0.0],
            "electrical protection device": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


def _chunks() -> list[CatalogueChunk]:
    return [
        CatalogueChunk("pump", 1, "hydraulic liquid mover"),
        CatalogueChunk("breaker", 2, "electrical protection device"),
    ]


def test_dense_index_orders_by_cosine() -> None:
    index = DenseIndex(_chunks(), np.asarray([[1.0, 0.0], [0.0, 1.0]]))

    results = index.search([0.9, 0.1], top_k=2)

    assert [item.chunk.chunk_id for item in results] == ["pump", "breaker"]
    assert results[0].score > results[1].score


def test_dense_index_rejects_query_dimension_mismatch() -> None:
    index = DenseIndex(_chunks(), np.asarray([[1.0, 0.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="dimension"):
        index.search([1.0, 0.0, 0.0], top_k=1)


def test_dense_cache_reuses_passage_vectors(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.pdf"
    catalogue.write_bytes(b"fake-pdf")
    embedder = _Embedder()

    first, first_hit = load_or_create_dense_index(
        catalogue,
        _chunks(),
        embedder,
        cache_dir=tmp_path / "cache",
        max_words=260,
        overlap_words=50,
    )
    second, second_hit = load_or_create_dense_index(
        catalogue,
        _chunks(),
        embedder,
        cache_dir=tmp_path / "cache",
        max_words=260,
        overlap_words=50,
    )

    assert first_hit is False
    assert second_hit is True
    assert embedder.calls == 1
    assert np.allclose(first.matrix, second.matrix)
