from __future__ import annotations

from pathlib import Path

import fitz

from rag_catalogue.pipeline import retrieve_catalogue_chunks, run_search


def _write_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox((50, 50, 550, 780), text, fontsize=10)
    document.save(path)
    document.close()


class _Embedding:
    model = "nvidia/test-embed"

    def __init__(self) -> None:
        self.passage_calls = 0
        self.query_calls: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls += 1
        return [
            [1.0, 0.0] if "semantic target" in text.casefold() else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]


class _Reranker:
    model = "nvidia/test-rerank"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [10.0 if "preferred passage" in text.casefold() else 0.0 for text in passages]


def test_dense_retrieval_adds_semantic_candidate(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "alpha beta exact lexical item",
            "semantic target described with unrelated vocabulary",
        ],
    )
    embedding = _Embedding()

    results = retrieve_catalogue_chunks(
        catalogue,
        "alpha beta conceptual request",
        top_k=2,
        candidate_k=4,
        embedding_client=embedding,
        cache_dir=tmp_path / "cache",
    )

    assert embedding.passage_calls == 1
    assert embedding.query_calls == ["alpha beta conceptual request"]
    semantic = next(item for item in results if item.chunk.page_number == 2)
    assert semantic.dense_score > 0.9
    assert semantic.dense_rank is not None


def test_reranker_controls_final_context_order(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "industrial component exact query terms",
            "preferred passage for the same industrial component",
        ],
    )
    reranker = _Reranker()

    results = retrieve_catalogue_chunks(
        catalogue,
        "industrial component",
        top_k=1,
        candidate_k=4,
        rerank_client=reranker,
    )

    assert reranker.calls
    assert results[0].chunk.page_number == 2
    assert results[0].rerank_score == 10.0


class _LLM:
    model = "nvidia/test-generation"

    def extract_need(self, _text: str) -> dict:
        return {
            "product_type": "industrial component",
            "source_reference": None,
            "attributes": [],
            "search_queries": ["industrial component"],
        }

    def select_reference(self, _need: dict, chunks: list) -> dict:
        page = chunks[0].chunk.page_number
        return {
            "reference": "REF-1",
            "reference_mode": "explicit",
            "reference_parts": [],
            "catalogue_pages": [page],
            "evidence": ["Reference REF-1"],
            "explanation": "explicit reference",
        }


def test_diagnostic_reports_all_models_and_stage_scores(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("industrial component", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(catalogue, ["Reference REF-1 industrial component preferred passage"])
    embedding = _Embedding()
    reranker = _Reranker()

    result = run_search(
        product,
        catalogue,
        _LLM(),
        top_k=1,
        candidate_k=4,
        embedding_client=embedding,
        rerank_client=reranker,
        cache_dir=tmp_path / "cache",
        include_diagnostics=True,
    )

    assert result["diagnostic"]["models"] == {
        "embedding": "nvidia/test-embed",
        "reranking": "nvidia/test-rerank",
        "generation": "nvidia/test-generation",
    }
    item = result["diagnostic"]["retrieval"][0]
    assert "dense_score" in item
    assert "fused_score" in item
    assert "rerank_score" in item


def test_adaptive_retrieval_depth_increases_every_100_pages() -> None:
    from rag_catalogue.pipeline import adaptive_retrieval_depth

    assert adaptive_retrieval_depth(100, top_k=8, candidate_k=32) == {
        "page_count": 100,
        "depth_steps": 0,
        "top_k": 8,
        "candidate_k": 32,
    }
    assert adaptive_retrieval_depth(101, top_k=8, candidate_k=32) == {
        "page_count": 101,
        "depth_steps": 1,
        "top_k": 10,
        "candidate_k": 48,
    }
    assert adaptive_retrieval_depth(250, top_k=8, candidate_k=32) == {
        "page_count": 250,
        "depth_steps": 2,
        "top_k": 12,
        "candidate_k": 64,
    }
    assert adaptive_retrieval_depth(1000, top_k=8, candidate_k=32) == {
        "page_count": 1000,
        "depth_steps": 9,
        "top_k": 20,
        "candidate_k": 128,
    }


def test_diagnostic_reports_adaptive_depth_for_large_catalogue(tmp_path: Path) -> None:
    product = tmp_path / "product.txt"
    product.write_text("industrial component", encoding="utf-8")
    catalogue = tmp_path / "catalogue.pdf"
    pages = ["irrelevant page"] * 100 + [
        "Reference REF-1 industrial component preferred passage"
    ]
    _write_pdf(catalogue, pages)

    result = run_search(
        product,
        catalogue,
        _LLM(),
        top_k=1,
        candidate_k=4,
        include_diagnostics=True,
    )

    metadata = result["diagnostic"]["retrieval_metadata"]
    assert metadata["catalogue_page_count"] == 101
    assert metadata["adaptive_depth_steps"] == 1
    assert metadata["requested_top_k"] == 1
    assert metadata["effective_top_k"] == 3
    assert metadata["requested_candidate_k"] == 4
    assert metadata["effective_candidate_k"] == 20
