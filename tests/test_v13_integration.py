from __future__ import annotations

from pathlib import Path

import fitz

from rag_catalogue import pipeline


def _write_pdf(path: Path, pages: list[str], toc: list[list[object]] | None = None) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox((50, 50, 550, 780), text, fontsize=10)
    if toc:
        document.set_toc(toc)
    document.save(path)
    document.close()


def test_v13_retrieval_reports_hierarchy_and_chunk_adaptive_profile(tmp_path: Path) -> None:
    catalogue = tmp_path / "catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "POMPES CENTRIFUGES specifications debit 20 m3/h hauteur 30 m",
            "Selection table pump PX20 flow 20 m3/h head 30 m",
            "Ordering Information pump PX 20 M code reference",
            "CAPTEURS INDUSTRIELS inductive sensor 24 V M12",
            "Selection table sensor SX24 M12",
        ],
        [[1, "Pompes", 1], [1, "Capteurs", 4]],
    )
    metadata: dict[str, object] = {}

    results = pipeline.retrieve_catalogue_chunks(
        catalogue,
        "pompe centrifuge debit 20 m3/h hauteur 30 m",
        top_k=3,
        candidate_k=12,
        retrieval_metadata=metadata,
        hierarchical=True,
        catalogue_profile="auto",
    )

    assert results
    assert metadata["hierarchical_enabled"] is True
    assert metadata["catalogue_section_count"] == 2
    assert metadata["catalogue_chunk_count"] >= 5
    assert metadata["selected_sections"]
    assert any(item.chunk.section_title == "Pompes" for item in results)
    assert any(item.chunk.kind == "ordering" for item in results)


def test_v13_context_quality_deduplicates_before_final_context(tmp_path: Path) -> None:
    catalogue = tmp_path / "duplicates.pdf"
    repeated = "Pompe centrifuge debit 20 m3/h hauteur 30 m moteur 4 kW"
    _write_pdf(catalogue, [repeated, repeated + ".", "Ordering Information PX 20 M"])
    metadata: dict[str, object] = {}

    results = pipeline.retrieve_catalogue_chunks(
        catalogue,
        repeated,
        top_k=3,
        candidate_k=12,
        retrieval_metadata=metadata,
        hierarchical=True,
    )

    assert results
    assert metadata["deduplicated_chunks"] >= 1
    result_ids = [item.chunk.chunk_id for item in results]
    assert not ({"p1-c1", "p2-c1"} <= set(result_ids))


class _Embedding:
    model = "nvidia/test-embed"

    def __init__(self) -> None:
        self.passage_calls = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls += 1
        return [[1.0, float(index + 1)] for index, _text in enumerate(texts)]

    def embed_query(self, _text: str) -> list[float]:
        return [1.0, 1.0]


def test_v13_prepare_catalogue_index_builds_dense_cache_and_map(tmp_path: Path) -> None:
    catalogue = tmp_path / "index.pdf"
    _write_pdf(catalogue, ["Pompes centrifuges", "Ordering Information PX 20 M"], [[1, "Pompes", 1]])
    embedding = _Embedding()

    prepare = getattr(pipeline, "prepare_catalogue_index", None)
    assert callable(prepare)
    report = prepare(
        catalogue,
        embedding_client=embedding,
        cache_dir=tmp_path / "cache",
        hierarchical=True,
    )

    assert embedding.passage_calls == 1
    assert report["page_count"] == 2
    assert report["section_count"] == 1
    assert report["chunk_count"] >= 2
    assert report["dense_cache_hit"] is False
