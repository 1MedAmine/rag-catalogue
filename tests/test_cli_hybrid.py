from __future__ import annotations

from pathlib import Path

import rag_catalogue_cli as cli


class _Generation:
    def __init__(self, **kwargs) -> None:
        self.model = kwargs["model"]
        self.kwargs = kwargs


class _Embedding:
    def __init__(self, **kwargs) -> None:
        self.model = kwargs["model"]
        self.kwargs = kwargs


class _Reranker:
    def __init__(self, **kwargs) -> None:
        self.model = kwargs["model"]
        self.kwargs = kwargs


def test_cli_wires_embedding_reranking_and_generation_independently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    product = tmp_path / "product.txt"
    catalogue = tmp_path / "catalogue.pdf"
    output = tmp_path / "result.json"
    product.write_text("product", encoding="utf-8")
    catalogue.write_bytes(b"%PDF")
    received: dict[str, object] = {}

    def fake_run_search(product_file, catalogue_pdf, llm, **kwargs):
        received.update(kwargs)
        received["llm"] = llm
        return {
            "reference": None,
            "catalogue_pages": [],
            "evidence": [],
            "explanation": "none",
            "validation_errors": [],
            "model": llm.model,
            "models": {
                "embedding": kwargs["embedding_client"].model,
                "reranking": kwargs["rerank_client"].model,
                "generation": llm.model,
            },
        }

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _Generation)
    monkeypatch.setattr(cli, "NvidiaEmbeddingClient", _Embedding)
    monkeypatch.setattr(cli, "NvidiaRerankClient", _Reranker)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--modele", "generation/model",
        "--raisonnement", "approfondi",
        "--modele-embedding", "embedding/model",
        "--modele-reranking", "rerank/model",
        "--embedding-url", "https://embed.test/v1/embeddings",
        "--rerank-url", "https://rank.test/reranking",
        "--candidate-k", "20",
        "--cache-dir", str(tmp_path / "cache"),
        "--sortie", str(output),
    ])

    assert code == 0
    assert received["llm"].model == "generation/model"
    assert received["llm"].kwargs["reasoning_mode"] == "approfondi"
    assert received["embedding_client"].model == "embedding/model"
    assert received["rerank_client"].model == "rerank/model"
    assert received["candidate_k"] == 20
    assert received["cache_dir"] == tmp_path / "cache"
    assert received["embedding_client"].kwargs["api_url"] == "https://embed.test/v1/embeddings"
    assert received["rerank_client"].kwargs["api_url"] == "https://rank.test/reranking"
    assert received["validate_result"] is True


def test_cli_can_disable_dense_and_reranking(tmp_path: Path, monkeypatch) -> None:
    product = tmp_path / "product.txt"
    catalogue = tmp_path / "catalogue.pdf"
    product.write_text("product", encoding="utf-8")
    catalogue.write_bytes(b"%PDF")
    received: dict[str, object] = {}

    def fake_run_search(*_args, **kwargs):
        received.update(kwargs)
        return {
            "reference": None,
            "catalogue_pages": [],
            "evidence": [],
            "explanation": "none",
            "validation_errors": [],
            "model": "generation/model",
        }

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _Generation)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--sans-embedding",
        "--sans-reranking",
    ])

    assert code == 0
    assert received["embedding_client"] is None
    assert received["rerank_client"] is None


def test_cli_enables_optional_proof_validation(tmp_path: Path, monkeypatch) -> None:
    product = tmp_path / "product.txt"
    catalogue = tmp_path / "catalogue.pdf"
    product.write_text("product", encoding="utf-8")
    catalogue.write_bytes(b"%PDF")
    received: dict[str, object] = {}

    def fake_run_search(*_args, **kwargs):
        received.update(kwargs)
        return {
            "reference": None,
            "catalogue_pages": [],
            "evidence": [],
            "explanation": "none",
            "validation_errors": [],
            "model": "generation/model",
        }

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _Generation)
    monkeypatch.setattr(cli, "NvidiaEmbeddingClient", _Embedding)
    monkeypatch.setattr(cli, "NvidiaRerankClient", _Reranker)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--validation-preuves",
    ])

    assert code == 0
    assert received["validate_result"] is True
