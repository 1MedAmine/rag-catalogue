from pathlib import Path

import rag_catalogue_cli as cli


class _FakeClient:
    def __init__(
        self, *, api_key: str, model: str, base_url: str, reasoning_mode: str = "normal"
    ) -> None:
        self.model = model
        self.reasoning_mode = reasoning_mode


class _FakeEmbedding:
    model = "embed"
    def __init__(self, **_kwargs) -> None:
        pass


class _FakeRerank:
    model = "rerank"
    def __init__(self, **_kwargs) -> None:
        pass


def _result() -> dict:
    return {
        "reference": None,
        "reference_mode": None,
        "reference_parts": [],
        "catalogue_pages": [],
        "evidence": [],
        "explanation": "not found",
        "validation_errors": [],
        "model": "fake",
    }


def test_cli_passes_explicit_provenance_and_enables_structural_validation_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cahier = tmp_path / "cahier.txt"
    provenance = tmp_path / "provenance.json"
    catalogue = tmp_path / "catalogue.pdf"
    output = tmp_path / "result.json"
    cahier.write_text("cahier", encoding="utf-8")
    provenance.write_text('{"contraintes_client":["2P"]}', encoding="utf-8")
    catalogue.write_bytes(b"%PDF")
    received = {}

    def fake_run_search(*_args, **kwargs):
        received.update(kwargs)
        return _result()

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _FakeClient)
    monkeypatch.setattr(cli, "NvidiaEmbeddingClient", _FakeEmbedding)
    monkeypatch.setattr(cli, "NvidiaRerankClient", _FakeRerank)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    code = cli.main([
        "chercher",
        "--fiche", str(cahier),
        "--provenance", str(provenance),
        "--catalogue", str(catalogue),
        "--sortie", str(output),
    ])

    assert code == 0
    assert received["provenance_file"] == provenance
    assert received["auto_discover_provenance"] is True
    assert received["validate_result"] is True


def test_cli_can_disable_auto_discovery_and_structural_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cahier = tmp_path / "cahier.txt"
    catalogue = tmp_path / "catalogue.pdf"
    cahier.write_text("cahier", encoding="utf-8")
    catalogue.write_bytes(b"%PDF")
    received = {}

    def fake_run_search(*_args, **kwargs):
        received.update(kwargs)
        return _result()

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _FakeClient)
    monkeypatch.setattr(cli, "NvidiaEmbeddingClient", _FakeEmbedding)
    monkeypatch.setattr(cli, "NvidiaRerankClient", _FakeRerank)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    code = cli.main([
        "chercher",
        "--fiche", str(cahier),
        "--catalogue", str(catalogue),
        "--sans-provenance-auto",
        "--sans-validation-preuves",
    ])

    assert code == 0
    assert received["provenance_file"] is None
    assert received["auto_discover_provenance"] is False
    assert received["validate_result"] is False
