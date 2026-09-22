from pathlib import Path

import rag_catalogue_cli as cli


def test_cli_search_accepts_v13_large_catalogue_options() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        "chercher",
        "--fiche", "fiche.txt",
        "--catalogue", "catalogue.pdf",
        "--ocr", "auto",
        "--ocr-lang", "eng",
        "--ocr-max-pages", "9",
        "--sans-hierarchie",
        "--profil-catalogue", "huge",
    ])

    assert args.ocr == "auto"
    assert args.ocr_lang == "eng"
    assert args.ocr_max_pages == 9
    assert args.sans_hierarchie is True
    assert args.profil_catalogue == "huge"


def test_cli_exposes_inspect_and_index_commands() -> None:
    parser = cli._build_parser()
    inspect_args = parser.parse_args(["inspecter", "--catalogue", "catalogue.pdf"])
    index_args = parser.parse_args(["indexer", "--catalogue", "catalogue.pdf"])

    assert inspect_args.command == "inspecter"
    assert index_args.command == "indexer"


def test_cli_wires_v13_options_to_run_search(tmp_path, monkeypatch) -> None:
    product = tmp_path / "fiche.txt"
    catalogue = tmp_path / "catalogue.pdf"
    product.write_text("pompe", encoding="utf-8")
    catalogue.write_bytes(b"%PDF")
    received = {}

    class Generation:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

    class Embedding:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

    class Rerank:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

    def fake_run_search(*_args, **kwargs):
        received.update(kwargs)
        return {
            "reference": None,
            "catalogue_pages": [],
            "explanation": "",
            "validation_errors": [],
            "model": "test",
        }

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", Generation)
    monkeypatch.setattr(cli, "NvidiaEmbeddingClient", Embedding)
    monkeypatch.setattr(cli, "NvidiaRerankClient", Rerank)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--ocr", "auto",
        "--ocr-lang", "fra+eng",
        "--ocr-max-pages", "7",
        "--profil-catalogue", "large",
        "--section-k", "6",
    ])

    assert code == 0
    assert received["ocr_mode"] == "auto"
    assert received["ocr_language"] == "fra+eng"
    assert received["ocr_max_pages"] == 7
    assert received["catalogue_profile"] == "large"
    assert received["section_k"] == 6
    assert received["hierarchy_enabled"] is True
