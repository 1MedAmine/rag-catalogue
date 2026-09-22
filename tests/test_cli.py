import json
from pathlib import Path

import rag_catalogue_cli as cli


class _FakeClient:
    def __init__(
        self, *, api_key: str, model: str, base_url: str, reasoning_mode: str = "normal"
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.reasoning_mode = reasoning_mode


def test_diagnostic_is_saved_separately_from_compact_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    product = tmp_path / "product.txt"
    catalogue = tmp_path / "catalogue.pdf"
    output = tmp_path / "result.json"
    product.write_text("test product", encoding="utf-8")
    catalogue.write_bytes(b"%PDF-placeholder")

    received: dict[str, object] = {}

    def fake_run_search(
        product_file: Path,
        catalogue_pdf: Path,
        llm: object,
        *,
        top_k: int,
        include_diagnostics: bool,
        **_kwargs,
    ) -> dict:
        received["include_diagnostics"] = include_diagnostics
        return {
            "reference": "AXR2PMC00006",
            "reference_mode": "constructed",
            "reference_parts": [],
            "catalogue_pages": [7],
            "evidence": ["proof"],
            "explanation": "match",
            "validation_errors": [],
            "model": "nvidia/test",
            "diagnostic": {
                "need": {"product_type": "protector"},
                "queries": ["protector 2P 6A"],
                "retrieval": [{"page": 7, "excerpt": "long diagnostic"}],
            },
        }

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _FakeClient)
    monkeypatch.setattr(cli, "run_search", fake_run_search)

    exit_code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--diagnostic",
        "--sortie", str(output),
    ])

    assert exit_code == 0
    assert received["include_diagnostics"] is True
    compact = json.loads(output.read_text(encoding="utf-8"))
    assert "diagnostic" not in compact
    diagnostic_path = tmp_path / "result_diagnostic.json"
    assert compact["diagnostic_file"] == str(diagnostic_path.resolve())
    assert json.loads(diagnostic_path.read_text(encoding="utf-8"))["retrieval"][0]["page"] == 7


def test_main_result_is_compact_by_default(tmp_path: Path, monkeypatch) -> None:
    product = tmp_path / "product.txt"
    catalogue = tmp_path / "catalogue.pdf"
    output = tmp_path / "result.json"
    product.write_text("test product", encoding="utf-8")
    catalogue.write_bytes(b"%PDF-placeholder")

    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _FakeClient)
    monkeypatch.setattr(
        cli,
        "run_search",
        lambda *_args, **_kwargs: {
            "reference": "AXR2PMC00006",
            "reference_mode": "constructed",
            "reference_parts": [{"position": 1, "code": "AXR", "page": 7}],
            "catalogue_pages": [7],
            "evidence": ["proof"],
            "explanation": "match",
            "validation_errors": [],
            "model": "nvidia/test",
        },
    )

    exit_code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--sortie", str(output),
    ])

    assert exit_code == 0
    compact = json.loads(output.read_text(encoding="utf-8"))
    assert compact == {
        "status": "found",
        "reference": "AXR2PMC00006",
        "catalogue_pages": [7],
        "explanation": "match",
        "validation_errors": [],
        "model": "nvidia/test",
    }


def test_details_flag_keeps_proof_fields(tmp_path: Path, monkeypatch) -> None:
    product = tmp_path / "product.txt"
    catalogue = tmp_path / "catalogue.pdf"
    output = tmp_path / "result.json"
    product.write_text("test product", encoding="utf-8")
    catalogue.write_bytes(b"%PDF-placeholder")

    full_result = {
        "reference": "AXR2PMC00006",
        "reference_mode": "constructed",
        "reference_parts": [{"position": 1, "code": "AXR", "page": 7}],
        "catalogue_pages": [7],
        "evidence": ["proof"],
        "explanation": "match",
        "validation_errors": [],
        "model": "nvidia/test",
    }
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(cli, "NvidiaChatClient", _FakeClient)
    monkeypatch.setattr(cli, "run_search", lambda *_args, **_kwargs: dict(full_result))

    exit_code = cli.main([
        "chercher",
        "--fiche", str(product),
        "--catalogue", str(catalogue),
        "--details",
        "--sortie", str(output),
    ])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == full_result


def test_search_cli_accepts_provenance_companion_options() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        "chercher",
        "--fiche", "cahier.txt",
        "--provenance", "provenance.json",
        "--catalogue", "catalogue.pdf",
        "--sans-provenance-auto",
    ])
    assert args.provenance == Path("provenance.json")
    assert args.sans_provenance_auto is True


def test_compact_result_keeps_ranked_candidates() -> None:
    result = {
        "reference": "HGD63H2PMCS0000C00003",
        "catalogue_pages": [41],
        "explanation": "Two relevant variants.",
        "validation_errors": [],
        "model": "nvidia/test",
        "candidates": [
            {
                "rank": 1,
                "reference": "HGD63H2PMCS0000C00003",
                "variant": "Deluxe",
                "match_level": "closest",
                "catalogue_pages": [41],
                "reason": "Closest documented variant.",
            },
            {
                "rank": 2,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": "Standard",
                "match_level": "alternative",
                "catalogue_pages": [42],
                "reason": "Compatible standard alternative.",
            },
        ],
    }

    compact = cli._compact_result(result)

    assert compact["status"] == "multiple_candidates"
    assert compact["reference"] == "HGD63H2PMCS0000C00003"
    assert [item["rank"] for item in compact["candidates"]] == [1, 2]
    assert compact["candidates"][1]["variant"] == "Standard"


def test_reasoning_mode_defaults_to_normal_and_accepts_all_profiles() -> None:
    parser = cli._build_parser()
    base = ["chercher", "--fiche", "fiche.txt", "--catalogue", "catalogue.pdf"]

    assert parser.parse_args(base).raisonnement == "normal"
    assert parser.parse_args(base + ["--raisonnement", "rapide"]).raisonnement == "rapide"
    assert parser.parse_args(base + ["--raisonnement", "approfondi"]).raisonnement == "approfondi"


def test_compact_result_marks_found_when_only_one_final_candidate_remains() -> None:
    result = {
        "reference": "REF-DEMO-032PMCS0000C00003",
        "catalogue_pages": [42],
        "explanation": "Un seul candidat valide a ete conserve.",
        "validation_errors": [],
        "model": "nvidia/test",
        "candidates": [
            {
                "rank": 1,
                "reference": "REF-DEMO-032PMCS0000C00003",
                "variant": "Standard",
                "match_level": "closest",
                "catalogue_pages": [42],
                "reason": "Correspondance validee.",
                "differences": [],
            }
        ],
    }

    compact = cli._compact_result(result)

    assert compact["status"] == "found"
    assert compact["reference"] == "REF-DEMO-032PMCS0000C00003"
    assert len(compact["candidates"]) == 1
