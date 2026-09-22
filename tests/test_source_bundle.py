import json
from pathlib import Path

from rag_catalogue.source_bundle import discover_provenance_file, load_product_source


def test_explicit_provenance_is_appended_without_replacing_full_cahier(tmp_path: Path) -> None:
    cahier = tmp_path / "cahier_produit_1.txt"
    provenance = tmp_path / "provenance_produit_1.json"
    cahier.write_text(
        "CAHIER COMPLET\nProduit source M9F11203\nDimensions et normes documentées.",
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps({
            "demande_client": "MCB 2P 3A Type C 10KA Schneider",
            "contraintes_client": ["2P", "3A", "Type C", "10KA"],
            "caracteristiques_produit": ["415 V AC", "125 V DC"],
            "informations_a_confirmer": ["tension d'utilisation AC ou DC"],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    source = load_product_source(cahier, provenance_file=provenance)
    rendered = source.render_for_llm()

    assert "CAHIER COMPLET" in rendered
    assert "Dimensions et normes documentées" in rendered
    assert "PROVENANCE STRUCTUREE" in rendered
    assert "MCB 2P 3A Type C 10KA Schneider" in rendered
    assert "415 V AC" in rendered
    assert source.provenance_path == provenance


def test_auto_discovery_prefers_rich_provenance_companion(tmp_path: Path) -> None:
    cahier = tmp_path / "cahier_produit_1.txt"
    rich = tmp_path / "provenance_produit_1.json"
    minimal = tmp_path / "entree_rag_produit_1.json"
    cahier.write_text("cahier", encoding="utf-8")
    rich.write_text('{"contraintes_client":["2P"],"caracteristiques_produit":["3A"]}', encoding="utf-8")
    minimal.write_text('{"contraintes_client":["2P"]}', encoding="utf-8")

    assert discover_provenance_file(cahier) == rich
    source = load_product_source(cahier)
    assert source.provenance_path == rich
    assert "3A" in source.render_for_llm()


def test_auto_discovery_can_be_disabled(tmp_path: Path) -> None:
    cahier = tmp_path / "cahier_produit_1.txt"
    provenance = tmp_path / "provenance_produit_1.json"
    cahier.write_text("cahier complet", encoding="utf-8")
    provenance.write_text('{"contraintes_client":["2P"]}', encoding="utf-8")

    source = load_product_source(cahier, auto_discover=False)

    assert source.provenance_path is None
    assert "PROVENANCE STRUCTUREE" not in source.render_for_llm()


def test_v4_provenance_is_rendered_as_canonical_rag_attributes(tmp_path: Path) -> None:
    cahier = tmp_path / "cahier_reference.txt"
    provenance = tmp_path / "provenance_reference.json"
    cahier.write_text("Cahier complet du produit source.", encoding="utf-8")
    preuve = "REF-400 inverseur manuel 400 A 4 pôles"
    provenance.write_text(
        json.dumps({
            "schema_version": "4.0",
            "type_demande": "reference_exacte",
            "identification_produit": {
                "statut": "confirmee",
                "fabricant": "Fabricant source",
                "reference": "REF-400",
                "preuve": preuve,
                "source": "https://fabricant.example/ref-400.pdf",
            },
            "contraintes_explicites": [
                {"champ": "nombre_de_poles", "valeur": "4P", "preuve": "4P", "confiance": "elevee"}
            ],
            "contraintes_derivees_obligatoires": [
                {"champ": "fonction", "valeur": "inverseur manuel", "preuve": preuve, "confiance": "elevee", "criticalite": "fonction"},
                {"champ": "courant_nominal", "valeur": "400 A", "preuve": preuve, "confiance": "elevee"},
            ],
            "criteres_classement": [
                {"champ": "nombre_de_poles", "valeur": "4 pôles", "preuve": preuve, "confiance": "elevee"}
            ],
            "capacites_optionnelles": [],
            "informations_a_confirmer": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    rendered = load_product_source(cahier, provenance_file=provenance).render_for_llm()

    assert "PROVENANCE CANONIQUE POUR LE RAG" in rendered
    assert '"provenance": "derived_required"' in rendered
    assert '"role": "constraint"' in rendered
    assert '"criticality": "fonction"' in rendered
    assert '"provenance": "source_product"' in rendered
    assert '"role": "selector"' in rendered
