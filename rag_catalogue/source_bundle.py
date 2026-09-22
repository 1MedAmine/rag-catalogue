"""Assemblage du cahier complet et de sa provenance structuree."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .source_text import extract_product_text
from provenance_v4 import migrer_provenance_legacy


_PROVENANCE_KEYS = (
    "schema_version",
    "type_demande",
    "identification_produit",
    "demande_client",
    "demande_client_brute",
    "produit_source",
    "contraintes_explicites",
    "contraintes_derivees_obligatoires",
    "criteres_classement",
    "contraintes_client",
    "caracteristiques_produit",
    "caracteristiques_source",
    "criteres_equivalence",
    "capacites_optionnelles",
    "informations_a_confirmer",
    "sources",
)


def _canonical_rag_provenance(raw: dict[str, Any]) -> dict[str, Any]:
    """Convertit V3.1/V4 en attributs stables compris par le RAG."""

    migrated = migrer_provenance_legacy(raw)
    attributes: list[dict[str, Any]] = []
    mappings = (
        ("contraintes_explicites", "constraint", "client"),
        ("contraintes_derivees_obligatoires", "constraint", "derived_required"),
        ("criteres_classement", "selector", "source_product"),
        ("capacites_optionnelles", "context", "optional"),
        ("informations_a_confirmer", "context", "to_confirm"),
    )
    for key, role, provenance in mappings:
        for item in migrated.get(key, []):
            if not isinstance(item, dict):
                continue
            value = str(item.get("valeur") or "").strip()
            name = str(item.get("champ") or "caracteristique").strip()
            if not value:
                continue
            attributes.append({
                "name": name,
                "value": value,
                "unit": item.get("unite"),
                "role": role,
                "provenance": provenance,
                "evidence": item.get("preuve"),
                "proofs": item.get("preuves", []),
                "proof_relation": item.get("relation_preuve"),
                "source": item.get("source"),
                "confidence": item.get("confiance"),
                "criticality": item.get("criticalite"),
            })
    return {
        "schema_version": "4.0",
        "request_type": migrated.get("type_demande", "indeterminee"),
        "source_identity": migrated.get("identification_produit", {}),
        "attributes": attributes,
        "open_questions": [
            str(item.get("valeur") or "").strip()
            for item in migrated.get("informations_a_confirmer", [])
            if isinstance(item, dict) and str(item.get("valeur") or "").strip()
        ],
    }


@dataclass(frozen=True)
class ProductSource:
    """Source primaire complete et compagnon de provenance facultatif."""

    product_path: Path
    primary_text: str
    provenance_path: Path | None = None
    provenance: dict[str, Any] | None = None

    def render_for_llm(self) -> str:
        """Rend le cahier integral puis le bloc structure, sans substitution."""

        if not self.provenance:
            return self.primary_text
        rendered = json.dumps(
            self.provenance,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        canonical = json.dumps(
            _canonical_rag_provenance(self.provenance),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        return (
            f"{self.primary_text}\n\n"
            "=== PROVENANCE STRUCTUREE (COMPLEMENT DU CAHIER, PAS UN REMPLACEMENT) ===\n"
            "Le cahier complet ci-dessus reste la source technique principale. "
            "Ce JSON sert uniquement a distinguer l'origine et le statut des valeurs.\n"
            f"{rendered}\n\n"
            "=== PROVENANCE CANONIQUE POUR LE RAG ===\n"
            "Les attributs ci-dessous sont une vue normalisee et non une nouvelle source.\n"
            f"{canonical}"
        )


def _suffix_from_cahier_stem(stem: str) -> str:
    prefixes = ("cahier_", "cahier-", "fiche_", "fiche-")
    folded = stem.casefold()
    for prefix in prefixes:
        if folded.startswith(prefix):
            return stem[len(prefix):]
    return stem


def discover_provenance_file(product_file: Path | str) -> Path | None:
    """Trouve un compagnon riche et previsible place a cote du cahier."""

    product_path = Path(product_file)
    parent = product_path.parent
    suffix = _suffix_from_cahier_stem(product_path.stem)
    candidates = [
        parent / f"provenance_{suffix}.json",
        parent / f"provenance-{suffix}.json",
        parent / f"entree_rag_{suffix}.json",
        parent / f"entree-rag-{suffix}.json",
        parent / f"{product_path.stem}_provenance.json",
        parent / f"{product_path.stem}.provenance.json",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and candidate.resolve() != product_path.resolve():
            return candidate
    return None


def _load_provenance(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON de provenance illisible en UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON de provenance invalide: {path}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Le JSON de provenance doit etre un objet: {path}")

    # Conserver les extensions utiles, mais verifier qu'au moins une cle connue
    # existe afin d'eviter d'annexer accidentellement un resultat sans rapport.
    if not any(key in raw for key in _PROVENANCE_KEYS):
        raise ValueError(
            "Le JSON compagnon ne contient aucune cle de provenance reconnue: "
            f"{path}"
        )
    return raw


def load_product_source(
    product_file: Path | str,
    *,
    provenance_file: Path | str | None = None,
    auto_discover: bool = True,
) -> ProductSource:
    """Charge le cahier complet et, si disponible, son JSON compagnon."""

    product_path = Path(product_file)
    primary_text = extract_product_text(product_path)
    provenance_path: Path | None
    if provenance_file is not None:
        provenance_path = Path(provenance_file)
        if not provenance_path.is_file():
            raise FileNotFoundError(f"JSON de provenance introuvable: {provenance_path}")
    elif auto_discover:
        provenance_path = discover_provenance_file(product_path)
    else:
        provenance_path = None

    provenance = _load_provenance(provenance_path) if provenance_path else None
    return ProductSource(
        product_path=product_path,
        primary_text=primary_text,
        provenance_path=provenance_path,
        provenance=provenance,
    )
