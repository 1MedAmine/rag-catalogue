"""Schéma canonique et durcissement déterministe de la provenance produit V4.

Le module reste générique : il ne connaît aucune marque, famille ou référence.
Il distingue les exigences explicites du client des contraintes techniques
certaines dérivées d'une référence exacte, puis conserve le reste comme
critères de classement, options ou informations à confirmer.
"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from typing import Any, Iterable


SCHEMA_VERSION = "4.0"
REQUEST_TYPES = {"reference_exacte", "description", "mixte", "indeterminee"}
IDENTITY_STATUSES = {"confirmee", "ambigue", "non_verifiee"}
CONFIDENCE_LEVELS = {"elevee", "moyenne", "faible"}
STRUCTURED_KEYS = (
    "contraintes_explicites",
    "contraintes_derivees_obligatoires",
    "criteres_classement",
    "capacites_optionnelles",
    "informations_a_confirmer",
)
CRITICALITIES = {
    "fonction",
    "configuration",
    "interface",
    "performance",
    "conformite",
    "securite",
    "secondaire",
    "indeterminee",
}
PROOF_RELATIONS = {
    "meme_ligne",
    "meme_colonne",
    "meme_tableau",
    "meme_bloc_produit",
    "codification_reference",
    "continuation_tableau",
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _fold(value: object) -> str:
    return unicodedata.normalize("NFKC", _clean(value)).casefold()


def _components(value: object) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.findall(r"\d+(?:[.,]\d+)?|[^\W\d_]+", text, re.UNICODE)


def _attested(value: object, source: object) -> bool:
    """Presence technique avec frontières alphanumériques et séparateurs libres."""

    components = _components(value)
    if not components:
        return False

    def component_pattern(component: str) -> str:
        decimal = re.fullmatch(r"(\d+)[.,](\d+)", component)
        if decimal:
            return rf"{re.escape(decimal.group(1))}[.,]{re.escape(decimal.group(2))}"
        return re.escape(component)

    body = r"[\W_]*".join(component_pattern(item) for item in components)
    pattern = re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE | re.UNICODE)
    normalised_source = unicodedata.normalize("NFKC", str(source or "")).casefold()
    return pattern.search(normalised_source) is not None


def _proof_attested(proof: object, documentation: object) -> bool:
    proof_text = _fold(proof)
    if not proof_text:
        return False
    documentation_text = _fold(documentation)
    return proof_text in documentation_text


def _normalise_request_type(value: object) -> str:
    folded = _fold(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "reference": "reference_exacte",
        "reference_seule": "reference_exacte",
        "recherche_par_reference": "reference_exacte",
        "exact_reference": "reference_exacte",
        "reference_exacte": "reference_exacte",
        "descriptive": "description",
        "description": "description",
        "mixed": "mixte",
        "mixte": "mixte",
        "unknown": "indeterminee",
        "indetermine": "indeterminee",
        "indeterminee": "indeterminee",
    }
    result = aliases.get(folded, folded)
    return result if result in REQUEST_TYPES else "indeterminee"


def _normalise_identity_status(value: object) -> str:
    folded = _fold(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "confirmed": "confirmee",
        "confirme": "confirmee",
        "confirmee": "confirmee",
        "exacte": "confirmee",
        "exact": "confirmee",
        "ambiguous": "ambigue",
        "ambigu": "ambigue",
        "ambigue": "ambigue",
        "unverified": "non_verifiee",
        "non_verifie": "non_verifiee",
        "non_verifiee": "non_verifiee",
        "unknown": "non_verifiee",
    }
    result = aliases.get(folded, folded)
    return result if result in IDENTITY_STATUSES else "non_verifiee"


def _normalise_confidence(value: object) -> str:
    folded = _fold(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "high": "elevee",
        "haute": "elevee",
        "eleve": "elevee",
        "élevée": "elevee",
        "elevee": "elevee",
        "medium": "moyenne",
        "moyen": "moyenne",
        "moyenne": "moyenne",
        "low": "faible",
        "faible": "faible",
    }
    result = aliases.get(folded, folded)
    return result if result in CONFIDENCE_LEVELS else "moyenne"


def _normalise_criticality(value: object, field: object = "") -> str:
    folded = _fold(value).replace("-", "_").replace(" ", "_")
    aliases = {
        "function": "fonction",
        "fonctionnelle": "fonction",
        "functional": "fonction",
        "configuration": "configuration",
        "selector": "configuration",
        "selection": "configuration",
        "interface": "interface",
        "mounting": "interface",
        "montage": "interface",
        "connection": "interface",
        "raccordement": "interface",
        "performance": "performance",
        "rating": "performance",
        "conformity": "conformite",
        "compliance": "conformite",
        "norme": "conformite",
        "standard": "conformite",
        "safety": "securite",
        "security": "securite",
        "secondary": "secondaire",
        "context": "secondaire",
    }
    result = aliases.get(folded, folded)
    if result in CRITICALITIES:
        return result

    # Repli générique limité aux noms de dimensions, jamais à une marque/référence.
    field_folded = _fold(field).replace("-", "_").replace(" ", "_")
    if any(token in field_folded for token in ("fonction", "function", "usage_principal")):
        return "fonction"
    if any(token in field_folded for token in ("montage", "mount", "interface", "raccord", "connection", "commande_front")):
        return "interface"
    if any(token in field_folded for token in ("mode", "configuration", "position", "pole", "phase", "transition")):
        return "configuration"
    if any(token in field_folded for token in ("norm", "conform", "certif")):
        return "conformite"
    if any(token in field_folded for token in ("secur", "protection")):
        return "securite"
    if any(token in field_folded for token in ("courant", "tension", "puissance", "debit", "pression", "capacity", "rating", "performance")):
        return "performance"
    return "indeterminee"


def _normalise_proofs(value: object, legacy_proof: object = None) -> list[dict[str, str]]:
    raw_values = value if isinstance(value, list) else []
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_values:
        if isinstance(raw, str):
            proof_type = "preuve"
            excerpt = _clean(raw)
        elif isinstance(raw, dict):
            proof_type = _clean(raw.get("type") or raw.get("role") or "preuve")
            excerpt = _clean(raw.get("extrait") or raw.get("preuve") or raw.get("evidence"))
        else:
            continue
        key = (_fold(proof_type), _fold(excerpt))
        if excerpt and key not in seen:
            seen.add(key)
            result.append({"type": proof_type or "preuve", "extrait": excerpt})
    legacy = _clean(legacy_proof)
    legacy_key = ("preuve", _fold(legacy))
    if legacy and legacy_key not in seen:
        result.append({"type": "preuve", "extrait": legacy})
    return result


def _normalise_proof_relation(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    relation_type = _fold(value.get("type")).replace("-", "_").replace(" ", "_")
    aliases = {
        "same_row": "meme_ligne",
        "same_column": "meme_colonne",
        "same_table": "meme_tableau",
        "same_product_block": "meme_bloc_produit",
        "ordering_code": "codification_reference",
        "continued_table": "continuation_tableau",
    }
    relation_type = aliases.get(relation_type, relation_type)
    proof = _clean(value.get("preuve") or value.get("evidence") or value.get("extrait"))
    if relation_type not in PROOF_RELATIONS or not proof:
        return None
    return {"type": relation_type, "preuve": proof}


def _normalise_item(value: object, *, default_field: str) -> dict[str, Any] | None:
    if isinstance(value, str):
        cleaned = _clean(value)
        if not cleaned:
            return None
        return {
            "champ": default_field,
            "valeur": cleaned,
            "unite": None,
            "preuve": None,
            "preuves": [],
            "relation_preuve": None,
            "source": None,
            "confiance": "moyenne",
            "criticalite": _normalise_criticality(None, default_field),
        }
    if not isinstance(value, dict):
        return None

    field = _clean(
        value.get("champ")
        or value.get("nom")
        or value.get("name")
        or value.get("critere")
        or value.get("caracteristique")
        or default_field
    )
    raw_value = (
        value.get("valeur")
        if "valeur" in value
        else value.get("value", value.get("texte"))
    )
    cleaned_value = _clean(raw_value)
    if not cleaned_value:
        return None
    unit = _clean(value.get("unite") or value.get("unit")) or None
    proof = _clean(
        value.get("preuve")
        or value.get("evidence")
        or value.get("expression_source")
    ) or None
    source = _clean(value.get("source") or value.get("url") or value.get("document")) or None
    confidence = _normalise_confidence(value.get("confiance") or value.get("confidence"))
    criticality = _normalise_criticality(
        value.get("criticalite") or value.get("criticite") or value.get("criticality"),
        field,
    )
    proofs = _normalise_proofs(value.get("preuves") or value.get("proofs"), proof)
    relation = _normalise_proof_relation(
        value.get("relation_preuve") or value.get("proof_relation")
    )
    return {
        "champ": field or default_field,
        "valeur": cleaned_value,
        "unite": unit,
        "preuve": proof,
        "preuves": proofs,
        "relation_preuve": relation,
        "source": source,
        "confiance": confidence,
        "criticalite": criticality,
    }


def _normalise_items(values: object, *, default_field: str) -> list[dict[str, Any]]:
    if values is None:
        return []
    raw_values = values if isinstance(values, list) else [values]
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_values:
        item = _normalise_item(raw, default_field=default_field)
        if item is None:
            continue
        key = (
            _fold(item["champ"]),
            _fold(item["valeur"]),
            _fold(item.get("unite")),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _normalise_identity(raw: object) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    return {
        "statut": _normalise_identity_status(
            source.get("statut") or source.get("status")
        ),
        "fabricant": _clean(
            source.get("fabricant") or source.get("manufacturer")
        ) or None,
        "reference": _clean(
            source.get("reference")
            or source.get("reference_source")
            or source.get("source_reference")
        ) or None,
        "preuve": _clean(source.get("preuve") or source.get("evidence")) or None,
        "source": _clean(source.get("source") or source.get("url")) or None,
    }


def _unique_strings(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean(raw)
        key = _fold(value)
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _merge_items(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for item in group:
            key = (
                _fold(item.get("champ")),
                _fold(item.get("valeur")),
                _fold(item.get("unite")),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(deepcopy(item))
    return result


def _item_values(items: Iterable[dict[str, Any]]) -> list[str]:
    return _unique_strings(item.get("valeur") for item in items)


def migrer_provenance_legacy(provenance: object) -> dict[str, Any]:
    """Convertit toute provenance connue vers la structure V4 non durcie."""

    source = provenance if isinstance(provenance, dict) else {}
    is_v4_source = bool(
        source.get("schema_version") == SCHEMA_VERSION
        or any(
            key in source
            for key in (
                "type_demande",
                "identification_produit",
                "contraintes_explicites",
                "contraintes_derivees_obligatoires",
                "criteres_classement",
            )
        )
    )
    explicit_source = (
        source.get("contraintes_explicites")
        if "contraintes_explicites" in source
        else source.get("contraintes_client", [])
    )
    ranking_source = source.get("criteres_classement")
    if ranking_source is None:
        ranking_source = []
        for key in (
            "caracteristiques_produit",
            "caracteristiques_source",
            "criteres_equivalence",
        ):
            raw = source.get(key)
            if isinstance(raw, list):
                ranking_source.extend(raw)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION if is_v4_source else "3.1",
        "type_demande": _normalise_request_type(source.get("type_demande")),
        "identification_produit": _normalise_identity(
            source.get("identification_produit") or source.get("produit_source")
        ),
        "contraintes_explicites": _normalise_items(
            explicit_source, default_field="exigence_client"
        ),
        "contraintes_derivees_obligatoires": _normalise_items(
            source.get("contraintes_derivees_obligatoires", []),
            default_field="caracteristique_structurante",
        ),
        "criteres_classement": _normalise_items(
            ranking_source, default_field="caracteristique_produit"
        ),
        "capacites_optionnelles": _normalise_items(
            source.get("capacites_optionnelles", []),
            default_field="capacite_optionnelle",
        ),
        "informations_a_confirmer": _normalise_items(
            source.get("informations_a_confirmer", []),
            default_field="information_a_confirmer",
        ),
        "sources": _unique_strings(source.get("sources", []) if isinstance(source.get("sources", []), list) else []),
    }


def _item_proof_attested(
    item: dict[str, Any],
    reference: str,
    documentation: str,
) -> bool:
    """Validate direct or structurally linked proof without domain knowledge."""

    value = item.get("valeur")
    legacy_proof = item.get("preuve")
    if legacy_proof and _proof_attested(legacy_proof, documentation):
        if _attested(reference, legacy_proof) and _attested(value, legacy_proof):
            return True

    proofs = item.get("preuves") if isinstance(item.get("preuves"), list) else []
    attested_proofs = [
        proof for proof in proofs
        if isinstance(proof, dict)
        and _proof_attested(proof.get("extrait"), documentation)
    ]
    reference_proved = any(
        _attested(reference, proof.get("extrait")) for proof in attested_proofs
    )
    value_proved = any(
        _attested(value, proof.get("extrait")) for proof in attested_proofs
    )
    relation = item.get("relation_preuve")
    relation_proved = bool(
        isinstance(relation, dict)
        and relation.get("type") in PROOF_RELATIONS
        and _proof_attested(relation.get("preuve"), documentation)
    )
    return bool(reference_proved and value_proved and relation_proved)


def normaliser_provenance_v4(
    demande_client: str,
    provenance: object,
    documentation: str = "",
) -> dict[str, Any]:
    """Migre et durcit une provenance sans inventer ni supprimer silencieusement.

    Les contraintes dérivées sont promues uniquement pour une référence exacte
    ou une demande mixte dont l'identité et chaque preuve sont attestées.
    """

    base = migrer_provenance_legacy(provenance)
    request_type = base["type_demande"]
    identity = base["identification_produit"]

    explicit_ok: list[dict[str, Any]] = []
    confirmations = list(base["informations_a_confirmer"])
    for item in base["contraintes_explicites"]:
        if _attested(item["valeur"], demande_client):
            explicit_ok.append(item)
        else:
            confirmations.append(item)

    reference = identity.get("reference")
    reference_confirmed = bool(
        request_type in {"reference_exacte", "mixte"}
        and identity.get("statut") == "confirmee"
        and isinstance(reference, str)
        and reference
        and _attested(reference, demande_client)
        and _attested(reference, documentation)
    )

    derived_ok: list[dict[str, Any]] = []
    ranking = list(base["criteres_classement"])
    if documentation:
        ranking = [
            item for item in ranking
            if _attested(item["valeur"], documentation)
        ]
    for item in base["contraintes_derivees_obligatoires"]:
        value_proved = _attested(item["valeur"], documentation)
        proof_proved = bool(reference) and _item_proof_attested(
            item, str(reference or ""), documentation
        )
        high_confidence = item.get("confiance") == "elevee"
        if reference_confirmed and value_proved and proof_proved and high_confidence:
            derived_ok.append(item)
            continue
        if identity.get("statut") in {"ambigue", "non_verifiee"} and request_type in {
            "reference_exacte",
            "mixte",
        }:
            confirmations.append(item)
        elif value_proved:
            ranking.append(item)
        else:
            confirmations.append(item)

    ranking = _merge_items(ranking)
    confirmations = _merge_items(confirmations)
    optional_source = list(base["capacites_optionnelles"])
    if documentation:
        optional_source = [
            item for item in optional_source
            if _attested(item["valeur"], documentation)
        ]
    optional = _merge_items(optional_source)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": base.get("source_schema_version", "3.1"),
        "type_demande": request_type,
        "identification_produit": identity,
        "contraintes_explicites": _merge_items(explicit_ok),
        "contraintes_derivees_obligatoires": _merge_items(derived_ok),
        "criteres_classement": ranking,
        "capacites_optionnelles_structurees": optional,
        "informations_a_confirmer_structurees": confirmations,
        "sources": list(base["sources"]),
    }

    # Vues de compatibilité V3.1 et consommateurs existants.
    result["contraintes_client"] = _item_values(result["contraintes_explicites"])
    result["caracteristiques_produit"] = _unique_strings(
        [
            *_item_values(result["contraintes_derivees_obligatoires"]),
            *_item_values(result["criteres_classement"]),
        ]
    )
    result["capacites_optionnelles"] = _item_values(optional)
    result["informations_a_confirmer"] = _item_values(confirmations)
    return result


def valeurs_provenance(provenance: dict[str, Any], cle: str) -> list[str]:
    """Retourne les valeurs d'une catégorie structurée ou de sa vue historique."""

    raw = provenance.get(cle, []) if isinstance(provenance, dict) else []
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return _item_values(raw)
    if isinstance(raw, list):
        return _unique_strings(raw)
    return []


def evaluer_completude_reference(provenance: object) -> dict[str, Any]:
    """Audit generic completeness for exact-reference or mixed requests.

    A source reference is not considered sufficiently described when the output
    contains only numeric ratings. At least one function and one additional
    structural dimension must be present among proven derived requirements.
    """

    base = migrer_provenance_legacy(provenance)
    request_type = base.get("type_demande")
    if request_type not in {"reference_exacte", "mixte"}:
        return {
            "applicable": False,
            "complete": True,
            "score": 1.0,
            "roles_presents": [],
            "roles_manquants": [],
        }

    items = base.get("contraintes_derivees_obligatoires", [])
    roles = {
        _normalise_criticality(item.get("criticalite"), item.get("champ"))
        for item in items
        if isinstance(item, dict)
    }
    structural = roles.intersection({
        "configuration", "interface", "performance", "conformite", "securite"
    })
    missing: list[str] = []
    if "fonction" not in roles:
        missing.append("fonction")
    if not structural:
        missing.append("dimension_structurante")
    complete = not missing
    score = min(1.0, (0.55 if "fonction" in roles else 0.0) + (0.35 if structural else 0.0) + min(0.10, len(items) * 0.025))
    return {
        "applicable": True,
        "complete": complete,
        "score": round(score, 3),
        "roles_presents": sorted(role for role in roles if role != "indeterminee"),
        "roles_manquants": missing,
    }


def documentation_depuis_preuves(provenance: object) -> str:
    """Reconstruit un corpus local minimal depuis les preuves déjà enregistrées.

    Cette fonction sert uniquement à la régénération hors ligne. Elle ne crée
    aucune donnée : elle concatène la référence, les valeurs et les preuves
    présentes dans le fichier de provenance.
    """

    base = migrer_provenance_legacy(provenance)
    identity = base["identification_produit"]
    fragments: list[object] = [
        identity.get("reference"),
        identity.get("preuve"),
    ]
    for key in STRUCTURED_KEYS:
        for item in base.get(key, []):
            fragments.extend((item.get("valeur"), item.get("preuve")))
            proofs = item.get("preuves")
            if isinstance(proofs, list):
                for proof in proofs:
                    if isinstance(proof, dict):
                        fragments.append(proof.get("extrait"))
                    else:
                        fragments.append(proof)
            relation = item.get("relation_preuve")
            if isinstance(relation, dict):
                fragments.append(relation.get("preuve"))
    return "\n".join(_unique_strings(fragments))
