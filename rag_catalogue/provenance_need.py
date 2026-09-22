"""Fusion déterministe d'une provenance V3.1/V4 dans le besoin catalogue.

Le LLM continue d'extraire la famille et les requêtes, mais il ne redécide pas
le statut des valeurs déjà classées par le pipeline de cahier. Ce module reste
strictement générique : il manipule des rôles et des preuves, pas des domaines
ou des références fabricant.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable


_PROVENANCE_PRIORITY = {
    "to_confirm": 1,
    "optional": 2,
    "source_product": 3,
    "derived_required": 4,
    "client": 5,
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(value).casefold())


def _identity_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    name = _clean(
        item.get("champ")
        or item.get("nom")
        or item.get("name")
    ).casefold()
    criticality = _clean(
        item.get("criticalite")
        or item.get("criticite")
        or item.get("role_equivalence")
    ).casefold()
    return criticality == "identite" or name in {
        "reference",
        "reference_origine",
        "référence",
        "référence_origine",
    }


def _normalise_item(
    item: object,
    *,
    default_name: str,
    role: str,
    provenance: str,
    constraint_scope: str | None = None,
) -> dict[str, Any] | None:
    if isinstance(item, str):
        value = _clean(item)
        if not value:
            return None
        name = default_name
        unit = None
        evidence = ""
        source = ""
        confidence = None
        criticality = None
    elif isinstance(item, dict):
        if _identity_item(item):
            return None
        name = _clean(
            item.get("champ")
            or item.get("nom")
            or item.get("critere")
            or item.get("name")
            or default_name
        )
        value = _clean(
            item.get("valeur")
            if "valeur" in item
            else item.get("value")
        )
        unit = _clean(item.get("unite") or item.get("unit")) or None
        if not value:
            value = _clean(item.get("valeur_complete"))
            unit = None
        if not value:
            return None
        evidence = _clean(
            item.get("preuve")
            or item.get("evidence")
            or item.get("expression_source")
        )
        source = _clean(
            item.get("source")
            or item.get("url")
            or item.get("source_url")
        )
        confidence = _clean(item.get("confiance") or item.get("confidence")) or None
        criticality = _clean(
            item.get("criticalite")
            or item.get("criticite")
            or item.get("role_equivalence")
        ) or None
    else:
        return None

    result: dict[str, Any] = {
        "name": name or default_name,
        "value": value,
        "unit": unit,
        "role": role,
        "provenance": provenance,
    }
    if constraint_scope:
        result["constraint_scope"] = constraint_scope
    if evidence:
        result["source_evidence"] = evidence
    if source:
        result["source_url"] = source
    if confidence:
        result["confidence"] = confidence
    if criticality:
        result["criticality"] = criticality
    return result


def _attributes_from(
    values: object,
    *,
    default_name: str,
    role: str,
    provenance: str,
    constraint_scope: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        attribute = _normalise_item(
            item,
            default_name=f"{default_name}_{index}",
            role=role,
            provenance=provenance,
            constraint_scope=constraint_scope,
        )
        if attribute is not None:
            result.append(attribute)
    return result


def _attribute_key(attribute: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _canonical(attribute.get("name")),
        _canonical(attribute.get("value")),
        _canonical(attribute.get("unit")),
    )


def _attribute_value_key(attribute: dict[str, Any]) -> str:
    """Canonicalise une valeur indépendamment de son découpage valeur/unité.

    Le LLM peut rendre ``{"value": "3", "unit": "A"}`` alors qu'un ancien
    JSON V3.1 contient simplement ``"3A"``. Les deux représentent la même
    donnée et ne doivent pas être ajoutées deux fois.
    """

    return _canonical(
        " ".join(
            part
            for part in (
                _clean(attribute.get("value")),
                _clean(attribute.get("unit")),
            )
            if part
        )
    )


_GENERIC_NAME_PREFIXES = (
    "contrainte_client_",
    "contrainte_explicite_",
    "contrainte_derivee_",
    "caracteristique_produit_",
    "critere_classement_",
    "information_a_confirmer_",
    "option_",
)


def _generic_attribute_name(attribute: dict[str, Any]) -> bool:
    name = _clean(attribute.get("name")).casefold()
    return any(name.startswith(prefix) for prefix in _GENERIC_NAME_PREFIXES)


def _merge_duplicate_attribute(
    previous: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    """Fusionne un doublon sans perdre le nom technique plus précis du LLM."""

    previous_priority = _PROVENANCE_PRIORITY.get(
        _clean(previous.get("provenance")), 0
    )
    new_priority = _PROVENANCE_PRIORITY.get(_clean(new.get("provenance")), 0)
    if new_priority < previous_priority:
        return previous

    # Les chaînes V3.1 ne portent pas de nom de champ. Le nom synthétique
    # ``contrainte_client_1`` ne doit pas remplacer ``rated current`` déjà
    # extrait du cahier. En revanche, son statut structuré fait autorité.
    if _generic_attribute_name(new) and not _generic_attribute_name(previous):
        merged = deepcopy(previous)
        for key in (
            "role",
            "provenance",
            "constraint_scope",
            "source_evidence",
            "source_url",
            "confidence",
            "criticality",
        ):
            if key in new:
                merged[key] = deepcopy(new[key])
        return merged
    return deepcopy(new)


def _merge_attributes(
    base: Iterable[dict[str, Any]],
    structured: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Déduplique en laissant la provenance la plus forte gagner."""

    result: list[dict[str, Any]] = []
    positions: dict[tuple[str, str, str], int] = {}
    value_positions: dict[str, list[int]] = {}
    combined = [
        *((False, item) for item in base),
        *((True, item) for item in structured),
    ]
    for is_structured, raw in combined:
        if not isinstance(raw, dict):
            continue
        attribute = deepcopy(raw)
        key = _attribute_key(attribute)
        if not any(key):
            continue
        previous_index = positions.get(key)
        if previous_index is None:
            value_key = _attribute_value_key(attribute)
            candidates = value_positions.get(value_key, []) if value_key else []
            structured_metadata = any(
                key in attribute
                for key in (
                    "constraint_scope",
                    "source_evidence",
                    "source_url",
                    "confidence",
                    "criticality",
                )
            )
            if candidates and (
                _generic_attribute_name(attribute)
                or (is_structured and structured_metadata and len(candidates) == 1)
            ):
                previous_index = candidates[0]
        if previous_index is None:
            positions[key] = len(result)
            result.append(attribute)
            value_key = _attribute_value_key(attribute)
            if value_key:
                value_positions.setdefault(value_key, []).append(len(result) - 1)
            continue
        previous = result[previous_index]
        merged = _merge_duplicate_attribute(previous, attribute)
        old_key = _attribute_key(previous)
        result[previous_index] = merged
        positions.pop(old_key, None)
        positions[_attribute_key(merged)] = previous_index
    return result


def _legacy_or_v4_attributes(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    new_explicit = provenance.get("contraintes_explicites")
    if isinstance(new_explicit, list):
        result.extend(_attributes_from(
            new_explicit,
            default_name="contrainte_explicite",
            role="constraint",
            provenance="client",
            constraint_scope="hard",
        ))
    elif isinstance(provenance.get("contraintes_client"), list):
        result.extend(_attributes_from(
            provenance.get("contraintes_client"),
            default_name="contrainte_client",
            role="constraint",
            provenance="client",
            constraint_scope="hard",
        ))

    result.extend(_attributes_from(
        provenance.get("contraintes_derivees_obligatoires"),
        default_name="contrainte_derivee",
        role="constraint",
        provenance="derived_required",
        constraint_scope="direct_equivalence",
    ))

    new_ranking = provenance.get("criteres_de_classement")
    if isinstance(new_ranking, list):
        result.extend(_attributes_from(
            new_ranking,
            default_name="critere_classement",
            role="selector",
            provenance="source_product",
        ))
    elif isinstance(provenance.get("caracteristiques_produit"), list):
        result.extend(_attributes_from(
            provenance.get("caracteristiques_produit"),
            default_name="caracteristique_produit",
            role="selector",
            provenance="source_product",
        ))

    result.extend(_attributes_from(
        provenance.get("capacites_optionnelles"),
        default_name="option",
        role="context",
        provenance="optional",
    ))
    result.extend(_attributes_from(
        provenance.get("informations_a_confirmer"),
        default_name="information_a_confirmer",
        role="context",
        provenance="to_confirm",
    ))
    return result


def _question_from_attribute(attribute: dict[str, Any]) -> str:
    name = _clean(attribute.get("name"))
    value = _clean(attribute.get("value"))
    unit = _clean(attribute.get("unit"))
    rendered_value = " ".join(part for part in (value, unit) if part)
    return f"{name} : {rendered_value}" if name else rendered_value


def merge_structured_provenance_into_need(
    need: dict[str, Any],
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fusionne le JSON compagnon dans le besoin sans réinterprétation LLM."""

    if not isinstance(need, dict):
        raise ValueError("le besoin extrait doit etre un objet")
    result = deepcopy(need)
    if not isinstance(provenance, dict):
        return result

    base_attributes = result.get("attributes")
    if not isinstance(base_attributes, list):
        base_attributes = []
    structured = _legacy_or_v4_attributes(provenance)
    is_v4 = bool(
        _clean(provenance.get("schema_version")).startswith("4")
        or any(
            key in provenance
            for key in (
                "identification_produit",
                "contraintes_explicites",
                "contraintes_derivees_obligatoires",
                "criteres_de_classement",
            )
        )
    )
    existing_questions = result.get("open_questions")
    if not is_v4 and isinstance(existing_questions, list) and existing_questions:
        # Le bundle V3.1 complet a déjà été lu par le LLM. Réinjecter ses
        # ambiguïtés sous forme d'attributs génériques dupliquerait la même
        # question dans une autre langue ou formulation.
        structured = [
            attribute
            for attribute in structured
            if attribute.get("provenance") != "to_confirm"
        ]
    result["attributes"] = _merge_attributes(
        [item for item in base_attributes if isinstance(item, dict)],
        structured,
    )

    identification = provenance.get("identification_produit")
    if isinstance(identification, dict):
        status = _clean(identification.get("statut") or identification.get("status"))
        if status in {"exacte", "probable"}:
            reference = _clean(
                identification.get("reference")
                or identification.get("reference_source")
            )
            manufacturer = _clean(
                identification.get("fabricant")
                or identification.get("manufacturer")
            )
            designation = _clean(
                identification.get("designation")
                or identification.get("product_type")
            )
            if reference:
                result["source_reference"] = reference
            if manufacturer:
                result["source_manufacturer"] = manufacturer
            if designation:
                result["product_type"] = designation

    questions = result.get("open_questions")
    if not isinstance(questions, list):
        questions = []
    seen_questions = {_canonical(question) for question in questions if isinstance(question, str)}
    for attribute in result["attributes"]:
        if attribute.get("provenance") != "to_confirm":
            continue
        question = _question_from_attribute(attribute)
        key = _canonical(question)
        if question and key not in seen_questions:
            seen_questions.add(key)
            questions.append(question)
    result["open_questions"] = questions[:12]
    result["structured_provenance_version"] = _clean(
        provenance.get("schema_version")
    ) or "3.1"
    result["request_type"] = _clean(provenance.get("type_demande")) or None
    return result
