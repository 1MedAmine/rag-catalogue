"""OpenAI-compatible NVIDIA LLM boundary for the catalogue RAG."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from .retrieval import RankedChunk
from .result_text import normalise_explanation


DEFAULT_NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_REASONING_MODE = "normal"
REASONING_PROFILES: dict[str, dict[str, Any]] = {
    "rapide": {
        "enable_thinking": False,
        "low_effort": None,
        "reasoning_budget": 0,
    },
    "normal": {
        "enable_thinking": True,
        "low_effort": True,
        "reasoning_budget": 2048,
    },
    "approfondi": {
        "enable_thinking": True,
        "low_effort": False,
        # Added to max_tokens rather than taken from it, so a larger budget
        # buys deliberation on catalogues with competing product families
        # without shortening the answer.
        "reasoning_budget": 8192,
    },
}


class LLMResponseError(ValueError):
    """The model response cannot be used as the required structured result."""


def _response_detail(response: Any | None) -> str:
    if response is None:
        return ""
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        for key in ("message", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
    text = getattr(response, "text", "")
    return " ".join(str(text or "").split())[:500]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a possibly fenced response."""

    if not isinstance(text, str) or not text.strip():
        raise LLMResponseError("Reponse LLM vide; objet JSON attendu")

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    excerpt = " ".join(text.strip().split())[:240]
    raise LLMResponseError(f"Objet JSON introuvable dans la reponse LLM: {excerpt}")


def repair_json_locally(text: str) -> dict[str, Any]:
    """Repair a small, bounded set of syntax defects without changing values.

    Supported defects are text appended in parentheses just outside a quoted
    JSON string and trailing commas before a closing array/object. Truncated
    output is deliberately not completed locally.
    """

    if not isinstance(text, str) or not text.strip():
        raise LLMResponseError("Reponse LLM vide; objet JSON attendu")
    repaired = text
    parenthetical = re.compile(
        r'"((?:[^"\\]|\\.)*)"\s*(\([^()\r\n]*\))(?=\s*[,}\]])'
    )
    for _ in range(4):
        updated = parenthetical.sub(lambda match: f'"{match.group(1)} {match.group(2)}"', repaired)
        if updated == repaired:
            break
        repaired = updated
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return extract_json_object(repaired)


def _looks_truncated_json(text: str) -> bool:
    if not isinstance(text, str):
        return True
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped[-1] not in "}]":
        return True
    in_string = False
    escaped = False
    braces = 0
    brackets = 0
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
    return in_string or braces != 0 or brackets != 0


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _normalise_generated_text(value: Any, *, fallback: str = "") -> str:
    """Normalise harmless free-text shapes emitted by the model."""

    return normalise_explanation(value, fallback=fallback)


def _normalise_generated_text_list(value: Any) -> list[str]:
    """Normalise one text or a list of texts without rejecting the response."""

    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = normalise_explanation(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _normalise_provenance(item: dict[str, Any]) -> str | None:
    raw = _first_present(
        item, "provenance", "origine", "source_type", "value_source",
    )
    value = _text_value(raw)
    if value is None:
        return None
    folded = value.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "client": "client",
        "demande_client": "client",
        "customer": "client",
        "source_product": "source_product",
        "derived_required": "derived_required",
        "contrainte_derivee": "derived_required",
        "contrainte_derivee_obligatoire": "derived_required",
        "produit_source": "source_product",
        "documentation": "source_product",
        "documented": "source_product",
        "optional": "optional",
        "option": "optional",
        "optionnel": "optional",
        "to_confirm": "to_confirm",
        "a_confirmer": "to_confirm",
        "ambiguous": "to_confirm",
        "ambigu": "to_confirm",
    }
    return aliases.get(folded)


def _normalise_role(item: dict[str, Any]) -> str | None:
    raw = _first_present(
        item,
        "role", "importance", "usage", "type_attribut", "attribute_role",
    )
    if isinstance(raw, bool):
        return "selector" if raw else "context"
    value = _text_value(raw)
    if value is None:
        changes_reference = _first_present(
            item, "changes_reference", "change_reference", "modifie_reference",
            "modifie_la_reference", "reference_defining",
        )
        if isinstance(changes_reference, bool):
            return "selector" if changes_reference else "context"
        required = _first_present(item, "required", "mandatory", "obligatoire", "critical")
        if isinstance(required, bool) and required:
            return "constraint"
        return None

    folded = value.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "selector": "selector",
        "selection": "selector",
        "reference": "selector",
        "reference_defining": "selector",
        "variant": "selector",
        "discriminant": "selector",
        "constraint": "constraint",
        "required": "constraint",
        "mandatory": "constraint",
        "critical": "constraint",
        "obligatoire": "constraint",
        "context": "context",
        "informative": "context",
        "information": "context",
        "secondary": "context",
        "non_selector": "context",
    }
    return aliases.get(folded)


def _normalise_attributes(raw: Any) -> list[dict[str, str | None]]:
    """Accept common JSON shapes without assuming a product domain."""

    if raw is None:
        return []

    items: list[Any]
    if isinstance(raw, dict):
        items = [{"name": name, "value": value} for name, value in raw.items()]
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    normalised: list[dict[str, str | None]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            value = item.strip()
            if value:
                normalised.append({
                    "name": f"attribute_{index}",
                    "value": value,
                    "unit": None,
                })
            continue
        if not isinstance(item, dict):
            continue

        name = _text_value(_first_present(
            item, "name", "nom", "attribute", "attribut",
            "characteristic", "caracteristique", "caractéristique",
        ))
        value_raw = _first_present(item, "value", "valeur", "val")
        unit = _text_value(_first_present(item, "unit", "unite", "unité"))

        if name is None and len(item) == 1:
            only_name, value_raw = next(iter(item.items()))
            name = _text_value(only_name)

        if isinstance(value_raw, dict):
            nested = value_raw
            value_raw = _first_present(nested, "value", "valeur", "val")
            if unit is None:
                unit = _text_value(_first_present(nested, "unit", "unite", "unité"))

        value = _text_value(value_raw)
        if name is None or value is None:
            continue
        attribute: dict[str, str | None] = {"name": name, "value": value, "unit": unit}
        role = _normalise_role(item)
        if role is not None:
            attribute["role"] = role
        provenance = _text_value(_first_present(
            item, "provenance", "origine", "source_status", "statut_source",
        ))
        if provenance is not None:
            folded = provenance.casefold().replace("-", "_").replace(" ", "_")
            aliases = {
                "client": "client",
                "demande_client": "client",
                "source_product": "source_product",
                "derived_required": "derived_required",
                "contrainte_derivee": "derived_required",
                "contrainte_derivee_obligatoire": "derived_required",
                "produit_source": "source_product",
                "documentation": "source_product",
                "to_confirm": "to_confirm",
                "a_confirmer": "to_confirm",
                "à_confirmer": "to_confirm",
            }
            if folded in aliases:
                attribute["provenance"] = aliases[folded]
        constraint_scope = _text_value(_first_present(
            item, "constraint_scope", "scope", "portee_contrainte",
        ))
        if constraint_scope is not None:
            folded_scope = constraint_scope.casefold().replace("-", "_").replace(" ", "_")
            if folded_scope in {"hard", "direct_equivalence"}:
                attribute["constraint_scope"] = folded_scope
        for source_key, aliases_for_key in (
            ("source_evidence", ("source_evidence", "preuve_source", "evidence")),
            ("source_url", ("source_url", "url_source")),
            ("confidence", ("confidence", "confiance")),
            ("criticality", ("criticality", "criticalite", "criticite")),
        ):
            metadata_value = _text_value(_first_present(item, *aliases_for_key))
            if metadata_value is not None:
                attribute[source_key] = metadata_value
        normalised.append(attribute)
    return normalised


def _normalise_open_questions(raw: Any) -> list[str]:
    candidates = [raw] if isinstance(raw, str) else raw
    if not isinstance(candidates, list):
        return []
    questions: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _text_value(candidate)
        if value is None:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        questions.append(value)
        if len(questions) == 12:
            break
    return questions


def _normalise_queries(raw: Any) -> list[str]:
    candidates = [raw] if isinstance(raw, str) else raw
    if not isinstance(candidates, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _text_value(candidate)
        if value is None:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(value)
        if len(queries) == 3:
            break
    return queries


def _fallback_query(
    product_type: str | None,
    attributes: list[dict[str, str | None]],
    product_text: str,
) -> str:
    parts: list[str] = []
    if product_type:
        parts.append(product_type)
    for attribute in attributes[:16]:
        fragment = " ".join(
            part for part in (
                attribute.get("name"),
                attribute.get("value"),
                attribute.get("unit"),
            )
            if isinstance(part, str) and part.strip()
        )
        if fragment:
            parts.append(fragment)
    if parts:
        return " ".join(parts)[:1800]
    return " ".join(product_text.split())[:1800]


def _normalise_need_response(
    result: dict[str, Any],
    product_text: str,
) -> dict[str, Any]:
    product_type = _text_value(_first_present(
        result, "product_type", "type_produit", "product", "produit", "type",
    ))
    source_manufacturer = _text_value(_first_present(
        result, "source_manufacturer", "fabricant_source", "manufacturer", "fabricant", "brand", "marque",
    ))
    source_family = _text_value(_first_present(
        result, "source_family", "famille_source", "family", "famille", "range", "gamme",
    ))
    source_reference = _text_value(_first_present(
        result, "source_reference", "reference_source", "source_ref",
        "reference", "référence", "ref",
    ))
    attributes = _normalise_attributes(_first_present(
        result, "attributes", "attributs", "caracteristiques",
        "caractéristiques", "specifications", "spécifications", "specs",
        "technical_attributes",
    ))
    search_queries = _normalise_queries(_first_present(
        result, "search_queries", "requetes_recherche", "requêtes_recherche",
        "queries", "requetes", "requêtes",
    ))
    raw_questions = _first_present(
        result, "open_questions", "questions_ouvertes", "informations_a_confirmer",
        "informations_à_confirmer", "points_a_confirmer",
    )
    open_questions = _normalise_queries(raw_questions)
    if not search_queries:
        fallback = _fallback_query(product_type, attributes, product_text)
        if fallback:
            search_queries = [fallback]

    return {
        "product_type": product_type,
        "source_manufacturer": source_manufacturer,
        "source_family": source_family,
        "source_reference": source_reference,
        "attributes": attributes,
        "open_questions": open_questions,
        "search_queries": search_queries,
        "source_excerpt": product_text[:12_000],
    }


def _positive_integer(value: Any, *, label: str) -> int:
    """Normalise harmless JSON number variants while rejecting unsafe values."""

    if isinstance(value, bool):
        raise LLMResponseError(f"{label} doit etre un entier positif")
    if type(value) is int:
        parsed = value
    elif type(value) is float and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise LLMResponseError(f"{label} doit etre un entier positif")
    if parsed <= 0:
        raise LLMResponseError(f"{label} doit etre un entier positif")
    return parsed


def _normalise_catalogue_pages(raw: Any) -> list[int]:
    """Normalise harmless page shapes without failing the whole generation.

    Accepted examples include ``42``, ``"page 42"``, ``["41", 42.0]`` and
    ``"pages 41 et 42"``. Invalid fragments are ignored; deterministic
    validation can still reject a candidate whose final page list is empty.
    """

    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    pages: list[int] = []
    seen: set[int] = set()

    def add(page: int) -> None:
        if page > 0 and page not in seen:
            seen.add(page)
            pages.append(page)

    for value in values:
        if isinstance(value, bool):
            continue
        if type(value) is int:
            add(value)
            continue
        if type(value) is float and value.is_integer():
            add(int(value))
            continue
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        if cleaned.isdigit():
            add(int(cleaned))
            continue
        if re.fullmatch(r"\d+\.0+", cleaned):
            add(int(float(cleaned)))
            continue
        for match in re.findall(r"\d+", cleaned):
            add(int(match))
    return pages


def _normalise_reference_parts(raw: Any) -> list[dict[str, Any]]:
    """Accept harmless model aliases while preserving proof metadata."""

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LLMResponseError("reference_parts doit etre une liste")

    parts: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise LLMResponseError("chaque reference_part doit etre un objet")
        code = _text_value(_first_present(item, "code", "value", "segment"))
        page = item.get("page")
        position = item.get("position", index)
        evidence = _text_value(item.get("evidence")) or ""
        if code is None:
            raise LLMResponseError("chaque reference_part doit contenir un code")
        try:
            page = _positive_integer(
                page, label="chaque reference_part.page"
            )
            position = _positive_integer(
                position, label="chaque reference_part.position"
            )
        except LLMResponseError as exc:
            raise LLMResponseError(
                "chaque reference_part doit contenir une page et une position entieres positives"
            ) from exc
        parts.append({
            "position": position,
            "code": code,
            "page": page,
            "evidence": evidence,
        })
    return parts


_MODE_ALIASES = {
    "verbatim": "explicit",
    "assembled": "constructed",
    "explicit": "explicit",
    "constructed": "constructed",
}


_ASSESSMENT_STATUS_ALIASES = {
    "satisfied": "satisfait",
    "match": "satisfait",
    "matched": "satisfait",
    "satisfait": "satisfait",
    "different": "different",
    "mismatch": "different",
    "différent": "different",
    "different": "different",
    "unproved": "non_prouve",
    "not_proved": "non_prouve",
    "unknown": "non_prouve",
    "non_verifiable": "non_prouve",
    "non_prouve": "non_prouve",
}
_ASSESSMENT_CRITICALITIES = {
    "fonction", "configuration", "interface", "performance",
    "conformite", "securite", "secondaire", "indeterminee",
}


def _normalise_constraint_assessment(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LLMResponseError("constraint_assessment doit etre une liste")
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LLMResponseError("chaque constraint_assessment doit etre un objet")
        field = _text_value(_first_present(item, "champ", "name", "attribute"))
        expected = _text_value(_first_present(item, "attendu", "expected"))
        found_raw = _first_present(item, "trouve", "found", "actual")
        found = _text_value(found_raw)
        status_raw = _text_value(_first_present(item, "statut", "status"))
        criticality_raw = _text_value(_first_present(item, "criticalite", "criticality"))
        provenance_raw = _text_value(_first_present(item, "provenance", "origine"))
        if field is None or expected is None or status_raw is None:
            raise LLMResponseError("constraint_assessment incomplet")
        status_key = status_raw.casefold().replace("-", "_").replace(" ", "_")
        status = _ASSESSMENT_STATUS_ALIASES.get(status_key)
        if status is None:
            raise LLMResponseError("statut de constraint_assessment invalide")
        criticality = (
            criticality_raw.casefold().replace("-", "_").replace(" ", "_")
            if criticality_raw else "indeterminee"
        )
        if criticality not in _ASSESSMENT_CRITICALITIES:
            criticality = "indeterminee"
        provenance = (
            provenance_raw.casefold().replace("-", "_").replace(" ", "_")
            if provenance_raw else "source_product"
        )
        result.append({
            "champ": field,
            "attendu": expected,
            "trouve": found,
            "statut": status,
            "criticalite": criticality,
            "provenance": provenance,
        })
    return result


def _normalise_selected_candidate(raw: Any, *, default_rank: int) -> dict[str, Any]:
    """Normalise one ranked catalogue candidate returned by the model."""

    if not isinstance(raw, dict):
        raise LLMResponseError("chaque candidat doit etre un objet JSON")
    reference = raw.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        raise LLMResponseError("chaque candidat doit contenir une reference")
    reference = reference.strip()

    rank_raw = raw.get("rank", default_rank)
    rank = _positive_integer(rank_raw, label="rang du candidat")
    pages = _normalise_catalogue_pages(raw.get("catalogue_pages"))
    evidence = _normalise_generated_text_list(raw.get("evidence"))

    reason = _normalise_generated_text(
        raw.get("reason", raw.get("explanation", ""))
    )
    if not reason:
        raise LLMResponseError("chaque candidat doit contenir une justification")

    variant = raw.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise LLMResponseError("variant doit etre une chaine ou null")
    match_level = raw.get("match_level", "alternative")
    if not isinstance(match_level, str) or not match_level.strip():
        raise LLMResponseError("match_level doit etre une chaine")

    reference_mode = raw.get("reference_mode")
    if reference_mode is not None:
        if not isinstance(reference_mode, str) or reference_mode not in _MODE_ALIASES:
            raise LLMResponseError(
                "reference_mode doit etre explicit, constructed ou null"
            )
        reference_mode = _MODE_ALIASES[reference_mode]
    reference_parts = _normalise_reference_parts(raw.get("reference_parts", []))
    if reference_mode is None and reference_parts:
        reference_mode = "constructed"
    elif reference_mode == "explicit":
        reference_parts = []

    differences = _normalise_generated_text_list(raw.get("differences", []))
    mandatory_differences = _normalise_generated_text_list(
        raw.get("mandatory_differences", [])
    )
    constraint_assessment = _normalise_constraint_assessment(
        raw.get("constraint_assessment", [])
    )
    if mandatory_differences and match_level.strip().casefold() == "equivalent":
        match_level = "alternative"

    return {
        "rank": rank,
        "reference": reference,
        "variant": variant.strip() if isinstance(variant, str) and variant.strip() else None,
        "match_level": match_level.strip(),
        "reference_mode": reference_mode,
        "reference_parts": reference_parts,
        "catalogue_pages": pages,
        "evidence": evidence,
        "reason": reason,
        "differences": differences,
        "mandatory_differences": mandatory_differences,
        "constraint_assessment": constraint_assessment,
    }


class NvidiaChatClient:
    """Two-call LLM service: extract the need, then rank catalogue references."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_NVIDIA_API_BASE,
        client: Any | None = None,
        reasoning_mode: str = DEFAULT_REASONING_MODE,
        max_attempts: int = 5,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key ne peut pas etre vide")
        if not model.strip():
            raise ValueError("model ne peut pas etre vide")
        self.model = model.strip()
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._session = requests.Session() if client is None else None
        if reasoning_mode not in REASONING_PROFILES:
            allowed = ", ".join(REASONING_PROFILES)
            raise ValueError(f"reasoning_mode invalide; valeurs autorisees: {allowed}")
        if max_attempts <= 0:
            raise ValueError("max_attempts doit etre strictement positif")
        self.reasoning_mode = reasoning_mode
        self._reasoning_profile = dict(REASONING_PROFILES[reasoning_mode])
        self._max_attempts = max_attempts
        self._supports_reasoning = "nemotron-3" in self.model.casefold()
        self._json_repair_attempts = 0
        self._json_repair_successes = 0
        self._json_local_attempts = 0
        self._json_local_successes = 0
        self._json_external_attempts = 0
        self._json_external_successes = 0
        self._json_truncated_responses = 0
        self._last_finish_reason: str | None = None
        if "nemotron-3-super" in self.model.casefold():
            self._temperature = 1.0
            self._top_p: float | None = 0.95
        else:
            self._temperature = 0.0
            self._top_p = None

    @property
    def reasoning_config(self) -> dict[str, Any]:
        profile = self._reasoning_profile if self._supports_reasoning else REASONING_PROFILES["rapide"]
        return {
            "mode": self.reasoning_mode,
            "enabled": bool(profile["enable_thinking"]),
            "low_effort": profile["low_effort"],
            "budget": int(profile["reasoning_budget"]),
        }

    @property
    def json_repair_config(self) -> dict[str, int]:
        return {
            "attempts": int(self._json_repair_attempts),
            "successes": int(self._json_repair_successes),
            "local_attempts": int(self._json_local_attempts),
            "local_successes": int(self._json_local_successes),
            "external_attempts": int(self._json_external_attempts),
            "external_successes": int(self._json_external_successes),
            "truncated_responses": int(self._json_truncated_responses),
        }

    def _reasoning_extra_body(
        self, *, force_no_thinking: bool = False
    ) -> tuple[dict[str, Any], int]:
        if force_no_thinking and self._supports_reasoning:
            return {"chat_template_kwargs": {"enable_thinking": False}}, 0
        if not self._supports_reasoning:
            return {}, 0
        profile = self._reasoning_profile
        enabled = bool(profile["enable_thinking"])
        kwargs: dict[str, Any] = {"enable_thinking": enabled}
        if enabled and profile["low_effort"] is not None:
            kwargs["low_effort"] = bool(profile["low_effort"])
        body: dict[str, Any] = {"chat_template_kwargs": kwargs}
        budget = int(profile["reasoning_budget"]) if enabled else 0
        if budget > 0:
            body["reasoning_budget"] = budget
        return body, budget

    @staticmethod
    def _retryable_failure(*, status_code: int | None, detail: str) -> bool:
        if status_code in {429, 500, 502, 503, 504}:
            return True
        folded = detail.casefold()
        return any(marker in folded for marker in (
            "resourceexhausted",
            "request limit reached",
            "rate limit",
            "temporarily unavailable",
            "service unavailable",
        ))

    @staticmethod
    def _retry_delay(attempt_index: int) -> float:
        return float(min(40, 5 * (2 ** attempt_index)))

    def _chat_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        force_no_thinking: bool = False,
    ) -> str:
        """Send one chat request and return the raw assistant text."""

        extra_body, reasoning_budget = self._reasoning_extra_body(
            force_no_thinking=force_no_thinking
        )
        self._last_finish_reason = None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "max_tokens": max_tokens + reasoning_budget,
            "stream": False,
        }
        if self._top_p is not None:
            payload["top_p"] = self._top_p

        if self._client is not None:
            for attempt in range(self._max_attempts):
                try:
                    client_payload = dict(payload)
                    if extra_body:
                        client_payload["extra_body"] = extra_body
                    response = self._client.chat.completions.create(**client_payload)
                    choices = getattr(response, "choices", None)
                    if not choices:
                        raise LLMResponseError("Le fournisseur LLM n'a retourne aucun choix")
                    content = getattr(getattr(choices[0], "message", None), "content", None)
                    finish_reason = getattr(choices[0], "finish_reason", None)
                    self._last_finish_reason = (
                        str(finish_reason) if finish_reason is not None else None
                    )
                    break
                except LLMResponseError:
                    raise
                except Exception as exc:
                    detail = " ".join(str(exc).split())
                    status_code = getattr(exc, "status_code", None)
                    if (
                        attempt + 1 < self._max_attempts
                        and self._retryable_failure(status_code=status_code, detail=detail)
                    ):
                        time.sleep(self._retry_delay(attempt))
                        continue
                    raise LLMResponseError(
                        "Appel NVIDIA impossible: "
                        f"modele={self.model}, detail={detail[:500]}"
                    ) from exc
            else:  # pragma: no cover - loop always breaks or raises
                raise LLMResponseError("Appel NVIDIA impossible apres plusieurs tentatives")
        else:
            assert self._session is not None
            endpoint = f"{self._base_url}/chat/completions"
            body: dict[str, Any] | None = None
            for attempt in range(self._max_attempts):
                response = None
                try:
                    response = self._session.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                        },
                        json={**payload, **extra_body},
                        timeout=300,
                    )
                    response.raise_for_status()
                    body = response.json()
                    break
                except requests.RequestException as exc:
                    actual_response = getattr(exc, "response", None) or response
                    detail = _response_detail(actual_response)
                    status_code = getattr(actual_response, "status_code", None)
                    if (
                        attempt + 1 < self._max_attempts
                        and self._retryable_failure(status_code=status_code, detail=detail)
                    ):
                        time.sleep(self._retry_delay(attempt))
                        continue
                    raise LLMResponseError(
                        "Appel NVIDIA impossible: "
                        f"modele={self.model}, url={endpoint}"
                        + (f", detail={detail}" if detail else "")
                    ) from exc
                except ValueError as exc:
                    raise LLMResponseError(
                        "Reponse NVIDIA non JSON: "
                        f"modele={self.model}, url={endpoint}"
                    ) from exc
            if body is None:
                raise LLMResponseError("Appel NVIDIA impossible apres plusieurs tentatives")
            try:
                choice = body["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
                self._last_finish_reason = (
                    str(finish_reason) if finish_reason is not None else None
                )
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMResponseError("Reponse NVIDIA sans contenu de message") from exc
        if not isinstance(content, str):
            raise LLMResponseError("Le fournisseur LLM n'a retourne aucun texte")
        return content

    def _repair_malformed_json(self, content: str, *, max_tokens: int) -> dict[str, Any]:
        """Ask the same model once to repair syntax only, then parse strictly."""

        self._json_repair_attempts += 1
        self._json_external_attempts += 1
        repaired_text = self._chat_text(
            system=(
                "Tu es un reparateur JSON strict. Repare uniquement la syntaxe de "
                "l'objet fourni: guillemets, virgules, crochets, accolades et echappements. "
                "Ne change aucune valeur technique, n'ajoute aucun candidat, ne supprime "
                "aucune information et ne fournis aucun commentaire hors JSON."
            ),
            user=(
                "Repare cette reponse pour produire exactement un objet JSON valide:\n\n"
                + content[:60_000]
            ),
            max_tokens=max(800, min(max_tokens, 4200)),
            force_no_thinking=True,
        )
        repaired = extract_json_object(repaired_text)
        self._json_repair_successes += 1
        self._json_external_successes += 1
        return repaired

    def _chat_json(self, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
        content = self._chat_text(system=system, user=user, max_tokens=max_tokens)
        try:
            return extract_json_object(content)
        except LLMResponseError as original_error:
            finish_reason = (self._last_finish_reason or "").casefold()
            truncated = finish_reason in {"length", "max_tokens"} or _looks_truncated_json(content)
            if truncated:
                self._json_truncated_responses += 1
            else:
                self._json_repair_attempts += 1
                self._json_local_attempts += 1
                try:
                    repaired = repair_json_locally(content)
                except LLMResponseError:
                    pass
                else:
                    self._json_repair_successes += 1
                    self._json_local_successes += 1
                    return repaired
            try:
                return self._repair_malformed_json(content, max_tokens=max_tokens)
            except LLMResponseError as repair_error:
                raise LLMResponseError(
                    f"{original_error}; reparation JSON automatique echouee: {repair_error}"
                ) from repair_error

    def extract_need(self, product_text: str) -> dict[str, Any]:
        """Extract a domain-neutral technical need and retrieval queries."""

        if not isinstance(product_text, str) or not product_text.strip():
            raise ValueError("le texte de la fiche produit ne peut pas etre vide")
        bounded_text = product_text[:60_000]
        raw_result = self._chat_json(
            system=(
                "Tu extrais fidelement les caracteristiques d'une fiche technique "
                "industrielle, quel que soit le type de produit. Tu ne dois utiliser "
                "que le texte fourni. Reponds uniquement par un objet JSON."
            ),
            user=(
                "Analyse la fiche technique ci-dessous. Retourne exactement ce schema:\n"
                "{\n"
                '  "product_type": "type generique du produit avec son abreviation technique internationale courante si elle est certaine, sans marque, gamme ni reference, ou null",\n'
                '  "source_manufacturer": "fabricant source ou null",\n'
                '  "source_family": "gamme ou famille commerciale source ou null",\n'
                '  "source_reference": "reference source explicitement presente ou null",\n'
                '  "attributes": [\n'
                '    {"name": "caracteristique", "value": "valeur", "unit": "unite ou null", "role": "selector|constraint|context", "provenance": "client|derived_required|source_product|optional|to_confirm"}\n'
                "  ],\n"
                '  "open_questions": ["ambiguite technique a confirmer"],\n'
                '  "search_queries": ["requete technique 1", "requete technique 2"]\n'
                "}\n"
                "Regles:\n"
                "- product_type doit rester generique, inclure l'abreviation ou l'acronyme technique international courant lorsqu'il est certain, et ne contenir ni fabricant, ni gamme, ni reference;\n"
                "- les selector doivent etre un noyau minimal inter-fabricants: au plus 12 caracteristiques qui permettent de choisir la variante correspondante dans un autre catalogue;\n"
                "- attribue role=selector aux caracteristiques principales qui choisissent directement une variante ou une reference chez plusieurs fabricants;\n"
                "- attribue role=constraint seulement aux exigences explicitement obligatoires dans la fiche ou la demande;\n"
                "- attribue role=context aux informations descriptives qui ne doivent pas bloquer le choix d'une reference;\n"
                "- provenance=client signifie explicitement demande par le client; provenance=derived_required signifie derivee d une reference exacte et prouvee, obligatoire pour l equivalence directe; provenance=source_product signifie atteste dans la fiche du produit source, utile au classement et potentiellement necessaire a l'equivalence technique sans devenir une contrainte dure; provenance=optional signifie option ou capacite documentaire non imposee; provenance=to_confirm signifie ambigu ou non tranche;\n"
                "- le JSON de provenance annexe complete le cahier: conserve les caracteristiques source_product utiles pour chercher l'equivalent, sans les transformer en contraintes client;\n"
                "- conserve provenance=derived_required et role=constraint pour les contraintes derivees deja prouvees; elles ne sont pas des demandes client mais definissent l equivalence directe;\n"
                "- place toute ambiguite importante dans open_questions et, si utile, produis plusieurs branches de recherche;\n"
                "- lorsque la meme performance est donnee par plusieurs valeurs mesurees selon plusieurs normes, tensions ou conditions, conserve une valeur principale comme selector et classe les autres en context sauf exigence explicite;\n"
                "- ne transforme pas toutes les dimensions, normes ou valeurs secondaires en contraintes obligatoires;\n"
                "- conserve les valeurs, tolerances, dimensions, materiaux, normes, interfaces et conditions explicitement presentes;\n"
                "- n'ajoute aucune caracteristique propre a une famille particuliere;\n"
                "- produis de 1 a 3 requetes descriptives utilisables dans un catalogue cible;\n"
                "- lorsque la fiche n\'est pas en anglais, inclus si possible une requete en anglais technique international avec l\'acronyme ou l\'abreviation courante du type de produit;\n"
                "- les requetes ne doivent contenir ni fabricant source, ni gamme source, ni reference source;\n"
                "- les requetes ne doivent contenir aucune marque, famille ou reference cible non presente dans la fiche source;\n"
                "- n'invente aucune valeur.\n\n"
                f"FICHE PRODUIT:\n{bounded_text}"
            ),
            max_tokens=1800,
        )
        result = _normalise_need_response(raw_result, bounded_text)
        if not result["search_queries"]:
            received_keys = ", ".join(sorted(raw_result)) or "aucune"
            raise LLMResponseError(
                "Le JSON du besoin ne permet pas de construire une requete; "
                f"cles recues: {received_keys}"
            )
        return result

    def select_reference(
        self,
        need: dict[str, Any],
        chunks: list[RankedChunk],
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("aucun passage catalogue n'a ete recupere")

        context_parts: list[str] = []
        for item in chunks:
            section = item.chunk.section_title or item.chunk.section_id or "non classee"
            context_parts.append(
                f"[SECTION {section} | KIND {item.chunk.kind} | "
                f"PAGE {item.chunk.page_number} | {item.chunk.chunk_id}]\n"
                f"{item.chunk.text}"
            )
        context = "\n\n".join(context_parts)
        validation_feedback = need.get("validation_feedback")
        previous_result = need.get("previous_result")
        repair_instruction = ""
        if isinstance(validation_feedback, list) and validation_feedback:
            empty_retry = "empty_candidate_list" in validation_feedback
            heading = (
                "CORRECTION APRES CONTROLE AUTOMATIQUE"
                if empty_retry
                else "CORRECTION APRES VALIDATION DETERMINISTE"
            )
            repair_instruction = (
                f"\n{heading}:\n"
                f"Erreurs: {json.dumps(validation_feedback, ensure_ascii=False)}\n"
                f"Reponse precedente rejetee: {json.dumps(previous_result, ensure_ascii=False)}\n"
                + (
                    "La premiere analyse a retourne candidates=[] alors que des passages "
                    "catalogue ont ete recuperes. Reanalyse toutes les tables de variantes "
                    "et toutes les cles de commande. Si aucune correspondance exacte "
                    "n existe, conserve les alternatives les plus proches et liste chaque "
                    "ecart dans differences. Retourne candidates=[] uniquement si aucune "
                    "reference commandable ne peut reellement etre prouvee ou construite.\n"
                    if empty_retry
                    else
                    "Corrige la reponse. Un prefixe de famille ou une reference partielle "
                    "n'est jamais une reference de commande complete.\n"
                )
            )

        selection_prompt = (
            "BESOIN TECHNIQUE SANS IDENTIFIANTS DE LA FICHE SOURCE:\n"
            f"{json.dumps(need, ensure_ascii=False, sort_keys=True)}\n"
            f"{repair_instruction}\n"
            "PASSAGES DU CATALOGUE:\n"
            f"{context}\n\n"
            "Retourne exactement cet objet JSON:\n"
            "{\n"
            '  "candidates": [{\n'
            '    "rank": 1,\n'
            '    "reference": "identifiant catalogue exact, sans description",\n'
            '    "variant": "nom de variante explicitement documente ou null",\n'
            '    "match_level": "closest" ou "equivalent" ou "alternative",\n'
            '    "reference_mode": "explicit" ou "constructed" ou null,\n'
            '    "reference_parts": [{"position": 1, "code": "segment exact", "page": 1, "evidence": "signification visible"}],\n'
            '    "catalogue_pages": [numeros de pages recuperes et reellement utilises],\n'
            '    "evidence": ["courtes preuves textuelles visibles dans ces pages"],\n'
            '    "reason": "pourquoi ce candidat est classe ici",\n'
            '    "differences": ["tous les ecarts utiles par rapport au besoin ou aux autres candidats"],\n'
            '    "mandatory_differences": ["ecart a une contrainte provenance=client ou derived_required; liste vide si aucun"]\n'
            '  }],\n'
            '  "explanation": "resume du classement et des hypotheses"\n'
            "}\n"
            "Contraintes obligatoires:\n"
            "- retourne de 0 a 5 candidats pertinents, sans doublon, classes du plus proche au moins proche;\n"
            "- le rang 1 est le meilleur candidat documente; les autres sont des alternatives compatibles;\n"
            "- les exigences provenance=client ou provenance=derived_required avec role=constraint definissent l equivalence directe;\n"
            "- un candidat est equivalent direct seulement si toutes les contraintes constraint_scope=hard ou constraint_scope=direct_equivalence sont satisfaites;\n"
            "- provenance=derived_required vient d une reference exacte prouvee: reporte tout ecart dans mandatory_differences; un ecart doit interdire match_level=equivalent, mais le candidat pertinent reste une alternative documentee;\n"
            "- les attributs provenance=source_product, meme role=selector, servent a classer la proximite technique mais ne doivent pas provoquer candidates=[] lorsqu une alternative documentee existe;\n"
            "- si aucune correspondance exacte n existe, conserve les alternatives les plus proches, indique match_level=closest ou alternative et liste tous les ecarts utiles dans differences;\n"
            "- un attribut sans role, ou dont le role est absent, n'est pas automatiquement obligatoire;\n"
            "- n'invente jamais qu'une variante Deluxe, premium ou standard est plus proche: justifie le classement avec le besoin et les preuves;\n"
            "- si plusieurs variantes sont techniquement a egalite, conserve-les et indique l'egalite ou le critere de departage;\n"
            "- catalogue_pages doit contenir au moins une PAGE presente dans le contexte pour chaque candidat;\n"
            "- si l'identifiant complet est ecrit tel quel, utilise reference_mode=explicit et reference_parts=[];\n"
            "- si l'identifiant est construit avec une cle de commande, utilise reference_mode=constructed et donne chaque segment dans l'ordre;\n"
            "- si un bloc STRUCTURED ORDER TEMPLATE indique SLOT COUNT=N, la reference construite et reference_parts doivent contenir exactement les N segments; ne retourne jamais seulement le prefixe, la gamme ou le modele de base;\n"
            "- EXAMPLE CODES indique uniquement l'ordre et le nombre des segments: remplace chaque valeur d'exemple par le code correspondant au besoin;\n"
            "- chaque segment constructed doit etre visible sur la page indiquee et leur concatenation doit former exactement reference;\n"
            "- reference contient seulement le code catalogue, sans courant, tension, courbe ou commentaire ajoute;\n"
            "- evidence doit citer les informations visibles dans les pages indiquees;\n"
            "- n'utilise jamais une reference provenant de la fiche source;\n"
            "- ne demande pas une egalite exacte pour un attribut role=context et ne l'utilise pas comme motif de rejet;\n"
            "- une difference de notation, de tension nominale conventionnelle ou de norme d'essai ne suffit pas a rejeter une variante si les selecteurs fonctionnels principaux correspondent;\n"
            "- pour les tables, utilise les blocs structures TABLE VARIANT COLUMN ou TABLE CODE MAP afin de ne pas melanger les colonnes;\n"
            "- candidates=[] signifie qu aucune reference commandable pertinente n est visible; une reference proche avec un ecart documente doit etre retournee comme alternative, pas rejetee."
        )

        result = self._chat_json(
            system=(
                "Tu identifies et classes toutes les references pertinentes dans un catalogue industriel, "
                "quel que soit le domaine du produit. Le contexte recupere est la seule source autorisee. "
                "Cherche chaque variante correspondante la plus proche puis les autres variantes pertinentes dans le catalogue cible, sans exiger "
                "que le fabricant cible utilise exactement la meme notation, la meme norme ou le meme libelle. "
                "Chaque reference doit etre uniquement l'identifiant catalogue exact, sans description technique ajoutee. "
                "Elle doit apparaitre explicitement dans le contexte, ou etre assemblable uniquement si une regle de codification "
                "complete et toutes ses valeurs sont explicitement presentes. Les blocs TABLE VARIANT COLUMN preservent les colonnes: "
                "ne melange jamais des valeurs provenant de deux colonnes differentes. Utilise en priorite les attributs role=selector; "
                "les attributs role=constraint explicitement obligatoires doivent etre compatibles ou satisfaits, tandis que les attributs "
                "role=context ne doivent jamais bloquer le choix. Les exigences provenance=client ou provenance=derived_required avec role=constraint definissent l equivalence directe, sans supprimer une alternative pertinente. Interprete aussi la provenance sans la confondre avec le role: "
                "provenance=client marque une demande explicite du client; provenance=derived_required marque une contrainte derivee d une reference exacte et prouvee; provenance=source_product marque une caracteristique attestee "
                "du produit source qui guide le classement mais peut differer pour une alternative proche si l ecart est explicite; provenance=optional ne doit jamais forcer une variante; "
                "provenance=to_confirm et open_questions signalent une ambiguite qui ne doit pas devenir une obligation. "
                "Quand une ambiguite ouvre plusieurs branches du catalogue, conserve toutes les branches compatibles et explique leur difference. "
                "Classe les candidats par proximite technique avec le produit source et les contraintes explicites. Une version premium, Deluxe "
                "ou renforcee ne doit passer devant une version standard que si des caracteristiques documentees du produit source justifient cette "
                "proximite; sinon signale l'egalite technique ou utilise un ordre deterministe clairement explique. Utilise PAGE CONTEXT pour identifier "
                "la famille et la fonction de chaque tableau. Ne retiens pas un produit qui ajoute une fonction technique majeure absente du type demande, "
                "sauf si aucune variante de meme fonction n'existe. Reponds uniquement par un objet JSON."
            ),
            user=selection_prompt,
            max_tokens=4200,
        )

        raw_candidates = result.get("candidates")
        if isinstance(raw_candidates, list):
            candidates = [
                _normalise_selected_candidate(item, default_rank=index)
                for index, item in enumerate(raw_candidates, start=1)
            ]
            candidates.sort(key=lambda item: (item["rank"], item["reference"]))
            seen_references: set[str] = set()
            unique_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                key = candidate["reference"].casefold()
                if key in seen_references:
                    continue
                seen_references.add(key)
                candidate["rank"] = len(unique_candidates) + 1
                unique_candidates.append(candidate)
                if len(unique_candidates) >= 5:
                    break
            explanation = _normalise_generated_text(
                result.get("explanation", "")
            )
            if not unique_candidates:
                return {
                    "reference": None,
                    "reference_mode": None,
                    "reference_parts": [],
                    "catalogue_pages": [],
                    "evidence": [],
                    "explanation": explanation.strip(),
                    "candidates": [],
                }
            best = unique_candidates[0]
            return {
                "reference": best["reference"],
                "reference_mode": best["reference_mode"],
                "reference_parts": best["reference_parts"],
                "catalogue_pages": best["catalogue_pages"],
                "evidence": best["evidence"],
                "explanation": explanation.strip() or best["reason"],
                "candidates": unique_candidates,
            }

        # Compatibilite avec les anciennes reponses V12 a reference unique.
        reference = result.get("reference")
        if reference is not None and not isinstance(reference, str):
            raise LLMResponseError("reference doit etre une chaine JSON ou null")
        if isinstance(reference, str):
            result["reference"] = reference.strip() or None
        result["catalogue_pages"] = _normalise_catalogue_pages(
            result.get("catalogue_pages")
        )
        result["evidence"] = _normalise_generated_text_list(
            result.get("evidence")
        )
        explanation = _normalise_generated_text(result.get("explanation"))
        reference_mode = result.get("reference_mode")
        if reference_mode is not None:
            if not isinstance(reference_mode, str) or reference_mode not in _MODE_ALIASES:
                raise LLMResponseError(
                    "reference_mode doit etre explicit, constructed ou null"
                )
            reference_mode = _MODE_ALIASES[reference_mode]
        reference_parts = _normalise_reference_parts(
            result.get("reference_parts", [])
        )
        if result.get("reference") is None:
            reference_mode = None
            reference_parts = []
        elif reference_mode is None and reference_parts:
            reference_mode = "constructed"
        elif reference_mode == "explicit":
            reference_parts = []
        result["reference_parts"] = reference_parts
        result["reference_mode"] = reference_mode
        result["explanation"] = explanation
        return result

    def repair_reference(
        self,
        need: dict[str, Any],
        chunks: list[RankedChunk],
        previous_result: dict[str, Any],
        validation_errors: list[str],
    ) -> dict[str, Any]:
        """Retry once with deterministic validation feedback."""

        repair_need = dict(need)
        repair_need["validation_feedback"] = list(validation_errors)
        repair_need["previous_result"] = previous_result
        return self.select_reference(repair_need, chunks)


class ResilientNvidiaChatClient:
    """Bounded primary/fallback router for structured catalogue generation."""

    def __init__(self, primary: Any, fallback: Any | None = None) -> None:
        if primary is None:
            raise ValueError("primary client is required")
        self.primary = primary
        self.fallback = fallback
        self.model = str(getattr(primary, "model", ""))
        self._model_used = self.model
        self._fallback_used = False
        self._fallback_reason: str | None = None
        self._stage_models: dict[str, str] = {}
        self._active_selection_client = primary

    @property
    def model_used(self) -> str:
        return self._model_used

    @property
    def kwargs(self) -> dict[str, Any]:
        """Compatibility view used by injected CLI test clients."""
        raw = getattr(self.primary, "kwargs", {})
        return dict(raw) if isinstance(raw, dict) else {}

    @property
    def reasoning_config(self) -> dict[str, Any]:
        config = getattr(self.primary, "reasoning_config", {})
        return dict(config) if isinstance(config, dict) else {}

    @property
    def json_repair_config(self) -> dict[str, int]:
        keys = {
            "attempts", "successes", "local_attempts", "local_successes",
            "external_attempts", "external_successes", "truncated_responses",
        }
        result = {key: 0 for key in keys}
        for client in (self.primary, self.fallback):
            config = getattr(client, "json_repair_config", None)
            if not isinstance(config, dict):
                continue
            for key in keys:
                result[key] += int(config.get(key, 0) or 0)
        return result

    @property
    def model_routing_config(self) -> dict[str, Any]:
        return {
            "primary_model": str(getattr(self.primary, "model", "")) or None,
            "fallback_model": str(getattr(self.fallback, "model", "")) or None,
            "model_used": self._model_used or None,
            "fallback_used": self._fallback_used,
            "fallback_reason": self._fallback_reason,
            "stage_models": dict(self._stage_models),
        }

    def _use_fallback(self, stage: str, reason: str, call):
        if self.fallback is None:
            raise LLMResponseError(reason)
        self._fallback_used = True
        self._fallback_reason = reason
        self._model_used = str(getattr(self.fallback, "model", ""))
        self._stage_models[stage] = self._model_used
        if stage == "select_reference":
            self._active_selection_client = self.fallback
        return call(self.fallback)

    def extract_need(self, product_text: str) -> dict[str, Any]:
        try:
            result = self.primary.extract_need(product_text)
        except LLMResponseError as exc:
            return self._use_fallback(
                "extract_need",
                f"primary_structured_output_error:{exc}",
                lambda client: client.extract_need(product_text),
            )
        self._model_used = str(getattr(self.primary, "model", ""))
        self._stage_models["extract_need"] = self._model_used
        return result

    def select_reference(self, need: dict[str, Any], chunks: list[RankedChunk]) -> dict[str, Any]:
        try:
            result = self.primary.select_reference(need, chunks)
        except LLMResponseError as exc:
            return self._use_fallback(
                "select_reference",
                f"primary_structured_output_error:{exc}",
                lambda client: client.select_reference(need, chunks),
            )
        unusable = bool(
            chunks
            and isinstance(result, dict)
            and result.get("reference") is None
            and not result.get("candidates")
        )
        if unusable and self.fallback is not None:
            return self._use_fallback(
                "select_reference",
                "primary_empty_candidates_with_context",
                lambda client: client.select_reference(need, chunks),
            )
        self._model_used = str(getattr(self.primary, "model", ""))
        self._stage_models["select_reference"] = self._model_used
        self._active_selection_client = self.primary
        return result

    def repair_reference(
        self,
        need: dict[str, Any],
        chunks: list[RankedChunk],
        previous_result: dict[str, Any],
        validation_errors: list[str],
    ) -> dict[str, Any]:
        active = self._active_selection_client
        repair = getattr(active, "repair_reference", None)
        if callable(repair):
            try:
                return repair(need, chunks, previous_result, validation_errors)
            except LLMResponseError as exc:
                if active is self.primary and self.fallback is not None:
                    fallback_repair = getattr(self.fallback, "repair_reference", None)
                    if callable(fallback_repair):
                        return self._use_fallback(
                            "repair_reference",
                            f"primary_repair_error:{exc}",
                            lambda _client: fallback_repair(
                                need, chunks, previous_result, validation_errors
                            ),
                        )
                raise
        return previous_result
