"""Tolerance helpers for free-form explanatory fields returned by LLMs."""

from __future__ import annotations

from typing import Any


_PREFERRED_KEYS = (
    "summary",
    "resume",
    "résumé",
    "explanation",
    "reason",
    "text",
    "texte",
    "message",
    "detail",
    "details",
)


def normalise_explanation(value: Any, *, fallback: str = "") -> str:
    """Return readable text from a string, list, or small JSON object.

    Generative models sometimes emit an explanation as a list of sentences or
    as an object such as ``{"summary": "...", "details": [...]}`` even when
    the requested schema says string.  Explanations are non-authoritative prose,
    so coercing those common shapes is safer than discarding an otherwise valid
    catalogue result.
    """

    parts: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        cleaned = " ".join(text.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            parts.append(cleaned)

    def collect(item: Any) -> None:
        if isinstance(item, str):
            add(item)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
            return
        if isinstance(item, dict):
            matched = False
            for key in _PREFERRED_KEYS:
                if key in item:
                    matched = True
                    collect(item[key])
            if not matched:
                for child in item.values():
                    collect(child)

    collect(value)
    if not parts and isinstance(fallback, str):
        add(fallback)
    return " ".join(parts)


def normalise_text_list(value: Any) -> list[str]:
    """Return a deduplicated list from one text or common generated shapes."""

    raw_items = value if isinstance(value, (list, tuple)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = normalise_explanation(item)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
