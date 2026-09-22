"""Quality controls for the context sent to the catalogue LLM."""

from __future__ import annotations

import re
import unicodedata

from .retrieval import RankedChunk


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "")).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _shingles(text: str, size: int = 4) -> set[tuple[str, ...]]:
    tokens = _normalise(text).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _variant_identity(text: str) -> str:
    match = re.search(
        r"(?:^|\n)(?:model|reference|type|designation|article)\s*:\s*([^\n]+)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return _normalise(match.group(1)) if match else ""


def _must_preserve_pair(left: RankedChunk, right: RankedChunk) -> bool:
    if left.chunk.kind != right.chunk.kind:
        return True
    if left.chunk.kind == "variant" or "structured table variant column" in left.chunk.text.casefold():
        left_identity = _variant_identity(left.chunk.text)
        right_identity = _variant_identity(right.chunk.text)
        if left_identity and right_identity and left_identity != right_identity:
            return True
    # Ordering maps can look nearly identical across product families; never
    # collapse them unless they carry the same section and exact content.
    if left.chunk.kind == "ordering" and left.chunk.section_id != right.chunk.section_id:
        return True
    return False


def deduplicate_ranked_chunks(
    items: list[RankedChunk],
    *,
    similarity_threshold: float = 0.94,
) -> tuple[list[RankedChunk], list[str]]:
    """Drop exact or near-duplicate prose while preserving distinct variants."""

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold doit etre compris entre 0 et 1")
    kept: list[RankedChunk] = []
    kept_signatures: list[tuple[str, set[tuple[str, ...]]]] = []
    removed: list[str] = []
    for item in items:
        normalised = _normalise(item.chunk.text)
        signature = _shingles(item.chunk.text)
        duplicate = False
        for previous, (previous_normalised, previous_signature) in zip(kept, kept_signatures, strict=False):
            if _must_preserve_pair(item, previous):
                continue
            if normalised == previous_normalised or _jaccard(signature, previous_signature) >= similarity_threshold:
                duplicate = True
                break
        if duplicate:
            removed.append(item.chunk.chunk_id)
            continue
        kept.append(item)
        kept_signatures.append((normalised, signature))
    return kept, removed


def diversify_ranked_chunks(
    items: list[RankedChunk],
    *,
    top_k: int,
    max_per_page: int = 2,
    max_per_section: int = 4,
) -> list[RankedChunk]:
    """Pack a diverse context while reserving structural evidence."""

    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if max_per_page <= 0 or max_per_section <= 0:
        raise ValueError("les limites de diversite doivent etre positives")
    if not items:
        return []

    result: list[RankedChunk] = []
    seen: set[str] = set()
    page_counts: dict[int, int] = {}
    section_counts: dict[str, int] = {}

    def add(item: RankedChunk, *, force: bool = False) -> bool:
        if item.chunk.chunk_id in seen or len(result) >= top_k:
            return False
        section_key = item.chunk.section_id or f"page-{item.chunk.page_number}"
        if not force:
            if page_counts.get(item.chunk.page_number, 0) >= max_per_page:
                return False
            if section_counts.get(section_key, 0) >= max_per_section:
                return False
        result.append(item)
        seen.add(item.chunk.chunk_id)
        page_counts[item.chunk.page_number] = page_counts.get(item.chunk.page_number, 0) + 1
        section_counts[section_key] = section_counts.get(section_key, 0) + 1
        return True

    # Keep one ordering/code map first when available, then one selection table.
    for reserved_kind in ("ordering", "selection"):
        reserved = next((item for item in items if item.chunk.kind == reserved_kind), None)
        if reserved is not None:
            add(reserved, force=True)

    for item in items:
        add(item)
    # If diversity constraints under-fill the context, relax them deterministically.
    if len(result) < min(top_k, len(items)):
        for item in items:
            add(item, force=True)
    return result[:top_k]
