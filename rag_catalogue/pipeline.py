"""End-to-end orchestration for ranked catalogue references."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from .pdf_text import CatalogueChunk, PdfInspection, PdfPage, chunk_pages, extract_pdf_pages, inspect_pdf
from .catalogue_map import CatalogueMap, CatalogueSection, build_catalogue_map, section_lookup
from .catalogue_navigation import (
    CatalogueNavigationMap,
    assess_local_evidence,
    build_navigation_map,
    expand_route_pages,
    navigation_chunks,
    plan_navigation,
)
from .context_quality import deduplicate_ranked_chunks, diversify_ranked_chunks
from .provenance_need import merge_structured_provenance_into_need
from .source_bundle import load_product_source
from .result_text import normalise_explanation, normalise_text_list
from .dense_retrieval import load_or_create_dense_index
from .retrieval import (
    HybridRetriever,
    RankedChunk,
    reciprocal_rank_fusion,
    rerank_ranked_chunks,
    rerank_ranked_chunks_batched,
    weighted_rank_fusion,
)


_ORDERING_HEADING = re.compile(
    r"\b(?:ordering|order)\s+(?:information|guidelines?|code)|"
    r"\bhow\s+to\s+order\b|\bcatalog(?:ue)?\s+(?:number|code)\b|"
    r"\bcodification\b|\binformations?\s+de\s+commande\b|"
    r"\breference\s+de\s+commande\b",
    flags=re.IGNORECASE,
)
_SELECTION_HEADING = re.compile(
    r"\bselection\s+(?:table|chart|guide)\b|"
    r"\btableau\s+de\s+selection\b|\bguide\s+de\s+selection\b",
    flags=re.IGNORECASE,
)

_MARKER_STOPWORDS = {
    "AC", "DC", "IEC", "EN", "ISO", "DIN", "IP",
    "MODEL", "TYPE", "FRAME", "STANDARD", "SELECTION",
    "ORDERING", "INFORMATION", "GUIDELINES", "REFERENCE", "CURRENT",
}
_PRIMARY_ORDERING_HEADING = re.compile(
    r"\bordering\s+guidelines?\b|\bhow\s+to\s+order\b|"
    r"\border(?:ing)?\s+code\b|\bcatalog(?:ue)?\s+(?:number|code)\b|"
    r"\bcodification\b|\breference\s+de\s+commande\b",
    flags=re.IGNORECASE,
)
_ACCESSORY_HEADING = re.compile(
    r"^.{0,120}\baccessor(?:y|ies|ie|ies)\b",
    flags=re.IGNORECASE,
)
_ORDERING_SCHEMA_TERMS = (
    "type", "model", "frame", "size", "variant", "option", "mounting",
    "frequency", "rated current", "voltage", "material", "connection",
    "number of poles", "characteristic", "code", "reference",
)


@dataclass(frozen=True, slots=True)
class CatalogueIndexBundle:
    pages: tuple[PdfPage, ...]
    chunks: tuple[CatalogueChunk, ...]
    retriever: HybridRetriever
    catalogue_map: CatalogueMap
    navigation_map: CatalogueNavigationMap
    inspection: PdfInspection


def _structural_markers(text: str) -> set[str]:
    """Extract rare-looking family/model markers without knowing a product domain."""

    markers: set[str] = set()
    for raw in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", text):
        token = raw.strip("_-")
        if len(token) < 3 or token in _MARKER_STOPWORDS:
            continue
        if not any(char.isalpha() for char in token):
            continue
        markers.add(token)
        prefix_match = re.match(r"[A-Z]{3,}", token)
        if prefix_match:
            prefix = prefix_match.group(0)
            if prefix not in _MARKER_STOPWORDS:
                markers.add(prefix)
    return markers



def _ordering_product_identity(text: str) -> tuple[str, str]:
    """Return the first product prefix and its visible description in an order map."""

    marker = "STRUCTURED TABLE CODE MAP"
    structured = text.split(marker, 1)[1] if marker in text else text
    for match in re.finditer(
        r"(?:^|\n)\s*([A-Z][A-Z_-]{2,11})\s*:\s*([^\n]{2,240})",
        structured,
    ):
        code = match.group(1).strip("_-")
        description = re.sub(r"\s+", " ", match.group(2)).strip(" -–—:;,.")
        if code in _MARKER_STOPWORDS or not description:
            continue
        return code, description
    return "", ""


def _ordering_family_codes(text: str) -> set[str]:
    """Extract the leading alphabetic type code from a structured order map."""

    code, _description = _ordering_product_identity(text)
    return {code} if code else set()


def _search_words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]{2,}", folded)


def _ordering_scope_rank(text: str, queries: Iterable[str]) -> tuple[float, int, str]:
    """Rank order maps by requested function, then prefer the least composite scope.

    A catalogue can contain a base product and a more complex device sharing most
    electrical values. Exact product prefixes/acronyms and description overlap win.
    When the query language gives no lexical match, the shorter visible product
    description is the safer base-function default instead of a composite variant.
    """

    code, description = _ordering_product_identity(text)
    if not description:
        return (0.0, 10_000, "")

    query_text = " ".join(str(query) for query in queries if isinstance(query, str))
    query_folded = " ".join(_search_words(query_text))
    query_words = set(query_folded.split())
    description_words_list = _search_words(description)
    description_words = set(description_words_list)

    direct_score = 0.0
    if code and re.search(rf"(?<![a-z0-9]){re.escape(code.casefold())}(?![a-z0-9])", query_text.casefold()):
        direct_score += 12.0

    first_line = text.splitlines()[0] if text else ""
    context_acronyms = {
        token
        for token in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{1,11}(?![A-Za-z0-9])", first_line)
        if token not in _MARKER_STOPWORDS
    }
    for acronym in context_acronyms:
        if re.search(
            rf"(?<![a-z0-9]){re.escape(acronym.casefold())}(?![a-z0-9])",
            query_text.casefold(),
        ):
            direct_score += 9.0
            break

    overlap = description_words & query_words
    if description_words:
        direct_score += 5.0 * len(overlap) / len(description_words)
    direct_score += 0.25 * len(overlap)

    normalised_description = " ".join(description_words_list)
    if normalised_description and normalised_description in query_folded:
        direct_score += 4.0

    composite_connectors = {"with", "plus", "combined", "integrated", "including"}
    complexity = len(description_words_list) + 2 * sum(
        1 for word in description_words_list if word in composite_connectors
    )
    return (-direct_score, complexity, normalised_description)



_CIRCLED_SLOT_SYMBOLS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_ORDER_TEMPLATE_COUNT = re.compile(r"(?m)^SLOT COUNT:\s*(\d+)\s*$")


def _order_template_block(text: str) -> str:
    """Extract a numbered ordering example without knowing the product family.

    Many industrial catalogues print one complete sample code followed by circled
    slot numbers and the corresponding field names.  Keeping that template lets
    the LLM and the validator distinguish a complete order reference from a mere
    family prefix.
    """

    symbol_number = {
        symbol: index + 1
        for index, symbol in enumerate(_CIRCLED_SLOT_SYMBOLS)
    }
    matches = list(re.finditer(
        f"[{re.escape(_CIRCLED_SLOT_SYMBOLS)}]",
        text,
    ))
    groups: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    for match in matches:
        if not current:
            current = [match]
            continue
        previous = current[-1]
        contiguous = not text[previous.end():match.start()].strip()
        sequential = (
            symbol_number[match.group()]
            == symbol_number[previous.group()] + 1
        )
        if contiguous and sequential:
            current.append(match)
        else:
            groups.append(current)
            current = [match]
    if current:
        groups.append(current)

    candidates = [
        group for group in groups
        if len(group) >= 3 and symbol_number[group[0].group()] == 1
    ]
    if not candidates:
        return ""
    run = max(candidates, key=len)
    slot_count = len(run)

    before = text[:run[0].start()]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/+:-]*", before)
    if len(tokens) < slot_count:
        return ""
    example_codes = tokens[-slot_count:]
    if any(len(code) > 40 for code in example_codes):
        return ""

    after = text[run[-1].end():]
    fields: list[str] = []
    cursor = 0
    for index in range(slot_count):
        symbol = _CIRCLED_SLOT_SYMBOLS[index]
        position = after.find(symbol, cursor)
        if position < 0:
            fields = []
            break
        field = re.sub(r"\s+", " ", after[cursor:position]).strip(" :-")
        if not field or len(field) > 160:
            fields = []
            break
        fields.append(field)
        cursor = position + len(symbol)

    lines = [
        "STRUCTURED ORDER TEMPLATE",
        f"SLOT COUNT: {slot_count}",
        "EXAMPLE CODES: " + " | ".join(example_codes),
    ]
    if len(fields) == slot_count:
        lines.append(
            "SLOT FIELDS: "
            + " | ".join(
                f"{index}={field}"
                for index, field in enumerate(fields, start=1)
            )
        )
    return "\n".join(lines)




_CODE_MAP_LINE = re.compile(
    r"(?m)^\s*([A-Za-z0-9][A-Za-z0-9._/+\-]{0,39})\s*:\s*(\S.*)$"
)
_SLOT_ALLOWED_CODES = re.compile(
    r"(?m)^SLOT\s+(\d+)\s+ALLOWED CODES:\s*(.+?)\s*$"
)


def _codes_in_structured_block(block: str) -> list[str]:
    """Return the catalogue codes printed as keys in one structured table block."""

    codes: list[str] = []
    seen: set[str] = set()
    for match in _CODE_MAP_LINE.finditer(block or ""):
        code = match.group(1).strip()
        key = _reference_key(code)
        if key and key not in seen:
            seen.add(key)
            codes.append(code)
    return codes


def _structured_slot_code_map(template: str, structured_blocks: list[str]) -> str:
    """Associate each ordering slot with its own catalogue code column.

    The association is learned from the complete example code printed by the
    catalogue.  It is therefore product-agnostic: the validator never knows
    that a slot is a frame, a curve, a size, or a material.  It only verifies
    that a code was selected from the table column demonstrated for that slot.
    """

    example_match = re.search(r"(?m)^EXAMPLE CODES:\s*(.+?)\s*$", template or "")
    if example_match is None:
        return ""
    examples = [item.strip() for item in example_match.group(1).split("|")]
    if len(examples) < 2 or any(not item for item in examples):
        return ""

    block_codes = [_codes_in_structured_block(block) for block in structured_blocks]
    block_keys = [{_reference_key(code) for code in codes} for codes in block_codes]
    candidates: dict[int, list[int]] = {}
    for slot, example in enumerate(examples, start=1):
        example_key = _reference_key(example)
        candidates[slot] = [
            index
            for index, keys in enumerate(block_keys)
            if example_key and example_key in keys
        ]

    assignments: dict[int, int] = {}
    changed = True
    while changed:
        changed = False
        for slot, possible in candidates.items():
            if slot in assignments:
                continue
            example_key = _reference_key(examples[slot - 1])
            remaining = []
            for block_index in possible:
                claimed_by_other_code = any(
                    assigned_block == block_index
                    and _reference_key(examples[other_slot - 1]) != example_key
                    for other_slot, assigned_block in assignments.items()
                )
                if not claimed_by_other_code:
                    remaining.append(block_index)
            candidates[slot] = remaining
            if len(remaining) == 1:
                assignments[slot] = remaining[0]
                changed = True

    lines = ["STRUCTURED SLOT CODE MAP"]
    for slot in range(1, len(examples) + 1):
        block_index = assignments.get(slot)
        if block_index is None:
            continue
        codes = block_codes[block_index]
        if codes:
            lines.append(f"SLOT {slot} ALLOWED CODES: " + " | ".join(codes))
    return "\n".join(lines) if len(lines) > 1 else ""


def _declared_slot_code_maps(
    pages: list[int],
    page_text: dict[int, str],
) -> dict[tuple[int, int], set[str]]:
    """Read deterministic per-slot allowed-code maps from retrieved pages."""

    result: dict[tuple[int, int], set[str]] = {}
    for page in pages:
        text = page_text.get(page, "")
        for match in _SLOT_ALLOWED_CODES.finditer(text):
            position = int(match.group(1))
            codes = {
                _reference_key(item.strip())
                for item in match.group(2).split("|")
                if _reference_key(item.strip())
            }
            if codes:
                result.setdefault((page, position), set()).update(codes)
    return result


def _declared_order_slot_counts(
    pages: list[int],
    page_text: dict[int, str],
) -> set[int]:
    counts: set[int] = set()
    for page in pages:
        text = page_text.get(page, "")
        if not _ORDERING_HEADING.search(text):
            continue
        for match in _ORDER_TEMPLATE_COUNT.finditer(text):
            value = int(match.group(1))
            if 2 <= value <= 50:
                counts.add(value)
    return counts


def _page_context_heading(text: str, *, maximum: int = 220) -> str:
    """Keep a page heading without reintroducing flattened table columns."""

    normalised = re.sub(r"\s+", " ", text).strip()
    if not normalised:
        return ""
    parenthetical_heading = re.match(r"^(.{1,120}?\([^)]{1,50}\))", normalised)
    if parenthetical_heading:
        return parenthetical_heading.group(1).strip()
    cut = len(normalised)
    for marker in (" Model ", " Reference Standard ", " Rated Current ", " ①", " Type ①"):
        position = normalised.find(marker)
        if position > 0:
            cut = min(cut, position)
    return normalised[:cut][:maximum].strip()

def _ordering_page_quality(text: str) -> float:
    """Score whether a page is a primary order-code schema rather than an accessory list."""

    if not _ORDERING_HEADING.search(text):
        return 0.0
    folded = re.sub(r"\s+", " ", text.casefold()).strip()
    score = 1.0
    if _PRIMARY_ORDERING_HEADING.search(text):
        score += 5.0
    heading = _ORDERING_HEADING.search(text)
    if heading is not None and heading.start() <= 80:
        score += 1.0
    if _ACCESSORY_HEADING.search(text):
        score -= 6.0

    circled_fields = len(re.findall(r"[①-⑳]", text))
    if circled_fields:
        score += min(5.0, circled_fields / 2.0)

    schema_hits = sum(1 for term in _ORDERING_SCHEMA_TERMS if term in folded)
    score += min(4.0, schema_hits * 0.4)
    return max(0.0, score)


def _ordering_expansions(
    core: list[RankedChunk],
    chunks: list[CatalogueChunk],
    ranked_by_id: dict[str, RankedChunk],
    queries: list[str],
    *,
    maximum_pages: int = 3,
) -> list[RankedChunk]:
    """Add distant base-product ordering pages, ranked by query and structure.

    Ordering pages are often far from the selection table. Accessory ordering
    pages can share more exact family codes than the base-product coding page,
    so non-accessory ordering guides are considered first.
    """

    if not core or maximum_pages <= 0:
        return []

    ordering_chunks = [chunk for chunk in chunks if _ORDERING_HEADING.search(chunk.text)]
    if not ordering_chunks:
        return []

    query_relevance = {chunk.chunk_id: 0.0 for chunk in ordering_chunks}
    query_text = " ".join(queries)

    def heading_identity_bonus(text: str) -> float:
        """Reward an exact product-class acronym in the order-page heading.

        This is domain-neutral: ``MCB`` in a query should prefer a page headed
        ``MCB Ordering Information`` over a neighbouring ``RCCB`` or ``RCBO``
        page, even when all pages repeat the same current and voltage values.
        """

        heading = text[:220]
        bonus = 0.0
        for token in re.findall(
            r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{1,11}(?![A-Za-z0-9])",
            heading,
        ):
            if token in _MARKER_STOPWORDS:
                continue
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                query_text,
                flags=re.IGNORECASE,
            ):
                bonus = max(bonus, 2.0)
        return bonus
    ordering_retriever = HybridRetriever(ordering_chunks)
    for query in queries:
        for candidate in ordering_retriever.search(
            query,
            top_k=len(ordering_chunks),
            weights=(0.45, 0.35, 0.20, 0.0),
        ):
            query_relevance[candidate.chunk.chunk_id] = max(
                query_relevance[candidate.chunk.chunk_id],
                candidate.score,
            )

    marker_core = [
        item for item in core
        if not _ORDERING_HEADING.search(item.chunk.text)
    ][:4]
    core_markers_by_rank = [
        _structural_markers(item.chunk.text) for item in marker_core
    ]
    candidates: list[tuple[bool, bool, bool, float, int, CatalogueChunk]] = []
    for chunk in ordering_chunks:
        chunk_markers = _structural_markers(chunk.text)
        best_structural = 0.0
        best_rank = len(core) + 1
        for rank, markers in enumerate(core_markers_by_rank, start=1):
            overlap = markers & chunk_markers
            if not overlap:
                continue
            specificity = sum(
                1.5 if any(char.isdigit() for char in token) else 1.0
                for token in overlap
            )
            score = specificity + 1.0 / rank
            if score > best_structural:
                best_structural = score
                best_rank = rank

        relevance = query_relevance.get(chunk.chunk_id, 0.0)
        heading = chunk.text[:260].casefold()
        accessory = bool(re.search(r"\baccessor(?:y|ies)\b|\baccessoire", heading))
        primary_guide = bool(_PRIMARY_ORDERING_HEADING.search(chunk.text))
        guide_bonus = 1.5 if primary_guide else 0.5
        base_bonus = 1.0 if not accessory else -2.0
        # Structural markers establish that an ordering page may belong to a
        # retrieved family, but repeated manufacturer/footer acronyms must not
        # outweigh the user's actual technical query. Cap that contribution so
        # query relevance decides between sibling ordering schemes (for example,
        # standard versus deluxe variants).
        structural_contribution = min(best_structural, 1.5)
        total = (
            structural_contribution
            + 6.0 * relevance
            + guide_bonus
            + base_bonus
            + heading_identity_bonus(chunk.text)
        )
        if best_structural > 0 or relevance >= 0.04:
            candidates.append((accessory, best_structural > 0, primary_guide, total, best_rank, chunk))

    # Do not discard globally relevant order maps merely because the first
    # lexical core happened to contain a neighbouring/composite family marker.
    # Large catalogues often repeat common units across MCB/RCCB/RCBO-like
    # sections; keeping the best query-relevant primary maps preserves recall,
    # while structural linkage still contributes to the score and ordering.

    candidates.sort(
        key=lambda item: (
            item[0],
            not item[2],
            -item[3],
            not item[1],
            item[4],
            item[5].page_number,
            item[5].chunk_id,
        )
    )

    selected: list[RankedChunk] = []
    seen_pages: set[int] = set()
    for _accessory, _linked, _primary, structural_score, _rank, chunk in candidates:
        if chunk.page_number in seen_pages:
            continue
        base = ranked_by_id.get(chunk.chunk_id)
        if base is None:
            base = RankedChunk(
                chunk=chunk,
                score=0.0,
                word_score=0.0,
                char_score=0.0,
                exact_score=0.0,
                code_score=0.0,
            )
        selected.append(
            RankedChunk(
                chunk=base.chunk,
                score=max(base.score, min(1.0, 0.10 + 0.04 * structural_score)),
                word_score=base.word_score,
                char_score=base.char_score,
                exact_score=base.exact_score,
                code_score=base.code_score,
            )
        )
        seen_pages.add(chunk.page_number)
        if len(seen_pages) >= maximum_pages:
            break
    return selected


def _selection_expansions(
    ordering: list[RankedChunk],
    chunks: list[CatalogueChunk],
    ranked_by_id: dict[str, RankedChunk],
    queries: list[str],
    *,
    maximum_pages: int = 3,
) -> list[RankedChunk]:
    """Link order-code pages back to their technical selection tables.

    The link is structural rather than domain-specific: family/model markers,
    visible table vocabulary and page qualifiers such as ``Standard Type`` are
    used. This prevents a good ordering page from being paired with a sibling
    product table merely because both repeat common electrical values.
    """

    if not ordering or maximum_pages <= 0:
        return []

    def technical_table_like(text: str) -> bool:
        heading = text[:1400]
        explicit_table = bool(
            _SELECTION_HEADING.search(heading)
            or (
                re.search(r"\bmodel\b", heading, flags=re.IGNORECASE)
                and re.search(
                    r"\brated\s+(?:current|voltage|power|capacity)\b|"
                    r"\breference\s+standard\b|\btechnical\s+data\b",
                    heading,
                    flags=re.IGNORECASE,
                )
            )
        )
        if explicit_table:
            return True

        # Some PDF extractors lose row labels but keep a code-like family
        # heading and many unit-bearing values. Recognise that layout without
        # assuming a particular product family.
        code_heading = bool(re.match(
            r"^\s*[A-Z][A-Z0-9_-]{2,}\s*(?:\([^)]{1,80}\))?",
            heading,
        ))
        unit_values = len(re.findall(
            r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?\s*"
            r"(?:[kMmunµ]?)(?:A|V|W|Hz|Pa|bar|m|mm|cm|kg|N|Nm|kA)"
            r"(?![A-Za-z])",
            heading,
            flags=re.IGNORECASE,
        ))
        standards = bool(re.search(r"\b(?:IEC|EN|ISO|DIN)\b", heading))
        return code_heading and unit_values >= 3 and standards

    def qualifiers(text: str) -> set[str]:
        heading = _page_context_heading(text, maximum=260).casefold()
        parenthetical = re.findall(r"\(([^)]{1,80})\)", heading)
        source = " ".join(parenthetical) if parenthetical else heading
        ignored = {"type", "model", "series", "selection", "table", "ordering", "information"}
        return {
            word
            for word in re.findall(r"[a-z]{3,}", source)
            if word not in ignored
        }

    candidates = [
        chunk
        for chunk in chunks
        if not _ORDERING_HEADING.search(chunk.text)
        and not _ACCESSORY_HEADING.search(chunk.text)
        and technical_table_like(chunk.text)
    ]
    if not candidates:
        return []

    query_scores = {chunk.chunk_id: 0.0 for chunk in candidates}
    candidate_retriever = HybridRetriever(candidates)
    for query in queries:
        for item in candidate_retriever.search(
            query,
            top_k=len(candidates),
            weights=(0.45, 0.35, 0.20, 0.0),
        ):
            query_scores[item.chunk.chunk_id] = max(
                query_scores[item.chunk.chunk_id], item.score
            )

    selected: list[RankedChunk] = []
    seen_pages: set[int] = set()
    for order_item in ordering[:4]:
        order_markers = _ordering_family_codes(order_item.chunk.text)
        if not order_markers:
            order_markers = _structural_markers(order_item.chunk.text)
        order_qualifiers = qualifiers(order_item.chunk.text)
        ranked_candidates: list[tuple[float, float, int, CatalogueChunk]] = []
        for chunk in candidates:
            chunk_markers = _structural_markers(chunk.text)
            overlap = order_markers & chunk_markers
            if not overlap:
                continue
            family_score = 0.0
            for marker in overlap:
                descendants = sum(
                    1
                    for candidate_marker in chunk_markers
                    if candidate_marker != marker
                    and candidate_marker.startswith(marker)
                )
                family_score += (
                    1.5 if any(char.isdigit() for char in marker) else 1.0
                ) + min(2.0, 0.4 * descendants)
            chunk_qualifiers = qualifiers(chunk.text)
            qualifier_overlap = len(order_qualifiers & chunk_qualifiers)
            qualifier_penalty = (
                4.0
                if order_qualifiers and chunk_qualifiers and qualifier_overlap == 0
                else 0.0
            )
            relevance = query_scores.get(chunk.chunk_id, 0.0)
            total = 4.0 * family_score + 6.0 * qualifier_overlap + 5.0 * relevance - qualifier_penalty
            ranked_candidates.append((total, relevance, chunk.page_number, chunk))

        ranked_candidates.sort(
            key=lambda item: (-item[0], -item[1], item[2], item[3].chunk_id)
        )
        for total, _relevance, page, chunk in ranked_candidates:
            if page in seen_pages:
                continue
            base = ranked_by_id.get(chunk.chunk_id)
            if base is None:
                base = RankedChunk(
                    chunk=chunk,
                    score=0.0,
                    word_score=0.0,
                    char_score=0.0,
                    exact_score=0.0,
                    code_score=0.0,
                )
            selected.append(
                RankedChunk(
                    chunk=base.chunk,
                    score=max(base.score, min(1.0, 0.12 + 0.03 * total)),
                    word_score=base.word_score,
                    char_score=base.char_score,
                    exact_score=base.exact_score,
                    code_score=base.code_score,
                    dense_score=base.dense_score,
                    lexical_rank=base.lexical_rank,
                    dense_rank=base.dense_rank,
                    fused_score=base.fused_score,
                    rerank_score=base.rerank_score,
                )
            )
            seen_pages.add(page)
            break
        if len(selected) >= maximum_pages:
            break
    return selected


def _normalise_queries(queries: str | Iterable[str]) -> list[str]:
    raw_queries = [queries] if isinstance(queries, str) else list(queries)
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        if not isinstance(query, str) or not query.strip():
            continue
        value = query.strip()
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    if not cleaned:
        raise ValueError("au moins une requete de recherche non vide est requise")
    return cleaned



_SOURCE_ORG_STOPWORDS = {
    "electric", "electrical", "electronics", "industrial", "industries",
    "company", "corporation", "corp", "group", "groupe", "inc", "ltd",
    "limited", "sa", "sas", "gmbh", "ag", "co",
}


def _source_identifier_fragments(need: dict[str, Any]) -> list[str]:
    """Return source-only literals to remove from target catalogue queries."""

    fragments: list[str] = []
    for key in ("source_reference", "source_family", "source_manufacturer"):
        value = need.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = value.strip()
        fragments.append(cleaned)
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", cleaned)
        for token in tokens:
            folded = token.casefold().strip("._-")
            if len(folded) < 3 or folded in _SOURCE_ORG_STOPWORDS:
                continue
            if key == "source_manufacturer" or any(char.isdigit() for char in token):
                fragments.append(token)
    return fragments

def _remove_literal(text: str, literal: str | None) -> str:
    if not isinstance(literal, str) or not literal.strip():
        return text
    pattern = re.compile(re.escape(literal.strip()), flags=re.IGNORECASE)
    return pattern.sub(" ", text)


def _clean_query(query: str, forbidden_values: Iterable[str | None]) -> str:
    cleaned = query
    for value in forbidden_values:
        cleaned = _remove_literal(cleaned, value)
        if isinstance(value, str):
            # Les requetes LLM peuvent reprendre seulement un morceau du fabricant
            # (ex. nom court pour raison sociale complète). Retirer aussi les
            # jetons source afin de ne jamais chercher la marque d'origine dans le
            # catalogue cible.
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
                if len(token) >= 3:
                    cleaned = re.sub(
                        rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                        " ",
                        cleaned,
                        flags=re.IGNORECASE,
                    )
    cleaned = re.sub(r"[|;,]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—:,.\t\n")
    return cleaned


def _attributes_for_selection(need: dict[str, Any]) -> list[dict[str, Any]]:
    attributes = need.get("attributes")
    if not isinstance(attributes, list):
        return []
    valid = [item for item in attributes if isinstance(item, dict)]
    has_roles = any(item.get("role") in {"selector", "constraint", "context"} for item in valid)
    if not has_roles:
        return valid
    selected = [item for item in valid if item.get("role") in {"selector", "constraint"}]
    return selected or valid


def _attribute_fragments(need: dict[str, Any], *, include_names: bool) -> list[str]:
    fragments: list[str] = []
    attributes = _attributes_for_selection(need)
    for attribute in attributes[:12]:
        if not isinstance(attribute, dict):
            continue
        parts: list[str] = []
        if include_names:
            name = attribute.get("name")
            if isinstance(name, str) and name.strip():
                parts.append(name.strip())
        value = attribute.get("value")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        unit = attribute.get("unit")
        if isinstance(unit, str) and unit.strip():
            parts.append(unit.strip())
        if parts:
            fragments.append(" ".join(parts))
    return fragments


def _source_wording_query(need: dict[str, Any], *, max_words: int = 60) -> str:
    """Keep the wording of the request itself, not only its normalised reading.

    Need extraction restates the product type in international terms, so a
    request written in the catalogue's own language loses the very words the
    catalogue uses for its sections. Searching with the original phrasing keeps
    that vocabulary available whatever the extraction chose to emit.
    """

    excerpt = need.get("source_excerpt")
    if not isinstance(excerpt, str) or not excerpt.strip():
        return ""
    text = re.sub(r"[#*_>`\[\]()]+", " ", excerpt)
    return " ".join(re.sub(r"\s+", " ", text).split()[:max_words])


def build_retrieval_queries(need: dict[str, Any]) -> list[str]:
    """Build generic catalogue queries while removing source-only identifiers."""

    forbidden = _source_identifier_fragments(need)
    candidates: list[str] = []

    source_wording = _source_wording_query(need)
    if source_wording:
        candidates.append(source_wording)

    raw_queries = need.get("search_queries")
    if isinstance(raw_queries, list):
        usable_queries = [query.strip() for query in raw_queries if isinstance(query, str) and query.strip()]
        usable_queries.sort(key=lambda query: (len(query.split()), len(query)))
        candidates.extend(usable_queries[:3])
    elif isinstance(raw_queries, str):
        candidates.append(raw_queries)

    product_type = need.get("product_type")
    product_prefix = product_type.strip() if isinstance(product_type, str) else ""
    named_fragments = _attribute_fragments(need, include_names=True)
    value_fragments = _attribute_fragments(need, include_names=False)
    if named_fragments:
        candidates.append(" ".join([product_prefix, *named_fragments]).strip())
    if value_fragments:
        candidates.append(" ".join([product_prefix, *value_fragments]).strip())

    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = _clean_query(candidate, forbidden)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value[:2000])
        if len(cleaned) == 6:
            break
    if not cleaned:
        raise ValueError("le besoin extrait ne permet pas de construire une requete catalogue")
    return cleaned



@lru_cache(maxsize=8)
def _catalogue_index_cached(
    path_value: str,
    file_size: int,
    modified_ns: int,
    max_words: int,
    overlap_words: int,
    ocr_mode: str,
    ocr_language: str,
    ocr_max_pages: int,
    min_native_chars: int,
) -> CatalogueIndexBundle:
    """Build a layout-aware immutable catalogue index once per file version."""

    del file_size, modified_ns  # values invalidate the cache key
    pages = extract_pdf_pages(
        path_value,
        table_pages=set(),
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    catalogue_map = build_catalogue_map(path_value, pages=pages)
    navigation_map = build_navigation_map(catalogue_map, pages)
    lookup = section_lookup(catalogue_map.sections)
    chunks = tuple(
        chunk_pages(
            pages,
            max_words=max_words,
            overlap_words=overlap_words,
            section_by_page=lookup,
        )
    )
    inspection = inspect_pdf(path_value, min_native_chars=min_native_chars)
    return CatalogueIndexBundle(
        pages=tuple(pages),
        chunks=chunks,
        retriever=HybridRetriever(list(chunks)),
        catalogue_map=catalogue_map,
        navigation_map=navigation_map,
        inspection=inspection,
    )


def _catalogue_index(
    catalogue_pdf: Path | str,
    *,
    max_words: int,
    overlap_words: int,
    ocr_mode: str = "auto",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
) -> CatalogueIndexBundle:
    path = Path(catalogue_pdf)
    if not path.is_file():
        raise FileNotFoundError(f"PDF introuvable: {path}")
    stat = path.stat()
    return _catalogue_index_cached(
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        max_words,
        overlap_words,
        ocr_mode,
        ocr_language,
        ocr_max_pages,
        min_native_chars,
    )


def _resolve_hierarchy_flag(
    *,
    hierarchy_enabled: bool | None = None,
    hierarchical: bool | None = None,
    hierarchy: bool | None = None,
    default: bool = True,
) -> bool:
    """Resolve backward-compatible hierarchy option aliases deterministically."""

    for value in (hierarchy_enabled, hierarchical, hierarchy):
        if value is not None:
            return bool(value)
    return bool(default)

def _section_summary_chunks(catalogue_map: Any) -> list[CatalogueChunk]:
    return [
        CatalogueChunk(
            chunk_id=f"section-{section.section_id}",
            page_number=section.start_page,
            text=section.summary,
            section_id=section.section_id,
            section_title=section.title,
            kind="section",
            markers=section.markers,
        )
        for section in catalogue_map.sections
        if str(section.summary or "").strip()
    ]


def _expand_section_ids(catalogue_map: Any, selected: Iterable[str]) -> list[str]:
    """Expand routed sections without crossing into unrelated product families.

    Immediate neighbours protect tables split at a section boundary. Nearby
    sections carrying a structural marker from the routed section are also
    included: industrial catalogues commonly place the family introduction,
    selection table, standard/deluxe variants and ordering key in consecutive
    sections with the same family/series marker.
    """

    sections = list(catalogue_map.sections)
    ordered = [section.section_id for section in sections]
    by_id = {section.section_id: section for section in sections}
    selected_order = [section_id for section_id in selected if section_id in by_id]
    wanted = set(selected_order)

    # Every routed section keeps its immediate neighbours. Marker propagation is
    # deliberately narrower: only the first three, family-like seeds (one or
    # two markers) may pull farther sections. This avoids a generic heading such
    # as "Ordering Information" carrying dozens of accessory markers and
    # widening the hierarchy to most of the catalogue.
    for section_id in selected_order:
        index = ordered.index(section_id)
        if index > 0:
            wanted.add(ordered[index - 1])
        if index + 1 < len(ordered):
            wanted.add(ordered[index + 1])

    for section_id in selected_order[:3]:
        seed = by_id[section_id]
        seed_markers = set(seed.markers)
        if not seed_markers or len(seed_markers) > 2:
            continue
        for candidate in sections:
            if candidate.section_id == seed.section_id:
                continue
            page_distance = max(
                0,
                seed.start_page - candidate.end_page,
                candidate.start_page - seed.end_page,
            )
            if page_distance > 8:
                continue
            if seed_markers.intersection(candidate.markers):
                wanted.add(candidate.section_id)

    return [section_id for section_id in ordered if section_id in wanted]


def _aggregate_lexical_lane(
    retriever: HybridRetriever,
    queries: list[str],
    *,
    top_k: int,
    weights: tuple[float, float, float, float],
    allowed_chunk_ids: set[str] | None = None,
) -> list[RankedChunk]:
    aggregated: dict[str, RankedChunk] = {}
    support_counts: dict[str, int] = {}
    for query in queries:
        for rank, candidate in enumerate(
            retriever.search(
                query,
                top_k=top_k,
                weights=weights,
                allowed_chunk_ids=allowed_chunk_ids,
            ),
            start=1,
        ):
            chunk_id = candidate.chunk.chunk_id
            support_counts[chunk_id] = support_counts.get(chunk_id, 0) + 1
            rank_bonus = 0.05 / rank
            previous = aggregated.get(chunk_id)
            if previous is None:
                aggregated[chunk_id] = RankedChunk(
                    chunk=candidate.chunk,
                    score=candidate.score + rank_bonus,
                    word_score=candidate.word_score,
                    char_score=candidate.char_score,
                    exact_score=candidate.exact_score,
                    code_score=candidate.code_score,
                    lexical_rank=rank,
                )
            else:
                aggregated[chunk_id] = RankedChunk(
                    chunk=previous.chunk,
                    score=max(previous.score, candidate.score + rank_bonus),
                    word_score=max(previous.word_score, candidate.word_score),
                    char_score=max(previous.char_score, candidate.char_score),
                    exact_score=max(previous.exact_score, candidate.exact_score),
                    code_score=max(previous.code_score, candidate.code_score),
                    lexical_rank=min(previous.lexical_rank or rank, rank),
                )
    return sorted(
        aggregated.values(),
        key=lambda item: (
            -(item.score + 0.02 * (support_counts[item.chunk.chunk_id] - 1)),
            item.chunk.page_number,
            item.chunk.chunk_id,
        ),
    )[:top_k]


def _aggregate_dense_lane(
    dense_index: Any,
    embedding_client: Any,
    queries: list[str],
    *,
    top_k: int,
    allowed_chunk_ids: set[str] | None = None,
    query_vectors: dict[str, Any] | None = None,
) -> list[RankedChunk]:
    dense_by_id: dict[str, RankedChunk] = {}
    cache = query_vectors if query_vectors is not None else {}
    for query in queries:
        if query not in cache:
            cache[query] = embedding_client.embed_query(query)
        query_vector = cache[query]
        for hit in dense_index.search(
            query_vector,
            top_k=top_k,
            allowed_chunk_ids=allowed_chunk_ids,
        ):
            previous = dense_by_id.get(hit.chunk.chunk_id)
            if previous is None or hit.score > previous.dense_score:
                dense_by_id[hit.chunk.chunk_id] = RankedChunk(
                    chunk=hit.chunk,
                    score=hit.score,
                    word_score=0.0,
                    char_score=0.0,
                    exact_score=0.0,
                    dense_score=hit.score,
                    dense_rank=hit.rank,
                )
    return sorted(
        dense_by_id.values(),
        key=lambda item: (-item.dense_score, item.chunk.page_number, item.chunk.chunk_id),
    )[:top_k]


def _structural_queries(queries: list[str]) -> list[str]:
    suffix = (
        "ordering information how to order codification reference code "
        "selection table technical data variant model"
    )
    return [f"{query} {suffix}"[:2400] for query in queries[:3]]


def adaptive_retrieval_profile(
    page_count: int,
    chunk_count: int,
    section_count: int,
    *,
    top_k: int,
    candidate_k: int | None,
    profile: str = "auto",
) -> dict[str, int | str]:
    """Scale retrieval from the actual index density, not page count alone."""

    if min(page_count, chunk_count, section_count) < 0:
        raise ValueError("les tailles du catalogue ne peuvent pas etre negatives")
    if top_k <= 0 or (candidate_k is not None and candidate_k <= 0):
        raise ValueError("profondeur de recherche invalide")
    aliases = {"auto": "auto", "compact": "standard", "standard": "standard", "large": "large", "grand": "large", "huge": "huge", "massif": "huge"}
    key = aliases.get(str(profile).casefold())
    if key is None:
        raise ValueError("profil catalogue inconnu")

    base_candidate = candidate_k or max(24, top_k * 4)
    if key == "standard":
        steps = 0
    else:
        page_steps = max(0, (page_count - 1) // 100) if page_count else 0
        chunk_steps = max(0, (chunk_count - 1) // 500) if chunk_count else 0
        # A catalogue may expose many fine-grained headings without being dense.
        # Let chunk density drive scaling first; sections add depth more slowly.
        section_steps = max(0, (section_count - 1) // 24) if section_count else 0
        steps = max(page_steps, chunk_steps, section_steps)
        if key == "large":
            steps = max(steps, 2)
        elif key == "huge":
            steps = max(steps, 4)

    effective_top = min(max(top_k, 24), top_k + 2 * steps)
    effective_candidate = min(max(base_candidate, 192), base_candidate + 16 * steps)
    effective_candidate = max(effective_candidate, effective_top)
    section_base = min(max(section_count, 1), 4) if section_count else 0
    section_k = min(section_count, section_base + steps) if section_count else 0
    return {
        "profile": key,
        "page_count": int(page_count),
        "chunk_count": int(chunk_count),
        "section_count": int(section_count),
        "depth_steps": int(steps),
        "top_k": int(effective_top),
        "candidate_k": int(effective_candidate),
        "section_k": int(section_k),
    }


def select_top_sections(
    ranked: list[RankedChunk],
    *,
    maximum: int,
) -> list[str]:
    """Choose sections by cumulative support across several matching chunks."""

    if maximum <= 0:
        return []
    aggregates: dict[str, dict[str, float | int]] = {}
    for position, item in enumerate(ranked, start=1):
        section_id = item.chunk.section_id
        if not section_id:
            continue
        score = (
            item.rerank_score
            if item.rerank_score is not None
            else item.fused_score or item.score
        )
        entry = aggregates.setdefault(section_id, {"sum": 0.0, "count": 0, "best": 0.0, "first": position})
        entry["sum"] = float(entry["sum"]) + max(0.0, float(score))
        entry["count"] = int(entry["count"]) + 1
        entry["best"] = max(float(entry["best"]), float(score))
        entry["first"] = min(int(entry["first"]), position)
    ordered = sorted(
        aggregates,
        key=lambda section: (
            -float(aggregates[section]["sum"]),
            -int(aggregates[section]["count"]),
            -float(aggregates[section]["best"]),
            int(aggregates[section]["first"]),
            section,
        ),
    )
    return ordered[:maximum]

def adaptive_retrieval_depth(
    page_count: int,
    *,
    top_k: int,
    candidate_k: int | None,
) -> dict[str, int]:
    """Increase retrieval depth after each complete block of 100 PDF pages.

    The default 8/32 profile becomes 10/48 for pages 101-200, 12/64
    for pages 201-300, and so on. Conservative caps avoid excessive
    reranking and context growth on very large catalogues. Explicitly larger
    user values are never reduced.
    """

    if page_count < 0:
        raise ValueError("page_count ne peut pas etre negatif")
    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if candidate_k is not None and candidate_k <= 0:
        raise ValueError("candidate_k doit etre strictement positif")

    depth_steps = max(0, (page_count - 1) // 100) if page_count else 0
    base_candidate = candidate_k or max(top_k * 4, 24)
    top_cap = max(top_k, 20)
    candidate_cap = max(base_candidate, 128)
    effective_top = min(top_cap, top_k + 2 * depth_steps)
    effective_candidate = min(candidate_cap, base_candidate + 16 * depth_steps)
    effective_candidate = max(effective_candidate, effective_top)
    return {
        "page_count": int(page_count),
        "depth_steps": int(depth_steps),
        "top_k": int(effective_top),
        "candidate_k": int(effective_candidate),
    }



def inspect_catalogue(
    catalogue_pdf: Path | str,
    *,
    ocr_mode: str = "off",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
) -> dict[str, Any]:
    """Return a local, API-free inspection of PDF extraction and hierarchy."""

    inspection = inspect_pdf(catalogue_pdf, min_native_chars=min_native_chars)
    pages = extract_pdf_pages(
        catalogue_pdf,
        table_pages=set(),
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    catalogue_map = build_catalogue_map(catalogue_pdf, pages=pages)
    navigation_map = build_navigation_map(catalogue_map, pages)
    return {
        "catalogue": str(Path(catalogue_pdf).resolve()),
        "page_count": inspection.page_count,
        "native_text_pages": list(inspection.native_text_pages),
        "low_text_pages": list(inspection.low_text_pages),
        "image_pages": list(inspection.image_pages),
        "ocr_candidate_pages": list(inspection.ocr_candidate_pages),
        "ocr_used_pages": [page.page_number for page in pages if page.ocr_used],
        "ocr_mode": ocr_mode,
        "section_source": catalogue_map.source,
        "section_count": len(catalogue_map.sections),
        "navigation_source": navigation_map.source,
        "navigation_structure_confidence": round(navigation_map.structure_confidence, 6),
        "navigation_structure_reasons": list(navigation_map.reasons),
        "navigation_node_count": len(navigation_map.nodes),
        "navigation_nodes": [
            {
                "node_id": node.node_id,
                "title": node.title,
                "start_page": node.start_page,
                "end_page": node.end_page,
                "level": node.level,
                "parent_id": node.parent_id,
                "source": node.source,
            }
            for node in navigation_map.nodes
        ],
        "sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
                "start_page": section.start_page,
                "end_page": section.end_page,
                "kind": section.kind,
                "markers": list(section.markers),
            }
            for section in catalogue_map.sections
        ],
    }


def index_catalogue(
    catalogue_pdf: Path | str,
    *,
    embedding_client: Any | None = None,
    cache_dir: Path | str = Path(".rag_cache"),
    max_words: int = 260,
    overlap_words: int = 50,
    hierarchy_enabled: bool | None = None,
    hierarchical: bool | None = None,
    hierarchy: bool | None = None,
    ocr_mode: str = "auto",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
) -> dict[str, Any]:
    """Prebuild the section-aware lexical index and optional dense cache."""

    hierarchy_on = _resolve_hierarchy_flag(
        hierarchy_enabled=hierarchy_enabled,
        hierarchical=hierarchical,
    )
    bundle = _catalogue_index(
        catalogue_pdf,
        max_words=max_words,
        overlap_words=overlap_words,
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    pages = list(bundle.pages)
    chunks = list(bundle.chunks)
    catalogue_map = bundle.catalogue_map
    navigation_map = bundle.navigation_map
    dense_cache_hit: bool | None = None
    if embedding_client is not None:
        _dense, dense_cache_hit = load_or_create_dense_index(
            catalogue_pdf,
            chunks,
            embedding_client,
            cache_dir=cache_dir,
            max_words=max_words,
            overlap_words=overlap_words,
        )
    return {
        "catalogue": str(Path(catalogue_pdf).resolve()),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "section_count": len(catalogue_map.sections),
        "section_source": catalogue_map.source,
        "navigation_source": navigation_map.source,
        "navigation_structure_confidence": round(navigation_map.structure_confidence, 6),
        "navigation_node_count": len(navigation_map.nodes),
        "hierarchy_enabled": hierarchy_on,
        "hierarchical_enabled": hierarchy_on,
        "ocr_mode": ocr_mode,
        "ocr_candidate_pages": list(bundle.inspection.ocr_candidate_pages),
        "ocr_used_pages": [page.page_number for page in pages if page.ocr_used],
        "embedding_model": str(getattr(embedding_client, "model", "")),
        "dense_index_created": embedding_client is not None,
        "dense_cache_hit": dense_cache_hit,
        "cache_dir": str(Path(cache_dir).resolve()),
    }


def prepare_catalogue_index(
    catalogue_pdf: Path | str,
    *,
    embedding_client: Any | None = None,
    cache_dir: Path | str = Path(".rag_cache"),
    max_words: int = 260,
    overlap_words: int = 50,
    hierarchy_enabled: bool | None = None,
    hierarchical: bool | None = None,
    hierarchy: bool | None = None,
    ocr_mode: str = "auto",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
) -> dict[str, Any]:
    """Prepare the reusable V13 index and return an auditable summary.

    ``hierarchical`` is accepted for API symmetry. The catalogue map is always
    built because it is inexpensive and can be ignored later by retrieval.
    """

    hierarchy_on = _resolve_hierarchy_flag(
        hierarchy_enabled=hierarchy_enabled,
        hierarchical=hierarchical,
        hierarchy=hierarchy,
    )
    hierarchy_enabled = hierarchy_on
    report = index_catalogue(
        catalogue_pdf,
        embedding_client=embedding_client,
        hierarchy_enabled=hierarchy_on,
        cache_dir=cache_dir,
        max_words=max_words,
        overlap_words=overlap_words,
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    report["hierarchy_enabled"] = hierarchy_on
    report["hierarchical_enabled"] = hierarchy_on
    return report


def retrieve_catalogue_chunks(
    catalogue_pdf: Path | str,
    queries: str | Iterable[str],
    *,
    top_k: int = 8,
    candidate_k: int | None = None,
    max_words: int = 260,
    overlap_words: int = 50,
    embedding_client: Any | None = None,
    rerank_client: Any | None = None,
    cache_dir: Path | str = Path(".rag_cache"),
    retrieval_metadata: dict[str, Any] | None = None,
    hierarchy_enabled: bool | None = None,
    hierarchical: bool | None = None,
    hierarchy: bool | None = None,
    catalogue_profile: str = "auto",
    section_k: int | None = None,
    ocr_mode: str = "auto",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
    rerank_batch_size: int = 48,
) -> list[RankedChunk]:
    """Navigate a structured catalogue, then retrieve locally or globally.

    V14 first selects a broad section and the most specific sub-section. It
    searches that route deeply, keeps one or two secondary branches, and
    automatically falls back to the normal global RAG when the structure, route
    or local evidence is insufficient. Exact codes and structural pages remain
    global safety lanes so an appendix reference cannot disappear.
    """

    if top_k <= 0:
        raise ValueError("top_k doit etre strictement positif")
    if candidate_k is not None and candidate_k <= 0:
        raise ValueError("candidate_k doit etre strictement positif")
    if section_k is not None and section_k <= 0:
        raise ValueError("section_k doit etre strictement positif")
    if rerank_batch_size <= 1:
        raise ValueError("rerank_batch_size doit etre superieur a 1")

    hierarchy_on = _resolve_hierarchy_flag(
        hierarchy_enabled=hierarchy_enabled,
        hierarchical=hierarchical,
    )

    query_list = _normalise_queries(queries)
    bundle = _catalogue_index(
        catalogue_pdf,
        max_words=max_words,
        overlap_words=overlap_words,
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    pages = list(bundle.pages)
    chunks = list(bundle.chunks)
    catalogue_map = bundle.catalogue_map
    retriever = bundle.retriever
    requested_top_k = top_k
    requested_candidate_k = candidate_k
    profile = adaptive_retrieval_profile(
        len(pages),
        len(chunks),
        len(catalogue_map.sections),
        top_k=top_k,
        candidate_k=candidate_k,
        profile=catalogue_profile,
    )
    top_k = int(profile["top_k"])
    candidate_k = int(profile["candidate_k"])
    effective_section_k = (
        min(len(catalogue_map.sections), section_k)
        if section_k is not None
        else int(profile["section_k"])
    )
    pool_size = min(len(chunks), max(top_k, candidate_k))

    inspection = bundle.inspection
    ocr_used_pages = [page.page_number for page in pages if page.ocr_used]
    if retrieval_metadata is not None:
        retrieval_metadata.update({
            "catalogue_page_count": len(pages),
            "catalogue_chunk_count": len(chunks),
            "catalogue_section_count": len(catalogue_map.sections),
            "catalogue_map_source": catalogue_map.source,
            "navigation_map_source": bundle.navigation_map.source,
            "navigation_node_count": len(bundle.navigation_map.nodes),
            "navigation_structure_confidence": round(bundle.navigation_map.structure_confidence, 6),
            "catalogue_profile": profile["profile"],
            "adaptive_depth_steps": profile["depth_steps"],
            "requested_top_k": requested_top_k,
            "effective_top_k": top_k,
            "requested_candidate_k": requested_candidate_k,
            "effective_candidate_k": candidate_k,
            "effective_section_k": effective_section_k,
            "hierarchy_enabled": hierarchy_on,
            "hierarchical_enabled": hierarchy_on,
            "ocr_mode": ocr_mode,
            "ocr_language": ocr_language,
            "ocr_candidate_pages": list(inspection.ocr_candidate_pages),
            "ocr_used_pages": ocr_used_pages,
            "low_text_pages": list(inspection.low_text_pages),
            "candidate_pool": pool_size,
        })

    navigation_map = bundle.navigation_map

    # Exact code/value retrieval is a permanent global safety lane. It is cheap
    # and protects references placed in appendices or outside the routed chapter.
    exact_lane = _aggregate_lexical_lane(
        retriever,
        query_list,
        top_k=pool_size,
        weights=(0.08, 0.07, 0.35, 0.50),
    )

    dense_index = None
    query_vector_cache: dict[str, Any] = {}
    dense_cache_hit: bool | None = None
    if embedding_client is not None:
        dense_index, dense_cache_hit = load_or_create_dense_index(
            catalogue_pdf,
            chunks,
            embedding_client,
            cache_dir=cache_dir,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    # V14 first navigates the catalogue map like a human: broad section,
    # specific sub-section, then a deep local search. If the map or local
    # evidence is weak, it automatically falls back to the normal global RAG.
    navigation_ranked: list[RankedChunk] = []
    route_plan = plan_navigation(navigation_map, [], maximum_routes=1)
    if hierarchy_on and effective_section_k > 0 and navigation_map.nodes:
        nav_chunks = navigation_chunks(navigation_map)
        if nav_chunks:
            nav_retriever = HybridRetriever(nav_chunks)
            route_lexical = _aggregate_lexical_lane(
                nav_retriever,
                query_list,
                top_k=min(len(nav_chunks), max(12, effective_section_k * 4)),
                weights=(0.55, 0.25, 0.20, 0.0),
            )
            navigation_ranked = weighted_rank_fusion(
                [("navigation_lexical", route_lexical, 1.0)],
                top_k=min(len(nav_chunks), max(12, effective_section_k * 4)),
            )
            route_plan = plan_navigation(
                navigation_map,
                navigation_ranked,
                maximum_routes=min(3, max(1, effective_section_k)),
            )

    primary_pages, secondary_pages = expand_route_pages(
        route_plan,
        page_count=len(pages),
        neighbour_radius=1,
    )
    primary_ids = {
        chunk.chunk_id for chunk in chunks if chunk.page_number in primary_pages
    }
    secondary_ids = {
        chunk.chunk_id for chunk in chunks if chunk.page_number in secondary_pages
    }

    primary_lexical: list[RankedChunk] = []
    primary_dense: list[RankedChunk] = []
    secondary_lexical: list[RankedChunk] = []
    secondary_dense: list[RankedChunk] = []
    if route_plan.strategy == "hierarchical" and primary_ids:
        primary_lexical = _aggregate_lexical_lane(
            retriever,
            query_list,
            top_k=pool_size,
            weights=(0.48, 0.27, 0.20, 0.05),
            allowed_chunk_ids=primary_ids,
        )
        if dense_index is not None and embedding_client is not None:
            primary_dense = _aggregate_dense_lane(
                dense_index,
                embedding_client,
                query_list,
                top_k=pool_size,
                allowed_chunk_ids=primary_ids,
                query_vectors=query_vector_cache,
            )
        if secondary_ids:
            secondary_lexical = _aggregate_lexical_lane(
                retriever,
                query_list,
                top_k=pool_size,
                weights=(0.48, 0.27, 0.20, 0.05),
                allowed_chunk_ids=secondary_ids,
            )
            if dense_index is not None and embedding_client is not None:
                secondary_dense = _aggregate_dense_lane(
                    dense_index,
                    embedding_client,
                    query_list,
                    top_k=pool_size,
                    allowed_chunk_ids=secondary_ids,
                    query_vectors=query_vector_cache,
                )

    local_lanes = [
        ("primary_lexical", primary_lexical, 1.60),
        ("primary_dense", primary_dense, 1.50),
        ("secondary_lexical", secondary_lexical, 1.15),
        ("secondary_dense", secondary_dense, 1.05),
    ]
    local_ranked = weighted_rank_fusion(local_lanes, top_k=pool_size)
    local_quality = assess_local_evidence(local_ranked[: min(24, len(local_ranked))], query_list)

    global_fallback_used = route_plan.global_fallback_used
    global_fallback_reason = route_plan.fallback_reason
    if route_plan.strategy == "hierarchical" and not local_quality.sufficient:
        global_fallback_used = True
        global_fallback_reason = "local_evidence_insufficient"

    global_lexical: list[RankedChunk] = []
    global_dense: list[RankedChunk] = []
    if not hierarchy_on or route_plan.strategy == "global" or global_fallback_used:
        global_lexical = _aggregate_lexical_lane(
            retriever,
            query_list,
            top_k=pool_size,
            weights=(0.45, 0.35, 0.20, 0.0),
        )
        if dense_index is not None and embedding_client is not None:
            global_dense = _aggregate_dense_lane(
                dense_index,
                embedding_client,
                query_list,
                top_k=pool_size,
                query_vectors=query_vector_cache,
            )

    structural_ids = {
        chunk.chunk_id
        for chunk in chunks
        if chunk.kind in {"ordering", "selection", "table"}
    }
    routed_ids = primary_ids | secondary_ids
    local_structural_ids = structural_ids & routed_ids
    local_structural_lane = (
        _aggregate_lexical_lane(
            retriever,
            _structural_queries(query_list),
            top_k=pool_size,
            weights=(0.50, 0.30, 0.15, 0.05),
            allowed_chunk_ids=local_structural_ids,
        )
        if local_structural_ids
        else []
    )
    # A low-weight global structural lane can recover a separated order-code
    # page without turning the whole request back into a global semantic search.
    global_structural_lane = (
        _aggregate_lexical_lane(
            retriever,
            _structural_queries(query_list),
            top_k=pool_size,
            weights=(0.50, 0.30, 0.15, 0.05),
            allowed_chunk_ids=structural_ids,
        )
        if structural_ids
        else []
    )

    lanes = [
        *local_lanes,
        ("local_structural", local_structural_lane, 1.30),
        ("global_exact", exact_lane, 1.25),
        ("global_structural_safety", global_structural_lane, 0.45),
        ("global_lexical", global_lexical, 1.00),
        ("global_dense", global_dense, 0.95),
    ]
    ranked = weighted_rank_fusion(lanes, top_k=pool_size)
    if not ranked:
        ranked = local_ranked or global_lexical or exact_lane or global_dense

    selected_page_set = primary_pages | secondary_pages
    selected_section_ids = list(dict.fromkeys(
        chunk.section_id
        for chunk in chunks
        if chunk.page_number in selected_page_set and chunk.section_id
    ))
    routed_section_ids = list(dict.fromkeys(
        section_id
        for node_id in (
            ([route_plan.primary_node_id] if route_plan.primary_node_id else [])
            + list(route_plan.secondary_node_ids)
        )
        for node in navigation_map.nodes
        if node.node_id == node_id
        for section_id in node.section_ids
    ))

    if retrieval_metadata is not None:
        by_id = {section.section_id: section for section in catalogue_map.sections}
        primary_node = navigation_map.node(route_plan.primary_node_id or "")
        retrieval_metadata.update({
            "dense_cache_hit": dense_cache_hit,
            "section_dense_cache_hit": None,
            "navigation_source": navigation_map.source,
            "navigation_structure_confidence": round(navigation_map.structure_confidence, 6),
            "navigation_structure_reasons": list(navigation_map.reasons),
            "navigation_strategy": (
                "hierarchical_with_global_fallback"
                if hierarchy_on and route_plan.strategy == "hierarchical" and global_fallback_used
                else (route_plan.strategy if hierarchy_on else "global")
            ),
            "navigation_route_confidence": round(route_plan.route_confidence, 6),
            "catalogue_route": list(route_plan.route_path),
            "primary_navigation_node_id": route_plan.primary_node_id,
            "secondary_navigation_node_ids": list(route_plan.secondary_node_ids),
            "primary_page_range": (
                [primary_node.start_page, primary_node.end_page]
                if primary_node is not None else []
            ),
            "primary_search_pages": sorted(primary_pages),
            "secondary_search_pages": sorted(secondary_pages),
            "global_fallback_used": bool(global_fallback_used or not hierarchy_on),
            "global_fallback_reason": (
                global_fallback_reason
                if hierarchy_on else "hierarchy_disabled"
            ),
            "local_evidence": {
                "coverage": round(local_quality.coverage, 6),
                "score": round(local_quality.score, 6),
                "sufficient": local_quality.sufficient,
                "has_reference_signal": local_quality.has_reference_signal,
                "has_structural_signal": local_quality.has_structural_signal,
                "matched_tokens": list(local_quality.matched_tokens),
                "missing_tokens": list(local_quality.missing_tokens),
            },
            "routed_section_ids": routed_section_ids,
            "selected_section_ids": selected_section_ids,
            "selected_sections": [
                {
                    "section_id": section_id,
                    "title": by_id[section_id].title,
                    "start_page": by_id[section_id].start_page,
                    "end_page": by_id[section_id].end_page,
                    "kind": by_id[section_id].kind,
                }
                for section_id in selected_section_ids
                if section_id in by_id
            ],
            "retrieval_lanes": {
                name: len(items) for name, items, _weight in lanes
            },
            "lexical_candidates": len(global_lexical),
            "dense_candidates": len(global_dense),
        })

    core = ranked[: min(top_k, len(ranked))]
    if not core:
        return []

    def _finalise_retrieval(items: list[RankedChunk]) -> list[RankedChunk]:
        context_target = min(len(items), max(top_k, min(24, top_k + 12)))
        rerank_pool = min(
            len(items),
            max(32, min(192, max(candidate_k, top_k * 4, context_target * 2))),
        )
        candidate_output, removed = deduplicate_ranked_chunks(items[:rerank_pool])
        if retrieval_metadata is not None:
            existing = list(retrieval_metadata.get("deduplicated_chunk_ids", []))
            retrieval_metadata["deduplicated_chunk_ids"] = list(
                dict.fromkeys([*existing, *removed])
            )
            retrieval_metadata["deduplicated_chunks"] = len(
                retrieval_metadata["deduplicated_chunk_ids"]
            )
            retrieval_metadata["pre_rerank_candidates"] = len(candidate_output)
        # Preserve structural coverage before the cross-encoder while keeping a
        # wider pool than the final LLM context.
        candidate_output = diversify_ranked_chunks(
            candidate_output,
            top_k=min(len(candidate_output), max(context_target * 3, top_k)),
            max_per_page=4,
            max_per_section=10,
        )
        if rerank_client is not None and candidate_output:
            rerank_query = "\n".join(query_list[:3])
            candidate_output = rerank_ranked_chunks_batched(
                rerank_query,
                candidate_output,
                rerank_client,
                top_k=min(max(context_target * 2, top_k), len(candidate_output)),
                batch_size=rerank_batch_size,
                metadata=retrieval_metadata,
            )
            if retrieval_metadata is not None:
                retrieval_metadata["reranked_candidates"] = len(candidate_output)
        candidate_output = diversify_ranked_chunks(
            candidate_output,
            top_k=min(context_target, len(candidate_output)),
            max_per_page=2,
            max_per_section=4,
        )
        if retrieval_metadata is not None:
            retrieval_metadata["final_context_chunks"] = len(candidate_output)
            retrieval_metadata["final_context_pages"] = sorted(
                {item.chunk.page_number for item in candidate_output}
            )
            retrieval_metadata["final_context_sections"] = list(dict.fromkeys(
                item.chunk.section_id for item in candidate_output if item.chunk.section_id
            ))
        return candidate_output

    ranked_by_id = {item.chunk.chunk_id: item for item in ranked}
    structural = _ordering_expansions(
        core,
        chunks,
        ranked_by_id,
        query_list,
    )
    selection_structural = _selection_expansions(
        structural,
        chunks,
        ranked_by_id,
        query_list,
    )
    structural_pages = {item.chunk.page_number for item in structural}

    # Industrial catalogues often continue a selection table or ordering key on
    # the next page. Add a small, score-ranked set of adjacent pages without
    # assuming any product family or manufacturer vocabulary.
    chunks_by_page: dict[int, list[CatalogueChunk]] = {}
    for chunk in chunks:
        chunks_by_page.setdefault(chunk.page_number, []).append(chunk)
    core_pages = {item.chunk.page_number for item in core}
    adjacent_pages = {
        page
        for core_page in core_pages
        for page in (core_page - 1, core_page + 1)
        if page > 0 and page in chunks_by_page and page not in core_pages
    }
    def adjacent_page_score(page: int) -> float:
        return max(
            (
                ranked_by_id.get(chunk.chunk_id).score
                for chunk in chunks_by_page[page]
                if ranked_by_id.get(chunk.chunk_id) is not None
            ),
            default=0.0,
        )

    selected_adjacent_pages: list[int] = []
    selected_set: set[int] = set()
    max_adjacent_pages = min(max(2, top_k), 8)

    # Product tables can start on the previous page and continue on the next.
    # Preserve both neighbours for the strongest catalogue-table pages before
    # generic ranking; many PDFs expose only generic headers such as "catalog
    # number / rated current" and never use the word "selection".
    def _catalogue_table_page(text: str) -> bool:
        heading = text[:1800]
        if _SELECTION_HEADING.search(heading):
            return True
        return bool(
            re.search(
                r"(?:num[eé]ro\s+de\s+catalogue|catalog(?:ue)?\s+number)"
                r".*(?:num[eé]ro\s+de\s+r[eé]f[eé]rence|reference|rated\s+current|courant\s+nominal)",
                heading,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )

    for core_item in core:
        if not _catalogue_table_page(core_item.chunk.text):
            continue
        core_page = core_item.chunk.page_number
        for neighbour in (core_page - 1, core_page + 1):
            if (
                neighbour > 0
                and neighbour in chunks_by_page
                and neighbour not in core_pages
                and neighbour not in selected_set
            ):
                selected_adjacent_pages.append(neighbour)
                selected_set.add(neighbour)
                if len(selected_adjacent_pages) >= max_adjacent_pages:
                    break
        if len(selected_adjacent_pages) >= max_adjacent_pages:
            break

    for core_item in core:
        core_page = core_item.chunk.page_number
        candidates = [
            page
            for page in (core_page - 1, core_page + 1)
            if page in adjacent_pages and page not in selected_set
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda page: (
                -adjacent_page_score(page),
                0 if page == core_page + 1 else 1,
                page,
            )
        )
        chosen = candidates[0]
        selected_adjacent_pages.append(chosen)
        selected_set.add(chosen)
        if len(selected_adjacent_pages) >= max_adjacent_pages:
            break

    if len(selected_adjacent_pages) < max_adjacent_pages:
        remaining = sorted(
            adjacent_pages - selected_set,
            key=lambda page: (-adjacent_page_score(page), page),
        )
        for page in remaining:
            selected_adjacent_pages.append(page)
            selected_set.add(page)
            if len(selected_adjacent_pages) >= max_adjacent_pages:
                break

    # Put the best technical passage first, then linked order-code schemas and
    # the remaining core results. Continuation pages come after the core so a
    # high-ranked ordering page can never be truncated away.
    output = [core[0]]
    seen_ids = {core[0].chunk.chunk_id}
    for item in structural:
        if item.chunk.chunk_id in seen_ids:
            continue
        output.append(item)
        seen_ids.add(item.chunk.chunk_id)
    for item in selection_structural:
        if item.chunk.chunk_id in seen_ids:
            continue
        output.append(item)
        seen_ids.add(item.chunk.chunk_id)

    # A catalogue table is often split into several text chunks on the same
    # physical page: the family heading can be retrieved while the orderable
    # row containing the exact current/reference remains in a sibling chunk.
    # Expand the strongest selected pages before reranking so the LLM sees both
    # the table identity and its rows. This is catalogue-generic and bounded.
    priority_pages: list[int] = []
    for selected_item in [*core[:4], *structural[:4], *selection_structural[:4]]:
        page = selected_item.chunk.page_number
        if page not in priority_pages:
            priority_pages.append(page)
    sibling_budget = min(24, max(6, top_k * 2))
    sibling_count = 0
    for page in priority_pages:
        page_items = sorted(
            chunks_by_page.get(page, []),
            key=lambda chunk: (
                -(ranked_by_id.get(chunk.chunk_id).score if ranked_by_id.get(chunk.chunk_id) else 0.0),
                chunk.chunk_id,
            ),
        )
        page_added = 0
        for chunk in page_items:
            if chunk.chunk_id in seen_ids:
                continue
            ranked_item = ranked_by_id.get(chunk.chunk_id)
            if ranked_item is None:
                ranked_item = RankedChunk(
                    chunk=chunk,
                    score=0.0,
                    word_score=0.0,
                    char_score=0.0,
                    exact_score=0.0,
                    code_score=0.0,
                )
            output.append(ranked_item)
            seen_ids.add(chunk.chunk_id)
            sibling_count += 1
            page_added += 1
            if page_added >= 4 or sibling_count >= sibling_budget:
                break
        if sibling_count >= sibling_budget:
            break

    for item in core[1:]:
        if item.chunk.chunk_id in seen_ids:
            continue
        output.append(item)
        seen_ids.add(item.chunk.chunk_id)
    for page in selected_adjacent_pages:
        for chunk in chunks_by_page[page]:
            if chunk.chunk_id in seen_ids:
                continue
            ranked_item = ranked_by_id.get(chunk.chunk_id)
            if ranked_item is None:
                ranked_item = RankedChunk(
                    chunk=chunk,
                    score=0.0,
                    word_score=0.0,
                    char_score=0.0,
                    exact_score=0.0,
                    code_score=0.0,
                )
            output.append(ranked_item)
            seen_ids.add(chunk.chunk_id)

    maximum_output = max(20, top_k + 12)
    output = output[:maximum_output]

    # Re-open only the selected pages for layout-aware table extraction. This
    # avoids flattening columns while keeping long-catalogue performance sane.
    selected_pages = {item.chunk.page_number for item in output}
    table_target_pages = set(selected_pages)
    if retrieval_metadata is not None:
        retrieval_metadata["pre_table_context_pages"] = sorted(selected_pages)
        retrieval_metadata["pre_table_context_order"] = [item.chunk.page_number for item in output]
    for core_item in core:
        if _SELECTION_HEADING.search(core_item.chunk.text):
            table_target_pages.add(core_item.chunk.page_number + 1)
    table_scan_pages = set(table_target_pages)
    table_scan_pages.update(page - 1 for page in table_target_pages if page > 1)
    enriched_pages = extract_pdf_pages(
        catalogue_pdf,
        table_pages=table_scan_pages,
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    page_plain_text = {page.page_number: page.text for page in pages}
    table_chunks: list[CatalogueChunk] = []
    for page in enriched_pages:
        if page.page_number not in table_target_pages or not page.structured_blocks:
            continue
        raw_page_text = page_plain_text.get(page.page_number, "")
        page_context = _page_context_heading(raw_page_text)
        is_ordering_page = bool(_ORDERING_HEADING.search(raw_page_text))
        if is_ordering_page:
            template = _order_template_block(raw_page_text)
            template_text = f"{template}\n" if template else ""
            slot_code_map = (
                _structured_slot_code_map(template, page.structured_blocks)
                if template
                else ""
            )
            slot_code_map_text = f"{slot_code_map}\n" if slot_code_map else ""
            base_page_chunk = next(iter(chunks_by_page.get(page.page_number, [])), None)
            table_chunks.append(
                CatalogueChunk(
                    chunk_id=f"p{page.page_number}-table-order",
                    page_number=page.page_number,
                    text=(
                        f"PAGE CONTEXT: {page_context}\n"
                        f"{template_text}"
                        f"{slot_code_map_text}"
                        "STRUCTURED TABLE CODE MAP\n"
                        + "\n\n".join(page.structured_blocks)
                    ),
                    section_id=base_page_chunk.section_id if base_page_chunk else "",
                    section_title=base_page_chunk.section_title if base_page_chunk else "",
                    kind="ordering",
                    markers=base_page_chunk.markers if base_page_chunk else (),
                )
            )
            continue
        for block_index, block in enumerate(page.structured_blocks, start=1):
            base_page_chunk = next(iter(chunks_by_page.get(page.page_number, [])), None)
            table_chunks.append(
                CatalogueChunk(
                    chunk_id=f"p{page.page_number}-table-{block_index}",
                    page_number=page.page_number,
                    text=(
                        f"PAGE CONTEXT: {page_context}\n"
                        f"STRUCTURED TABLE VARIANT COLUMN\n{block}"
                    ),
                    section_id=base_page_chunk.section_id if base_page_chunk else "",
                    section_title=base_page_chunk.section_title if base_page_chunk else "",
                    kind="variant",
                    markers=base_page_chunk.markers if base_page_chunk else (),
                )
            )

    if structural_pages:
        table_chunks = [
            chunk
            for chunk in table_chunks
            if not chunk.chunk_id.endswith("-table-order")
            or chunk.page_number in structural_pages
        ]
    if retrieval_metadata is not None:
        retrieval_metadata["table_target_pages"] = sorted(table_target_pages)
        retrieval_metadata["structured_table_pages"] = sorted({chunk.page_number for chunk in table_chunks})
    if not table_chunks:
        return _finalise_retrieval(output)

    def is_identity_chunk(chunk: CatalogueChunk) -> bool:
        return bool(re.search(
            r"(?:^|\n)(?:model|reference|type|designation|article)\s*:",
            chunk.text,
            flags=re.IGNORECASE,
        ))

    table_retriever = HybridRetriever(table_chunks)
    table_scores: dict[str, RankedChunk] = {}
    table_pool = min(len(table_chunks), max(top_k * 3, 12))
    for query in query_list:
        for candidate in table_retriever.search(
            query,
            top_k=table_pool,
            weights=(0.45, 0.35, 0.20, 0.0),
        ):
            previous = table_scores.get(candidate.chunk.chunk_id)
            if previous is None or candidate.score > previous.score:
                table_scores[candidate.chunk.chunk_id] = candidate

    # Guarantee query-relevant structured evidence from the strongest pages
    # already selected by the plain-text retrieval. A global table ranking can
    # otherwise be flooded by dozens of similar blocks from one large page and
    # silently drop the exact row on a neighbouring product page.
    priority_page_order: list[int] = []
    for item in output:
        page = item.chunk.page_number
        if page not in priority_page_order:
            priority_page_order.append(page)
    priority_page_order = priority_page_order[: min(16, max(6, top_k))]
    table_chunks_by_page: dict[int, list[CatalogueChunk]] = {}
    for chunk in table_chunks:
        table_chunks_by_page.setdefault(chunk.page_number, []).append(chunk)
    for page in priority_page_order:
        page_chunks = table_chunks_by_page.get(page, [])
        if not page_chunks:
            continue
        page_retriever = HybridRetriever(page_chunks)
        for query in query_list:
            for candidate in page_retriever.search(
                query,
                top_k=min(2, len(page_chunks)),
                weights=(0.45, 0.35, 0.20, 0.0),
            ):
                previous = table_scores.get(candidate.chunk.chunk_id)
                if previous is None or candidate.score > previous.score:
                    table_scores[candidate.chunk.chunk_id] = candidate

    if retrieval_metadata is not None:
        retrieval_metadata["table_score_pages_before_identity"] = sorted({item.chunk.page_number for item in table_scores.values()})
        retrieval_metadata["priority_table_pages"] = priority_page_order

    # Ordering schemas and variant identity records are structural evidence. Keep
    # them even when their wording differs from the query language.
    for chunk in table_chunks:
        if not chunk.chunk_id.endswith("-table-order") and not is_identity_chunk(chunk):
            continue
        table_scores.setdefault(
            chunk.chunk_id,
            RankedChunk(
                chunk=chunk,
                score=0.0,
                word_score=0.0,
                char_score=0.0,
                exact_score=0.0,
                code_score=0.0,
            ),
        )

    structural_page_rank = {
        item.chunk.page_number: rank
        for rank, item in enumerate(structural)
    }
    ordering_items = [
        item
        for item in table_scores.values()
        if item.chunk.chunk_id.endswith("-table-order")
    ]
    ordering_items.sort(
        key=lambda item: (
            _ordering_scope_rank(item.chunk.text, query_list),
            structural_page_rank.get(item.chunk.page_number, 10_000),
            -item.score,
            item.chunk.page_number,
        )
    )
    primary_ordering = ordering_items[0] if ordering_items else None
    primary_markers = (
        _ordering_family_codes(primary_ordering.chunk.text)
        if primary_ordering is not None
        else set()
    )
    if not primary_markers and primary_ordering is not None:
        primary_markers = _structural_markers(primary_ordering.chunk.text)

    def family_overlap(item: RankedChunk) -> float:
        overlap = primary_markers & _structural_markers(item.chunk.text)
        return sum(
            1.5 if any(char.isdigit() for char in marker) else 1.0
            for marker in overlap
        )

    def context_qualifiers(item: RankedChunk) -> set[str]:
        first_line = item.chunk.text.splitlines()[0] if item.chunk.text else ""
        match = re.search(r"\(([^)]{1,60})\)", first_line)
        if match is None:
            return set()
        return {
            token
            for token in re.findall(r"[A-Za-z]{3,}", match.group(1).casefold())
            if token not in {"type", "model", "series"}
        }

    primary_qualifiers = (
        context_qualifiers(primary_ordering)
        if primary_ordering is not None
        else set()
    )

    def qualifier_overlap(item: RankedChunk) -> int:
        return len(primary_qualifiers & context_qualifiers(item))

    identity_items = [
        item
        for item in table_scores.values()
        if is_identity_chunk(item.chunk)
        and not item.chunk.chunk_id.endswith("-table-order")
    ]
    identity_items.sort(
        key=lambda item: (
            -family_overlap(item),
            -qualifier_overlap(item),
            -item.score,
            item.chunk.page_number,
            item.chunk.chunk_id,
        )
    )
    remaining_ordering = [
        item for item in ordering_items if item is not primary_ordering
    ]
    remaining_ordering.sort(
        key=lambda item: (
            -family_overlap(item),
            -qualifier_overlap(item),
            structural_page_rank.get(item.chunk.page_number, 10_000),
            -item.score,
            item.chunk.page_number,
        )
    )
    other_items = [
        item
        for item in table_scores.values()
        if item not in ordering_items and item not in identity_items
    ]
    other_items.sort(
        key=lambda item: (
            -family_overlap(item),
            -item.score,
            item.chunk.page_number,
            item.chunk.chunk_id,
        )
    )

    selected_tables: list[RankedChunk] = []
    selected_table_ids: set[str] = set()
    preferred_sequence: list[RankedChunk] = []
    if primary_ordering is not None:
        preferred_sequence.append(primary_ordering)
    # Put the candidate columns that share the primary order-map family and
    # qualifier (for example standard vs premium) before sibling schemas.
    identity_head = identity_items[:10]
    preferred_sequence.extend(identity_head)
    preferred_sequence.extend(remaining_ordering)
    preferred_sequence.extend(identity_items[len(identity_head):])
    preferred_sequence.extend(other_items)
    for item in preferred_sequence:
        if item.chunk.chunk_id in selected_table_ids:
            continue
        selected_tables.append(item)
        selected_table_ids.add(item.chunk.chunk_id)
        if len(selected_tables) >= max(16, top_k * 2):
            break

    if retrieval_metadata is not None:
        retrieval_metadata["selected_table_pages"] = [item.chunk.page_number for item in selected_tables]
        retrieval_metadata["selected_table_chunk_ids"] = [item.chunk.chunk_id for item in selected_tables]

    structured_pages = {
        chunk.page_number for chunk in table_chunks
    }

    # Keep one high-relevance plain-text chunk beside one structured block for
    # each priority table page. Table extraction may preserve identifiers while
    # omitting the current/voltage row (or vice versa); discarding all native
    # text from a structured page loses exactly the evidence needed to select
    # a nearby catalogue alternative.
    plain_by_page: dict[int, list[RankedChunk]] = {}
    nonstructured_plain: list[RankedChunk] = []
    for item in output:
        page = item.chunk.page_number
        if page in structured_pages:
            plain_by_page.setdefault(page, []).append(item)
        elif not (
            structural_pages
            and _ORDERING_HEADING.search(item.chunk.text)
            and page not in structural_pages
        ):
            nonstructured_plain.append(item)

    table_by_page: dict[int, list[RankedChunk]] = {}
    for item in selected_tables:
        table_by_page.setdefault(item.chunk.page_number, []).append(item)

    # Promote native support by technical relevance rather than by page order.
    # A late page containing an exact orderable row must survive even when many
    # earlier pages contain generic table headers or coincidental values.
    support_limit = min(12, max(4, top_k // 2))
    native_support = _select_native_table_support(
        selected_tables,
        plain_by_page,
        maximum=support_limit,
    )
    support_ids = {item.chunk.chunk_id for item in native_support}
    support_pages = {item.chunk.page_number for item in native_support}
    if retrieval_metadata is not None:
        retrieval_metadata["native_table_support_pages"] = [
            item.chunk.page_number for item in native_support
        ]
        retrieval_metadata["native_table_support_chunk_ids"] = [
            item.chunk.chunk_id for item in native_support
        ]

    structured_head = min(len(selected_tables), min(8, max(4, top_k // 2)))
    final_output: list[RankedChunk] = [
        *selected_tables[:structured_head],
        *native_support,
        *selected_tables[structured_head:],
    ]
    for page in priority_page_order:
        remaining = [
            item for item in plain_by_page.get(page, [])
            if item.chunk.chunk_id not in support_ids
        ]
        if remaining:
            final_output.append(remaining[0])
    final_output.extend(nonstructured_plain)
    seen_final: set[str] = set()
    unique_output: list[RankedChunk] = []
    for item in final_output:
        if item.chunk.chunk_id in seen_final:
            continue
        seen_final.add(item.chunk.chunk_id)
        unique_output.append(item)
    return _finalise_retrieval(unique_output)


def _select_native_table_support(
    selected_tables: list[RankedChunk],
    plain_by_page: dict[int, list[RankedChunk]],
    *,
    maximum: int,
) -> list[RankedChunk]:
    """Select native-text support only when structured blocks lose evidence.

    Native PDF text may rescue a row omitted by table extraction, but it may
    also flatten several table columns together.  A native chunk is therefore
    promoted only when it contributes a catalogue/reference token or multiple
    technical values that are absent from the structured chunks on that page.
    """

    if maximum <= 0 or not selected_tables or not plain_by_page:
        return []

    table_page_order: list[int] = []
    structured_by_page: dict[int, list[str]] = {}
    for item in selected_tables:
        page = item.chunk.page_number
        if page not in table_page_order:
            table_page_order.append(page)
        structured_by_page.setdefault(page, []).append(item.chunk.text)
    page_position = {page: index for index, page in enumerate(table_page_order)}

    reference_pattern = re.compile(
        r"\b(?=[A-Z0-9][A-Z0-9,./-]{4,}\b)"
        r"(?=[A-Z0-9,./-]*[A-Z])(?=[A-Z0-9,./-]*\d)"
        r"[A-Z0-9][A-Z0-9,./-]*\b"
    )
    technical_pattern = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:kA|mA|A|kV|VDC|VAC|V|Hz|mm|cm|kg|W)\b"
        r"|\b\d+(?:[.,]\d+)?\s*[x×]\s*\d+(?:[.,]\d+)?(?:\s*mm)?\b",
        flags=re.IGNORECASE,
    )

    def evidence_tokens(text: str) -> tuple[set[str], set[str]]:
        upper = text.upper()
        references = {match.group(0).casefold() for match in reference_pattern.finditer(upper)}
        technical = {
            re.sub(r"\s+", "", match.group(0)).replace("×", "x").casefold()
            for match in technical_pattern.finditer(text)
        }
        return references, technical

    supports: list[tuple[float, int, RankedChunk]] = []
    for page in table_page_order:
        page_items = plain_by_page.get(page, [])
        if not page_items:
            continue
        structured_text = "\n".join(structured_by_page.get(page, []))
        structured_refs, structured_technical = evidence_tokens(structured_text)

        def support_score(item: RankedChunk) -> tuple[float, float, float, float, str]:
            plain_refs, plain_technical = evidence_tokens(item.chunk.text)
            added_refs = plain_refs - structured_refs
            added_technical = plain_technical - structured_technical
            evidence_bonus = 0.45 * min(4, len(added_refs)) + 0.20 * min(5, len(added_technical))
            score = (
                float(item.score)
                + 1.35 * float(item.exact_score)
                + 0.55 * float(item.code_score)
                + 0.20 * float(item.word_score)
                + 0.10 * float(item.char_score)
                + evidence_bonus
            )
            return (
                score,
                float(item.exact_score),
                float(item.code_score),
                float(item.score),
                item.chunk.chunk_id,
            )

        eligible: list[RankedChunk] = []
        for item in page_items:
            plain_refs, plain_technical = evidence_tokens(item.chunk.text)
            added_refs = plain_refs - structured_refs
            added_technical = plain_technical - structured_technical
            if added_refs or (
                len(added_technical) >= 2 and float(item.exact_score) >= 0.25
            ):
                eligible.append(item)
        if not eligible:
            continue

        best = max(eligible, key=support_score)
        best_score = support_score(best)[0]
        supports.append((best_score, page_position[page], best))

    supports.sort(key=lambda item: (-item[0], item[1], item[2].chunk.chunk_id))
    return [item for _score, _position, item in supports[:maximum]]


def _catalogue_need(need: dict[str, Any], queries: list[str]) -> dict[str, Any]:
    """Remove source identifiers while preserving technical context and ambiguity."""

    attributes = need.get("attributes")
    valid_attributes = (
        [item for item in attributes if isinstance(item, dict)]
        if isinstance(attributes, list)
        else []
    )
    has_provenance = any(
        item.get("provenance") in {"client", "derived_required", "source_product", "optional", "to_confirm"}
        for item in valid_attributes
    )
    # Compatibilite V11 : un ancien besoin sans provenance continue d'envoyer
    # seulement selectors/constraints au selecteur. En V12, le cahier+JSON
    # marque explicitement la provenance, donc le contexte source utile et les
    # ambiguïtés sont conserves pour la decision finale.
    safe_attributes = (
        valid_attributes
        if has_provenance
        else [
            item for item in valid_attributes
            if item.get("role") in {"selector", "constraint"}
        ] or valid_attributes
    )
    open_questions = need.get("open_questions")
    safe_questions = (
        [item for item in open_questions if isinstance(item, str) and item.strip()]
        if isinstance(open_questions, list)
        else []
    )
    return {
        "product_type": need.get("product_type"),
        "attributes": safe_attributes,
        "open_questions": safe_questions,
        "search_queries": list(queries),
    }


def _reference_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())




_VARIANT_PATTERNS = (
    (re.compile(r"\b(?:standard(?: type)?|version standard|type standard)\b", re.I), "Standard"),
    (re.compile(r"\b(?:deluxe(?: type)?|version deluxe|type deluxe)\b", re.I), "Deluxe"),
)
_EXPLICIT_DELUXE_REQUIREMENT = re.compile(
    r"\b(?:deluxe|premium)\b.{0,80}\b(?:required|requested|mandatory|exig[eé]e?|demand[eé]e?|obligatoire)\b"
    r"|\b(?:required|requested|mandatory|exig[eé]e?|demand[eé]e?|obligatoire)\b.{0,80}\b(?:deluxe|premium)\b",
    re.I,
)


def _infer_candidate_variant(candidate: dict[str, Any]) -> str | None:
    variant = candidate.get("variant")
    if isinstance(variant, str) and variant.strip():
        text = variant.strip()
        for pattern, canonical in _VARIANT_PATTERNS:
            if pattern.search(text):
                return canonical
        return text
    corpus = " ".join(
        [
            normalise_explanation(candidate.get("reason")),
            " ".join(normalise_text_list(candidate.get("evidence", []))),
            " ".join(normalise_text_list(candidate.get("differences", []))),
            " ".join(normalise_text_list(candidate.get("mandatory_differences", []))),
        ]
    )
    for pattern, canonical in _VARIANT_PATTERNS:
        if pattern.search(corpus):
            return canonical
    return None


def _deluxe_explicitly_required(candidates: list[dict[str, Any]]) -> bool:
    corpus = " ".join(
        normalise_explanation(value)
        for candidate in candidates
        for value in (
            candidate.get("reason"),
            candidate.get("evidence"),
            candidate.get("differences"),
            candidate.get("mandatory_differences"),
        )
    )
    return bool(_EXPLICIT_DELUXE_REQUIREMENT.search(corpus))


def _prefer_base_variant(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for candidate in candidates:
        item = dict(candidate)
        item["variant"] = _infer_candidate_variant(item)
        enriched.append(item)
    if _deluxe_explicitly_required(enriched):
        return enriched
    has_standard = any(item.get("variant") == "Standard" for item in enriched)
    has_deluxe = any(item.get("variant") == "Deluxe" for item in enriched)
    if has_standard and has_deluxe:
        priority = {"Standard": 0, "Deluxe": 1}
        enriched.sort(
            key=lambda item: (
                priority.get(item.get("variant"), 2),
                item.get("rank", 999),
                _reference_key(item.get("reference")),
            )
        )
    return enriched


def _french_variant_reason(candidate: dict[str, Any], *, rank: int, total: int) -> str | None:
    variant = candidate.get("variant")
    variants = {item for item in (candidate.get("_all_variants") or []) if item}
    if variants == {"Standard", "Deluxe"}:
        if variant == "Standard" and rank == 1:
            return (
                "Respecte les critères techniques confirmés. La version Standard est "
                "classée en premier, car aucune exigence ne justifie une version Deluxe."
            )
        if variant == "Deluxe":
            return (
                "Respecte les mêmes critères techniques confirmés. La version Deluxe est "
                "proposée comme alternative, sans exigence spécifique la rendant prioritaire."
            )
    return None


def _candidate_label(candidate: dict[str, Any]) -> str:
    reference = str(candidate.get("reference") or "").strip()
    variant = candidate.get("variant")
    if isinstance(variant, str) and variant.strip():
        return f"{reference} ({variant.strip()})"
    return reference


def _sentence(value: Any) -> str:
    text = normalise_explanation(value).strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


_STALE_RANKING_SENTENCE = re.compile(
    r"\b(?:rank(?:ed|ing)?|first|second|third|premi[eè]r(?:e)?|"
    r"deuxi[eè]me|troisi[eè]me|rang|class[eé](?:e)?|top candidate)\b",
    flags=re.IGNORECASE,
)


def _clean_final_candidate_reason(value: Any, *, rank: int, total: int) -> str:
    """Remove statements tied to a pre-filter ranking that is no longer true."""

    text = normalise_explanation(value).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    kept = [sentence for sentence in sentences if not _STALE_RANKING_SENTENCE.search(sentence)]
    cleaned = " ".join(kept).strip()
    if cleaned:
        return cleaned
    if total == 1:
        return "Reference catalogue validee et conservee comme candidat unique."
    if rank == 1:
        return "Reference catalogue validee et conservee comme meilleur candidat."
    return f"Reference catalogue validee comme alternative de rang {rank}."


_STALE_DIFFERENCE = re.compile(
    r"\b(?:deluxe|standard|premium|variant(?:e)?|alternative|candidate|candidat|"
    r"rank(?:ed|ing)?|rang|class[eé](?:e)?)\b",
    flags=re.IGNORECASE,
)
_NO_DIFFERENCE = re.compile(
    r"^\s*(?:no|none|aucun(?:e)?|sans)\b.*\bdiff[eé]rence",
    flags=re.IGNORECASE,
)


def _clean_final_candidate_differences(value: Any, *, total: int) -> list[str]:
    """Keep source-product gaps while removing comparisons to vanished peers."""

    differences = [
        item for item in normalise_text_list(value)
        if not _NO_DIFFERENCE.search(item)
    ]
    if total != 1:
        return differences
    return [
        item for item in differences
        if not _STALE_DIFFERENCE.search(item)
    ]


def _normalise_constraint_assessment_for_pipeline(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    status_aliases = {
        "satisfied": "satisfait", "match": "satisfait", "satisfait": "satisfait",
        "different": "different", "mismatch": "different", "différent": "different",
        "unproved": "non_prouve", "not_proved": "non_prouve",
        "unknown": "non_prouve", "non_verifiable": "non_prouve",
        "non_prouve": "non_prouve",
    }
    for item in raw:
        if not isinstance(item, dict):
            continue
        field = str(item.get("champ") or item.get("name") or "").strip()
        expected = str(item.get("attendu") or item.get("expected") or "").strip()
        found_raw = item.get("trouve", item.get("found"))
        found = None if found_raw is None else str(found_raw).strip() or None
        status_raw = str(item.get("statut") or item.get("status") or "").strip()
        status_key = status_raw.casefold().replace("-", "_").replace(" ", "_")
        status = status_aliases.get(status_key)
        if not field or not expected or status is None:
            continue
        criticality = str(
            item.get("criticalite") or item.get("criticality") or "indeterminee"
        ).strip().casefold().replace("-", "_").replace(" ", "_")
        if criticality not in {
            "fonction", "configuration", "interface", "performance",
            "conformite", "securite", "secondaire", "indeterminee",
        }:
            criticality = "indeterminee"
        provenance = str(
            item.get("provenance") or item.get("origine") or "source_product"
        ).strip().casefold().replace("-", "_").replace(" ", "_")
        result.append({
            "champ": field,
            "attendu": expected,
            "trouve": found,
            "statut": status,
            "criticalite": criticality,
            "provenance": provenance,
        })
    return result


def _candidate_equivalence_status(candidate: dict[str, Any]) -> str:
    assessment = _normalise_constraint_assessment_for_pipeline(
        candidate.get("constraint_assessment", [])
    )
    mandatory = [
        item for item in assessment
        if item.get("provenance") in {"client", "derived_required"}
    ]
    if any(
        item.get("statut") == "different" and item.get("criticalite") == "fonction"
        for item in mandatory
    ):
        return "non_equivalent"
    if any(item.get("statut") == "non_prouve" for item in mandatory):
        return "non_verifiable"
    if any(item.get("statut") == "different" for item in mandatory):
        return "alternative_conditionnelle"
    if normalise_text_list(candidate.get("mandatory_differences", [])):
        return "alternative_conditionnelle"

    match_level = str(candidate.get("match_level") or "").strip().casefold()
    aliases = {
        "equivalent": "equivalent_direct",
        "equivalent_direct": "equivalent_direct",
        "alternative_conditionnelle": "alternative_conditionnelle",
        "non_equivalent": "non_equivalent",
        "non_verifiable": "non_verifiable",
    }
    return aliases.get(match_level, "proximite_documentee")


def _finalise_candidate_collection(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute ranks and rank-dependent text after filtering/deduplication."""

    candidates = _prefer_base_variant(candidates)
    total = len(candidates)
    all_variants = [item.get("variant") for item in candidates]
    final: list[dict[str, Any]] = []
    for rank, raw_candidate in enumerate(candidates, start=1):
        candidate = dict(raw_candidate)
        candidate["_all_variants"] = all_variants
        candidate["rank"] = rank
        candidate["mandatory_differences"] = normalise_text_list(
            candidate.get("mandatory_differences", [])
        )
        candidate["constraint_assessment"] = _normalise_constraint_assessment_for_pipeline(
            candidate.get("constraint_assessment", [])
        )
        status = _candidate_equivalence_status(candidate)
        candidate["equivalence_status"] = status
        if status in {
            "alternative_conditionnelle", "non_equivalent", "non_verifiable"
        }:
            candidate["match_level"] = status
        french_reason = _french_variant_reason(candidate, rank=rank, total=total)
        candidate["reason"] = french_reason or _clean_final_candidate_reason(
            candidate.get("reason"),
            rank=rank,
            total=total,
        )
        candidate["differences"] = _clean_final_candidate_differences(
            candidate.get("differences", []),
            total=total,
        )
        if rank > 1 and set(all_variants) == {"Standard", "Deluxe"}:
            candidate["differences"] = [
                "Variante Deluxe au lieu de la variante Standard ; les critères techniques confirmés restent satisfaits."
            ]
        candidate.pop("_all_variants", None)
        final.append(candidate)
    return final


def _final_candidate_explanation(candidates: list[dict[str, Any]]) -> str:
    """Build a summary only from candidates that survived final normalisation.

    The model's original summary may mention candidates later rejected or
    deduplicated. Rebuilding it here keeps the explanation aligned with the
    final ranks, primary reference, and candidate count.
    """

    if not candidates:
        return ""

    best = candidates[0]
    best_label = _candidate_label(best)
    best_reason = _sentence(best.get("reason"))

    if len(candidates) == 1:
        summary = f"Un seul candidat valide a ete conserve : {best_label}."
        return f"{summary} {best_reason}".strip()

    summary = (
        f"{len(candidates)} candidats valides ont ete conserves. "
        f"Le candidat classe premier est {best_label}."
    )
    if best_reason:
        summary += f" {best_reason}"

    alternatives: list[str] = []
    for candidate in candidates[1:]:
        label = _candidate_label(candidate)
        reason = normalise_explanation(candidate.get("reason")).strip().rstrip(".!?")
        alternatives.append(f"{label} — {reason}" if reason else label)
    if alternatives:
        summary += " Alternatives : " + "; ".join(alternatives) + "."
    return summary.strip()


def _proof_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _page_texts(chunks: list[RankedChunk]) -> dict[int, str]:
    by_page: dict[int, list[str]] = {}
    for item in chunks:
        by_page.setdefault(item.chunk.page_number, []).append(item.chunk.text)
    return {
        page: " ".join(dict.fromkeys(parts))
        for page, parts in by_page.items()
    }


def _normalise_reference_parts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    parts: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        code = item.get("code", item.get("value", item.get("segment")))
        page = item.get("page")
        position = item.get("position", index)
        evidence = item.get("evidence", "")
        if not isinstance(code, str) or not code.strip():
            continue
        if type(page) is not int or page <= 0:
            continue
        if type(position) is not int or position <= 0:
            position = index
        parts.append({
            "position": position,
            "code": code.strip(),
            "page": page,
            "evidence": evidence.strip() if isinstance(evidence, str) else "",
            "field": item.get("field") if isinstance(item.get("field"), str) else "",
        })
    parts.sort(key=lambda item: item["position"])
    return parts


def _constructed_reference_is_proved(
    reference: str,
    parts: list[dict[str, Any]],
    pages: list[int],
    page_texts: dict[int, str],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if len(parts) < 2:
        return False, ["missing_reference_parts"]

    assembled = "".join(_reference_key(part["code"]) for part in parts)
    if assembled != _reference_key(reference):
        errors.append("constructed_parts_do_not_match_reference")

    ordering_page_seen = False
    for part in parts:
        page = part["page"]
        code_key = _reference_key(part["code"])
        page_text = page_texts.get(page, "")
        if page not in pages or not page_text:
            errors.append("constructed_part_page_not_proved")
            continue
        if _ORDERING_HEADING.search(page_text):
            ordering_page_seen = True
        if not code_key or code_key not in _proof_text(page_text):
            errors.append("constructed_part_not_proved")
            continue
        evidence = part.get("evidence", "")
        if evidence and _proof_text(evidence) not in _proof_text(page_text):
            errors.append("constructed_part_evidence_not_proved")

    if not ordering_page_seen:
        errors.append("ordering_page_not_proved")
    return not errors, list(dict.fromkeys(errors))


def _safe_null_result(errors: list[str], explanation: str = "") -> dict[str, Any]:
    unique_errors = list(dict.fromkeys(errors))
    detail = ", ".join(unique_errors)
    message = "Resultat rejete: aucune reference catalogue suffisamment prouvee."
    if detail:
        message += f" Motifs: {detail}."
    if explanation.strip():
        message += f" Reponse LLM: {explanation.strip()}"
    return {
        "reference": None,
        "reference_mode": None,
        "reference_parts": [],
        "catalogue_pages": [],
        "evidence": [],
        "explanation": message,
        "validation_errors": unique_errors,
    }


def _page_text_by_number(chunks: list[RankedChunk]) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for item in chunks:
        grouped.setdefault(item.chunk.page_number, []).append(item.chunk.text)
    return {page: "\n".join(texts) for page, texts in grouped.items()}


def _reference_is_explicit(reference: str, pages: list[int], page_text: dict[int, str]) -> bool:
    key = _reference_key(reference)
    if not key:
        return False
    return any(key in _reference_key(page_text.get(page, "")) for page in pages)


def _code_is_visible(code: str, text: str) -> bool:
    value = code.strip()
    if not value:
        return False
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _constructed_reference_errors(
    reference: str,
    parts_raw: Any,
    pages: list[int],
    page_text: dict[int, str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(parts_raw, list) or len(parts_raw) < 2:
        return ["invalid_reference_parts"]

    parts: list[tuple[int, str, int]] = []
    for item in parts_raw:
        if not isinstance(item, dict):
            errors.append("invalid_reference_parts")
            continue
        position = item.get("position")
        code = item.get("code")
        page = item.get("page")
        if type(position) is not int or position <= 0:
            errors.append("invalid_reference_part_position")
            continue
        if not isinstance(code, str) or not code.strip():
            errors.append("invalid_reference_part_code")
            continue
        if type(page) is not int or page <= 0:
            errors.append("invalid_reference_part_page")
            continue
        parts.append((position, code.strip(), page))

    if errors:
        return list(dict.fromkeys(errors))
    parts.sort(key=lambda item: item[0])
    if [position for position, _code, _page in parts] != list(range(1, len(parts) + 1)):
        errors.append("invalid_reference_part_order")

    declared_counts = _declared_order_slot_counts(pages, page_text)
    if declared_counts and len(parts) not in declared_counts:
        errors.append("incomplete_constructed_reference")

    assembled = "".join(code for _position, code, _page in parts)
    if _reference_key(assembled) != _reference_key(reference):
        errors.append("constructed_reference_mismatch")

    if any(page not in pages for _position, _code, page in parts):
        errors.append("constructed_part_page_not_cited")

    slot_code_maps = _declared_slot_code_maps(pages, page_text)
    for position, code, page in parts:
        allowed = slot_code_maps.get((page, position))
        if allowed is not None and _reference_key(code) not in allowed:
            errors.append("constructed_part_wrong_slot")
            break

    for _position, code, page in parts:
        if not _code_is_visible(code, page_text.get(page, "")):
            errors.append("constructed_part_not_proved")
            break

    if not any(_ORDERING_HEADING.search(page_text.get(page, "")) for page in pages):
        errors.append("missing_ordering_schema")
    return list(dict.fromkeys(errors))


def _normalise_page_numbers(raw: Any) -> list[int]:
    """Accept common LLM page shapes while preserving deterministic validation."""

    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    pages: list[int] = []
    seen: set[int] = set()
    for value in values:
        candidates: list[int] = []
        if isinstance(value, bool):
            continue
        if type(value) is int:
            candidates = [value]
        elif type(value) is float and value.is_integer():
            candidates = [int(value)]
        elif isinstance(value, str):
            cleaned = value.strip()
            if cleaned.isdigit():
                candidates = [int(cleaned)]
            elif re.fullmatch(r"\d+\.0+", cleaned):
                candidates = [int(float(cleaned))]
            else:
                candidates = [int(item) for item in re.findall(r"\d+", cleaned)]
        for page in candidates:
            if page > 0 and page not in seen:
                seen.add(page)
                pages.append(page)
    return pages


def validate_catalogue_result(
    raw_result: Any,
    source_need: dict[str, Any],
    chunks: list[RankedChunk],
) -> dict[str, Any]:
    """Reject source-copying, descriptive pseudo-references, and unproved codes."""

    if not isinstance(raw_result, dict):
        return _safe_null_result(["invalid_result_object"])

    errors: list[str] = []
    reference = raw_result.get("reference")
    if reference is not None and not isinstance(reference, str):
        errors.append("invalid_reference_type")
        reference = None
    if isinstance(reference, str):
        reference = reference.strip() or None

    pages_raw = raw_result.get("catalogue_pages")
    pages = _normalise_page_numbers(pages_raw)
    if reference is not None and not pages:
        errors.append("invalid_catalogue_pages")

    evidence_raw = raw_result.get("evidence")
    evidence = (
        [item.strip() for item in evidence_raw if isinstance(item, str) and item.strip()]
        if isinstance(evidence_raw, list)
        else []
    )
    if not isinstance(evidence_raw, list):
        errors.append("invalid_evidence")

    explanation = normalise_explanation(raw_result.get("explanation"))

    allowed_pages = {item.chunk.page_number for item in chunks}
    page_text = _page_text_by_number(chunks)
    source_reference = source_need.get("source_reference")
    if reference is not None:
        if _reference_key(reference) and _reference_key(reference) == _reference_key(source_reference):
            errors.append("source_reference_reused")
        if not pages:
            errors.append("missing_catalogue_page")
        elif any(page not in allowed_pages for page in pages):
            errors.append("catalogue_page_not_retrieved")
        if not evidence:
            errors.append("missing_catalogue_evidence")

        mode_raw = raw_result.get("reference_mode")
        mode_aliases = {
            "explicit": "explicit",
            "verbatim": "explicit",
            "constructed": "constructed",
            "assembled": "constructed",
        }
        mode = (
            mode_aliases.get(mode_raw.casefold())
            if isinstance(mode_raw, str)
            else None
        )
        reference_parts = _normalise_reference_parts(
            raw_result.get("reference_parts")
        )
        explicit = _reference_is_explicit(reference, pages, page_text)
        if mode is None and explicit:
            mode = "explicit"
        elif mode is None and reference_parts:
            mode = "constructed"
        if mode == "explicit":
            if not explicit:
                errors.extend([
                    "reference_not_proved",
                    "reference_not_explicit_or_constructed",
                ])
        elif mode == "constructed":
            errors.extend(
                _constructed_reference_errors(
                    reference,
                    reference_parts,
                    pages,
                    page_text,
                )
            )
        else:
            errors.extend([
                "reference_not_proved",
                "reference_not_explicit_or_constructed",
            ])

    errors = list(dict.fromkeys(errors))
    if errors and reference is not None:
        return _safe_null_result(errors, explanation)

    if reference is not None and mode == "constructed":
        reference = "".join(part["code"] for part in reference_parts)

    return {
        "reference": reference,
        "reference_mode": mode if reference is not None else None,
        "reference_parts": (
            reference_parts
            if reference is not None and mode == "constructed"
            else []
        ),
        "catalogue_pages": pages if reference is not None else [],
        "evidence": evidence if reference is not None else [],
        "explanation": explanation,
        "validation_errors": errors,
    }


def validate_catalogue_result_collection(
    raw_result: Any,
    source_need: dict[str, Any],
    chunks: list[RankedChunk],
) -> dict[str, Any]:
    """Validate every ranked candidate and keep all catalogue-proved references."""

    if not isinstance(raw_result, dict) or not isinstance(
        raw_result.get("candidates"), list
    ):
        return validate_catalogue_result(raw_result, source_need, chunks)

    valid_candidates: list[dict[str, Any]] = []
    rejected_errors: list[str] = []
    seen: set[str] = set()
    for index, raw_candidate in enumerate(raw_result["candidates"], start=1):
        if not isinstance(raw_candidate, dict):
            rejected_errors.append("invalid_candidate_object")
            continue
        payload = {
            "reference": raw_candidate.get("reference"),
            "reference_mode": raw_candidate.get("reference_mode"),
            "reference_parts": raw_candidate.get("reference_parts", []),
            "catalogue_pages": raw_candidate.get("catalogue_pages", []),
            "evidence": raw_candidate.get("evidence", []),
            "explanation": raw_candidate.get(
                "reason", raw_candidate.get("explanation", "")
            ),
        }
        validated = validate_catalogue_result(payload, source_need, chunks)
        reference = validated.get("reference")
        if not isinstance(reference, str) or not reference:
            rejected_errors.extend(validated.get("validation_errors", []))
            continue
        key = _reference_key(reference)
        if key in seen:
            continue
        seen.add(key)

        rank_raw = raw_candidate.get("rank", index)
        if type(rank_raw) is int and rank_raw > 0:
            rank = rank_raw
        elif type(rank_raw) is float and rank_raw.is_integer() and rank_raw > 0:
            rank = int(rank_raw)
        elif isinstance(rank_raw, str) and rank_raw.strip().isdigit() and int(rank_raw) > 0:
            rank = int(rank_raw.strip())
        else:
            rejected_errors.append("invalid_candidate_rank")
            continue

        variant = raw_candidate.get("variant")
        if variant is not None and not isinstance(variant, str):
            rejected_errors.append("invalid_candidate_variant")
            variant = None
        match_level = raw_candidate.get("match_level", "alternative")
        if not isinstance(match_level, str) or not match_level.strip():
            match_level = "alternative"
        reason = normalise_explanation(
            raw_candidate.get(
                "reason", raw_candidate.get("explanation", validated["explanation"])
            ),
            fallback=validated["explanation"],
        )
        differences = normalise_text_list(raw_candidate.get("differences", []))
        mandatory_differences = normalise_text_list(
            raw_candidate.get("mandatory_differences", [])
        )
        if mandatory_differences and match_level.strip().casefold() == "equivalent":
            match_level = "alternative"

        valid_candidates.append({
            "rank": rank,
            "reference": reference,
            "variant": variant.strip() if isinstance(variant, str) and variant.strip() else None,
            "match_level": match_level.strip(),
            "reference_mode": validated["reference_mode"],
            "reference_parts": validated["reference_parts"],
            "catalogue_pages": validated["catalogue_pages"],
            "evidence": validated["evidence"],
            "reason": reason.strip() or validated["explanation"],
            "differences": differences,
            "mandatory_differences": mandatory_differences,
            "constraint_assessment": _normalise_constraint_assessment_for_pipeline(
                raw_candidate.get("constraint_assessment", [])
            ),
            "validation_errors": [],
        })

    valid_candidates.sort(
        key=lambda item: (item["rank"], _reference_key(item["reference"]))
    )
    valid_candidates = valid_candidates[:5]
    valid_candidates = _finalise_candidate_collection(valid_candidates)

    explanation = normalise_explanation(raw_result.get("explanation", ""))
    if not valid_candidates:
        return _safe_null_result(
            list(dict.fromkeys(rejected_errors or ["no_proved_candidate"])),
            explanation,
        ) | {"candidates": []}

    best = valid_candidates[0]
    final_explanation = _final_candidate_explanation(valid_candidates)
    return {
        "reference": best["reference"],
        "reference_mode": best["reference_mode"],
        "reference_parts": best["reference_parts"],
        "catalogue_pages": best["catalogue_pages"],
        "evidence": best["evidence"],
        "explanation": final_explanation,
        "validation_errors": [],
        "candidates": valid_candidates,
    }


def normalise_generation_result(raw_result: Any) -> dict[str, Any]:
    """Apply schema normalisation and preserve every ranked catalogue candidate."""

    if isinstance(raw_result, dict) and isinstance(raw_result.get("candidates"), list):
        raw_candidates = raw_result["candidates"]
        normalised_candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, dict):
                raise ValueError("chaque candidat doit etre un objet")
            candidate_payload = {
                "reference": raw_candidate.get("reference"),
                "reference_mode": raw_candidate.get("reference_mode"),
                "reference_parts": raw_candidate.get("reference_parts", []),
                "catalogue_pages": raw_candidate.get("catalogue_pages", []),
                "evidence": raw_candidate.get("evidence", []),
                "explanation": raw_candidate.get(
                    "reason", raw_candidate.get("explanation", "")
                ),
            }
            candidate = normalise_generation_result(candidate_payload)
            reference = candidate.get("reference")
            if not isinstance(reference, str) or not reference:
                continue
            key = _reference_key(reference)
            if key in seen:
                continue
            seen.add(key)

            rank_raw = raw_candidate.get("rank", index)
            if type(rank_raw) is int and rank_raw > 0:
                rank = rank_raw
            elif type(rank_raw) is float and rank_raw.is_integer() and rank_raw > 0:
                rank = int(rank_raw)
            elif isinstance(rank_raw, str) and rank_raw.strip().isdigit() and int(rank_raw) > 0:
                rank = int(rank_raw.strip())
            else:
                raise ValueError("rank doit etre un entier positif")

            variant = raw_candidate.get("variant")
            if variant is not None and not isinstance(variant, str):
                raise ValueError("variant doit etre une chaine ou null")
            match_level = raw_candidate.get("match_level", "alternative")
            if not isinstance(match_level, str) or not match_level.strip():
                raise ValueError("match_level doit etre une chaine")
            reason = normalise_explanation(
                raw_candidate.get(
                    "reason", raw_candidate.get("explanation", "")
                ),
                fallback=candidate["explanation"],
            )
            differences = normalise_text_list(
                raw_candidate.get("differences", [])
            )
            mandatory_differences = normalise_text_list(
                raw_candidate.get("mandatory_differences", [])
            )
            if mandatory_differences and match_level.strip().casefold() == "equivalent":
                match_level = "alternative"

            normalised_candidates.append({
                "rank": rank,
                "reference": reference,
                "variant": variant.strip() if isinstance(variant, str) and variant.strip() else None,
                "match_level": match_level.strip(),
                "reference_mode": candidate["reference_mode"],
                "reference_parts": candidate["reference_parts"],
                "catalogue_pages": candidate["catalogue_pages"],
                "evidence": candidate["evidence"],
                "reason": reason.strip() or candidate["explanation"],
                "differences": differences,
                "mandatory_differences": mandatory_differences,
                "constraint_assessment": _normalise_constraint_assessment_for_pipeline(
                    raw_candidate.get("constraint_assessment", [])
                ),
                "validation_errors": [],
            })

        normalised_candidates.sort(
            key=lambda item: (item["rank"], _reference_key(item["reference"]))
        )
        normalised_candidates = normalised_candidates[:5]
        normalised_candidates = _finalise_candidate_collection(normalised_candidates)

        explanation = normalise_explanation(
            raw_result.get("explanation", ""),
            fallback=(normalised_candidates[0]["reason"] if normalised_candidates else ""),
        )
        if not normalised_candidates:
            return {
                "reference": None,
                "reference_mode": None,
                "reference_parts": [],
                "catalogue_pages": [],
                "evidence": [],
                "explanation": explanation.strip(),
                "validation_errors": [],
                "candidates": [],
            }
        best = normalised_candidates[0]
        final_explanation = _final_candidate_explanation(normalised_candidates)
        return {
            "reference": best["reference"],
            "reference_mode": best["reference_mode"],
            "reference_parts": best["reference_parts"],
            "catalogue_pages": best["catalogue_pages"],
            "evidence": best["evidence"],
            "explanation": final_explanation,
            "validation_errors": [],
            "candidates": normalised_candidates,
        }

    if not isinstance(raw_result, dict):
        raise ValueError("le LLM n'a pas retourne un objet JSON")
    reference = raw_result.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise ValueError("reference doit etre une chaine ou null")
    if isinstance(reference, str):
        reference = reference.strip() or None

    pages = _normalise_page_numbers(raw_result.get("catalogue_pages", []))

    evidence = normalise_text_list(raw_result.get("evidence", []))

    explanation = normalise_explanation(raw_result.get("explanation", ""))

    mode_raw = raw_result.get("reference_mode")
    mode_aliases = {
        "explicit": "explicit",
        "verbatim": "explicit",
        "constructed": "constructed",
        "assembled": "constructed",
    }
    if mode_raw is None:
        mode = None
    elif isinstance(mode_raw, str) and mode_raw.casefold() in mode_aliases:
        mode = mode_aliases[mode_raw.casefold()]
    else:
        raise ValueError("reference_mode doit etre explicit, constructed ou null")

    parts = _normalise_reference_parts(raw_result.get("reference_parts", []))
    if reference is None:
        mode = None
        parts = []
        pages = []
        evidence = []
    elif mode == "constructed" and parts:
        reference = "".join(part["code"] for part in parts)
    elif mode == "explicit":
        parts = []

    return {
        "reference": reference,
        "reference_mode": mode,
        "reference_parts": parts,
        "catalogue_pages": pages,
        "evidence": evidence,
        "explanation": explanation.strip(),
        "validation_errors": [],
    }


def _diagnostic_payload(
    need: dict[str, Any],
    queries: list[str],
    chunks: list[RankedChunk],
    *,
    models: dict[str, str],
    retrieval_metadata: dict[str, Any],
) -> dict[str, Any]:
    safe_need = {
        key: value
        for key, value in need.items()
        if key != "source_excerpt"
    }
    return {
        "need": safe_need,
        "queries": list(queries),
        "models": dict(models),
        "retrieval_metadata": dict(retrieval_metadata),
        "retrieval": [
            {
                "rank": rank,
                "page": item.chunk.page_number,
                "chunk_id": item.chunk.chunk_id,
                "section_id": item.chunk.section_id,
                "section_title": item.chunk.section_title,
                "kind": item.chunk.kind,
                "markers": list(item.chunk.markers),
                "score": round(item.score, 6),
                "word_score": round(item.word_score, 6),
                "char_score": round(item.char_score, 6),
                "exact_score": round(item.exact_score, 6),
                "code_score": round(item.code_score, 6),
                "dense_score": round(item.dense_score, 6),
                "lexical_rank": item.lexical_rank,
                "dense_rank": item.dense_rank,
                "fused_score": round(item.fused_score, 6),
                "rerank_score": (
                    round(item.rerank_score, 6)
                    if item.rerank_score is not None
                    else None
                ),
                "excerpt": item.chunk.text[:1200],
            }
            for rank, item in enumerate(chunks, start=1)
        ],
    }


def _generation_has_candidate(raw_result: Any) -> bool:
    """Return whether a raw generation contains at least one usable reference."""

    if not isinstance(raw_result, dict):
        return False
    candidates = raw_result.get("candidates")
    if isinstance(candidates, list):
        return any(
            isinstance(candidate, dict)
            and isinstance(candidate.get("reference"), str)
            and bool(candidate["reference"].strip())
            for candidate in candidates
        )
    reference = raw_result.get("reference")
    return isinstance(reference, str) and bool(reference.strip())


def run_search(
    product_file: Path | str,
    catalogue_pdf: Path | str,
    llm: Any,
    *,
    top_k: int = 8,
    candidate_k: int | None = None,
    embedding_client: Any | None = None,
    rerank_client: Any | None = None,
    cache_dir: Path | str = Path(".rag_cache"),
    include_diagnostics: bool = False,
    validate_result: bool = False,
    provenance_file: Path | str | None = None,
    auto_discover_provenance: bool = True,
    hierarchical: bool = True,
    hierarchy_enabled: bool | None = None,
    catalogue_profile: str = "auto",
    section_k: int | None = None,
    ocr_mode: str = "auto",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
    rerank_batch_size: int = 64,
) -> dict[str, Any]:
    """Extract a need, run hybrid retrieval, rerank, then generate ranked references."""

    if hierarchy_enabled is not None:
        hierarchical = bool(hierarchy_enabled)

    product_source = load_product_source(
        product_file,
        provenance_file=provenance_file,
        auto_discover=auto_discover_provenance,
    )
    product_text = product_source.render_for_llm()
    need = llm.extract_need(product_text)
    if not isinstance(need, dict):
        raise ValueError("le LLM n'a pas produit un besoin JSON")
    need = merge_structured_provenance_into_need(
        need,
        product_source.provenance,
    )
    queries = build_retrieval_queries(need)
    retrieval_metadata: dict[str, Any] = {
        "input_mode": (
            "full_cahier_plus_provenance"
            if product_source.provenance_path is not None
            else "full_cahier_only"
        ),
        "provenance_file": (
            str(product_source.provenance_path.resolve())
            if product_source.provenance_path is not None
            else None
        ),
        "structured_provenance_version": need.get(
            "structured_provenance_version"
        ),
        "request_type": need.get("request_type"),
    }

    ranked_chunks = retrieve_catalogue_chunks(
        catalogue_pdf,
        queries,
        top_k=top_k,
        candidate_k=candidate_k,
        embedding_client=embedding_client,
        rerank_client=rerank_client,
        cache_dir=cache_dir,
        retrieval_metadata=retrieval_metadata,
        hierarchy_enabled=hierarchical,
        catalogue_profile=catalogue_profile,
        section_k=section_k,
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
        rerank_batch_size=rerank_batch_size,
    )
    selection_need = _catalogue_need(need, queries)
    raw_result = llm.select_reference(selection_need, ranked_chunks)
    generation_attempts = 1
    if ranked_chunks and not _generation_has_candidate(raw_result):
        retry = getattr(llm, "repair_reference", None)
        if callable(retry):
            raw_result = retry(
                selection_need,
                ranked_chunks,
                raw_result,
                ["empty_candidate_list"],
            )
        else:
            retry_need = dict(selection_need)
            retry_need["validation_feedback"] = ["empty_candidate_list"]
            retry_need["previous_result"] = raw_result
            raw_result = llm.select_reference(retry_need, ranked_chunks)
        generation_attempts = 2
    retrieval_metadata["generation_attempts"] = generation_attempts
    retrieval_metadata["empty_candidate_retry"] = generation_attempts == 2
    if validate_result:
        result = validate_catalogue_result_collection(raw_result, need, ranked_chunks)

        repairable_errors = {
            "incomplete_constructed_reference",
            "constructed_reference_mismatch",
            "invalid_reference_parts",
            "invalid_reference_part_order",
            "reference_not_explicit_or_constructed",
        }
        repair = getattr(llm, "repair_reference", None)
        if (
            callable(repair)
            and repairable_errors.intersection(result.get("validation_errors", []))
        ):
            repaired_raw = repair(
                selection_need,
                ranked_chunks,
                raw_result,
                list(result.get("validation_errors", [])),
            )
            result = validate_catalogue_result_collection(repaired_raw, need, ranked_chunks)
    else:
        result = normalise_generation_result(raw_result)

    generation_model = str(
        getattr(llm, "model_used", getattr(llm, "model", ""))
    )
    models = {
        "embedding": str(getattr(embedding_client, "model", "")),
        "reranking": str(getattr(rerank_client, "model", "")),
        "generation": generation_model,
    }
    routing = getattr(llm, "model_routing_config", None)
    if isinstance(routing, dict):
        primary_model = routing.get("primary_model")
        fallback_model = routing.get("fallback_model")
        if primary_model:
            models["generation_primary"] = str(primary_model)
        if fallback_model:
            models["generation_fallback"] = str(fallback_model)
        result["model_routing"] = dict(routing)
        retrieval_metadata["model_routing"] = dict(routing)
    result["model"] = generation_model
    result["models"] = models
    reasoning = getattr(llm, "reasoning_config", None)
    if isinstance(reasoning, dict):
        result["reasoning"] = dict(reasoning)
    json_repair = getattr(llm, "json_repair_config", None)
    if isinstance(json_repair, dict):
        for key in (
            "attempts", "successes", "local_attempts", "local_successes",
            "external_attempts", "external_successes", "truncated_responses",
        ):
            retrieval_metadata[f"json_repair_{key}"] = int(
                json_repair.get(key, 0) or 0
            )
    if include_diagnostics:
        result["diagnostic"] = _diagnostic_payload(
            need,
            queries,
            ranked_chunks,
            models=models,
            retrieval_metadata=retrieval_metadata,
        )
    return result
