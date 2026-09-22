"""Human-like catalogue navigation for large industrial PDFs.

V14 separates two questions:

1. Is the catalogue structured enough to navigate like a human?
2. If so, which broad section and sub-section should be searched first?

The module is deliberately domain-neutral.  It uses PDF outlines, textual tables
of contents, layout headings, page ranges and retrieval evidence.  When the
structure or route is weak, callers fall back to the normal global RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, Sequence

from .catalogue_map import CatalogueMap, CatalogueSection
from .pdf_text import CatalogueChunk, PdfPage
from .retrieval import RankedChunk


_TOC_HEADING = re.compile(
    r"\b(?:table\s+of\s+contents|general\s+contents|contents|sommaire|index)\b",
    re.IGNORECASE,
)
_GENERIC_TITLES = {
    "page", "front matter", "catalogue", "catalog", "contents", "sommaire",
    "table of contents", "index", "product catalogue", "general contents",
}
_TOC_RANGE = re.compile(
    r"^(?P<title>.*?[A-Za-zÀ-ÖØ-öø-ÿ®™)])\s+"
    r"(?P<start>\d{1,4})\s*[-–—]\s*(?P<end>\d{1,4})\s*$"
)

_TOC_SINGLE_WITH_LEADER = re.compile(
    r"^(?P<title>.*?[A-Za-zÀ-ÖØ-öø-ÿ®™)])(?:\s*\.{2,}\s*|\s{3,})"
    r"(?P<start>\d{1,4})\s*$"
)
_REFERENCE_SIGNAL = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._/+\-]{4,}\b)"
    r"(?=[A-Za-z0-9._/+\-]*[A-Za-z])(?=[A-Za-z0-9._/+\-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9._/+\-]{3,}(?![A-Za-z0-9])"
)
_QUERY_STOPWORDS = {
    "avec", "sans", "pour", "dans", "type", "produit", "product", "the",
    "and", "from", "une", "des", "les", "sur", "par", "catalogue", "catalog",
    "reference", "référence", "modele", "modèle", "technical", "technique",
}


def _fold(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _tokens(text: object) -> list[str]:
    return [token for token in _fold(text).split() if len(token) >= 2 and token not in _QUERY_STOPWORDS]


def _normalise_title(text: object) -> str:
    value = " ".join(str(text or "").replace("\u00a0", " ").split())
    value = re.sub(r"\s*\.{2,}\s*", " ", value)
    return value.strip(" -–—:;,.|")[:180]


@dataclass(frozen=True, slots=True)
class NavigationNode:
    node_id: str
    title: str
    start_page: int
    end_page: int
    page_numbers: tuple[int, ...]
    level: int
    parent_id: str | None
    summary: str
    source: str
    section_ids: tuple[str, ...] = ()

    @property
    def span(self) -> int:
        return max(1, self.end_page - self.start_page + 1)


@dataclass(frozen=True, slots=True)
class CatalogueNavigationMap:
    nodes: tuple[NavigationNode, ...]
    source: str
    structure_confidence: float
    reasons: tuple[str, ...]
    page_count: int

    def node(self, node_id: str) -> NavigationNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def route_path(self, node_id: str) -> tuple[str, ...]:
        by_id = {node.node_id: node for node in self.nodes}
        node = by_id.get(node_id)
        if node is None:
            return ()
        path: list[str] = []
        visited: set[str] = set()
        while node is not None and node.node_id not in visited:
            visited.add(node.node_id)
            path.append(node.title)
            node = by_id.get(node.parent_id or "")
        return tuple(reversed(path))


@dataclass(frozen=True, slots=True)
class NavigationPlan:
    strategy: str
    structure_confidence: float
    route_confidence: float
    primary_node_id: str | None
    secondary_node_ids: tuple[str, ...]
    route_path: tuple[str, ...]
    primary_pages: frozenset[int]
    secondary_pages: frozenset[int]
    global_fallback_used: bool
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class LocalEvidenceQuality:
    coverage: float
    score: float
    sufficient: bool
    has_reference_signal: bool
    has_structural_signal: bool
    matched_tokens: tuple[str, ...]
    missing_tokens: tuple[str, ...]


def _parse_toc_line(line: str) -> list[tuple[str, int, int]]:
    value = _normalise_title(line)
    if not value or _TOC_HEADING.fullmatch(_fold(value)):
        return []
    visible_ranges = list(re.finditer(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})", value))
    direct = _TOC_RANGE.match(value) or _TOC_SINGLE_WITH_LEADER.match(value)
    raw_matches: list[object]
    if direct is not None and len(visible_ranges) <= 1:
        raw_matches = [direct]
    else:
        flattened = re.sub(
            r"^(?:table\s+of\s+contents|general\s+contents|contents|sommaire|index)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        range_matches = list(re.finditer(r"(\d{1,4})\s*[-–—]\s*(\d{1,4})", flattened))
        cursor = 0
        raw_matches = []
        for range_match in range_matches:
            title_part = flattened[cursor:range_match.start()]
            raw_matches.append((title_part, range_match.group(1), range_match.group(2)))
            cursor = range_match.end()

    parsed: list[tuple[str, int, int]] = []
    for match in raw_matches:
        if isinstance(match, tuple):
            title = _normalise_title(match[0])
            start = int(match[1])
            end = int(match[2])
        else:
            title = _normalise_title(match.group("title"))
            start = int(match.group("start"))
            end_group = match.groupdict().get("end")
            end = int(end_group) if end_group else start
        if title and len(_fold(title)) >= 3:
            parsed.append((title, start, end))
    return parsed


def _parse_toc_entries(pages: Sequence[PdfPage], page_count: int) -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    scan_limit = min(len(pages), max(8, min(32, page_count // 10 + 8)))
    toc_open = False
    continuation_misses = 0
    for page in pages[:scan_limit]:
        heading_context = f"{page.heading}\n{page.text[:500]}"
        has_heading = bool(_TOC_HEADING.search(heading_context))
        if has_heading:
            toc_open = True
            continuation_misses = 0
        elif not toc_open:
            continue

        page_entries: list[tuple[str, int, int]] = []
        for raw_line in page.text.splitlines():
            page_entries.extend(_parse_toc_line(raw_line))

        for title, start, end in page_entries:
            if not (1 <= start <= end <= page_count):
                continue
            key = (_fold(title), start, end)
            if key in seen:
                continue
            seen.add(key)
            entries.append((title, start, end))

        if page_entries:
            continuation_misses = 0
        elif toc_open and not has_heading:
            continuation_misses += 1
            if continuation_misses >= 1:
                break
    return entries


def _title_overlap(title: str, page: PdfPage) -> float:
    wanted = set(_tokens(title))
    if not wanted:
        return 0.0
    observed = set(_tokens(f"{page.heading} {page.text[:700]}"))
    return len(wanted & observed) / len(wanted)


def _infer_page_offset(entries: Sequence[tuple[str, int, int]], pages: Sequence[PdfPage]) -> int:
    """Infer a small printed-page/PDF-page offset from visible headings."""

    by_page = {page.page_number: page for page in pages}
    if len(entries) < 3:
        return 0
    scored: list[tuple[float, int, int]] = []
    for offset in range(-6, 7):
        total = 0.0
        supports = 0
        for title, start, _end in entries[:36]:
            page = by_page.get(start + offset)
            if page is None:
                continue
            overlap = _title_overlap(title, page)
            if overlap >= 0.45:
                supports += 1
                total += overlap
        scored.append((total, supports, offset))
    scored.sort(reverse=True)
    best_total, best_supports, best_offset = scored[0]
    zero_total, zero_supports, _ = next(item for item in scored if item[2] == 0)
    if best_offset and best_supports >= 2 and (
        best_total >= zero_total + 0.8 or best_supports >= zero_supports + 2
    ):
        return best_offset
    return 0


def _entry_parent_indices(entries: Sequence[tuple[str, int, int]]) -> list[int | None]:
    parents: list[int | None] = []
    for index, (_title, start, end) in enumerate(entries):
        candidates: list[tuple[int, int]] = []
        own_span = end - start
        for other_index, (_other_title, other_start, other_end) in enumerate(entries):
            if index == other_index:
                continue
            other_span = other_end - other_start
            if other_start <= start and end <= other_end and other_span > own_span:
                candidates.append((other_span, other_index))
        parents.append(min(candidates)[1] if candidates else None)
    return parents


def _section_ids_for_range(
    catalogue_map: CatalogueMap,
    start_page: int,
    end_page: int,
) -> tuple[str, ...]:
    return tuple(
        section.section_id
        for section in catalogue_map.sections
        if section.start_page <= end_page and start_page <= section.end_page
    )


def _page_summary(pages_by_number: dict[int, PdfPage], start: int, end: int) -> str:
    snippets: list[str] = []
    candidates = [start, start + 1, end]
    for page_number in candidates:
        page = pages_by_number.get(page_number)
        if page is None:
            continue
        heading = _normalise_title(page.heading)
        text = " ".join(page.text.split())[:420]
        piece = " — ".join(value for value in (heading, text) if value)
        if piece and piece not in snippets:
            snippets.append(piece)
    return " ".join(snippets)[:1400]


def _nodes_from_text_toc(
    entries: Sequence[tuple[str, int, int]],
    catalogue_map: CatalogueMap,
    pages: Sequence[PdfPage],
) -> tuple[NavigationNode, ...]:
    offset = _infer_page_offset(entries, pages)
    adjusted: list[tuple[str, int, int]] = []
    for title, start, end in entries:
        shifted_start = max(1, min(catalogue_map.page_count, start + offset))
        shifted_end = max(shifted_start, min(catalogue_map.page_count, end + offset))
        adjusted.append((title, shifted_start, shifted_end))
    parents = _entry_parent_indices(adjusted)
    pages_by_number = {page.page_number: page for page in pages}
    levels: dict[int, int] = {}

    def level_for(index: int, trail: set[int] | None = None) -> int:
        if index in levels:
            return levels[index]
        trail = set() if trail is None else set(trail)
        if index in trail:
            levels[index] = 1
            return 1
        trail.add(index)
        parent = parents[index]
        value = 1 if parent is None else level_for(parent, trail) + 1
        levels[index] = min(value, 8)
        return levels[index]

    nodes: list[NavigationNode] = []
    for index, (title, start, end) in enumerate(adjusted, start=1):
        parent_index = parents[index - 1]
        node_id = f"n{index:03d}"
        child_titles = [
            adjusted[child_index][0]
            for child_index, parent in enumerate(parents)
            if parent == index - 1
        ]
        summary = (
            f"NAVIGATION: {title}. Pages {start}-{end}. "
            + ("Sous-sections: " + "; ".join(child_titles) + ". " if child_titles else "")
            + _page_summary(pages_by_number, start, end)
        ).strip()
        nodes.append(
            NavigationNode(
                node_id=node_id,
                title=title,
                start_page=start,
                end_page=end,
                page_numbers=tuple(range(start, end + 1)),
                level=level_for(index - 1),
                parent_id=f"n{parent_index + 1:03d}" if parent_index is not None else None,
                summary=summary,
                source="text-toc",
                section_ids=_section_ids_for_range(catalogue_map, start, end),
            )
        )
    return tuple(nodes)


def _nodes_from_sections(catalogue_map: CatalogueMap) -> tuple[NavigationNode, ...]:
    nodes: list[NavigationNode] = []
    stack: list[tuple[int, str]] = []
    for index, section in enumerate(catalogue_map.sections, start=1):
        while stack and stack[-1][0] >= section.level:
            stack.pop()
        parent_id = stack[-1][1] if stack else None
        node_id = f"n{index:03d}"
        stack.append((section.level, node_id))
        nodes.append(
            NavigationNode(
                node_id=node_id,
                title=section.title,
                start_page=section.start_page,
                end_page=section.end_page,
                page_numbers=section.page_numbers,
                level=max(1, section.level),
                parent_id=parent_id,
                summary=section.summary,
                source=catalogue_map.source,
                section_ids=(section.section_id,),
            )
        )
    return tuple(nodes)


def _structure_confidence(
    nodes: Sequence[NavigationNode],
    *,
    source: str,
    page_count: int,
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    if source == "text-toc":
        base = 0.82
        reasons.append("textual_toc")
    elif source == "toc":
        base = 0.86
        reasons.append("pdf_outline")
    elif source == "ocr-layout":
        base = 0.42
        reasons.append("ocr_layout_headings")
    else:
        base = 0.36
        reasons.append("layout_headings")

    if not nodes:
        return 0.0, ("no_navigation_nodes",)
    meaningful = [
        node for node in nodes
        if _fold(node.title) not in _GENERIC_TITLES
        and not _fold(node.title).startswith("page ")
        and len(_tokens(node.title)) >= 1
    ]
    meaningful_ratio = len(meaningful) / len(nodes)
    hierarchy_ratio = sum(node.level > 1 for node in nodes) / len(nodes)
    covered = {
        page
        for node in nodes
        for page in node.page_numbers
        if 1 <= page <= page_count
    }
    coverage = len(covered) / max(1, page_count)
    unique_titles = len({_fold(node.title) for node in meaningful}) / max(1, len(meaningful))

    score = base
    score += 0.12 * min(1.0, len(nodes) / 8)
    score += 0.10 * meaningful_ratio
    score += 0.06 * unique_titles
    score += 0.06 * min(1.0, coverage / 0.60)
    score += 0.05 * min(1.0, hierarchy_ratio / 0.20)
    if len(nodes) == 1:
        score -= 0.35
        reasons.append("single_section")
    if meaningful_ratio < 0.5:
        score -= 0.25
        reasons.append("generic_titles")
    if source == "layout" and len(nodes) <= 2:
        score -= 0.15
    return max(0.0, min(1.0, score)), tuple(reasons)


def build_navigation_map(
    catalogue_map: CatalogueMap,
    pages: Iterable[PdfPage],
) -> CatalogueNavigationMap:
    values = tuple(sorted(pages, key=lambda page: page.page_number))
    entries = _parse_toc_entries(values, catalogue_map.page_count)
    if len(entries) >= 2:
        nodes = _nodes_from_text_toc(entries, catalogue_map, values)
        source = "text-toc"
    else:
        nodes = _nodes_from_sections(catalogue_map)
        source = catalogue_map.source
    confidence, reasons = _structure_confidence(
        nodes,
        source=source,
        page_count=catalogue_map.page_count,
    )
    return CatalogueNavigationMap(
        nodes=nodes,
        source=source,
        structure_confidence=confidence,
        reasons=reasons,
        page_count=catalogue_map.page_count,
    )


def navigation_chunks(navigation: CatalogueNavigationMap) -> list[CatalogueChunk]:
    return [
        CatalogueChunk(
            chunk_id=f"navigation-{node.node_id}",
            page_number=node.start_page,
            text=node.summary,
            section_id=node.node_id,
            section_title=node.title,
            kind="navigation",
        )
        for node in navigation.nodes
        if node.summary.strip()
    ]


_ROUTE_AID_MARKERS = (
    "guide de choix",
    "guide du choix",
    "guide de selection",
    "selection guide",
    "selection chart",
    "choice guide",
    "choosing guide",
)


def _is_navigation_aid(title: object) -> bool:
    """A section that points at products rather than listing orderable ones."""

    folded = _fold(title)
    if folded in _GENERIC_TITLES:
        return True
    return any(marker in folded for marker in _ROUTE_AID_MARKERS)


def _node_is_ancestor(
    navigation: CatalogueNavigationMap,
    ancestor_id: str,
    descendant_id: str,
) -> bool:
    by_id = {node.node_id: node for node in navigation.nodes}
    node = by_id.get(descendant_id)
    visited: set[str] = set()
    while node is not None and node.parent_id and node.node_id not in visited:
        visited.add(node.node_id)
        if node.parent_id == ancestor_id:
            return True
        node = by_id.get(node.parent_id)
    return False


def _route_signal(item: RankedChunk) -> float:
    # Choosing a chapter is a question of subject, so the signal has to come
    # from term weighting. word_score and char_score are TF-IDF based, and
    # code_score only fires on alphanumeric part numbers, which are decisive on
    # their own. exact_score is a plain set overlap: bare figures such as "250"
    # or "400" weigh as much as "interrupteur-sectionneur", which hands the
    # route to whichever section carries the most tables rather than to the one
    # about the right product. It stays out of routing and keeps its role in
    # ranking chunks once the section is chosen.
    lexical = max(item.word_score, item.char_score, item.code_score)
    if lexical > 0:
        return max(0.0, min(1.0, lexical))
    fused = item.fused_score or item.score
    return max(0.0, min(1.0, fused * 8.0))


def plan_navigation(
    navigation: CatalogueNavigationMap,
    ranked_nodes: Sequence[RankedChunk],
    *,
    maximum_routes: int = 3,
    minimum_structure_confidence: float = 0.50,
    minimum_route_confidence: float = 0.42,
) -> NavigationPlan:
    if maximum_routes <= 0:
        raise ValueError("maximum_routes doit etre strictement positif")
    if navigation.structure_confidence < minimum_structure_confidence:
        return NavigationPlan(
            strategy="global",
            structure_confidence=navigation.structure_confidence,
            route_confidence=0.0,
            primary_node_id=None,
            secondary_node_ids=(),
            route_path=(),
            primary_pages=frozenset(),
            secondary_pages=frozenset(),
            global_fallback_used=True,
            fallback_reason="structure_unreliable",
        )
    by_id = {node.node_id: node for node in navigation.nodes}
    candidates: list[tuple[float, int, NavigationNode, RankedChunk]] = []
    for position, item in enumerate(ranked_nodes, start=1):
        node = by_id.get(item.chunk.section_id)
        if node is None:
            continue
        signal = _route_signal(item)
        specificity = 0.035 * max(0, node.level - 1)
        specificity += 0.06 * (1.0 - min(1.0, node.span / max(1, navigation.page_count)))
        candidates.append((signal + specificity, position, node, item))
    if not candidates:
        return NavigationPlan(
            strategy="global",
            structure_confidence=navigation.structure_confidence,
            route_confidence=0.0,
            primary_node_id=None,
            secondary_node_ids=(),
            route_path=(),
            primary_pages=frozenset(),
            secondary_pages=frozenset(),
            global_fallback_used=True,
            fallback_reason="route_not_found",
        )
    candidates.sort(key=lambda value: (-value[0], value[1], value[2].span, value[2].node_id))

    # When a parent and a child have nearly identical support, a human enters
    # the more specific child rather than searching the entire parent chapter.
    # A selection guide or a summary scores well on a need: it restates the
    # whole chapter's vocabulary. It holds no orderable reference though, so
    # routing there guarantees a search with nothing to find. Such a section
    # stays available as secondary context but never becomes the destination.
    destinations = [item for item in candidates if not _is_navigation_aid(item[2].title)]
    if not destinations:
        destinations = candidates
    primary = destinations[0]
    for candidate in destinations[1:]:
        if _node_is_ancestor(navigation, primary[2].node_id, candidate[2].node_id):
            if candidate[0] >= primary[0] * 0.82:
                primary = candidate
                break
    primary_score, _position, primary_node, primary_item = primary

    # If many independent sections are essentially tied, a narrow route would
    # hide valid alternatives. In that case V14 deliberately returns to the
    # normal global RAG instead of pretending that one chapter is certain.
    # Only sections that could themselves be the destination make the route
    # ambiguous; a guide scoring well alongside them is not a rival chapter.
    competitive_nodes: list[NavigationNode] = []
    for score, _pos, node, _item in destinations:
        if score < primary_score * 0.86:
            continue
        if _node_is_ancestor(navigation, node.node_id, primary_node.node_id):
            continue
        if _node_is_ancestor(navigation, primary_node.node_id, node.node_id):
            continue
        competitive_nodes.append(node)
    if len(competitive_nodes) > maximum_routes:
        return NavigationPlan(
            strategy="global",
            structure_confidence=navigation.structure_confidence,
            route_confidence=0.35,
            primary_node_id=primary_node.node_id,
            secondary_node_ids=tuple(node.node_id for node in competitive_nodes[: maximum_routes - 1]),
            route_path=navigation.route_path(primary_node.node_id),
            primary_pages=frozenset(primary_node.page_numbers),
            secondary_pages=frozenset(
                page for node in competitive_nodes[: maximum_routes - 1] for page in node.page_numbers
            ),
            global_fallback_used=True,
            fallback_reason="multiple_competing_sections",
        )

    secondary: list[NavigationNode] = []
    for score, _pos, node, _item in candidates:
        if node.node_id == primary_node.node_id:
            continue
        if _node_is_ancestor(navigation, node.node_id, primary_node.node_id):
            continue
        if _node_is_ancestor(navigation, primary_node.node_id, node.node_id):
            continue
        same_parent = bool(primary_node.parent_id) and node.parent_id == primary_node.parent_id
        if primary_node.parent_id and not same_parent:
            continue
        threshold = primary_score * (0.22 if same_parent else 0.80)
        if score < max(0.12, threshold):
            continue
        secondary.append(node)
        if len(secondary) >= max(0, maximum_routes - 1):
            break

    second_score = next(
        (score for score, _pos, node, _item in candidates if node.node_id != primary_node.node_id),
        0.0,
    )
    margin = max(0.0, min(1.0, primary_score - second_score + 0.25))
    route_confidence = max(
        0.0,
        min(
            1.0,
            0.45 * navigation.structure_confidence
            + 0.40 * _route_signal(primary_item)
            + 0.15 * margin,
        ),
    )
    if route_confidence < minimum_route_confidence:
        return NavigationPlan(
            strategy="global",
            structure_confidence=navigation.structure_confidence,
            route_confidence=route_confidence,
            primary_node_id=primary_node.node_id,
            secondary_node_ids=tuple(node.node_id for node in secondary),
            route_path=navigation.route_path(primary_node.node_id),
            primary_pages=frozenset(primary_node.page_numbers),
            secondary_pages=frozenset(page for node in secondary for page in node.page_numbers),
            global_fallback_used=True,
            fallback_reason="route_low_confidence",
        )
    return NavigationPlan(
        strategy="hierarchical",
        structure_confidence=navigation.structure_confidence,
        route_confidence=route_confidence,
        primary_node_id=primary_node.node_id,
        secondary_node_ids=tuple(node.node_id for node in secondary),
        route_path=navigation.route_path(primary_node.node_id),
        primary_pages=frozenset(primary_node.page_numbers),
        secondary_pages=frozenset(page for node in secondary for page in node.page_numbers),
        global_fallback_used=False,
        fallback_reason=None,
    )


def expand_route_pages(
    plan: NavigationPlan,
    *,
    page_count: int,
    neighbour_radius: int = 1,
) -> tuple[set[int], set[int]]:
    if neighbour_radius < 0:
        raise ValueError("neighbour_radius doit etre positif ou nul")

    def expanded(values: Iterable[int]) -> set[int]:
        result: set[int] = set()
        for page in values:
            for candidate in range(page - neighbour_radius, page + neighbour_radius + 1):
                if 1 <= candidate <= page_count:
                    result.add(candidate)
        return result

    primary = expanded(plan.primary_pages)
    secondary = expanded(plan.secondary_pages) - primary
    return primary, secondary


def assess_local_evidence(
    candidates: Sequence[RankedChunk],
    queries: Sequence[str],
) -> LocalEvidenceQuality:
    combined = " ".join(item.chunk.text for item in candidates[:24])
    observed = set(_tokens(combined))
    # Queries restate one need in several wordings, and often in several
    # languages. Pooling their tokens lets a translated variant that shares no
    # vocabulary with the catalogue drag coverage below the fallback threshold,
    # so the branch is judged on the wording that fits it best.
    variants = [tokens for tokens in (_tokens(query) for query in queries) if tokens]
    if not variants:
        variants = [[]]
    matched, missing, coverage = (), (), 0.0
    for query_tokens in variants:
        unique = list(dict.fromkeys(query_tokens))
        hits = tuple(token for token in unique if token in observed)
        ratio = len(hits) / max(1, len(unique))
        if ratio >= coverage:
            matched = hits
            missing = tuple(token for token in unique if token not in observed)
            coverage = ratio
    has_reference = bool(_REFERENCE_SIGNAL.search(combined))
    has_structural = any(
        item.chunk.kind in {"table", "selection", "ordering", "variant", "technical"}
        for item in candidates
    )
    exact_signal = max((item.exact_score + item.code_score for item in candidates), default=0.0)
    score = max(
        0.0,
        min(1.0, 0.62 * coverage + 0.18 * bool(has_reference) + 0.12 * bool(has_structural) + 0.08 * min(1.0, exact_signal)),
    )
    sufficient = bool(candidates) and coverage >= 0.52 and (has_reference or has_structural) and score >= 0.55
    return LocalEvidenceQuality(
        coverage=coverage,
        score=score,
        sufficient=sufficient,
        has_reference_signal=has_reference,
        has_structural_signal=has_structural,
        matched_tokens=matched,
        missing_tokens=missing,
    )
