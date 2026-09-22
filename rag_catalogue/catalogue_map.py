"""Hierarchical map of a catalogue PDF.

The map is domain-neutral. It prefers the PDF outline when available, then
falls back to strong layout headings and finally to bounded page windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from typing import TYPE_CHECKING, Iterable

import fitz

if TYPE_CHECKING:
    from .pdf_text import PdfPage


_ORDERING = re.compile(
    r"\b(?:ordering|order)\s+(?:information|guidelines?|code)|"
    r"\bhow\s+to\s+order\b|\bcatalog(?:ue)?\s+(?:number|code)\b|"
    r"\bcodification\b|\binformations?\s+de\s+commande\b|"
    r"\breference\s+de\s+commande\b",
    re.IGNORECASE,
)
_SELECTION = re.compile(
    r"\bselection\s+(?:table|chart|guide)\b|"
    r"\btableau\s+de\s+selection\b|\bguide\s+de\s+selection\b",
    re.IGNORECASE,
)
_TECHNICAL = re.compile(
    r"\btechnical\s+(?:data|specifications?)\b|"
    r"\bproduct\s+(?:data|performance|specifications?)\b|"
    r"\bdimensions?\b|\bspecifications?\b|"
    r"\bcaracteristiques?\s+techniques?\b|\bdonnees?\s+techniques?\b",
    re.IGNORECASE,
)
_GENERIC_HEADINGS = {
    "catalogue", "catalog", "product catalogue", "product catalog",
    "contents", "table of contents", "sommaire", "index",
}
_MARKER_STOPWORDS = {
    "AC", "DC", "IEC", "EN", "ISO", "DIN", "IP", "MODEL", "TYPE",
    "FRAME", "STANDARD", "SELECTION", "ORDERING", "INFORMATION",
    "GUIDELINES", "REFERENCE", "CURRENT", "VOLTAGE", "TECHNICAL",
}


@dataclass(frozen=True, slots=True)
class CatalogueSection:
    section_id: str
    title: str
    start_page: int
    end_page: int
    page_numbers: tuple[int, ...]
    kind: str
    markers: tuple[str, ...]
    summary: str
    level: int = 1


@dataclass(frozen=True, slots=True)
class CatalogueMap:
    path: Path
    page_count: int
    sections: tuple[CatalogueSection, ...]
    source: str

    def section_for_page(self, page_number: int) -> CatalogueSection:
        if page_number <= 0 or page_number > self.page_count:
            raise ValueError("page hors du catalogue")
        for section in self.sections:
            if section.start_page <= page_number <= section.end_page:
                return section
        # Defensive fallback; a valid map should cover every page.
        return CatalogueSection(
            section_id="s000",
            title=f"Page {page_number}",
            start_page=page_number,
            end_page=page_number,
            page_numbers=(page_number,),
            kind="text",
            markers=(),
            summary=f"Page {page_number}",
        )

    def pages_for_sections(self, section_ids: Iterable[str]) -> set[int]:
        wanted = set(section_ids)
        return {
            page
            for section in self.sections
            if section.section_id in wanted
            for page in section.page_numbers
        }


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def page_kind(page: "PdfPage") -> str:
    combined = f"{page.heading}\n{page.text[:1800]}"
    if _ORDERING.search(combined):
        return "ordering"
    if _SELECTION.search(combined):
        return "selection"
    if page.structured_blocks:
        return "table"
    if _TECHNICAL.search(combined):
        return "technical"
    return "text"


def structural_markers(text: str) -> tuple[str, ...]:
    markers: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{2,}(?![A-Za-z0-9])", text):
        token = raw.strip("_-")
        if (
            len(token) < 3
            or token in _MARKER_STOPWORDS
            or not any(char.isalpha() for char in token)
        ):
            continue
        candidates = [token]
        prefix = re.match(r"[A-Z]{3,}", token)
        if prefix:
            candidates.append(prefix.group(0))
        for candidate in candidates:
            if candidate not in _MARKER_STOPWORDS and candidate not in seen:
                seen.add(candidate)
                markers.append(candidate)
    return tuple(markers[:24])


def _usable_heading(page: "PdfPage") -> str:
    heading = " ".join(str(page.heading or "").split()).strip(" -–—:;,.|")
    folded = _fold(heading)
    if not heading or folded in _GENERIC_HEADINGS:
        return ""
    if len(heading) > 180 or len(heading.split()) > 22:
        return ""
    return heading


def _similar_heading(left: str, right: str) -> bool:
    a, b = _fold(left), _fold(right)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    left_tokens, right_tokens = set(a.split()), set(b.split())
    union = left_tokens | right_tokens
    return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.58


def _section_title(page: "PdfPage", *, fallback: str) -> str:
    heading = _usable_heading(page)
    if heading:
        return heading
    kind = page_kind(page)
    if kind == "ordering":
        return "Ordering / codification"
    if kind == "selection":
        return "Selection guide"
    words = page.text.split()
    compact = " ".join(words[:10]).strip(" -–—:;,.|")
    return compact[:120] or fallback


def _make_section(index: int, pages: list["PdfPage"], *, title: str | None = None, level: int = 1) -> CatalogueSection:
    first = pages[0]
    kinds = [page_kind(page) for page in pages]
    kind = next((value for value in kinds if value in {"ordering", "selection"}), kinds[0])
    section_title = title or _section_title(
        first,
        fallback=f"Pages {first.page_number}-{pages[-1].page_number}",
    )
    marker_values: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for marker in structural_markers(f"{page.heading}\n{page.text[:2400]}"):
            if marker not in seen:
                seen.add(marker)
                marker_values.append(marker)
    snippets = []
    for page in pages[:3]:
        text = " ".join(page.text.split())
        if text:
            snippets.append(text[:500])
    summary = (
        f"SECTION {index}: {section_title}. Pages {first.page_number}-{pages[-1].page_number}. "
        f"Kind: {kind}. Markers: {' '.join(marker_values[:12])}. "
        + " ".join(snippets)
    ).strip()
    return CatalogueSection(
        section_id=f"s{index:03d}",
        title=section_title,
        start_page=first.page_number,
        end_page=pages[-1].page_number,
        page_numbers=tuple(page.page_number for page in pages),
        kind=kind,
        markers=tuple(marker_values[:24]),
        summary=summary,
        level=level,
    )


def build_catalogue_sections(
    pages: Iterable["PdfPage"],
    *,
    max_pages: int = 12,
    heading_threshold: float = 0.55,
) -> list[CatalogueSection]:
    """Group contiguous pages using strong headings and bounded windows."""

    values = sorted(list(pages), key=lambda page: page.page_number)
    if max_pages <= 0:
        raise ValueError("max_pages doit etre strictement positif")
    if not values:
        return []

    groups: list[list[PdfPage]] = []
    current: list["PdfPage"] = []
    current_title = ""
    current_kind = "text"

    for page in values:
        heading = _usable_heading(page)
        kind = page_kind(page)
        strong_heading = bool(heading) and page.heading_confidence >= heading_threshold
        structural_boundary = kind == "ordering" and current and kind != current_kind
        heading_boundary = (
            strong_heading
            and current
            and current_title
            and not _similar_heading(heading, current_title)
        )
        size_boundary = bool(current) and len(current) >= max_pages

        if structural_boundary or heading_boundary or size_boundary:
            groups.append(current)
            current = []
            current_title = ""
            current_kind = "text"

        if not current:
            current_title = heading
            current_kind = kind
        current.append(page)

    if current:
        groups.append(current)

    return [_make_section(index, group) for index, group in enumerate(groups, start=1)]


def _sections_from_toc(path: Path, pages: list["PdfPage"]) -> list[CatalogueSection]:
    try:
        with fitz.open(path) as document:
            toc = document.get_toc(simple=True)
    except (fitz.FileDataError, RuntimeError, ValueError):
        return []
    entries: list[tuple[int, str, int]] = []
    for raw in toc:
        if not isinstance(raw, list) or len(raw) < 3:
            continue
        level, title, page = raw[:3]
        if type(level) is not int or type(page) is not int or not isinstance(title, str):
            continue
        title = " ".join(title.split()).strip()
        if title and 1 <= page <= len(pages):
            entries.append((level, title, page))
    if not entries:
        return []

    # Multiple outline entries can start on the same page. The deepest one is
    # the most specific section for that page.
    by_page: dict[int, tuple[int, str]] = {}
    for level, title, page in entries:
        previous = by_page.get(page)
        if previous is None or level >= previous[0]:
            by_page[page] = (level, title)
    starts = sorted(by_page)
    sections: list[CatalogueSection] = []
    page_by_number = {page.page_number: page for page in pages}
    for index, start_page in enumerate(starts, start=1):
        end_page = (starts[index] - 1) if index < len(starts) else len(pages)
        group = [
            page_by_number[page_number]
            for page_number in range(start_page, end_page + 1)
            if page_number in page_by_number
        ]
        if group:
            level, title = by_page[start_page]
            sections.append(_make_section(index, group, title=title, level=level))
    if sections and sections[0].start_page > 1:
        prefix = [page_by_number[p] for p in range(1, sections[0].start_page) if p in page_by_number]
        if prefix:
            sections.insert(0, _make_section(0, prefix, title="Front matter", level=1))
    # Reassign stable IDs after optional prefix insertion.
    return [
        CatalogueSection(
            section_id=f"s{index:03d}",
            title=section.title,
            start_page=section.start_page,
            end_page=section.end_page,
            page_numbers=section.page_numbers,
            kind=section.kind,
            markers=section.markers,
            summary=section.summary,
            level=section.level,
        )
        for index, section in enumerate(sections, start=1)
    ]


def build_catalogue_map_from_pages(
    path: Path | str,
    pages: Iterable["PdfPage"],
) -> CatalogueMap:
    """Build a catalogue map from already extracted pages.

    This is the OCR-safe entry point used by the V13 indexer: the page text may
    come from native extraction or from the bounded OCR fallback, while the PDF
    outline is still preferred whenever it exists.
    """

    pdf = Path(path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF introuvable: {pdf}")
    values = sorted(list(pages), key=lambda page: page.page_number)
    if not values:
        raise ValueError("aucune page disponible pour construire la carte catalogue")
    toc_sections = _sections_from_toc(pdf, values)
    sections = toc_sections or build_catalogue_sections(values)
    return CatalogueMap(
        path=pdf,
        page_count=len(values),
        sections=tuple(sections),
        source="toc" if toc_sections else (
            "ocr-layout" if any(getattr(page, "ocr_used", False) for page in values) else "layout"
        ),
    )


@lru_cache(maxsize=16)
def _build_catalogue_map_cached(path_value: str, size: int, modified_ns: int) -> CatalogueMap:
    del size, modified_ns
    path = Path(path_value)
    from .pdf_text import extract_pdf_pages
    pages = extract_pdf_pages(path, table_pages=set())
    return build_catalogue_map_from_pages(path, pages)


def build_catalogue_map(
    path: Path | str,
    *,
    pages: Iterable["PdfPage"] | None = None,
) -> CatalogueMap:
    pdf = Path(path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF introuvable: {pdf}")
    if pages is None:
        stat = pdf.stat()
        return _build_catalogue_map_cached(str(pdf.resolve()), stat.st_size, stat.st_mtime_ns)
    values = sorted(list(pages), key=lambda page: page.page_number)
    if not values:
        raise ValueError("aucune page disponible pour construire la carte catalogue")
    toc_sections = _sections_from_toc(pdf, values)
    sections = toc_sections or build_catalogue_sections(values)
    return CatalogueMap(
        path=pdf,
        page_count=max(page.page_number for page in values),
        sections=tuple(sections),
        source="toc" if toc_sections else "layout",
    )


def section_lookup(
    sections: Iterable[CatalogueSection],
) -> dict[int, tuple[str, str, str, tuple[str, ...]]]:
    lookup: dict[int, tuple[str, str, str, tuple[str, ...]]] = {}
    for section in sections:
        metadata = (section.section_id, section.title, section.kind, section.markers)
        for page_number in section.page_numbers:
            lookup[page_number] = metadata
    return lookup
