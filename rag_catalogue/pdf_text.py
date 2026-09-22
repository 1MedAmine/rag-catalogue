"""PDF text extraction and page-aware catalogue chunking."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from pathlib import Path
import re
from typing import TYPE_CHECKING, Iterable

import fitz

from .encoding_repair import detect_delta, repair_text

if TYPE_CHECKING:
    from .catalogue_map import CatalogueMap


@dataclass(frozen=True, slots=True)
class PdfPage:
    page_number: int
    text: str
    structured_blocks: tuple[str, ...] = ()
    heading: str = field(default="", compare=False)
    heading_confidence: float = field(default=0.0, compare=False)
    text_quality: float = field(default=1.0, compare=False)
    ocr_used: bool = field(default=False, compare=False)


@dataclass(frozen=True, slots=True)
class PdfInspection:
    page_count: int
    native_text_pages: tuple[int, ...]
    low_text_pages: tuple[int, ...]
    image_pages: tuple[int, ...]
    ocr_candidate_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CatalogueChunk:
    chunk_id: str
    page_number: int
    text: str
    section_id: str = field(default="", compare=False)
    section_title: str = field(default="", compare=False)
    kind: str = field(default="text", compare=False)
    markers: tuple[str, ...] = field(default=(), compare=False)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def inspect_pdf(
    path: Path | str,
    *,
    min_native_chars: int = 40,
) -> PdfInspection:
    """Inspect native extraction coverage and locate OCR fallback candidates."""

    if min_native_chars < 0:
        raise ValueError("min_native_chars doit etre positif ou nul")
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")
    native_text_pages: list[int] = []
    low_text_pages: list[int] = []
    image_pages: list[int] = []
    ocr_candidate_pages: list[int] = []
    try:
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                native = _normalise_whitespace(page.get_text("text", sort=True))
                has_images = bool(page.get_images(full=True))
                if len(native) >= min_native_chars:
                    native_text_pages.append(index)
                else:
                    low_text_pages.append(index)
                if has_images:
                    image_pages.append(index)
                if len(native) < min_native_chars and has_images:
                    ocr_candidate_pages.append(index)
            page_count = document.page_count
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise ValueError(f"PDF illisible: {pdf_path}") from exc
    return PdfInspection(
        page_count=page_count,
        native_text_pages=tuple(native_text_pages),
        low_text_pages=tuple(low_text_pages),
        image_pages=tuple(image_pages),
        ocr_candidate_pages=tuple(ocr_candidate_pages),
    )




def _page_layout(page: fitz.Page) -> dict:
    try:
        layout = page.get_text("dict", sort=True)
    except (RuntimeError, ValueError, TypeError):
        return {}
    return layout if isinstance(layout, dict) else {}


def _iter_spans(layout: dict) -> Iterable[dict]:
    for block in layout.get("blocks", []) or []:
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            for span in line.get("spans", []) or []:
                if isinstance(span, dict):
                    yield span


def _page_font_deltas(layout: dict) -> list[int]:
    """Offsets that restore whole font groups on this page, if any."""

    per_font: dict[str, list[str]] = {}
    for span in _iter_spans(layout):
        text = str(span.get("text") or "")
        if text.strip():
            per_font.setdefault(str(span.get("font") or ""), []).append(text)
    found: list[int] = []
    for chunks in per_font.values():
        delta = detect_delta(" ".join(chunks))
        if delta:
            found.append(delta)
    return found


@lru_cache(maxsize=32)
def _document_deltas(
    path_value: str,
    file_size: int,
    modified_ns: int,
) -> tuple[int, ...]:
    """Learn the catalogue's encoding offsets from a sample of its pages.

    A single font can mix sound and offset spans on one page, so offsets are
    learnt document-wide where whole groups give a clear signal, then applied
    span by span. Sampling keeps this affordable on very large catalogues.
    """

    del file_size, modified_ns  # only used to invalidate the cache key
    counts: dict[int, int] = {}
    try:
        with fitz.open(Path(path_value)) as document:
            total = document.page_count
            step = max(1, total // 120)
            for index in range(0, total, step):
                for delta in _page_font_deltas(_page_layout(document.load_page(index))):
                    counts[delta] = counts.get(delta, 0) + 1
    except (fitz.FileDataError, RuntimeError, ValueError):
        return ()
    return tuple(sorted((d for d, n in counts.items() if n >= 2), key=lambda d: -counts[d]))


def _page_text(page: fitz.Page, layout: dict, deltas: tuple[int, ...]) -> str:
    """Native page text, rebuilt span by span only when the catalogue needs it."""

    if not deltas:
        return page.get_text("text", sort=True)
    lines: list[str] = []
    for block in layout.get("blocks", []) or []:
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            parts = [
                repair_text(str(span.get("text") or ""), deltas)
                for span in line.get("spans", []) or []
                if isinstance(span, dict)
            ]
            rendered = "".join(parts)
            if rendered.strip():
                lines.append(rendered)
        lines.append("")
    return "\n".join(lines)


def _text_quality(text: str) -> float:
    """Estimate whether native extraction contains usable catalogue text."""

    value = str(text or "")
    if not value.strip():
        return 0.0
    visible = [char for char in value if not char.isspace()]
    if not visible:
        return 0.0
    alnum_ratio = sum(char.isalnum() for char in visible) / len(visible)
    word_count = len(re.findall(r"\b\w+\b", value, flags=re.UNICODE))
    length_score = min(1.0, word_count / 24.0)
    return max(0.0, min(1.0, 0.65 * alnum_ratio + 0.35 * length_score))


def _layout_heading(
    page: fitz.Page,
    fallback_text: str,
    layout: dict,
    deltas: tuple[int, ...],
) -> tuple[str, float]:
    """Pick a probable page heading from font size, weight and vertical position."""

    candidates: list[tuple[float, str]] = []
    page_height = max(float(page.rect.height), 1.0)
    for block in layout.get("blocks", []) if isinstance(layout, dict) else []:
        if not isinstance(block, dict) or block.get("type") != 0:
            continue
        for line in block.get("lines", []) or []:
            if not isinstance(line, dict):
                continue
            spans = line.get("spans", []) or []
            line_text = _normalise_whitespace(" ".join(
                repair_text(str(span.get("text") or ""), deltas)
                for span in spans
                if isinstance(span, dict)
            ))
            if not line_text or len(line_text) > 180 or len(line_text.split()) > 22:
                continue
            sizes = [float(span.get("size") or 0.0) for span in spans if isinstance(span, dict)]
            size = max(sizes, default=0.0)
            y0 = min(
                (float((span.get("bbox") or [0, page_height])[1]) for span in spans if isinstance(span, dict)),
                default=page_height,
            )
            bold = any(
                "bold" in str(span.get("font") or "").casefold()
                or int(span.get("flags") or 0) & 16
                for span in spans
                if isinstance(span, dict)
            )
            top_bonus = max(0.0, 1.0 - y0 / (page_height * 0.45))
            score = size / 18.0 + (0.3 if bold else 0.0) + 0.45 * top_bonus
            score -= max(0, len(line_text) - 80) / 240.0
            candidates.append((score, line_text))
    if candidates:
        score, heading = max(candidates, key=lambda item: (item[0], -len(item[1])))
        confidence = max(0.0, min(1.0, score / 1.7))
        return heading, confidence
    first_line = _normalise_whitespace(str(fallback_text or "").split("\n", 1)[0])
    if first_line and len(first_line) <= 140:
        return first_line, 0.25
    return "", 0.0


def _infer_chunk_kind(page: PdfPage) -> str:
    text = f"{page.heading} {page.text[:1200]}"
    if re.search(r"\b(?:ordering|order)\s+(?:information|guidelines?|code)|\bhow\s+to\s+order\b|\bcodification\b|information de commande", text, re.IGNORECASE):
        return "ordering"
    if re.search(r"\bselection\s+(?:table|chart|guide)\b|tableau de selection", text, re.IGNORECASE):
        return "selection"
    if page.structured_blocks:
        return "table"
    if re.search(r"\btechnical\s+(?:data|specifications?)\b|\bdimensions?\b|caracteristiques techniques", text, re.IGNORECASE):
        return "technical"
    return "text"


def _clean_cell(value: object, deltas: tuple[int, ...] = ()) -> str:
    if value is None:
        return ""
    return _normalise_whitespace(repair_text(str(value), deltas))


def _meaningful_rows(
    rows: Iterable[list[object]],
    deltas: tuple[int, ...] = (),
) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [_clean_cell(cell, deltas) for cell in row]
        if any(cells):
            cleaned.append(cells)
    return cleaned


def _looks_like_row_labels(values: list[str]) -> bool:
    nonempty = [value for value in values if value]
    if len(nonempty) < 3:
        return False
    label_like = 0
    code_like = 0
    for value in nonempty:
        letters = sum(char.isalpha() for char in value)
        digits = sum(char.isdigit() for char in value)
        words = value.split()
        if letters and (len(words) >= 2 or digits == 0 or len(value) > 12):
            label_like += 1
        if letters and digits and len(words) <= 2 and len(value) <= 16:
            code_like += 1
    return label_like >= max(2, int(len(nonempty) * 0.6)) and label_like > code_like




def _looks_like_identifier_cell(value: str) -> bool:
    compact = value.replace(" ", "")
    if not compact or len(value) > 80:
        return False
    has_letter = any(char.isalpha() for char in value)
    has_digit = any(char.isdigit() for char in value)
    return has_letter and has_digit and len(value.split()) <= 8


def _is_continuation_table(rows: list[list[str]], previous_labels: list[str] | None) -> bool:
    if previous_labels is None or len(previous_labels) != len(rows) or len(rows) < 3:
        return False
    if not rows or len(rows[0]) < 1:
        return False
    previous_first = previous_labels[0].casefold().strip() if previous_labels else ""
    current_first = rows[0][0].casefold().strip()
    if not current_first or current_first == previous_first:
        return False
    first_row_values = [cell for cell in rows[0] if cell]
    if not first_row_values:
        return False
    identifier_count = sum(_looks_like_identifier_cell(cell) for cell in first_row_values)
    return identifier_count >= max(1, (len(first_row_values) + 1) // 2)




_RECORD_HEADER_RE = re.compile(
    r"\b(?:model|variant|reference|product|type|item|designation|article)\b",
    flags=re.IGNORECASE,
)


def _is_variant_schema_table(rows: list[list[str]]) -> bool:
    """Identify a product-variant table, not a simple two-column code map."""

    if len(rows) < 4:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    first_column = [row[0] for row in rows]
    if not _looks_like_row_labels(first_column):
        return False
    if width >= 3:
        return True
    first_label = first_column[0] if first_column else ""
    first_value = rows[0][1] if len(rows[0]) > 1 else ""
    return bool(_RECORD_HEADER_RE.search(first_label)) and bool(first_value)


def _record_block(labels: list[str], values: list[str]) -> str:
    lines = [
        f"{label}: {value}"
        for label, value in zip(labels, values, strict=False)
        if label and value
    ]
    return "\n".join(lines)


def _mapping_block(rows: list[list[str]]) -> str:
    lines: list[str] = []
    for row in rows:
        nonempty = [cell for cell in row if cell]
        if len(nonempty) < 2:
            continue
        left, right = nonempty[0], nonempty[1]
        # Ignore broken layout artefacts such as one-letter fragments.
        if len(left.replace(" ", "")) < 1 or len(right.replace(" ", "")) < 2:
            continue
        lines.append(f"{left}: {right}")
    return "\n".join(lines)


def _extract_table_blocks(
    page: fitz.Page,
    previous_row_labels: list[str] | None,
    cell_deltas: tuple[int, ...] = (),
) -> tuple[list[str], list[str] | None]:
    """Serialize table columns independently so values cannot leak across variants."""

    try:
        # PyMuPDF currently prints a layout-package suggestion directly to the
        # process streams on first table analysis. A CLI that emits JSON must
        # keep stdout/stderr machine-readable, so contain this library notice.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            finder = page.find_tables()
        tables = list(getattr(finder, "tables", []) or [])
    except (AttributeError, RuntimeError, ValueError):
        return [], None

    blocks: list[str] = []
    continuation_labels = previous_row_labels
    labels_for_next_page: list[str] | None = None
    for table in tables:
        try:
            rows = _meaningful_rows(table.extract(), cell_deltas)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            continue
        if not rows:
            continue
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        if width < 2:
            continue

        if _is_continuation_table(rows, continuation_labels):
            assert continuation_labels is not None
            for column in range(width):
                block = _record_block(
                    continuation_labels, [row[column] for row in rows]
                )
                if block:
                    blocks.append(block)
            # A carried schema applies only to the immediately following table/page.
            continuation_labels = None
            continue

        if _is_variant_schema_table(rows):
            labels = [row[0] for row in rows]
            labels_for_next_page = labels
            for column in range(1, width):
                block = _record_block(labels, [row[column] for row in rows])
                if block:
                    blocks.append(block)
            continue

        block = _mapping_block(rows)
        if block:
            blocks.append(block)

    # Preserve order while removing duplicates produced by overlapping table detections.
    unique_blocks = list(dict.fromkeys(blocks))
    return unique_blocks, labels_for_next_page


def _blocks_for_page_text(blocks: list[str]) -> str:
    records: list[str] = []
    for block in blocks:
        records.append(block.replace(": ", " = "))
    return "\n\n".join(records)


@lru_cache(maxsize=32)
def _extract_raw_pdf_pages_cached(
    path_value: str,
    file_size: int,
    modified_ns: int,
) -> tuple[tuple[int, str, str, float, float], ...]:
    """Cache native page text plus lightweight layout metadata."""

    pdf_path = Path(path_value)
    pages: list[tuple[int, str, str, float, float]] = []
    deltas = _document_deltas(path_value, file_size, modified_ns)
    try:
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                layout = _page_layout(page)
                raw_multiline = _page_text(page, layout, deltas)
                raw_text = _normalise_whitespace(raw_multiline)
                heading, heading_confidence = _layout_heading(
                    page, raw_multiline, layout, deltas
                )
                pages.append((
                    index,
                    raw_text,
                    heading,
                    heading_confidence,
                    _text_quality(raw_text),
                ))
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise ValueError(f"PDF illisible: {pdf_path}") from exc
    return tuple(pages)


def _ocr_page_text(
    page: fitz.Page,
    *,
    language: str,
    dpi: int = 150,
) -> str:
    """Run PyMuPDF's Tesseract bridge for one page, when available."""

    try:
        text_page = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
        return _normalise_whitespace(page.get_text("text", textpage=text_page, sort=True))
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return ""


def _extract_pages_with_optional_ocr(
    path_value: str,
    native_pages: tuple[tuple[int, str, str, float, float], ...],
    *,
    ocr_mode: str,
    ocr_language: str,
    ocr_max_pages: int,
    min_native_chars: int,
) -> tuple[tuple[int, str, str, float, float, bool], ...]:
    if ocr_mode not in {"off", "auto", "force"}:
        raise ValueError("ocr_mode doit etre off, auto ou force")
    if ocr_max_pages < 0:
        raise ValueError("ocr_max_pages doit etre positif ou nul")
    if min_native_chars < 0:
        raise ValueError("min_native_chars doit etre positif ou nul")
    if ocr_mode == "off" or ocr_max_pages == 0:
        return tuple((*item, False) for item in native_pages)

    by_page = {item[0]: item for item in native_pages}
    attempts = 0
    output: list[tuple[int, str, str, float, float, bool]] = []
    try:
        with fitz.open(path_value) as document:
            for page_number in range(1, document.page_count + 1):
                native = by_page[page_number]
                _number, raw_text, heading, confidence, quality = native
                page = document.load_page(page_number - 1)
                has_images = bool(page.get_images(full=True))
                should_ocr = (
                    ocr_mode == "force"
                    or (ocr_mode == "auto" and len(raw_text) < min_native_chars and has_images)
                )
                used = False
                if should_ocr and attempts < ocr_max_pages:
                    attempts += 1
                    ocr_text = _ocr_page_text(page, language=ocr_language)
                    if len(ocr_text) > len(raw_text):
                        raw_text = ocr_text
                        quality = _text_quality(raw_text)
                        used = True
                output.append((page_number, raw_text, heading, confidence, quality, used))
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise ValueError(f"PDF illisible: {path_value}") from exc
    return tuple(output)


@lru_cache(maxsize=512)
def _extract_table_page_cached(
    path_value: str,
    file_size: int,
    modified_ns: int,
    page_number: int,
    previous_labels: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Cache one page's expensive table detection across overlapping requests."""

    pdf_path = Path(path_value)
    deltas = _document_deltas(path_value, file_size, modified_ns)
    try:
        with fitz.open(pdf_path) as document:
            if page_number <= 0 or page_number > document.page_count:
                return (), ()
            page = document.load_page(page_number - 1)
            blocks, labels = _extract_table_blocks(
                page,
                list(previous_labels) if previous_labels else None,
                deltas,
            )
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise ValueError(f"PDF illisible: {pdf_path}") from exc
    return tuple(blocks), tuple(labels or ())


def extract_pdf_pages(
    path: Path | str,
    *,
    table_pages: set[int] | None = None,
    ocr_mode: str = "off",
    ocr_language: str = "eng",
    ocr_max_pages: int = 12,
    min_native_chars: int = 40,
) -> list[PdfPage]:
    """Extract pages and preserve tables only for explicitly requested pages."""

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")
    stat = pdf_path.stat()
    path_value = str(pdf_path.resolve())
    native_pages = _extract_raw_pdf_pages_cached(
        path_value,
        stat.st_size,
        stat.st_mtime_ns,
    )
    raw_pages = _extract_pages_with_optional_ocr(
        path_value,
        native_pages,
        ocr_mode=ocr_mode,
        ocr_language=ocr_language,
        ocr_max_pages=ocr_max_pages,
        min_native_chars=min_native_chars,
    )
    if not raw_pages:
        raise ValueError(f"Aucun texte extractible dans le PDF: {pdf_path}")

    available_page_numbers = {page_number for page_number, _text, _heading, _confidence, _quality, _ocr_used in raw_pages}
    requested = (
        available_page_numbers
        if table_pages is None
        else set(table_pages) & available_page_numbers
    )

    pages: list[PdfPage] = []
    previous_labels: tuple[str, ...] = ()
    previous_table_page: int | None = None
    for page_number, raw_text, heading, heading_confidence, text_quality, ocr_used in raw_pages:
        blocks: tuple[str, ...] = ()
        if page_number in requested:
            carried_labels = (
                previous_labels
                if previous_table_page is not None
                and page_number == previous_table_page + 1
                else ()
            )
            blocks, previous_labels = _extract_table_page_cached(
                path_value,
                stat.st_size,
                stat.st_mtime_ns,
                page_number,
                carried_labels,
            )
            previous_table_page = page_number
        else:
            previous_labels = ()
            previous_table_page = None

        table_text = _blocks_for_page_text(list(blocks))
        text = raw_text
        if table_text:
            text = _normalise_whitespace(
                f"{raw_text} STRUCTURED TABLE RECORDS {table_text}"
            )
        pages.append(
            PdfPage(
                page_number=page_number,
                text=text,
                structured_blocks=blocks,
                heading=heading,
                heading_confidence=heading_confidence,
                text_quality=text_quality,
                ocr_used=ocr_used,
            )
        )
    if not any(page.text.strip() for page in pages):
        raise ValueError(f"Aucun texte extractible dans le PDF: {pdf_path}")
    return pages


def chunk_pages(
    pages: list[PdfPage],
    *,
    max_words: int = 260,
    overlap_words: int = 50,
    section_by_page: dict[int, tuple[str, str, str, tuple[str, ...]]] | None = None,
    section_map: "CatalogueMap | None" = None,
) -> list[CatalogueChunk]:
    """Split each page independently while retaining page provenance."""

    if max_words <= 0:
        raise ValueError("max_words doit etre strictement positif")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words doit etre compris entre 0 et max_words - 1")

    chunks: list[CatalogueChunk] = []
    step = max_words - overlap_words
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        part = 1
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            section_id = ""
            section_title = ""
            kind = _infer_chunk_kind(page)
            markers: tuple[str, ...] = ()
            if section_by_page and page.page_number in section_by_page:
                section_id, section_title, section_kind, markers = section_by_page[page.page_number]
                # The page-local structure is more precise than a section-wide
                # label. A section can contain a selection table followed by an
                # ordering page; never erase that ordering signal.
                if (
                    kind == "text"
                    and section_kind in {"ordering", "selection", "table", "technical", "variant"}
                ):
                    kind = section_kind
            elif section_map is not None:
                section = section_map.section_for_page(page.page_number)
                section_id = section.section_id
                section_title = section.title
            chunks.append(
                CatalogueChunk(
                    chunk_id=f"p{page.page_number}-c{part}",
                    page_number=page.page_number,
                    text=" ".join(words[start:end]),
                    section_id=section_id,
                    section_title=section_title,
                    kind=kind,
                    markers=markers,
                )
            )
            if end == len(words):
                break
            start += step
            part += 1
    return chunks
