from pathlib import Path

import fitz

from rag_catalogue.pdf_text import CatalogueChunk, PdfPage, chunk_pages, extract_pdf_pages


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page un REF-DEMO-03 disjoncteur 10 kA 2P courbe C")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page deux courant nominal 16 A code 00016")
    doc.save(path)
    doc.close()


def test_extract_pdf_pages_uses_one_based_page_numbers(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf)

    pages = extract_pdf_pages(pdf)

    assert pages == [
        PdfPage(page_number=1, text="Page un REF-DEMO-03 disjoncteur 10 kA 2P courbe C"),
        PdfPage(page_number=2, text="Page deux courant nominal 16 A code 00016"),
    ]


def test_chunk_pages_preserves_page_and_overlap() -> None:
    pages = [PdfPage(page_number=7, text="un deux trois quatre cinq six sept huit")]

    chunks = chunk_pages(pages, max_words=5, overlap_words=2)

    assert chunks == [
        CatalogueChunk(chunk_id="p7-c1", page_number=7, text="un deux trois quatre cinq"),
        CatalogueChunk(chunk_id="p7-c2", page_number=7, text="quatre cinq six sept huit"),
    ]


def _draw_table(page, origin_x: float, origin_y: float, col_widths: list[float], rows: list[list[str]], row_height: float = 28) -> None:
    xs = [origin_x]
    for width in col_widths:
        xs.append(xs[-1] + width)
    ys = [origin_y + row_height * index for index in range(len(rows) + 1)]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            page.insert_textbox(
                (xs[col_index] + 3, ys[row_index] + 3, xs[col_index + 1] - 3, ys[row_index + 1] - 3),
                value,
                fontsize=7,
            )


def test_extract_pdf_pages_serializes_columns_as_independent_product_records(tmp_path: Path) -> None:
    pdf = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    _draw_table(
        page,
        40,
        60,
        [130, 180, 180],
        [
            ["Model", "AX63N, 6 kA", "AX63H, 10 kA"],
            ["No. of Poles", "1P, 2P", "1P, 2P"],
            ["Rated Current", "1, 6, 16 A", "1, 6, 16 A"],
            ["Breaking Capacity", "6 kA", "10 kA"],
        ],
    )
    doc.save(pdf)
    doc.close()

    blocks = "\n".join(extract_pdf_pages(pdf)[0].structured_blocks)

    assert "Model: AX63N, 6 kA" in blocks
    assert "Breaking Capacity: 6 kA" in blocks
    assert "Model: AX63H, 10 kA" in blocks
    assert "Breaking Capacity: 10 kA" in blocks


def test_extract_pdf_pages_reuses_previous_row_labels_for_continuation_table(tmp_path: Path) -> None:
    pdf = tmp_path / "continuation.pdf"
    doc = fitz.open()
    first = doc.new_page()
    _draw_table(
        first,
        40,
        60,
        [120, 170],
        [
            ["Model", "AX63N"],
            ["Poles", "1P, 2P"],
            ["Current", "1, 6, 16 A"],
            ["Capacity", "6 kA"],
        ],
    )
    second = doc.new_page()
    _draw_table(
        second,
        40,
        60,
        [170, 170],
        [
            ["AX63P", "AX63U"],
            ["1P, 2P", "1P, 2P"],
            ["1, 6, 16 A", "1, 6, 16 A"],
            ["10 kA", "15 kA"],
        ],
    )
    doc.save(pdf)
    doc.close()

    pages = extract_pdf_pages(pdf)
    continuation = "\n".join(pages[1].structured_blocks)

    assert "Model: AX63P" in continuation
    assert "Poles: 1P, 2P" in continuation
    assert "Capacity: 10 kA" in continuation
    assert "Model: AX63U" in continuation
    assert "Capacity: 15 kA" in continuation


def test_extract_pdf_pages_does_not_pollute_stdout_with_table_warning(
    tmp_path: Path,
    capsys,
) -> None:
    pdf = tmp_path / "quiet-table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    _draw_table(
        page,
        40,
        60,
        [130, 180],
        [
            ["Model", "AX63P"],
            ["Poles", "2P"],
            ["Current", "6 A"],
            ["Capacity", "10 kA"],
        ],
    )
    doc.save(pdf)
    doc.close()

    extract_pdf_pages(pdf)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_overlapping_table_requests_reuse_each_page_extraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf = tmp_path / "cached-tables.pdf"
    doc = fitz.open()
    for model in ("AX63P", "AX63H", "AX63U"):
        page = doc.new_page()
        _draw_table(
            page,
            40,
            60,
            [130, 180],
            [
                ["Model", model],
                ["Poles", "2P"],
                ["Current", "6 A"],
                ["Capacity", "10 kA"],
            ],
        )
    doc.save(pdf)
    doc.close()

    original_find_tables = fitz.Page.find_tables
    calls: list[int] = []

    def counted_find_tables(page, *args, **kwargs):
        calls.append(page.number + 1)
        return original_find_tables(page, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "find_tables", counted_find_tables)

    extract_pdf_pages(pdf, table_pages={1, 2})
    extract_pdf_pages(pdf, table_pages={1, 2, 3})

    assert calls.count(1) == 1
    assert calls.count(2) == 1
    assert calls.count(3) == 1
