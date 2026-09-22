from pathlib import Path

import fitz

from rag_catalogue.pdf_text import PdfInspection, inspect_pdf


def test_inspect_pdf_identifies_low_text_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "inspect.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Une page avec suffisamment de texte technique pour extraction native.")
    doc.new_page()
    doc.save(pdf)
    doc.close()

    inspection = inspect_pdf(pdf, min_native_chars=20)

    assert isinstance(inspection, PdfInspection)
    assert inspection.page_count == 2
    assert inspection.native_text_pages == (1,)
    assert inspection.low_text_pages == (2,)
    assert inspection.ocr_candidate_pages == ()


def test_inspect_pdf_marks_image_only_pages_as_ocr_candidates(tmp_path: Path) -> None:
    pdf = tmp_path / "image.pdf"
    image_pdf = fitz.open()
    image_page = image_pdf.new_page(width=200, height=200)
    pix = image_page.get_pixmap()
    png = pix.tobytes("png")
    image_pdf.close()

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_image(page.rect, stream=png)
    doc.save(pdf)
    doc.close()

    inspection = inspect_pdf(pdf, min_native_chars=20)

    assert inspection.low_text_pages == (1,)
    assert inspection.image_pages == (1,)
    assert inspection.ocr_candidate_pages == (1,)
