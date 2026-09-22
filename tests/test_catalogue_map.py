from pathlib import Path

import fitz

from rag_catalogue.catalogue_map import build_catalogue_map
from rag_catalogue.pdf_text import PdfPage, chunk_pages


def test_catalogue_map_uses_pdf_toc_and_assigns_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "toc.pdf"
    doc = fitz.open()
    for index in range(6):
        page = doc.new_page()
        page.insert_text((72, 72), f"Contenu page {index + 1}")
    doc.set_toc([[1, "Pompes", 1], [2, "Pompes centrifuges", 2], [1, "Moteurs", 4]])
    doc.save(pdf)
    doc.close()

    catalogue_map = build_catalogue_map(pdf)

    assert catalogue_map.section_for_page(1).title == "Pompes"
    assert catalogue_map.section_for_page(2).title == "Pompes centrifuges"
    assert catalogue_map.section_for_page(3).title == "Pompes centrifuges"
    assert catalogue_map.section_for_page(4).title == "Moteurs"
    assert catalogue_map.section_for_page(6).title == "Moteurs"


def test_catalogue_map_falls_back_to_large_font_headings(tmp_path: Path) -> None:
    pdf = tmp_path / "headings.pdf"
    doc = fitz.open()
    first = doc.new_page()
    first.insert_text((72, 72), "CAPTEURS INDUSTRIELS", fontsize=18)
    first.insert_text((72, 110), "Specifications generales", fontsize=9)
    second = doc.new_page()
    second.insert_text((72, 72), "Table de selection", fontsize=9)
    third = doc.new_page()
    third.insert_text((72, 72), "POMPES CENTRIFUGES", fontsize=18)
    third.insert_text((72, 110), "Courbes hydrauliques", fontsize=9)
    doc.save(pdf)
    doc.close()

    catalogue_map = build_catalogue_map(pdf)

    assert catalogue_map.section_for_page(1).title == "CAPTEURS INDUSTRIELS"
    assert catalogue_map.section_for_page(2).title == "CAPTEURS INDUSTRIELS"
    assert catalogue_map.section_for_page(3).title == "POMPES CENTRIFUGES"


def test_chunk_pages_attaches_section_metadata(tmp_path: Path) -> None:
    pdf = tmp_path / "sections.pdf"
    doc = fitz.open()
    for index in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {index + 1}")
    doc.set_toc([[1, "Roulements", 1]])
    doc.save(pdf)
    doc.close()
    catalogue_map = build_catalogue_map(pdf)

    chunks = chunk_pages(
        [PdfPage(page_number=1, text="roulement radial acier")],
        section_map=catalogue_map,
    )

    assert chunks[0].section_id == catalogue_map.section_for_page(1).section_id
    assert chunks[0].section_title == "Roulements"
