"""Lecture generique de la fiche produit depuis PDF ou texte."""

from __future__ import annotations

from pathlib import Path

from .pdf_text import extract_pdf_pages


_TEXT_SUFFIXES = {".txt", ".md", ".json"}


def extract_product_text(path: Path | str) -> str:
    """Retourne le contenu de la fiche produit avec une provenance lisible."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Fiche produit introuvable: {source_path}")

    suffix = source_path.suffix.casefold()
    if suffix == ".pdf":
        pages = extract_pdf_pages(source_path)
        rendered_pages = []
        for page in pages:
            content = page.text
            if page.structured_blocks:
                content = content + "\n\n" + "\n\n".join(page.structured_blocks)
            rendered_pages.append(f"[PAGE {page.page_number}]\n{content}")
        return "\n\n".join(rendered_pages)

    if suffix in _TEXT_SUFFIXES:
        try:
            text = source_path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Fichier texte illisible en UTF-8: {source_path}"
            ) from exc
        if not text:
            raise ValueError(f"La fiche produit texte est vide: {source_path}")
        return f"[FICHIER TEXTE {source_path.name}]\n{text}"

    raise ValueError(
        "Format de fiche non pris en charge: "
        f"{source_path.suffix or '(sans extension)'}. Utilise PDF, TXT, MD ou JSON."
    )
