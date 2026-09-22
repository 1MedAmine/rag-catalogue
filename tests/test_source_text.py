from pathlib import Path

import pytest

from rag_catalogue.source_text import extract_product_text


def test_extract_product_text_reads_utf8_sig_txt(tmp_path: Path) -> None:
    source = tmp_path / "produit.txt"
    source.write_text("\ufeffDisjoncteur 2P 6 A courbe C 10 kA", encoding="utf-8")

    result = extract_product_text(source)

    assert result == "[FICHIER TEXTE produit.txt]\nDisjoncteur 2P 6 A courbe C 10 kA"


def test_extract_product_text_rejects_empty_txt(tmp_path: Path) -> None:
    source = tmp_path / "vide.txt"
    source.write_text("  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="vide"):
        extract_product_text(source)
