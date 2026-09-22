"""Offset-encoding repair for catalogues whose font subsets lack a ToUnicode map."""

from rag_catalogue.encoding_repair import detect_delta, french_score, repair_text, shift_text


def test_detects_the_offset_of_a_shifted_french_span() -> None:
    garbled = (
        '+DR 2(1". RNMS CDR HMSDQQTOSDTQR RDBSHNMMDTQR '
        'LTKSHONK@HQDR L@MTDKR ONTQ K@ CHRSQHATSHNM'
    )

    assert detect_delta(garbled) == 33


def test_repairs_a_shifted_span_into_readable_french() -> None:
    garbled = "(MSDQQTOSDTQR RDBSHNMMDTQR ONTQ K@ CHRSQHATSHNM C ġMDQFHD"

    repaired = repair_text(garbled, (33, 29))

    assert "interrupteurs" in repaired.lower()
    assert "sectionneurs" in repaired.lower()
    assert "distribution" in repaired.lower()


def test_leaves_sound_french_untouched() -> None:
    sound = "Interrupteurs-sectionneurs pour la distribution d'energie de 125 a 5000 A"

    assert detect_delta(sound) == 0
    assert repair_text(sound, (33, 29, -32)) == sound


def test_leaves_reference_codes_untouched() -> None:
    codes = "3032 4025 2698 4020 CT 43"

    assert repair_text(codes, (33, 29, -32)) == codes


def test_shift_preserves_spacing_and_restores_control_code_gaps() -> None:
    # Offset fonts encode their word gap as a control code rather than U+0020.
    assert shift_text("ONTQ\x03ETRHAKDR", 33) == "pour fusibles"
    assert shift_text("ONTQ ETRHAKDR", 33) == "pour fusibles"


def test_french_score_separates_prose_from_a_shifted_stream() -> None:
    prose = "les interrupteurs sectionneurs pour la distribution"
    shifted = shift_text(prose, -33)

    assert french_score(prose) > french_score(shifted) + 0.2
