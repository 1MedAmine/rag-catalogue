"""Repair catalogue text extracted from fonts with an offset character encoding.

Some catalogues embed font subsets whose ToUnicode CMap is missing or shifted.
PyMuPDF then returns the raw character codes, so a French page reads as
``,QWHUUXSWHXUVVHFWLRQQHXUV`` instead of ``Interrupteurs-sectionneurs``.
Every affected span uses a single constant offset, so shifting the code points
back restores the text.

Detection scores French letter-bigram frequency rather than whole words: an
offset font often drops its space glyphs, so the repaired text can read as
``desinterrupteurs`` and no dictionary lookup would fire. A span is repaired
only when the shifted variant looks markedly more French than the original.
"""

from __future__ import annotations

import re
import unicodedata

# Frequent French bigrams; enough to separate real prose from a shifted stream.
_FRENCH_BIGRAMS = frozenset({
    "es", "en", "de", "nt", "on", "re", "le", "te", "er", "an", "se", "et",
    "la", "ai", "it", "ou", "me", "is", "ur", "ne", "ci", "ns", "ie", "ue",
    "us", "ar", "ce", "ra", "el", "ti", "ri", "in", "ss", "io", "ta", "di",
    "si", "sa", "ur", "ec", "ct", "pr", "tr", "ss", "ll", "mm", "pe", "po",
    "or", "om", "im", "am", "em", "al", "il", "ol", "ul", "eu", "au", "ou",
    "ph", "ch", "qu", "gn", "rs", "st", "nc", "nd", "rt", "rd", "lo", "li",
    "ni", "no", "na", "mi", "mo", "ma", "co", "ca", "cu", "du", "da", "do",
    "fi", "fo", "fa", "ve", "vi", "va", "so", "su", "to", "tu", "pa", "pi",
})

_KEYWORDS = (
    "interrupteur", "sectionneur", "calibre", "reference", "protection",
    "commande", "coffret", "fusible", "contact", "auxiliaire", "tension",
    "poignee", "raccordement", "dimensions", "accessoires", "utilisation",
    "caracteristiques", "montage", "distribution", "alimentation",
)

_NON_LETTER = re.compile(r"[^a-z]+")

_MIN_LETTERS = 40
_MIN_TARGET_SCORE = 0.46
_MIN_GAIN = 0.17
_SAMPLE_CHARS = 1200

# Offsets seen in affected catalogues, plus their immediate neighbourhood.
_CANDIDATE_DELTAS: tuple[int, ...] = tuple(
    sorted({d for base in (29, 33, 1, 3, 13, 16, 47, 32, 30, 28, 31) for d in (base, -base)})
)


def shift_text(text: str, delta: int) -> str:
    """Shift every non-space code point by ``delta``; spaces carry no glyph."""

    if not delta:
        return text
    out: list[str] = []
    for char in text:
        if char.isspace():
            out.append(char)
            continue
        if ord(char) < 0x20:
            # Offset fonts encode their word gap as a control code, not U+0020.
            out.append(" ")
            continue
        code = ord(char) + delta
        out.append(chr(code) if 0 <= code <= 0x10FFFF else char)
    return "".join(out)


def _ascii_letters(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return _NON_LETTER.sub(" ", stripped)


def french_score(text: str) -> float:
    """Rough 0..1 likelihood that ``text`` is French prose."""

    letters = _ascii_letters(text)
    bigrams = [
        letters[i:i + 2]
        for i in range(len(letters) - 1)
        if letters[i] != " " and letters[i + 1] != " "
    ]
    if len(bigrams) < 5:
        return 0.0
    ratio = sum(1 for bigram in bigrams if bigram in _FRENCH_BIGRAMS) / len(bigrams)
    bonus = 0.12 if any(word in letters for word in _KEYWORDS) else 0.0
    # Several offsets yield the same letters in different case; prose is mostly
    # lower case, so this settles the tie towards the true offset.
    cased = [char for char in text if char.isalpha()]
    case_hint = 0.05 * (sum(1 for char in cased if char.islower()) / len(cased)) if cased else 0.0
    return min(1.0, ratio + bonus + case_hint)


def _letter_count(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def detect_delta(text: str) -> int:
    """Return the offset that restores ``text``, or 0 when it is already sound."""

    sample = text[:_SAMPLE_CHARS]
    if _letter_count(sample) < _MIN_LETTERS:
        return 0
    source_score = french_score(sample)
    best_delta = 0
    best_score = max(source_score + _MIN_GAIN, _MIN_TARGET_SCORE)
    for delta in _CANDIDATE_DELTAS:
        score = french_score(shift_text(sample, delta))
        if score >= best_score:
            best_delta, best_score = delta, score
    return best_delta


def repair_text(text: str, deltas: tuple[int, ...]) -> str:
    """Repair a standalone string using offsets already learnt for the document."""

    letters = _letter_count(text)
    if not deltas or letters < 6:
        return text
    source_score = french_score(text)
    if source_score >= _MIN_TARGET_SCORE:
        return text
    # Headings arrive one word per span, too short to judge on the usual budget,
    # so they have to clear a higher bar before being rewritten.
    if letters < 14:
        floor, gain = 0.62, 0.30
    else:
        floor, gain = 0.40, _MIN_GAIN
    best_text, best_score = text, max(source_score + gain, floor)
    for delta in deltas:
        candidate = shift_text(text, delta)
        score = french_score(candidate)
        if score >= best_score:
            best_text, best_score = candidate, score
    return best_text
