"""Deterministic PDF/OCR text normalization for exact quotation checks."""

import re
import unicodedata


def normalized_text(value):
    text = unicodedata.normalize("NFKD", str(value or "")).lower()
    for old, new in {
        "μ": "u",
        "µ": "u",
        "±": "+/-",
        "−": "-",
        "∆": "delta",
        "δ": "delta",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
    }.items():
        text = text.replace(old, new)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def normalized_quote_text(value):
    """Normalize deterministic PDF/OCR artifacts without fuzzy substitutions."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    text = "".join(
        " " if char in "\r\n\t" else char
        for char in text
        if char in "\r\n\t" or unicodedata.category(char) != "Cc"
    )
    text = normalized_text(text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)

    def join_ocr_letters(match):
        letters = re.sub(r"\s+", "", match.group())
        if letters[0] in {"a", "i"} and len(letters) >= 5:
            return letters[0] + " " + letters[1:]
        return letters

    text = re.sub(
        r"(?<![a-z])(?:[a-z]\s+){2,}[a-z](?![a-z])",
        join_ocr_letters,
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


__all__ = ["normalized_text", "normalized_quote_text"]
