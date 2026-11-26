"""Canonical DOI handling used by all manifest inputs."""

import re
import unicodedata
from urllib.parse import unquote


class DOIError(ValueError):
    """Raised when a value is not a usable DOI."""


_PREFIX = re.compile(
    r"^(?:doi\s*:\s*|https?://(?:(?:dx\.)?doi\.org)/)",
    re.IGNORECASE,
)
_DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_PAIRS = {")": "(", "]": "[", "}": "{"}


def normalize_doi(value):
    """Return a stable lowercase DOI, rejecting malformed identifiers.

    URL/prefix forms, percent escapes, surrounding citation punctuation, and
    Unicode compatibility variants are normalized. Punctuation that is part of
    a balanced DOI suffix is retained.
    """
    if not isinstance(value, str):
        raise DOIError("DOI must be a string")
    doi = unicodedata.normalize("NFKC", unquote(value)).strip()
    doi = _PREFIX.sub("", doi, count=1).strip()
    doi = doi.strip("\"'<>")
    while doi and doi[-1] in ".,;:":
        doi = doi[:-1]
    while doi and doi[-1] in _PAIRS:
        close = doi[-1]
        if doi.count(close) > doi.count(_PAIRS[close]):
            doi = doi[:-1]
        else:
            break
    doi = doi.strip().lower()
    if not doi or any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in doi):
        raise DOIError("DOI is empty or contains whitespace/control characters")
    if not _DOI.fullmatch(doi):
        raise DOIError("DOI must match 10.<registrant>/<suffix>")
    return doi
