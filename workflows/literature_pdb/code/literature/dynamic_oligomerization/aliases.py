"""Deterministic, audited bounds for external identity aliases."""

import re


# RCSB can place every homologous locus tag in one source-gene string.  These
# limits keep prompt, retrieval, and identity work proportional to the entity,
# while the audit explicitly records that a returned list may be incomplete.
MIN_GENE_ALIAS_CHARS = 2
MAX_GENE_ALIAS_CHARS = 64
MAX_GENE_ALIASES_PER_ENTITY = 64
MAX_MATCH_LABEL_CHARS = 96
MAX_IDENTITY_LABELS_PER_TARGET = 128
MAX_RETRIEVAL_TERMS = 128
MAX_COMPETITOR_TERMS = 128

GENERIC_LABELS = frozenset({"protein", "enzyme", "protease"})
_GENE_SPLIT = re.compile(r"[,;/\s]+")
_GENE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_LOCUS_TAG = re.compile(r"^(?:[A-Za-z0-9]{2,10}_\d{3,}|[A-Z]{2,}\d{3,})$")


def _ordered_unique(values):
    """Deduplicate case-insensitively with stable lexical representatives."""
    selected = {}
    for value in sorted(
        (str(value).strip() for value in values if str(value).strip()),
        key=lambda item: (item.casefold(), item),
    ):
        selected.setdefault(value.casefold(), value)
    return [selected[key] for key in sorted(selected)]


def normalize_gene_aliases(
    values,
    limit=MAX_GENE_ALIASES_PER_ENTITY,
    prior_audit=None,
):
    """Split, validate, deduplicate, and cap RCSB gene aliases.

    Returns ``(aliases, audit)``.  Comma, semicolon, slash, and whitespace are
    separators.  Only 2--64 character ASCII identifier tokens are accepted.
    ``complete`` is false whenever upstream or current normalization truncated
    aliases, so consumers never mistake a capped list for the complete source.
    """
    if limit < 0:
        raise ValueError("gene alias limit must be nonnegative")
    if values is None:
        values = []
    if isinstance(values, str):
        values = [values]
    raw_values = [str(value) for value in values if str(value).strip()]
    candidates = [
        token
        for raw in raw_values
        for token in _GENE_SPLIT.split(raw.strip())
        if token
    ]
    valid = [
        token
        for token in candidates
        if MIN_GENE_ALIAS_CHARS <= len(token) <= MAX_GENE_ALIAS_CHARS
        and _GENE_TOKEN.fullmatch(token)
    ]
    unique = sorted(
        _ordered_unique(valid),
        key=lambda value: (
            bool(_LOCUS_TAG.fullmatch(value)),
            value.casefold(),
            value,
        ),
    )
    aliases = unique[:limit]
    audit = {
        "normalization": "split_comma_semicolon_slash_whitespace_ascii_identifier_v1",
        "selection_policy": "non_locus_like_identifiers_then_locus_tags_lexically",
        "input_value_count": len(raw_values),
        "candidate_token_count": len(candidates),
        "valid_alias_count": len(unique),
        "returned_alias_count": len(aliases),
        "invalid_token_count": len(candidates) - len(valid),
        "duplicate_token_count": len(valid) - len(unique),
        "truncated_alias_count": max(0, len(unique) - len(aliases)),
        "truncated": len(unique) > len(aliases),
        "complete": len(unique) <= len(aliases),
        "limit": limit,
        "min_token_chars": MIN_GENE_ALIAS_CHARS,
        "max_token_chars": MAX_GENE_ALIAS_CHARS,
    }
    if isinstance(prior_audit, dict):
        for field in (
            "input_value_count",
            "candidate_token_count",
            "valid_alias_count",
            "invalid_token_count",
            "duplicate_token_count",
            "truncated_alias_count",
        ):
            try:
                audit[field] = max(audit[field], int(prior_audit.get(field, 0)))
            except (TypeError, ValueError):
                pass
        audit["returned_alias_count"] = len(aliases)
        audit["truncated_alias_count"] = max(
            audit["truncated_alias_count"],
            max(0, audit["valid_alias_count"] - len(aliases)),
        )
        audit["truncated"] = bool(
            prior_audit.get("truncated") or audit["truncated_alias_count"]
        )
        audit["complete"] = not audit["truncated"]
    return aliases, audit


def bounded_match_labels(values, limit, max_chars=MAX_MATCH_LABEL_CHARS):
    """Return safe, informative labels and an explicit truncation audit."""
    if limit < 0:
        raise ValueError("label limit must be nonnegative")
    raw = [str(value).strip() for value in values if str(value).strip()]
    valid = []
    invalid = 0
    generic = 0
    for value in raw:
        normalized = " ".join(value.casefold().split()).strip(" \t\r\n.,;:")
        if normalized in GENERIC_LABELS:
            generic += 1
            continue
        if not (MIN_GENE_ALIAS_CHARS <= len(value) <= max_chars):
            invalid += 1
            continue
        if not re.search(r"[A-Za-z0-9]", value):
            invalid += 1
            continue
        valid.append(value)
    unique = _ordered_unique(valid)
    selected = unique[:limit]
    return selected, {
        "input_label_count": len(raw),
        "valid_label_count": len(unique),
        "returned_label_count": len(selected),
        "invalid_length_or_specificity_count": invalid,
        "generic_label_count": generic,
        "duplicate_label_count": len(valid) - len(unique),
        "truncated_label_count": max(0, len(unique) - len(selected)),
        "truncated": len(unique) > len(selected),
        "complete": len(unique) <= len(selected),
        "limit": limit,
        "max_label_chars": max_chars,
    }
