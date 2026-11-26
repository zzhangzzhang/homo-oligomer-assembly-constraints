"""Deterministic offline audit of rejected contextual EvidenceRef quotes."""

import difflib
import re
import unicodedata
from collections import Counter

from .serialization import canonical_json
from .quote_normalization import normalized_quote_text

QUOTE_AUDIT_VERSION = "contextual_quote_audit_v1"
ROOT_CAUSES = (
    "harmless_typography_unicode_whitespace_hyphenation",
    "model_dropped_inserted_punctuation_or_words",
    "quote_spans_multiple_sentences_but_remains_contiguous",
    "quote_copied_from_neighboring_or_wrong_source_id",
    "column_or_ocr_extraction_artifact",
    "truncated_ellipsized_or_paraphrased_model_text",
    "page_or_section_mismatch",
    "genuinely_absent",
)
ROLE_NAMES = {
    "identity_coreference": "identity/coreference",
    "construct_identity": "construct",
    "target_attribution": "target attribution",
    "citation_scope": "citation",
}


def _iter_invalid_evidence(value, path="$"):
    if isinstance(value, dict):
        verification = value.get("verification")
        if (
            isinstance(verification, dict)
            and verification.get("verified") is False
            and "quote" in value
            and "source_id" in value
        ):
            yield path, value
        for key, child in value.items():
            if key in {
                "component_diagnostics",
                "component_validation",
                "verification_summary",
            }:
                continue
            yield from _iter_invalid_evidence(child, path + "." + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_invalid_evidence(child, "%s[%d]" % (path, index))


def _component_path(path):
    match = re.match(
        r"(\$\.(?:paper_target_identities|constructs|state_claims|event_claims|"
        r"excluded_candidates|unassigned_candidates)\[\d+\])",
        path,
    )
    return match.group(1) if match else "$"


def _component_map(result):
    mapped = {}
    for field in (
        "paper_target_identities",
        "constructs",
        "state_claims",
        "event_claims",
        "excluded_candidates",
        "unassigned_candidates",
    ):
        for index, item in enumerate(result.get(field) or []):
            mapped["$.%s[%d]" % (field, index)] = item
    return mapped


def _window_similarity(query, text, normalize=None):
    query = str(query or "")
    text = str(text or "")
    if normalize:
        query = normalize(query)
        text = normalize(text)
    if not query:
        return 0.0
    if query in text:
        return 1.0
    matcher = difflib.SequenceMatcher(None, query, text, autojunk=False)
    match = matcher.find_longest_match(0, len(query), 0, len(text))
    start = max(0, match.b - match.a)
    candidates = [text]
    for shift in (-20, -8, 0, 8, 20):
        candidate_start = max(0, start + shift)
        candidates.append(
            text[candidate_start : candidate_start + len(query) + 40]
        )
    return max(
        difflib.SequenceMatcher(None, query, candidate, autojunk=False).ratio()
        for candidate in candidates
    )


def _fast_fuzzy_score(quote, text):
    query = normalized_quote_text(quote).casefold()
    source = normalized_quote_text(text).casefold()
    if not query:
        return 0.0
    if query in source:
        return 1.0
    query_tokens = re.findall(r"\w+", query)
    source_tokens = set(re.findall(r"\w+", source))
    if not query_tokens:
        return 0.0
    overlap = sum(token in source_tokens for token in query_tokens) / len(
        query_tokens
    )
    candidates = []
    if len(source) <= max(500, len(query) * 4):
        candidates.append(source)
    anchors = sorted(
        {token for token in query_tokens if len(token) >= 4},
        key=lambda token: (-len(token), token),
    )[:5]
    for anchor in anchors:
        for match in list(re.finditer(re.escape(anchor), source))[:5]:
            start = max(0, match.start() - len(query) // 2)
            candidates.append(source[start : start + len(query) + 80])
    sequence = max(
        (
            difflib.SequenceMatcher(
                None,
                query,
                candidate,
                autojunk=False,
            ).ratio()
            for candidate in candidates
        ),
        default=0.0,
    )
    return 0.4 * overlap + 0.6 * sequence


def _aggressive_typography(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    translations = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
    }
    for old, new in translations.items():
        text = text.replace(old, new)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,.;:!?%()\[\]])\s*", r"\1", text)
    return text.casefold().strip()


def _unicode_punctuation(value):
    text = normalized_quote_text(value)
    for old, new in {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }.items():
        text = text.replace(old, new)
    return text


def _normalization_gap_features(quote, source):
    features = []
    transforms = {
        "unicode_punctuation_mapping": _unicode_punctuation,
        "inline_spaced_hyphen_join": lambda value: re.sub(
            r"(?<=\w)-\s+(?=\w)",
            "",
            normalized_quote_text(value),
        ),
        "bidirectional_punctuation_spacing": lambda value: re.sub(
            r"\s*([,.;:!?%()\[\]])\s*",
            r"\1",
            normalized_quote_text(value),
        ),
    }
    for name, transform in transforms.items():
        transformed_quote = transform(quote)
        transformed_source = transform(source)
        if transformed_quote and transformed_quote in transformed_source:
            features.append(name)
    return features


def _punctuation_free(value):
    return " ".join(
        re.findall(r"\w+", normalized_quote_text(value).casefold())
    )


def _matches(blocks, quote, normalized=False):
    matches = []
    needle = normalized_quote_text(quote) if normalized else quote
    for block in blocks:
        text = (
            normalized_quote_text(block.get("text") or "")
            if normalized
            else str(block.get("text") or "")
        )
        if needle and needle in text:
            matches.append(block)
    return matches


def _similarity_bin(score):
    if score >= 1.0:
        return "1.00"
    if score >= 0.95:
        return "0.95-0.99"
    if score >= 0.90:
        return "0.90-0.94"
    if score >= 0.80:
        return "0.80-0.89"
    if score >= 0.60:
        return "0.60-0.79"
    return "0.00-0.59"


def _otherwise_valid_claims(result, path, evidence_path):
    components = _component_map(result)
    component = components.get(path, {})
    affected = []
    direct_claim = path.startswith(("$.state_claims[", "$.event_claims["))
    if direct_claim:
        other = [
            item
            for item in component.get("component_validation", {}).get("errors", [])
            if item.get("path") != evidence_path
        ]
        if not other:
            affected.append(path)
        return affected

    if path.startswith("$.paper_target_identities["):
        group_id = str(component.get("group_id") or "")
        allowed = {"dependent_identity_invalid", "dependent_construct_invalid"}
        for field in ("state_claims", "event_claims"):
            for index, claim in enumerate(result.get(field) or []):
                selected = str(
                    claim.get("target_attribution", {}).get("selected_group_id") or ""
                )
                errors = claim.get("component_validation", {}).get("errors", [])
                if selected == group_id and all(
                    item.get("error_code") in allowed for item in errors
                ):
                    affected.append("$.%s[%d]" % (field, index))
    elif path.startswith("$.constructs["):
        construct_id = str(component.get("construct_id") or "")
        for field in ("state_claims", "event_claims"):
            for index, claim in enumerate(result.get(field) or []):
                errors = claim.get("component_validation", {}).get("errors", [])
                if claim.get("construct_id") == construct_id and all(
                    item.get("error_code") == "dependent_construct_invalid"
                    for item in errors
                ):
                    affected.append("$.%s[%d]" % (field, index))
    return affected


def _classify(reference, claimed, blocks, exact_elsewhere, normalized_elsewhere):
    verification = reference["verification"]
    code = verification.get("error_code")
    quote = str(reference.get("quote") or "")
    claimed_text = str((claimed or {}).get("text") or "")
    if code == "evidence_provenance_mismatch":
        return "page_or_section_mismatch"
    if exact_elsewhere or normalized_elsewhere:
        return "quote_copied_from_neighboring_or_wrong_source_id"
    joined = "\n".join(str(block.get("text") or "") for block in blocks)
    joined_normalized = normalized_quote_text(joined)
    if (
        quote
        and (
            quote in joined
            or normalized_quote_text(quote) in joined_normalized
        )
        and len(re.findall(r"[.!?](?:\s|$)", quote)) >= 2
    ):
        return "quote_spans_multiple_sentences_but_remains_contiguous"
    aggressive_quote = _aggressive_typography(quote)
    aggressive_source = _aggressive_typography(claimed_text)
    if aggressive_quote and aggressive_quote in aggressive_source:
        return "harmless_typography_unicode_whitespace_hyphenation"
    fuzzy = _fast_fuzzy_score(quote, claimed_text)
    if (
        re.search(r"(?:\b\w\s+){2,}\w\b|\w-\s+\w", claimed_text)
        and fuzzy >= 0.70
    ):
        return "column_or_ocr_extraction_artifact"
    quote_words = _punctuation_free(quote)
    source_words = _punctuation_free(claimed_text)
    if quote_words and quote_words in source_words:
        return "model_dropped_inserted_punctuation_or_words"
    if fuzzy >= 0.84:
        return "model_dropped_inserted_punctuation_or_words"
    if "..." in quote or "\u2026" in quote or fuzzy >= 0.45:
        return "truncated_ellipsized_or_paraphrased_model_text"
    return "genuinely_absent"


def audit_contextual_quotes(verified_rows, dossiers):
    """Audit every rejected quote against its source and all dossier blocks."""
    records = []
    for row in sorted(verified_rows, key=lambda item: item["custom_id"]):
        custom_id = row["custom_id"]
        result = row["result"]
        dossier = dossiers[custom_id]
        blocks = list(dossier.get("source_blocks") or [])
        by_source = {
            str(block.get("source_id") or ""): block for block in blocks
        }
        order = {
            str(block.get("source_id") or ""): index
            for index, block in enumerate(blocks)
        }
        for path, reference in _iter_invalid_evidence(result):
            source_id = str(reference.get("source_id") or "")
            claimed = by_source.get(source_id)
            quote = str(reference.get("quote") or "")
            exact_matches = _matches(blocks, quote)
            normalized_matches = _matches(blocks, quote, normalized=True)
            exact_elsewhere = [
                block
                for block in exact_matches
                if str(block.get("source_id") or "") != source_id
            ]
            normalized_elsewhere = [
                block
                for block in normalized_matches
                if str(block.get("source_id") or "") != source_id
            ]
            quote_tokens = set(
                re.findall(r"\w+", normalized_quote_text(quote).casefold())
            )
            block_ranks = []
            for block in blocks:
                block_tokens = set(
                    re.findall(
                        r"\w+",
                        normalized_quote_text(
                            block.get("text") or ""
                        ).casefold(),
                    )
                )
                block_ranks.append(
                    (
                        len(quote_tokens & block_tokens),
                        str(block.get("source_id") or ""),
                        block,
                    )
                )
            ranked_blocks = [
                item[2]
                for item in sorted(block_ranks, reverse=True)[:5]
            ]
            best_block = None
            best_fuzzy = 0.0
            for block in ranked_blocks:
                score = _fast_fuzzy_score(
                    quote,
                    str(block.get("text") or ""),
                )
                if score > best_fuzzy:
                    best_fuzzy = score
                    best_block = block
            candidate_blocks = exact_elsewhere or normalized_elsewhere
            unique_correctable = len(candidate_blocks) == 1
            candidate = candidate_blocks[0] if unique_correctable else None
            claimed_text = str((claimed or {}).get("text") or "")
            component_path = _component_path(path)
            affected = _otherwise_valid_claims(
                result,
                component_path,
                path,
            )
            role = ROLE_NAMES.get(
                str(reference.get("evidence_role") or ""),
                str(reference.get("evidence_role") or "unknown"),
            )
            classification = _classify(
                reference,
                claimed,
                blocks,
                exact_elsewhere,
                normalized_elsewhere,
            )
            normalization_features = _normalization_gap_features(
                quote,
                claimed_text,
            )
            neighbor_distance = None
            if candidate is not None and source_id in order:
                neighbor_distance = abs(
                    order[str(candidate.get("source_id") or "")] - order[source_id]
                )
            records.append(
                {
                    "custom_id": custom_id,
                    "doi": row["doi"],
                    "cohort": row["cohort"],
                    "evidence_path": path,
                    "component_path": component_path,
                    "evidence_role": role,
                    "error_code": reference["verification"].get("error_code"),
                    "root_cause": classification,
                    "normalization_gap_features": normalization_features,
                    "quote": quote,
                    "claimed_source_id": source_id,
                    "claimed_page": reference.get("page"),
                    "claimed_section": reference.get("section"),
                    "source_block_exists": claimed is not None,
                    "claimed_source_text": claimed_text,
                    "literal_similarity_claimed_source": _window_similarity(
                        quote,
                        claimed_text,
                    ),
                    "normalized_similarity_claimed_source": _window_similarity(
                        quote,
                        claimed_text,
                        normalized_quote_text,
                    ),
                    "fuzzy_similarity_claimed_source": _fast_fuzzy_score(
                        quote,
                        claimed_text,
                    ),
                    "best_paper_fuzzy_similarity": best_fuzzy,
                    "best_paper_source_id": (
                        str(best_block.get("source_id") or "")
                        if best_block
                        else ""
                    ),
                    "best_paper_source_text": (
                        str(best_block.get("text") or "") if best_block else ""
                    ),
                    "exact_match_source_ids": [
                        str(block.get("source_id") or "")
                        for block in exact_matches
                    ],
                    "normalized_match_source_ids": [
                        str(block.get("source_id") or "")
                        for block in normalized_matches
                    ],
                    "unique_correctable_source": unique_correctable,
                    "correctable_source_id": (
                        str(candidate.get("source_id") or "")
                        if candidate
                        else ""
                    ),
                    "correctable_source_page": (
                        candidate.get("page") if candidate else None
                    ),
                    "correctable_source_section": (
                        candidate.get("section") if candidate else None
                    ),
                    "correctable_source_neighbor_distance": neighbor_distance,
                    "affects_otherwise_valid_claim": bool(affected),
                    "otherwise_valid_claim_paths": affected,
                }
            )

    return summarize_contextual_quote_records(records)


def summarize_contextual_quote_records(records):
    """Summarize canonically ordered per-paper quote-audit records."""
    by_role = Counter(item["evidence_role"] for item in records)
    by_cause = Counter(item["root_cause"] for item in records)
    normalization_features = Counter(
        feature
        for item in records
        for feature in item["normalization_gap_features"]
    )
    similarity_distributions = {}
    for field in (
        "literal_similarity_claimed_source",
        "normalized_similarity_claimed_source",
        "fuzzy_similarity_claimed_source",
        "best_paper_fuzzy_similarity",
    ):
        similarity_distributions[field] = dict(
            sorted(Counter(_similarity_bin(item[field]) for item in records).items())
        )
    recommendations = []
    for feature, count in sorted(normalization_features.items()):
        safe = feature == "unicode_punctuation_mapping"
        recommendations.append(
            {
                "type": "normalization_review",
                "feature": feature,
                "affected_reference_count": count,
                "recommendation": (
                    "Unicode dash/quote canonicalization is narrowly deterministic "
                    "and should be considered with exact regression fixtures."
                    if safe
                    else "Do not enable globally without manual review: this layout "
                    "rewrite can erase semantically meaningful punctuation."
                ),
                "automatically_applied": False,
            }
        )
    typography_count = by_cause[
        "harmless_typography_unicode_whitespace_hyphenation"
    ]
    if typography_count:
        recommendations.append(
            {
                "type": "combined_layout_normalization_review",
                "affected_reference_count": typography_count,
                "recommendation": (
                    "Do not add the audit's combined punctuation/spacing/"
                    "hyphenation rewrite to verification: most cases require "
                    "multiple transformations and could erase meaningful prose."
                ),
                "automatically_applied": False,
            }
        )
    correctable = sum(item["unique_correctable_source"] for item in records)
    if correctable:
        recommendations.append(
            {
                "type": "source_id_correction_review",
                "affected_reference_count": correctable,
                "recommendation": (
                    "A source correction is defensible only when the quote has one "
                    "literal or allowlisted-normalized match in the complete dossier; "
                    "retain the model source_id and record any correction separately."
                ),
                "automatically_applied": False,
            }
        )
    summary = {
        "audit_version": QUOTE_AUDIT_VERSION,
        "invalid_reference_count": len(records),
        "affected_paper_count": len({item["custom_id"] for item in records}),
        "counts_by_evidence_role": dict(sorted(by_role.items())),
        "counts_by_root_cause": {
            cause: by_cause[cause] for cause in ROOT_CAUSES
        },
        "normalization_gap_feature_counts": dict(
            sorted(normalization_features.items())
        ),
        "similarity_distributions": similarity_distributions,
        "unique_correctable_source_count": correctable,
        "affects_otherwise_valid_claim_count": sum(
            item["affects_otherwise_valid_claim"] for item in records
        ),
        "fuzzy_accepted_count": 0,
        "raw_responses_altered": False,
        "recommendations": recommendations,
    }
    return {"summary": summary, "records": records}


def render_quote_audit_examples(audit, examples_per_cause=2):
    """Render bounded, deterministic examples for human review."""
    lines = [
        "# Terra non-exact quote audit",
        "",
        "No rejected quote was fuzzy-accepted and no raw response was altered.",
        "",
    ]
    records = audit["records"]
    for cause in ROOT_CAUSES:
        examples = [item for item in records if item["root_cause"] == cause][
            :examples_per_cause
        ]
        if not examples:
            continue
        lines.extend(["## " + cause.replace("_", " "), ""])
        for item in examples:
            lines.extend(
                [
                    "- DOI: `%s`; path: `%s`; role: `%s`"
                    % (
                        item["doi"],
                        item["evidence_path"],
                        item["evidence_role"],
                    ),
                    "  - Claimed source: `%s`; best paper source: `%s`"
                    % (
                        item["claimed_source_id"],
                        item["best_paper_source_id"],
                    ),
                    "  - Similarity (literal / normalized / fuzzy / paper-best): "
                    "%.3f / %.3f / %.3f / %.3f"
                    % (
                        item["literal_similarity_claimed_source"],
                        item["normalized_similarity_claimed_source"],
                        item["fuzzy_similarity_claimed_source"],
                        item["best_paper_fuzzy_similarity"],
                    ),
                    "  - Affects otherwise valid claim: `%s`"
                    % str(item["affects_otherwise_valid_claim"]).lower(),
                    "  - Model quote: %s"
                    % canonical_json(item["quote"]),
                    "  - Claimed block: %s"
                    % canonical_json(item["claimed_source_text"][:1000]),
                    "  - Best paper block: %s"
                    % canonical_json(item["best_paper_source_text"][:1000]),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "QUOTE_AUDIT_VERSION",
    "audit_contextual_quotes",
    "render_quote_audit_examples",
    "summarize_contextual_quote_records",
]
