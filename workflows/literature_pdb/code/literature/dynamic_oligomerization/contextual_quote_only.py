#!/usr/bin/env python3
"""Build a simple Terra-semantics plus exact-quote literature layer.

Terra remains responsible for target attribution and scientific interpretation.
The only independent evidence check performed here is whether every result
quote occurs literally, or after the allowlisted PDF normalization, somewhere
in the supplied paper blocks. Identity, construct, event-shape, provenance,
and Mini-pointer diagnostics are retained in the historical strict rebuild but
do not gate claims in this layer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .doi import normalize_doi
from .serialization import canonical_json
from .state_classification import reported_state_classifications
from .quote_normalization import normalized_quote_text


# This version identifies the exact acceptance policy in every tiered claim.
POLICY_VERSION = "terra_semantics_exact_result_quotes_v1"

# Normalized exact matching permits only deterministic PDF/OCR cleanup; fuzzy
# or paraphrase matches never pass the quote gate.
EXACT_MATCH_TYPES = {"literal_exact", "allowlisted_normalized_exact"}
DIRECT_CLAIM_ORIGINS = {
    "current_paper_experiment",
    "current_paper_interpretation",
}
DIRECT_CITATION_SCOPES = {"current_paper", "mixed_separable"}
SELECTED_ATTRIBUTIONS = {"exact_target", "construct_of_target"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            yield value


def _atomic_text(path: Path, text: str) -> None:
    """Write text durably without exposing a partially written final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tier(name: str, reasons: list[str]) -> dict[str, Any]:
    """Return the common audit payload attached to every candidate claim."""
    return {
        "initial_tier": name,
        "tier_reasons": sorted(set(reasons)),
        "policy_version": POLICY_VERSION,
        "deterministic_high_confidence": False,
        "deterministic_high_confidence_status": "not_applicable",
    }


def verify_quote_anywhere(
    quote: str,
    source_blocks: list[dict[str, Any]],
    normalized_blocks: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Find a quote anywhere in the supplied paper, ignoring block metadata."""
    quote = str(quote or "")
    if not quote:
        return {"verified": False, "match_type": "invalid", "reason": "empty_quote"}

    # Prefer a byte-for-byte substring match and retain its actual location.
    for block in source_blocks:
        text = str(block.get("text") or "")
        start = text.find(quote)
        if start >= 0:
            return {
                "verified": True,
                "match_type": "literal_exact",
                "source_id": str(block.get("source_id") or ""),
                "character_start": start,
                "character_end": start + len(quote),
            }

    # Search normalized text only after literal matching fails. Block metadata
    # supplied by Terra is deliberately ignored: the quote may occur anywhere
    # in the paper dossier.
    normalized_quote = normalized_quote_text(quote)
    if normalized_quote:
        if normalized_blocks is None:
            normalized_blocks = [
                (
                    str(block.get("source_id") or ""),
                    normalized_quote_text(str(block.get("text") or "")),
                )
                for block in source_blocks
            ]
        for source_id, normalized_source in normalized_blocks:
            start = normalized_source.find(normalized_quote)
            if start >= 0:
                return {
                    "verified": True,
                    "match_type": "allowlisted_normalized_exact",
                    "normalization": "normalized_quote_text_v1",
                    "source_id": source_id,
                    "normalized_character_start": start,
                    "normalized_character_end": start + len(normalized_quote),
                }
    return {
        "verified": False,
        "match_type": "invalid",
        "reason": "quote_not_exact_anywhere_in_paper",
    }


def _terra_semantic_reasons(
    claim: dict[str, Any], canonical_group_ids: set[str]
) -> tuple[str, list[str]]:
    """Map Terra's own semantic fields to a transparent inclusion decision."""
    attribution = claim.get("target_attribution") or {}
    assignment = str(attribution.get("assignment_status") or "")
    status = str(attribution.get("target_attribution_status") or "")
    selected = str(attribution.get("selected_group_id") or "")

    # Ambiguity is retained as its own auditable tier instead of being treated
    # as evidence against the candidate.
    if assignment in {"ambiguous", "unassigned"} or status in {
        "ambiguous",
        "unresolved",
    }:
        return "ambiguous_candidate", ["terra_target_assignment_ambiguous"]

    reasons: list[str] = []
    # These are direct translations of Terra's structured labels. The script
    # does not independently recompute identity or biological interpretation.
    if assignment != "selected":
        reasons.append("terra_target_not_selected")
    if status not in SELECTED_ATTRIBUTIONS:
        reasons.append("terra_target_not_exact_or_construct")
    if selected not in canonical_group_ids:
        reasons.append("selected_target_not_in_dossier")
    if claim.get("claim_origin") not in DIRECT_CLAIM_ORIGINS:
        reasons.append("terra_not_current_paper_claim")
    citation = claim.get("citation_scope") or {}
    if citation.get("status") not in DIRECT_CITATION_SCOPES:
        reasons.append("terra_citation_scope_not_direct")
    if claim.get("biological_context") == "crystallographic":
        reasons.append("terra_crystallographic_only")
    if claim.get("assembly_class") == "partner_templated":
        reasons.append("terra_partner_templated")
    if claim.get("method_result_link_type") == "unresolved":
        reasons.append("terra_method_result_link_unresolved")
    if reasons:
        return "excluded_by_policy", reasons
    return "llm_validated", ["terra_semantics_and_exact_result_quotes"]


def apply_quote_only_policy(
    parsed: dict[str, Any], dossier: dict[str, Any]
) -> dict[str, Any]:
    """Attach quote-only tiers to one schema-valid Terra result."""
    result = copy.deepcopy(parsed)
    expected_doi = normalize_doi(dossier["paper_identity"]["doi"])
    actual_doi = normalize_doi(result["paper_doi"])
    if actual_doi != expected_doi:
        # A DOI mismatch indicates the wrong dossier/result pairing, so stop
        # rather than emitting a misleading per-claim exclusion.
        raise ValueError(f"paper DOI mismatch: {actual_doi} != {expected_doi}")

    source_blocks = list(dossier.get("source_blocks") or [])
    # Pre-normalize the paper once; a dossier may contain many evidence quotes.
    normalized_blocks = [
        (
            str(block.get("source_id") or ""),
            normalized_quote_text(str(block.get("text") or "")),
        )
        for block in source_blocks
    ]
    canonical_group_ids = {
        str(target.get("group_id") or "")
        for target in dossier.get("canonical_targets") or []
    }
    for field in ("state_claims", "event_claims"):
        for claim in result.get(field) or []:
            evidence = list(claim.get("result_evidence") or [])
            for reference in evidence:
                reference["verification"] = verify_quote_anywhere(
                    str(reference.get("quote") or ""),
                    source_blocks,
                    normalized_blocks,
                )

            # A claim is accepted only when it has evidence and every result
            # quote passes one of the two exact-match modes.
            all_exact = bool(evidence) and all(
                reference.get("verification", {}).get("match_type")
                in EXACT_MATCH_TYPES
                for reference in evidence
            )
            if not all_exact:
                claim["deterministic_tiering"] = _tier(
                    "excluded_by_policy", ["result_quote_not_exact"]
                )
                continue
            tier, reasons = _terra_semantic_reasons(claim, canonical_group_ids)
            claim["deterministic_tiering"] = _tier(tier, reasons)

    # Terra has already separated these candidates from state/event claims.
    # Preserve that decision while assigning the common offline tier format.
    for item in result.get("excluded_candidates") or []:
        item["deterministic_tiering"] = _tier(
            "excluded_by_policy",
            ["terra_excluded_candidate", str(item.get("reason") or "unspecified")],
        )
    for item in result.get("unassigned_candidates") or []:
        tier = (
            "excluded_by_policy"
            if item.get("reason") == "nonmanifest_protein"
            else "ambiguous_candidate"
        )
        item["deterministic_tiering"] = _tier(
            tier, ["terra_unassigned_candidate", str(item.get("reason") or "unspecified")]
        )
    result["offline_policy"] = {
        "version": POLICY_VERSION,
        "independent_gate": "all_result_quotes_exact_anywhere_in_supplied_paper",
        "terra_fields_used_without_deterministic_revalidation": [
            "target_attribution",
            "claim_origin",
            "citation_scope",
            "biological_context",
            "assembly_class",
            "method_result_link_type",
        ],
        "non_gating_diagnostics": [
            "identity_graph",
            "construct_graph",
            "event_shape",
            "source_block_metadata",
            "candidate_pointer_accounting",
        ],
    }
    return result


def _claim_tiers(result: dict[str, Any]) -> Counter:
    counts: Counter = Counter()
    for field in (
        "state_claims",
        "event_claims",
        "excluded_candidates",
        "unassigned_candidates",
    ):
        for item in result.get(field) or []:
            counts[item["deterministic_tiering"]["initial_tier"]] += 1
    return counts


def _classification_rows(
    record: dict[str, Any],
    dossier: dict[str, Any],
) -> list[dict[str, str]]:
    result = record["result"]
    # This helper considers only llm_validated state/event claims and collapses
    # their monomer/oligomer signals to one class per selected target group.
    classes = reported_state_classifications(result)
    target_index = {
        str(target.get("group_id") or ""): target
        for target in dossier.get("canonical_targets") or []
    }
    rows = []
    for group_id, state_class in sorted(classes.items()):
        target = target_index.get(group_id)
        if target is None:
            continue
        names = target.get("target_names") or []
        rows.append(
            {
                "doi": normalize_doi(record["doi"]),
                "group_id": group_id,
                "uniprot_accession": str(target.get("uniprot_accession") or ""),
                "target_name": str(names[0] if names else ""),
                "state_classification": state_class,
            }
        )
    return rows


def build_quote_only_layer(
    parsed_results: Path,
    dossier_dir: Path,
    output_dir: Path,
    old_tiered_results: Path | None = None,
) -> dict[str, Any]:
    """Build immutable quote-only JSONL, classes, and transition audit."""
    if output_dir.exists():
        # Publication rebuilds are append-only: never overwrite an old layer.
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    # The old layer is read solely for transition reporting. It cannot change
    # quote verification, tiers, or classifications in the new layer.
    old_claim_tiers: dict[tuple[str, str, str], str] = {}
    old_classes: dict[tuple[str, str], str] = {}
    if old_tiered_results is not None:
        for record in _jsonl(old_tiered_results):
            custom_id = str(record.get("custom_id") or "")
            result = record.get("result") or {}
            for field in ("state_claims", "event_claims"):
                for claim in result.get(field) or []:
                    old_claim_tiers[
                        (custom_id, field, str(claim.get("claim_id") or ""))
                    ] = str(
                        claim.get("deterministic_tiering", {}).get("initial_tier")
                        or ""
                    )
            for group_id, value in reported_state_classifications(result).items():
                old_classes[(normalize_doi(record["doi"]), group_id)] = value

    tier_counts: Counter = Counter()
    claim_transitions: Counter = Counter()
    new_classes: dict[tuple[str, str], str] = {}
    class_rows: list[dict[str, str]] = []
    paper_count = 0
    tiered_path = output_dir / "tiered_results.jsonl"
    with tiered_path.open("w", encoding="utf-8", newline="") as output:
        for record in _jsonl(parsed_results):
            custom_id = str(record.get("custom_id") or "")
            dossier_path = dossier_dir / f"{custom_id}.json"
            dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
            result = apply_quote_only_policy(record["result"], dossier)
            revised = {
                "custom_id": custom_id,
                "doi": normalize_doi(record["doi"]),
                "cohort": record.get("cohort"),
                # Inputs are schema-valid Terra results; integrity failures
                # raise and stop the rebuild instead of becoming paper errors.
                "technical_error": False,
                "result": result,
            }
            output.write(canonical_json(revised) + "\n")
            paper_count += 1
            tier_counts.update(_claim_tiers(result))
            for field in ("state_claims", "event_claims"):
                for claim in result.get(field) or []:
                    new_tier = str(
                        claim["deterministic_tiering"]["initial_tier"]
                    )
                    old_tier = old_claim_tiers.get(
                        (custom_id, field, str(claim.get("claim_id") or "")),
                        "not_in_old_layer",
                    )
                    claim_transitions[(old_tier, new_tier)] += 1
            for row in _classification_rows(revised, dossier):
                key = (row["doi"], row["group_id"])
                if key in new_classes:
                    raise ValueError(f"duplicate DOI×group classification: {key}")
                new_classes[key] = row["state_classification"]
                class_rows.append(row)

    # This compact table is the canonical DOI×target input for structural joins.
    classification_path = output_dir / "target_classifications.csv"
    with classification_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doi",
                "group_id",
                "uniprot_accession",
                "target_name",
                "state_classification",
                "previous_state_classification",
                "classification_changed",
            ],
        )
        writer.writeheader()
        for row in sorted(
            class_rows, key=lambda value: (value["doi"], value["group_id"])
        ):
            previous = old_classes.get((row["doi"], row["group_id"]), "")
            writer.writerow(
                {
                    **row,
                    "previous_state_classification": previous,
                    "classification_changed": str(
                        bool(previous and previous != row["state_classification"])
                    ).lower(),
                }
            )

    removed_keys = sorted(set(old_classes) - set(new_classes))
    transitions = Counter()
    for key in set(old_classes) | set(new_classes):
        transitions[(old_classes.get(key, "No class"), new_classes.get(key, "No class"))] += 1
    summary = {
        "policy_version": POLICY_VERSION,
        "paper_count": paper_count,
        "claim_tier_counts": dict(sorted(tier_counts.items())),
        "claim_tier_transitions": [
            {"old_tier": old, "new_tier": new, "count": count}
            for (old, new), count in sorted(claim_transitions.items())
        ],
        "target_classification_count": len(new_classes),
        "target_classification_counts": dict(Counter(new_classes.values())),
        "target_classification_transitions": [
            {"old_class": old, "new_class": new, "count": count}
            for (old, new), count in sorted(transitions.items())
        ],
        "old_classified_targets_removed": len(removed_keys),
        "inputs": {
            "parsed_results": str(parsed_results.resolve()),
            "parsed_results_sha256": _sha256(parsed_results),
            "dossier_dir": str(dossier_dir.resolve()),
            "old_tiered_results": (
                str(old_tiered_results.resolve()) if old_tiered_results else ""
            ),
            "old_tiered_results_sha256": (
                _sha256(old_tiered_results) if old_tiered_results else ""
            ),
        },
        "outputs": {
            "tiered_results": str(tiered_path.resolve()),
            "target_classifications": str(classification_path.resolve()),
        },
    }
    _atomic_text(
        output_dir / "summary.json", canonical_json(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-results", type=Path, required=True)
    parser.add_argument("--dossier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-tiered-results", type=Path)
    args = parser.parse_args()
    summary = build_quote_only_layer(
        args.parsed_results,
        args.dossier_dir,
        args.output_dir,
        args.old_tiered_results,
    )
    print(canonical_json(summary, indent=2))


if __name__ == "__main__":
    main()
