#!/usr/bin/env python3
"""Build a claim-level Figure 6 audit without redistributing article text."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


WORKFLOW = Path(__file__).resolve().parents[2]
OUT = Path(
    os.environ.get(
        "LITERATURE_AUDIT_OUT",
        str(WORKFLOW / "data/literature_audit"),
    )
)
ROOT = Path(
    os.environ.get(
        "LITERATURE_PIPELINE_ROOT",
        "/path/to/external-data/literature/"
        "dynamic_oligomerization_publication_v1/contextual_reextraction_v4/"
        "contextual_terra_all_v6_2_batch_20260823",
    )
)
OLD_ROOT = ROOT / "rebuilds/tolerant-11e517d13642a7d2"
FINAL_ROOT = ROOT / "rebuilds/quote_only_v1_20260915"
OLD_TIERED = OLD_ROOT / "tiered_results.jsonl"
FINAL_TIERED = FINAL_ROOT / "tiered_results.jsonl"
TARGET_CLASSES = FINAL_ROOT / "target_classifications.csv"
REQUEST_META = ROOT / "request_metadata.jsonl"
DOSSIERS = ROOT / "dossiers"
RAW_RESPONSES = ROOT / "raw_responses"
API_ERRORS = ROOT / "api_errors"

sys.path.insert(
    0, str(WORKFLOW / "code/literature")
)
from dynamic_oligomerization.quote_normalization import (  # noqa: E402
    normalized_quote_text,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def hash_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tiers_for_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": {
            claim.get("claim_id", f"index:{index}"): claim.get(
                "deterministic_tiering", {}
            ).get("initial_tier")
            for index, claim in enumerate(result.get("state_claims", []))
        },
        "event": {
            claim.get("claim_id", f"index:{index}"): claim.get(
                "deterministic_tiering", {}
            ).get("initial_tier")
            for index, claim in enumerate(result.get("event_claims", []))
        },
        "excluded": [
            claim.get("deterministic_tiering", {}).get("initial_tier")
            for claim in result.get("excluded_candidates", [])
        ],
        "unassigned": [
            claim.get("deterministic_tiering", {}).get("initial_tier")
            for claim in result.get("unassigned_candidates", [])
        ],
    }


def collect_old_tiers() -> tuple[dict[str, Any], dict[str, str]]:
    tiers = {}
    statuses = {}
    for row in iter_jsonl(OLD_TIERED):
        custom_id = row["custom_id"]
        if row.get("technical_error"):
            statuses[custom_id] = str(row.get("error_phase") or "technical_error")
            tiers[custom_id] = None
        else:
            statuses[custom_id] = "parsed_and_tiered"
            tiers[custom_id] = tiers_for_result(row["result"])
    return tiers, statuses


def load_classes() -> tuple[dict[tuple[str, str], dict[str, str]], int]:
    with TARGET_CLASSES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["doi"], row["group_id"]): row for row in rows}
    return result, len(rows)


def evidence_nodes(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if "source_id" in value and "quote" in value:
            yield value
        for child in value.values():
            yield from evidence_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_nodes(child)


def group_from_claim(claim: dict[str, Any]) -> str:
    attribution = claim.get("target_attribution") or {}
    group = attribution.get("selected_group_id")
    if group:
        return str(group)
    for reference in claim.get("candidate_references", []):
        group = reference.get("originating_group_id")
        if group:
            return str(group)
    return ""


def source_maps(dossier: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    blocks = dossier.get("source_blocks", [])
    mapping = {}
    for block in blocks:
        mapping[str(block.get("source_id", ""))] = block
        mapping.setdefault(str(block.get("base_source_id", "")), block)
    return mapping, blocks


def prepare_block(block: dict[str, Any]) -> dict[str, Any]:
    if "_audit_text" not in block:
        text = str(block.get("text") or "")
        normalized = normalized_quote_text(text)
        block["_audit_text"] = text
        block["_audit_normalized"] = normalized
        block["_audit_sha256"] = hash_text(text)
        block["_audit_normalized_sha256"] = hash_text(normalized)
    return block


def resolve_evidence(
    evidence: dict[str, Any],
    block_map: dict[str, dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    quote = str(evidence.get("quote") or "")
    verification = evidence.get("verification") or {}
    claimed = str(evidence.get("source_id") or "")
    candidate_ids = [
        str(verification.get("source_id") or ""),
        claimed,
        claimed.split(".window.", 1)[0],
    ]
    block = next((block_map[value] for value in candidate_ids if value in block_map), None)
    match_type = str(verification.get("match_type") or "")
    if block is not None:
        block = prepare_block(block)
    if block is not None and quote and quote in block["_audit_text"]:
        match_type = match_type or "literal_exact"
    elif block is None or quote not in block["_audit_text"]:
        exact = [
            prepare_block(candidate)
            for candidate in blocks
            if quote and quote in str(candidate.get("text") or "")
        ]
        if exact:
            block = exact[0]
            match_type = match_type or "literal_exact"
        elif quote:
            normalized = normalized_quote_text(quote)
            matched = [
                prepare_block(candidate)
                for candidate in blocks
                if normalized
                and normalized
                in prepare_block(candidate)["_audit_normalized"]
            ]
            if matched:
                block = matched[0]
                match_type = match_type or "allowlisted_normalized_exact"
    normalized_quote = normalized_quote_text(quote)
    return {
        "role": evidence.get("evidence_role"),
        "claimed_source_id": claimed,
        "matched_source_id": (block or {}).get("source_id"),
        "page": evidence.get("page") or (block or {}).get("page"),
        "section": evidence.get("section") or (block or {}).get("section"),
        "occurrence_index": evidence.get("occurrence_index"),
        "source_block_sha256": block["_audit_sha256"] if block else None,
        "normalized_source_block_sha256": (
            block["_audit_normalized_sha256"] if block else None
        ),
        "quote_sha256": hash_text(quote),
        "normalized_quote_sha256": hash_text(normalized_quote),
        "quote_character_count": len(quote),
        "match_type": match_type or ("unresolved" if block is None else "source_only"),
        "character_start": verification.get("character_start"),
        "character_end": verification.get("character_end"),
        "used_for_final_quote_gate": evidence.get("evidence_role") == "result",
    }


def old_tier(
    old: dict[str, Any] | None, kind: str, index: int, claim: dict[str, Any]
) -> tuple[str | None, str]:
    if old is None:
        return None, "technical_or_missing_old_row"
    if kind in {"state", "event"}:
        claim_id = claim.get("claim_id", f"index:{index}")
        return old[kind].get(claim_id), "claim_id"
    values = old[kind]
    return (values[index] if index < len(values) else None), "collection_index"


def claim_row(
    *,
    custom_id: str,
    doi: str,
    kind: str,
    index: int,
    claim: dict[str, Any],
    old: dict[str, Any] | None,
    dossier: dict[str, Any],
    classes: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    collection_name = {
        "state": "state_claims",
        "event": "event_claims",
        "excluded": "excluded_candidates",
        "unassigned": "unassigned_candidates",
    }[kind]
    group_id = group_from_claim(claim)
    canonical = {
        row.get("group_id"): row for row in dossier.get("canonical_targets", [])
    }.get(group_id, {})
    construct_id = str(claim.get("construct_id") or "")
    construct = next(
        (
            row
            for row in dossier.get("result_constructs", [])
            if row.get("construct_id") == construct_id
        ),
        {},
    )
    # Final result constructs, not dossier candidates, carry the adjudicated graph.
    block_map, blocks = source_maps(dossier)
    seen = set()
    refs = []
    for evidence in evidence_nodes(claim):
        key = (
            evidence.get("evidence_role"),
            evidence.get("source_id"),
            evidence.get("occurrence_index"),
            evidence.get("quote"),
        )
        if key in seen:
            continue
        seen.add(key)
        refs.append(resolve_evidence(evidence, block_map, blocks))
    previous_tier, join_method = old_tier(old, kind, index, claim)
    new_tier = claim.get("deterministic_tiering", {}).get("initial_tier")
    class_row = classes.get((doi, group_id))
    attribution = claim.get("target_attribution") or {}
    citation_scope = claim.get("citation_scope") or {}
    if kind == "state":
        assembly = claim.get("assembly_class")
        context = claim.get("biological_context")
        subunit_count = claim.get("subunit_count")
    elif kind == "event":
        assembly = {
            "from": claim.get("from_assembly_classes"),
            "to": claim.get("to_assembly_classes"),
        }
        context = claim.get("biological_context")
        subunit_count = {
            "from": claim.get("from_target_subunit_counts"),
            "to": claim.get("to_target_subunit_counts"),
        }
    else:
        assembly = None
        context = None
        subunit_count = None
    return {
        "audit_schema_version": "literature_claim_audit_v1",
        "custom_id": custom_id,
        "doi": doi,
        "claim_path": f"/result/{collection_name}/{index}",
        "claim_kind": kind,
        "claim_id": claim.get("claim_id"),
        "claim_object_sha256": canonical_hash(claim),
        "target": {
            "assignment_status": attribution.get("assignment_status"),
            "target_attribution_status": attribution.get(
                "target_attribution_status"
            ),
            "selected_group_id": group_id,
            "uniprot_accession": canonical.get("uniprot_accession"),
            "candidate_target_ids": attribution.get("candidate_target_ids", []),
            "subject_text_sha256": (
                hash_text(attribution.get("subject_text"))
                if attribution.get("subject_text")
                else None
            ),
            "attribution_basis_sha256": (
                hash_text(attribution.get("attribution_basis"))
                if attribution.get("attribution_basis")
                else None
            ),
        },
        "construct": {
            "construct_id": construct_id or None,
            "selected_group_id": construct.get("selected_group_id"),
            "relation_to_parent": construct.get("relation_to_parent"),
            "parent_construct_id": construct.get("parent_construct_id"),
            "resolution_status": construct.get("resolution_status"),
            "construct_text_sha256": (
                hash_text(construct.get("construct_text"))
                if construct.get("construct_text")
                else None
            ),
        },
        "semantics": {
            "subunit_count": subunit_count,
            "assembly_class": assembly,
            "biological_context": context,
            "method_text_sha256": (
                hash_text(claim.get("method")) if claim.get("method") else None
            ),
            "method_result_link_type": claim.get("method_result_link_type"),
            "claim_origin": claim.get("claim_origin"),
            "citation_scope": citation_scope.get("status"),
        },
        "decision": {
            "old_tier": previous_tier,
            "old_tier_join": join_method,
            "new_tier": new_tier,
            "policy_version": claim.get("deterministic_tiering", {}).get(
                "policy_version"
            ),
            "tier_reasons": claim.get("deterministic_tiering", {}).get(
                "tier_reasons", []
            ),
            "exclusion_reason": claim.get("reason"),
            "exclusion_reason_detail_sha256": (
                hash_text(claim.get("reason_detail"))
                if claim.get("reason_detail")
                else None
            ),
        },
        "final_target_class": {
            "state_classification": (
                class_row.get("state_classification") if class_row else None
            ),
            "previous_state_classification": (
                class_row.get("previous_state_classification")
                if class_row
                else None
            ),
            "classification_changed": (
                str(class_row.get("classification_changed", "")).lower()
                in {"1", "true", "yes"}
                if class_row
                else None
            ),
            "claim_contributes": bool(
                kind in {"state", "event"}
                and new_tier == "llm_validated"
                and class_row is not None
            ),
        },
        "evidence_refs": refs,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_tiers, old_statuses = collect_old_tiers()
    classes, class_count = load_classes()
    shutil.copy2(TARGET_CLASSES, OUT / "target_classifications.csv")

    counts = Counter()
    transition_counts = Counter()
    final_counts_by_request = Counter()
    evidence_match_counts = Counter()
    accepted_result_evidence_failures = 0
    dossier_summary_by_request = {}
    claim_output = OUT / "literature_claim_audit.jsonl"
    with claim_output.open("w", encoding="utf-8") as output:
        for row in iter_jsonl(FINAL_TIERED):
            custom_id = row["custom_id"]
            doi = row["doi"]
            result = row["result"]
            dossier = json.loads((DOSSIERS / f"{custom_id}.json").read_text())
            dossier["result_constructs"] = result.get("constructs", [])
            dossier_summary_by_request[custom_id] = {
                "paper_bytes": dossier.get("paper_identity", {}).get("paper_bytes"),
                "source_block_count": len(dossier.get("source_blocks", [])),
            }
            collections = [
                ("state", result.get("state_claims", [])),
                ("event", result.get("event_claims", [])),
                ("excluded", result.get("excluded_candidates", [])),
                ("unassigned", result.get("unassigned_candidates", [])),
            ]
            for kind, claims in collections:
                for index, claim in enumerate(claims):
                    audit = claim_row(
                        custom_id=custom_id,
                        doi=doi,
                        kind=kind,
                        index=index,
                        claim=claim,
                        old=old_tiers.get(custom_id),
                        dossier=dossier,
                        classes=classes,
                    )
                    output.write(
                        json.dumps(
                            audit,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    new_tier = audit["decision"]["new_tier"]
                    old_tier_value = audit["decision"]["old_tier"]
                    counts[new_tier] += 1
                    transition_counts[(old_tier_value, new_tier)] += 1
                    final_counts_by_request[custom_id] += 1
                    for evidence in audit["evidence_refs"]:
                        evidence_match_counts[evidence["match_type"]] += 1
                    if (
                        kind in {"state", "event"}
                        and new_tier == "llm_validated"
                    ):
                        result_evidence = [
                            evidence
                            for evidence in audit["evidence_refs"]
                            if evidence["used_for_final_quote_gate"]
                        ]
                        if not result_evidence or any(
                            evidence["match_type"]
                            not in {
                                "literal_exact",
                                "allowlisted_normalized_exact",
                            }
                            for evidence in result_evidence
                        ):
                            accepted_result_evidence_failures += 1

    request_output = OUT / "literature_request_audit.jsonl"
    request_count = 0
    response_count = 0
    error_count = 0
    with request_output.open("w", encoding="utf-8") as output:
        for metadata in iter_jsonl(REQUEST_META):
            custom_id = metadata["custom_id"]
            if custom_id not in dossier_summary_by_request:
                dossier_path = DOSSIERS / f"{custom_id}.json"
                if dossier_path.is_file():
                    dossier = json.loads(dossier_path.read_text())
                    dossier_summary_by_request[custom_id] = {
                        "paper_bytes": dossier.get("paper_identity", {}).get(
                            "paper_bytes"
                        ),
                        "source_block_count": len(dossier.get("source_blocks", [])),
                    }
            response = RAW_RESPONSES / f"{custom_id}.json"
            error = API_ERRORS / f"{custom_id}.json"
            if response.is_file():
                response_kind = "response"
                response_sha = sha256(response)
                response_count += 1
            elif error.is_file():
                response_kind = "api_error"
                response_sha = sha256(error)
                error_count += 1
            else:
                response_kind = old_statuses.get(custom_id, "missing")
                response_sha = None
            audit = {
                "audit_schema_version": "literature_request_audit_v1",
                "custom_id": custom_id,
                "doi": metadata["doi"],
                "cohort": metadata.get("cohort"),
                "dossier_file_sha256": metadata.get("dossier_file_sha256"),
                "dossier_content_sha256": metadata.get(
                    "dossier_content_sha256"
                ),
                "paper_sha256": metadata.get("paper_sha256"),
                "paper_bytes": dossier_summary_by_request.get(custom_id, {}).get(
                    "paper_bytes"
                ),
                "source_block_count": dossier_summary_by_request.get(
                    custom_id, {}
                ).get("source_block_count"),
                "request_sha256": metadata.get("request_sha256"),
                "request_body_sha256": metadata.get("request_body_sha256"),
                "user_prompt_sha256": metadata.get("user_prompt_sha256"),
                "system_prompt_sha256": metadata.get("system_prompt_sha256"),
                "response_kind": response_kind,
                "response_file_sha256": response_sha,
                "old_pipeline_status": old_statuses.get(custom_id),
                "final_claim_count": final_counts_by_request.get(custom_id, 0),
            }
            output.write(
                json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n"
            )
            request_count += 1

    expected = {
        "request_rows": 20812,
        "response_rows": 20789,
        "api_error_rows": 23,
        "claim_rows": 72811,
        "tier_counts": {
            "llm_validated": 31120,
            "excluded_by_policy": 40439,
            "ambiguous_candidate": 1252,
        },
        "target_class_rows": 12263,
    }
    observed = {
        "request_rows": request_count,
        "response_rows": response_count,
        "api_error_rows": error_count,
        "claim_rows": sum(counts.values()),
        "tier_counts": dict(counts),
        "target_class_rows": class_count,
    }
    if observed != expected:
        raise RuntimeError({"expected": expected, "observed": observed})
    if accepted_result_evidence_failures:
        raise RuntimeError(
            {
                "accepted_result_evidence_failures":
                    accepted_result_evidence_failures
            }
        )

    forbidden = {
        "quote",
        "text",
        "source_text",
        "mention_text",
        "population_text",
        "reason_detail",
        "attribution_basis",
        "extraction_summary",
    }
    for path in (claim_output, request_output):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                stack = [row]
                while stack:
                    value = stack.pop()
                    if isinstance(value, dict):
                        overlap = forbidden & set(value)
                        if overlap:
                            raise RuntimeError(
                                f"{path}:{line_number}: forbidden keys {overlap}"
                            )
                        stack.extend(value.values())
                    elif isinstance(value, list):
                        stack.extend(value)

    inputs = {
        "request_metadata.jsonl": sha256(REQUEST_META),
        "old_tiered_results.jsonl": sha256(OLD_TIERED),
        "final_tiered_results.jsonl": sha256(FINAL_TIERED),
        "target_classifications.csv": sha256(TARGET_CLASSES),
    }
    outputs = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in (
            request_output,
            claim_output,
            OUT / "target_classifications.csv",
        )
    }
    manifest = {
        "audit_schema_version": "literature_audit_manifest_v1",
        "copyright_policy": (
            "No article text, quotations, prompts, model responses, or "
            "free-text semantic explanations are included."
        ),
        "expected": expected,
        "observed": observed,
        "evidence_validation": {
            "match_type_counts": dict(evidence_match_counts),
            "accepted_result_evidence_failures":
                accepted_result_evidence_failures,
        },
        "tier_transitions": [
            {"old_tier": old, "new_tier": new, "count": count}
            for (old, new), count in sorted(
                transition_counts.items(), key=lambda item: str(item[0])
            )
        ],
        "inputs": inputs,
        "outputs": outputs,
        "generator_sha256": sha256(Path(__file__)),
        "limitations": [
            "The final historical policy trusts Terra semantics for attribution, construct, context, assembly class, provenance, and method linkage.",
            "Hash-only audit rows permit traceability but not semantic review without lawful access to source articles/dossiers.",
            "Targets without an accepted classification are absent rather than represented as Unknown.",
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()
