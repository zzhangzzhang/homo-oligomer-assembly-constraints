"""Human-readable contextual evidence exports with transparent decisions."""

import csv
import json
import os
import tempfile
from pathlib import Path

from .doi import normalize_doi
from .state_classification import reported_state_classifications


HUMAN_EXPORT_COLUMNS = (
    "doi",
    "uniprot_accession",
    "target_name",
    "evidence_type",
    "construct",
    "reported_state_or_transition",
    "state_classification",
    "direction",
    "dynamic_regime",
    "condition_linked",
    "trigger_type",
    "trigger",
    "accepted",
    "reason",
    "quote_verified",
    "quote",
)

def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError("blank JSONL line at %s:%d" % (path, line_number))
            rows.append(json.loads(raw))
    return rows


def _evidence_payload(claim, evidence_type):
    evidence_values = (
        claim.get("result_evidence") or []
        if evidence_type in {"state", "event"}
        else claim.get("evidence") or []
    )
    quotes = []
    statuses = []
    for evidence in evidence_values:
        verification = evidence.get("verification") or {}
        quote = str(evidence.get("quote") or "")
        if quote and quote not in quotes:
            quotes.append(quote)
        match_type = str(verification.get("match_type") or "")
        if match_type in {"literal_exact", "allowlisted_normalized_exact"}:
            statuses.append("verified")
        elif verification.get("verified") is False or match_type == "invalid":
            statuses.append("invalid")
        else:
            statuses.append("unknown")
    observed = set(statuses)
    if observed == {"verified"}:
        quote_verified = "true"
    elif "invalid" in observed and "verified" in observed:
        quote_verified = "partial"
    elif "invalid" in observed:
        quote_verified = "false"
    else:
        quote_verified = "unknown"
    return "\n\n".join(quotes), quote_verified


def _target_index(dossier):
    return {
        str(target["group_id"]): {
            "uniprot_accession": str(target.get("uniprot_accession") or ""),
            "target_name": (
                str((target.get("target_names") or [""])[0])
                if target.get("target_names")
                else ""
            ),
        }
        for target in dossier.get("canonical_targets") or []
    }


def _simple_direction(claim):
    mapped = []
    for value in claim.get("direction_processes") or []:
        simple = "none" if value == "no_net_direction" else str(value)
        if simple not in mapped:
            mapped.append(simple)
    return "+".join(mapped) if mapped else "unknown"


def _simple_dynamic_regime(claim):
    value = str(claim.get("dynamic_regime") or "unknown")
    if value == "unresolved_heterogeneity":
        return "unknown"
    return value


def _decision(claim):
    tiering = claim.get("deterministic_tiering") or {}
    tier = str(tiering.get("initial_tier") or "technical_error")
    accepted = tier == "llm_validated"
    if accepted:
        return "true", "all validation gates passed"
    parts = []
    reason = str(claim.get("reason") or "")
    if reason:
        parts.append(reason)
    for value in tiering.get("tier_reasons") or []:
        value = str(value)
        if value and value not in parts and value != "hard_exclusion":
            parts.append(value)
    validation = claim.get("component_validation") or {}
    for value in validation.get("error_codes") or []:
        value = str(value)
        if value and value not in parts:
            parts.append(value)
    detail = str(claim.get("reason_detail") or "")
    if detail:
        parts.append(detail)
    if not parts:
        parts.append(tier)
    return "false", "; ".join(parts)


def _claim_group_ids(claim, targets, constructs):
    attribution = claim.get("target_attribution") or {}
    selected = str(attribution.get("selected_group_id") or "")
    if selected in targets:
        return [selected]
    candidate_ids = [
        str(value)
        for value in attribution.get("candidate_target_ids") or []
        if str(value) in targets
    ]
    if candidate_ids:
        return sorted(set(candidate_ids))
    construct = constructs.get(str(claim.get("construct_id") or "")) or {}
    construct_target = str(construct.get("selected_group_id") or "")
    return [construct_target] if construct_target in targets else []


def _reported_state_or_transition(claim, evidence_type):
    if evidence_type == "state":
        return str(claim.get("state_label") or "")
    if evidence_type == "event":
        from_states = " + ".join(str(value) for value in claim.get("from_states") or [])
        to_states = " + ".join(str(value) for value in claim.get("to_states") or [])
        if from_states or to_states:
            return "%s -> %s" % (from_states or "unknown", to_states or "unknown")
    return ""


def _display_target(group_ids, targets, classifications):
    selected = [targets[group_id] for group_id in group_ids if group_id in targets]
    accessions = sorted(
        {item["uniprot_accession"] for item in selected if item["uniprot_accession"]}
    )
    names = sorted({item["target_name"] for item in selected if item["target_name"]})
    states = sorted(
        {classifications.get(group_id, "Unknown") for group_id in group_ids}
    )
    return {
        "uniprot_accession": " | ".join(accessions),
        "target_name": " | ".join(names),
        "state_classification": (
            states[0] if len(states) == 1 else "Unknown"
        ),
    }


def _merge_display_values(first, second):
    values = []
    for combined in (first, second):
        for value in str(combined or "").split(" | "):
            if value and value not in values:
                values.append(value)
    return " | ".join(values)


def _visible_control_characters(value):
    """Render embedded PDF control bytes visibly in the human-facing TSV."""
    return "".join(
        chr(0x2400 + ord(character))
        if ord(character) < 32 and character not in "\t\n\r"
        else character
        for character in str(value)
    )


def _collapse_repeated_evidence(rows):
    """Merge identical or nested quotes when every classification agrees."""
    collapsed = []
    ignored = {"construct", "trigger", "quote"}
    base_columns = [
        column for column in HUMAN_EXPORT_COLUMNS if column not in ignored
    ]
    by_base = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item["doi"],
            item["uniprot_accession"],
            item["evidence_type"],
            -len(item["quote"]),
            item["quote"],
        ),
    ):
        base = tuple(row[column] for column in base_columns)
        merged = None
        candidates = by_base.setdefault(base, [])
        for existing in candidates:
            quote = row["quote"]
            existing_quote = existing["quote"]
            if (
                quote == existing_quote
                or (quote and quote in existing_quote)
                or (existing_quote and existing_quote in quote)
            ):
                merged = existing
                break
        if merged is None:
            copied = dict(row)
            collapsed.append(copied)
            candidates.append(copied)
            continue
        merged["construct"] = _merge_display_values(
            merged["construct"], row["construct"]
        )
        merged["trigger"] = _merge_display_values(
            merged["trigger"], row["trigger"]
        )
        if len(row["quote"]) > len(merged["quote"]):
            merged["quote"] = row["quote"]
    return sorted(
        collapsed,
        key=lambda row: (
            row["doi"],
            row["uniprot_accession"],
            {
                "state": 0,
                "event": 1,
                "excluded_candidate": 2,
                "unassigned_candidate": 3,
            }.get(row["evidence_type"], 4),
            row["reported_state_or_transition"],
            row["accepted"] != "true",
            row["quote"],
        ),
    )


def build_human_readable_rows(tiered_rows, dossiers, accepted_only=True):
    """Build compact evidence rows and collapse display-level repetitions."""
    rows = []
    for record in tiered_rows:
        if record.get("technical_error"):
            continue
        custom_id = str(record.get("custom_id") or "")
        dossier = dossiers.get(custom_id)
        if dossier is None:
            raise ValueError("missing dossier for custom_id %s" % custom_id)
        result = record.get("result") or {}
        classifications = reported_state_classifications(result)
        targets = _target_index(dossier)
        constructs = {
            str(item.get("construct_id") or ""): item
            for item in result.get("constructs") or []
        }
        doi = normalize_doi(record.get("doi") or result.get("paper_doi"))

        for evidence_type, field in (
            ("state", "state_claims"),
            ("event", "event_claims"),
            ("excluded_candidate", "excluded_candidates"),
            ("unassigned_candidate", "unassigned_candidates"),
        ):
            for claim in result.get(field) or []:
                accepted, reason = _decision(claim)
                if accepted_only and accepted != "true":
                    continue
                group_ids = _claim_group_ids(claim, targets, constructs)
                target = _display_target(group_ids, targets, classifications)
                quote, quote_verified = _evidence_payload(claim, evidence_type)
                if not quote and accepted == "true":
                    raise ValueError(
                        "accepted claim %s has no displayable evidence quote"
                        % claim.get("claim_id")
                    )
                if evidence_type == "event":
                    direction = _simple_direction(claim)
                    dynamic_regime = _simple_dynamic_regime(claim)
                    condition_linked = (
                        "true" if claim.get("condition_linked") else "false"
                    )
                    trigger_type = str(claim.get("trigger_type") or "none")
                    trigger = str(claim.get("trigger_name") or "")
                else:
                    direction = ""
                    dynamic_regime = ""
                    condition_linked = ""
                    trigger_type = ""
                    trigger = ""
                construct = constructs.get(
                    str(claim.get("construct_id") or "")
                ) or {}
                rows.append(
                    {
                        "doi": doi,
                        "uniprot_accession": target["uniprot_accession"],
                        "target_name": target["target_name"],
                        "evidence_type": evidence_type,
                        "construct": str(construct.get("construct_text") or ""),
                        "reported_state_or_transition": (
                            _reported_state_or_transition(claim, evidence_type)
                        ),
                        "state_classification": target["state_classification"],
                        "direction": direction,
                        "dynamic_regime": dynamic_regime,
                        "condition_linked": condition_linked,
                        "trigger_type": trigger_type,
                        "trigger": trigger,
                        "accepted": accepted,
                        "reason": reason,
                        "quote_verified": quote_verified,
                        "quote": quote,
                    }
                )
    return _collapse_repeated_evidence(rows)


def export_human_readable_contextual(
    tiered_results_path,
    dossier_dir,
    output_tsv,
    accepted_only=True,
):
    """Write a human-readable evidence TSV and return export counts."""
    tiered_rows = _read_jsonl(tiered_results_path)
    dossier_root = Path(dossier_dir)
    dossiers = {}
    for record in tiered_rows:
        custom_id = str(record.get("custom_id") or "")
        path = dossier_root / (custom_id + ".json")
        if not path.is_file():
            raise ValueError("missing dossier file: %s" % path)
        dossiers[custom_id] = json.loads(path.read_text(encoding="utf-8"))
    rows = build_human_readable_rows(
        tiered_rows, dossiers, accepted_only=accepted_only
    )

    destination = Path(output_tsv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=HUMAN_EXPORT_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(
                {
                    key: _visible_control_characters(value)
                    for key, value in row.items()
                }
                for row in rows
            )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    state_counts = {}
    target_classifications = {}
    accepted_rows = [row for row in rows if row["accepted"] == "true"]
    for row in accepted_rows:
        state_counts[row["state_classification"]] = (
            state_counts.get(row["state_classification"], 0) + 1
        )
        target_key = (row["doi"], row["uniprot_accession"])
        previous = target_classifications.setdefault(
            target_key, row["state_classification"]
        )
        if previous != row["state_classification"]:
            raise ValueError("inconsistent target state classification")
    target_counts = {}
    for value in target_classifications.values():
        target_counts[value] = target_counts.get(value, 0) + 1
    return {
        "output_tsv": str(destination.resolve()),
        "row_count": len(rows),
        "accepted_only": bool(accepted_only),
        "accepted_row_count": len(accepted_rows),
        "not_accepted_row_count": len(rows) - len(accepted_rows),
        "state_row_count": sum(
            row["evidence_type"] == "state" for row in accepted_rows
        ),
        "event_row_count": sum(
            row["evidence_type"] == "event" for row in accepted_rows
        ),
        "row_counts_by_state_classification": dict(sorted(state_counts.items())),
        "target_count": len(target_classifications),
        "target_counts_by_state_classification": dict(sorted(target_counts.items())),
    }


__all__ = [
    "HUMAN_EXPORT_COLUMNS",
    "build_human_readable_rows",
    "export_human_readable_contextual",
    "reported_state_classifications",
]
