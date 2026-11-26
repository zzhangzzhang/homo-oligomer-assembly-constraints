#!/usr/bin/env python3
"""Reproduce the GPT-5.4-mini mono/homo denominator and Terra funnel.

This is a read-only audit. It joins immutable structural manifests, the
canonical DOI×UniProt manifest, mini outputs, the DOI-complete Terra cohort,
accepted Terra claims, and the reverse-discordance screen. Derived reports are
written to a new caller-supplied directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dynamic_oligomerization.state_classification import (
    reported_state_classifications,
)
from dynamic_oligomerization.doi import DOIError, normalize_doi


ROOT = Path(
    "/path/to/external-data/literature/dynamic_oligomerization_publication_v1"
)
SCREEN_ROOT = Path("/path/to/external-data/pdb_pairformer_screen")
CANONICAL = ROOT / "full_manifest_publication_v3/manifest.jsonl"
CANONICAL_AUDIT = ROOT / "full_manifest_publication_v3/audit.json"
MINI_RUN = ROOT / "full_batch_v5_20260822_v2"
CONTEXT_INDEX = ROOT / "contextual_reextraction_v4/context_index_v4.sqlite3"
TERRA_RUN = (
    ROOT
    / "contextual_reextraction_v4/contextual_terra_all_v6_2_batch_20260823"
)
TERRA_REBUILD = TERRA_RUN / "rebuilds/tolerant-11e517d13642a7d2"
MONOMER_MANIFEST = SCREEN_ROOT / "full_screen/manifest_monomer.csv"
HOMO_MANIFEST = SCREEN_ROOT / "full_screen/manifest_homo.csv"
PDB_DOI = Path("/path/to/external-data/literature/pdb_to_doi.json")
REVERSE_SCREEN = (
    SCREEN_ROOT
    / "full_analysis/reverse_assembly_discordance/"
    "v5_20260903T124800Z/full_candidate_screen.csv"
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def safe_doi(value: Any) -> str:
    try:
        return normalize_doi(value)
    except (DOIError, TypeError, ValueError):
        return ""


def manifest_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["pdb"].strip().upper(), row["uniprot"].strip().upper())
            for row in csv.DictReader(handle)
        }


def source_category(
    monomer_links: set[tuple[str, str]], homo_links: set[tuple[str, str]]
) -> str:
    if monomer_links and homo_links:
        return "both"
    if monomer_links:
        return "monomer_only"
    if homo_links:
        return "homo_only"
    return "outside"


MONOMER_TERMS = ("monomer",)
OLIGOMER_TERMS = (
    "dimer",
    "trimer",
    "tetramer",
    "pentamer",
    "hexamer",
    "heptamer",
    "octamer",
    "nonamer",
    "decamer",
    "dodecamer",
    "oligomer",
    "multimer",
)


def text_signals(values: Iterable[Any]) -> set[str]:
    text = " ".join(str(value or "") for value in values).casefold()
    signals: set[str] = set()
    if any(term in text for term in MONOMER_TERMS):
        signals.add("monomer")
    if any(term in text for term in OLIGOMER_TERMS):
        signals.add("oligomer")
    return signals


def mini_candidate_class(result: dict[str, Any]) -> str:
    states = list(result.get("state_observations") or [])
    events = list(result.get("dynamic_events") or [])
    if not states and not events:
        return "no_candidate"
    signals: set[str] = set()
    for state in states:
        if state.get("assembly_type") != "discrete_free_homo_oligomer":
            continue
        count = int(state.get("subunit_count") or 0)
        if count == 1:
            signals.add("monomer")
        elif count > 1:
            signals.add("oligomer")
        signals.update(
            text_signals([state.get("state_label"), state.get("population_text")])
        )
    for event in events:
        counts = list(event.get("from_target_subunit_counts") or []) + list(
            event.get("to_target_subunit_counts") or []
        )
        if 1 in counts:
            signals.add("monomer")
        if any(int(value or 0) > 1 for value in counts):
            signals.add("oligomer")
        signals.update(
            text_signals(
                list(event.get("from_states") or [])
                + list(event.get("to_states") or [])
            )
        )
    if signals == {"monomer", "oligomer"}:
        return "both_multiple"
    if signals == {"monomer"}:
        return "monomer_only"
    if signals == {"oligomer"}:
        return "homo_oligomer_only"
    if events and not states:
        return "event_only_other"
    return "other_unknown"


def exact_direct_claim(claim: dict[str, Any]) -> bool:
    if (
        (claim.get("deterministic_tiering") or {}).get("initial_tier")
        != "llm_validated"
    ):
        return False
    if claim.get("claim_origin") != "current_paper_experiment":
        return False
    if (claim.get("citation_scope") or {}).get("status") != "current_paper":
        return False
    evidence = list(claim.get("result_evidence") or [])
    return bool(evidence) and all(
        bool((item.get("verification") or {}).get("verified"))
        and (item.get("verification") or {}).get("match_type")
        in {"literal_exact", "allowlisted_normalized_exact"}
        for item in evidence
    )


def structural_summary(
    pairs: set[tuple[str, str]], pdb_doi: dict[str, str]
) -> dict[str, int]:
    doi_pairs = {
        (pdb_doi.get(pdb, ""), up)
        for pdb, up in pairs
        if pdb_doi.get(pdb, "")
    }
    return {
        "pdb_uniprot_links": len(pairs),
        "unique_pdb_entries": len({pdb for pdb, _ in pairs}),
        "unique_uniprot_accessions": len({up for _, up in pairs}),
        "pdb_entries_with_normalized_doi": len(
            {pdb for pdb, _ in pairs if pdb_doi.get(pdb, "")}
        ),
        "unique_normalized_dois": len({doi for doi, _ in doi_pairs}),
        "unique_doi_uniprot": len(doi_pairs),
    }


def count_card_groups(cards: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(cards)
    return {
        "target_cards": len(rows),
        "unique_dois": len({row["doi"] for row in rows}),
        "unique_uniprot_accessions": len({row["uniprot"] for row in rows}),
        "unique_pdb_entries": len(
            {
                pdb
                for row in rows
                for pdb in row["all_card_pdbs"]
            }
        ),
        "all_card_pdb_uniprot_links": len(
            {
                link
                for row in rows
                for link in row["all_links"]
            }
        ),
        "monomer_manifest_links": len(
            {
                link
                for row in rows
                for link in row["monomer_links"]
            }
        ),
        "homo_manifest_links": len(
            {
                link
                for row in rows
                for link in row["homo_links"]
            }
        ),
    }


def add_metric(
    output: list[dict[str, Any]],
    stage: str,
    source: str,
    classification: str,
    unit: str,
    count: int,
    denominator: int | None = None,
    note: str = "",
) -> None:
    output.append(
        {
            "stage": stage,
            "source": source,
            "classification": classification,
            "unit": unit,
            "count": count,
            "denominator": "" if denominator is None else denominator,
            "rate": (
                ""
                if denominator in (None, 0)
                else f"{count / denominator:.12g}"
            ),
            "note": note,
        }
    )


def run(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    monomer_pairs = manifest_pairs(MONOMER_MANIFEST)
    homo_pairs = manifest_pairs(HOMO_MANIFEST)
    raw_pdb_doi = json.loads(PDB_DOI.read_text())
    pdb_doi = {
        str(pdb).upper(): safe_doi(doi)
        for pdb, doi in raw_pdb_doi.items()
    }

    cards: dict[str, dict[str, Any]] = {}
    for group in read_jsonl(CANONICAL):
        doi = safe_doi(group["doi"])
        up = str(group["uniprot_accession"]).upper()
        all_links = {
            (str(pdb).upper(), up) for pdb in group.get("pdb_ids") or []
        }
        monomer_links = all_links & monomer_pairs
        homo_links = all_links & homo_pairs
        cards[group["group_id"]] = {
            "group_id": group["group_id"],
            "doi": doi,
            "uniprot": up,
            "all_card_pdbs": sorted({pdb for pdb, _ in all_links}),
            "all_links": all_links,
            "monomer_links": monomer_links,
            "homo_links": homo_links,
            "source": source_category(monomer_links, homo_links),
            "mini_response": False,
            "mini_candidate": False,
            "mini_candidate_class": "no_response",
            "terra_submitted": False,
            "terra_accepted_class": "",
            "terra_direct_exact_class": "",
        }

    for row in read_jsonl(MINI_RUN / "parsed_results.jsonl"):
        card = cards[row["group_id"]]
        result = row.get("result") or {}
        candidate_class = mini_candidate_class(result)
        card["mini_response"] = True
        card["mini_candidate_class"] = candidate_class
        card["mini_candidate"] = candidate_class != "no_candidate"

    cohort = json.loads((TERRA_RUN / "cohort_manifest.json").read_text())
    submitted_dois = {safe_doi(row["doi"]) for row in cohort["dossiers"]}
    for card in cards.values():
        card["terra_submitted"] = card["doi"] in submitted_dois

    accepted_class_by_group: dict[str, str] = {}
    direct_class_by_group: dict[str, str] = {}
    for entry in read_jsonl(TERRA_REBUILD / "tiered_results.jsonl"):
        result = entry.get("result") or {}
        accepted_class_by_group.update(reported_state_classifications(result))
        direct_result = dict(result)
        direct_result["state_claims"] = [
            claim
            for claim in result.get("state_claims") or []
            if exact_direct_claim(claim)
        ]
        direct_result["event_claims"] = [
            claim
            for claim in result.get("event_claims") or []
            if exact_direct_claim(claim)
        ]
        direct_class_by_group.update(
            reported_state_classifications(direct_result)
        )
    for group_id, card in cards.items():
        card["terra_accepted_class"] = accepted_class_by_group.get(group_id, "")
        card["terra_direct_exact_class"] = direct_class_by_group.get(group_id, "")

    reverse_rows = {}
    with REVERSE_SCREEN.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            reverse_rows[(row["pdb"].upper(), row["uniprot"].upper())] = row
    for card in cards.values():
        linked = [
            reverse_rows[link]
            for link in card["monomer_links"]
            if link in reverse_rows
        ]
        card["reverse_link_count"] = len(linked)
        card["reverse_direct_exact_positive_links"] = sum(
            row["screen_reason"]
            != "no_accepted_direct_positive_exact_doi_uniprot_evidence"
            for row in linked
        )
        card["reverse_strict_links"] = sum(
            row["screen_pass"].casefold() == "true" for row in linked
        )

    all_cards = list(cards.values())
    source_order = ("monomer_only", "homo_only", "both", "outside")
    mini_classes = (
        "monomer_only",
        "homo_oligomer_only",
        "both_multiple",
        "event_only_other",
        "other_unknown",
    )
    terra_classes = (
        "Monomer-only",
        "Oligomer-only",
        "Multiple states",
        "Unknown",
    )

    source_partition = {
        source: count_card_groups(
            row for row in all_cards if row["source"] == source
        )
        for source in source_order
    }
    mini_by_source: dict[str, Any] = {}
    for source in source_order:
        source_cards = [row for row in all_cards if row["source"] == source]
        candidates = [row for row in source_cards if row["mini_candidate"]]
        candidate_units = count_card_groups(candidates)
        mini_by_source[source] = {
            "input_target_cards": len(source_cards),
            "candidate_target_cards": len(candidates),
            "candidate_rate": len(candidates) / len(source_cards),
            "input_dois": len({row["doi"] for row in source_cards}),
            "candidate_dois": len({row["doi"] for row in candidates}),
            "candidate_units": candidate_units,
            "candidate_content": dict(
                sorted(Counter(row["mini_candidate_class"] for row in candidates).items())
            ),
        }

    linked_rollups = {}
    for label, predicate in {
        "monomer_linked": lambda row: bool(row["monomer_links"]),
        "homo_linked": lambda row: bool(row["homo_links"]),
    }.items():
        group = [row for row in all_cards if predicate(row)]
        candidates = [row for row in group if row["mini_candidate"]]
        linked_rollups[label] = {
            "input_target_cards": len(group),
            "candidate_target_cards": len(candidates),
            "candidate_rate": len(candidates) / len(group),
            "candidate_content": dict(
                sorted(Counter(row["mini_candidate_class"] for row in candidates).items())
            ),
        }

    candidate_dois = {
        row["doi"] for row in all_cards if row["mini_candidate"]
    }
    candidate_pool_cards = [
        row for row in all_cards if row["doi"] in candidate_dois
    ]
    excluded_dois = {
        safe_doi(row["doi"]) for row in cohort["excluded_dossiers"]
    }
    preparation_excluded_cards = [
        row for row in candidate_pool_cards if row["doi"] in excluded_dois
    ]
    submitted = [row for row in all_cards if row["terra_submitted"]]
    terra_partition = {
        source: count_card_groups(
            row for row in submitted if row["source"] == source
        )
        for source in source_order
    }
    for source in source_order:
        source_cards = [row for row in submitted if row["source"] == source]
        terra_partition[source]["cards_with_own_mini_candidate"] = sum(
            row["mini_candidate"] for row in source_cards
        )
        terra_partition[source]["sibling_cards_without_own_candidate"] = sum(
            not row["mini_candidate"] for row in source_cards
        )

    terra_accepted: dict[str, Any] = {}
    terra_direct: dict[str, Any] = {}
    for source in source_order:
        source_cards = [row for row in submitted if row["source"] == source]
        terra_accepted[source] = {
            classification: count_card_groups(
                row
                for row in source_cards
                if row["terra_accepted_class"] == classification
            )
            for classification in terra_classes
        }
        terra_direct[source] = {
            classification: count_card_groups(
                row
                for row in source_cards
                if row["terra_direct_exact_class"] == classification
            )
            for classification in terra_classes
        }

    monomer_submitted = [
        row for row in submitted if bool(row["monomer_links"])
    ]
    reverse = {
        "submitted_monomer_linked_target_cards": len(monomer_submitted),
        "submitted_monomer_pdb_uniprot_links": sum(
            row["reverse_link_count"] for row in monomer_submitted
        ),
        "direct_exact_positive_target_cards": sum(
            row["reverse_direct_exact_positive_links"] > 0
            for row in monomer_submitted
        ),
        "direct_exact_positive_pdb_uniprot_links": sum(
            row["reverse_direct_exact_positive_links"]
            for row in monomer_submitted
        ),
        "strict_target_cards": sum(
            row["reverse_strict_links"] > 0 for row in monomer_submitted
        ),
        "strict_pdb_uniprot_links": sum(
            row["reverse_strict_links"] for row in monomer_submitted
        ),
    }

    homo_monomer_evidence = [
        row
        for row in submitted
        if row["homo_links"]
        and row["terra_direct_exact_class"] in {"Monomer-only", "Multiple states"}
    ]
    opposite_direction = {
        "direct_exact_monomer_or_multiple_target_cards": len(
            homo_monomer_evidence
        ),
        "unique_dois": len({row["doi"] for row in homo_monomer_evidence}),
        "homo_pdb_uniprot_links": len(
            {
                link
                for row in homo_monomer_evidence
                for link in row["homo_links"]
            }
        ),
        "unique_homo_pdb_entries": len(
            {
                pdb
                for row in homo_monomer_evidence
                for pdb, _ in row["homo_links"]
            }
        ),
        "classification": dict(
            sorted(
                Counter(
                    row["terra_direct_exact_class"]
                    for row in homo_monomer_evidence
                ).items()
            )
        ),
        "note": (
            "No homo-side construct/assembly compatibility screen is applied "
            "here; this is not reciprocal to the strict reverse screen."
        ),
    }

    mini_config = json.loads((MINI_RUN / "run_config.json").read_text())
    mini_accounting = json.loads((MINI_RUN / "accounting.json").read_text())
    canonical_audit = json.loads(CANONICAL_AUDIT.read_text())
    report = {
        "schema_version": "mini_mono_homo_funnel_v1",
        "definitions": {
            "target_card": "unique normalized DOI×UniProt canonical group",
            "pdb_uniprot_link": "unique PDB entry paired with its selected UniProt",
            "candidate": (
                "any raw mini state_observations or dynamic_events item; "
                "an unadjudicated retrieval pointer, not accepted evidence"
            ),
            "source_partition": (
                "card-level membership determined by intersecting its PDB×UniProt "
                "links with the two full-screen structural manifests"
            ),
        },
        "inputs": {
            "canonical_manifest": str(CANONICAL),
            "canonical_audit": str(CANONICAL_AUDIT),
            "mini_run": str(MINI_RUN),
            "terra_run": str(TERRA_RUN),
            "monomer_manifest": str(MONOMER_MANIFEST),
            "homo_manifest": str(HOMO_MANIFEST),
            "reverse_screen": str(REVERSE_SCREEN),
        },
        "starting_structural_manifests": {
            "monomer": structural_summary(monomer_pairs, pdb_doi),
            "homo": structural_summary(homo_pairs, pdb_doi),
            "overlap_pdb_uniprot_links": len(monomer_pairs & homo_pairs),
            "overlap_pdb_entries": len(
                {pdb for pdb, _ in monomer_pairs}
                & {pdb for pdb, _ in homo_pairs}
            ),
            "policy": (
                "Upstream labels were parsed from the RCSB structure-page "
                "'Global Stoichiometry' string: values containing Monomer entered "
                "monomers.json and values containing Homo N-mer entered "
                "homooligomers.csv. screen_full_manifest.py then selects one SIFTS "
                "UniProt per PDB, requires a qid35 Pairformer matrix with length "
                "<=500 and an AU mmCIF; homo also requires assembly1 coordinates. "
                "Neither step re-adjudicates every deposited assembly record."
            ),
        },
        "canonical_manifest": {
            "target_cards": len(all_cards),
            "unique_dois": len({row["doi"] for row in all_cards}),
            "source_partition": source_partition,
            "upstream_counts": canonical_audit["counts"],
            "upstream_exclusion_reason_counts": canonical_audit[
                "exclusion_reason_counts"
            ],
        },
        "mini_input_and_output": {
            "model": mini_config["model"],
            "prompt_version": mini_config["prompt_version"],
            "schema_version": mini_config["schema_version"],
            "requests_prepared": mini_config["request_count"],
            "unique_source_files_verified": mini_config[
                "unique_source_files_verified"
            ],
            "accounting": mini_accounting,
            "candidate_target_cards": sum(
                row["mini_candidate"] for row in all_cards
            ),
            "candidate_dois": len(
                candidate_dois
            ),
            "candidate_items": sum(
                1
                for row in read_jsonl(MINI_RUN / "parsed_results.jsonl")
                for _ in (
                    list((row.get("result") or {}).get("state_observations") or [])
                    + list((row.get("result") or {}).get("dynamic_events") or [])
                )
            ),
            "by_source": mini_by_source,
            "linked_rollups": linked_rollups,
        },
        "terra_submission": {
            "candidate_doi_pool": cohort["source_dossier_count"],
            "candidate_doi_pool_target_cards": len(candidate_pool_cards),
            "candidate_doi_pool_source_partition": {
                source: count_card_groups(
                    row
                    for row in candidate_pool_cards
                    if row["source"] == source
                )
                for source in source_order
            },
            "preparation_exclusions": cohort["excluded_dossier_count"],
            "preparation_excluded_target_cards": len(preparation_excluded_cards),
            "preparation_excluded_source_partition": dict(
                sorted(
                    Counter(
                        row["source"] for row in preparation_excluded_cards
                    ).items()
                )
            ),
            "submitted_dois": cohort["doi_count"],
            "submitted_target_cards": len(submitted),
            "submitted_cards_with_own_mini_candidate": sum(
                row["mini_candidate"] for row in submitted
            ),
            "submitted_sibling_cards_without_own_candidate": sum(
                not row["mini_candidate"] for row in submitted
            ),
            "source_partition": terra_partition,
        },
        "terra_accepted_target_classifications": {
            "all_llm_validated_claims_by_source": terra_accepted,
            "direct_current_paper_exact_claims_by_source": terra_direct,
            "classification_policy": (
                "Per target, monomer/oligomer signals are combined across accepted "
                "state claims and event endpoints; multiple means both signals."
            ),
        },
        "downstream_reverse_discordance": reverse,
        "opposite_homo_to_monomer_evidence_intersection": opposite_direction,
    }

    with (output_dir / "funnel_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    card_fields = [
        "group_id",
        "doi",
        "uniprot",
        "source",
        "all_pdb_count",
        "monomer_link_count",
        "homo_link_count",
        "mini_response",
        "mini_candidate",
        "mini_candidate_class",
        "terra_submitted",
        "terra_accepted_class",
        "terra_direct_exact_class",
        "reverse_link_count",
        "reverse_direct_exact_positive_links",
        "reverse_strict_links",
    ]
    with (output_dir / "target_card_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=card_fields)
        writer.writeheader()
        for row in sorted(all_cards, key=lambda item: item["group_id"]):
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in card_fields},
                    "all_pdb_count": len(row["all_card_pdbs"]),
                    "monomer_link_count": len(row["monomer_links"]),
                    "homo_link_count": len(row["homo_links"]),
                }
            )

    metrics: list[dict[str, Any]] = []
    for source in source_order:
        total = mini_by_source[source]["input_target_cards"]
        add_metric(metrics, "mini_input", source, "all", "target_card", total)
        add_metric(
            metrics,
            "mini_output",
            source,
            "any_candidate",
            "target_card",
            mini_by_source[source]["candidate_target_cards"],
            total,
        )
        for classification in mini_classes:
            add_metric(
                metrics,
                "mini_output",
                source,
                classification,
                "target_card",
                mini_by_source[source]["candidate_content"].get(classification, 0),
                total,
            )
        submitted_source = [
            row for row in submitted if row["source"] == source
        ]
        add_metric(
            metrics,
            "terra_submission",
            source,
            "all",
            "target_card",
            len(submitted_source),
        )
        for classification in terra_classes:
            add_metric(
                metrics,
                "terra_accepted",
                source,
                classification,
                "target_card",
                terra_accepted[source][classification]["target_cards"],
                len(submitted_source),
            )
            add_metric(
                metrics,
                "terra_direct_current_exact",
                source,
                classification,
                "target_card",
                terra_direct[source][classification]["target_cards"],
                len(submitted_source),
            )
    with (output_dir / "metric_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
