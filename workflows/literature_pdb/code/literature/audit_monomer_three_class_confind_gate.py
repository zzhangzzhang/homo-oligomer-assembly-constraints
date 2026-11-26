#!/usr/bin/env python3
"""Audit ConFind/qid35 readiness for monomer-linked three-class records.

This read-only audit reconstructs exact canonical card membership, checks
existing qid35 Pairformer matrices, compares against the existing homo-specific
ConFind score universe, and independently reapplies the deposited-assembly
monomer annotation. It does not generate ConFind contacts or run inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dynamic_oligomerization.doi import normalize_doi
from reverse_assembly_helpers import (
    assembly_monomer_decision,
    assembly_rows_from_block,
    load_cif_block,
)


ROOT = Path("/path/to/external-data/pdb_pairformer_screen")
LIT = Path("/path/to/external-data/literature/dynamic_oligomerization_publication_v1")
MONOMER_MANIFEST = ROOT / "full_screen/manifest_monomer.csv"
CANONICAL = LIT / "full_manifest_publication_v3/manifest.jsonl"
CARD_AUDIT = (
    ROOT
    / "full_analysis/denominator_audit/"
    "mini_mono_homo_funnel_v3_20260903/target_card_audit.csv"
)
HOMO_CONFIND_SCORES = (
    ROOT
    / "confind_bio_assembly1/full_analysis/20260823/"
    "pairformer_qid15_qid35_summary.csv"
)
HOMO_CONFIND_MANIFEST = ROOT / "full_screen/manifest_homo.csv"
HOMO_CONFIND_CONTACTS = ROOT / "confind_bio_assembly1/contacts"
CLASSES = ("Monomer-only", "Oligomer-only", "Multiple states")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    output_dir: Path,
    target_classifications: Path | None = None,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    monomer = {
        (row["pdb"].upper(), row["uniprot"].upper()): row
        for row in read_csv(MONOMER_MANIFEST)
    }
    homo_pairs = {
        (row["pdb"].upper(), row["uniprot"].upper())
        for row in read_csv(HOMO_CONFIND_MANIFEST)
    }
    if set(monomer) & homo_pairs:
        raise RuntimeError("monomer/homo manifest PDB×UniProt overlap is nonzero")
    card_audit = {row["group_id"]: row for row in read_csv(CARD_AUDIT)}
    revised_classes = (
        {
            row["group_id"]: row["state_classification"]
            for row in read_csv(target_classifications)
        }
        if target_classifications is not None
        else {}
    )
    confind = {
        (row["pdb"].upper(), row["uniprot"].upper()): row
        for row in read_csv(HOMO_CONFIND_SCORES)
    }

    records: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for group in read_jsonl(CANONICAL):
        audit = card_audit[group["group_id"]]
        class_name = (
            revised_classes.get(group["group_id"], "")
            if target_classifications is not None
            else audit["terra_direct_exact_class"]
        )
        if class_name not in CLASSES:
            continue
        doi = normalize_doi(group["doi"])
        uniprot = group["uniprot_accession"].upper()
        exact_links = {
            (pdb.upper(), uniprot) for pdb in group["pdb_ids"]
        } & set(monomer)
        for pdb, up in exact_links:
            prior = seen.get((pdb, up))
            identity = (doi, class_name)
            if prior is not None and prior != identity:
                raise RuntimeError(
                    f"{pdb}×{up} maps to multiple DOI/class identities: "
                    f"{prior}, {identity}"
                )
            seen[(pdb, up)] = identity
            manifest = monomer[(pdb, up)]
            matrix_path = Path(manifest["pf_path"])
            score = confind.get((pdb, up))
            intra_contact_path = (
                HOMO_CONFIND_CONTACTS / f"{pdb.lower()}-assembly1_intra.npy"
            )
            inter_contact_path = (
                HOMO_CONFIND_CONTACTS / f"{pdb.lower()}-assembly1_inter.npy"
            )
            try:
                decision = assembly_monomer_decision(
                    assembly_rows_from_block(
                        load_cif_block(manifest["au_cif_path"])
                    )
                )
                assembly_error = ""
            except Exception as error:
                decision = {
                    "strict_monomer": False,
                    "reason": f"mmcif_error:{type(error).__name__}",
                    "assembly_count": 0,
                    "assembly_ids": [],
                    "oligomeric_counts": [],
                }
                assembly_error = str(error)
            records.append(
                {
                    "pdb": pdb,
                    "uniprot": up,
                    "doi": doi,
                    "doi_uniprot_key": f"{doi}|{up}",
                    "direct_current_paper_exact_class": class_name,
                    "canonical_group_id": group["group_id"],
                    "exact_card_pdb_membership": True,
                    "qid35_matrix_path": str(matrix_path),
                    "qid35_matrix_exists": matrix_path.exists(),
                    "existing_confind_intra_contact_map": intra_contact_path.exists(),
                    "existing_confind_inter_contact_map": inter_contact_path.exists(),
                    "existing_confind_score": score is not None,
                    "existing_intra_recall_topx_qid35": (
                        score["intra_recall_topx_qid35"] if score else ""
                    ),
                    "intra_recall_gt_0_4_gate": (
                        (
                            float(score["intra_recall_topx_qid35"]) > 0.4
                            if score["intra_recall_topx_qid35"]
                            else False
                        )
                        if score
                        else "not_computable_missing_confind"
                    ),
                    "strict_all_deposited_assemblies_monomer": decision[
                        "strict_monomer"
                    ],
                    "assembly_reason": decision["reason"],
                    "assembly_count": decision["assembly_count"],
                    "assembly_ids": ";".join(decision["assembly_ids"]),
                    "assembly_oligomeric_counts": ";".join(
                        "" if value is None else str(value)
                        for value in decision["oligomeric_counts"]
                    ),
                    "assembly_read_error": assembly_error,
                    "construct_filter_applied": False,
                    "next_analysis_endpoint": (
                        "outcome_blind_CA_prescreen_then_isolated_dimer_"
                        "ConFind_qid35_scoring"
                    ),
                }
            )

    records.sort(
        key=lambda row: (
            CLASSES.index(row["direct_current_paper_exact_class"]),
            row["doi"],
            row["uniprot"],
            row["pdb"],
        )
    )
    pair_keys = {(row["pdb"], row["uniprot"]) for row in records}
    if len(records) != len(pair_keys):
        raise RuntimeError("duplicate PDB×UniProt records")
    if target_classifications is None and len(records) != 2443:
        raise RuntimeError(f"expected 2,443 records, found {len(records)}")
    if any(not row["qid35_matrix_exists"] for row in records):
        raise RuntimeError("qid35 matrix coverage is incomplete")

    class_rows: list[dict[str, Any]] = []
    for class_name in CLASSES:
        subset = [
            row
            for row in records
            if row["direct_current_paper_exact_class"] == class_name
        ]
        scored = [row for row in subset if row["existing_confind_score"]]
        passed = [
            row
            for row in scored
            if row["intra_recall_gt_0_4_gate"] is True
        ]
        strict = [
            row
            for row in subset
            if row["strict_all_deposited_assemblies_monomer"]
        ]
        failed = [
            row
            for row in subset
            if row["assembly_reason"]
            == "at_least_one_deposited_assembly_nonmonomeric"
        ]
        class_rows.append(
            {
                "literature_class": class_name,
                "unique_doi_uniprot_keys": len(
                    {(row["doi"], row["uniprot"]) for row in subset}
                ),
                "unique_pdb_uniprot_records": len(subset),
                "unique_uniprot_accessions": len(
                    {row["uniprot"] for row in subset}
                ),
                "unique_qid35_matrix_files": len(
                    {row["qid35_matrix_path"] for row in subset}
                ),
                "qid35_covered_records": sum(
                    row["qid35_matrix_exists"] for row in subset
                ),
                "existing_confind_scored_records": len(scored),
                "existing_confind_intra_contact_maps": sum(
                    row["existing_confind_intra_contact_map"] for row in subset
                ),
                "intra_recall_gt_0_4_records": (
                    len(passed) if scored else "not_computable"
                ),
                "strict_monomer_assembly_records": len(strict),
                "explicit_nonmonomeric_assembly_records": len(failed),
            }
        )

    all_uniprots = {row["uniprot"] for row in records}
    all_keys = {(row["doi"], row["uniprot"]) for row in records}
    assembly_reasons = Counter(row["assembly_reason"] for row in records)
    class_assembly = defaultdict(Counter)
    for row in records:
        class_assembly[row["direct_current_paper_exact_class"]][
            row["assembly_reason"]
        ] += 1

    anchor = {
        row["literature_class"]: (
            row["unique_doi_uniprot_keys"],
            row["unique_pdb_uniprot_records"],
            row["unique_uniprot_accessions"],
        )
        for row in class_rows
    }
    expected_key_record = {
        "Monomer-only": (923, 1406),
        "Oligomer-only": (268, 442),
        "Multiple states": (415, 595),
    }
    if target_classifications is None and {
        key: values[:2] for key, values in anchor.items()
    } != expected_key_record:
        raise RuntimeError(f"class anchor mismatch: {anchor}")
    if target_classifications is None and len(all_keys) != 1606:
        raise RuntimeError(f"expected 1,606 unique DOI×UniProt keys, found {len(all_keys)}")

    write_csv(output_dir / "monomer_three_class_records.csv", records)
    write_csv(output_dir / "class_filter_accounting.csv", class_rows)
    assembly_rows = [
        {
            "literature_class": class_name,
            "assembly_reason": reason,
            "unique_pdb_uniprot_records": count,
        }
        for class_name in CLASSES
        for reason, count in sorted(class_assembly[class_name].items())
    ]
    write_csv(output_dir / "assembly_recheck_summary.csv", assembly_rows)

    inputs = {
        "canonical_manifest": CANONICAL,
        "target_card_audit": CARD_AUDIT,
        "monomer_manifest": MONOMER_MANIFEST,
        "homo_manifest": HOMO_CONFIND_MANIFEST,
        "existing_homo_confind_scores": HOMO_CONFIND_SCORES,
    }
    if target_classifications is not None:
        inputs["target_classifications"] = target_classifications
    audit = {
        "schema_version": "monomer_three_class_confind_gate_v1",
        "definitions": {
            "class_policy": (
                "mutually exclusive DOI×UniProt direct current-paper exact "
                "Monomer-only, Oligomer-only, or Multiple states"
            ),
            "exact_membership": (
                "PDB×UniProt must occur in both the canonical card PDB list "
                "and manifest_monomer.csv"
            ),
            "confind_top_x": (
                "for each retained isolated dimer, X=|intrachain ConFind ∪ "
                "interface ConFind| at UniProt separation >=6; C-alpha "
                "geometry prescreens candidates but does not define the final "
                "scored contacts"
            ),
            "figure_gate_to_mirror": (
                "three-class label and qid35 intra_recall_topx >0.4 only; "
                "do not add n_bio_confind>=50"
            ),
            "monomer_endpoint": (
                "outcome-blind C-alpha geometry prescreens individual ASU/XTAL "
                "candidate dimers; full-atom isolated-dimer ConFind contacts "
                "then define the qid35 intrachain-quality and interface-recovery "
                "endpoints"
            ),
        },
        "counts": {
            "unique_doi_uniprot_keys": len(all_keys),
            "unique_pdb_uniprot_records": len(records),
            "unique_uniprot_accessions": len(all_uniprots),
            "qid35_covered_records": sum(
                row["qid35_matrix_exists"] for row in records
            ),
            "unique_qid35_matrix_files": len(
                {row["qid35_matrix_path"] for row in records}
            ),
            "existing_confind_scored_records": sum(
                row["existing_confind_score"] for row in records
            ),
            "existing_confind_intra_contact_maps": sum(
                row["existing_confind_intra_contact_map"] for row in records
            ),
            "existing_confind_inter_contact_maps": sum(
                row["existing_confind_inter_contact_map"] for row in records
            ),
            "intra_recall_gt_0_4_records": "not_computable_missing_confind",
            "assembly_recheck_reasons": dict(sorted(assembly_reasons.items())),
        },
        "by_class": class_rows,
        "validation": {
            "records_unique": len(records) == len(pair_keys),
            "classes_mutually_exclusive": True,
            "exact_card_membership_all_records": True,
            "qid35_matrix_coverage_complete": True,
            "monomer_homo_manifest_pair_overlap": 0,
            "existing_homo_confind_score_overlap": sum(
                row["existing_confind_score"] for row in records
            ),
            "existing_homo_confind_contact_map_overlap": sum(
                row["existing_confind_intra_contact_map"] for row in records
            ),
        },
        "inputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in inputs.items()
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    methods = f"""# Monomer-linked three-class ConFind/qid35 gate audit

## Result

The starting cohort has {len(records):,} unique exact-card-member PDB×UniProt records
representing {len(all_keys):,} DOI×UniProt paper–protein keys. Every record has its existing
qid35 Pairformer matrix, so Pairformer inference need not be rerun.

The current ConFind contact maps and score table were generated from the
disjoint homo manifest. They cover {sum(row["existing_confind_score"] for row in records):,}/{len(records):,} monomer-linked records.
Consequently, the exact number passing `intra_recall_topx_qid35 > 0.4` is not
yet computable. Existing historical CACA intrachain contacts are not
substituted for ConFind.

## Policy to mirror

The final homo plotting script selects only a three-class literature label and
strict qid35 `intra_recall_topx > 0.4`. The older v2 enrichment analysis also
requires `n_bio_confind >= 50` and complete biological-interface metrics. The
final plotting policy is the matched quality gate.

For monomer structures, ConFind intrachain recovery is a Pairformer-quality
gate. Cα geometry is used only for outcome-blind ASU/XTAL candidate-dimer
prescreening. Each retained candidate is rebuilt as a full-atom, two-chain P1
dimer; ConFind then defines its intrachain and interface contacts, and qid35 is
scored at candidate-specific X=|intra union interface|. Interface recovery is
therefore measured against ConFind contacts rather than Cα contacts, followed
by same-interface PISA and condition analysis.

The deposited-assembly policy was independently reevaluated for every record
and is retained as an annotation here; construct compatibility was not used.
"""
    (output_dir / "METHODS_AND_NEXT_STEP.md").write_text(methods, encoding="utf-8")
    output_hashes = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.name != "output_checksums.json"
    }
    (output_dir / "output_checksums.json").write_text(
        json.dumps(output_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-classifications", type=Path)
    args = parser.parse_args()
    run(
        args.output_dir.resolve(),
        (
            args.target_classifications.resolve()
            if args.target_classifications is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
