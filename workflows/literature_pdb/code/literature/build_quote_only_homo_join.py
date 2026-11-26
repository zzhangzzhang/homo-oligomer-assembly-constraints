#!/usr/bin/env python3
"""Directly join homo ConFind scores, PISA, DOI, and quote-only classes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote


SCHEMA_VERSION = "quote_only_homo_direct_join_v1"
CLASS_ORDER = {"Monomer-only", "Multiple states", "Oligomer-only"}
PISA_FIELDS = (
    "diss_energy_kcal_mol",
    "protein_interface_area_A2",
    "unique_hydrophobic_interface_residue_pct",
)
JOIN_FIELDS = (
    "pdb_normalized",
    "uniprot_normalized",
    "doi_normalized",
    "literature_class",
    "target_name",
    "has_doi",
    "exact_target_match",
    "three_class_match",
    "match_status",
    "pisa_status",
    *PISA_FIELDS,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_doi(value: Any) -> str:
    doi = unquote(str(value or "")).strip().strip("\"'").casefold()
    for prefix in ("doi:", "https://doi.org/", "http://doi.org/", "https://dx.doi.org/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return "".join(doi.split()).rstrip(".,;:")


def normalize_pdb(value: Any) -> str:
    return str(value or "").strip().upper()


def normalize_uniprot(value: Any) -> str:
    return str(value or "").strip().upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_index(
    rows: list[dict[str, str]],
    key,
    label: str,
) -> dict[Any, dict[str, str]]:
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise ValueError(f"duplicate {label}: {value}")
        result[value] = row
    return result


def run(
    scores_path: Path,
    pisa_path: Path,
    pdb_to_doi_path: Path,
    target_classifications_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    scores = read_csv(scores_path)
    if not scores:
        raise ValueError("score table is empty")
    score_fields = list(scores[0])
    for required in ("pdb", "uniprot", "intra_recall_topx_qid35"):
        if required not in score_fields:
            raise ValueError(f"score table lacks {required}")
    score_index = unique_index(
        scores,
        lambda row: (normalize_pdb(row["pdb"]), normalize_uniprot(row["uniprot"])),
        "score PDB×UniProt",
    )

    pisa_rows = read_csv(pisa_path)
    pisa_index = unique_index(
        pisa_rows,
        lambda row: (normalize_pdb(row["pdb"]), normalize_uniprot(row["uniprot"])),
        "PISA PDB×UniProt",
    )
    missing_pisa_fields = {"pdb", "uniprot", "status", *PISA_FIELDS} - set(
        pisa_rows[0] if pisa_rows else ()
    )
    if missing_pisa_fields:
        raise ValueError(f"PISA table lacks fields: {sorted(missing_pisa_fields)}")

    raw_doi_map = json.loads(pdb_to_doi_path.read_text(encoding="utf-8"))
    pdb_to_doi = {}
    for raw_pdb, raw_doi in raw_doi_map.items():
        pdb = normalize_pdb(raw_pdb)
        doi = normalize_doi(raw_doi)
        if pdb in pdb_to_doi and pdb_to_doi[pdb] != doi:
            raise ValueError(f"conflicting DOI mapping for PDB {pdb}")
        pdb_to_doi[pdb] = doi

    classification_rows = read_csv(target_classifications_path)
    class_index = unique_index(
        classification_rows,
        lambda row: (
            normalize_doi(row["doi"]),
            normalize_uniprot(row["uniprot_accession"]),
        ),
        "classification DOI×UniProt",
    )

    output_rows = []
    status_counts = Counter()
    for (pdb, uniprot), score in sorted(score_index.items()):
        doi = pdb_to_doi.get(pdb, "")
        classification = class_index.get((doi, uniprot))
        class_name = (
            str(classification.get("state_classification") or "")
            if classification
            else ""
        )
        target_name = (
            str(classification.get("target_name") or "") if classification else ""
        )
        pisa = pisa_index.get((pdb, uniprot))
        pisa_status = str(pisa.get("status") or "") if pisa else ""
        if not doi:
            match_status = "no_doi"
        elif classification:
            match_status = "quote_only_exact_doi_uniprot_match"
        else:
            match_status = "no_quote_only_class"
        row = dict(score)
        row.update(
            {
                "pdb_normalized": pdb,
                "uniprot_normalized": uniprot,
                "doi_normalized": doi,
                "literature_class": class_name,
                "target_name": target_name,
                "has_doi": str(bool(doi)).lower(),
                "exact_target_match": str(bool(classification)).lower(),
                "three_class_match": str(class_name in CLASS_ORDER).lower(),
                "match_status": match_status,
                "pisa_status": pisa_status,
            }
        )
        for field in PISA_FIELDS:
            row[field] = pisa.get(field, "") if pisa else ""
        status_counts[match_status] += 1
        output_rows.append(row)

    output_path = output_dir / "homo_joined_all_score_rows.csv"
    output_fields = score_fields + list(JOIN_FIELDS)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    inputs = (
        scores_path,
        pisa_path,
        pdb_to_doi_path,
        target_classifications_path,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(output_rows),
        "unique_pdb_uniprot_count": len(score_index),
        "pisa_rows_joined": sum(
            (row["pdb_normalized"], row["uniprot_normalized"]) in pisa_index
            for row in output_rows
        ),
        "match_status_counts": dict(sorted(status_counts.items())),
        "three_class_rows": sum(
            row["literature_class"] in CLASS_ORDER for row in output_rows
        ),
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
        },
        "invariants": {
            "score_rows_unique_by_pdb_uniprot": True,
            "pisa_rows_unique_by_pdb_uniprot": True,
            "classifications_unique_by_doi_uniprot": True,
            "raw_score_columns_preserved": score_fields,
            "stale_legacy_evidence_row_count_removed": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--pisa", required=True, type=Path)
    parser.add_argument("--pdb-to-doi", required=True, type=Path)
    parser.add_argument("--target-classifications", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run(
        args.scores.resolve(),
        args.pisa.resolve(),
        args.pdb_to_doi.resolve(),
        args.target_classifications.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
