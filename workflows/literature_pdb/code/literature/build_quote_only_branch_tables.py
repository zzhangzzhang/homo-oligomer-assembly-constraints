#!/usr/bin/env python3
"""Build quote-only branch composition without legacy enrichment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


CLASSES = ("Monomer-only", "Multiple states", "Oligomer-only", "Unknown")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def run(
    target_classifications: Path,
    target_card_audit: Path,
    base_funnel: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    classifications = read_csv(target_classifications)
    by_group = {}
    for row in classifications:
        group_id = row["group_id"]
        if group_id in by_group:
            raise ValueError(f"duplicate target classification: {group_id}")
        by_group[group_id] = row["state_classification"]
    cards = read_csv(target_card_audit)
    if len({row["group_id"] for row in cards}) != len(cards):
        raise ValueError("duplicate group_id in target card audit")

    composition = []
    for branch, link_field in (
        ("monomer_linked", "monomer_link_count"),
        ("homo_linked", "homo_link_count"),
    ):
        submitted = [
            row
            for row in cards
            if truth(row["terra_submitted"]) and int(row[link_field]) > 0
        ]
        submitted_keys = len(submitted)
        submitted_records = sum(int(row[link_field]) for row in submitted)
        classified_keys = 0
        classified_records = 0
        for class_name in CLASSES:
            selected = [
                row
                for row in submitted
                if by_group.get(row["group_id"], "") == class_name
            ]
            keys = len(selected)
            records = sum(int(row[link_field]) for row in selected)
            classified_keys += keys
            classified_records += records
            composition.append(
                {
                    "branch": branch,
                    "direct_exact_class": class_name,
                    "unique_doi_uniprot_keys": keys,
                    "unique_pdb_uniprot_records": records,
                    "submitted_doi_uniprot_denominator": submitted_keys,
                    "submitted_pdb_uniprot_denominator": submitted_records,
                    "note": "quote-only rebuild; includes both-source cards/links",
                }
            )
        composition.append(
            {
                "branch": branch,
                "direct_exact_class": "No direct exact accepted class",
                "unique_doi_uniprot_keys": submitted_keys - classified_keys,
                "unique_pdb_uniprot_records": submitted_records - classified_records,
                "submitted_doi_uniprot_denominator": submitted_keys,
                "submitted_pdb_uniprot_denominator": submitted_records,
                "note": "submitted branch membership minus four classifications",
            }
        )

    composition_path = output_dir / "terra_direct_exact_class_composition.csv"
    with composition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(composition[0]))
        writer.writeheader()
        writer.writerows(composition)
    funnel_path = output_dir / "side_by_side_funnel.csv"
    shutil.copy2(base_funnel, funnel_path)

    inputs = (target_classifications, target_card_audit, base_funnel)
    summary = {
        "schema_version": "quote_only_branch_tables_v1",
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "outputs": {
            composition_path.name: sha256(composition_path),
            funnel_path.name: sha256(funnel_path),
        },
        "composition": composition,
        "legacy_enrichment_dependency": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-classifications", required=True, type=Path)
    parser.add_argument("--target-card-audit", required=True, type=Path)
    parser.add_argument("--base-funnel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run(
        args.target_classifications.resolve(),
        args.target_card_audit.resolve(),
        args.base_funnel.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
