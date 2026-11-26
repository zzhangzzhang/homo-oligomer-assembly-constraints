#!/usr/bin/env python3
"""Build the PDB-level monomer scatter table from reusable ConFind/PISA jobs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLASS_ORDER = ("Monomer-only", "Multiple states", "Oligomer-only")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pisa(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_csv(path):
            interface_id = row["interface_id"]
            if interface_id in rows and rows[interface_id] != row:
                raise RuntimeError(f"conflicting PISA row: {interface_id}")
            rows[interface_id] = row
    return rows


def run(
    target_classifications: Path,
    confind_roots: list[Path],
    pisa_summaries: list[Path],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    classes = {
        row["group_id"]: row["state_classification"]
        for row in read_csv(target_classifications)
    }
    pisa = load_pisa(pisa_summaries)

    eligible: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    manifest_jobs = 0
    missing_done = []
    for root in confind_roots:
        manifest = read_csv(root / "candidate_manifest_primary.csv")
        manifest_jobs += len(manifest)
        for row in manifest:
            class_name = classes.get(row["canonical_group_id"], "")
            if class_name not in CLASS_ORDER:
                continue
            done_path = root / "jobs" / row["interface_id"] / "done.json"
            if not done_path.is_file():
                missing_done.append(row["interface_id"])
                continue
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("status") != "success":
                continue
            if float(done.get("intra_recall_at_candidate_x") or 0) <= 0.4:
                continue
            if int(done.get("n_interface_confind_pairs") or 0) < 10:
                continue
            done["direct_current_paper_exact_class"] = class_name
            eligible[(str(done["pdb"]), str(done["uniprot"]))].append(done)
    if missing_done:
        raise RuntimeError(
            f"{len(missing_done)} ConFind jobs are incomplete; "
            f"first={missing_done[0]}"
        )

    output_rows = []
    for (pdb, uniprot), candidates in sorted(eligible.items()):
        selected = sorted(
            candidates,
            key=lambda row: (
                -float(row["interface_recall"]),
                -int(row["interface_hits"]),
                -int(row["n_interface_confind_pairs"]),
                str(row["interface_id"]),
            ),
        )[0]
        pisa_row = pisa.get(str(selected["interface_id"]), {})
        pisa_status = str(pisa_row.get("status") or "")
        dissociation = (
            pisa_row.get("dissociation_energy_kcal_mol", "")
            if pisa_status == "success"
            else ""
        )
        area = (
            pisa_row.get("protein_interface_area_A2", "")
            if pisa_status == "success"
            else ""
        )
        output_rows.append(
            {
                "pdb": pdb,
                "uniprot": uniprot,
                "doi": selected["doi"],
                "doi_uniprot_key": selected["doi_uniprot_key"],
                "canonical_group_id": selected["canonical_group_id"],
                "direct_current_paper_exact_class": selected[
                    "direct_current_paper_exact_class"
                ],
                "primary_candidate_count_ge10": len(candidates),
                "best_recall_interface_id": selected["interface_id"],
                "best_recall_interface_recall": selected["interface_recall"],
                "best_recall_interface_hits": selected["interface_hits"],
                "best_recall_confind_inter_pairs": selected[
                    "n_interface_confind_pairs"
                ],
                "best_recall_intra_recall_at_candidate_x": selected[
                    "intra_recall_at_candidate_x"
                ],
                "best_recall_pisa_status": pisa_status or "not_run",
                "best_recall_pisa_dissociation_energy_kcal_mol": dissociation,
                "best_recall_pisa_protein_interface_area_A2": area,
                "best_pisa_dissociation_energy_kcal_mol": dissociation,
                "best_pisa_protein_interface_area_A2": area,
            }
        )
    table_path = output_dir / "pdb_best_interfaces.csv"
    write_csv(table_path, output_rows)
    summary = {
        "schema_version": "quote_only_monomer_pdb_table_v1",
        "manifest_jobs": manifest_jobs,
        "eligible_pdb_uniprot_records": len(output_rows),
        "eligible_by_class": dict(
            Counter(
                row["direct_current_paper_exact_class"] for row in output_rows
            )
        ),
        "same_interface_pisa_records": sum(
            row["best_recall_pisa_status"] == "success" for row in output_rows
        ),
        "same_interface_pisa_by_class": dict(
            Counter(
                row["direct_current_paper_exact_class"]
                for row in output_rows
                if row["best_recall_pisa_status"] == "success"
            )
        ),
        "inputs": {
            "target_classifications": str(target_classifications),
            "confind_roots": [str(path) for path in confind_roots],
            "pisa_summaries": [str(path) for path in pisa_summaries],
        },
        "output": str(table_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-classifications", type=Path, required=True)
    parser.add_argument("--confind-root", type=Path, action="append", required=True)
    parser.add_argument("--pisa-summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.target_classifications,
        args.confind_root,
        args.pisa_summary,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
