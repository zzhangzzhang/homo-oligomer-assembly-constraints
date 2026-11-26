#!/usr/bin/env python3
"""Build EPPIC/model-scaling and Pairformer-QID contact-recovery tables.

The reported quantity is class-specific recovery among the top-X predictions,
where X is the number of true inter-or-intra contacts. It is not conventional
precision at X.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEQSEP_K = 6
DEFAULT_EXCLUSIONS = ("2w6a", "2fgz")
MODEL_SPECS = (
    ("esm2_8m", "ESM2", 8, "esm2", "esm2_8m"),
    ("esm2_35m", "ESM2", 35, "esm2", "esm2_35m"),
    ("esm2_150m", "ESM2", 150, "esm2", "esm2_150m"),
    ("esm2_650m", "ESM2", 650, "esm2", "esm2_650m"),
    ("esm2_3b", "ESM2", 3_000, "esm2", "esm2_3b"),
    ("esm2_15b", "ESM2", 15_000, "esm2", "esm2_15b"),
    (
        "esmc_300m_refit",
        "ESM-C refit",
        300,
        "esmc",
        "esmc300m_overlay_17_liblinear",
    ),
    (
        "esmc_600m_refit",
        "ESM-C refit",
        600,
        "esmc",
        "esmc600m_overlay_17_liblinear",
    ),
    (
        "esmc_6b_refit",
        "ESM-C refit",
        6_000,
        "esmc",
        "esmc6b_overlay_17_liblinear",
    ),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eppic-root",
        type=Path,
        required=True,
        help="EPPIC root containing lists, labels, masks, and official ESM2 maps.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Root containing all release prediction directories.",
    )
    parser.add_argument(
        "--esm2-prediction-root",
        type=Path,
        help=(
            "Optional separate root for official ESM2 maps; defaults to "
            "--prediction-root."
        ),
    )
    parser.add_argument(
        "--pairformer-csv",
        type=Path,
        help="Pairformer QID evaluation CSV (default: EPPIC root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUSIONS),
        help="Case-insensitive PDB IDs excluded from every model and panel.",
    )
    return parser.parse_args()


def load_ids(path: Path) -> list[str]:
    with path.open() as handle:
        return [str(row[0]).lower() for row in json.load(handle)]


def valid_pair_mask(residue_mask: np.ndarray, length: int) -> np.ndarray:
    resolved = np.outer(residue_mask.astype(bool), residue_mask.astype(bool))
    sequence_separation = np.triu(
        np.ones((length, length), dtype=bool), k=SEQSEP_K
    )
    return resolved & sequence_separation


def evaluate_map(
    pdb_id: str,
    prediction_path: Path,
    contact_dir: Path,
    mask_dir: Path,
) -> dict[str, float | int]:
    inter = np.load(contact_dir / f"{pdb_id}-assembly1_inter.npy")
    intra = np.load(contact_dir / f"{pdb_id}-assembly1_intra.npy")
    residue_mask = np.load(mask_dir / f"{pdb_id}-assembly1_mask.npy")
    prediction = np.load(prediction_path)

    length = inter.shape[0]
    expected = (length, length)
    if intra.shape != expected or prediction.shape != expected:
        raise ValueError(
            f"{pdb_id}: target/prediction shapes are "
            f"{inter.shape}/{intra.shape}/{prediction.shape}"
        )
    if residue_mask.shape != (length,):
        raise ValueError(
            f"{pdb_id}: residue mask {residue_mask.shape}, expected {(length,)}"
        )

    valid = valid_pair_mask(residue_mask, length)
    inter_only = (inter == 1) & (intra == 0) & valid
    intra_only = (intra == 1) & (inter == 0) & valid
    either = ((inter == 1) | (intra == 1)) & valid
    top_x = int(either.sum())
    if top_x == 0:
        raise ValueError(f"{pdb_id}: no valid ground-truth contacts")

    scores = np.where(valid, prediction, -np.inf)
    flat_indices = np.argpartition(scores.ravel(), -top_x)[-top_x:]
    row_indices, column_indices = np.unravel_index(flat_indices, scores.shape)

    n_inter = int(inter_only.sum())
    n_intra = int(intra_only.sum())
    hits_inter = int(inter_only[row_indices, column_indices].sum())
    hits_intra = int(intra_only[row_indices, column_indices].sum())
    return {
        "L": length,
        "top_x": top_x,
        "n_inter_only": n_inter,
        "n_intra_only": n_intra,
        "inter_recovery": hits_inter / n_inter if n_inter else np.nan,
        "intra_recovery": hits_intra / n_intra if n_intra else np.nan,
    }


def prediction_directory(
    source: str,
    directory: str,
    prediction_root: Path,
    esm2_prediction_root: Path,
) -> Path:
    root = esm2_prediction_root if source == "esm2" else prediction_root
    return root / directory


def build_contact_table(
    eppic_root: Path,
    prediction_root: Path,
    esm2_prediction_root: Path,
    sets: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    contact_dir = eppic_root / "contact_matrices"
    mask_dir = eppic_root / "mask"

    for set_name, pdb_ids in sets.items():
        for pdb_id in pdb_ids:
            for model, family, size_m, source, directory in MODEL_SPECS:
                pred_dir = prediction_directory(
                    source, directory, prediction_root, esm2_prediction_root
                )
                pred_path = pred_dir / f"{pdb_id}.npy"
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Missing prediction: {pred_path}")
                result = evaluate_map(pdb_id, pred_path, contact_dir, mask_dir)
                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "set": set_name,
                        "model": model,
                        "family": family,
                        "size_m": size_m,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def summarize_contacts(per_pdb: pd.DataFrame) -> pd.DataFrame:
    return (
        per_pdb.groupby(
            ["set", "model", "family", "size_m"], as_index=False, sort=False
        )
        .agg(
            n=("pdb_id", "nunique"),
            mean_inter_recovery=("inter_recovery", "mean"),
            mean_intra_recovery=("intra_recovery", "mean"),
        )
        .sort_values(["family", "size_m", "set"])
        .reset_index(drop=True)
    )


def curate_pairformer(
    csv_path: Path,
    sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    required = {"pdb", "qid", "base_inter", "base_intra"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} lacks columns: {sorted(missing)}")

    set_by_pdb = {
        pdb_id: set_name for set_name, pdb_ids in sets.items() for pdb_id in pdb_ids
    }
    frame["pdb_id"] = frame["pdb"].astype(str).str.lower()
    frame = frame[frame["pdb_id"].isin(set_by_pdb)].copy()
    frame["set"] = frame["pdb_id"].map(set_by_pdb)
    frame = frame.rename(
        columns={
            "base_inter": "inter_recovery",
            "base_intra": "intra_recovery",
        }
    )
    keep = [
        "pdb_id",
        "set",
        "qid",
        "inter_recovery",
        "intra_recovery",
        "n_inter",
        "rand_inter_mean",
        "rand_inter_max",
        "rand_intra_mean",
        "rand_intra_max",
    ]
    frame = frame[keep].sort_values(["set", "pdb_id", "qid"]).reset_index(drop=True)

    qids = sorted(frame["qid"].unique())
    expected_rows = sum(len(ids) for ids in sets.values()) * len(qids)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Pairformer table has {len(frame)} rows; expected {expected_rows} "
            f"for {len(qids)} QID cutoffs and the common cohort"
        )
    coverage = frame.groupby(["set", "qid"])["pdb_id"].nunique()
    for set_name, pdb_ids in sets.items():
        if not (coverage.loc[set_name] == len(pdb_ids)).all():
            raise ValueError(f"Incomplete Pairformer coverage for {set_name}")

    summary = (
        frame.groupby(["set", "qid"], as_index=False)
        .agg(
            n=("pdb_id", "nunique"),
            mean_inter_recovery=("inter_recovery", "mean"),
            mean_intra_recovery=("intra_recovery", "mean"),
        )
        .sort_values(["set", "qid"])
        .reset_index(drop=True)
    )
    return frame, summary


def main() -> None:
    args = parse_args()
    eppic_root = args.eppic_root.resolve()
    prediction_root = args.prediction_root.resolve()
    esm2_prediction_root = (
        args.esm2_prediction_root.resolve()
        if args.esm2_prediction_root
        else prediction_root
    )
    pairformer_csv = (
        args.pairformer_csv.resolve()
        if args.pairformer_csv
        else eppic_root / "qid_vs_random_eval_masked.csv"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = {pdb_id.lower() for pdb_id in args.exclude}
    sets = {
        "bio": [
            pdb_id
            for pdb_id in load_ids(eppic_root / "DCbio_clean.json")
            if pdb_id not in excluded
        ],
        "xtal": [
            pdb_id
            for pdb_id in load_ids(eppic_root / "DCxtal_clean.json")
            if pdb_id not in excluded
        ],
    }
    expected_counts = {"bio": 40, "xtal": 74}
    actual_counts = {name: len(ids) for name, ids in sets.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected cohort counts: {actual_counts}")

    contact_table = build_contact_table(
        eppic_root, prediction_root, esm2_prediction_root, sets
    )
    contact_summary = summarize_contacts(contact_table)
    pairformer_table, pairformer_summary = curate_pairformer(pairformer_csv, sets)

    contact_table.to_csv(output_dir / "contact_scaling_per_pdb.csv", index=False)
    contact_summary.to_csv(output_dir / "contact_scaling_summary.csv", index=False)
    pairformer_table.to_csv(output_dir / "pairformer_qid_per_pdb.csv", index=False)
    pairformer_summary.to_csv(
        output_dir / "pairformer_qid_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "pdb_id": "2w6a",
                "reason": "Excluded globally by the analysis specification.",
            },
            {
                "pdb_id": "2fgz",
                "reason": (
                    "Excluded globally: target length 1083, but the available "
                    "Pairformer prediction is truncated to 1024."
                ),
            },
        ]
    ).to_csv(output_dir / "exclusions.csv", index=False)

    print(f"Common cohort: {actual_counts} ({sum(actual_counts.values())} total)")
    print(f"Contact rows: {len(contact_table)}")
    print(f"Pairformer rows: {len(pairformer_table)}")
    print(f"Wrote publication tables to {output_dir}")


if __name__ == "__main__":
    main()
