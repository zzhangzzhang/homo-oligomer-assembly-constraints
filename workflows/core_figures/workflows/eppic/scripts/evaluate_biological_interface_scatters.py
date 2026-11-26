#!/usr/bin/env python3
"""Evaluate contact-map variants on the 40 biological EPPIC interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_eppic_contact_recovery import evaluate_map, load_ids


MODEL_DIRECTORIES = {
    "esm3_1.4b_catjac": "esm3_1.4b_catjac",
    "esm2_650m_catjac": "esm2_650m_catjac",
    "esm2_650m_official": "esm2_650m",
    "esm2_650m_refit": "esm2_650m_overlay_17_liblinear",
    "esmc_300m_catjac": "esmc_300m_catjac",
    "esmc_300m_refit": "esmc300m_overlay_17_liblinear",
    "esmc_600m_catjac": "esmc_600m_catjac",
    "esmc_600m_refit": "esmc600m_overlay_17_liblinear",
    "esmc_6b_catjac": "esmc_6b_catjac",
    "esmc_6b_refit": "esmc6b_overlay_17_liblinear",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eppic-root",
        type=Path,
        default=project_root / "data" / "evaluation",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=project_root / "data" / "predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "data"
        / "processed"
        / "scatter_comparisons_per_pdb.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eppic_root = args.eppic_root.resolve()
    prediction_root = args.prediction_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    pdb_ids = [
        pdb_id
        for pdb_id in load_ids(eppic_root / "DCbio_clean.json")
        if pdb_id != "2w6a"
    ]
    if len(pdb_ids) != 40:
        raise ValueError(f"Expected 40 biological interfaces, found {len(pdb_ids)}")

    rows: list[dict[str, object]] = []
    contact_dir = eppic_root / "contact_matrices"
    mask_dir = eppic_root / "mask"
    for pdb_id in pdb_ids:
        for model, directory in MODEL_DIRECTORIES.items():
            prediction_path = prediction_root / directory / f"{pdb_id}.npy"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            result = evaluate_map(pdb_id, prediction_path, contact_dir, mask_dir)
            rows.append({"pdb_id": pdb_id, "model": model, **result})

    frame = pd.DataFrame(rows).sort_values(["pdb_id", "model"]).reset_index(drop=True)
    coverage = frame.groupby("model")["pdb_id"].nunique()
    if len(frame) != 400 or not (coverage == 40).all():
        raise ValueError(f"Incomplete model coverage:\n{coverage}")
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} rows for {len(pdb_ids)} proteins to {output}")


if __name__ == "__main__":
    main()
