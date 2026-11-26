#!/usr/bin/env python3
"""
Rebuild combined_df.csv for AlphaFold multimer runs.

This recomputes structure contact metrics and merges them with AF log metrics,
following the same logic used in notebooks/extract_data.ipynb.

Default target:
  /path/to/external-data/olig_af_batch_multimer/
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

from lib.log_metrics import analyze_results
from lib.structure_metrics import analyze_structures
from lib.run_completeness import pdb_ids_with_min_seeds


def load_pdb_ids(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def rebuild_combined_df(
    data_path: str,
    structure_dir: str,
    json_path: str,
    out_csv: str = None,
    min_seeds: int = None,
):
    pdb_ids = load_pdb_ids(json_path)
    print(f"Loaded {len(pdb_ids)} PDB IDs from {json_path}")

    if min_seeds is not None:
        ready = pdb_ids_with_min_seeds(data_path, pdb_ids, min_seeds=min_seeds)
        print(
            f"--min-seeds {min_seeds}: {len(ready)}/{len(pdb_ids)} PDBs with all "
            f"qid{{0,30}}×qb{{True,False}} runs having ≥{min_seeds} seeds (model_1 r0)"
        )
        pdb_ids = ready
        if not pdb_ids:
            raise RuntimeError("No PDBs passed the min-seeds filter. Check paths and runs.")

    print(f"Parsing AF logs from {data_path}")
    af_df = analyze_results(data_path)
    af_df = af_df[af_df["pdb_id"].isin(pdb_ids)]
    print(f"AF metrics rows (filtered to selected PDBs): {len(af_df)}")

    dfs = []
    for pdb_id in tqdm(pdb_ids, desc="structure_analysis"):
        cif_path = os.path.join(structure_dir, f"{pdb_id}-assembly1.cif")
        if not os.path.exists(cif_path):
            continue
        try:
            df = analyze_structures(data_path, pdb_id, cif_path)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] {pdb_id}: {e}")

    if not dfs:
        raise RuntimeError("No structure-analysis rows produced. Check paths and CIF files.")

    structure_df = pd.concat(dfs, ignore_index=True)
    print(f"Structure metrics rows: {len(structure_df)}")

    key_cols = ["pdb_id", "qid", "qb", "maskrate", "seed", "model", "recycle"]
    combined_df = pd.merge(
        af_df,
        structure_df,
        on=key_cols,
        how="inner",
    )
    print(f"Merged rows: {len(combined_df)}")

    if out_csv is None:
        out_csv = os.path.join(data_path, "combined_df.csv")
    combined_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # quick sanity summary
    qid_vals = sorted(combined_df["qid"].astype(str).unique().tolist())
    qb_vals = sorted(combined_df["qb"].astype(str).unique().tolist())
    print(f"qid values: {qid_vals}")
    print(f"qb values: {qb_vals}")
    print(
        combined_df.groupby(["qid", "qb"]).size().reset_index(name="n_rows").sort_values(["qid", "qb"]).to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description="Rebuild combined_df.csv for AF multimer runs")
    parser.add_argument("--data-path", default="/path/to/external-data/olig_af_batch_multimer/")
    parser.add_argument("--structure-dir", default="/path/to/external-data/msa_pairformer/confind/cif/")
    parser.add_argument("--json-path", default="/path/to/external-data/msa_pairformer/high_diff_pdbs.json")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=None,
        metavar="N",
        help="Only PDBs where each qid 0/30 and qb True/False has ≥N seeds at recycle 0 (model_1)",
    )
    args = parser.parse_args()

    rebuild_combined_df(
        data_path=args.data_path,
        structure_dir=args.structure_dir,
        json_path=args.json_path,
        out_csv=args.out_csv,
        min_seeds=args.min_seeds,
    )


if __name__ == "__main__":
    main()

