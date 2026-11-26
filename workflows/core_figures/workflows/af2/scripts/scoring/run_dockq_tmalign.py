#!/usr/bin/env python3
"""
Run DockQ and TM-align analysis on all experiment sets.

Outputs one CSV per experiment set into its base directory:
  - dockq_tmalign.csv

Usage:
    python3 run_dockq_tmalign.py                # recycle 0 only (default)
    python3 run_dockq_tmalign.py --all-recycles  # all recycles
"""

import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from lib.structural_scoring import analyze_prediction_dir
from lib.run_completeness import pdb_ids_with_min_seeds

# --- Configuration ---
json_path = "/path/to/external-data/msa_pairformer/high_diff_pdbs.json"
structure_dir = "/path/to/external-data/msa_pairformer/confind/cif/"

# All four experiment directories
EXPERIMENTS = {
    "baseline": "/path/to/external-data/olig_af_batch_multimer/",
    "inter_expand": "/path/to/external-data/olig_af_batch_multimer_expand/",
    "offdiag": "/path/to/external-data/olig_af_batch_multimer_offdiag/",
    "intra_expand": "/path/to/external-data/olig_af_batch_multimer_intra_expand/",
    "ondiag": "/path/to/external-data/olig_af_batch_multimer_ondiag/",
}


def _process_one_pdb(args_tuple):
    """Worker function for parallel PDB processing."""
    data_path, pdb_id, cif_path, recycle_0_only = args_tuple
    try:
        results = analyze_prediction_dir(
            data_path,
            pdb_id,
            cif_path,
            run_dockq_flag=True,
            run_tm_flag=True,
            recycle_0_only=recycle_0_only,
        )
        return results
    except Exception as e:
        print(f"  Error processing {pdb_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Run DockQ and TM-align analysis")
    parser.add_argument(
        "--all-recycles",
        action="store_true",
        help="Process all recycles (default: recycle 0 only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if dockq_tmalign.csv already exists (fixes stale/wrong CSVs)",
    )
    parser.add_argument(
        "--json-path",
        default=json_path,
        help="JSON list or dict of PDB IDs to consider",
    )
    parser.add_argument(
        "--structure-dir",
        default=structure_dir,
        help="directory containing <pdb>-assembly1.cif references",
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override/add an experiment directory; may be repeated",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=None,
        metavar="N",
        help="Only PDBs where each qid 0/30 × qb True/False has ≥N seeds at recycle 0",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help='Comma-separated keys from EXPERIMENTS (e.g. "baseline") or "all"',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        metavar="N",
        help="Number of parallel workers (default: 32)",
    )
    args = parser.parse_args()

    recycle_0_only = not args.all_recycles
    recycle_label = "all recycles" if args.all_recycles else "recycle 0 only"

    with open(args.json_path, "r") as f:
        pdb_ids = json.load(f)
        if isinstance(pdb_ids, dict):
            pdb_ids = list(pdb_ids.keys())

    print(f"Loaded {len(pdb_ids)} PDB IDs from {args.json_path} ({recycle_label})")

    experiments = dict(EXPERIMENTS)
    for value in args.experiment_dir:
        if "=" not in value:
            raise SystemExit("--experiment-dir must use NAME=PATH")
        name, path = value.split("=", 1)
        experiments[name.strip()] = path.strip()

    if args.experiments.strip().lower() == "all":
        exp_items = list(experiments.items())
    else:
        keys = [k.strip() for k in args.experiments.split(",") if k.strip()]
        exp_items = [(k, experiments[k]) for k in keys if k in experiments]
        unknown = [k for k in keys if k not in experiments]
        if unknown:
            raise SystemExit(
                f"Unknown experiment keys: {unknown}. Valid: {list(experiments)}"
            )

    for exp_name, data_path in exp_items:
        out_csv = os.path.join(data_path, "dockq_tmalign.csv")

        if os.path.exists(out_csv) and not args.overwrite:
            print(
                f"\n>>> [{exp_name}] Already exists: {out_csv}, skipping. Use --overwrite to recompute."
            )
            continue

        if not os.path.exists(data_path):
            print(f"\n>>> [{exp_name}] Directory not found: {data_path}, skipping.")
            continue

        exp_pdb_ids = pdb_ids
        if args.min_seeds is not None:
            exp_pdb_ids = pdb_ids_with_min_seeds(
                data_path, pdb_ids, min_seeds=args.min_seeds
            )
            print(
                f"\n>>> [{exp_name}] --min-seeds {args.min_seeds}: "
                f"{len(exp_pdb_ids)}/{len(pdb_ids)} PDBs on disk"
            )
            if not exp_pdb_ids:
                print(f">>> [{exp_name}] No PDBs passed filter, skipping.")
                continue

        print(f"\n{'='*60}")
        print(f"Processing: {exp_name} ({data_path}) — {recycle_label}")
        print(f"{'='*60}")

        work_items = []
        for pdb_id in exp_pdb_ids:
            cif_path = os.path.join(
                args.structure_dir, f"{pdb_id}-assembly1.cif"
            )
            if not os.path.exists(cif_path):
                print(f"  CIF not found for {pdb_id}, skipping.")
                continue
            pdb_dir = os.path.join(data_path, pdb_id)
            if not os.path.exists(pdb_dir):
                continue
            work_items.append((data_path, pdb_id, cif_path, recycle_0_only))

        all_results = []
        n_workers = min(args.workers, len(work_items)) if work_items else 1
        print(f"  Processing {len(work_items)} PDBs with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_process_one_pdb, item): item[1] for item in work_items}
            for future in tqdm(as_completed(futures), total=len(futures), desc=exp_name):
                results = future.result()
                if results:
                    all_results.extend(results)

        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(out_csv, index=False)
            print(f"\nSaved {len(df)} rows to {out_csv}")

            print(f"\nSummary for {exp_name} ({recycle_label}):")
            print(f"  Proteins: {df['pdb_id'].nunique()}")
            print(f"  DockQ mean: {df['DockQ'].mean():.4f} (median: {df['DockQ'].median():.4f})")
            print(
                f"  TM-score mean: {df['tm_score'].mean():.4f} (median: {df['tm_score'].median():.4f})"
            )
        else:
            print(f"\nNo results for {exp_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
