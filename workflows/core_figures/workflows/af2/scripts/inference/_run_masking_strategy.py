#!/usr/bin/env python3
"""CLI shared by the four fixed covariance-masking strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _af2_multimer_common import (
    MaskStrategy,
    configure_cuda,
    load_pdb_ids,
    run_prediction,
)


def parse_args(description: str) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=root / "data" / "cohort_manifest.csv",
    )
    parser.add_argument(
        "--msa-dir",
        type=Path,
        required=True,
        help="external directory containing <pdb>_id90.a3m inputs",
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        required=True,
        help="external directory containing assembly contact matrices",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--pdb-ids", nargs="*")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-recycles", type=int, default=6)
    parser.add_argument("--recycle-tolerance", type=float, default=0.5)
    return parser.parse_args()


def run_strategy(strategy: MaskStrategy, description: str) -> None:
    args = parse_args(description)
    configure_cuda(args.gpu)
    pdb_ids = (
        [value.lower() for value in args.pdb_ids]
        if args.pdb_ids
        else load_pdb_ids(args.cohort.resolve())
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "run_config.json").open("w") as handle:
        json.dump(
            {
                "model_type": "alphafold2_multimer_v3",
                "copies": 2,
                "qid": 0,
                "query_bias": True,
                "mask_strategy": strategy,
                "mask_rate": 1.0,
                "models": args.models or "all five multimer models",
                "num_seeds": args.num_seeds,
                "num_recycles": args.num_recycles,
                "recycle_tolerance": args.recycle_tolerance,
                "mlm_replace_fraction": 0.15,
                "cohort": str(args.cohort.resolve()),
                "msa_dir": str(args.msa_dir.resolve()),
                "contact_dir": str(args.contact_dir.resolve()),
            },
            handle,
            indent=2,
        )
    for pdb_id in pdb_ids:
        run_prediction(
            pdb_id=pdb_id,
            qid=0,
            query_bias=True,
            mask_strategy=strategy,
            msa_dir=args.msa_dir.resolve(),
            contact_dir=args.contact_dir.resolve(),
            output_root=args.output_root.resolve(),
            num_seeds=args.num_seeds,
            num_recycles=args.num_recycles,
            recycle_tolerance=args.recycle_tolerance,
            model_names=args.models,
        )
