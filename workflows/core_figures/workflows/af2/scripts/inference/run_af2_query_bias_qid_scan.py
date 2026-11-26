#!/usr/bin/env python3
"""Run the four AF-Multimer query-bias/QID conditions used in the figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _af2_multimer_common import configure_cuda, load_pdb_ids, run_prediction


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--pdb-ids", nargs="*")
    parser.add_argument("--qid", nargs="+", type=int, choices=[0, 30], default=[0, 30])
    parser.add_argument(
        "--query-bias",
        choices=["both", "on", "off"],
        default="both",
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-recycles", type=int, default=6)
    parser.add_argument("--recycle-tolerance", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda(args.gpu)
    pdb_ids = (
        [value.lower() for value in args.pdb_ids]
        if args.pdb_ids
        else load_pdb_ids(args.cohort.resolve())
    )
    query_bias_values = {
        "both": [True, False],
        "on": [True],
        "off": [False],
    }[args.query_bias]
    conditions = [
        (qid, query_bias)
        for qid in args.qid
        for query_bias in query_bias_values
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "run_config.json").open("w") as handle:
        json.dump(
            {
                "model_type": "alphafold2_multimer_v3",
                "copies": 2,
                "qid": args.qid,
                "query_bias": query_bias_values,
                "models": args.models or "all five multimer models",
                "num_seeds": args.num_seeds,
                "num_recycles": args.num_recycles,
                "recycle_tolerance": args.recycle_tolerance,
                "mlm_replace_fraction": 0.15,
                "cohort": str(args.cohort.resolve()),
                "msa_dir": str(args.msa_dir.resolve()),
            },
            handle,
            indent=2,
        )
    for pdb_id in pdb_ids:
        for qid, query_bias in conditions:
            run_prediction(
                pdb_id=pdb_id,
                qid=qid,
                query_bias=query_bias,
                mask_strategy="none",
                msa_dir=args.msa_dir.resolve(),
                contact_dir=None,
                output_root=args.output_root.resolve(),
                num_seeds=args.num_seeds,
                num_recycles=args.num_recycles,
                recycle_tolerance=args.recycle_tolerance,
                model_names=args.models,
            )


if __name__ == "__main__":
    main()
