#!/usr/bin/env python3
"""Explicit CLI for the frozen MSA-Pairformer execution implementation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


IMPLEMENTATION = Path(__file__).with_name("_run_pairformer_impl.py")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=root / "data/upstream/uniprots_le500_with_msa.json",
    )
    parser.add_argument("--msa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qids", nargs="+", type=int, default=[35])
    parser.add_argument("--shard", default="0/1", metavar="INDEX/COUNT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hhsuite-bin", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not IMPLEMENTATION.is_file():
        raise FileNotFoundError(IMPLEMENTATION)
    if not args.targets.is_file():
        raise FileNotFoundError(args.targets)
    with args.targets.open(encoding="utf-8") as handle:
        targets = json.load(handle)
    if not isinstance(targets, list) or not targets:
        raise ValueError("--targets must contain a nonempty JSON list")
    if not args.msa_dir.is_dir():
        raise FileNotFoundError(args.msa_dir)
    if any(value < 0 or value > 100 for value in args.qids):
        raise ValueError("--qids values must be percentages from 0 to 100")
    shard_parts = args.shard.split("/")
    if len(shard_parts) != 2:
        raise ValueError("--shard must use INDEX/COUNT")
    index, count = (int(value) for value in shard_parts)
    if count < 1 or index < 0 or index >= count:
        raise ValueError("invalid --shard INDEX/COUNT")

    environment = os.environ.copy()
    environment.update(
        {
            "PAIRFORMER_TARGET_LIST": str(args.targets.resolve()),
            "PAIRFORMER_MSA_DIR": str(args.msa_dir.resolve()),
            "PAIRFORMER_OUT_DIR": str(args.output_dir.resolve()),
            "PAIRFORMER_QIDS": ",".join(str(value) for value in args.qids),
            "PAIRFORMER_SHARD": args.shard,
            "PAIRFORMER_DEVICE": args.device,
        }
    )
    if args.hhsuite_bin is not None:
        environment["HHSUITE_BIN"] = str(args.hhsuite_bin.resolve())
    command = [args.python, str(IMPLEMENTATION)]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "targets": len(targets),
                    "qids": args.qids,
                    "shard": args.shard,
                    "device": args.device,
                    "msa_dir": environment["PAIRFORMER_MSA_DIR"],
                    "output_dir": environment["PAIRFORMER_OUT_DIR"],
                },
                indent=2,
            )
        )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
