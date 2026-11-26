#!/usr/bin/env python3
"""Score qid15 and qid35 Pairformer matrices against ConFind contacts.

Both qids use the same assembly-1 ConFind contact sets and the same denominator:

    X = number of unique (intra-chain union biological-interface) contacts
        with UniProt sequence separation >= 6.

The top-X Pairformer positions are selected from the corresponding upper
triangle.  This script does not rerun ConFind.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


DEFAULT_MANIFEST = (
    "/path/to/external-data/pdb_pairformer_screen/full_screen/manifest_homo.csv"
)
DEFAULT_CONTACT_DIR = (
    "/path/to/external-data/pdb_pairformer_screen/confind_bio_assembly1/contacts"
)
DEFAULT_PAIRFORMER_DIR = "/path/to/external-data/pdb_pairformer_screen/pairformer"
DEFAULT_OUTPUT = (
    "/path/to/external-data/pdb_pairformer_screen/confind_bio_assembly1/"
    "pairformer_qid15_qid35_summary.csv"
)

BASE_FIELDS = [
    "pdb", "uniprot", "L_up",
    "n_intra_confind", "n_bio_confind", "x_union_confind",
]
METRIC_NAMES = [
    "topx_size",
    "n_union_hits_topx", "union_precision_topx",
    "n_intra_hits_topx", "intra_recall_topx",
    "n_bio_hits_topx", "bio_recall_topx",
    "bio_zero_support",
    "mean_pf_intra", "mean_pf_bio",
]
OUTPUT_FIELDS = (
    BASE_FIELDS
    + [f"{name}_qid15" for name in METRIC_NAMES]
    + [f"{name}_qid35" for name in METRIC_NAMES]
    + [
        "delta_union_precision_qid15_minus_qid35",
        "delta_bio_recall_qid15_minus_qid35",
        "intra_contact_path", "inter_contact_path",
    ]
)
FAILURE_FIELDS = ["pdb", "uniprot", "stage", "reason"]


def load_manifest(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def pair_set(array: np.ndarray, min_sep: int) -> set[tuple[int, int]]:
    if array.size == 0:
        return set()
    return {
        (int(i), int(j))
        for i, j in np.asarray(array).reshape(-1, 2)
        if int(j) - int(i) >= min_sep
    }


def load_contacts(
    intra_path: str,
    inter_path: str,
    min_sep: int,
) -> tuple[set, set, int]:
    intra_matrix = np.load(intra_path, allow_pickle=False, mmap_mode="r")
    inter_matrix = np.load(inter_path, allow_pickle=False, mmap_mode="r")
    if (
        intra_matrix.ndim != 2
        or inter_matrix.ndim != 2
        or intra_matrix.shape[0] != intra_matrix.shape[1]
        or intra_matrix.shape != inter_matrix.shape
    ):
        raise ValueError(
            f"invalid intra/inter matrix shapes: "
            f"{intra_matrix.shape}, {inter_matrix.shape}"
        )

    def matrix_pairs(matrix: np.ndarray) -> set[tuple[int, int]]:
        rows, columns = np.where(np.triu(matrix, k=min_sep) > 0)
        return {
            (int(i) + 1, int(j) + 1)
            for i, j in zip(rows, columns)
        }

    return (
        matrix_pairs(intra_matrix),
        matrix_pairs(inter_matrix),
        int(intra_matrix.shape[0]),
    )


def mean_scores(matrix: np.ndarray, pairs: set[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    values = [float(matrix[i - 1, j - 1]) for i, j in pairs]
    return float(np.mean(values))


def score_one_matrix(
    matrix_path: str,
    L_up: int,
    intra: set[tuple[int, int]],
    bio: set[tuple[int, int]],
    min_sep: int,
) -> dict:
    matrix = np.loadtxt(matrix_path)
    if matrix.shape != (L_up, L_up):
        raise ValueError(
            f"Pairformer shape {matrix.shape} does not match L_up={L_up}"
        )
    matrix = 0.5 * (matrix + matrix.T)
    rows, cols = np.triu_indices(L_up, k=min_sep)
    structural_union = intra | bio
    requested_x = len(structural_union)
    topx_size = min(requested_x, len(rows))

    if topx_size:
        scores = matrix[rows, cols]
        order = np.argsort(-scores, kind="mergesort")[:topx_size]
        top_pairs = {
            (int(rows[index]) + 1, int(cols[index]) + 1)
            for index in order
        }
    else:
        top_pairs = set()

    union_hits = len(top_pairs & structural_union)
    intra_hits = len(top_pairs & intra)
    bio_hits = len(top_pairs & bio)
    return {
        "topx_size": topx_size,
        "n_union_hits_topx": union_hits,
        "union_precision_topx": (
            union_hits / topx_size if topx_size else None
        ),
        "n_intra_hits_topx": intra_hits,
        "intra_recall_topx": intra_hits / len(intra) if intra else None,
        "n_bio_hits_topx": bio_hits,
        "bio_recall_topx": bio_hits / len(bio) if bio else None,
        "bio_zero_support": bool(bio and bio_hits == 0),
        "mean_pf_intra": mean_scores(matrix, intra),
        "mean_pf_bio": mean_scores(matrix, bio),
    }


def process_one(row: dict, config: dict) -> tuple[dict | None, dict | None]:
    pdb = row["pdb"].upper()
    uniprot = row["uniprot"]
    L_up = int(row["L_up"])
    intra_path = os.path.join(
        config["contact_dir"], f"{pdb.lower()}-assembly1_intra.npy"
    )
    inter_path = os.path.join(
        config["contact_dir"], f"{pdb.lower()}-assembly1_inter.npy"
    )
    stage = "contacts"
    try:
        if not os.path.exists(intra_path):
            raise FileNotFoundError(intra_path)
        if not os.path.exists(inter_path):
            raise FileNotFoundError(inter_path)
        intra, bio, matrix_length = load_contacts(
            intra_path, inter_path, config["min_sep"]
        )
        if matrix_length != L_up:
            raise ValueError(
                f"contact matrix length {matrix_length} does not match L_up={L_up}"
            )

        output = {
            "pdb": pdb,
            "uniprot": uniprot,
            "L_up": L_up,
            "n_intra_confind": len(intra),
            "n_bio_confind": len(bio),
            "x_union_confind": len(intra | bio),
            "intra_contact_path": intra_path,
            "inter_contact_path": inter_path,
        }
        per_qid = {}
        for qid in (15, 35):
            stage = f"pairformer_qid{qid}"
            matrix_path = os.path.join(
                config["pairformer_dir"],
                f"{uniprot}_qid{qid}_hhfilter.txt",
            )
            metrics = score_one_matrix(
                matrix_path, L_up, intra, bio, config["min_sep"]
            )
            per_qid[qid] = metrics
            for name, value in metrics.items():
                output[f"{name}_qid{qid}"] = value

        output["delta_union_precision_qid15_minus_qid35"] = (
            value_or_nan(per_qid[15]["union_precision_topx"])
            - value_or_nan(per_qid[35]["union_precision_topx"])
        )
        output["delta_bio_recall_qid15_minus_qid35"] = (
            value_or_nan(per_qid[15]["bio_recall_topx"])
            - value_or_nan(per_qid[35]["bio_recall_topx"])
        )
        return output, None
    except Exception as error:
        return None, {
            "pdb": pdb,
            "uniprot": uniprot,
            "stage": stage,
            "reason": (
                f"{type(error).__name__}: {error} | "
                + traceback.format_exc(limit=2).replace("\n", " ")
            ),
        }


def value_or_nan(value) -> float:
    return float(value) if value is not None else float("nan")


def atomic_write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--contact-dir", default=DEFAULT_CONTACT_DIR)
    parser.add_argument("--pairformer-dir", default=DEFAULT_PAIRFORMER_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-sep", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdbs", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    if args.pdbs:
        selected = {pdb.upper() for pdb in args.pdbs}
        rows = [row for row in rows if row["pdb"].upper() in selected]
    if args.limit is not None:
        rows = rows[: args.limit]
    config = {
        "contact_dir": args.contact_dir,
        "pairformer_dir": args.pairformer_dir,
        "min_sep": args.min_sep,
    }

    outputs = []
    failures = []
    if args.workers <= 1:
        iterator = (process_one(row, config) for row in rows)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(process_one, row, config) for row in rows]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for output, failure in tqdm(iterator, total=len(rows)):
            if output is not None:
                outputs.append(output)
            if failure is not None:
                failures.append(failure)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    outputs.sort(key=lambda row: row["pdb"])
    failures.sort(key=lambda row: row["pdb"])
    atomic_write_csv(args.output, OUTPUT_FIELDS, outputs)
    failure_path = os.path.splitext(args.output)[0] + ".failed.csv"
    atomic_write_csv(failure_path, FAILURE_FIELDS, failures)
    print(f"Scored:   {len(outputs):,} -> {args.output}")
    print(f"Failures: {len(failures):,} -> {failure_path}")


if __name__ == "__main__":
    main()
