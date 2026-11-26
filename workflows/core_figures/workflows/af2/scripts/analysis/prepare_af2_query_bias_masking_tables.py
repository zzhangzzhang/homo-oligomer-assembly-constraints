#!/usr/bin/env python3
"""Build strict, balanced observation and paired-delta plotting tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUERY_EXPECTED = 109 * 4 * 5 * 5
MASKING_EXPECTED = 109 * 5 * 5 * 5
QUERY_DELTA_EXPECTED = 109 * 2 * 5 * 5
MASKING_DELTA_EXPECTED = 109 * 4 * 5 * 5

STRATEGY_FILES = {
    "baseline": "baseline_dockq_tmalign.csv",
    "offdiag": "offdiag_dockq_tmalign.csv",
    "ondiag": "ondiag_dockq_tmalign.csv",
    "inter_expand": "inter_expand_dockq_tmalign.csv",
    "intra_expand": "intra_expand_dockq_tmalign.csv",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=root / "data" / "cohort_manifest.csv",
    )
    parser.add_argument(
        "--query-metrics",
        type=Path,
        default=root / "data" / "raw" / "query_bias" / "dockq_tmalign.csv",
    )
    parser.add_argument(
        "--masking-dir",
        type=Path,
        default=root / "data" / "raw" / "masking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument(
        "--write-cohort-manifest",
        type=Path,
        help="optional path for a regenerated logical cohort manifest",
    )
    return parser.parse_args()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns={"DockQ": "dockq", "qb": "query_bias"}).copy()
    result["pdb_id"] = result["pdb_id"].astype(str).str.lower()
    result["query_bias"] = (
        result["query_bias"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    result["qid"] = pd.to_numeric(result["qid"], errors="raise").astype(int)
    result["maskrate"] = pd.to_numeric(
        result["maskrate"], errors="raise"
    ).astype(int)
    result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(int)
    result["recycle"] = pd.to_numeric(
        result["recycle"], errors="raise"
    ).astype(int)
    result["model"] = result["model"].astype(str)
    for metric in ("tm_score", "dockq"):
        result[metric] = pd.to_numeric(result[metric], errors="raise")
    return result


def assert_balanced(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    expected_rows: int,
    expected_cohort: set[str],
    table_name: str,
) -> None:
    if len(frame) != expected_rows:
        raise AssertionError(
            f"{table_name}: expected {expected_rows} rows, found {len(frame)}"
        )
    if frame.duplicated(keys).any():
        duplicates = frame.loc[frame.duplicated(keys, keep=False), keys]
        raise AssertionError(
            f"{table_name}: duplicate keys\n{duplicates.head().to_string(index=False)}"
        )
    observed = set(frame["pdb_id"])
    if observed != expected_cohort:
        raise AssertionError(
            f"{table_name}: cohort mismatch; missing={sorted(expected_cohort-observed)}, "
            f"extra={sorted(observed-expected_cohort)}"
        )
    if set(frame["model"]) != {f"model_{value}" for value in range(1, 6)}:
        raise AssertionError(f"{table_name}: expected models model_1 through model_5")
    if set(frame["seed"]) != set(range(5)):
        raise AssertionError(f"{table_name}: expected seeds 0 through 4")
    metrics = frame[["tm_score", "dockq"]].to_numpy()
    if not np.isfinite(metrics).all():
        raise AssertionError(f"{table_name}: non-finite TM-score or DockQ")
    if ((metrics < 0) | (metrics > 1)).any():
        raise AssertionError(f"{table_name}: TM-score or DockQ outside [0, 1]")


def prepare_query_table(
    source: Path,
    cohort: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = normalize(pd.read_csv(source))
    frame = frame.loc[
        frame["pdb_id"].isin(cohort)
        & frame["recycle"].eq(0)
        & frame["maskrate"].eq(0)
        & frame["qid"].isin([0, 30])
        & frame["query_bias"].notna()
    ].copy()
    frame["condition"] = (
        "QID "
        + frame["qid"].astype(str)
        + ", query bias "
        + frame["query_bias"].map({True: "on", False: "off"})
    )
    columns = [
        "pdb_id",
        "qid",
        "query_bias",
        "condition",
        "model",
        "seed",
        "recycle",
        "tm_score",
        "dockq",
    ]
    frame = frame[columns].sort_values(
        ["pdb_id", "qid", "query_bias", "model", "seed"]
    )
    keys = ["pdb_id", "qid", "query_bias", "model", "seed"]
    assert_balanced(
        frame,
        keys=keys,
        expected_rows=QUERY_EXPECTED,
        expected_cohort=cohort,
        table_name="query-bias observations",
    )

    pair_keys = ["pdb_id", "qid", "model", "seed"]
    off = frame.loc[~frame["query_bias"], pair_keys + ["tm_score", "dockq"]]
    on = frame.loc[frame["query_bias"], pair_keys + ["tm_score", "dockq"]]
    deltas = on.merge(
        off,
        on=pair_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_on", "_off"),
    )
    deltas["tm_score_delta"] = (
        deltas["tm_score_on"] - deltas["tm_score_off"]
    )
    deltas["dockq_delta"] = deltas["dockq_on"] - deltas["dockq_off"]
    if len(deltas) != QUERY_DELTA_EXPECTED:
        raise AssertionError(
            f"query-bias deltas: expected {QUERY_DELTA_EXPECTED}, "
            f"found {len(deltas)}"
        )
    return frame, deltas.sort_values(pair_keys)


def prepare_masking_table(
    source_dir: Path,
    cohort: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for strategy, filename in STRATEGY_FILES.items():
        frame = normalize(pd.read_csv(source_dir / filename))
        intended_maskrate = 0 if strategy == "baseline" else 100
        frame = frame.loc[
            frame["pdb_id"].isin(cohort)
            & frame["recycle"].eq(0)
            & frame["qid"].eq(0)
            & frame["query_bias"].eq(True)
            & frame["maskrate"].eq(intended_maskrate)
        ].copy()
        frame["strategy"] = strategy
        tables.append(frame)

    columns = [
        "pdb_id",
        "strategy",
        "model",
        "seed",
        "recycle",
        "tm_score",
        "dockq",
    ]
    observations = pd.concat(tables, ignore_index=True)[columns]
    observations = observations.sort_values(
        ["pdb_id", "strategy", "model", "seed"]
    )
    keys = ["pdb_id", "strategy", "model", "seed"]
    assert_balanced(
        observations,
        keys=keys,
        expected_rows=MASKING_EXPECTED,
        expected_cohort=cohort,
        table_name="masking observations",
    )

    pair_keys = ["pdb_id", "model", "seed"]
    baseline = observations.loc[
        observations["strategy"].eq("baseline"),
        pair_keys + ["tm_score", "dockq"],
    ]
    treatment = observations.loc[
        ~observations["strategy"].eq("baseline")
    ].copy()
    deltas = treatment.merge(
        baseline,
        on=pair_keys,
        how="inner",
        validate="many_to_one",
        suffixes=("_strategy", "_baseline"),
    )
    deltas["tm_score_delta"] = (
        deltas["tm_score_strategy"] - deltas["tm_score_baseline"]
    )
    deltas["dockq_delta"] = (
        deltas["dockq_strategy"] - deltas["dockq_baseline"]
    )
    if len(deltas) != MASKING_DELTA_EXPECTED:
        raise AssertionError(
            f"masking deltas: expected {MASKING_DELTA_EXPECTED}, "
            f"found {len(deltas)}"
        )
    return observations, deltas.sort_values(
        ["pdb_id", "strategy", "model", "seed"]
    )


def main() -> None:
    args = parse_args()
    if args.cohort.suffix.lower() == ".csv":
        cohort_frame = pd.read_csv(args.cohort)
        if "pdb_id" not in cohort_frame:
            raise ValueError("Cohort CSV must contain pdb_id")
        cohort_values = cohort_frame["pdb_id"].astype(str).tolist()
    else:
        with args.cohort.open() as handle:
            cohort_values = json.load(handle)
        if isinstance(cohort_values, dict):
            cohort_values = list(cohort_values)
    cohort = {str(value).lower() for value in cohort_values}
    if len(cohort) != 109:
        raise AssertionError(f"Expected 109 proteins, found {len(cohort)}")

    query, query_deltas = prepare_query_table(args.query_metrics, cohort)
    masking, masking_deltas = prepare_masking_table(args.masking_dir, cohort)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "af2_query_bias_observations.csv": query,
        "af2_query_bias_paired_deltas.csv": query_deltas,
        "af2_masking_observations.csv": masking,
        "af2_masking_paired_deltas.csv": masking_deltas,
    }
    for filename, frame in outputs.items():
        destination = args.output_dir / filename
        frame.to_csv(destination, index=False)
        print(f"Wrote {len(frame):,} rows: {destination}")

    if args.write_cohort_manifest is not None:
        cohort_manifest = pd.DataFrame(
            {
                "cohort_index": range(1, len(cohort_values) + 1),
                "pdb_id": [str(value).lower() for value in cohort_values],
            }
        )
        cohort_manifest["msa_artifact"] = cohort_manifest["pdb_id"].map(
            lambda value: f"msa/{value}_id90.a3m"
        )
        cohort_manifest["native_artifact"] = cohort_manifest["pdb_id"].map(
            lambda value: f"cif/{value}-assembly1.cif"
        )
        cohort_manifest["inter_contact_artifact"] = cohort_manifest["pdb_id"].map(
            lambda value: f"contact_matrices/{value}-assembly1_inter.npy"
        )
        cohort_manifest["intra_contact_artifact"] = cohort_manifest["pdb_id"].map(
            lambda value: f"contact_matrices/{value}-assembly1_intra.npy"
        )
        args.write_cohort_manifest.parent.mkdir(parents=True, exist_ok=True)
        cohort_manifest.to_csv(args.write_cohort_manifest, index=False)


if __name__ == "__main__":
    main()
