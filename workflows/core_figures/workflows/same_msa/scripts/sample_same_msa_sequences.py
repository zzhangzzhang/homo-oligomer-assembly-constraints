#!/usr/bin/env python3
"""Select a deterministic set of aligned homologs across identity bins.

This is a cleaned replacement for the sampling cell in
``fig2004_sameMSA_3epw.ipynb``. It fixes the historical off-by-one error by
pairing each calculated identity with the same A3M record, and stores source
record indices, headers, identities, and aligned sequences for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PROTEINS = (
    "1BRTA",
    "1FCYA",
    "1KQFC",
    "1RKUA",
    "1TKEA",
    "1TQHA",
    "3H7AA",
    "3o1n",
    "4COGA",
    "4RY0A",
    "4XOMA",
)
BIN_INDICES = tuple(range(1, 9))
DEFAULT_SEED = 20260731


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=root / "data" / "alignments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "sampled_sequences",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples-per-bin", type=int, default=3)
    return parser.parse_args()


def read_a3m(path: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(
                        {
                            "source_index": len(records),
                            "header": header,
                            "aligned_sequence": "".join(sequence_parts),
                        }
                    )
                header = line[1:]
                sequence_parts = []
            elif header is None:
                raise ValueError(f"{path}: sequence before first header")
            else:
                sequence_parts.append(line)
    if header is not None:
        records.append(
            {
                "source_index": len(records),
                "header": header,
                "aligned_sequence": "".join(sequence_parts),
            }
        )
    if len(records) < 2:
        raise ValueError(f"{path}: expected query plus homolog records")
    return records


def remove_lowercase(sequence: str) -> str:
    return "".join(character for character in sequence if not character.islower())


def identity_to_query(query: str, aligned_sequence: str) -> float:
    aligned_without_insertions = remove_lowercase(aligned_sequence)
    if len(aligned_without_insertions) != len(query):
        raise ValueError(
            "A3M record length after removing insertions does not match query: "
            f"{len(aligned_without_insertions)} != {len(query)}"
        )
    matches = sum(
        query_character == candidate_character
        for query_character, candidate_character in zip(
            query, aligned_without_insertions
        )
    )
    return matches / len(query)


def protein_seed(global_seed: int, protein: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{protein}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_protein(
    protein: str,
    alignment_path: Path,
    global_seed: int,
    samples_per_bin: int,
) -> dict[str, object]:
    records = read_a3m(alignment_path)
    query = str(records[0]["aligned_sequence"])
    pools: dict[int, list[dict[str, str | int | float]]] = {
        bin_index: [] for bin_index in BIN_INDICES
    }

    for record in records[1:]:
        identity = identity_to_query(query, str(record["aligned_sequence"]))
        bin_index = int(identity * 10)
        if bin_index in pools:
            pools[bin_index].append({**record, "identity": identity})

    rng = random.Random(protein_seed(global_seed, protein))
    sampled: dict[str, list[dict[str, str | int | float]]] = {}
    for bin_index in BIN_INDICES:
        pool = pools[bin_index]
        count = min(samples_per_bin, len(pool))
        sampled[str(bin_index)] = rng.sample(pool, count)

    return {
        "schema_version": 1,
        "protein": protein,
        "global_seed": global_seed,
        "protein_seed": protein_seed(global_seed, protein),
        "samples_per_bin": samples_per_bin,
        "identity_bin_convention": "[bin/10, (bin+1)/10)",
        "query": records[0],
        "candidate_counts": {
            str(bin_index): len(pools[bin_index]) for bin_index in BIN_INDICES
        },
        "samples": sampled,
    }


def main() -> None:
    args = parse_args()
    alignment_dir = args.alignment_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_rows = []
    for protein in PROTEINS:
        alignment_path = alignment_dir / f"{protein}.a3m"
        if not alignment_path.is_file():
            raise FileNotFoundError(f"Missing alignment: {alignment_path}")
        result = sample_protein(
            protein,
            alignment_path,
            global_seed=args.seed,
            samples_per_bin=args.samples_per_bin,
        )
        output_path = output_dir / f"{protein}.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

        candidate_counts = result["candidate_counts"]
        sampled = result["samples"]
        coverage_rows.append(
            {
                "protein": protein,
                **{
                    f"bin_{bin_index}_candidates": candidate_counts[str(bin_index)]
                    for bin_index in BIN_INDICES
                },
                **{
                    f"bin_{bin_index}_sampled": len(sampled[str(bin_index)])
                    for bin_index in BIN_INDICES
                },
            }
        )

    import csv

    coverage_path = output_dir / "sampling_coverage.csv"
    with coverage_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    incomplete = [
        row["protein"]
        for row in coverage_rows
        if any(
            row[f"bin_{bin_index}_sampled"] < args.samples_per_bin
            for bin_index in BIN_INDICES
        )
    ]
    if incomplete != ["3o1n"]:
        raise ValueError(
            "Expected only 3o1n to have an incomplete bin; "
            f"found {incomplete}"
        )
    if coverage_rows[PROTEINS.index("3o1n")]["bin_8_sampled"] != 1:
        raise ValueError("Expected exactly one 3o1n sample in identity bin 8")
    print(f"Wrote deterministic samples for {len(PROTEINS)} proteins")
    print("Coverage exception: 3o1n bin 8 has one available sequence")


if __name__ == "__main__":
    main()
