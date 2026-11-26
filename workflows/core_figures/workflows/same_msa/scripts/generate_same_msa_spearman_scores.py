#!/usr/bin/env python3
"""Generate, archive, and score same-MSA categorical Jacobians with ESM2.

The numerical convention follows the original analysis: ``fx - fx_h``,
four-axis centering, symmetrization, float16 archival precision, top-L
reference-contact ranking, an RMS coefficient threshold, and SciPy Spearman
correlation. Jacobians are stored as losslessly compressed float16 NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import esm
import numpy as np
import torch
from scipy.stats import spearmanr


MODEL_CHECKPOINTS = {
    "150m": "esm2_t30_150M_UR50D.pt",
    "650m": "esm2_t33_650M_UR50D.pt",
    "3b": "esm2_t36_3B_UR50D.pt",
    "15b": "esm2_t48_15B_UR50D.pt",
}
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
FIELDNAMES = (
    "model",
    "protein",
    "identity_bin",
    "replicate",
    "source_index",
    "sequence_identity",
    "n_aligned_residues",
    "n_masked_coefficients",
    "spearman_r",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_CHECKPOINTS)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=root / "data" / "sampled_sequences",
    )
    parser.add_argument(
        "--jacobian-dir",
        type=Path,
        default=root / "data" / "jacobians",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "derived",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--inference-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Model inference precision; use float16 for ESM2-15B on 80 GB GPUs.",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=list(PROTEINS),
        choices=PROTEINS,
    )
    return parser.parse_args()


def center_and_symmetrize(jacobian: np.ndarray) -> np.ndarray:
    for axis in range(4):
        jacobian -= jacobian.mean(axis, keepdims=True)
    return (jacobian + jacobian.transpose(2, 3, 0, 1)) / 2


def contacts_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    centered = jacobian.copy()
    for axis in range(4):
        centered -= centered.mean(axis, keepdims=True)
    frobenius = np.sqrt(np.square(centered).sum((1, 3)))
    np.fill_diagonal(frobenius, 0)
    column_sum = frobenius.sum(0, keepdims=True)
    row_sum = frobenius.sum(1, keepdims=True)
    corrected = frobenius - (column_sum * row_sum) / frobenius.sum()
    np.fill_diagonal(corrected, 0)
    return (corrected + corrected.T) / 2


def save_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


class ESM2Jacobian:
    def __init__(
        self, checkpoint: Path, device: str, inference_dtype: str
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        self.device = torch.device(device)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
            str(checkpoint)
        )
        if inference_dtype == "float16":
            self.model = self.model.half()
        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def __call__(self, sequence: str) -> np.ndarray:
        tokens = self.batch_converter([("sequence", sequence)])[-1]
        length = len(sequence)

        def logits(input_tokens: torch.Tensor) -> np.ndarray:
            output = self.model(input_tokens)["logits"]
            return output[..., 1 : length + 1, 4:24].cpu().numpy()

        with torch.no_grad():
            wild_type = logits(tokens.to(self.device))[0]
            tiled = torch.tile(tokens, [20, 1]).to(self.device)
            mutated = np.zeros((length, 20, length, 20))
            for position in range(length):
                substitutions = torch.clone(tiled)
                substitutions[:, position + 1] = torch.arange(
                    4, 24, device=self.device
                )
                mutated[position] = logits(substitutions)
        return wild_type - mutated


def model_sequence(aligned_sequence: str) -> str:
    return re.sub(r"[^a-zA-Z]", "", aligned_sequence).upper()


def load_or_generate_query(
    calculator: ESM2Jacobian,
    aligned_sequence: str,
    archive_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if archive_path.is_file():
        with np.load(archive_path) as archive:
            return archive["jacobian"], archive["contact"]
    full_precision = center_and_symmetrize(
        calculator(model_sequence(aligned_sequence))
    )
    contact = contacts_from_jacobian(full_precision).astype(np.float16)
    jacobian = full_precision.astype(np.float16)
    save_compressed(archive_path, jacobian=jacobian, contact=contact)
    return jacobian, contact


def load_or_generate_sample(
    calculator: ESM2Jacobian,
    aligned_sequence: str,
    archive_path: Path,
) -> np.ndarray:
    if archive_path.is_file():
        with np.load(archive_path) as archive:
            return archive["jacobian"]
    full_precision = center_and_symmetrize(
        calculator(model_sequence(aligned_sequence))
    )
    jacobian = full_precision.astype(np.float16)
    save_compressed(archive_path, jacobian=jacobian)
    return jacobian


def score_jacobians(
    reference_contact: np.ndarray,
    reference_jacobian: np.ndarray,
    sample_jacobian: np.ndarray,
    aligned_sequence: str,
) -> tuple[float, int, int]:
    letters_only = re.sub(r"[^a-zA-Z]", "", aligned_sequence)
    without_insertions = re.sub(r"[a-z]", "", aligned_sequence)
    sample_indices = [
        index for index, character in enumerate(letters_only) if character.isupper()
    ]
    reference_indices = [
        index
        for index, character in enumerate(without_insertions)
        if character.isupper()
    ]
    if len(sample_indices) != len(reference_indices):
        raise ValueError("Sample/reference aligned residue counts differ")
    if not sample_indices:
        return np.nan, 0, 0
    if max(sample_indices) >= sample_jacobian.shape[0]:
        raise ValueError("Sample alignment index exceeds Jacobian length")
    if max(reference_indices) >= reference_jacobian.shape[0]:
        raise ValueError("Reference alignment index exceeds Jacobian length")

    sample = sample_jacobian[
        np.ix_(sample_indices, range(20), sample_indices, range(20))
    ]
    reference = reference_jacobian[
        np.ix_(reference_indices, range(20), reference_indices, range(20))
    ]
    reference_scores = reference_contact[
        np.ix_(reference_indices, reference_indices)
    ]
    length = len(sample_indices)
    if length < 2:
        return np.nan, length, 0

    triangle = np.triu_indices(length, 1)
    sample_pairs = sample[triangle[0], :, triangle[1], :]
    reference_pairs = reference[triangle[0], :, triangle[1], :]
    order = reference_scores[triangle[0], triangle[1]].argsort()[::-1]
    selected_reference = reference_pairs[order[:length]]
    threshold = np.sqrt(np.square(selected_reference).mean())
    mask = np.abs(selected_reference) > threshold
    selected_sample = sample_pairs[order[:length]][mask]
    selected_reference = selected_reference[mask]
    if not selected_sample.size or not selected_reference.size:
        return np.nan, length, 0
    coefficient = spearmanr(
        selected_sample.flatten(), selected_reference.flatten()
    ).statistic
    return float(coefficient), length, int(mask.sum())


def load_existing(path: Path) -> tuple[list[dict[str, str]], set[tuple[str, str, str]]]:
    if not path.is_file():
        return [], set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = {
        (row["protein"], row["identity_bin"], row["replicate"]) for row in rows
    }
    return rows, keys


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint_root.resolve() / MODEL_CHECKPOINTS[args.model]
    sample_dir = args.sample_dir.resolve()
    jacobian_root = args.jacobian_dir.resolve() / args.model
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calculator = ESM2Jacobian(
        checkpoint=checkpoint,
        device=args.device,
        inference_dtype=args.inference_dtype,
    )
    for protein in args.proteins:
        output_path = (
            output_dir / "replicates" / args.model / f"{protein}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows, completed = load_existing(output_path)
        with (sample_dir / f"{protein}.json").open() as handle:
            sample_data = json.load(handle)

        query_aligned = sample_data["query"]["aligned_sequence"]
        query_archive = jacobian_root / protein / "query.npz"
        print(f"[{args.model}] {protein}: query", flush=True)
        reference_jacobian, reference_contact = load_or_generate_query(
            calculator, query_aligned, query_archive
        )

        for bin_index in range(1, 9):
            for replicate, sample in enumerate(
                sample_data["samples"][str(bin_index)]
            ):
                key = (protein, str(bin_index), str(replicate))
                archive_path = (
                    jacobian_root / protein / f"{bin_index}_{replicate}.npz"
                )
                print(
                    f"[{args.model}] {protein} bin={bin_index} rep={replicate}",
                    flush=True,
                )
                sample_jacobian = load_or_generate_sample(
                    calculator,
                    sample["aligned_sequence"],
                    archive_path,
                )
                if key not in completed:
                    coefficient, n_aligned, n_masked = score_jacobians(
                        reference_contact,
                        reference_jacobian,
                        sample_jacobian,
                        sample["aligned_sequence"],
                    )
                    rows.append(
                        {
                            "model": args.model,
                            "protein": protein,
                            "identity_bin": bin_index,
                            "replicate": replicate,
                            "source_index": sample["source_index"],
                            "sequence_identity": sample["identity"],
                            "n_aligned_residues": n_aligned,
                            "n_masked_coefficients": n_masked,
                            "spearman_r": coefficient,
                        }
                    )
                    completed.add(key)
                    write_rows(output_path, rows)
                del sample_jacobian
                gc.collect()

        del reference_contact, reference_jacobian
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Wrote {len(rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
