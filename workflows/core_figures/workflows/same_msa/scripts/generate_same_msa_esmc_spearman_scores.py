#!/usr/bin/env python3
"""Generate, archive, and score same-MSA categorical Jacobians with ESM-C."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path


def configure_visible_device() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cuda:0")
    arguments, _ = parser.parse_known_args()
    if arguments.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = arguments.device.split(":", 1)[1]
        return "cuda:0"
    return arguments.device


REMAPPED_DEVICE = configure_visible_device()
os.environ.setdefault("HF_HOME", "/path/to/external-data/hf_cache")
os.environ.setdefault("TORCH_HOME", "/path/to/external-data/torch_cache")

import numpy as np
import torch

import generate_same_msa_spearman_scores as common


VARIANTS = {
    "300m": "esmc_300m",
    "600m": "esmc_600m",
    "6b": "biohub/ESMC-6B",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--model",
        help="Override the registered model name or pinned local 6B snapshot.",
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--substitution-batch-size",
        type=int,
        default=0,
        help="Substitutions per forward pass; 0 selects by sequence length.",
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=list(common.PROTEINS),
        choices=common.PROTEINS,
    )
    return parser.parse_args()


def automatic_batch_size(length: int, variant: str) -> int:
    if variant == "6b":
        if length <= 256:
            return 8
        if length <= 512:
            return 4
        if length <= 900:
            return 2
        return 1
    if length <= 300:
        return 20
    if length <= 500:
        return 5
    if length <= 800:
        return 2
    return 1


def alphabet_from_tokenizer(tokenizer) -> list[str]:
    inverse = {
        token_index: token
        for token, token_index in tokenizer.get_vocab().items()
    }
    return [inverse[index] for index in range(4, 24)]


class ESMCJacobian:
    def __init__(
        self,
        variant: str,
        model_name: str,
        device: str,
        substitution_batch_size: int,
    ) -> None:
        self.variant = variant
        self.device = torch.device(device)
        self.requested_batch_size = substitution_batch_size
        if variant == "6b":
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            ).to(self.device).eval()
            self.backend = "transformers"
        else:
            from esm.models.esmc import ESMC

            self.model = ESMC.from_pretrained(
                model_name, device=self.device
            ).eval()
            self.model = self.model.to(dtype=torch.float32)
            self.tokenizer = self.model.tokenizer
            self.backend = "esm"
        self.alphabet = alphabet_from_tokenizer(self.tokenizer)
        if set(self.alphabet) != set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"Unexpected ESM-C amino-acid vocabulary: {self.alphabet}")

    def encode(self, sequence: str) -> torch.Tensor:
        if self.backend == "transformers":
            return self.tokenizer(
                sequence, return_tensors="pt"
            )["input_ids"][0].to(self.device)
        from esm.sdk.api import ESMProtein

        return self.model.encode(
            ESMProtein(sequence=sequence)
        ).sequence.to(self.device)

    def logits(self, input_tokens: torch.Tensor, length: int) -> np.ndarray:
        if self.backend == "transformers":
            output = self.model(input_ids=input_tokens).logits
        else:
            output = self.model(
                sequence_tokens=input_tokens
            ).sequence_logits
        return (
            output[..., 1 : length + 1, 4:24]
            .float()
            .cpu()
            .numpy()
        )

    def __call__(self, sequence: str) -> np.ndarray:
        tokens = self.encode(sequence)
        length = len(sequence)
        if tokens.shape[0] != length + 2:
            raise ValueError(
                f"Expected {length + 2} tokens including BOS/EOS, "
                f"found {tokens.shape[0]}"
            )
        batch_size = (
            self.requested_batch_size
            if self.requested_batch_size
            else automatic_batch_size(length, self.variant)
        )
        with torch.no_grad():
            wild_type = self.logits(tokens[None], length)[0]
            mutated = np.empty(
                (length, 20, length, 20), dtype=np.float32
            )
            for position in range(length):
                for start in range(0, 20, batch_size):
                    stop = min(start + batch_size, 20)
                    substitutions = tokens.unsqueeze(0).repeat(
                        stop - start, 1
                    )
                    substitutions[:, position + 1] = torch.arange(
                        4 + start, 4 + stop, device=self.device
                    )
                    mutated[position, start:stop] = self.logits(
                        substitutions, length
                    )
        return wild_type - mutated


def main() -> None:
    args = parse_args()
    if args.substitution_batch_size < 0:
        raise ValueError("--substitution-batch-size cannot be negative")
    if REMAPPED_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_label = f"esmc_{args.variant}"
    model_name = args.model or VARIANTS[args.variant]
    sample_dir = args.sample_dir.resolve()
    jacobian_root = args.jacobian_dir.resolve() / model_label
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calculator = ESMCJacobian(
        variant=args.variant,
        model_name=model_name,
        device=REMAPPED_DEVICE,
        substitution_batch_size=args.substitution_batch_size,
    )
    print(
        f"Loaded {model_label} ({model_name}); "
        f"alphabet={''.join(calculator.alphabet)}",
        flush=True,
    )

    for protein in args.proteins:
        output_path = (
            output_dir / "replicates" / model_label / f"{protein}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows, completed = common.load_existing(output_path)
        with (sample_dir / f"{protein}.json").open() as handle:
            sample_data = json.load(handle)

        query_aligned = sample_data["query"]["aligned_sequence"]
        query_archive = jacobian_root / protein / "query.npz"
        print(f"[{model_label}] {protein}: query", flush=True)
        reference_jacobian, reference_contact = common.load_or_generate_query(
            calculator, query_aligned, query_archive
        )

        for bin_index in range(1, 9):
            for replicate, sample in enumerate(
                sample_data["samples"][str(bin_index)]
            ):
                key = (protein, str(bin_index), str(replicate))
                archive_path = (
                    jacobian_root
                    / protein
                    / f"{bin_index}_{replicate}.npz"
                )
                print(
                    f"[{model_label}] {protein} "
                    f"bin={bin_index} rep={replicate}",
                    flush=True,
                )
                sample_jacobian = common.load_or_generate_sample(
                    calculator,
                    sample["aligned_sequence"],
                    archive_path,
                )
                if key not in completed:
                    coefficient, n_aligned, n_masked = (
                        common.score_jacobians(
                            reference_contact,
                            reference_jacobian,
                            sample_jacobian,
                            sample["aligned_sequence"],
                        )
                    )
                    rows.append(
                        {
                            "model": model_label,
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
                    common.write_rows(output_path, rows)
                del sample_jacobian
                gc.collect()

        del reference_contact, reference_jacobian
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Wrote {len(rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
