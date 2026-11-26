#!/usr/bin/env python3
"""Generate full categorical-Jacobian contact maps with ESM-C."""

from __future__ import annotations

import argparse
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

import numpy as np
import torch
from tqdm import tqdm

from _catjac_contact_common import (
    TQDM_BAR_FORMAT,
    clean_sequence,
    jacobian_to_contacts,
    load_dataset,
    save_contact_map,
    select_shard,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["300m", "600m", "6b"], required=True)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Registered model name for 300M/600M; pinned local snapshot or "
            "exact repository identifier for 6B."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ids-path",
        type=Path,
        default=project_root / "data" / "evaluation" / "all_pdb_ids.json",
    )
    parser.add_argument(
        "--sequences-path",
        type=Path,
        default=project_root / "data" / "evaluation" / "sequences.json",
    )
    parser.add_argument(
        "--substitution-batch-size",
        type=int,
        default=0,
        help="Substitutions per forward pass; 0 selects a length-based value.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
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
    inverse_vocabulary = {
        token_index: token for token, token_index in tokenizer.get_vocab().items()
    }
    return [inverse_vocabulary[index] for index in range(4, 24)]


def load_esm_backend(model_name: str, device: torch.device):
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(model_name, device=device).eval()
    model = model.to(dtype=torch.float32)
    return model, alphabet_from_tokenizer(model.tokenizer)


def load_transformers_backend(model_name: str, device: torch.device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, dtype=dtype
    ).to(device).eval()
    return model, tokenizer, alphabet_from_tokenizer(tokenizer)


@torch.no_grad()
def categorical_jacobian_esm(
    sequence: str,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    from esm.sdk.api import ESMProtein

    tokens = model.encode(ESMProtein(sequence=sequence)).sequence.to(device)
    length = len(sequence)

    def logits(input_tokens: torch.Tensor) -> np.ndarray:
        values = model(sequence_tokens=input_tokens).sequence_logits[
            ..., 1 : length + 1, 4:24
        ]
        return values.float().cpu().numpy()

    wild_type = logits(tokens[None])[0]
    jacobian = np.empty((length, 20, length, 20), dtype=np.float32)
    for position in tqdm(
        range(length),
        leave=False,
        bar_format=TQDM_BAR_FORMAT,
        desc="Residue substitutions",
    ):
        for start in range(0, 20, batch_size):
            stop = min(start + batch_size, 20)
            perturbed = tokens.unsqueeze(0).repeat(stop - start, 1)
            perturbed[:, position + 1] = torch.arange(
                4 + start, 4 + stop, device=device
            )
            jacobian[position, start:stop] = logits(perturbed)
    return jacobian - wild_type


@torch.no_grad()
def categorical_jacobian_transformers(
    sequence: str,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokens = tokenizer(sequence, return_tensors="pt")["input_ids"][0].to(device)
    length = len(sequence)
    if tokens.shape[0] != length + 2:
        raise ValueError(
            f"Expected {length + 2} tokens including BOS/EOS, got {tokens.shape[0]}"
        )

    def logits(input_tokens: torch.Tensor) -> np.ndarray:
        values = model(input_ids=input_tokens).logits[
            ..., 1 : length + 1, 4:24
        ]
        return values.float().cpu().numpy()

    wild_type = logits(tokens[None])[0]
    jacobian = np.empty((length, 20, length, 20), dtype=np.float32)
    for position in tqdm(
        range(length),
        leave=False,
        bar_format=TQDM_BAR_FORMAT,
        desc="Residue substitutions",
    ):
        for start in range(0, 20, batch_size):
            stop = min(start + batch_size, 20)
            perturbed = tokens.unsqueeze(0).repeat(stop - start, 1)
            perturbed[:, position + 1] = torch.arange(
                4 + start, 4 + stop, device=device
            )
            jacobian[position, start:stop] = logits(perturbed)
    return jacobian - wild_type


def main() -> None:
    args = parse_args()
    if args.substitution_batch_size < 0:
        raise ValueError("--substitution-batch-size cannot be negative")
    if REMAPPED_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(REMAPPED_DEVICE)
    output_dir = args.output_dir.resolve()

    pdb_ids, sequences = load_dataset(
        args.ids_path.resolve(), args.sequences_path.resolve()
    )
    selected = select_shard(
        pdb_ids,
        args.shard,
        args.num_shards,
        args.reverse,
        {pdb_id.lower() for pdb_id in args.exclude},
    )

    if args.variant == "6b":
        model, tokenizer, model_alphabet = load_transformers_backend(
            args.model, device
        )
    else:
        model, model_alphabet = load_esm_backend(args.model, device)
        tokenizer = None

    for pdb_id in tqdm(selected, desc="Proteins", bar_format=TQDM_BAR_FORMAT):
        output_path = output_dir / f"{pdb_id}.npy"
        if output_path.is_file() and not args.overwrite:
            continue
        if pdb_id not in sequences:
            raise KeyError(f"No sequence for {pdb_id}")
        sequence = clean_sequence(sequences[pdb_id])
        batch_size = (
            args.substitution_batch_size
            if args.substitution_batch_size
            else automatic_batch_size(len(sequence), args.variant)
        )
        if args.variant == "6b":
            jacobian = categorical_jacobian_transformers(
                sequence, model, tokenizer, device, batch_size
            )
        else:
            jacobian = categorical_jacobian_esm(
                sequence, model, device, batch_size
            )
        contacts = jacobian_to_contacts(jacobian, model_alphabet)
        save_contact_map(output_path, contacts, len(sequence))


if __name__ == "__main__":
    main()
