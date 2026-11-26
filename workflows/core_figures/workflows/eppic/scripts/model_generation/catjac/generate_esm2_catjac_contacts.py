#!/usr/bin/env python3
"""Generate full categorical-Jacobian contact maps with ESM2."""

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

import esm
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
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--substitution-batch-size", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device):
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    return model.to(device).eval(), alphabet


@torch.no_grad()
def categorical_jacobian(
    sequence: str,
    model,
    alphabet,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return the L×20×L×20 all-substitution logit Jacobian."""
    _, _, tokens = alphabet.get_batch_converter()([("sequence", sequence)])
    tokens = tokens.to(device)
    length = len(sequence)

    def logits(input_tokens: torch.Tensor) -> np.ndarray:
        values = model(input_tokens)["logits"][..., 1 : length + 1, 4:24]
        return values.float().cpu().numpy()

    wild_type = logits(tokens)[0]
    jacobian = np.empty((length, 20, length, 20), dtype=np.float32)
    for position in tqdm(
        range(length),
        leave=False,
        bar_format=TQDM_BAR_FORMAT,
        desc="Residue substitutions",
    ):
        for start in range(0, 20, batch_size):
            stop = min(start + batch_size, 20)
            perturbed = tokens.repeat(stop - start, 1)
            perturbed[:, position + 1] = torch.arange(
                4 + start, 4 + stop, device=device
            )
            jacobian[position, start:stop] = logits(perturbed)
    return jacobian - wild_type


def main() -> None:
    args = parse_args()
    if args.substitution_batch_size < 1:
        raise ValueError("--substitution-batch-size must be positive")
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
    model, alphabet = load_model(args.checkpoint.resolve(), device)
    model_alphabet = list(alphabet.all_toks[4:24])

    for pdb_id in tqdm(selected, desc="Proteins", bar_format=TQDM_BAR_FORMAT):
        output_path = output_dir / f"{pdb_id}.npy"
        if output_path.is_file() and not args.overwrite:
            continue
        if pdb_id not in sequences:
            raise KeyError(f"No sequence for {pdb_id}")
        sequence = clean_sequence(sequences[pdb_id])
        jacobian = categorical_jacobian(
            sequence,
            model,
            alphabet,
            device,
            args.substitution_batch_size,
        )
        contacts = jacobian_to_contacts(jacobian, model_alphabet)
        save_contact_map(output_path, contacts, len(sequence))


if __name__ == "__main__":
    main()
