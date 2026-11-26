"""Shared categorical-Jacobian post-processing and dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CANONICAL_ALPHABET = "AFILVMWYDEKRHNQSTGPC"
TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[elapsed: {elapsed} remaining: {remaining}]"
)


def clean_sequence(sequence: str) -> str:
    """Uppercase a sequence and retain alphabetic residue symbols."""
    cleaned = "".join(character for character in sequence.upper() if character.isalpha())
    if not cleaned:
        raise ValueError("Sequence is empty after cleaning")
    return cleaned


def jacobian_to_contacts(
    jacobian: np.ndarray,
    model_alphabet: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Convert an L×20×L×20 categorical Jacobian into an L×L contact score.

    This preserves the exact published pipeline: reorder amino-acid channels,
    symmetrize couplings, center over all four axes, take the Frobenius norm,
    remove the diagonal, and apply average-product correction (APC).
    """
    if jacobian.ndim != 4:
        raise ValueError(f"Expected a rank-4 Jacobian, received {jacobian.shape}")
    length_x, alphabet_x, length_y, alphabet_y = jacobian.shape
    if length_x != length_y or alphabet_x != 20 or alphabet_y != 20:
        raise ValueError(
            "Expected shape (L, 20, L, 20), received "
            f"{jacobian.shape}"
        )

    alphabet = list(model_alphabet)
    if len(alphabet) != 20 or set(alphabet) != set(CANONICAL_ALPHABET):
        raise ValueError(f"Invalid model amino-acid alphabet: {alphabet}")
    channel_order = [alphabet.index(amino_acid) for amino_acid in CANONICAL_ALPHABET]

    couplings = jacobian.copy()
    couplings = couplings[:, channel_order, :, :]
    couplings = couplings[:, :, :, channel_order]
    couplings = (couplings + couplings.transpose(2, 3, 0, 1)) / 2
    for axis in range(4):
        couplings -= couplings.mean(axis, keepdims=True)

    contacts = np.sqrt(np.square(couplings).sum((1, 3)))
    np.fill_diagonal(contacts, 0)
    total = contacts.sum()
    if not np.isfinite(total) or total == 0:
        raise ValueError("Cannot apply APC to a zero or non-finite contact matrix")
    average_product = (
        contacts.sum(0, keepdims=True)
        * contacts.sum(1, keepdims=True)
        / total
    )
    contacts = contacts - average_product
    np.fill_diagonal(contacts, 0)
    contacts = contacts.astype(np.float32, copy=False)
    if not np.isfinite(contacts).all():
        raise ValueError("Contact matrix contains non-finite values")
    return contacts


def load_dataset(ids_path: Path, sequences_path: Path) -> tuple[list[str], dict[str, str]]:
    with ids_path.open() as handle:
        pdb_ids = [str(pdb_id).lower() for pdb_id in json.load(handle)]
    with sequences_path.open() as handle:
        raw_sequences = json.load(handle)
    sequences = {str(key).lower(): str(value) for key, value in raw_sequences.items()}
    return pdb_ids, sequences


def select_shard(
    pdb_ids: list[str],
    shard: int,
    num_shards: int,
    reverse: bool,
    excluded: set[str],
) -> list[str]:
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"Invalid shard {shard} for {num_shards} shards")
    selected = [pdb_id for pdb_id in pdb_ids if pdb_id not in excluded]
    selected = selected[shard::num_shards]
    return selected[::-1] if reverse else selected


def save_contact_map(path: Path, contacts: np.ndarray, expected_length: int) -> None:
    expected_shape = (expected_length, expected_length)
    if contacts.shape != expected_shape:
        raise ValueError(f"Contact shape {contacts.shape}; expected {expected_shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, contacts.astype(np.float32, copy=False))
