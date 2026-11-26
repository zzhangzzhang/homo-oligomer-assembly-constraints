#!/usr/bin/env python3
"""Shared, path-independent AF-Multimer v3 inference implementation."""

from __future__ import annotations

import gc
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np


MaskStrategy = Literal[
    "none",
    "offdiag",
    "ondiag",
    "inter_expand",
    "intra_expand",
]


def load_pdb_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "pdb_id" not in rows[0]:
            raise ValueError(f"CSV cohort lacks pdb_id: {path}")
        values = [row["pdb_id"] for row in rows]
    else:
        with path.open() as handle:
            values = json.load(handle)
    if isinstance(values, dict):
        values = list(values)
    return [str(value).lower() for value in values]


def read_query_sequence(path: Path) -> str:
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            sequence = line.strip().upper()
            if sequence:
                return sequence
    raise ValueError(f"No query sequence in {path}")


def run_hhfilter(
    input_path: Path,
    output_path: Path,
    qid: int,
    identity: int = 90,
    coverage: int = 75,
) -> None:
    subprocess.run(
        [
            "hhfilter",
            "-qid",
            str(qid),
            "-id",
            str(identity),
            "-cov",
            str(coverage),
            "-i",
            str(input_path),
            "-o",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"hhfilter did not create {output_path}")


def expand_binary_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    expanded = np.zeros_like(mask, dtype=bool)
    rows, columns = np.where(mask)
    length = mask.shape[0]
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            shifted_rows = np.clip(rows + row_offset, 0, length - 1)
            shifted_columns = np.clip(columns + column_offset, 0, length - 1)
            expanded[shifted_rows, shifted_columns] = True
    return expanded


def build_covariance_mask(
    strategy: MaskStrategy,
    contact_dir: Path,
    pdb_id: str,
    query_length: int,
    copies: int = 2,
) -> np.ndarray | None:
    if strategy == "none":
        return None

    inter = np.load(contact_dir / f"{pdb_id}-assembly1_inter.npy")
    if inter.shape != (query_length, query_length):
        raise ValueError(
            f"{pdb_id}: inter matrix {inter.shape} does not match "
            f"query length {query_length}"
        )
    intra = None
    if strategy in {"inter_expand", "intra_expand"}:
        intra = np.load(contact_dir / f"{pdb_id}-assembly1_intra.npy")
        if intra.shape != inter.shape:
            raise ValueError(
                f"{pdb_id}: intra/inter matrix shapes differ: "
                f"{intra.shape}/{inter.shape}"
            )

    length = query_length
    full_length = copies * length
    if strategy == "ondiag":
        full_mask = np.ones((full_length, full_length), dtype=float)
        for copy_index in range(copies):
            start = copy_index * length
            full_mask[start : start + length, start : start + length] = 0
        return full_mask

    full_mask = np.zeros((full_length, full_length), dtype=float)
    diagonal_block = np.ones((length, length), dtype=float)
    if strategy == "inter_expand":
        exclusive = (inter == 1) & (intra == 0)
        diagonal_block[expand_binary_mask(exclusive)] = 0
    elif strategy == "intra_expand":
        exclusive = (intra == 1) & (inter == 0)
        diagonal_block[expand_binary_mask(exclusive)] = 0
    elif strategy != "offdiag":
        raise ValueError(f"Unknown mask strategy: {strategy}")

    for copy_index in range(copies):
        start = copy_index * length
        full_mask[start : start + length, start : start + length] = diagonal_block
    return full_mask


def run_prediction(
    *,
    pdb_id: str,
    qid: int,
    query_bias: bool,
    mask_strategy: MaskStrategy,
    msa_dir: Path,
    contact_dir: Path | None,
    output_root: Path,
    num_seeds: int = 5,
    num_recycles: int = 6,
    recycle_tolerance: float = 0.5,
    copies: int = 2,
    model_names: list[str] | None = None,
) -> None:
    from colabdesign import clear_mem, mk_af_model
    from colabdesign.af.contrib import predict
    from colabdesign.shared.protein import _np_rmsd

    suffix = "" if mask_strategy == "none" else "_mask100"
    job_name = f"{pdb_id}_qid{qid}_qb{query_bias}{suffix}"
    output_dir = output_root / pdb_id / job_name
    pdb_dir = output_dir / "pdb"
    recycle_dir = pdb_dir / "recycles"
    best_archive = pdb_dir / "best.npz"
    if best_archive.is_file():
        with np.load(best_archive, allow_pickle=True) as archive:
            previous_seeds = (
                int(archive["num_seeds"]) if "num_seeds" in archive else 1
            )
            previous_models = (
                set(archive["model_names"].astype(str).tolist())
                if "model_names" in archive
                else None
            )
        models_satisfied = (
            model_names is None
            and (previous_models is None or len(previous_models) == 5)
        ) or (
            model_names is not None
            and previous_models is not None
            and set(model_names).issubset(previous_models)
        )
        if previous_seeds >= num_seeds and models_satisfied:
            print(f"Skipping completed {job_name}")
            return

    recycle_dir.mkdir(parents=True, exist_ok=True)
    source_msa = msa_dir / f"{pdb_id}_id90.a3m"
    if not source_msa.is_file():
        raise FileNotFoundError(f"Missing MSA: {source_msa}")
    sequence = read_query_sequence(source_msa)
    filtered_msa = output_dir / "msa.filt.a3m"
    run_hhfilter(source_msa, filtered_msa, qid=qid)
    msa, deletion_matrix = predict.parse_a3m(str(filtered_msa))

    rank_by = "multi" if copies > 1 else "plddt"
    model_options = {
        "num_msa": 512,
        "num_extra_msa": 1024,
        "num_templates": 1,
        "use_cluster_profile": True,
        "model_type": "alphafold2_multimer_v3",
        "use_templates": False,
        "protocol": "hallucination",
        "best_metric": rank_by,
        "optimize_seq": False,
        "query_bias": query_bias,
        "debug": False,
    }
    clear_mem()
    af = mk_af_model(use_mlm=True, **model_options)
    af.prep_inputs([len(sequence)], copies=copies, seed=0)
    af.set_msa(msa, deletion_matrix)

    previous_length = 0
    for copy_index in range(copies):
        if copy_index > 0:
            af._inputs["residue_index"][previous_length:] += 32
        previous_length += len(sequence)

    if mask_strategy != "none":
        if contact_dir is None:
            raise ValueError("A contact directory is required for masking")
        covariance_mask = build_covariance_mask(
            mask_strategy,
            contact_dir=contact_dir,
            pdb_id=pdb_id,
            query_length=len(sequence),
            copies=copies,
        )
        af._inputs["cov_mask"] = covariance_mask

    af.set_opt("mlm", replace_fraction=0.15)
    af._tmp = {
        "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
        "log": [],
        "best": {},
    }
    prediction_info = []
    print_keys = ("plddt", "ptm", "i_ptm", "multi")
    models_to_run = list(af._model_names)
    if model_names is not None:
        unknown = set(model_names) - set(models_to_run)
        if unknown:
            raise ValueError(
                f"Unknown model names {sorted(unknown)}; available={models_to_run}"
            )
        models_to_run = [name for name in models_to_run if name in model_names]
    with (output_dir / "log.txt").open("w") as log_handle:
        for seed in range(num_seeds):
            af.set_seed(seed)
            for model_name in models_to_run:
                recycle = 0
                af._inputs.pop("prev", None)
                previous_positions = None
                stop_recycling = False
                while recycle < num_recycles + 1:
                    af.predict(dropout=False, models=[model_name], verbose=False)
                    af._inputs["prev"] = af.aux["prev"]
                    if copies > 1 and "i_ptm" in af.aux["log"]:
                        af.aux["log"]["multi"] = (
                            0.8 * af.aux["log"]["i_ptm"]
                            + 0.2 * af.aux["log"]["ptm"]
                        )
                    message = (
                        f"seed={seed} model={model_name} recycle={recycle}"
                    )
                    for key in print_keys:
                        if key in af.aux["log"]:
                            message += f" {key}={af.aux['log'][key]:.3f}"

                    current_positions = af.aux["atom_positions"][:, 1]
                    if recycle > 0:
                        rmsd = _np_rmsd(
                            previous_positions,
                            current_positions,
                            use_jax=False,
                        )
                        stop_recycling = rmsd < recycle_tolerance
                        message += f" rmsd_tol={rmsd:.3f}"
                    previous_positions = current_positions
                    log_handle.write(message + "\n")
                    af.save_current_pdb(
                        str(
                            recycle_dir
                            / f"{model_name}_r{recycle}_seed{seed}.pdb"
                        )
                    )
                    if recycle == num_recycles or stop_recycling:
                        tag = f"{model_name}_seed{seed}"
                        prediction_info.append(
                            [tag, message, af.aux["log"][rank_by]]
                        )
                        af._save_results(
                            save_best=True,
                            best_metric=rank_by,
                            metric_higher_better=True,
                            verbose=False,
                        )
                        af.save_current_pdb(str(pdb_dir / f"{tag}.pdb"))
                        af._k += 1
                    if stop_recycling:
                        break
                    recycle += 1

    ranking = np.argsort([row[2] for row in prediction_info])[::-1]
    best = af._tmp["best"]["aux"]
    af.save_pdb(str(pdb_dir / "best.pdb"))
    np.savez_compressed(
        best_archive,
        plddt=best["plddt"].astype(np.float16),
        pae=best.get("pae", np.array([])).astype(np.float16),
        tag=np.array(prediction_info[ranking[0]][0]),
        metrics=np.array(prediction_info[ranking[0]][1]),
        num_seeds=np.array(num_seeds),
        model_names=np.array(models_to_run),
    )
    del af
    clear_mem()
    gc.collect()


def configure_cuda(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
