#!/usr/bin/env python3
"""Run canonical no-mask and state-exclusive-mask AF2-pTM conditions.

This consolidates the historical 06/06b/06c launch branches into one
condition-level resumable runner without changing the reported model settings.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


CONDITIONS = ("no_mask", "mask_excl1", "mask_excl2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-dict", type=Path, required=True)
    parser.add_argument("--exclusive-dir", type=Path, required=True)
    parser.add_argument("--msa-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=["no_mask", "mask_excl1"],
        help="conditions to run; defaults to the two conditions shown in S4",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="ID1_ID2",
        help="restrict to one or more conformation pairs",
    )
    parser.add_argument(
        "--prediction-id",
        action="append",
        default=[],
        help="restrict predictions to selected PDB/query IDs",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--passes", type=int, default=7)
    parser.add_argument("--recycle-tolerance", type=float, default=0.5)
    parser.add_argument("--qid", type=int, default=0)
    parser.add_argument("--max-pairwise-identity", type=int, default=90)
    parser.add_argument("--coverage", type=int, default=75)
    parser.add_argument("--mask-rate", type=float, default=1.0)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--max-sequence-length", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_query_sequence(seq_dict: dict, pdb_id: str) -> str:
    try:
        return str(seq_dict[pdb_id]["1"])
    except KeyError as exc:
        raise KeyError(f"No entity-1 sequence for {pdb_id}") from exc


def build_full_mask(
    aligned_mask: np.ndarray,
    alignment_indices: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """Map an aligned-frame binary mask to full sequence coordinates."""
    result = np.zeros((sequence_length, sequence_length), dtype=np.int8)
    indices = np.asarray(alignment_indices, dtype=int)
    rows, columns = np.where(np.asarray(aligned_mask) == 1)
    result[indices[rows], indices[columns]] = 1
    return result


def filtered_msa(
    *,
    source: Path,
    destination: Path,
    qid: int,
    pairwise_identity: int,
    coverage: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "hhfilter",
            "-qid",
            str(qid),
            "-id",
            str(pairwise_identity),
            "-cov",
            str(coverage),
            "-i",
            str(source),
            "-o",
            str(destination),
        ],
        check=True,
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"HHfilter produced no MSA: {destination}")


def run_condition(
    *,
    pdb_id: str,
    pair_tag: str,
    condition: str,
    binary_mask: Optional[np.ndarray],
    sequence: str,
    args: argparse.Namespace,
) -> None:
    from colabdesign import clear_mem, mk_af_model
    from colabdesign.af.contrib import predict
    from colabdesign.shared.protein import _np_rmsd

    job_name = f"{pdb_id}_{pair_tag}_{condition}"
    job_dir = args.output_dir / job_name
    pdb_dir = job_dir / "pdb"
    recycle_dir = pdb_dir / "recycles"
    best_archive = pdb_dir / "best.npz"
    if best_archive.is_file() and not args.overwrite:
        print(f"SKIP {job_name}: completed")
        return

    source_msa = args.msa_dir / f"{pdb_id.lower()}_raw.a3m"
    if not source_msa.is_file():
        raise FileNotFoundError(source_msa)
    recycle_dir.mkdir(parents=True, exist_ok=True)
    msa_path = job_dir / "msa.filt.a3m"
    filtered_msa(
        source=source_msa,
        destination=msa_path,
        qid=args.qid,
        pairwise_identity=args.max_pairwise_identity,
        coverage=args.coverage,
    )
    msa, deletion_matrix = predict.parse_a3m(str(msa_path))

    clear_mem()
    model = mk_af_model(
        use_mlm=True,
        num_msa=512,
        num_extra_msa=1024,
        num_templates=1,
        use_cluster_profile=True,
        model_type="alphafold2_ptm",
        use_templates=False,
        protocol="hallucination",
        best_metric="plddt",
        optimize_seq=False,
        query_bias=True,
        debug=False,
        data_dir=str(args.params_dir),
    )
    model.prep_inputs([len(sequence)], copies=1, seed=0)
    model.set_msa(msa, deletion_matrix)
    if binary_mask is not None:
        covariance_mask = np.ones((len(sequence), len(sequence)), dtype=float)
        covariance_mask[np.asarray(binary_mask) == 1] = 1.0 - args.mask_rate
        model._inputs["cov_mask"] = covariance_mask

    model.set_opt("mlm", replace_fraction=0.15)
    model._tmp = {
        "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
        "log": [],
        "best": {},
    }
    model_names = list(model._model_names)
    if args.models:
        unknown = set(args.models) - set(model_names)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        model_names = [name for name in model_names if name in args.models]

    info = []
    with (job_dir / "log.txt").open("w", encoding="utf-8") as handle:
        for seed in args.seeds:
            model.set_seed(seed)
            for model_name in model_names:
                model._inputs.pop("prev", None)
                previous_positions = None
                for pass_index in range(args.passes):
                    model.predict(
                        dropout=False,
                        models=[model_name],
                        verbose=False,
                    )
                    model._inputs["prev"] = model.aux["prev"]
                    message = (
                        f"seed={seed} model={model_name} pass={pass_index}"
                        f" plddt={model.aux['log']['plddt']:.3f}"
                        f" ptm={model.aux['log']['ptm']:.3f}"
                    )
                    current_positions = model.aux["atom_positions"][:, 1]
                    stop = False
                    if previous_positions is not None:
                        rmsd = _np_rmsd(
                            previous_positions,
                            current_positions,
                            use_jax=False,
                        )
                        message += f" rmsd_tol={rmsd:.3f}"
                        stop = rmsd < args.recycle_tolerance
                    previous_positions = current_positions
                    handle.write(message + "\n")
                    model.save_current_pdb(
                        str(
                            recycle_dir
                            / f"{model_name}_r{pass_index}_seed{seed}.pdb"
                        )
                    )
                    if pass_index == args.passes - 1 or stop:
                        tag = f"{model_name}_seed{seed}"
                        info.append(
                            [tag, message, float(model.aux["log"]["plddt"])]
                        )
                        model._save_results(
                            save_best=True,
                            best_metric="plddt",
                            metric_higher_better=True,
                            verbose=False,
                        )
                        model.save_current_pdb(str(pdb_dir / f"{tag}.pdb"))
                        model._k += 1
                        break

    rank = np.argsort([row[2] for row in info])[::-1]
    best = model._tmp["best"]["aux"]
    model.save_pdb(str(pdb_dir / "best.pdb"))
    np.savez_compressed(
        best_archive,
        plddt=best["plddt"].astype(np.float16),
        pae=best.get("pae", np.array([])).astype(np.float16),
        tag=np.array(info[rank[0]][0]),
        metrics=np.array(info[rank[0]][1]),
        models=np.array(model_names),
        seeds=np.array(args.seeds),
        condition=np.array(condition),
    )
    print(f"DONE {job_name}")
    del model
    clear_mem()
    gc.collect()


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not 0.0 <= args.mask_rate <= 1.0:
        raise ValueError("--mask-rate must be between zero and one")
    if args.passes < 1:
        raise ValueError("--passes must be positive")
    seq_dict = json.loads(args.seq_dict.read_text(encoding="utf-8"))
    requested_pairs = {value.upper() for value in args.pair}
    requested_ids = {value.upper() for value in args.prediction_id}
    mask_files = sorted(args.exclusive_dir.glob("*.npz"))
    if not mask_files:
        raise FileNotFoundError(f"No NPZ masks in {args.exclusive_dir}")

    for mask_path in mask_files:
        parts = mask_path.stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected mask filename: {mask_path.name}")
        _, id1, id2 = parts[:3]
        id1, id2 = id1.upper(), id2.upper()
        pair_tag = f"{id1}_{id2}"
        if requested_pairs and pair_tag not in requested_pairs:
            continue
        data = np.load(mask_path)
        entries = [
            (id1, data["aln_idx1"]),
            (id2, data["aln_idx2"]),
        ]
        for pdb_id, indices in entries:
            if requested_ids and pdb_id not in requested_ids:
                continue
            sequence = read_query_sequence(seq_dict, pdb_id)
            if len(sequence) > args.max_sequence_length:
                print(f"SKIP {pdb_id}: length {len(sequence)}")
                continue
            masks = {
                "no_mask": None,
                "mask_excl1": build_full_mask(
                    data["excl1"], indices, len(sequence)
                ),
                "mask_excl2": build_full_mask(
                    data["excl2"], indices, len(sequence)
                ),
            }
            for condition in args.conditions:
                run_condition(
                    pdb_id=pdb_id,
                    pair_tag=pair_tag,
                    condition=condition,
                    binary_mask=masks[condition],
                    sequence=sequence,
                    args=args,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
