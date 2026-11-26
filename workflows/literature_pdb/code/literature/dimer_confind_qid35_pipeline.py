#!/usr/bin/env python3
"""Resumable dimer-only ConFind/qid35 lattice-interface pipeline.

Stages:
  enumerate  Outcome-blind robust ASU/XTAL enumeration and workload manifests.
  run        Full-atom two-chain P1 ConFind plus candidate-specific qid35 score.
  finalize   Rebuild deterministic summaries from per-job atomic JSON records.
  canary     Run one small primary candidate per literature class and 3O1N.
  self-check Validate manifests, completed jobs, and frozen scientific policy.

The production ``run`` stage is never invoked by ``enumerate`` or ``canary``
without an explicit bounded job selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Mapping

import gemmi
import numpy as np

from reverse_assembly_helpers import (
    parse_sifts,
)

STRUCTURAL_CODE = Path(__file__).resolve().parents[1] / "structural"
if str(STRUCTURAL_CODE) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_CODE))
from confind_bioassembly_screen import (  # noqa: E402
    parse_sifts_for_uniprot,
    prepare_confind_pdb,
)


DEFAULT_INPUT = Path(
    "/path/to/external-data/pdb_pairformer_screen/full_analysis/denominator_audit/"
    "monomer_three_class_confind_gate_v2_20260904/"
    "monomer_three_class_records.csv"
)
DEFAULT_OUT = Path(
    "/path/to/external-data/pdb_pairformer_screen/full_analysis/"
    "dimer_confind_qid35_v2_20260904"
)
DEFAULT_CONFIND = Path(
    "/path/to/external-home/sam/homooligomer/confind-msl-bin/confind"
)
DEFAULT_ROTLIB = Path(
    "/path/to/external-home/sam/homooligomer/confind-msl-bin/rotlibs/DB-2010"
)
SIFTS_DIR = Path(
    "/path/to/external-data/pdb_pairformer_screen/full_screen/sifts_xml"
)
HOMO_MANIFEST = Path(
    "/path/to/external-data/pdb_pairformer_screen/full_screen/manifest_homo.csv"
)
BIOASSEMBLY_DIR = Path("/path/to/external-data/pdb_bioassembly_all")
CLASS_ORDER = ("Monomer-only", "Oligomer-only", "Multiple states")
ENDPOINT_RE = re.compile(r"^([^,]+),(-?\d+)([A-Za-z]?)$")


def truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chain_ca(
    chain: gemmi.Chain, mapping: Mapping[int, int]
) -> tuple[np.ndarray, list[int]]:
    coords, positions = [], []
    for residue in chain.get_polymer():
        if not residue.label_seq:
            continue
        up = mapping.get(int(residue.label_seq))
        if up is None:
            continue
        atom = next((item for item in residue if item.name == "CA"), None)
        if atom is None:
            continue
        coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        positions.append(int(up))
    return np.asarray(coords, dtype=float), positions


def lattice_range(
    au_bbox: tuple[np.ndarray, np.ndarray],
    image_bbox: tuple[np.ndarray, np.ndarray],
    orth: np.ndarray,
    cutoff: float,
) -> list[tuple[int, int, int]]:
    vectors = (orth[:, 0], orth[:, 1], orth[:, 2])
    norms = tuple(np.linalg.norm(vector) for vector in vectors)
    au_min, au_max = au_bbox
    image_min, image_max = image_bbox
    span = (au_max - au_min) + (image_max - image_min) + 2 * cutoff
    n_max = max(2, int(np.ceil(float(np.max(span)) / min(norms))) + 1)
    valid = []
    for left in range(-n_max, n_max + 1):
        for middle in range(-n_max, n_max + 1):
            for right in range(-n_max, n_max + 1):
                shift = (
                    left * vectors[0]
                    + middle * vectors[1]
                    + right * vectors[2]
                )
                if np.all(image_max + shift + cutoff >= au_min) and np.all(
                    image_min + shift - cutoff <= au_max
                ):
                    valid.append((left, middle, right))
    return valid


def invariant_interface_signature(
    source_positions: list[int],
    partner_positions: list[int],
    indices: np.ndarray,
    squared_distances: np.ndarray,
) -> str:
    """Rigid-motion, translation, and chain-swap invariant interface identity."""
    contacts = sorted(
        (
            min(
                int(source_positions[int(source_index)]),
                int(partner_positions[int(partner_index)]),
            ),
            max(
                int(source_positions[int(source_index)]),
                int(partner_positions[int(partner_index)]),
            ),
            round(
                float(
                    np.sqrt(
                        squared_distances[
                            int(source_index), int(partner_index)
                        ]
                    )
                ),
                3,
            ),
        )
        for source_index, partner_index in indices
    )
    return hashlib.sha1(
        json.dumps(contacts, separators=(",", ":")).encode()
    ).hexdigest()


def enumerate_one(
    source: dict[str, str], cutoff: float, min_sep: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdb, up = source["pdb"].upper(), source["uniprot"].upper()
    structure = gemmi.read_structure(source["au_cif_path"])
    mapping = parse_sifts(source["sifts_path"], up)
    if len(structure) == 0:
        return [], {"status": "no_coordinate_model"}
    spacegroup = structure.find_spacegroup()
    if spacegroup is None or not structure.cell.is_crystal():
        return [], {"status": "not_crystallographic"}
    model = structure[0]
    chains = {
        chain.name: chain
        for chain in model
        if chain.name in mapping and mapping[chain.name]
    }
    tables = {
        name: chain_ca(chain, mapping[name]) for name, chain in chains.items()
    }
    tables = {name: value for name, value in tables.items() if len(value[0])}
    if not tables:
        return [], {"status": "no_mapped_ca"}
    frac = np.asarray(structure.cell.frac.mat, dtype=float)
    orth = np.asarray(structure.cell.orth.mat, dtype=float)
    all_coords = np.concatenate([value[0] for value in tables.values()])
    au_bbox = (all_coords.min(axis=0), all_coords.max(axis=0))
    grouped: dict[str, dict[str, Any]] = {}
    transforms_evaluated = 0
    for source_name in sorted(tables):
        source_coords, source_pos = tables[source_name]
        source_bbox = (source_coords.min(axis=0), source_coords.max(axis=0))
        for partner_name in sorted(tables):
            partner_coords, partner_pos = tables[partner_name]
            partner_frac = partner_coords @ frac.T
            for op_index, operation in enumerate(spacegroup.operations()):
                matrix = np.asarray(operation.rot, dtype=float) / operation.DEN
                translation = (
                    np.asarray(operation.tran, dtype=float) / operation.DEN
                )
                base = (partner_frac @ matrix.T + translation) @ orth.T
                image_bbox = (base.min(axis=0), base.max(axis=0))
                for lattice in lattice_range(
                    au_bbox, image_bbox, orth, cutoff
                ):
                    if (
                        op_index == 0
                        and lattice == (0, 0, 0)
                        and source_name == partner_name
                    ):
                        continue
                    shift = (
                        lattice[0] * orth[:, 0]
                        + lattice[1] * orth[:, 1]
                        + lattice[2] * orth[:, 2]
                    )
                    partner = base + shift
                    partner_bbox = (
                        partner.min(axis=0),
                        partner.max(axis=0),
                    )
                    if np.any(partner_bbox[1] + cutoff < source_bbox[0]) or np.any(
                        partner_bbox[0] - cutoff > source_bbox[1]
                    ):
                        continue
                    transforms_evaluated += 1
                    d2 = (
                        (
                            source_coords[:, None, :]
                            - partner[None, :, :]
                        )
                        ** 2
                    ).sum(axis=2)
                    indices = np.argwhere(d2 < cutoff * cutoff)
                    if not len(indices):
                        continue
                    scoreable = {
                        tuple(
                            sorted(
                                (
                                    int(source_pos[int(i)]),
                                    int(partner_pos[int(j)]),
                                )
                            )
                        )
                        for i, j in indices
                        if abs(
                            int(source_pos[int(i)])
                            - int(partner_pos[int(j)])
                        )
                        >= min_sep
                    }
                    signature = invariant_interface_signature(
                        source_pos, partner_pos, indices, d2
                    )
                    category = (
                        "ASU"
                        if op_index == 0
                        and lattice == (0, 0, 0)
                        and source_name != partner_name
                        else "XTAL"
                    )
                    transform_id = (
                        f"{source_name}-{partner_name}@{operation.triplet()}"
                        f"@({lattice[0]},{lattice[1]},{lattice[2]})"
                    )
                    candidate = {
                        "category": category,
                        "source_chain": source_name,
                        "partner_chain": partner_name,
                        "op_index": op_index,
                        "op_triplet": operation.triplet(),
                        "frac_matrix_json": json.dumps(matrix.tolist()),
                        "frac_translation_json": json.dumps(
                            translation.tolist()
                        ),
                        "lattice_json": json.dumps(list(lattice)),
                        "transform_id": transform_id,
                        "physical_signature": signature,
                        "raw_ca_atom_contacts": len(indices),
                        "scoreable_ca_pairs": len(scoreable),
                        "minimum_ca_distance_A": float(np.sqrt(d2.min())),
                    }
                    previous = grouped.get(signature)
                    key = lambda item: (
                        -item["scoreable_ca_pairs"],
                        -item["raw_ca_atom_contacts"],
                        item["minimum_ca_distance_A"],
                        item["transform_id"],
                    )
                    if previous is None or key(candidate) < key(previous):
                        grouped[signature] = candidate
    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            -item["scoreable_ca_pairs"],
            -item["raw_ca_atom_contacts"],
            item["minimum_ca_distance_A"],
            item["transform_id"],
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["interface_rank"] = rank
        row["physical_interface_count"] = len(ranked)
        row["interface_id"] = f"{pdb}_{row['category']}_i{rank:04d}"
    return ranked, {
        "status": "ok",
        "transforms_evaluated": transforms_evaluated,
        "physical_interfaces": len(ranked),
    }


def mapped_chain(
    original: gemmi.Chain,
    mapping: Mapping[int, int],
    name: str,
) -> gemmi.Chain:
    output = gemmi.Chain(name)
    seen: set[int] = set()
    for residue in original.get_polymer():
        if not residue.label_seq:
            continue
        up = mapping.get(int(residue.label_seq))
        if up is None or up in seen:
            continue
        seen.add(up)
        cloned = residue.clone()
        cloned.seqid = gemmi.SeqId(int(up), " ")
        cloned.label_seq = int(up)
        output.add_residue(cloned)
    if len(output) == 0:
        raise ValueError(f"no mapped residues for chain {original.name}")
    return output


def transform_chain(
    chain: gemmi.Chain,
    matrix: np.ndarray,
    translation: np.ndarray,
    lattice: np.ndarray,
    source: gemmi.Structure,
) -> None:
    frac = np.asarray(source.cell.frac.mat, dtype=float)
    orth = np.asarray(source.cell.orth.mat, dtype=float)
    for residue in chain:
        for atom in residue:
            point = np.asarray(
                [atom.pos.x, atom.pos.y, atom.pos.z], dtype=float
            )
            fractional = frac @ point
            result = orth @ (
                matrix @ fractional + translation + lattice
            )
            atom.pos = gemmi.Position(*map(float, result))


def write_candidate_pdb(job: Mapping[str, Any], path: Path) -> dict[str, Any]:
    structure = gemmi.read_structure(job["au_cif_path"])
    mapping = parse_sifts(job["sifts_path"], job["uniprot"])
    model = structure[0]
    source_original = next(
        chain for chain in model if chain.name == job["source_chain"]
    )
    partner_original = next(
        chain for chain in model if chain.name == job["partner_chain"]
    )
    chain_a = mapped_chain(
        source_original, mapping[job["source_chain"]], "A"
    )
    chain_b = mapped_chain(
        partner_original, mapping[job["partner_chain"]], "B"
    )
    transform_chain(
        chain_b,
        np.asarray(json.loads(job["frac_matrix_json"]), dtype=float),
        np.asarray(json.loads(job["frac_translation_json"]), dtype=float),
        np.asarray(json.loads(job["lattice_json"]), dtype=float),
        structure,
    )
    output = gemmi.Structure()
    output.name = job["interface_id"]
    output.cell = gemmi.UnitCell(200, 200, 200, 90, 90, 90)
    output.spacegroup_hm = "P 1"
    out_model = gemmi.Model("1")
    out_model.add_chain(chain_a)
    out_model.add_chain(chain_b)
    output.add_model(out_model)
    output.write_pdb(str(path))
    return {
        "chain_ids": ["A", "B"],
        "mapped_residues_A": len(chain_a),
        "mapped_residues_B": len(chain_b),
        "atoms_A": sum(len(residue) for residue in chain_a),
        "atoms_B": sum(len(residue) for residue in chain_b),
        "protein_only": True,
        "full_atom": True,
    }


def run_confind(
    pdb_path: Path,
    raw_path: Path,
    executable: Path,
    rotlib: Path,
    timeout: int,
) -> None:
    result = subprocess.run(
        [
            str(executable),
            "--p",
            str(pdb_path),
            "--o",
            str(raw_path),
            "--rLib",
            str(rotlib),
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ConFind exit {result.returncode}: "
            f"{(result.stderr or result.stdout)[-2000:]}"
        )
    if not raw_path.exists():
        raise RuntimeError("ConFind did not create raw output")


def parse_contacts(
    raw_path: Path, length: int, threshold: float, min_sep: int
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[str, np.ndarray], int]:
    intra: dict[tuple[int, int], float] = {}
    inter: dict[tuple[int, int], float] = {}
    chain1, position1, chain2, position2, scores, pairs_i, pairs_j, flags = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    parsed = 0
    with raw_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("contact"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            match1, match2 = ENDPOINT_RE.match(parts[1]), ENDPOINT_RE.match(parts[2])
            if match1 is None or match2 is None:
                continue
            score = float(parts[3])
            if score <= threshold:
                continue
            c1, p1 = match1.group(1), int(match1.group(2))
            c2, p2 = match2.group(1), int(match2.group(2))
            if c1 not in {"A", "B"} or c2 not in {"A", "B"}:
                continue
            if not (1 <= p1 <= length and 1 <= p2 <= length):
                continue
            parsed += 1
            left, right = sorted((p1, p2))
            is_inter = c1 != c2
            chain1.append(c1)
            position1.append(p1)
            chain2.append(c2)
            position2.append(p2)
            scores.append(score)
            pairs_i.append(left)
            pairs_j.append(right)
            flags.append(is_inter)
            target = inter if is_inter else intra
            pair = (left, right)
            target[pair] = max(score, target.get(pair, float("-inf")))
    intra = {
        pair: score for pair, score in intra.items() if pair[1] - pair[0] >= min_sep
    }
    inter = {
        pair: score for pair, score in inter.items() if pair[1] - pair[0] >= min_sep
    }
    arrays = {
        "chain1": np.asarray(chain1, dtype="U1"),
        "position1": np.asarray(position1, dtype=np.int32),
        "chain2": np.asarray(chain2, dtype="U1"),
        "position2": np.asarray(position2, dtype=np.int32),
        "score": np.asarray(scores, dtype=np.float32),
        "collapsed_i": np.asarray(pairs_i, dtype=np.int32),
        "collapsed_j": np.asarray(pairs_j, dtype=np.int32),
        "is_inter": np.asarray(flags, dtype=bool),
    }
    return intra, inter, arrays, parsed


def top_pairs(matrix: np.ndarray, count: int, min_sep: int) -> set[tuple[int, int]]:
    rows, columns = np.triu_indices(matrix.shape[0], k=min_sep)
    count = min(count, len(rows))
    if count == 0:
        return set()
    order = np.argsort(-matrix[rows, columns], kind="mergesort")[:count]
    return {
        (int(rows[index]) + 1, int(columns[index]) + 1)
        for index in order
    }


def score_qid35(
    matrix_path: str,
    length: int,
    intra: Mapping[tuple[int, int], float],
    inter: Mapping[tuple[int, int], float],
    min_sep: int,
) -> dict[str, Any]:
    matrix = np.loadtxt(matrix_path)
    if matrix.shape != (length, length):
        raise ValueError(
            f"qid35 shape {matrix.shape} does not match L_up={length}"
        )
    matrix = 0.5 * (matrix + matrix.T)
    intra_set, inter_set = set(intra), set(inter)
    union = intra_set | inter_set
    candidate_top = top_pairs(matrix, len(union), min_sep)
    intra_top = top_pairs(matrix, len(intra_set), min_sep)
    interface_hits = len(candidate_top & inter_set)
    intra_hits = len(candidate_top & intra_set)
    union_hits = len(candidate_top & union)
    return {
        "candidate_specific_x": len(union),
        "n_intra_confind_pairs": len(intra_set),
        "n_interface_confind_pairs": len(inter_set),
        "interface_hits": interface_hits,
        "interface_recall": (
            interface_hits / len(inter_set) if inter_set else None
        ),
        "inter_fraction_of_topx": (
            interface_hits / len(candidate_top) if candidate_top else None
        ),
        "intra_hits": intra_hits,
        "intra_recall_at_candidate_x": (
            intra_hits / len(intra_set) if intra_set else None
        ),
        "union_precision": (
            union_hits / len(candidate_top) if candidate_top else None
        ),
        "standardized_intra_top_i_hits": len(intra_top & intra_set),
        "standardized_intra_top_i_recall": (
            len(intra_top & intra_set) / len(intra_set)
            if intra_set
            else None
        ),
    }


def job_paths(out_dir: Path, interface_id: str) -> dict[str, Path]:
    directory = out_dir / "jobs" / interface_id
    return {
        "dir": directory,
        "done": directory / "done.json",
        "failure": directory / "failure.json",
        "input_gz": directory / "input.pdb.gz",
        "raw_gz": directory / "confind_raw.txt.gz",
        "contacts": directory / "contacts_lossless.npz",
    }


def run_job(job: dict[str, str], config: dict[str, Any]) -> tuple[str, str, float]:
    started = time.time()
    paths = job_paths(Path(config["out_dir"]), job["interface_id"])
    if paths["done"].exists() and config["resume"]:
        return job["interface_id"], "skipped", 0.0
    paths["dir"].mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=job["interface_id"] + "_") as tmp:
        temporary = Path(tmp)
        pdb_path = temporary / "input.pdb"
        raw_path = temporary / "raw.txt"
        try:
            prep = write_candidate_pdb(job, pdb_path)
            run_confind(
                pdb_path,
                raw_path,
                Path(config["confind"]),
                Path(config["rotlib"]),
                int(config["timeout"]),
            )
            intra, inter, arrays, n_parsed = parse_contacts(
                raw_path,
                int(job["L_up"]),
                float(config["threshold"]),
                int(config["min_sep"]),
            )
            metrics = score_qid35(
                job["qid35_matrix_path"],
                int(job["L_up"]),
                intra,
                inter,
                int(config["min_sep"]),
            )
            metadata = {
                "interface_id": job["interface_id"],
                "pdb": job["pdb"],
                "uniprot": job["uniprot"],
                "doi": job["doi"],
                "literature_class": job[
                    "direct_current_paper_exact_class"
                ],
                "category": job["category"],
                "transform_id": job["transform_id"],
                "source_chain": job["source_chain"],
                "partner_chain": job["partner_chain"],
                "frac_matrix": json.loads(job["frac_matrix_json"]),
                "frac_translation": json.loads(
                    job["frac_translation_json"]
                ),
                "lattice": json.loads(job["lattice_json"]),
                "threshold_strictly_greater_than": config["threshold"],
                "min_sequence_separation": config["min_sep"],
            }
            with open(paths["contacts"].with_suffix(".npz.part"), "wb") as handle:
                np.savez_compressed(
                    handle,
                    **arrays,
                    metadata_json=np.asarray(json.dumps(metadata)),
                )
            os.replace(
                paths["contacts"].with_suffix(".npz.part"),
                paths["contacts"],
            )
            with pdb_path.open("rb") as source, gzip.open(
                paths["input_gz"], "wb"
            ) as destination:
                shutil.copyfileobj(source, destination)
            with raw_path.open("rb") as source, gzip.open(
                paths["raw_gz"], "wb"
            ) as destination:
                shutil.copyfileobj(source, destination)
            result = {
                **{key: job[key] for key in job},
                **prep,
                **metrics,
                "confind_records_above_threshold": n_parsed,
                "elapsed_seconds": time.time() - started,
                "input_pdb_gz": str(paths["input_gz"]),
                "raw_confind_gz": str(paths["raw_gz"]),
                "lossless_contacts_npz": str(paths["contacts"]),
                "status": "success",
            }
            atomic_json(paths["done"], result)
            if paths["failure"].exists():
                paths["failure"].unlink()
            return job["interface_id"], "success", result["elapsed_seconds"]
        except Exception as error:
            atomic_json(
                paths["failure"],
                {
                    "interface_id": job["interface_id"],
                    "pdb": job["pdb"],
                    "stage": "run_job",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.time() - started,
                },
            )
            return job["interface_id"], "failure", time.time() - started


def enumerate_stage(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_rows = read_csv(args.input)
    monomer_manifest = {
        (row["pdb"].upper(), row["uniprot"].upper()): row
        for row in read_csv(args.monomer_manifest)
    }
    candidates, pdb_rows = [], []
    for index, source in enumerate(source_rows, 1):
        manifest = monomer_manifest[(source["pdb"], source["uniprot"])]
        if int(manifest["L_up"]) > 500:
            raise RuntimeError(f"length >500: {source['pdb']}")
        merged = {
            **source,
            **manifest,
            "sifts_path": str(
                args.sifts_dir / f"{source['pdb'].lower()}.xml.gz"
            ),
            "cohort": (
                "primary_strict_monomer"
                if truth(source["strict_all_deposited_assemblies_monomer"])
                else "sensitivity_nonmonomeric_recheck"
            ),
        }
        try:
            interfaces, accounting = enumerate_one(
                merged, args.ca_cutoff, args.min_sep
            )
            status = accounting["status"]
        except Exception as error:
            interfaces = []
            accounting = {"status": f"error:{type(error).__name__}:{error}"}
            status = accounting["status"]
        passing = [
            row
            for row in interfaces
            if row["scoreable_ca_pairs"] >= args.minimum_ca_pairs
        ]
        for row in passing:
            candidates.append(
                {
                    **merged,
                    **row,
                    "candidate_count_ge10_for_pdb": len(passing),
                    "candidate_selection_policy": (
                        f"outcome_blind_raw_CA<{args.ca_cutoff:g}A_"
                        f"scoreable_pairs>={args.minimum_ca_pairs}"
                    ),
                }
            )
        pdb_rows.append(
            {
                "pdb": merged["pdb"],
                "uniprot": merged["uniprot"],
                "doi": merged["doi"],
                "direct_current_paper_exact_class": merged[
                    "direct_current_paper_exact_class"
                ],
                "cohort": merged["cohort"],
                "L_up": merged["L_up"],
                "enumeration_status": status,
                "physical_interfaces": len(interfaces),
                "interfaces_ge10": len(passing),
                "transforms_evaluated": accounting.get(
                    "transforms_evaluated", 0
                ),
            }
        )
        if index % 100 == 0:
            print(f"enumerated {index:,}/{len(source_rows):,}", flush=True)
    candidates.sort(
        key=lambda row: (
            row["cohort"],
            row["pdb"],
            int(row["interface_rank"]),
        )
    )
    atomic_csv(args.out_dir / "candidate_manifest_all.csv", candidates)
    for cohort, filename in [
        ("primary_strict_monomer", "candidate_manifest_primary.csv"),
        (
            "sensitivity_nonmonomeric_recheck",
            "candidate_manifest_sensitivity.csv",
        ),
    ]:
        rows = [row for row in candidates if row["cohort"] == cohort]
        if rows:
            atomic_csv(args.out_dir / filename, rows)
    atomic_csv(args.out_dir / "pdb_enumeration_accounting.csv", pdb_rows)
    workload = {}
    for cohort in [
        "primary_strict_monomer",
        "sensitivity_nonmonomeric_recheck",
    ]:
        sources = [row for row in pdb_rows if row["cohort"] == cohort]
        jobs = [row for row in candidates if row["cohort"] == cohort]
        workload[cohort] = {
            "pdb_records": len(sources),
            "unique_doi_uniprot_keys": len(
                {(row["doi"], row["uniprot"]) for row in sources}
            ),
            "crystallographic_records": sum(
                row["enumeration_status"] == "ok" for row in sources
            ),
            "raw_physical_interfaces": sum(
                int(row["physical_interfaces"]) for row in sources
            ),
            "interfaces_ge10_scoreable_ca_pairs": len(jobs),
            "jobs": len(jobs),
            "class_pdb_records": dict(
                Counter(
                    row["direct_current_paper_exact_class"]
                    for row in sources
                )
            ),
            "class_jobs": dict(
                Counter(
                    row["direct_current_paper_exact_class"] for row in jobs
                )
            ),
        }
    atomic_json(
        args.out_dir / "workload.json",
        {
            "schema_version": "dimer_confind_qid35_v2",
            "frozen_policy": {
                "length_max": 500,
                "ca_cutoff_strict_less_than_A": args.ca_cutoff,
                "minimum_scoreable_ca_pairs": args.minimum_ca_pairs,
                "pairformer_used_for_candidate_selection": False,
                "all_passing_interfaces_retained": True,
                "higher_order_assemblies": "out_of_scope",
                "construct_filter": False,
            },
            "workload": workload,
            "enumeration_status": dict(
                Counter(row["enumeration_status"] for row in pdb_rows)
            ),
            "inputs": {
                "source": {
                    "path": str(args.input),
                    "sha256": sha256(args.input),
                },
                "monomer_manifest": {
                    "path": str(args.monomer_manifest),
                    "sha256": sha256(args.monomer_manifest),
                },
            },
        },
    )
    write_methods(args.out_dir)
    print(json.dumps(workload, indent=2))


def selected_jobs(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.manifest)
    rows.sort(key=lambda row: row["interface_id"])
    rows = rows[args.shard :: args.num_shards]
    if args.interface_ids:
        wanted = set(args.interface_ids)
        rows = [row for row in rows if row["interface_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def split_phase_manifests(out_dir: Path, primary_manifest: Path) -> None:
    rows = read_csv(primary_manifest)
    interesting = [
        row
        for row in rows
        if row["direct_current_paper_exact_class"]
        in {"Oligomer-only", "Multiple states"}
    ]
    controls = [
        row
        for row in rows
        if row["direct_current_paper_exact_class"] == "Monomer-only"
    ]
    if len(interesting) != 2365 or len(controls) != 3128:
        raise RuntimeError(
            f"phase counts mismatch: {len(interesting)}, {len(controls)}"
        )
    all_ids = {row["interface_id"] for row in rows}
    interesting_ids = {row["interface_id"] for row in interesting}
    control_ids = {row["interface_id"] for row in controls}
    if interesting_ids & control_ids or interesting_ids | control_ids != all_ids:
        raise RuntimeError("phase manifests are not a disjoint exact union")
    for path, values in [
        (out_dir / "candidate_manifest_phase1_interesting.csv", interesting),
        (out_dir / "candidate_manifest_phase2_controls.csv", controls),
    ]:
        if path.exists():
            existing = read_csv(path)
            if [row["interface_id"] for row in existing] != [
                row["interface_id"] for row in values
            ]:
                raise RuntimeError(f"existing immutable phase manifest differs: {path}")
        else:
            atomic_csv(path, values)
    print(
        f"phase1={len(interesting):,}; phase2={len(controls):,}; "
        f"union={len(all_ids):,}; disjoint=true"
    )


def status_report(
    out_dir: Path,
    manifest: Path,
    screen_session: str,
    run_log: Path | None = None,
) -> dict[str, Any]:
    rows = read_csv(manifest)
    ids = {row["interface_id"] for row in rows}
    completed_rows = []
    failures = []
    mtimes = []
    for interface_id in ids:
        paths = job_paths(out_dir, interface_id)
        if paths["done"].exists():
            completed_rows.append(json.loads(paths["done"].read_text()))
            mtimes.append(paths["done"].stat().st_mtime)
        elif paths["failure"].exists():
            failures.append(json.loads(paths["failure"].read_text()))
    class_total = Counter(
        row["direct_current_paper_exact_class"] for row in rows
    )
    class_completed = Counter(
        row["direct_current_paper_exact_class"] for row in completed_rows
    )
    elapsed = [float(row["elapsed_seconds"]) for row in completed_rows]
    throughput = None
    eta = None
    if len(mtimes) >= 2 and max(mtimes) > min(mtimes):
        throughput = (len(mtimes) - 1) / (max(mtimes) - min(mtimes))
        eta = (len(rows) - len(completed_rows)) / throughput
    try:
        result = subprocess.run(
            ["/usr/bin/screen", "-ls"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        screen_output = (result.stdout + result.stderr).strip()
        screen_active = screen_session in screen_output
    except Exception as error:
        screen_output = f"{type(error).__name__}: {error}"
        screen_active = None
    current_run = None
    if run_log is not None and run_log.exists():
        matches = re.findall(
            r"progress=(\d[\d,]*)/(\d[\d,]*) "
            r"success=(\d[\d,]*) failure=(\d[\d,]*) "
            r"skipped=(\d[\d,]*) elapsed_s=([\d.]+) "
            r"jobs_per_hour=([\d.]+) eta_s=([^\s]+)",
            run_log.read_text(errors="replace"),
        )
        if matches:
            values = matches[-1]
            current_run = {
                "completed_including_skips": int(values[0].replace(",", "")),
                "manifest_total": int(values[1].replace(",", "")),
                "new_successes": int(values[2].replace(",", "")),
                "failures": int(values[3].replace(",", "")),
                "preexisting_skipped": int(values[4].replace(",", "")),
                "elapsed_seconds": float(values[5]),
                "jobs_per_hour_including_skip_scan": float(values[6]),
                "eta_seconds_reported": (
                    None if values[7] == "unknown" else float(values[7])
                ),
            }
            elapsed_run = current_run["elapsed_seconds"]
            current_run["new_successes_per_hour"] = (
                current_run["new_successes"] / elapsed_run * 3600
                if elapsed_run
                else None
            )
    return {
        "manifest": str(manifest),
        "total_jobs": len(rows),
        "completed_successes": len(completed_rows),
        "failures_without_success": len(failures),
        "percent_complete": 100 * len(completed_rows) / len(rows),
        "class_total": dict(sorted(class_total.items())),
        "class_completed": dict(sorted(class_completed.items())),
        "elapsed_seconds": {
            "minimum": min(elapsed) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "p90": float(np.quantile(elapsed, 0.9)) if elapsed else None,
            "maximum": max(elapsed) if elapsed else None,
        },
        "observed_jobs_per_hour": (
            throughput * 3600 if throughput is not None else None
        ),
        "eta_seconds": eta,
        "screen_session": screen_session,
        "screen_active": screen_active,
        "screen_ls": screen_output,
        "current_run_log": str(run_log) if run_log is not None else None,
        "current_run": current_run,
    }


def limit_worker_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"


def rolling_process_jobs(
    jobs: list[dict[str, str]],
    config: dict[str, Any],
    workers: int,
    max_inflight: int,
    progress_interval: int = 10,
) -> dict[str, Any]:
    """Run a deterministic job stream with a bounded rolling future queue."""
    started = time.time()
    counts = Counter()
    completed = 0
    next_index = 0
    last_report = started
    with ProcessPoolExecutor(
        max_workers=workers, initializer=limit_worker_threads
    ) as pool:
        inflight: dict[Any, int] = {}
        while next_index < len(jobs) and len(inflight) < max_inflight:
            future = pool.submit(run_job, jobs[next_index], config)
            inflight[future] = next_index
            next_index += 1
        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in finished:
                job_index = inflight.pop(future)
                try:
                    _, outcome, _ = future.result()
                except Exception as error:
                    outcome = "failure"
                    print(
                        f"scheduler_worker_exception interface_id="
                        f"{jobs[job_index]['interface_id']} "
                        f"error={type(error).__name__}:{error}",
                        flush=True,
                    )
                counts[outcome] += 1
                completed += 1
                if next_index < len(jobs):
                    replacement = pool.submit(
                        run_job, jobs[next_index], config
                    )
                    inflight[replacement] = next_index
                    next_index += 1
            now = time.time()
            if (
                completed % progress_interval == 0
                or now - last_report >= 30
                or completed == len(jobs)
            ):
                elapsed = now - started
                throughput = completed / elapsed if elapsed else 0.0
                eta = (
                    (len(jobs) - completed) / throughput
                    if throughput
                    else None
                )
                eta_text = f"{eta:.1f}" if eta is not None else "unknown"
                print(
                    f"progress={completed:,}/{len(jobs):,} "
                    f"success={counts['success']:,} "
                    f"failure={counts['failure']:,} "
                    f"skipped={counts['skipped']:,} "
                    f"elapsed_s={elapsed:.1f} "
                    f"jobs_per_hour={throughput * 3600:.2f} "
                    f"eta_s={eta_text}",
                    flush=True,
                )
                last_report = now
    return {
        "total": len(jobs),
        "success": counts["success"],
        "failure": counts["failure"],
        "skipped": counts["skipped"],
        "wall_seconds": time.time() - started,
    }


def run_stage(args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= 64:
        raise SystemExit("--workers must be in [1,64]")
    jobs = selected_jobs(args)
    config = {
        "out_dir": str(args.out_dir),
        "confind": str(args.confind),
        "rotlib": str(args.rotlib),
        "timeout": args.timeout,
        "threshold": args.threshold,
        "min_sep": args.min_sep,
        "resume": args.resume,
    }
    max_inflight = args.max_inflight or args.workers * 2
    if max_inflight < args.workers:
        raise SystemExit("--max-inflight must be >= --workers")
    print(
        f"rolling jobs={len(jobs):,}; workers={args.workers}; "
        f"max_inflight={max_inflight}",
        flush=True,
    )
    summary = rolling_process_jobs(
        jobs, config, args.workers, max_inflight, args.progress_interval
    )
    print("run_summary=" + json.dumps(summary, sort_keys=True), flush=True)
    if not args.no_finalize:
        finalize_stage(args.out_dir)


def finalize_stage(out_dir: Path) -> None:
    accounting_path = out_dir / "pdb_enumeration_accounting.csv"
    manifest_path = out_dir / "candidate_manifest_all.csv"
    if accounting_path.exists() and manifest_path.exists():
        pdb_accounting = read_csv(accounting_path)
        candidate_manifest = read_csv(manifest_path)
        workload_rows = []
        for cohort in (
            "primary_strict_monomer",
            "sensitivity_nonmonomeric_recheck",
        ):
            for class_name in CLASS_ORDER:
                pdb_rows = [
                    row
                    for row in pdb_accounting
                    if row["cohort"] == cohort
                    and row["direct_current_paper_exact_class"] == class_name
                ]
                jobs = [
                    row
                    for row in candidate_manifest
                    if row["cohort"] == cohort
                    and row["direct_current_paper_exact_class"] == class_name
                ]
                workload_rows.append(
                    {
                        "cohort": cohort,
                        "literature_class": class_name,
                        "pdb_uniprot_records": len(pdb_rows),
                        "unique_doi_uniprot_keys": len(
                            {(row["doi"], row["uniprot"]) for row in pdb_rows}
                        ),
                        "crystallographic_records": sum(
                            row["enumeration_status"] == "ok"
                            for row in pdb_rows
                        ),
                        "raw_physical_interfaces": sum(
                            int(row["physical_interfaces"]) for row in pdb_rows
                        ),
                        "interfaces_ge10_scoreable_ca_pairs": len(jobs),
                        "confind_jobs": len(jobs),
                        "pdb_uniprot_records_with_jobs": len(
                            {(row["pdb"], row["uniprot"]) for row in jobs}
                        ),
                        "doi_uniprot_keys_with_jobs": len(
                            {(row["doi"], row["uniprot"]) for row in jobs}
                        ),
                    }
                )
        atomic_csv(out_dir / "workload_by_class.csv", workload_rows)
        workload_path = out_dir / "workload.json"
        if workload_path.exists():
            workload = json.loads(workload_path.read_text())
            for cohort in (
                "primary_strict_monomer",
                "sensitivity_nonmonomeric_recheck",
            ):
                jobs = [
                    row
                    for row in candidate_manifest
                    if row["cohort"] == cohort
                ]
                workload["workload"][cohort][
                    "pdb_records_with_jobs"
                ] = len({(row["pdb"], row["uniprot"]) for row in jobs})
                workload["workload"][cohort][
                    "doi_uniprot_keys_with_jobs"
                ] = len({(row["doi"], row["uniprot"]) for row in jobs})
            atomic_json(workload_path, workload)
    done = [
        json.loads(path.read_text())
        for path in sorted((out_dir / "jobs").glob("*/done.json"))
    ]
    failed = [
        json.loads(path.read_text())
        for path in sorted((out_dir / "jobs").glob("*/failure.json"))
        if not (path.parent / "done.json").exists()
    ]
    if done:
        atomic_csv(out_dir / "completed_candidate_scores.csv", done)
    if failed:
        atomic_csv(out_dir / "failed_candidates.csv", failed)
    atomic_json(
        out_dir / "status.json",
        {
            "completed": len(done),
            "failed_without_success": len(failed),
            "elapsed_seconds": {
                "sum": sum(float(row["elapsed_seconds"]) for row in done),
                "median": (
                    float(np.median([row["elapsed_seconds"] for row in done]))
                    if done
                    else None
                ),
                "minimum": (
                    min(row["elapsed_seconds"] for row in done) if done else None
                ),
                "maximum": (
                    max(row["elapsed_seconds"] for row in done) if done else None
                ),
            },
        },
    )
    canary_path = out_dir / "CANARY_REPORT.json"
    if canary_path.exists():
        canary = json.loads(canary_path.read_text())
        monomer = canary["monomer_canaries"]
        successful = [row for row in monomer if row["success"]]
        manifest = read_csv(out_dir / "candidate_manifest_primary.csv")
        measured_seconds = sum(row["elapsed_seconds"] for row in successful)
        measured_residues = sum(
            int(
                next(
                    item["L_up"]
                    for item in manifest
                    if item["interface_id"] == row["interface_id"]
                )
            )
            for row in successful
        )
        total_residues = sum(int(row["L_up"]) for row in manifest)
        length_scaled_seconds_32 = (
            measured_seconds / measured_residues * total_residues / 32
        )
        bytes_per_residue = (
            canary["artifact_storage"]["canary_job_artifact_bytes"]
            / measured_residues
        )
        canary["production_planning"] = {
            "primary_jobs": len(manifest),
            "primary_job_residue_sum": total_residues,
            "measured_small_canary_seconds_per_residue": (
                measured_seconds / measured_residues
            ),
            "length_scaled_wall_hours_at_32_workers": (
                length_scaled_seconds_32 / 3600
            ),
            "recommended_wall_planning_range_hours_at_32_workers": [
                length_scaled_seconds_32 / 3600,
                2 * length_scaled_seconds_32 / 3600,
            ],
            "length_scaled_compressed_storage_gb": (
                bytes_per_residue * total_residues / 1e9
            ),
            "recommended_storage_headroom_gb": (
                2 * bytes_per_residue * total_residues / 1e9
            ),
            "caveat": (
                "three monomer canaries were deliberately the shortest "
                "eligible records; scaling is by summed length and may still "
                "understate contact-density, I/O, timeout, and retry costs"
            ),
        }
        atomic_json(canary_path, canary)
        lines = [
            "# Dimer-only ConFind/qid35 canary report",
            "",
            "## Monomer candidate dimers",
            "",
            "| Literature class | PDB/interface | Runtime (s) | Status |",
            "|---|---|---:|---|",
            *[
                f"| {row['literature_class']} | {row['pdb']} / "
                f"{row['interface_id']} | {row['elapsed_seconds']:.2f} | "
                f"{'success' if row['success'] else 'failed'} |"
                for row in monomer
            ],
            "",
            "All inputs were full-atom, protein-only two-chain P1 structures "
            "with chain IDs A/B and exact SIFTS-to-UniProt renumbering. Candidate "
            "selection used only raw Cα geometry; no Pairformer field was read.",
            "",
            "The repeated monomer candidate reproduced all discrete qid35/ConFind "
            f"metrics exactly: **{canary['monomer_repeat']['deterministic_metrics']}**.",
            "",
            "## Simple homo assembly-1 semantic control",
            "",
            f"3O1N has exactly {canary['homo_simple_dimer_canary']['whole_assembly_target_chains']} "
            "mapped target chains, so its isolated pair is its complete target "
            "assembly-1 dimer. Two independent ConFind runs returned "
            f"{canary['homo_simple_dimer_canary']['intra_pairs']} intrachain and "
            f"{canary['homo_simple_dimer_canary']['inter_pairs']} interchain "
            "scoreable pairs with identical pair sets. This validates dimer "
            "semantics only; no higher-order inference is made.",
            "",
            "## Production planning",
            "",
            f"- Primary jobs: {len(manifest):,}",
            f"- Length-scaled wall estimate at 32 workers: "
            f"{length_scaled_seconds_32 / 3600:.1f} h",
            f"- Planning range with 2× headroom: "
            f"{length_scaled_seconds_32 / 3600:.1f}–"
            f"{2 * length_scaled_seconds_32 / 3600:.1f} h",
            f"- Length-scaled compressed storage estimate: "
            f"{bytes_per_residue * total_residues / 1e9:.2f} GB; "
            f"reserve at least {2 * bytes_per_residue * total_residues / 1e9:.1f} GB",
            "",
            "These estimates are grounded in three deliberately short canaries "
            "and remain uncertain for longer/contact-dense structures and retries.",
        ]
        (out_dir / "CANARY_REPORT.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    provenance = {
        "schema_version": "dimer_confind_qid35_v2",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "configuration": {
            "confind_score_strictly_greater_than": 0.01,
            "minimum_sequence_separation": 6,
            "qid": 35,
            "maximum_production_workers": 32,
            "default_timeout_seconds": 1800,
            "higher_order_assemblies": "out_of_scope",
        },
        "immutable_source_outputs_modified": False,
        "full_production_run_started": False,
    }
    atomic_json(out_dir / "provenance.json", provenance)
    top_level = [
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "output_checksums.json"
    ]
    atomic_json(
        out_dir / "output_checksums.json",
        {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(top_level)
        },
    )


def raw_contact_dict(path: Path, length: int, threshold: float, min_sep: int):
    return parse_contacts(path, length, threshold, min_sep)[:2]


def homo_canary(args: argparse.Namespace) -> dict[str, Any]:
    manifest = next(
        row
        for row in read_csv(HOMO_MANIFEST)
        if row["pdb"].upper() == args.homo_pdb.upper()
    )
    pdb = manifest["pdb"].lower()
    assembly = BIOASSEMBLY_DIR / f"{pdb}-assembly1.cif.gz"
    sifts = SIFTS_DIR / f"{pdb}.xml.gz"
    mapping = parse_sifts_for_uniprot(sifts, manifest["uniprot"])
    directory = args.out_dir / "canary" / manifest["pdb"].upper()
    directory.mkdir(parents=True, exist_ok=True)
    input_pdb = directory / "whole_assembly1_target.pdb"
    prep = prepare_confind_pdb(
        str(assembly), mapping, str(input_pdb), int(manifest["L_up"])
    )
    if prep["n_target_chains"] != 2:
        return {
            "pdb": manifest["pdb"],
            "status": "skipped_not_simple_dimer",
            "n_target_chains": prep["n_target_chains"],
        }
    outputs, elapsed = [], []
    for replicate in (1, 2):
        raw = directory / f"replicate{replicate}.txt"
        started = time.time()
        run_confind(input_pdb, raw, args.confind, args.rotlib, args.timeout)
        elapsed.append(time.time() - started)
        intra, inter, _, parsed = parse_contacts(
            raw, int(manifest["L_up"]), args.threshold, args.min_sep
        )
        outputs.append((intra, inter))
        with raw.open("rb") as source, gzip.open(
            directory / f"replicate{replicate}.txt.gz", "wb"
        ) as destination:
            shutil.copyfileobj(source, destination)
        raw.unlink()
    return {
        "pdb": manifest["pdb"],
        "uniprot": manifest["uniprot"],
        "status": "success",
        "whole_assembly_target_chains": prep["n_target_chains"],
        "isolated_pair_is_whole_target_assembly": True,
        "intra_pairs": len(outputs[0][0]),
        "inter_pairs": len(outputs[0][1]),
        "deterministic_pair_sets": (
            outputs[0][0] == outputs[1][0]
            and outputs[0][1] == outputs[1][1]
        ),
        "replicate_elapsed_seconds": elapsed,
        "higher_order_inference": False,
    }


def canary_stage(args: argparse.Namespace) -> None:
    rows = read_csv(args.manifest)
    primary = [row for row in rows if row["cohort"] == "primary_strict_monomer"]
    chosen = []
    for class_name in CLASS_ORDER:
        options = [
            row
            for row in primary
            if row["direct_current_paper_exact_class"] == class_name
        ]
        options.sort(
            key=lambda row: (
                int(row["L_up"]),
                -int(row["scoreable_ca_pairs"]),
                row["interface_id"],
            )
        )
        chosen.append(options[0])
    config = {
        "out_dir": str(args.out_dir),
        "confind": str(args.confind),
        "rotlib": str(args.rotlib),
        "timeout": args.timeout,
        "threshold": args.threshold,
        "min_sep": args.min_sep,
        "resume": False,
    }
    results = []
    for row in chosen:
        interface_id, outcome, elapsed = run_job(row, config)
        results.append(
            {
                "interface_id": interface_id,
                "pdb": row["pdb"],
                "literature_class": row[
                    "direct_current_paper_exact_class"
                ],
                "success": outcome == "success",
                "elapsed_seconds": elapsed,
            }
        )
    first = chosen[0]
    original = json.loads(
        job_paths(args.out_dir, first["interface_id"])["done"].read_text()
    )
    repeat_dir = args.out_dir / "canary_repeat"
    repeat_config = {**config, "out_dir": str(repeat_dir)}
    _, repeat_outcome, repeat_elapsed = run_job(first, repeat_config)
    repeated = json.loads(
        job_paths(repeat_dir, first["interface_id"])["done"].read_text()
    )
    metric_fields = [
        "candidate_specific_x",
        "n_intra_confind_pairs",
        "n_interface_confind_pairs",
        "interface_hits",
        "intra_hits",
        "union_precision",
        "standardized_intra_top_i_recall",
    ]
    repeat_success = repeat_outcome == "success"
    deterministic_metrics = repeat_success and all(
        original[field] == repeated[field] for field in metric_fields
    )
    homo = homo_canary(args)
    artifacts = [
        path
        for path in args.out_dir.glob("jobs/*/*")
        if path.is_file()
    ]
    artifact_bytes = sum(path.stat().st_size for path in artifacts)
    report = {
        "monomer_canaries": results,
        "monomer_repeat": {
            "interface_id": first["interface_id"],
            "success": repeat_success,
            "elapsed_seconds": repeat_elapsed,
            "deterministic_metrics": deterministic_metrics,
        },
        "homo_simple_dimer_canary": homo,
        "selection_validation": {
            "classes": sorted(row["literature_class"] for row in results),
            "pairformer_fields_used_in_enumeration": [],
            "minimum_scoreable_ca_pairs": 10,
        },
        "artifact_storage": {
            "canary_job_artifact_bytes": artifact_bytes,
            "successful_monomer_jobs": sum(row["success"] for row in results),
            "mean_bytes_per_successful_monomer_job": (
                artifact_bytes / sum(row["success"] for row in results)
                if any(row["success"] for row in results)
                else None
            ),
        },
    }
    atomic_json(args.out_dir / "CANARY_REPORT.json", report)
    finalize_stage(args.out_dir)
    print(json.dumps(report, indent=2))


def self_check(out_dir: Path) -> None:
    candidates = read_csv(out_dir / "candidate_manifest_all.csv")
    ids = [row["interface_id"] for row in candidates]
    assert len(ids) == len(set(ids))
    assert all(int(row["L_up"]) <= 500 for row in candidates)
    assert all(int(row["scoreable_ca_pairs"]) >= 10 for row in candidates)
    assert all(
        "pf" not in row["candidate_selection_policy"].casefold()
        and "pairformer" not in row["candidate_selection_policy"].casefold()
        for row in candidates
    )
    for path in (out_dir / "jobs").glob("*/done.json"):
        row = json.loads(path.read_text())
        assert row["chain_ids"] == ["A", "B"]
        assert row["protein_only"] and row["full_atom"]
        assert float(row["confind_records_above_threshold"]) >= 0
    print(
        f"self-check passed: {len(candidates):,} unique jobs; "
        f"{sum(1 for _ in (out_dir / 'jobs').glob('*/done.json'))} completed"
    )


def write_methods(out_dir: Path) -> None:
    text = """# Dimer-only ConFind/qid35 pipeline

## Frozen primary policy

Candidate discovery is outcome-blind. All physically deduplicated ASU/XTAL
interfaces with at least 10 unique mapped Cα pairs at strict distance <8 Å and
UniProt separation >=6 are retained. Pairformer scores are not read during
enumeration. Only pairwise dimers are generated; higher-order reconstruction
is prohibited.

Each candidate is written as a protein-only, full-atom two-chain P1 PDB. Chains
are named A/B, non-target and nonpolymer content is excluded, and mapped
residues are renumbered to exact UniProt positions. ConFind contacts are kept
only when score is strictly >0.01; scoreable pairs additionally require
|i-j|>=6. Lossless parsed artifacts retain original chain endpoints, positions,
scores, collapsed UniProt pairs, interface transform, and provenance.

The primary historical-compatible score uses X=|I union C|, a symmetrized qid35
matrix, and the top-X eligible upper-triangle pairs. Outputs explicitly report
interface hits/recall, inter fraction of top-X, intra hits, intra recall at
candidate X, and union precision. Top-|I| intrachain recall is sensitivity only.

The primary cohort is the 2,193 strict deposited-monomer PDB×UniProt records.
The 250 assembly-recheck failures are retained separately. Construct
compatibility is not filtered.

## Resumability

Every candidate writes an atomic `done.json`; successful jobs are skipped with
`--resume`. Raw ConFind text is gzip-preserved and a lossless NPZ stores parsed
records. Sharding is deterministic. Work is submitted in batches of at most
twice the process count.

## Production command

After canary review:

```bash
PYTHONPATH=. python homooligomer/pdb_screen/literature_parsing/dimer_confind_qid35_pipeline.py run \\
  --out-dir /path/to/external-data/pdb_pairformer_screen/full_analysis/dimer_confind_qid35_v2_20260904 \\
  --manifest /path/to/external-data/pdb_pairformer_screen/full_analysis/dimer_confind_qid35_v2_20260904/candidate_manifest_primary.csv \\
  --workers 32 --timeout 1800 --resume
```

Do not launch this command until worker count and canary timing are approved.
"""
    (out_dir / "METHODS.md").write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="stage", required=True)
    enum = sub.add_parser("enumerate")
    enum.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    enum.add_argument(
        "--monomer-manifest",
        type=Path,
        default=Path(
            "/path/to/external-data/pdb_pairformer_screen/full_screen/manifest_monomer.csv"
        ),
    )
    enum.add_argument("--sifts-dir", type=Path, default=SIFTS_DIR)
    enum.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    enum.add_argument("--ca-cutoff", type=float, default=8.0)
    enum.add_argument("--minimum-ca-pairs", type=int, default=10)
    enum.add_argument("--min-sep", type=int, default=6)
    for name in ("run", "canary"):
        command = sub.add_parser(name)
        command.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
        command.add_argument(
            "--manifest",
            type=Path,
            default=DEFAULT_OUT / "candidate_manifest_primary.csv",
        )
        command.add_argument("--confind", type=Path, default=DEFAULT_CONFIND)
        command.add_argument("--rotlib", type=Path, default=DEFAULT_ROTLIB)
        command.add_argument("--timeout", type=int, default=1800)
        command.add_argument("--threshold", type=float, default=0.01)
        command.add_argument("--min-sep", type=int, default=6)
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--max-inflight", type=int)
        command.add_argument("--progress-interval", type=int, default=10)
        command.add_argument("--no-finalize", action="store_true")
        command.add_argument("--resume", action="store_true")
        command.add_argument("--limit", type=int)
        command.add_argument("--shard", type=int, default=0)
        command.add_argument("--num-shards", type=int, default=1)
        command.add_argument("--interface-ids", nargs="+")
    canary = sub.choices["canary"]
    canary.add_argument("--homo-pdb", default="3O1N")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    check = sub.add_parser("self-check")
    check.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    split = sub.add_parser("split-phases")
    split.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    split.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUT / "candidate_manifest_primary.csv",
    )
    status = sub.add_parser("status")
    status.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    status.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUT / "candidate_manifest_primary.csv",
    )
    status.add_argument("--screen-session", default="dimer-confind-qid35")
    status.add_argument("--run-log", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "num_shards") and not 0 <= args.shard < args.num_shards:
        raise SystemExit("invalid shard")
    if args.stage == "enumerate":
        enumerate_stage(args)
    elif args.stage == "run":
        run_stage(args)
    elif args.stage == "canary":
        canary_stage(args)
    elif args.stage == "finalize":
        finalize_stage(args.out_dir)
    elif args.stage == "self-check":
        self_check(args.out_dir)
    elif args.stage == "split-phases":
        split_phase_manifests(args.out_dir, args.manifest)
    elif args.stage == "status":
        print(
            json.dumps(
                status_report(
                    args.out_dir,
                    args.manifest,
                    args.screen_session,
                    args.run_log,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
