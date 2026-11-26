#!/usr/bin/env python3
"""Generate ConFind contacts for homo-oligomer biological assembly 1.

The old ConFind pipeline indexes contacts by the first mmCIF entity sequence
and can collapse assembly copies that share an author chain ID.  This runner
instead:

1. reads the already-downloaded biological assembly 1;
2. keeps only chains mapping to the manifest's target UniProt;
3. gives every assembly copy a distinct one-character PDB chain ID;
4. renumbers observed residues to UniProt positions before running ConFind;
5. saves separate within-chain and cross-chain (biological-interface) pairs.

ConFind's existing operational threshold is retained: contact score > 0.01.
No asymmetric-unit or crystallographic-symmetry contacts are generated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
from tqdm import tqdm


DEFAULT_MANIFEST = (
    "/path/to/external-data/pdb_pairformer_screen/full_screen/manifest_homo.csv"
)
DEFAULT_ASSEMBLY_DIR = "/path/to/external-data/pdb_bioassembly_all"
DEFAULT_SIFTS_DIR = "/path/to/external-data/pdb_pairformer_screen/full_screen/sifts_xml"
DEFAULT_OUT_DIR = (
    "/path/to/external-data/pdb_pairformer_screen/confind_bio_assembly1"
)
DEFAULT_CONFIND = (
    "/path/to/external-home/sam/homooligomer/confind-msl-bin/confind"
)
DEFAULT_ROTLIB = (
    "/path/to/external-home/sam/homooligomer/"
    "confind-msl-bin/rotlibs/DB-2010"
)

SIFTS_NS = "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}"
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
ASSEMBLY_COPY_RE = re.compile(r"^(.+?)(?:[-_]\d+)$")
CONFIND_ENDPOINT_RE = re.compile(r"^([^,]+),(-?\d+)([A-Za-z]?)$")

SUMMARY_FIELDS = [
    "pdb", "uniprot", "L_up", "assembly",
    "n_target_chains", "n_observed_target_residues", "n_atoms",
    "n_confind_records", "n_records_above_threshold",
    "n_intra_pairs_all", "n_bio_pairs_all",
    "n_intra_pairs_sep6", "n_bio_pairs_sep6", "x_union_sep6",
    "intra_contact_path", "inter_contact_path",
]
FAILURE_FIELDS = ["pdb", "uniprot", "stage", "reason"]


def load_manifest(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_sifts_for_uniprot(path: str, target_uniprot: str) -> dict[str, dict[int, int]]:
    """Return source author chain -> label_seq -> UniProt position."""
    with gzip.open(path, "rb") as handle:
        root = ET.parse(handle).getroot()

    by_chain: dict[str, dict[int, int]] = {}
    for entity in root.findall(f"{SIFTS_NS}entity"):
        if entity.get("type") != "protein":
            continue
        for segment in entity.findall(f"{SIFTS_NS}segment"):
            residues = segment.find(f"{SIFTS_NS}listResidue")
            if residues is None:
                continue
            for residue in residues.findall(f"{SIFTS_NS}residue"):
                try:
                    label_seq = int(residue.get("dbResNum"))
                except (TypeError, ValueError):
                    continue
                pdb_xref = None
                up_xref = None
                for xref in residue.findall(f"{SIFTS_NS}crossRefDb"):
                    source = xref.get("dbSource")
                    if source == "PDB":
                        pdb_xref = xref
                    elif (
                        source == "UniProt"
                        and xref.get("dbAccessionId") == target_uniprot
                    ):
                        up_xref = xref
                if pdb_xref is None or up_xref is None:
                    continue
                chain = pdb_xref.get("dbChainId")
                try:
                    up_position = int(up_xref.get("dbResNum"))
                except (TypeError, ValueError):
                    continue
                if chain:
                    by_chain.setdefault(chain, {})[label_seq] = up_position
    return by_chain


def source_chain_for_assembly_chain(
    assembly_chain: str,
    sifts_chains: set[str],
) -> str | None:
    """Resolve an assembly-copy chain to the source author chain safely."""
    if assembly_chain in sifts_chains:
        return assembly_chain
    match = ASSEMBLY_COPY_RE.match(assembly_chain)
    if match and match.group(1) in sifts_chains:
        return match.group(1)
    return None


def clone_residue_at_uniprot_position(
    residue: gemmi.Residue,
    up_position: int,
) -> gemmi.Residue:
    """Clone one observed residue and renumber it to its UniProt position."""
    cloned = residue.clone()
    cloned.seqid = gemmi.SeqId(up_position, " ")
    cloned.label_seq = up_position
    return cloned


def prepare_confind_pdb(
    assembly_path: str,
    sifts_mapping: dict[str, dict[int, int]],
    output_pdb: str,
    L_up: int,
) -> dict:
    """Write target-UniProt assembly chains as a copy-safe ConFind PDB."""
    structure = gemmi.read_structure(assembly_path)
    if len(structure) == 0:
        raise ValueError("assembly has no coordinate models")
    source_model = structure[0]

    output = gemmi.Structure()
    output.name = structure.name
    output_model = gemmi.Model("1")

    chain_metadata = {}
    n_residues = 0
    n_atoms = 0
    used_generated_ids = 0

    for assembly_chain in source_model:
        source_chain = source_chain_for_assembly_chain(
            assembly_chain.name, set(sifts_mapping)
        )
        if source_chain is None:
            continue
        label_to_up = sifts_mapping[source_chain]
        kept = []
        seen_positions = set()
        for residue in assembly_chain.get_polymer():
            if not residue.label_seq:
                continue
            up_position = label_to_up.get(int(residue.label_seq))
            if up_position is None or not 1 <= up_position <= L_up:
                continue
            if up_position in seen_positions:
                continue
            seen_positions.add(up_position)
            cloned = clone_residue_at_uniprot_position(residue, up_position)
            if len(cloned) == 0:
                continue
            kept.append(cloned)

        if not kept:
            continue
        if used_generated_ids >= len(PDB_CHAIN_IDS):
            raise ValueError(
                f"target assembly has more than {len(PDB_CHAIN_IDS)} chains; "
                "legacy PDB cannot represent them safely"
            )
        generated_id = PDB_CHAIN_IDS[used_generated_ids]
        used_generated_ids += 1
        out_chain = gemmi.Chain(generated_id)
        for residue in kept:
            out_chain.add_residue(residue)
            n_residues += 1
            n_atoms += len(residue)
        output_model.add_chain(out_chain)
        chain_metadata[generated_id] = {
            "assembly_chain": assembly_chain.name,
            "source_chain": source_chain,
            "n_residues": len(kept),
        }

    if len(chain_metadata) == 0:
        raise ValueError("no assembly chains map to the target UniProt")
    if n_atoms > 99_999:
        raise ValueError(
            f"target assembly has {n_atoms:,} atoms; legacy PDB atom limit is 99,999"
        )

    # Gemmi copies a model when it is added to a structure, so add it only
    # after all chains and residues have been populated.
    output.add_model(output_model)
    output.write_pdb(output_pdb)
    return {
        "chain_metadata": chain_metadata,
        "n_target_chains": len(chain_metadata),
        "n_observed_target_residues": n_residues,
        "n_atoms": n_atoms,
    }


def run_confind(
    input_pdb: str,
    output_contacts: str,
    executable: str,
    rotlib: str,
    timeout_seconds: int,
) -> None:
    command = [
        executable,
        "--p", input_pdb,
        "--o", output_contacts,
        "--rLib", rotlib,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"ConFind exit code {completed.returncode}: {message[-2000:]}"
        )
    if not os.path.exists(output_contacts):
        raise RuntimeError("ConFind completed without creating an output file")


def parse_endpoint(token: str) -> tuple[str, int] | None:
    match = CONFIND_ENDPOINT_RE.match(token)
    if not match:
        return None
    chain = match.group(1)
    position = int(match.group(2))
    return chain, position


def parse_confind_contacts(
    path: str,
    valid_chains: set[str],
    L_up: int,
    threshold: float,
) -> tuple[dict, dict, int, int]:
    """Return max-score dictionaries for intra and biological-interface pairs."""
    intra: dict[tuple[int, int], float] = {}
    bio: dict[tuple[int, int], float] = {}
    n_records = 0
    n_above = 0
    with open(path) as handle:
        for line in handle:
            if not line.startswith("contact"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            n_records += 1
            try:
                score = float(parts[3])
            except ValueError:
                continue
            if score <= threshold:
                continue
            endpoint1 = parse_endpoint(parts[1])
            endpoint2 = parse_endpoint(parts[2])
            if endpoint1 is None or endpoint2 is None:
                continue
            chain1, pos1 = endpoint1
            chain2, pos2 = endpoint2
            if chain1 not in valid_chains or chain2 not in valid_chains:
                continue
            if not (1 <= pos1 <= L_up and 1 <= pos2 <= L_up):
                continue
            if pos1 == pos2 and chain1 == chain2:
                continue
            pair = (pos1, pos2) if pos1 < pos2 else (pos2, pos1)
            target = intra if chain1 == chain2 else bio
            target[pair] = max(score, target.get(pair, float("-inf")))
            n_above += 1
    return intra, bio, n_records, n_above


def pairs_and_scores(
    scores: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(scores)
    pairs = np.asarray(ordered, dtype=np.int32).reshape(-1, 2)
    values = np.asarray([scores[pair] for pair in ordered], dtype=np.float32)
    return pairs, values


def save_one_contact_npz(
    path: str,
    scores_by_pair: dict,
    metadata: dict,
) -> None:
    pairs, scores = pairs_and_scores(scores_by_pair)
    temporary = path + ".part"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            pairs=pairs,
            scores=scores,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
    os.replace(temporary, path)


def save_binary_contact_matrix(
    path: str,
    scores_by_pair: dict[tuple[int, int], float],
    length: int,
) -> None:
    """Save the same symmetric int8 matrix format as the legacy pipeline."""
    matrix = np.zeros((length, length), dtype=np.int8)
    for (position1, position2) in scores_by_pair:
        index1 = position1 - 1
        index2 = position2 - 1
        matrix[index1, index2] = 1
        matrix[index2, index1] = 1
    temporary = path + ".part"
    with open(temporary, "wb") as handle:
        np.save(handle, matrix)
    os.replace(temporary, path)


def count_sep(pairs: set[tuple[int, int]], min_sep: int) -> int:
    return sum(1 for i, j in pairs if j - i >= min_sep)


def build_summary(
    metadata: dict,
    intra: set[tuple[int, int]],
    bio: set[tuple[int, int]],
    intra_path: str,
    inter_path: str,
) -> dict:
    intra_sep6 = {pair for pair in intra if pair[1] - pair[0] >= 6}
    bio_sep6 = {pair for pair in bio if pair[1] - pair[0] >= 6}
    return {
        "pdb": metadata["pdb"],
        "uniprot": metadata["uniprot"],
        "L_up": metadata["L_up"],
        "assembly": 1,
        "n_target_chains": len(metadata["chain_metadata"]),
        "n_observed_target_residues": metadata.get(
            "n_observed_target_residues", ""
        ),
        "n_atoms": metadata.get("n_atoms", ""),
        "n_confind_records": metadata.get("n_confind_records", ""),
        "n_records_above_threshold": metadata.get(
            "n_records_above_threshold", ""
        ),
        "n_intra_pairs_all": len(intra),
        "n_bio_pairs_all": len(bio),
        "n_intra_pairs_sep6": len(intra_sep6),
        "n_bio_pairs_sep6": len(bio_sep6),
        "x_union_sep6": len(intra_sep6 | bio_sep6),
        "intra_contact_path": intra_path,
        "inter_contact_path": inter_path,
    }


def pair_set_from_matrix(path: str) -> tuple[set[tuple[int, int]], int]:
    matrix = np.load(path, allow_pickle=False, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"contact matrix is not square: {matrix.shape}")
    row_indices, column_indices = np.where(np.triu(matrix) > 0)
    pairs = {
        (int(i) + 1, int(j) + 1)
        for i, j in zip(row_indices, column_indices)
    }
    return pairs, int(matrix.shape[0])


def summary_from_binary_matrices(
    row: dict,
    intra_path: str,
    inter_path: str,
) -> dict:
    intra, intra_length = pair_set_from_matrix(intra_path)
    bio, inter_length = pair_set_from_matrix(inter_path)
    expected_length = int(row["L_up"])
    if intra_length != expected_length or inter_length != expected_length:
        raise ValueError(
            f"contact matrix length mismatch: {intra_length}, {inter_length}, "
            f"expected {expected_length}"
        )
    metadata = {
        "pdb": row["pdb"].upper(),
        "uniprot": row["uniprot"],
        "L_up": expected_length,
        "chain_metadata": {},
    }
    return build_summary(metadata, intra, bio, intra_path, inter_path)


def pair_set_from_array(array: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(i), int(j))
        for i, j in np.asarray(array, dtype=np.int32).reshape(-1, 2)
    }


def load_combined_sparse_contact(
    combined_path: str,
) -> tuple[dict, dict, dict]:
    with np.load(combined_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        intra_pairs = np.asarray(data["intra_pairs"], dtype=np.int32)
        intra_scores = np.asarray(data["intra_scores"], dtype=np.float32)
        inter_pairs = np.asarray(data["bio_pairs"], dtype=np.int32)
        inter_scores = np.asarray(data["bio_scores"], dtype=np.float32)
    intra = {
        tuple(map(int, pair)): float(score)
        for pair, score in zip(intra_pairs, intra_scores)
    }
    inter = {
        tuple(map(int, pair)): float(score)
        for pair, score in zip(inter_pairs, inter_scores)
    }
    return intra, inter, metadata


def load_split_sparse_contacts(
    intra_path: str,
    inter_path: str,
) -> tuple[dict, dict, dict]:
    with np.load(intra_path, allow_pickle=False) as data:
        intra_pairs = np.asarray(data["pairs"], dtype=np.int32)
        intra_scores = np.asarray(data["scores"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"]))
    with np.load(inter_path, allow_pickle=False) as data:
        inter_pairs = np.asarray(data["pairs"], dtype=np.int32)
        inter_scores = np.asarray(data["scores"], dtype=np.float32)
        inter_metadata = json.loads(str(data["metadata_json"]))
    if (
        inter_metadata["pdb"] != metadata["pdb"]
        or inter_metadata["uniprot"] != metadata["uniprot"]
    ):
        raise ValueError("intra/inter sparse metadata do not match")
    intra = {
        tuple(map(int, pair)): float(score)
        for pair, score in zip(intra_pairs, intra_scores)
    }
    inter = {
        tuple(map(int, pair)): float(score)
        for pair, score in zip(inter_pairs, inter_scores)
    }
    return intra, inter, metadata


def process_one(row: dict, config: dict) -> tuple[dict | None, dict | None]:
    pdb = row["pdb"].lower()
    uniprot = row["uniprot"]
    L_up = int(row["L_up"])
    intra_path = os.path.join(
        config["contacts_dir"], f"{pdb}-assembly1_intra.npy"
    )
    inter_path = os.path.join(
        config["contacts_dir"], f"{pdb}-assembly1_inter.npy"
    )
    sparse_intra_path = os.path.join(
        config["sparse_intra_dir"], f"{pdb}-assembly1_intra.npz"
    )
    sparse_inter_path = os.path.join(
        config["sparse_inter_dir"], f"{pdb}-assembly1_inter.npz"
    )
    combined_path = os.path.join(config["combined_dir"], f"{pdb}.npz")

    if config["resume"]:
        if os.path.exists(intra_path) and os.path.exists(inter_path):
            return summary_from_binary_matrices(row, intra_path, inter_path), None
        if os.path.exists(sparse_intra_path) and os.path.exists(
            sparse_inter_path
        ):
            intra, bio, metadata = load_split_sparse_contacts(
                sparse_intra_path, sparse_inter_path
            )
            save_binary_contact_matrix(intra_path, intra, L_up)
            save_binary_contact_matrix(inter_path, bio, L_up)
            return build_summary(
                metadata, set(intra), set(bio), intra_path, inter_path
            ), None
        if os.path.exists(combined_path):
            intra, bio, metadata = load_combined_sparse_contact(combined_path)
            save_binary_contact_matrix(intra_path, intra, L_up)
            save_binary_contact_matrix(inter_path, bio, L_up)
            return build_summary(
                metadata, set(intra), set(bio), intra_path, inter_path
            ), None

    stage = "setup"
    work_dir = tempfile.mkdtemp(prefix=f"{pdb}_", dir=config["tmp_dir"])
    try:
        assembly_path = os.path.join(
            config["assembly_dir"], f"{pdb}-assembly1.cif.gz"
        )
        sifts_path = os.path.join(config["sifts_dir"], f"{pdb}.xml.gz")
        if not os.path.exists(assembly_path):
            raise FileNotFoundError(f"missing assembly: {assembly_path}")
        if not os.path.exists(sifts_path):
            raise FileNotFoundError(f"missing SIFTS XML: {sifts_path}")

        stage = "sifts"
        sifts_mapping = parse_sifts_for_uniprot(sifts_path, uniprot)
        if not sifts_mapping:
            raise ValueError("no SIFTS residues map to target UniProt")

        stage = "prepare_pdb"
        pdb_path = os.path.join(work_dir, f"{pdb}-assembly1.target.pdb")
        preparation = prepare_confind_pdb(
            assembly_path, sifts_mapping, pdb_path, L_up
        )

        stage = "confind"
        raw_path = os.path.join(work_dir, f"{pdb}-assembly1.cf.txt")
        run_confind(
            pdb_path,
            raw_path,
            config["confind"],
            config["rotlib"],
            config["timeout_seconds"],
        )

        stage = "parse_contacts"
        valid_chains = set(preparation["chain_metadata"])
        intra, bio, n_records, n_above = parse_confind_contacts(
            raw_path,
            valid_chains,
            L_up,
            config["threshold"],
        )
        intra_set = set(intra)
        bio_set = set(bio)
        intra_sep6 = {pair for pair in intra_set if pair[1] - pair[0] >= 6}
        bio_sep6 = {pair for pair in bio_set if pair[1] - pair[0] >= 6}
        union_sep6 = intra_sep6 | bio_sep6

        metadata = {
            "pdb": pdb.upper(),
            "uniprot": uniprot,
            "L_up": L_up,
            "assembly": 1,
            "threshold": config["threshold"],
            "chain_metadata": preparation["chain_metadata"],
            "n_observed_target_residues": preparation[
                "n_observed_target_residues"
            ],
            "n_atoms": preparation["n_atoms"],
            "n_confind_records": n_records,
            "n_records_above_threshold": n_above,
        }
        stage = "save"
        save_binary_contact_matrix(intra_path, intra, L_up)
        save_binary_contact_matrix(inter_path, bio, L_up)

        summary = {
            "pdb": pdb.upper(),
            "uniprot": uniprot,
            "L_up": L_up,
            "assembly": 1,
            "n_target_chains": preparation["n_target_chains"],
            "n_observed_target_residues": preparation[
                "n_observed_target_residues"
            ],
            "n_atoms": preparation["n_atoms"],
            "n_confind_records": n_records,
            "n_records_above_threshold": n_above,
            "n_intra_pairs_all": len(intra_set),
            "n_bio_pairs_all": len(bio_set),
            "n_intra_pairs_sep6": len(intra_sep6),
            "n_bio_pairs_sep6": len(bio_sep6),
            "x_union_sep6": len(union_sep6),
            "intra_contact_path": intra_path,
            "inter_contact_path": inter_path,
        }
        return summary, None
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        if config["verbose_failures"]:
            reason += " | " + traceback.format_exc(limit=3).replace("\n", " ")
        failure = {
            "pdb": pdb.upper(),
            "uniprot": uniprot,
            "stage": stage,
            "reason": reason,
        }
        return None, failure
    finally:
        if not config["keep_temp"]:
            shutil.rmtree(work_dir, ignore_errors=True)


def append_rows(path: str, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def load_completed(summary_path: str) -> set[str]:
    if not os.path.exists(summary_path):
        return set()
    with open(summary_path, newline="") as handle:
        return {
            row["pdb"].upper()
            for row in csv.DictReader(handle)
            if row.get("pdb")
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--assembly-dir", default=DEFAULT_ASSEMBLY_DIR)
    parser.add_argument("--sifts-dir", default=DEFAULT_SIFTS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--confind", default=DEFAULT_CONFIND)
    parser.add_argument("--rotlib", default=DEFAULT_ROTLIB)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min-sep", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--run-name",
        default="matrices",
        help="Label used in resumable summary and failure CSV filenames.",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdbs", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--verbose-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must be in [0, --num-shards)")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        raise SystemExit("--run-name may contain only letters, digits, _, ., and -")

    contacts_dir = os.path.join(args.out_dir, "contacts")
    sparse_intra_dir = os.path.join(contacts_dir, "intra")
    sparse_inter_dir = os.path.join(contacts_dir, "inter")
    tmp_dir = os.path.join(args.out_dir, "tmp")
    os.makedirs(contacts_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    summary_path = os.path.join(
        args.out_dir,
        f"summary_{args.run_name}.shard{args.shard}of{args.num_shards}.csv",
    )
    failures_path = os.path.join(
        args.out_dir,
        f"failed_{args.run_name}.shard{args.shard}of{args.num_shards}.csv",
    )

    rows = load_manifest(args.manifest)
    if args.pdbs:
        selected = {pdb.upper() for pdb in args.pdbs}
        rows = [row for row in rows if row["pdb"].upper() in selected]
    rows.sort(key=lambda row: row["pdb"])
    rows = rows[args.shard :: args.num_shards]
    if args.resume:
        completed = load_completed(summary_path)
        rows = [row for row in rows if row["pdb"].upper() not in completed]
    if args.limit is not None:
        rows = rows[: args.limit]

    config = {
        "assembly_dir": args.assembly_dir,
        "sifts_dir": args.sifts_dir,
        "combined_dir": contacts_dir,
        "contacts_dir": contacts_dir,
        "sparse_intra_dir": sparse_intra_dir,
        "sparse_inter_dir": sparse_inter_dir,
        "tmp_dir": tmp_dir,
        "confind": args.confind,
        "rotlib": args.rotlib,
        "threshold": args.threshold,
        "min_sep": args.min_sep,
        "timeout_seconds": args.timeout_seconds,
        "resume": args.resume,
        "keep_temp": args.keep_temp,
        "verbose_failures": args.verbose_failures,
    }
    print(
        f"ConFind assembly1 jobs={len(rows):,}, workers={args.workers}, "
        f"shard={args.shard}/{args.num_shards}, threshold>{args.threshold}"
    )

    summaries = []
    failures = []
    if args.workers <= 1:
        iterator = (process_one(row, config) for row in rows)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(process_one, row, config) for row in rows]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for summary, failure in tqdm(iterator, total=len(rows)):
            if summary is not None:
                summaries.append(summary)
            if failure is not None:
                failures.append(failure)
            if len(summaries) + len(failures) >= 50:
                append_rows(summary_path, SUMMARY_FIELDS, summaries)
                append_rows(failures_path, FAILURE_FIELDS, failures)
                summaries.clear()
                failures.clear()
    finally:
        if args.workers > 1:
            pool.shutdown(wait=True)
        append_rows(summary_path, SUMMARY_FIELDS, summaries)
        append_rows(failures_path, FAILURE_FIELDS, failures)

    print(f"Summary:  {summary_path}")
    print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
