#!/usr/bin/env python3
"""Run local PISA directly on deposited biological assembly 1.

The target UniProt chains are first selected with SIFTS while retaining their
deposited assembly-1 coordinates.  The official ``pdbegroup/pisa`` Docker
image is then invoked with ``--as-is`` so PISA does not generate
crystallographic mates.  Compact per-PDB JSON files retain assembly metrics,
protein-protein interface areas, and residue-level hydrophobic composition;
temporary PISA sessions and XML files are removed after parsing.
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
from tqdm import tqdm


DEFAULT_MANIFEST = (
    "/path/to/external-data/pdb_pairformer_screen/confind_bio_assembly1/"
    "manifest_homo_verified_a04_pdbcan50_500.csv"
)
DEFAULT_ASSEMBLY_DIR = "/path/to/external-data/pdb_bioassembly_all"
DEFAULT_SIFTS_DIR = "/path/to/external-data/pdb_pairformer_screen/full_screen/sifts_xml"
DEFAULT_OUT_DIR = "/path/to/external-data/pisa/bioassembly1_target_local"
DEFAULT_IMAGE = "pdbegroup/pisa:latest"
DEFAULT_CONFIG = "/usr/share/pisa/setup/pisa_cfg_tmp"
SIFTS_NS = "{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}"
ASSEMBLY_COPY_RE = re.compile(r"^(.+?)(?:[-_]\d+)$")
HYDROPHOBIC_3 = {
    "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR",
}
ASSEMBLY_FLOATS = {
    "diss_energy", "diss_energy_0", "asa", "bsa", "entropy",
    "entropy_0", "diss_area", "int_energy",
}
ASSEMBLY_INTS = {
    "serial_no", "id", "size", "mmsize", "freesize", "n_uc",
    "n_diss", "symNumber",
}
SUMMARY_FIELDS = [
    "pdb", "status", "diss_energy_kcal_mol", "diss_energy_0_kcal_mol",
    "uniprot", "L_up", "n_target_chains", "n_target_residues",
    "assembly_asa_A2", "assembly_bsa_A2", "diss_area_A2",
    "int_energy_kcal_mol", "mmsize", "n_diss", "symmetry_number",
    "n_protein_interfaces", "protein_interface_area_A2",
    "protein_interface_bsa_A2", "n_interface_residue_instances",
    "n_hydrophobic_interface_residue_instances",
    "hydrophobic_interface_residue_pct",
    "n_unique_interface_sequence_positions",
    "n_unique_hydrophobic_interface_sequence_positions",
    "unique_hydrophobic_interface_residue_pct", "raw_json",
]
FAILURE_FIELDS = ["pdb", "stage", "reason"]


def load_manifest(path: str) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def parse_sifts_for_uniprot(
    path: str,
    target_uniprot: str,
) -> dict[str, dict[int, int]]:
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
                    if xref.get("dbSource") == "PDB":
                        pdb_xref = xref
                    elif (
                        xref.get("dbSource") == "UniProt"
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
    if assembly_chain in sifts_chains:
        return assembly_chain
    match = ASSEMBLY_COPY_RE.match(assembly_chain)
    if match and match.group(1) in sifts_chains:
        return match.group(1)
    return None


def write_target_assembly(
    assembly_path: str,
    sifts_path: str,
    target_uniprot: str,
    length: int,
    output_path: str,
) -> dict:
    """Write only target-UniProt chains in their assembly-1 coordinates."""
    mapping = parse_sifts_for_uniprot(sifts_path, target_uniprot)
    if not mapping:
        raise ValueError("no SIFTS residues map to target UniProt")
    source = gemmi.read_structure(assembly_path)
    if len(source) == 0:
        raise ValueError("assembly has no coordinate models")
    output = gemmi.Structure()
    output.name = source.name
    output.cell = source.cell
    output.spacegroup_hm = source.spacegroup_hm
    model = gemmi.Model("1")
    n_target_chains = 0
    n_target_residues = 0

    for assembly_chain in source[0]:
        source_chain = source_chain_for_assembly_chain(
            assembly_chain.name, set(mapping)
        )
        if source_chain is None:
            continue
        label_to_uniprot = mapping[source_chain]
        output_chain = gemmi.Chain(assembly_chain.name)
        seen_positions = set()
        for residue in assembly_chain.get_polymer():
            if not residue.label_seq:
                continue
            position = label_to_uniprot.get(int(residue.label_seq))
            if (
                position is None
                or not 1 <= position <= length
                or position in seen_positions
            ):
                continue
            seen_positions.add(position)
            cloned = residue.clone()
            cloned.seqid = gemmi.SeqId(position, " ")
            cloned.label_seq = position
            output_chain.add_residue(cloned)
        if len(output_chain) == 0:
            continue
        model.add_chain(output_chain)
        n_target_chains += 1
        n_target_residues += len(output_chain)

    if n_target_chains < 2:
        raise ValueError(
            f"target assembly has {n_target_chains} mapped polymer chain(s)"
        )
    output.add_model(model)
    output.make_mmcif_document().write_file(output_path)
    return {
        "n_target_chains": n_target_chains,
        "n_target_residues": n_target_residues,
    }


def number(text: str | None, kind=float):
    value = (text or "").strip()
    if not value:
        return None
    try:
        return kind(value)
    except ValueError:
        return None


def parse_assembly_xml(path: str) -> dict:
    root = ET.parse(path).getroot()
    assembly = root.find(".//asu_complex/assembly")
    if assembly is None:
        description = (root.findtext("status_description") or "").strip()
        raise ValueError(f"PISA as-is assembly metrics missing: {description}")
    output = {}
    for field in ASSEMBLY_FLOATS:
        output[field] = number(assembly.findtext(field), float)
    for field in ASSEMBLY_INTS:
        output[field] = number(assembly.findtext(field), int)
    output["formula"] = (assembly.findtext("formula") or "").strip()
    output["composition"] = (assembly.findtext("composition") or "").strip()
    return output


def parse_interface_xml(path: str) -> dict:
    root = ET.parse(path).getroot()
    protein_interfaces = []
    residue_instances = set()
    unique_positions: dict[tuple[str, str], str] = {}

    for interface in root.findall("interface"):
        molecules = interface.findall("molecule")
        if len(molecules) != 2:
            continue
        if any(
            (molecule.findtext("class") or "").strip() != "Protein"
            for molecule in molecules
        ):
            continue
        interface_area = number(interface.findtext("int_area"), float) or 0.0
        molecule_bsa = 0.0
        chains = []
        for molecule in molecules:
            chain = (molecule.findtext("chain_id") or "").strip()
            chains.append(chain)
            molecule_bsa += number(
                molecule.findtext("int_area"), float
            ) or 0.0
            for residue in molecule.findall("./residues/residue"):
                bsa = number(residue.findtext("bsa"), float) or 0.0
                if bsa <= 0:
                    continue
                name = (residue.findtext("name") or "").strip().upper()
                seq_num = (residue.findtext("seq_num") or "").strip()
                label_seq = (
                    residue.findtext("label_seq_num") or seq_num
                ).strip()
                insertion = (residue.findtext("ins_code") or "").strip()
                residue_instances.add((chain, seq_num, insertion, name))
                unique_positions[(label_seq, name)] = name
        protein_interfaces.append(
            {
                "id": number(interface.findtext("id"), int),
                "chains": chains,
                "interface_area_A2": interface_area,
                "buried_surface_area_A2": molecule_bsa,
                "solvation_energy_kcal_mol": number(
                    interface.findtext("int_solv_en"), float
                ),
                "stabilization_energy_kcal_mol": number(
                    interface.findtext("stab_en"), float
                ),
                "p_value": number(interface.findtext("pvalue"), float),
            }
        )

    n_residues = len(residue_instances)
    n_hydrophobic = sum(
        residue[-1] in HYDROPHOBIC_3 for residue in residue_instances
    )
    n_unique = len(unique_positions)
    n_unique_hydrophobic = sum(
        name in HYDROPHOBIC_3 for name in unique_positions.values()
    )
    return {
        "protein_interfaces": protein_interfaces,
        "n_protein_interfaces": len(protein_interfaces),
        "protein_interface_area_A2": sum(
            interface["interface_area_A2"]
            for interface in protein_interfaces
        ),
        "protein_interface_bsa_A2": sum(
            interface["buried_surface_area_A2"]
            for interface in protein_interfaces
        ),
        "n_interface_residue_instances": n_residues,
        "n_hydrophobic_interface_residue_instances": n_hydrophobic,
        "hydrophobic_interface_residue_pct": (
            100 * n_hydrophobic / n_residues if n_residues else None
        ),
        "n_unique_interface_sequence_positions": n_unique,
        "n_unique_hydrophobic_interface_sequence_positions": (
            n_unique_hydrophobic
        ),
        "unique_hydrophobic_interface_residue_pct": (
            100 * n_unique_hydrophobic / n_unique if n_unique else None
        ),
    }


def atomic_json(path: str, value: dict) -> None:
    temporary = path + ".part"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def process_one(row: dict, config: dict) -> tuple[dict | None, dict | None]:
    pdb = row["pdb"].lower()
    uniprot = row["uniprot"]
    length = int(row["L_up"])
    output_path = os.path.join(config["raw_dir"], f"{pdb}.json")
    if config["resume"] and os.path.exists(output_path):
        with open(output_path) as handle:
            return json.load(handle), None

    stage = "setup"
    work_dir = tempfile.mkdtemp(prefix=f"{pdb}_", dir=config["tmp_dir"])
    try:
        input_name = f"{pdb}-assembly1.cif.gz"
        input_path = os.path.join(config["assembly_dir"], input_name)
        sifts_path = os.path.join(config["sifts_dir"], f"{pdb}.xml.gz")
        if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        if not os.path.exists(sifts_path):
            raise FileNotFoundError(sifts_path)
        target_name = f"{pdb}-assembly1-target.cif"
        target_path = os.path.join(work_dir, target_name)
        stage = "target_assembly"
        target_metadata = write_target_assembly(
            input_path,
            sifts_path,
            uniprot,
            length,
            target_path,
        )
        session = f"pisa_{pdb}"
        interface_xml = os.path.join(work_dir, "interfaces.xml")
        assembly_xml = os.path.join(work_dir, "assembly.xml")
        script = (
            "set -e; "
            f"pisa {session} -analyse /data/{target_name} --as-is "
            f"{config['pisa_config']} >/dev/null; "
            f"pisa {session} -xml interface {config['pisa_config']} "
            "> /data/interfaces.xml; "
            f"pisa {session} -xml assembly {config['pisa_config']} "
            "> /data/assembly.xml; "
            f"rm -rf /data/{session}; "
            "chmod 0666 /data/interfaces.xml /data/assembly.xml"
        )
        command = [
            "docker", "run", "--rm",
            "--cpu-shares", str(config["docker_cpu_shares"]),
            "-v", f"{work_dir}:/data",
            config["image"], "bash", "-lc", script,
        ]
        stage = "pisa"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config["timeout_seconds"],
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"PISA container exit {completed.returncode}: {message[-2000:]}"
            )

        stage = "parse"
        assembly = parse_assembly_xml(assembly_xml)
        interfaces = parse_interface_xml(interface_xml)
        result = {
            "pdb": pdb.upper(),
            "uniprot": uniprot,
            "L_up": length,
            "status": "success",
            **target_metadata,
            **assembly,
            **interfaces,
            "raw_json": output_path,
        }
        stage = "save"
        atomic_json(output_path, result)
        return result, None
    except Exception as error:
        return None, {
            "pdb": pdb.upper(),
            "stage": stage,
            "reason": (
                f"{type(error).__name__}: {error} | "
                + traceback.format_exc(limit=2).replace("\n", " ")
            ),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def summary_row(result: dict) -> dict:
    return {
        "pdb": result["pdb"],
        "status": result["status"],
        "uniprot": result["uniprot"],
        "L_up": result["L_up"],
        "n_target_chains": result["n_target_chains"],
        "n_target_residues": result["n_target_residues"],
        "diss_energy_kcal_mol": result.get("diss_energy"),
        "diss_energy_0_kcal_mol": result.get("diss_energy_0"),
        "assembly_asa_A2": result.get("asa"),
        "assembly_bsa_A2": result.get("bsa"),
        "diss_area_A2": result.get("diss_area"),
        "int_energy_kcal_mol": result.get("int_energy"),
        "mmsize": result.get("mmsize"),
        "n_diss": result.get("n_diss"),
        "symmetry_number": result.get("symNumber"),
        "n_protein_interfaces": result.get("n_protein_interfaces"),
        "protein_interface_area_A2": result.get(
            "protein_interface_area_A2"
        ),
        "protein_interface_bsa_A2": result.get(
            "protein_interface_bsa_A2"
        ),
        "n_interface_residue_instances": result.get(
            "n_interface_residue_instances"
        ),
        "n_hydrophobic_interface_residue_instances": result.get(
            "n_hydrophobic_interface_residue_instances"
        ),
        "hydrophobic_interface_residue_pct": result.get(
            "hydrophobic_interface_residue_pct"
        ),
        "n_unique_interface_sequence_positions": result.get(
            "n_unique_interface_sequence_positions"
        ),
        "n_unique_hydrophobic_interface_sequence_positions": result.get(
            "n_unique_hydrophobic_interface_sequence_positions"
        ),
        "unique_hydrophobic_interface_residue_pct": result.get(
            "unique_hydrophobic_interface_residue_pct"
        ),
        "raw_json": result["raw_json"],
    }


def write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    temporary = path + ".part"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--assembly-dir", default=DEFAULT_ASSEMBLY_DIR)
    parser.add_argument("--sifts-dir", default=DEFAULT_SIFTS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--pisa-config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--docker-cpu-shares", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdbs", nargs="+")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = os.path.join(args.out_dir, "raw")
    tmp_dir = os.path.join(args.out_dir, "tmp")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    rows = load_manifest(args.manifest)
    if args.pdbs:
        selected = {pdb.lower() for pdb in args.pdbs}
        rows = [row for row in rows if row["pdb"].lower() in selected]
    if args.limit is not None:
        rows = rows[: args.limit]
    config = {
        "assembly_dir": args.assembly_dir,
        "sifts_dir": args.sifts_dir,
        "raw_dir": raw_dir,
        "tmp_dir": tmp_dir,
        "image": args.image,
        "pisa_config": args.pisa_config,
        "docker_cpu_shares": args.docker_cpu_shares,
        "timeout_seconds": args.timeout_seconds,
        "resume": args.resume,
    }

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, row, config) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, failure = future.result()
            if result is not None:
                results.append(result)
            if failure is not None:
                failures.append(failure)

    results.sort(key=lambda row: row["pdb"])
    failures.sort(key=lambda row: row["pdb"])
    write_csv(
        os.path.join(args.out_dir, "pisa_bioassembly1_summary.csv"),
        SUMMARY_FIELDS,
        [summary_row(result) for result in results],
    )
    write_csv(
        os.path.join(args.out_dir, "pisa_bioassembly1_failed.csv"),
        FAILURE_FIELDS,
        failures,
    )
    print(f"Success: {len(results):,}; failures: {len(failures):,}")


if __name__ == "__main__":
    main()
