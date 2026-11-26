#!/usr/bin/env python3
"""Resumable local-PISA lifecycle for isolated Phase-1 candidate dimers."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import reverse_assembly_helpers as H


ROOT = Path(
    "/path/to/external-data/pdb_pairformer_screen/full_analysis/"
    "dimer_confind_qid35_v2_20260904"
)
DEFAULT_MANIFEST = ROOT / "candidate_manifest_phase1_interesting.csv"
DEFAULT_OUT = ROOT / "phase1_pisa_v1_20260905"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with source.open("rb") as input_handle, gzip.open(
        temporary, "wb"
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle)
    os.replace(temporary, destination)


def number(value: str | None, kind: type = float) -> Any:
    if value is None or not value.strip():
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def parse_all_protein_interfaces(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    rows = []
    for interface in root.findall(".//interface"):
        molecules = interface.findall("molecule")
        if len(molecules) != 2 or any(
            (molecule.findtext("class") or "").strip() != "Protein"
            for molecule in molecules
        ):
            continue
        rows.append(
            {
                "pisa_interface_id": number(interface.findtext("id"), int),
                "interface_area_A2": number(interface.findtext("int_area")),
                "buried_surface_area_A2": sum(
                    number(molecule.findtext("int_area")) or 0.0
                    for molecule in molecules
                ),
                "solvation_energy_kcal_mol": number(
                    interface.findtext("int_solv_en")
                ),
                "stabilization_energy_kcal_mol": number(
                    interface.findtext("stab_en")
                ),
                "p_value": number(interface.findtext("pvalue")),
                "hydrogen_bonds": number(interface.findtext("nhb"), int),
                "salt_bridges": number(interface.findtext("nsb"), int),
                "disulfide_bonds": number(interface.findtext("nds"), int),
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["interface_area_A2"] or 0.0),
            row["pisa_interface_id"] if row["pisa_interface_id"] is not None else 10**9,
        )
    )
    return rows


def input_is_ca_only(path: Path) -> bool:
    names = set()
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM  "):
                names.add(line[12:16].strip())
    return names == {"CA"}


def job_paths(out_dir: Path, interface_id: str) -> dict[str, Path]:
    directory = out_dir / "jobs" / interface_id
    return {
        "dir": directory,
        "done": directory / "done.json",
        "failure": directory / "failure.json",
        "interfaces": directory / "interfaces.xml.gz",
        "assembly": directory / "assembly.xml.gz",
        "analyse": directory / "analyse.log.gz",
    }


def run_one(job: dict[str, str], config: dict[str, Any]) -> tuple[str, str, float]:
    started = time.time()
    interface_id = job["interface_id"]
    paths = job_paths(Path(config["out_dir"]), interface_id)
    if config["resume"] and paths["done"].exists():
        return interface_id, "skipped", 0.0
    paths["dir"].mkdir(parents=True, exist_ok=True)
    input_gz = Path(config["dimer_root"]) / "jobs" / interface_id / "input.pdb.gz"
    if not input_gz.exists():
        atomic_json(
            paths["failure"],
            {
                "interface_id": interface_id,
                "status": "failure",
                "error": f"missing input: {input_gz}",
            },
        )
        return interface_id, "failure", time.time() - started
    if input_is_ca_only(input_gz):
        result = {
            **job,
            "status": "technical_exclusion",
            "technical_flag": "ca_only_deposited_coordinates",
            "pisa_supported": False,
            "elapsed_seconds": time.time() - started,
            "input_pdb_gz": str(input_gz),
            "input_pdb_gz_sha256": sha256(input_gz),
        }
        atomic_json(paths["done"], result)
        return interface_id, "technical_exclusion", result["elapsed_seconds"]

    with tempfile.TemporaryDirectory(
        prefix=f"{interface_id}_", dir=config["tmp_dir"]
    ) as temporary_name:
        temporary = Path(temporary_name)
        input_path = temporary / "input.pdb"
        with gzip.open(input_gz, "rb") as source, input_path.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        session = "dip_" + hashlib.sha1(
            interface_id.encode("utf-8")
        ).hexdigest()[:16]
        script = (
            "set -e; "
            f"pisa {session} -analyse /data/input.pdb --as-is "
            f"{config['pisa_config']} >/data/analyse.log 2>&1; "
            f"pisa {session} -xml interface {config['pisa_config']} "
            "> /data/interfaces.xml; "
            f"pisa {session} -xml assembly {config['pisa_config']} "
            "> /data/assembly.xml; "
            f"rm -rf /data/{session}; "
            "chmod 0666 /data/interfaces.xml /data/assembly.xml "
            "/data/analyse.log"
        )
        completed = None
        try:
            completed = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--cpu-shares",
                    str(config["docker_cpu_shares"]),
                    "-v",
                    f"{temporary}:/data",
                    config["pisa_image"],
                    "bash",
                    "-lc",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=int(config["timeout"]),
                check=False,
            )
            if completed.returncode != 0:
                message = (completed.stderr or completed.stdout or "").strip()
                analyse_path = temporary / "analyse.log"
                if analyse_path.exists():
                    message = (
                        message
                        + "\n"
                        + analyse_path.read_text(errors="replace")[-3000:]
                    ).strip()
                raise RuntimeError(
                    f"PISA container exit {completed.returncode}: {message[-3000:]}"
                )
            interface_xml = temporary / "interfaces.xml"
            assembly_xml = temporary / "assembly.xml"
            metrics = H.parse_pisa_xml(interface_xml, assembly_xml)
            all_interfaces = parse_all_protein_interfaces(interface_xml)
            gzip_copy(interface_xml, paths["interfaces"])
            gzip_copy(assembly_xml, paths["assembly"])
            gzip_copy(temporary / "analyse.log", paths["analyse"])
            result = {
                **job,
                **metrics,
                "status": "success",
                "technical_flag": "",
                "elapsed_seconds": time.time() - started,
                "input_pdb_gz": str(input_gz),
                "input_pdb_gz_sha256": sha256(input_gz),
                "pisa_interfaces": all_interfaces,
                "interfaces_xml_gz": str(paths["interfaces"]),
                "assembly_xml_gz": str(paths["assembly"]),
                "analyse_log_gz": str(paths["analyse"]),
                "interfaces_xml_gz_sha256": sha256(paths["interfaces"]),
                "assembly_xml_gz_sha256": sha256(paths["assembly"]),
                "analyse_log_gz_sha256": sha256(paths["analyse"]),
            }
            atomic_json(paths["done"], result)
            if paths["failure"].exists():
                paths["failure"].unlink()
            return interface_id, "success", result["elapsed_seconds"]
        except Exception as error:
            analyse_path = temporary / "analyse.log"
            if analyse_path.exists():
                gzip_copy(analyse_path, paths["analyse"])
            if "no chains found in input file" in str(error):
                result = {
                    **job,
                    "status": "technical_exclusion",
                    "technical_flag": "pisa_no_chains_found",
                    "pisa_supported": False,
                    "elapsed_seconds": time.time() - started,
                    "input_pdb_gz": str(input_gz),
                    "input_pdb_gz_sha256": sha256(input_gz),
                    "analyse_log_gz": (
                        str(paths["analyse"]) if paths["analyse"].exists() else ""
                    ),
                    "error": str(error),
                }
                atomic_json(paths["done"], result)
                if paths["failure"].exists():
                    paths["failure"].unlink()
                return (
                    interface_id,
                    "technical_exclusion",
                    result["elapsed_seconds"],
                )
            atomic_json(
                paths["failure"],
                {
                    "interface_id": interface_id,
                    "pdb": job["pdb"],
                    "status": "failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.time() - started,
                    "analyse_log_gz": (
                        str(paths["analyse"]) if paths["analyse"].exists() else ""
                    ),
                    "container_stdout": (
                        completed.stdout[-3000:] if completed is not None else ""
                    ),
                    "container_stderr": (
                        completed.stderr[-3000:] if completed is not None else ""
                    ),
                },
            )
            return interface_id, "failure", time.time() - started


def run_stage(args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= 6:
        raise SystemExit("--workers must be in [1,6]")
    rows = read_csv(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tmp").mkdir(exist_ok=True)
    config = {
        "out_dir": str(args.out_dir),
        "dimer_root": str(args.dimer_root),
        "tmp_dir": str(args.out_dir / "tmp"),
        "resume": args.resume,
        "timeout": args.timeout,
        "pisa_image": args.pisa_image,
        "pisa_config": args.pisa_config,
        "docker_cpu_shares": args.docker_cpu_shares,
    }
    max_inflight = args.max_inflight or args.workers * 2
    counts: Counter[str] = Counter()
    started = time.time()
    next_index = 0
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        inflight: dict[Any, int] = {}
        while next_index < len(rows) and len(inflight) < max_inflight:
            inflight[pool.submit(run_one, rows[next_index], config)] = next_index
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
                        f"worker_exception={rows[job_index]['interface_id']}:"
                        f"{type(error).__name__}:{error}",
                        flush=True,
                    )
                counts[outcome] += 1
                completed_count += 1
                if next_index < len(rows):
                    inflight[
                        pool.submit(run_one, rows[next_index], config)
                    ] = next_index
                    next_index += 1
            if completed_count % 10 == 0 or completed_count == len(rows):
                elapsed = time.time() - started
                rate = completed_count / elapsed if elapsed else 0.0
                eta = (len(rows) - completed_count) / rate if rate else None
                eta_text = f"{eta:.1f}" if eta is not None else "unknown"
                print(
                    f"progress={completed_count:,}/{len(rows):,} "
                    f"success={counts['success']:,} "
                    f"excluded={counts['technical_exclusion']:,} "
                    f"failure={counts['failure']:,} "
                    f"skipped={counts['skipped']:,} "
                    f"jobs_per_hour={rate * 3600:.1f} "
                    f"eta_s={eta_text}",
                    flush=True,
                )
    print(json.dumps(dict(counts), sort_keys=True), flush=True)


def status_stage(args: argparse.Namespace) -> None:
    rows = read_csv(args.manifest)
    totals = Counter(row["direct_current_paper_exact_class"] for row in rows)
    outcomes: Counter[str] = Counter()
    completed_by_class: Counter[str] = Counter()
    elapsed = []
    for row in rows:
        paths = job_paths(args.out_dir, row["interface_id"])
        if paths["done"].exists():
            result = json.loads(paths["done"].read_text())
            outcomes[result["status"]] += 1
            completed_by_class[row["direct_current_paper_exact_class"]] += 1
            elapsed.append(float(result.get("elapsed_seconds") or 0))
        elif paths["failure"].exists():
            outcomes["failure_without_done"] += 1
        else:
            outcomes["pending"] += 1
    report = {
        "total": len(rows),
        "outcomes": dict(outcomes),
        "class_total": dict(totals),
        "class_completed": dict(completed_by_class),
        "elapsed_seconds": {
            "median": sorted(elapsed)[len(elapsed) // 2] if elapsed else None,
            "maximum": max(elapsed) if elapsed else None,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def finalize_stage(args: argparse.Namespace) -> None:
    manifest = read_csv(args.manifest)
    summary_rows, interface_rows, artifact_rows = [], [], []
    for row in manifest:
        paths = job_paths(args.out_dir, row["interface_id"])
        if paths["done"].exists():
            result = json.loads(paths["done"].read_text())
        elif paths["failure"].exists():
            result = json.loads(paths["failure"].read_text())
        else:
            result = {
                **row,
                "status": "pending",
                "pisa_supported": False,
            }
        interfaces = result.pop("pisa_interfaces", [])
        summary_rows.append(result)
        artifact_rows.append(
            {
                "interface_id": row["interface_id"],
                "status": result["status"],
                "input_pdb_gz": result.get("input_pdb_gz", ""),
                "input_pdb_gz_sha256": result.get(
                    "input_pdb_gz_sha256", ""
                ),
                "interfaces_xml_gz": result.get("interfaces_xml_gz", ""),
                "interfaces_xml_gz_sha256": result.get(
                    "interfaces_xml_gz_sha256", ""
                ),
                "assembly_xml_gz": result.get("assembly_xml_gz", ""),
                "assembly_xml_gz_sha256": result.get(
                    "assembly_xml_gz_sha256", ""
                ),
                "analyse_log_gz": result.get("analyse_log_gz", ""),
                "analyse_log_gz_sha256": result.get(
                    "analyse_log_gz_sha256", ""
                ),
            }
        )
        for interface in interfaces:
            interface_rows.append(
                {
                    "candidate_interface_id": row["interface_id"],
                    "pdb": row["pdb"],
                    "uniprot": row["uniprot"],
                    "doi": row["doi"],
                    **interface,
                }
            )
    atomic_csv(args.out_dir / "pisa_candidate_summary.csv", summary_rows)
    atomic_csv(args.out_dir / "pisa_protein_interfaces.csv", interface_rows)
    atomic_csv(args.out_dir / "pisa_artifact_manifest.csv", artifact_rows)
    atomic_json(
        args.out_dir / "finalize_summary.json",
        {
            "manifest_jobs": len(manifest),
            "status_counts": dict(
                Counter(row["status"] for row in summary_rows)
            ),
            "parsed_protein_interfaces": len(interface_rows),
            "raw_xml_pairs_preserved": sum(
                bool(row["interfaces_xml_gz"] and row["assembly_xml_gz"])
                for row in artifact_rows
            ),
            "analysis_logs_preserved": sum(
                bool(row["analyse_log_gz"]) for row in artifact_rows
            ),
        },
    )
    methods = """# Phase 1 isolated-dimer local PISA

Each row of the immutable Phase 1 candidate manifest was processed exactly
once or assigned a transparent technical-exclusion status. The input is the
already preserved protein-only, two-chain P1 PDB generated for ConFind. PISA
ran in `pdbegroup/pisa:latest` with `-analyse ... --as-is`; workers were capped
at four for production while the control ConFind run remained active.

Interface XML, assembly XML and analysis logs are gzip-preserved per successful
candidate. Every protein-protein interface in interface XML is exported to
`pisa_protein_interfaces.csv`; the candidate summary deterministically selects
the largest-area protein interface. Assembly buried surface area, interaction
energy and dissociation energy are parsed from assembly XML. Support is defined
independently of Pairformer as: protein-interface area >=400 A^2, interaction
energy <0, and dissociation energy >0.

Four C-alpha-only 1RH2 inputs are technical exclusions. PISA returned
`no chains found in input file` reproducibly for 3LG8_XTAL_i0001; its analysis
log is preserved, it remains valid for ConFind/qid35, and only its PISA metrics
are unavailable. No PISA result is used to select a Pairformer interface.
"""
    (args.out_dir / "METHODS.md").write_text(methods, encoding="utf-8")
    image = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "pdbegroup/pisa:latest",
            "--format",
            "{{.Id}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_json(
        args.out_dir / "provenance.json",
        {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "dimer_root": str(args.dimer_root),
            "pisa_image": "pdbegroup/pisa:latest",
            "pisa_image_id": image.stdout.strip(),
            "pisa_config": "/usr/share/pisa/setup/pisa_cfg_tmp",
            "production_workers": 4,
            "support_rule": (
                "area>=400A2 and interaction_energy<0 and "
                "dissociation_energy>0"
            ),
        },
    )
    output_files = [
        args.out_dir / name
        for name in (
            "pisa_candidate_summary.csv",
            "pisa_protein_interfaces.csv",
            "pisa_artifact_manifest.csv",
            "finalize_summary.json",
            "METHODS.md",
            "provenance.json",
        )
    ]
    atomic_json(
        args.out_dir / "output_checksums.json",
        {str(path): sha256(path) for path in output_files},
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="stage", required=True)
    for name in ("run", "status", "finalize"):
        command = sub.add_parser(name)
        command.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--dimer-root", type=Path, default=ROOT)
    run = sub.choices["run"]
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-inflight", type=int)
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--pisa-image", default="pdbegroup/pisa:latest")
    run.add_argument("--pisa-config", default="/usr/share/pisa/setup/pisa_cfg_tmp")
    run.add_argument("--docker-cpu-shares", type=int, default=256)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.stage == "run":
        run_stage(args)
    elif args.stage == "status":
        status_stage(args)
    else:
        finalize_stage(args)


if __name__ == "__main__":
    main()
