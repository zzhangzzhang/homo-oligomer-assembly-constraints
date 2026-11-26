"""Deterministic entity-level target grouping and manifest audits."""

import csv
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .aliases import normalize_gene_aliases
from .doi import DOIError, normalize_doi
from .hashing import sha256_file
from .rcsb import read_cache
from .serialization import canonical_json


MANIFEST_FIELDS = [
    "group_id",
    "doi",
    "uniprot_accession",
    "target_names",
    "pdb_ids",
    "entities",
    "paper_path",
    "paper_sha256",
    "paper_bytes",
]


def _json_load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_doi_mappings(path):
    """Load either DOI→PDB-list or PDB→DOI JSON without reading label data."""
    raw = _json_load(path)
    if not isinstance(raw, dict):
        raise ValueError("DOI mapping must be a JSON object")
    pairs = []
    for key, value in raw.items():
        if isinstance(value, list):
            for pdb_id in value:
                pairs.append((key, pdb_id))
        elif value is None or isinstance(value, str):
            if value:
                pairs.append((value, key))
        else:
            raise ValueError("unsupported DOI mapping value for %r" % key)
    return pairs


def canonicalize_mappings(pairs):
    mapping = defaultdict(set)
    audit = []
    raw_dois = defaultdict(set)
    for raw_doi, raw_pdb in pairs:
        pdb_id = str(raw_pdb or "").strip().upper()
        try:
            doi = normalize_doi(raw_doi)
        except DOIError as exc:
            audit.append(
                {
                    "scope": "mapping",
                    "doi_raw": str(raw_doi),
                    "pdb_id": pdb_id,
                    "reason": "invalid_doi",
                    "detail": str(exc),
                }
            )
            continue
        if not pdb_id:
            audit.append(
                {
                    "scope": "mapping",
                    "doi": doi,
                    "pdb_id": "",
                    "reason": "missing_pdb_id",
                    "detail": "",
                }
            )
            continue
        mapping[doi].add(pdb_id)
        raw_dois[doi].add(str(raw_doi))
    normalized = {doi: sorted(mapping[doi]) for doi in sorted(mapping)}
    merges = [
        {"doi": doi, "raw_dois": sorted(values), "raw_doi_count": len(values)}
        for doi, values in sorted(raw_dois.items())
        if len(values) > 1
    ]
    return normalized, sorted(audit, key=_audit_key), merges


def _status_rows(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        rows = []
        for key, value in raw.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("doi", key)
                rows.append(row)
        return rows
    raise ValueError("download status must be an object or list")


def canonicalize_downloads(path):
    downloads = defaultdict(list)
    invalid = []
    for row in _status_rows(_json_load(path)):
        raw_doi = row.get("doi")
        try:
            doi = normalize_doi(raw_doi)
        except DOIError as exc:
            invalid.append(
                {
                    "scope": "download",
                    "doi_raw": str(raw_doi),
                    "reason": "invalid_doi",
                    "detail": str(exc),
                }
            )
            continue
        downloads[doi].append(dict(row))
    for rows in downloads.values():
        rows.sort(key=lambda row: (str(row.get("file") or ""), str(row.get("source") or "")))
    return downloads, sorted(invalid, key=_audit_key)


def _provenance_file(path, role):
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    if not source.is_file():
        raise ValueError("provenance source is not a file: %s" % source)
    return {
        "path": str(source),
        "role": role,
        "bytes": stat.st_size,
        "sha256": sha256_file(source),
    }


def development_baseline_snapshot(source_paths):
    """Freeze deterministic ``(role, path)`` provenance as development data."""
    files = [
        _provenance_file(path, role)
        for role, path in sorted(source_paths)
        if path is not None
    ]
    digest_payload = [
        {
            "role": row["role"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in files
    ]
    snapshot_sha256 = hashlib.sha256(
        json.dumps(
            digest_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "role": "development",
        "holdout_eligible": False,
        "provenance_schema_version": 1,
        "files": files,
        "snapshot_sha256": snapshot_sha256,
    }


def development_run_snapshot(sources):
    """Recursively freeze named pilot artifacts as development-only data."""
    expanded = []
    for role, raw_path in sorted(sources):
        source = Path(raw_path).expanduser().resolve()
        if source.is_file():
            expanded.append((role, source))
            continue
        if not source.is_dir():
            raise ValueError("development source does not exist: %s" % source)
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            expanded.append(("%s/%s" % (role, relative), path))
    snapshot = development_baseline_snapshot(expanded)
    snapshot["source_roots"] = [
        {"role": role, "path": str(Path(path).expanduser().resolve())}
        for role, path in sorted(sources)
    ]
    return snapshot


def _paper_format_error(path):
    """Return a deterministic reason when a supported paper is plainly corrupt."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with path.open("rb") as handle:
            return None if handle.read(5) == b"%PDF-" else "invalid_pdf_header"
    if suffix in {".xml", ".nxml"}:
        try:
            ET.parse(path)
        except (ET.ParseError, OSError):
            return "invalid_xml"
        return None
    return "unsupported_paper_type"


def _memoized_sha256(path, checksum_cache):
    """Hash a stable file identity once while retaining content conflict checks."""
    stat = path.stat()
    key = (
        str(path.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if key not in checksum_cache:
        checksum_cache[key] = sha256_file(path)
    return checksum_cache[key]


def select_paper(doi, rows, checksum_cache=None):
    checksum_cache = checksum_cache if checksum_cache is not None else {}
    rejected = []
    candidates = []
    for row in rows:
        if row.get("status") != "success":
            rejected.append("status_%s" % (row.get("status") or "missing"))
            continue
        raw_path = row.get("file")
        if not raw_path:
            rejected.append("missing_path")
            continue
        path = Path(raw_path).expanduser()
        try:
            size = path.stat().st_size
            if not path.is_file() or size <= 0 or not os.access(str(path), os.R_OK):
                rejected.append("unreadable_or_empty")
                continue
            format_error = _paper_format_error(path)
            if format_error:
                rejected.append(format_error)
                continue
            candidates.append(
                (str(path.resolve()), _memoized_sha256(path, checksum_cache), size)
            )
        except OSError:
            rejected.append("unreadable_or_missing")
    if not candidates:
        return None, sorted(set(rejected)) or ["missing_download_status"]
    candidates.sort()
    unique_content = {(digest, size) for _, digest, size in candidates}
    if len(unique_content) > 1:
        return None, ["conflicting_successful_papers"]
    path, digest, size = candidates[0]
    return {"paper_path": path, "paper_sha256": digest, "paper_bytes": size}, []


def _paper_rejection_category(reason):
    """Group strict paper failures without erasing their exact reason."""
    if reason in {"missing_download_status", "missing_path", "unreadable_or_missing"}:
        return "missing"
    if reason == "unreadable_or_empty":
        return "unreadable"
    if reason in {"invalid_pdf_header", "invalid_xml", "unsupported_paper_type"}:
        return "invalid_format"
    if reason == "conflicting_successful_papers":
        return "conflicting_successful_papers"
    if reason.startswith("status_"):
        return "status_not_success"
    return "other"


def _entity_identity_error(pdb_id, entity, duplicate_ids):
    entity_id = entity.get("entity_id")
    rcsb_entity_id = entity.get("rcsb_entity_id")
    if not entity_id:
        return "ambiguous_entity_mapping", "missing entity_id"
    if entity_id in duplicate_ids:
        return "ambiguous_entity_mapping", "duplicate entity_id"
    if rcsb_entity_id and rcsb_entity_id.upper() != ("%s_%s" % (pdb_id, entity_id)).upper():
        return "ambiguous_entity_mapping", "inconsistent rcsb_entity_id"
    accessions = sorted(set(entity.get("uniprot_accessions") or []))
    if not accessions:
        return "no_uniprot_mapping", ""
    if len(accessions) > 1:
        return "unresolved_multi_accession_or_chimeric_identity", ",".join(accessions)
    return None, None


def _entry_identity_metadata(record, accession):
    """Expose an entry title only when it maps to one exact target."""
    accessions = sorted(
        {
            str(value).strip().upper()
            for value in record.get("entry_distinct_uniprot_accessions") or []
            if str(value).strip()
        }
    )
    polymer_entity_count = record.get("entry_polymer_entity_count")
    safe_title = None
    if polymer_entity_count == 1 and accessions == [accession]:
        title = record.get("entry_title")
        if isinstance(title, str) and title.strip():
            safe_title = title.strip()
    return {
        "entry_title": safe_title,
        "entry_polymer_entity_count": polymer_entity_count,
        "entry_distinct_uniprot_accessions": accessions,
    }


def group_targets(doi_mapping, rcsb_records):
    groups = {}
    exclusions = []
    for doi in sorted(doi_mapping):
        for pdb_id in doi_mapping[doi]:
            record = rcsb_records.get(pdb_id)
            if not record or record.get("status") != "ok":
                status = (record or {}).get("status") or "missing"
                if status == "not_found":
                    reason = "rcsb_entry_not_found"
                elif status in {"missing", "retry_exhausted"}:
                    reason = "rcsb_mapping_unresolved"
                else:
                    reason = "rcsb_mapping_failed"
                exclusions.append(
                    {
                        "scope": "pdb",
                        "doi": doi,
                        "pdb_id": pdb_id,
                        "entity_id": "",
                        "reason": reason,
                        "detail": status,
                    }
                )
                continue
            entities = record.get("entities") or []
            entity_id_counts = defaultdict(int)
            for entity in entities:
                if entity.get("entity_id"):
                    entity_id_counts[entity["entity_id"]] += 1
            duplicate_ids = {
                entity_id for entity_id, count in entity_id_counts.items() if count > 1
            }
            if not entities:
                exclusions.append(
                    {
                        "scope": "pdb",
                        "doi": doi,
                        "pdb_id": pdb_id,
                        "entity_id": "",
                        "reason": "no_polymer_entities",
                        "detail": "",
                    }
                )
            for entity in entities:
                reason, detail = _entity_identity_error(pdb_id, entity, duplicate_ids)
                if reason:
                    exclusions.append(
                        {
                            "scope": "entity",
                            "doi": doi,
                            "pdb_id": pdb_id,
                            "entity_id": entity.get("entity_id") or "",
                            "reason": reason,
                            "detail": detail,
                        }
                    )
                    continue
                accession = entity["uniprot_accessions"][0].upper()
                key = (doi, accession)
                group = groups.setdefault(
                    key,
                    {"doi": doi, "uniprot_accession": accession, "entities": []},
                )
                genes, gene_audit = normalize_gene_aliases(
                    entity.get("source_genes"),
                    prior_audit=entity.get("source_gene_alias_audit"),
                )
                canonical_entity = {
                    "pdb_id": pdb_id,
                    "entity_id": entity["entity_id"],
                    "chains": sorted(set(entity.get("chains") or [])),
                    "description": entity.get("description"),
                    "polymer_type": entity.get("polymer_type"),
                    "mutation_description": entity.get("mutation_description"),
                    "sequence_length": entity.get("sequence_length"),
                    "mutation_count": entity.get("mutation_count"),
                    "source_organisms": sorted(set(entity.get("source_organisms") or [])),
                    "source_genes": genes,
                    "source_gene_alias_audit": gene_audit,
                }
                canonical_entity.update(_entry_identity_metadata(record, accession))
                group["entities"].append(canonical_entity)
    for group in groups.values():
        group["entities"].sort(key=lambda row: (row["pdb_id"], row["entity_id"]))
    return groups, sorted(exclusions, key=_audit_key)


def _group_id(doi, accession):
    token = hashlib.sha256((doi + "\t" + accession).encode("utf-8")).hexdigest()[:20]
    return "doi_uniprot_" + token


def finalize_manifest(groups, downloads):
    manifest = []
    exclusions = []
    checksum_cache = {}
    paper_by_doi = {
        doi: select_paper(doi, downloads.get(doi, []), checksum_cache)
        for doi in sorted({key[0] for key in groups})
    }
    for (doi, accession), group in sorted(groups.items()):
        paper, errors = paper_by_doi[doi]
        if paper is None:
            exclusions.append(
                {
                    "scope": "target_group",
                    "doi": doi,
                    "pdb_id": ",".join(sorted({row["pdb_id"] for row in group["entities"]})),
                    "entity_id": "",
                    "uniprot_accession": accession,
                    "reason": "paper_unavailable",
                    "detail": ",".join(errors),
                }
            )
            continue
        names = sorted(
            {row["description"] for row in group["entities"] if row.get("description")}
        )
        row = {
            "group_id": _group_id(doi, accession),
            "doi": doi,
            "uniprot_accession": accession,
            "target_names": names,
            "pdb_ids": sorted({entity["pdb_id"] for entity in group["entities"]}),
            "entities": group["entities"],
        }
        row.update(paper)
        manifest.append(row)
    manifest.sort(key=lambda row: (row["doi"], row["uniprot_accession"]))
    return manifest, sorted(exclusions, key=_audit_key)


def _reconcile_stage(canonical_sources, new_keys):
    legacy_keys = set(canonical_sources)
    details = []
    for key in sorted(new_keys | legacy_keys):
        if key in new_keys and key not in legacy_keys:
            action = "addition"
        elif key in legacy_keys and key not in new_keys:
            action = "exclusion"
        elif len(canonical_sources[key]) > 1:
            action = "merge"
        else:
            action = "retained"
        details.append(
            {
                "doi": key[0],
                "uniprot_accession": key[1],
                "action": action,
                "legacy_keys": canonical_sources.get(key, []),
            }
        )
    return {
        "canonicalizable_legacy_count": len(legacy_keys),
        "canonical_count": len(new_keys),
        "additions": sum(row["action"] == "addition" for row in details),
        "merges": sum(row["action"] == "merge" for row in details),
        "exclusions": sum(row["action"] == "exclusion" for row in details),
        "retained": sum(row["action"] == "retained" for row in details),
    }, details


def _increment_reason_counts(counts, reasons):
    for reason in sorted(set(reasons)):
        counts[reason] += 1


def reconcile_legacy(path, groups, manifest, downloads):
    """Compare legacy and canonical groups at explicitly matched paper stages."""
    if not path:
        return {"provided": False}, []
    raw = _json_load(path)
    if not isinstance(raw, dict):
        raise ValueError("legacy groups must be a JSON object")
    canonical_sources = defaultdict(list)
    legacy_doi_by_key = {}
    unresolved = []
    strict_rejection_reason_counts = defaultdict(int)
    strict_rejection_category_counts = defaultdict(int)
    status_success_strict_rejection_reason_counts = defaultdict(int)
    for key in sorted(raw):
        if "::" not in key:
            unresolved.append({"legacy_key": key, "reason": "malformed_legacy_key"})
            _increment_reason_counts(
                strict_rejection_reason_counts, ["malformed_legacy_key"]
            )
            _increment_reason_counts(
                strict_rejection_category_counts, ["invalid_legacy_identity"]
            )
            continue
        raw_doi, accession = key.rsplit("::", 1)
        try:
            doi = normalize_doi(raw_doi)
        except DOIError:
            unresolved.append({"legacy_key": key, "reason": "invalid_legacy_doi"})
            _increment_reason_counts(
                strict_rejection_reason_counts, ["invalid_legacy_doi"]
            )
            _increment_reason_counts(
                strict_rejection_category_counts, ["invalid_legacy_identity"]
            )
            continue
        legacy_doi_by_key[key] = doi
        if accession in {"SINGLE", "NO_UNIPROT"}:
            unresolved.append({"legacy_key": key, "reason": "unresolved_legacy_target"})
            continue
        canonical_sources[(doi, accession.upper())].append(key)
    checksum_cache = {}
    status_success_dois = {
        doi
        for doi, rows in downloads.items()
        if any(row.get("status") == "success" for row in rows)
    }
    paper_selection = {
        doi: select_paper(doi, downloads.get(doi, []), checksum_cache)
        for doi in sorted(set(legacy_doi_by_key.values()))
    }
    strict_readable_dois = {
        doi for doi, (paper, _) in paper_selection.items() if paper is not None
    }
    status_success_raw_count = sum(
        doi in status_success_dois for doi in legacy_doi_by_key.values()
    )
    strict_readable_raw_count = sum(
        doi in strict_readable_dois for doi in legacy_doi_by_key.values()
    )
    for key, doi in sorted(legacy_doi_by_key.items()):
        paper, reasons = paper_selection[doi]
        if paper is not None:
            continue
        _increment_reason_counts(strict_rejection_reason_counts, reasons)
        _increment_reason_counts(
            strict_rejection_category_counts,
            [_paper_rejection_category(reason) for reason in reasons],
        )
        if doi in status_success_dois:
            _increment_reason_counts(
                status_success_strict_rejection_reason_counts, reasons
            )
    paper_gated_sources = defaultdict(list)
    for key, values in canonical_sources.items():
        if key[0] in strict_readable_dois:
            paper_gated_sources[key].extend(values)
    raw_stage, raw_details = _reconcile_stage(canonical_sources, set(groups))
    paper_stage, paper_details = _reconcile_stage(
        paper_gated_sources,
        {(row["doi"], row["uniprot_accession"]) for row in manifest},
    )
    raw_stage.update(
        {
            "legacy_stage": "raw",
            "legacy_stage_count": len(raw),
            "canonical_stage": "before_strict_paper_gate",
        }
    )
    paper_stage.update(
        {
            "legacy_stage": "strict_readable",
            "legacy_stage_count": strict_readable_raw_count,
            "canonical_stage": "after_strict_paper_gate",
        }
    )
    details = [
        dict(row, stage="before_paper_gate") for row in raw_details
    ] + [dict(row, stage="after_paper_gate") for row in paper_details]
    summary = {
        "provided": True,
        "legacy_group_count": len(raw),
        "legacy_status_success_group_count": status_success_raw_count,
        "legacy_strict_readable_group_count": strict_readable_raw_count,
        # Historical callers supplied a status-file success count under this
        # ambiguous name. Keep that interpretation, and expose strict
        # readability separately so unlike stages are never compared.
        "legacy_paper_gated_group_count": status_success_raw_count,
        "legacy_paper_gated_count_semantics": "status_success_legacy",
        "stage_definitions": {
            "raw": "all legacy mapping entries before DOI or target validation",
            "status_success": (
                "legacy entries with a normalizable DOI and at least one "
                "download-status row marked success; files are not inspected"
            ),
            "strict_readable": (
                "legacy entries whose DOI has one unambiguous, readable, "
                "supported-format paper selected by select_paper"
            ),
            "canonicalizable": (
                "unique normalized DOI x non-placeholder UniProt keys"
            ),
            "canonical": "unique normalized DOI x exact RCSB-mapped UniProt keys",
        },
        "rejection_count_semantics": (
            "counts are legacy-entry reason incidences; one rejected entry can "
            "contribute once to each distinct select_paper reason"
        ),
        "strict_legacy_rejection_reason_counts": dict(
            sorted(strict_rejection_reason_counts.items())
        ),
        "strict_legacy_rejection_category_counts": dict(
            sorted(strict_rejection_category_counts.items())
        ),
        "status_success_but_strict_rejection_reason_counts": dict(
            sorted(status_success_strict_rejection_reason_counts.items())
        ),
        "before_paper_gate": raw_stage,
        "after_paper_gate": paper_stage,
        "unresolved_legacy": unresolved,
    }
    return summary, details


def _audit_key(row):
    return tuple(
        str(row.get(field) or "")
        for field in ("scope", "doi", "doi_raw", "pdb_id", "entity_id", "reason", "detail")
    )


def _write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _write_tsv(path, rows, fields=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({field for row in rows for field in row})
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                        if isinstance(row.get(field), (list, dict))
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def build_manifest(
    doi_mapping_path,
    rcsb_cache_path,
    download_status_path,
    manifest_jsonl_path,
    manifest_tsv_path,
    audit_json_path,
    exclusions_tsv_path,
    legacy_groups_path=None,
    reconciliation_tsv_path=None,
    expected_legacy_count=None,
    expected_legacy_paper_gated_count=None,
):
    pairs = load_doi_mappings(doi_mapping_path)
    mapping, mapping_exclusions, doi_merges = canonicalize_mappings(pairs)
    rcsb_records = read_cache(rcsb_cache_path)
    groups, entity_exclusions = group_targets(mapping, rcsb_records)
    downloads, download_exclusions = canonicalize_downloads(download_status_path)
    manifest, paper_exclusions = finalize_manifest(groups, downloads)
    exclusions = sorted(
        mapping_exclusions + entity_exclusions + download_exclusions + paper_exclusions,
        key=_audit_key,
    )
    reconciliation, reconciliation_rows = reconcile_legacy(
        legacy_groups_path, groups, manifest, downloads
    )
    implementation_dir = Path(__file__).resolve().parent
    development_baseline = development_baseline_snapshot(
        [
            ("code_doi_normalization", implementation_dir / "doi.py"),
            ("code_manifest_builder", Path(__file__)),
            ("code_rcsb_cache", implementation_dir / "rcsb.py"),
            ("input_doi_mapping", doi_mapping_path),
            ("input_download_status", download_status_path),
            ("input_legacy_groups", legacy_groups_path),
            ("input_rcsb_cache", rcsb_cache_path),
        ]
    )
    blockers = []
    unresolved_rcsb = sum(
        row.get("reason") == "rcsb_mapping_unresolved"
        and row.get("detail") in {"missing", "retry_exhausted"}
        for row in exclusions
    )
    if unresolved_rcsb:
        blockers.append("%d PDB mappings remain unresolved" % unresolved_rcsb)
    if expected_legacy_count is not None and reconciliation.get("provided"):
        if reconciliation["legacy_group_count"] != expected_legacy_count:
            blockers.append(
                "legacy count is %d, expected %d"
                % (reconciliation["legacy_group_count"], expected_legacy_count)
            )
    if expected_legacy_paper_gated_count is not None and reconciliation.get("provided"):
        if (
            reconciliation["legacy_status_success_group_count"]
            != expected_legacy_paper_gated_count
        ):
            blockers.append(
                "status-success legacy count is %d, expected historical "
                "paper-gated count %d"
                % (
                    reconciliation["legacy_status_success_group_count"],
                    expected_legacy_paper_gated_count,
                )
            )
    audit = {
        "policy": {
            "unit": "normalized DOI x exact RCSB-mapped UniProt accession",
            "excluded_identity_reasons": [
                "no_uniprot_mapping",
                "ambiguous_entity_mapping",
                "unresolved_multi_accession_or_chimeric_identity",
            ],
            "metadata_source": "RCSB identity and concise construct metadata only",
        },
        "counts": {
            "input_pairs": len(pairs),
            "normalized_dois": len(mapping),
            "input_pdbs": len({pdb for values in mapping.values() for pdb in values}),
            "resolved_target_groups_before_paper_gate": len(groups),
            "manifest_groups": len(manifest),
            "exclusions": len(exclusions),
        },
        "doi_merges": doi_merges,
        "exclusion_reason_counts": {
            reason: sum(row.get("reason") == reason for row in exclusions)
            for reason in sorted({row.get("reason") for row in exclusions})
        },
        "reconciliation": reconciliation,
        "development_baseline": development_baseline,
        "full_data_blockers": blockers,
    }
    _write_jsonl(manifest_jsonl_path, manifest)
    _write_tsv(manifest_tsv_path, manifest, MANIFEST_FIELDS)
    _write_json(audit_json_path, audit)
    _write_tsv(exclusions_tsv_path, exclusions)
    if reconciliation_tsv_path:
        _write_tsv(
            reconciliation_tsv_path,
            reconciliation_rows,
            ["stage", "doi", "uniprot_accession", "action", "legacy_keys"],
        )
    return manifest, audit, exclusions
