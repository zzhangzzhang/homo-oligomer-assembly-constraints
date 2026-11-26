"""Predictor-blind paper-level context for candidate re-extraction.

This module consumes only immutable v5 manifest, parsed-output, and request
metadata artifacts. Candidate claims are retained as retrieval pointers, never
as accepted findings or adjudicated target attribution.
"""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .doi import normalize_doi
from .hashing import sha256_file
from .serialization import canonical_json
from .text import extract_text, normalize_text

INDEX_SCHEMA_VERSION = "contextual_paper_index_v1"
DOSSIER_SCHEMA_VERSION = "contextual_paper_dossier_v1"
DEFAULT_TOKEN_CAP = 160_000
MIN_TOKEN_CAP = 120_000
MAX_TOKEN_CAP = 200_000
STREAM_COMMIT_ROWS = 500
CHAIN_SEED = hashlib.sha256(b"contextual-jsonl-chain-v1").hexdigest()

TARGET_FIELDS = (
    "group_id",
    "doi",
    "uniprot_accession",
    "target_names",
    "pdb_ids",
    "entities",
    "paper_path",
    "paper_sha256",
    "paper_bytes",
)
ENTITY_FIELDS = (
    "pdb_id",
    "entity_id",
    "chains",
    "description",
    "polymer_type",
    "mutation_description",
    "sequence_length",
    "mutation_count",
    "source_organisms",
    "source_genes",
    "source_gene_alias_audit",
    "entry_title",
    "entry_polymer_entity_count",
    "entry_distinct_uniprot_accessions",
)
PASSAGE_FIELDS = (
    "source_id",
    "base_source_id",
    "source_passage_id",
    "section",
    "page",
    "block",
    "block_index",
    "sentence_start",
    "sentence_end",
    "char_start",
    "char_end",
    "provenance",
    "layout_diagnostics",
    "element_path",
    "layout_transform",
    "requires_abstention",
    "text",
)
POINTER_FORBIDDEN_PARTS = (
    "accept",
    "adjudicat",
    "curat",
    "validat",
    "verified",
    "rejection",
    "identity_anchor_ids",
    "deterministic_decision",
)
PREDICTOR_FORBIDDEN_PARTS = (
    "pisa",
    "coevolution",
    "co_evolution",
    "interface_size",
    "interface_area",
    "interface_hydrophobic",
    "hydrophobicity",
    "accepted",
    "acceptance",
    "acceptance_label",
    "accepted_label",
    "curated_label",
    "validation_label",
    "deterministic_identity_decision",
    "deterministic_target_decision",
)
POLICY = {
    "candidate_claim_role": "retrieval_pointer_not_truth",
    "originating_group_role": "pointer_not_adjudicated_attribution",
    "no_candidate_recall_eligibility": (
        "doi_complete_and_no_group_in_candidate_groups"
    ),
    "no_candidate_previous_outputs_used": False,
    "candidate_fields_removed": list(POINTER_FORBIDDEN_PARTS),
    "predictor_fields_forbidden": list(PREDICTOR_FORBIDDEN_PARTS),
    "bibliography_policy": (
        "omit_only_explicit_or_conservative_terminal_reference_blocks"
    ),
    "context_priority": [
        "candidate_and_base_blocks",
        "candidate_neighbors_plus_minus_2",
        "identity_anchor_blocks",
        "referenced_captions_and_headings",
        "front_matter",
        "complete_relevant_sections",
        "document_order_fill",
    ],
    "default_token_cap": DEFAULT_TOKEN_CAP,
    "token_encoding": "o200k_base",
    "noncontiguous_blocks_joined": False,
}


class ContextualIndexError(RuntimeError):
    """Raised when contextual provenance or index invariants do not hold."""

    def __init__(self, message, code="contextual_index_error", details=None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def _sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _policy_sha256():
    return _sha256_json(POLICY)


def _code_identity():
    root = Path(__file__).resolve().parent
    paths = [
        Path(__file__).resolve(),
        root / "text.py",
        root / "serialization.py",
        root / "manifest.py",
    ]
    files = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.name)
    ]
    return {"files": files, "sha256": _sha256_json(files)}


def _atomic_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _path_identity(path):
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    if not source.is_file():
        raise ContextualIndexError("contextual input is not a file: %s" % source)
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _canonical_target(row):
    target = {field: row.get(field) for field in TARGET_FIELDS if field in row}
    required = (
        "group_id",
        "doi",
        "uniprot_accession",
        "paper_path",
        "paper_sha256",
    )
    missing = [field for field in required if not str(target.get(field) or "").strip()]
    if missing:
        raise ContextualIndexError(
            "manifest row lacks required fields: %s" % ",".join(missing)
        )
    target["group_id"] = str(target["group_id"])
    target["doi"] = normalize_doi(target["doi"])
    target["uniprot_accession"] = str(target["uniprot_accession"]).upper()
    target["target_names"] = sorted(
        {
            str(value).strip()
            for value in target.get("target_names") or []
            if str(value).strip()
        }
    )
    target["pdb_ids"] = sorted(
        {
            str(value).strip().upper()
            for value in target.get("pdb_ids") or []
            if str(value).strip()
        }
    )
    target["entities"] = sorted(
        [
            {field: entity.get(field) for field in ENTITY_FIELDS if field in entity}
            for entity in target.get("entities") or []
            if isinstance(entity, dict)
        ],
        key=lambda item: (
            str(item.get("pdb_id") or ""),
            str(item.get("entity_id") or ""),
            canonical_json(item),
        ),
    )
    target["paper_path"] = str(Path(target["paper_path"]).expanduser().resolve())
    target["paper_sha256"] = str(target["paper_sha256"]).lower()
    if target.get("paper_bytes") is not None:
        target["paper_bytes"] = int(target["paper_bytes"])
    return target


def _forbidden_pointer_key(key):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return any(part in normalized for part in POINTER_FORBIDDEN_PARTS)


def sanitize_candidate_pointer(value):
    """Remove all adjudication fields while preserving extracted claim hints."""
    if isinstance(value, dict):
        return {
            str(key): sanitize_candidate_pointer(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if not _forbidden_pointer_key(key)
        }
    if isinstance(value, list):
        return [sanitize_candidate_pointer(item) for item in value]
    return value


def _candidate_pointers(row):
    result = row.get("result") if isinstance(row.get("result"), dict) else row
    pointers = []
    for kind, field in (("state", "state_observations"), ("event", "dynamic_events")):
        values = result.get(field) or []
        if not isinstance(values, list):
            raise ContextualIndexError("%s must be an array" % field)
        for ordinal, candidate in enumerate(values):
            if not isinstance(candidate, dict):
                raise ContextualIndexError("%s candidate must be an object" % kind)
            pointers.append(
                {
                    "candidate_kind": kind,
                    "candidate_ordinal": ordinal,
                    "pointer_status": "unadjudicated_hint",
                    "candidate": sanitize_candidate_pointer(candidate),
                }
            )
    return pointers


def _clean_passage(passage):
    if not isinstance(passage, dict):
        return None
    cleaned = {
        field: passage.get(field) for field in PASSAGE_FIELDS if field in passage
    }
    if not str(cleaned.get("source_id") or ""):
        return None
    return cleaned


def _raw_identity_anchors(metadata):
    graph = metadata.get("identity_graph")
    if not isinstance(graph, dict):
        return []
    nodes = {
        str(node.get("node_id")): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    anchors = {}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source_id") or "")
        quote = str(edge.get("anchor_quote") or "")
        if not source_id or not quote or source_id.startswith("manifest:"):
            continue
        node = nodes.get(str(edge.get("source_node_id") or ""), {})
        item = {
            "source_id": source_id,
            "anchor_quote": quote,
            "source_node_label": str(node.get("label") or ""),
            "source_node_type": str(node.get("node_type") or ""),
        }
        if isinstance(edge.get("provenance"), dict):
            item["provenance"] = sanitize_candidate_pointer(edge["provenance"])
        item["anchor_candidate_id"] = "anchor_" + _sha256_json(item)[:20]
        anchors[item["anchor_candidate_id"]] = item
    return [anchors[key] for key in sorted(anchors)]


def _create_schema(connection):
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=FULL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE streams (
            role TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            device INTEGER NOT NULL,
            inode INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL DEFAULT 0,
            line_count INTEGER NOT NULL DEFAULT 0,
            chain_sha256 TEXT NOT NULL,
            complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0, 1))
        );
        CREATE TABLE targets (
            group_id TEXT PRIMARY KEY,
            doi TEXT NOT NULL,
            accession TEXT NOT NULL,
            target_json TEXT NOT NULL
        );
        CREATE INDEX targets_doi_idx ON targets (doi, accession, group_id);
        CREATE TABLE candidate_groups (
            group_id TEXT PRIMARY KEY REFERENCES targets(group_id),
            custom_id TEXT NOT NULL,
            candidate_count INTEGER NOT NULL
        );
        CREATE TABLE candidates (
            group_id TEXT NOT NULL REFERENCES candidate_groups(group_id),
            kind TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            pointer_json TEXT NOT NULL,
            PRIMARY KEY (group_id, kind, ordinal)
        );
        CREATE TABLE request_context (
            group_id TEXT PRIMARY KEY REFERENCES candidate_groups(group_id),
            custom_id TEXT NOT NULL,
            source_sha256 TEXT,
            selected_passages_json TEXT NOT NULL,
            identity_anchors_json TEXT NOT NULL
        );
        """
    )


def _open_new_index(index_path, inputs):
    path = Path(index_path).expanduser().resolve()
    if path.exists():
        raise ContextualIndexError(
            "contextual index already exists; use explicit resume: %s" % path
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=60)
    try:
        _create_schema(connection)
        code = _code_identity()
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "policy_json": canonical_json(POLICY),
            "policy_sha256": _policy_sha256(),
            "code_json": canonical_json(code),
            "code_sha256": code["sha256"],
            "build_complete": "0",
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        for role, identity in inputs.items():
            connection.execute(
                "INSERT INTO streams "
                "(role,path,bytes,mtime_ns,device,inode,byte_offset,line_count,"
                "chain_sha256,complete) VALUES (?,?,?,?,?,?,?,?,?,0)",
                (
                    role,
                    identity["path"],
                    identity["bytes"],
                    identity["mtime_ns"],
                    identity["device"],
                    identity["inode"],
                    0,
                    0,
                    CHAIN_SEED,
                ),
            )
        connection.commit()
        return connection
    except Exception:
        connection.close()
        if path.exists():
            path.unlink()
        raise


def _metadata(connection):
    return dict(connection.execute("SELECT key,value FROM metadata").fetchall())


def _verify_runtime_identity(connection, require_complete=False):
    metadata = _metadata(connection)
    if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ContextualIndexError("contextual index schema mismatch")
    if metadata.get("policy_sha256") != _policy_sha256():
        raise ContextualIndexError("contextual index policy provenance mismatch")
    if metadata.get("code_sha256") != _code_identity()["sha256"]:
        raise ContextualIndexError("contextual index code provenance mismatch")
    if require_complete and metadata.get("build_complete") != "1":
        raise ContextualIndexError("contextual index is incomplete")
    return metadata


def _verify_stream_identities(connection, inputs):
    recorded = {
        row[0]: {
            "path": row[1],
            "bytes": row[2],
            "mtime_ns": row[3],
            "device": row[4],
            "inode": row[5],
        }
        for row in connection.execute(
            "SELECT role,path,bytes,mtime_ns,device,inode FROM streams"
        )
    }
    if recorded != inputs:
        raise ContextualIndexError("contextual index input provenance mismatch")


def _chain_step(previous, raw):
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _stream_jsonl(
    connection,
    role,
    processor,
    interrupt_after_records=None,
):
    row = connection.execute(
        "SELECT path,bytes,byte_offset,line_count,chain_sha256,complete "
        "FROM streams WHERE role=?",
        (role,),
    ).fetchone()
    if row is None:
        raise ContextualIndexError("missing stream state: %s" % role)
    path, expected_bytes, offset, line_count, chain, complete = row
    if complete:
        return 0
    processed = 0
    pending = 0
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        while True:
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                raise ContextualIndexError(
                    "%s contains a blank JSONL line at %d" % (role, line_count + 1)
                )
            try:
                value = json.loads(raw)
            except Exception as exc:
                raise ContextualIndexError(
                    "%s has invalid JSON at line %d" % (role, line_count + 1)
                ) from exc
            if not isinstance(value, dict):
                raise ContextualIndexError(
                    "%s line %d is not an object" % (role, line_count + 1)
                )
            processor(connection, value)
            chain = _chain_step(chain, raw)
            line_count += 1
            processed += 1
            pending += 1
            offset = handle.tell()
            if pending >= STREAM_COMMIT_ROWS:
                connection.execute(
                    "UPDATE streams SET byte_offset=?,line_count=?,chain_sha256=? "
                    "WHERE role=?",
                    (offset, line_count, chain, role),
                )
                connection.commit()
                pending = 0
            if (
                interrupt_after_records is not None
                and processed >= interrupt_after_records
            ):
                connection.execute(
                    "UPDATE streams SET byte_offset=?,line_count=?,chain_sha256=? "
                    "WHERE role=?",
                    (offset, line_count, chain, role),
                )
                connection.commit()
                raise RuntimeError("simulated contextual indexing interruption")
    if offset != expected_bytes:
        raise ContextualIndexError(
            "%s byte count changed while streaming (%d != %d)"
            % (role, offset, expected_bytes)
        )
    connection.execute(
        "UPDATE streams SET byte_offset=?,line_count=?,chain_sha256=?,complete=1 "
        "WHERE role=?",
        (offset, line_count, chain, role),
    )
    connection.commit()
    return processed


def _index_manifest(connection, row):
    target = _canonical_target(row)
    connection.execute(
        "INSERT INTO targets (group_id,doi,accession,target_json) VALUES (?,?,?,?)",
        (
            target["group_id"],
            target["doi"],
            target["uniprot_accession"],
            canonical_json(target),
        ),
    )


def _index_parsed(connection, row):
    pointers = _candidate_pointers(row)
    if not pointers:
        return
    group_id = str(row.get("group_id") or "")
    custom_id = str(row.get("custom_id") or "")
    if not group_id or not custom_id:
        raise ContextualIndexError("candidate-bearing parsed row lacks IDs")
    if (
        connection.execute(
            "SELECT 1 FROM targets WHERE group_id=?", (group_id,)
        ).fetchone()
        is None
    ):
        raise ContextualIndexError(
            "parsed candidate has unknown group_id: %s" % group_id
        )
    connection.execute(
        "INSERT INTO candidate_groups (group_id,custom_id,candidate_count) "
        "VALUES (?,?,?)",
        (group_id, custom_id, len(pointers)),
    )
    for pointer in pointers:
        pointer["originating_group_id"] = group_id
        pointer["originating_group_role"] = "pointer_not_adjudicated_attribution"
        connection.execute(
            "INSERT INTO candidates (group_id,kind,ordinal,pointer_json) "
            "VALUES (?,?,?,?)",
            (
                group_id,
                pointer["candidate_kind"],
                pointer["candidate_ordinal"],
                canonical_json(pointer),
            ),
        )


def _index_request_metadata(connection, row):
    group_id = str(row.get("group_id") or "")
    candidate = connection.execute(
        "SELECT custom_id FROM candidate_groups WHERE group_id=?", (group_id,)
    ).fetchone()
    if candidate is None:
        return
    custom_id = str(row.get("custom_id") or "")
    if custom_id != candidate[0]:
        raise ContextualIndexError(
            "request metadata custom_id mismatch for %s" % group_id
        )
    passages = [
        cleaned
        for cleaned in (
            _clean_passage(item) for item in row.get("selected_passages") or []
        )
        if cleaned is not None
    ]
    passages.sort(
        key=lambda item: (
            int(item.get("block_index", item.get("block", 0)) or 0),
            str(item.get("source_id") or ""),
            int(item.get("char_start", 0) or 0),
        )
    )
    connection.execute(
        "INSERT INTO request_context "
        "(group_id,custom_id,source_sha256,selected_passages_json,"
        "identity_anchors_json) VALUES (?,?,?,?,?)",
        (
            group_id,
            custom_id,
            str(row.get("source_sha256") or "").lower() or None,
            canonical_json(passages),
            canonical_json(_raw_identity_anchors(row)),
        ),
    )


def _finalize_index(connection):
    missing_context = [
        row[0]
        for row in connection.execute(
            "SELECT c.group_id FROM candidate_groups c "
            "LEFT JOIN request_context r ON r.group_id=c.group_id "
            "WHERE r.group_id IS NULL ORDER BY c.group_id"
        )
    ]
    if missing_context:
        raise ContextualIndexError(
            "candidate groups lack request metadata: %s"
            % ",".join(missing_context[:10])
        )
    inconsistent = connection.execute(
        "SELECT doi FROM targets GROUP BY doi "
        "HAVING COUNT(DISTINCT json_extract(target_json,'$.paper_sha256')) != 1 "
        "OR COUNT(DISTINCT json_extract(target_json,'$.paper_path')) != 1"
    ).fetchall()
    if inconsistent:
        raise ContextualIndexError(
            "DOI targets disagree on paper identity: %s" % inconsistent[0][0]
        )
    stream_hashes = {
        role: digest
        for role, digest in connection.execute(
            "SELECT role,chain_sha256 FROM streams WHERE complete=1 ORDER BY role"
        )
    }
    if len(stream_hashes) != 3:
        raise ContextualIndexError("not all contextual input streams are complete")
    connection.execute(
        "INSERT OR REPLACE INTO metadata (key,value) VALUES (?,?)",
        ("input_chain_hashes_json", canonical_json(stream_hashes)),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata (key,value) VALUES (?,?)",
        ("inputs_sha256", _sha256_json(stream_hashes)),
    )
    connection.execute("UPDATE metadata SET value='1' WHERE key='build_complete'")
    connection.commit()


def build_context_index(
    manifest_path,
    parsed_results_path,
    request_metadata_path,
    index_path,
    resume=False,
    _interrupt_role=None,
    _interrupt_after_records=None,
):
    """Build or safely resume the streaming SQLite candidate-pointer index."""
    inputs = {
        "manifest": _path_identity(manifest_path),
        "parsed_results": _path_identity(parsed_results_path),
        "request_metadata": _path_identity(request_metadata_path),
    }
    destination = Path(index_path).expanduser().resolve()
    if destination.exists():
        if not resume:
            raise ContextualIndexError(
                "contextual index already exists; pass resume=True to verify and resume"
            )
        connection = sqlite3.connect(str(destination), timeout=60)
        connection.execute("PRAGMA foreign_keys=ON")
        _verify_runtime_identity(connection)
        _verify_stream_identities(connection, inputs)
    else:
        connection = _open_new_index(destination, inputs)
    try:
        for role, processor in (
            ("manifest", _index_manifest),
            ("parsed_results", _index_parsed),
            ("request_metadata", _index_request_metadata),
        ):
            interrupt = _interrupt_after_records if role == _interrupt_role else None
            _stream_jsonl(connection, role, processor, interrupt)
        _finalize_index(connection)
    finally:
        connection.close()
    return inspect_context_index(destination)


def _open_verified_index(index_path):
    path = Path(index_path).expanduser().resolve()
    if not path.is_file():
        raise ContextualIndexError("contextual index does not exist: %s" % path)
    connection = sqlite3.connect(str(path), timeout=60)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        metadata = _verify_runtime_identity(connection, require_complete=True)
        for role, source_path, size, mtime_ns, device, inode in connection.execute(
            "SELECT role,path,bytes,mtime_ns,device,inode FROM streams ORDER BY role"
        ):
            current = _path_identity(source_path)
            expected = {
                "path": source_path,
                "bytes": size,
                "mtime_ns": mtime_ns,
                "device": device,
                "inode": inode,
            }
            if current != expected:
                raise ContextualIndexError(
                    "contextual index source provenance mismatch: %s" % role
                )
        return connection, metadata
    except Exception:
        connection.close()
        raise


def inspect_context_index(index_path, doi=None):
    """Return deterministic JSON-serializable index provenance and counts."""
    connection, metadata = _open_verified_index(index_path)
    try:
        streams = [
            {
                "role": row[0],
                "path": row[1],
                "bytes": row[2],
                "line_count": row[3],
                "chain_sha256": row[4],
                "complete": bool(row[5]),
            }
            for row in connection.execute(
                "SELECT role,path,bytes,line_count,chain_sha256,complete "
                "FROM streams ORDER BY role"
            )
        ]
        result = {
            "index_path": str(Path(index_path).expanduser().resolve()),
            "schema_version": metadata["schema_version"],
            "build_complete": metadata["build_complete"] == "1",
            "policy_sha256": metadata["policy_sha256"],
            "code_sha256": metadata["code_sha256"],
            "inputs_sha256": metadata.get("inputs_sha256"),
            "counts": {
                "targets": connection.execute(
                    "SELECT COUNT(*) FROM targets"
                ).fetchone()[0],
                "dois": connection.execute(
                    "SELECT COUNT(DISTINCT doi) FROM targets"
                ).fetchone()[0],
                "candidate_groups": connection.execute(
                    "SELECT COUNT(*) FROM candidate_groups"
                ).fetchone()[0],
                "candidate_dois": connection.execute(
                    "SELECT COUNT(DISTINCT t.doi) FROM candidate_groups c "
                    "JOIN targets t ON t.group_id=c.group_id"
                ).fetchone()[0],
                "no_candidate_dois": connection.execute(
                    "SELECT COUNT(DISTINCT t.doi) FROM targets t "
                    "WHERE NOT EXISTS ("
                    "SELECT 1 FROM targets sibling "
                    "JOIN candidate_groups c ON c.group_id=sibling.group_id "
                    "WHERE sibling.doi=t.doi)"
                ).fetchone()[0],
                "state_candidates": connection.execute(
                    "SELECT COUNT(*) FROM candidates WHERE kind='state'"
                ).fetchone()[0],
                "event_candidates": connection.execute(
                    "SELECT COUNT(*) FROM candidates WHERE kind='event'"
                ).fetchone()[0],
            },
            "streams": streams,
        }
        if doi is not None:
            normalized = normalize_doi(doi)
            result["doi"] = normalized
            result["doi_targets"] = [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT target_json FROM targets WHERE doi=? "
                    "ORDER BY accession,group_id",
                    (normalized,),
                )
            ]
            result["doi_candidate_groups"] = [
                row[0]
                for row in connection.execute(
                    "SELECT c.group_id FROM candidate_groups c "
                    "JOIN targets t ON t.group_id=c.group_id "
                    "WHERE t.doi=? ORDER BY c.group_id",
                    (normalized,),
                )
            ]
        return result
    finally:
        connection.close()


def select_candidate_dois(index_path, limit):
    """Select the first candidate-bearing DOIs in canonical order."""
    limit = int(limit)
    if limit < 1:
        raise ValueError("candidate DOI limit must be positive")
    connection, _ = _open_verified_index(index_path)
    try:
        return [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT t.doi FROM candidate_groups c "
                "JOIN targets t ON t.group_id=c.group_id "
                "ORDER BY t.doi LIMIT ?",
                (limit,),
            )
        ]
    finally:
        connection.close()


def select_candidate_dois_hashed(index_path, limit, seed=0):
    """Select a stable hash-seeded pilot from candidate-bearing DOIs."""
    limit = int(limit)
    if limit < 1:
        raise ValueError("candidate DOI limit must be positive")
    seed = str(seed)
    connection, metadata = _open_verified_index(index_path)
    try:
        eligible = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT t.doi FROM candidate_groups c "
                "JOIN targets t ON t.group_id=c.group_id "
                "ORDER BY t.doi"
            )
        ]
    finally:
        connection.close()
    eligible_sha256 = _sha256_json(eligible)
    ranked = sorted(
        eligible,
        key=lambda doi: (
            hashlib.sha256((seed + doi).encode("utf-8")).hexdigest(),
            doi,
        ),
    )
    selected = ranked[:limit]
    selection_identity = {
        "selection_version": "candidate_doi_hashed_pilot_selection_v2",
        "seed": seed,
        "requested_limit": limit,
        "eligible_doi_count": len(eligible),
        "eligible_dois_sha256": eligible_sha256,
        "selected_dois": selected,
        "ordered_eligible_dois": ranked,
        "index_inputs_sha256": metadata["inputs_sha256"],
        "index_policy_sha256": metadata["policy_sha256"],
        "index_code_sha256": metadata["code_sha256"],
        "predictor_blind": True,
        "candidate_only": True,
        "doi_complete": True,
    }
    selection_identity["selection_sha256"] = _sha256_json(selection_identity)
    return selection_identity


def select_no_candidate_dois(index_path, limit, seed="0", reserve_count=0):
    """Select a stable DOI-complete predictor-blind recall sample."""
    limit = int(limit)
    if limit < 1:
        raise ValueError("no-candidate DOI limit must be positive")
    reserve_count = int(reserve_count)
    if reserve_count < 0:
        raise ValueError("no-candidate reserve count cannot be negative")
    seed = str(seed)
    connection, metadata = _open_verified_index(index_path)
    try:
        eligible = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT t.doi FROM targets t "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM targets sibling "
                "JOIN candidate_groups c ON c.group_id=sibling.group_id "
                "WHERE sibling.doi=t.doi) "
                "ORDER BY t.doi"
            )
        ]
    finally:
        connection.close()
    eligible_sha256 = _sha256_json(eligible)
    ranked = sorted(
        eligible,
        key=lambda doi: (
            hashlib.sha256((seed + "\0" + doi).encode("utf-8")).hexdigest(),
            doi,
        ),
    )
    selected = ranked[:limit]
    technical_reserve = ranked[limit : limit + reserve_count]
    selection_identity = {
        "selection_version": "no_candidate_doi_recall_selection_v3",
        "seed": seed,
        "requested_limit": limit,
        "technical_reserve_count": reserve_count,
        "eligible_doi_count": len(eligible),
        "eligible_dois_sha256": eligible_sha256,
        "selected_dois": selected,
        "ordered_eligible_dois": ranked,
        "technical_reserve_dois": technical_reserve,
        "index_inputs_sha256": metadata["inputs_sha256"],
        "index_policy_sha256": metadata["policy_sha256"],
        "index_code_sha256": metadata["code_sha256"],
        "predictor_blind": True,
        "doi_complete": True,
    }
    selection_identity["selection_sha256"] = _sha256_json(selection_identity)
    return selection_identity


def _is_reference_section(section):
    normalized = re.sub(r"[^a-z]+", " ", str(section or "").casefold()).strip()
    return normalized in {
        "references",
        "reference",
        "bibliography",
        "literature cited",
        "works cited",
    }


def _reference_like(text):
    value = str(text or "")
    signals = (
        bool(re.search(r"\b(?:19|20)\d{2}\b", value)),
        bool(re.search(r"\bdoi\s*:", value, re.I)),
        bool(re.search(r"\bet\s+al\.", value, re.I)),
        bool(re.match(r"\s*(?:\[\d+\]|\d+[.)])\s+", value)),
    )
    return sum(signals) >= 2


def classify_bibliography_blocks(passages):
    """Mark only explicit XML references or conservative terminal PDF lists."""
    classifications = [False] * len(passages)
    for index, passage in enumerate(passages):
        path = str(
            passage.get("element_path")
            or (passage.get("provenance") or {}).get("element_path")
            or ""
        ).casefold()
        if _is_reference_section(passage.get("section")) or re.search(
            r"/(?:ref-list|ref|bibliography)(?:\[|/)", path
        ):
            classifications[index] = True
    heading = None
    threshold = max(0, len(passages) // 2)
    for index, passage in enumerate(passages):
        text = normalize_text(str(passage.get("text") or ""))
        if index >= threshold and _is_reference_section(text) and len(text) <= 40:
            heading = index
            break
    if heading is not None:
        classifications[heading] = True
        trailing = passages[heading + 1 :]
        reference_flags = [_reference_like(item.get("text")) for item in trailing]
        if reference_flags and sum(reference_flags) >= max(
            1, len(reference_flags) // 2
        ):
            for relative, flag in enumerate(reference_flags, 1):
                if flag:
                    classifications[heading + relative] = True
    return classifications


def _xml_structural_blocks(path):
    if Path(path).suffix.lower() not in {".xml", ".nxml"}:
        return [], {}
    root = ET.parse(Path(path)).getroot()
    paths = {}
    structural = []
    counters = {"article-title": 0, "heading": 0}

    def walk(element, element_path, inherited_section, parent_tag=""):
        order = len(paths)
        paths[element_path] = order
        tag = element.tag.rsplit("}", 1)[-1].lower()
        section = inherited_section
        if tag == "sec":
            section_title = next(
                (
                    normalize_text(" ".join(child.itertext()))
                    for child in element
                    if child.tag.rsplit("}", 1)[-1].lower() == "title"
                ),
                "",
            )
            if section_title:
                section = section_title
        if tag in {"article-title", "title"}:
            parent_is_section_title = tag == "title" and parent_tag == "sec"
            if tag == "article-title" or parent_is_section_title:
                text = normalize_text(" ".join(element.itertext()))
                if text:
                    kind = "article-title" if tag == "article-title" else "heading"
                    counters[kind] += 1
                    source_id = (
                        "xml.front.title%d" % counters[kind]
                        if kind == "article-title"
                        else "xml.heading%d" % counters[kind]
                    )
                    structural.append(
                        {
                            "_document_order": order,
                            "source_id": source_id,
                            "base_source_id": source_id,
                            "section": "title" if kind == "article-title" else section,
                            "page": "unknown",
                            "block": 0,
                            "block_index": 0,
                            "sentence_start": 1,
                            "sentence_end": 1,
                            "char_start": 0,
                            "char_end": len(text),
                            "provenance": {
                                "page": "unknown",
                                "block": 0,
                                "sentence_start": 1,
                                "sentence_end": 1,
                                "char_start": 0,
                                "char_end": len(text),
                                "offset_basis": "normalized_xml_element_text",
                                "element_path": element_path,
                                "element_tag": tag,
                                "base_source_id": source_id,
                            },
                            "element_path": element_path,
                            "requires_abstention": False,
                            "text": text,
                        }
                    )
        child_counts = {}
        for child in element:
            child_tag = child.tag.rsplit("}", 1)[-1].lower()
            child_counts[child_tag] = child_counts.get(child_tag, 0) + 1
            child_path = "%s/%s[%d]" % (
                element_path,
                child_tag,
                child_counts[child_tag],
            )
            walk(child, child_path, section, tag)

    root_tag = root.tag.rsplit("}", 1)[-1].lower()
    walk(root, "/%s[1]" % root_tag, "unknown")
    return structural, paths


def _extract_contextual_text(path):
    passages = extract_text(path)
    structural, path_order = _xml_structural_blocks(path)
    if not structural:
        return passages
    ordered = []
    for index, passage in enumerate(passages):
        element_path = str(
            passage.get("element_path")
            or (passage.get("provenance") or {}).get("element_path")
            or ""
        )
        item = dict(passage)
        item["_document_order"] = path_order.get(element_path, len(path_order) + index)
        ordered.append(item)
    ordered.extend(structural)
    ordered.sort(
        key=lambda item: (
            item["_document_order"],
            0 if str(item.get("source_id") or "").startswith("xml.heading") else 1,
            str(item.get("source_id") or ""),
        )
    )
    for item in ordered:
        item.pop("_document_order", None)
    return ordered


def _source_block(passage, document_index):
    block = {
        field: passage.get(field)
        for field in PASSAGE_FIELDS
        if field in passage and field != "source_passage_id"
    }
    block["document_index"] = int(document_index)
    block["source_id"] = str(block.get("source_id") or "")
    block["base_source_id"] = str(block.get("base_source_id") or block["source_id"])
    block["section"] = str(block.get("section") or "unknown")
    block["page"] = str(block.get("page") or "unknown")
    block["block"] = int(block.get("block", block.get("block_index", 0)) or 0)
    block["block_index"] = int(block.get("block_index", block["block"]) or 0)
    return block


def _candidate_source_ids(pointers):
    ids = set()
    quotes = set()

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("source_id"), str) and value["source_id"]:
                ids.add(value["source_id"])
            if isinstance(value.get("quote"), str) and value["quote"]:
                quotes.add(value["quote"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(pointers)
    return ids, quotes


def _resolve_required_blocks(blocks, pointers, windows, anchors):
    source_map = {}
    for index, block in enumerate(blocks):
        for key in ("source_id", "base_source_id"):
            source_map.setdefault(str(block.get(key) or ""), set()).add(index)
    window_map = {}
    for window in windows:
        values = {
            str(window.get("source_id") or ""),
            str(window.get("base_source_id") or ""),
            str(window.get("source_passage_id") or ""),
        }
        for value in values:
            if value:
                window_map.setdefault(value, set()).update(
                    item for item in values if item
                )
    candidate_ids, candidate_quotes = _candidate_source_ids(pointers)
    candidate_blocks = set()
    unresolved = []
    for source_id in sorted(candidate_ids):
        aliases = set(window_map.get(source_id, ())) | {source_id}
        matches = set()
        for alias in aliases:
            matches.update(source_map.get(alias, ()))
        if not matches:
            for quote in sorted(candidate_quotes, key=lambda item: (-len(item), item)):
                matches.update(
                    index
                    for index, block in enumerate(blocks)
                    if quote in block.get("text", "")
                )
        if matches:
            candidate_blocks.update(matches)
        else:
            unresolved.append(
                {
                    "kind": "candidate_source",
                    "source_id": source_id,
                    "reason": "not_recovered",
                }
            )
    anchor_blocks = set()
    for anchor in anchors:
        source_id = str(anchor.get("source_id") or "")
        matches = set(source_map.get(source_id, ()))
        if not matches:
            quote = str(anchor.get("anchor_quote") or "")
            matches.update(
                index
                for index, block in enumerate(blocks)
                if quote and quote in block.get("text", "")
            )
        if matches:
            anchor_blocks.update(matches)
        else:
            unresolved.append(
                {
                    "kind": "identity_anchor",
                    "anchor_candidate_id": anchor.get("anchor_candidate_id"),
                    "source_id": source_id,
                    "reason": "not_recovered",
                }
            )
    neighbors = {
        neighbor
        for index in candidate_blocks | anchor_blocks
        for neighbor in range(max(0, index - 2), min(len(blocks), index + 3))
    }
    return candidate_blocks, anchor_blocks, neighbors, unresolved


def _block_is_heading(block):
    text = str(block.get("text") or "").strip()
    return bool(text) and len(text) <= 160 and not re.search(r"[.!?]\s*$", text)


def _selection_plan(blocks, pointers, windows, anchors):
    candidate, anchor, neighbors, unresolved = _resolve_required_blocks(
        blocks, pointers, windows, anchors
    )
    required = candidate | anchor | neighbors
    captions_headings = set()
    for index in candidate | anchor:
        if str(blocks[index].get("section") or "") in {
            "figure_or_table_caption",
            "table",
        }:
            captions_headings.add(index)
        for prior in range(index - 1, max(-1, index - 4), -1):
            if _block_is_heading(blocks[prior]):
                captions_headings.add(prior)
                break
    required.update(captions_headings)
    front = {
        index
        for index, block in enumerate(blocks)
        if index < 3
        or str(block.get("section") or "").casefold()
        in {"title", "abstract", "front", "front matter"}
    }
    relevant_sections = {
        str(blocks[index].get("section") or "")
        for index in candidate | anchor
        if str(blocks[index].get("section") or "") not in {"", "unknown"}
    }
    section_blocks = {
        index
        for index, block in enumerate(blocks)
        if str(block.get("section") or "") in relevant_sections
    }
    ordered = []
    reasons = {}

    def add(indices, reason):
        for index in sorted(indices):
            reasons.setdefault(index, []).append(reason)
            if index not in ordered:
                ordered.append(index)

    add(required, "required_candidate_anchor_or_neighbor")
    add(front, "front_matter")
    add(section_blocks, "complete_relevant_section")
    add(range(len(blocks)), "document_order_fill")
    return ordered, reasons, required, unresolved


def _encoding():
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("contextual dossiers require tiktoken") from exc
    return tiktoken.get_encoding("o200k_base")


def _token_count(value, encoding):
    return len(encoding.encode(canonical_json(value)))


def _assert_no_forbidden_keys(value, path="$"):
    violations = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if any(part in normalized for part in PREDICTOR_FORBIDDEN_PARTS):
                violations.append("%s.%s" % (path, key))
            violations.extend(_assert_no_forbidden_keys(child, "%s.%s" % (path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _assert_no_forbidden_keys(child, "%s[%d]" % (path, index))
            )
    return violations


def assert_predictor_blind(dossier):
    """Reject structured predictor/acceptance leakage from a dossier."""
    violations = _assert_no_forbidden_keys(dossier)
    if violations:
        raise ContextualIndexError(
            "predictor-blind audit failed: %s" % ",".join(sorted(violations))
        )
    return {
        "passed": True,
        "scope": "structured_keys_excluding_literal_paper_text",
        "forbidden_feature_families": list(PREDICTOR_FORBIDDEN_PARTS),
        "violations": [],
    }


def _dossier_hash_payload(dossier):
    value = json.loads(canonical_json(dossier))
    value.pop("token_audit", None)
    hashes = value.get("hashes") or {}
    hashes.pop("dossier_content_sha256", None)
    return value


def _finish_dossier(dossier, token_cap, encoding):
    dossier["predictor_blind_audit"] = assert_predictor_blind(dossier)
    dossier.setdefault("hashes", {})["dossier_content_sha256"] = _sha256_json(
        _dossier_hash_payload(dossier)
    )
    payload = json.loads(canonical_json(dossier))
    payload.pop("token_audit", None)
    dossier["token_audit"] = {
        "encoding": "o200k_base",
        "token_cap": token_cap,
        "payload_token_count": _token_count(payload, encoding),
        "serialized_dossier_token_count": 0,
        "within_cap": False,
        "before_model_framing": True,
    }
    previous = None
    for _ in range(8):
        count = _token_count(dossier, encoding)
        dossier["token_audit"]["serialized_dossier_token_count"] = count
        dossier["token_audit"]["within_cap"] = count <= token_cap
        state = (count, dossier["token_audit"]["within_cap"])
        if state == previous:
            break
        previous = state
    return dossier


def _load_doi_records(connection, doi, allow_no_candidates=False):
    targets = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT target_json FROM targets WHERE doi=? ORDER BY accession,group_id",
            (doi,),
        )
    ]
    if not targets:
        raise ContextualIndexError("DOI is not present in contextual index: %s" % doi)
    candidate_rows = connection.execute(
        "SELECT c.group_id,c.pointer_json FROM candidates c "
        "JOIN targets t ON t.group_id=c.group_id WHERE t.doi=? "
        "ORDER BY c.group_id,CASE c.kind WHEN 'state' THEN 0 ELSE 1 END,c.ordinal",
        (doi,),
    ).fetchall()
    if not candidate_rows and not allow_no_candidates:
        raise ContextualIndexError(
            "DOI has no indexed state/event candidates; "
            "set allow_no_candidates=True explicitly: %s" % doi
        )
    pointers_by_group = {}
    for group_id, pointer_json in candidate_rows:
        pointers_by_group.setdefault(group_id, []).append(json.loads(pointer_json))
    context = {
        row[0]: {
            "selected_passages": json.loads(row[1]),
            "identity_anchors": json.loads(row[2]),
            "source_sha256": row[3],
        }
        for row in connection.execute(
            "SELECT r.group_id,r.selected_passages_json,r.identity_anchors_json,"
            "r.source_sha256 FROM request_context r "
            "JOIN targets t ON t.group_id=r.group_id WHERE t.doi=? "
            "ORDER BY r.group_id",
            (doi,),
        )
    }
    return targets, pointers_by_group, context


def _paper_identity(targets):
    identities = {
        (
            target["paper_path"],
            target["paper_sha256"],
            target.get("paper_bytes"),
        )
        for target in targets
    }
    if len(identities) != 1:
        raise ContextualIndexError("DOI target cards disagree on paper identity")
    path, expected, size = next(iter(identities))
    actual = sha256_file(path)
    if actual != expected:
        raise ContextualIndexError(
            "checksum mismatch for contextual paper source: %s" % path
        )
    current_size = Path(path).stat().st_size
    if size is not None and int(size) != current_size:
        raise ContextualIndexError(
            "byte-size mismatch for contextual paper source: %s" % path
        )
    return {
        "doi": targets[0]["doi"],
        "paper_path": path,
        "paper_sha256": actual,
        "paper_bytes": current_size,
        "checksum_verified": True,
    }


def _target_cards(targets):
    group_ids = [target["group_id"] for target in targets]
    cards = []
    for target in targets:
        card = dict(target)
        card.pop("paper_path", None)
        card.pop("paper_sha256", None)
        card.pop("paper_bytes", None)
        card["same_paper_competitor_group_ids"] = [
            group_id for group_id in group_ids if group_id != target["group_id"]
        ]
        cards.append(card)
    return cards


def _base_dossier(
    paper,
    targets,
    pointers_by_group,
    anchors,
    blocks,
    bibliography,
    unresolved,
    index_metadata,
    selection_mode,
    omitted,
    dossier_mode,
):
    bundles = [
        {
            "originating_group_id": group_id,
            "originating_group_role": "pointer_not_adjudicated_attribution",
            "bundle_status": "candidate_hints_not_truth",
            "candidates": pointers_by_group[group_id],
        }
        for group_id in sorted(pointers_by_group)
    ]
    dossier = {
        "dossier_schema_version": DOSSIER_SCHEMA_VERSION,
        "paper_identity": paper,
        "canonical_targets": _target_cards(targets),
        "candidate_bundles": bundles,
        "raw_identity_anchor_candidates": anchors,
        "source_blocks": blocks,
        "bibliography_audit": {
            "policy": POLICY["bibliography_policy"],
            "omitted_bibliography_only_blocks": bibliography,
            "in_text_citation_markers_retained": True,
        },
        "context_audit": {
            "selection_mode": selection_mode,
            "noncontiguous_blocks_joined": False,
            "required_context_unresolved": unresolved,
            "omitted_nonbibliography_blocks": omitted,
            "required_candidate_or_anchor_omitted": bool(unresolved),
        },
        "hashes": {
            "paper_sha256": paper["paper_sha256"],
            "index_inputs_sha256": index_metadata["inputs_sha256"],
            "index_code_sha256": index_metadata["code_sha256"],
            "index_policy_sha256": index_metadata["policy_sha256"],
            "dossier_code_sha256": _code_identity()["sha256"],
            "dossier_policy_sha256": _policy_sha256(),
        },
    }
    if dossier_mode == "fresh_no_candidate_recall_sample":
        dossier["dossier_mode"] = dossier_mode
        dossier["context_audit"].update(
            {
                "previous_candidate_outputs_used": False,
                "request_context_used": False,
                "fresh_no_candidate_recall": True,
            }
        )
    return dossier


def build_doi_dossier(
    index_path,
    doi,
    token_cap=DEFAULT_TOKEN_CAP,
    allow_no_candidates=False,
    _minimum_token_cap=MIN_TOKEN_CAP,
):
    """Build one deterministic, checksum-verified paper-level dossier."""
    token_cap = int(token_cap)
    if not _minimum_token_cap <= token_cap <= MAX_TOKEN_CAP:
        raise ValueError(
            "token_cap must be between %d and %d" % (_minimum_token_cap, MAX_TOKEN_CAP)
        )
    normalized = normalize_doi(doi)
    connection, index_metadata = _open_verified_index(index_path)
    try:
        targets, pointers_by_group, context = _load_doi_records(
            connection,
            normalized,
            allow_no_candidates=allow_no_candidates,
        )
    finally:
        connection.close()
    paper = _paper_identity(targets)
    source_hashes = {
        item["source_sha256"] for item in context.values() if item.get("source_sha256")
    }
    if source_hashes and source_hashes != {paper["paper_sha256"]}:
        raise ContextualIndexError(
            "request metadata source checksum disagrees with manifest paper"
        )
    extracted = _extract_contextual_text(paper["paper_path"])
    bibliography_flags = classify_bibliography_blocks(extracted)
    bibliography = []
    clean_blocks = []
    for original_index, (passage, bibliography_only) in enumerate(
        zip(extracted, bibliography_flags)
    ):
        block = _source_block(passage, original_index)
        if bibliography_only:
            bibliography.append(
                {
                    "source_id": block["source_id"],
                    "document_index": original_index,
                    "section": block["section"],
                    "page": block["page"],
                    "text_sha256": hashlib.sha256(
                        str(block.get("text") or "").encode("utf-8")
                    ).hexdigest(),
                    "reason": "bibliography_only",
                }
            )
        else:
            clean_blocks.append(block)
    if not clean_blocks:
        raise ContextualIndexError(
            "checksum-valid paper extraction yielded no non-bibliography "
            "source blocks for DOI %s" % normalized,
            code="no_nonbibliography_source_blocks",
            details={
                "doi": normalized,
                "paper_sha256": paper["paper_sha256"],
                "extracted_block_count": len(extracted),
                "bibliography_block_count": len(bibliography),
            },
        )
    pointers = [
        pointer
        for group_id in sorted(pointers_by_group)
        for pointer in pointers_by_group[group_id]
    ]
    windows = [
        passage
        for group_id in sorted(context)
        for passage in context[group_id]["selected_passages"]
    ]
    anchors_by_id = {
        anchor["anchor_candidate_id"]: anchor
        for group_id in sorted(context)
        for anchor in context[group_id]["identity_anchors"]
    }
    anchors = [anchors_by_id[key] for key in sorted(anchors_by_id)]
    dossier_mode = (
        "candidate_context_reextraction"
        if pointers_by_group
        else "fresh_no_candidate_recall_sample"
    )
    ordered, reasons, required, unresolved = _selection_plan(
        clean_blocks, pointers, windows, anchors
    )
    encoding = _encoding()
    full_blocks = []
    for index, block in enumerate(clean_blocks):
        item = dict(block)
        item["selection_reasons"] = reasons.get(index, ["full_context"])
        full_blocks.append(item)
    dossier = _base_dossier(
        paper,
        targets,
        pointers_by_group,
        anchors,
        full_blocks,
        bibliography,
        unresolved,
        index_metadata,
        "full_nonbibliography_text",
        [],
        dossier_mode,
    )
    dossier = _finish_dossier(dossier, token_cap, encoding)
    if dossier["token_audit"]["within_cap"]:
        dossier["context_audit"]["cleaned_nonbibliography_block_count"] = len(
            clean_blocks
        )
        dossier["context_audit"]["included_block_count"] = len(full_blocks)
        dossier["context_audit"]["cap_exceeded_due_to_required_context"] = False
        return _finish_dossier(dossier, token_cap, encoding)

    selected = list(sorted(required))
    selected_set = set(selected)

    def materialize(indices):
        values = []
        for chosen in sorted(indices):
            item = dict(clean_blocks[chosen])
            item["selection_reasons"] = reasons[chosen]
            values.append(item)
        return values

    def omission_records(indices):
        return [
            {
                "source_id": clean_blocks[item]["source_id"],
                "document_index": clean_blocks[item]["document_index"],
                "reason": "token_budget",
            }
            for item in range(len(clean_blocks))
            if item not in indices
        ]

    baseline = _base_dossier(
        paper,
        targets,
        pointers_by_group,
        anchors,
        materialize(selected_set),
        bibliography,
        unresolved,
        index_metadata,
        "adaptive_over_cap",
        omission_records(selected_set),
        dossier_mode,
    )
    baseline = _finish_dossier(baseline, token_cap, encoding)
    estimated_tokens = baseline["token_audit"]["serialized_dossier_token_count"]
    # Per-block canonical token costs permit a linear selection pass. Exact
    # whole-dossier counts are still recomputed below before export.
    for index in ordered:
        if index in selected_set:
            continue
        item = dict(clean_blocks[index])
        item["selection_reasons"] = reasons[index]
        cost = _token_count(item, encoding) + 4
        if estimated_tokens + cost <= token_cap - 128:
            selected.append(index)
            selected_set.add(index)
            estimated_tokens += cost
    selected_blocks = []
    for index in sorted(selected_set):
        item = dict(clean_blocks[index])
        item["selection_reasons"] = reasons[index]
        selected_blocks.append(item)
    omitted = [
        {
            "source_id": clean_blocks[index]["source_id"],
            "document_index": clean_blocks[index]["document_index"],
            "reason": "token_budget",
        }
        for index in range(len(clean_blocks))
        if index not in selected_set
    ]
    dossier = _base_dossier(
        paper,
        targets,
        pointers_by_group,
        anchors,
        selected_blocks,
        bibliography,
        unresolved,
        index_metadata,
        "adaptive_over_cap",
        omitted,
        dossier_mode,
    )
    dossier["context_audit"].update(
        {
            "cleaned_nonbibliography_block_count": len(clean_blocks),
            "included_block_count": len(selected_blocks),
            "cap_exceeded_due_to_required_context": False,
        }
    )
    dossier = _finish_dossier(dossier, token_cap, encoding)
    while not dossier["token_audit"]["within_cap"] and any(
        index not in required for index in selected
    ):
        removable = next(index for index in reversed(selected) if index not in required)
        selected.remove(removable)
        selected_set.remove(removable)
        selected_blocks = materialize(selected_set)
        omitted = omission_records(selected_set)
        dossier = _base_dossier(
            paper,
            targets,
            pointers_by_group,
            anchors,
            selected_blocks,
            bibliography,
            unresolved,
            index_metadata,
            "adaptive_over_cap",
            omitted,
            dossier_mode,
        )
        dossier["context_audit"].update(
            {
                "cleaned_nonbibliography_block_count": len(clean_blocks),
                "included_block_count": len(selected_blocks),
                "cap_exceeded_due_to_required_context": False,
            }
        )
        dossier = _finish_dossier(dossier, token_cap, encoding)
    if not dossier["token_audit"]["within_cap"]:
        dossier["context_audit"]["cap_exceeded_due_to_required_context"] = True
        dossier = _finish_dossier(dossier, token_cap, encoding)
    return dossier


def export_dossiers(
    index_path,
    dois,
    output_dir,
    token_cap=DEFAULT_TOKEN_CAP,
    allow_no_candidates=False,
    selection_audit=None,
    requested_count=None,
    skip_empty_context=False,
    backfill_unextractable=False,
):
    """Export selected DOI dossiers without overwriting existing artifacts."""
    normalized = []
    seen = set()
    for value in dois:
        doi = normalize_doi(value)
        if doi not in seen:
            normalized.append(doi)
            seen.add(doi)
    if not normalized:
        raise ValueError("at least one DOI is required")
    if requested_count is None:
        requested_count = len(normalized)
    requested_count = int(requested_count)
    backfill_unextractable = bool(backfill_unextractable or skip_empty_context)
    if requested_count < 1 or (
        requested_count > len(normalized) and not backfill_unextractable
    ):
        raise ValueError(
            "requested dossier count must be between 1 and the DOI pool size"
        )
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ContextualIndexError(
            "dossier output directory already exists: %s" % destination
        )
    destination.mkdir(parents=True)
    exported = []
    excluded = []
    try:
        for doi in normalized:
            try:
                dossier = build_doi_dossier(
                    index_path,
                    doi,
                    token_cap=token_cap,
                    allow_no_candidates=allow_no_candidates,
                )
            except ContextualIndexError as exc:
                if not (
                    backfill_unextractable
                    and exc.code == "no_nonbibliography_source_blocks"
                ):
                    raise
                exclusion = {
                    "doi": doi,
                    "reason": exc.code,
                    "paper_sha256": exc.details.get("paper_sha256"),
                }
                exclusion["exclusion_sha256"] = _sha256_json(exclusion)
                excluded.append(exclusion)
                continue
            if not dossier.get("source_blocks"):
                # Defensive invariant: build_doi_dossier must raise before this.
                raise ContextualIndexError(
                    "dossier unexpectedly contains no source blocks for DOI %s" % doi,
                    code="no_nonbibliography_source_blocks",
                    details={"doi": doi},
                )
            filename = (
                "doi_" + hashlib.sha256(doi.encode("utf-8")).hexdigest()[:20] + ".json"
            )
            path = destination / filename
            _atomic_json(path, dossier)
            exported.append(
                {
                    "doi": doi,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "tokens": dossier["token_audit"]["serialized_dossier_token_count"],
                }
            )
            if len(exported) == requested_count:
                break
        eligibility_exhausted = len(exported) != requested_count
        if eligibility_exhausted and not backfill_unextractable:
            raise ContextualIndexError(
                "only %d of %d requested dossiers had usable source context"
                % (len(exported), requested_count)
            )
        manifest = {
            "schema_version": DOSSIER_SCHEMA_VERSION,
            "index_path": str(Path(index_path).expanduser().resolve()),
            "token_cap": int(token_cap),
            "dossiers": exported,
            "requested_dossier_count": requested_count,
            "exported_dossier_count": len(exported),
            "eligibility_exhausted": eligibility_exhausted,
        }
        if excluded:
            manifest["technical_exclusions"] = excluded
            manifest["technical_replacement_count"] = sum(
                item["doi"] not in set((selection_audit or {}).get("selected_dois", []))
                for item in exported
            )
        if allow_no_candidates:
            manifest["allow_no_candidates"] = True
        if selection_audit is not None:
            audit = json.loads(canonical_json(selection_audit))
            sampled_dois = list(audit.get("selected_dois") or [])
            exported_dois = [item["doi"] for item in exported]
            excluded_dois = [item["doi"] for item in excluded]
            backfill_dois = [
                doi for doi in exported_dois if doi not in set(sampled_dois)
            ]
            audit.update(
                {
                    "sampled_dois": sampled_dois,
                    "sampled_count": len(sampled_dois),
                    "sampled_dois_sha256": _sha256_json(sampled_dois),
                    "excluded_unextractable": excluded,
                    "excluded_unextractable_count": len(excluded),
                    "excluded_unextractable_dois_sha256": _sha256_json(excluded_dois),
                    "backfill_dois": backfill_dois,
                    "backfill_count": len(backfill_dois),
                    "backfill_dois_sha256": _sha256_json(backfill_dois),
                    "final_exported_dois": exported_dois,
                    "final_exported_count": len(exported_dois),
                    "final_exported_dois_sha256": _sha256_json(exported_dois),
                    "eligibility_exhausted": eligibility_exhausted,
                }
            )
            backfill_identity = {
                key: audit[key]
                for key in (
                    "sampled_dois_sha256",
                    "excluded_unextractable",
                    "backfill_dois_sha256",
                    "final_exported_dois_sha256",
                    "eligibility_exhausted",
                )
            }
            audit["export_backfill_audit_sha256"] = _sha256_json(backfill_identity)
            manifest["selection_audit"] = audit
        _atomic_json(destination / "export_manifest.json", manifest)
        return manifest
    except Exception:
        if destination.exists():
            for path in destination.iterdir():
                if path.is_file():
                    path.unlink()
            if not any(destination.iterdir()):
                destination.rmdir()
        raise


__all__ = [
    "DEFAULT_TOKEN_CAP",
    "DOSSIER_SCHEMA_VERSION",
    "INDEX_SCHEMA_VERSION",
    "MAX_TOKEN_CAP",
    "MIN_TOKEN_CAP",
    "ContextualIndexError",
    "assert_predictor_blind",
    "build_context_index",
    "build_doi_dossier",
    "classify_bibliography_blocks",
    "export_dossiers",
    "inspect_context_index",
    "sanitize_candidate_pointer",
    "select_candidate_dois",
    "select_candidate_dois_hashed",
    "select_no_candidate_dois",
]
