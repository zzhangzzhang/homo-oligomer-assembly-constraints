"""Durable prepare, synchronous execute, and offline build lifecycle for v6."""

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
from pathlib import Path

from .contextual_extractor import (
    DEFAULT_REASONING_EFFORT,
    prepare_contextual_request,
)
from .contextual_prompt import (
    CONTEXTUAL_PROMPT_VERSION,
    ContextualDossierRefusal,
    contextual_system_prompt_sha256,
    validate_contextual_dossier,
)
from .contextual_quote_audit import (
    audit_contextual_quotes,
    render_quote_audit_examples,
    summarize_contextual_quote_records,
)
from .contextual_schema import (
    CONTEXTUAL_SCHEMA_VERSION,
    contextual_schema_sha256,
)
from .contextual_validation import (
    ContextualValidationError,
    derive_contextual_tiers_tolerant,
    supplied_candidate_keys,
    tier_counts,
    validate_contextual_schema,
    verify_contextual_result_tolerant,
)
from .doi import normalize_doi
from .lifecycle import _object_dict
from .hashing import sha256_file
from .serialization import canonical_json
from .api_credentials import load_api_key

CONTEXTUAL_PILOT_LIFECYCLE_VERSION = "contextual_pilot_lifecycle_v1"
CONTEXTUAL_TOLERANT_REBUILD_VERSION = "contextual_tolerant_rebuild_v1"
DEFAULT_CONTEXTUAL_PILOT_MODEL = "gpt-5.6-sol"
DEFAULT_CONTEXTUAL_MAX_COMPLETION_TOKENS = 20_000
DEFAULT_CONTEXTUAL_EXPECTED_OUTPUT_TOKENS = 4_000
DEFAULT_CONTEXTUAL_WORKERS = 4
DEFAULT_CONTEXTUAL_BUILD_WORKERS = 16
DEFAULT_CONTEXTUAL_BUILD_RECORD_TIMEOUT_SECONDS = 300
CONTEXTUAL_BUILD_IN_FLIGHT_MULTIPLIER = 4
CONTEXTUAL_PARALLEL_BUILD_VERSION = "contextual_parallel_build_v1"
DEFAULT_CONTEXTUAL_MAX_RETRIES = 3
DEFAULT_CONTEXTUAL_RETRY_BACKOFF_SECONDS = 1.0
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConflictError",
    "InternalServerError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "TimeoutError",
}


class ContextualPilotError(RuntimeError):
    """Raised when a pilot artifact or lifecycle invariant is unsafe."""


def _validate_build_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("contextual build workers must be a positive integer")
    return workers


def _parallel_build_identity(workers):
    return {
        "version": CONTEXTUAL_PARALLEL_BUILD_VERSION,
        "executor": "ProcessPoolExecutor",
        "workers": workers,
        "max_in_flight": workers * CONTEXTUAL_BUILD_IN_FLIGHT_MULTIPLIER,
        "merge_order": "custom_id",
        "timing": {
            "clock": "monotonic",
            "per_record_timeout_seconds": (
                DEFAULT_CONTEXTUAL_BUILD_RECORD_TIMEOUT_SECONDS
            ),
        },
    }


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value):
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _atomic_bytes(path, content):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    _atomic_bytes(
        path,
        (canonical_json(value, indent=2) + "\n").encode("utf-8"),
    )


def _write_build_progress(
    path,
    phase,
    processed,
    total,
    workers,
    started_at,
    detail="",
):
    elapsed = max(0.0, time.monotonic() - started_at)
    rate = processed / elapsed if processed and elapsed else 0.0
    remaining = max(0, total - processed)
    eta = remaining / rate if rate else None
    _atomic_json(
        path,
        {
            "phase": phase,
            "detail": str(detail),
            "processed": int(processed),
            "total": int(total),
            "percent": (100.0 * processed / total if total else 0.0),
            "workers": int(workers),
            "elapsed_seconds": elapsed,
            "papers_per_second": rate,
            "eta_seconds": eta,
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        },
    )


def _atomic_jsonl(path, rows, sort_key=None):
    ordered = list(rows)
    if sort_key is not None:
        ordered.sort(key=sort_key)
    content = "".join(canonical_json(row) + "\n" for row in ordered)
    _atomic_bytes(path, content.encode("utf-8"))


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ContextualPilotError("cannot read JSON artifact %s: %s" % (path, error))
    return value


def _read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                raise ContextualPilotError(
                    "JSONL must be nonblank and newline terminated: %s:%d"
                    % (path, line_number)
                )
            try:
                row = json.loads(raw)
            except ValueError as error:
                raise ContextualPilotError(
                    "invalid JSONL at %s:%d: %s" % (path, line_number, error)
                )
            if not isinstance(row, dict):
                raise ContextualPilotError(
                    "JSONL row is not an object at %s:%d" % (path, line_number)
                )
            rows.append(row)
    return rows


def _safe_response_name(custom_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(custom_id or "")):
        raise ContextualPilotError("unsafe contextual custom_id: %r" % custom_id)
    return str(custom_id) + ".json"


def _code_identity():
    package = Path(__file__).resolve().parent
    names = (
        "contextual.py",
        "contextual_extractor.py",
        "contextual_pilot.py",
        "contextual_prompt.py",
        "contextual_quote_audit.py",
        "contextual_schema.py",
        "contextual_validation.py",
        "doi.py",
        "serialization.py",
        "validation.py",
    )
    files = {
        name: sha256_file(package / name)
        for name in names
    }
    return {"files": files, "sha256": _sha256_json(files)}


def _token_estimates(request):
    body_json = canonical_json(request["body"])
    request_json = canonical_json(request)
    try:
        import tiktoken
    except ImportError:
        return {
            "method": "ceil_utf8_characters_divided_by_4",
            "encoding": "",
            "request_characters": len(request_json),
            "body_characters": len(body_json),
            "estimated_request_tokens": (len(request_json) + 3) // 4,
            "estimated_body_tokens": (len(body_json) + 3) // 4,
        }
    encoding = tiktoken.get_encoding("o200k_base")
    return {
        "method": "tiktoken_canonical_json",
        "encoding": encoding.name,
        "request_characters": len(request_json),
        "body_characters": len(body_json),
        "estimated_request_tokens": len(encoding.encode(request_json)),
        "estimated_body_tokens": len(encoding.encode(body_json)),
    }


def _pricing_record(
    input_price_per_million,
    output_price_per_million,
    cached_input_price_per_million,
):
    pricing = {
        "input_per_million_tokens_usd": float(input_price_per_million),
        "output_per_million_tokens_usd": float(output_price_per_million),
        "cached_input_per_million_tokens_usd": float(
            cached_input_price_per_million
        ),
    }
    if any(value < 0 for value in pricing.values()):
        raise ValueError("contextual pilot prices must be non-negative")
    return pricing


def _manifest_dossier_path(export_dir, entry):
    relative = Path(str(entry.get("path") or "")).name
    if not relative:
        raise ContextualPilotError("export manifest dossier path is empty")
    path = export_dir / relative
    if not path.is_file():
        raise ContextualPilotError(
            "exported dossier is missing beside its manifest: %s" % path
        )
    return path


def _dossier_refusal_reason_code(error):
    message = str(error).strip()
    known = {
        "required candidate or identity context is unresolved": (
            "required_context_unresolved"
        ),
        "required candidate or identity context was omitted": (
            "required_context_omitted"
        ),
        "required context exceeds the dossier token cap": (
            "required_context_exceeds_token_cap"
        ),
    }
    if message in known:
        return known[message]
    slug = re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")
    return "dossier_refusal_%s" % (slug[:120] or "unspecified")


def _exclusion_fields(exclude_unpreparable_dossiers, exclusions):
    records = list(exclusions)
    return {
        "exclude_unpreparable_dossiers": bool(exclude_unpreparable_dossiers),
        "excluded_dossier_count": len(records),
        "excluded_dossiers_sha256": _sha256_json(records),
        "excluded_dossiers": records,
    }


def _validate_exclusion_audit(value):
    if "excluded_dossiers" not in value:
        return None
    exclusions = value.get("excluded_dossiers")
    if not isinstance(exclusions, list):
        raise ContextualPilotError("contextual dossier exclusion list is malformed")
    if value.get("excluded_dossier_count") != len(exclusions):
        raise ContextualPilotError("contextual dossier exclusion count mismatch")
    if value.get("excluded_dossiers_sha256") != _sha256_json(exclusions):
        raise ContextualPilotError("contextual dossier exclusion checksum mismatch")
    if exclusions and value.get("exclude_unpreparable_dossiers") is not True:
        raise ContextualPilotError("contextual dossier exclusions lack opt-in mode")
    required = {
        "doi",
        "path",
        "file_sha256",
        "error_type",
        "message",
        "reason_code",
    }
    seen_dois = set()
    for record in exclusions:
        if not isinstance(record, dict) or not required <= set(record):
            raise ContextualPilotError("contextual dossier exclusion is malformed")
        doi = normalize_doi(record["doi"])
        if (
            doi != record["doi"]
            or doi in seen_dois
            or not re.fullmatch(r"[0-9a-f]{64}", str(record["file_sha256"]))
            or record["error_type"] != "ContextualDossierRefusal"
            or not str(record["path"])
            or not str(record["message"])
            or not str(record["reason_code"])
        ):
            raise ContextualPilotError(
                "contextual dossier exclusion identity is malformed"
            )
        seen_dois.add(doi)
    return exclusions


def _load_exports(export_dirs, exclude_unpreparable_dossiers=False):
    exports = []
    dossiers = []
    exclusions = []
    seen_dois = set()
    for export_index, value in enumerate(export_dirs):
        export_dir = Path(value).expanduser().resolve()
        manifest_path = export_dir / "export_manifest.json"
        if not manifest_path.is_file():
            raise ContextualPilotError(
                "dossier export lacks export_manifest.json: %s" % export_dir
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest = _read_json(manifest_path)
        entries = manifest.get("dossiers")
        if not isinstance(entries, list) or not entries:
            raise ContextualPilotError("dossier export manifest is empty or malformed")
        export_record = {
            "export_index": export_index,
            "source_directory": str(export_dir),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_bytes(manifest_bytes),
            "manifest_copy": "cohort_manifests/export-%03d.json"
            % export_index,
            "dossier_count": len(entries),
            "accepted_dossier_count": 0,
            "excluded_dossier_count": 0,
        }
        exports.append((export_record, manifest_bytes, manifest))
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ContextualPilotError("export manifest dossier entry is malformed")
            dossier_path = _manifest_dossier_path(export_dir, entry)
            actual_file_hash = sha256_file(dossier_path)
            expected_file_hash = str(entry.get("sha256") or "").lower()
            if actual_file_hash != expected_file_hash:
                raise ContextualPilotError(
                    "exported dossier file checksum mismatch: %s" % dossier_path
                )
            dossier = _read_json(dossier_path)
            try:
                validation = validate_contextual_dossier(dossier)
            except ContextualDossierRefusal as error:
                if not exclude_unpreparable_dossiers:
                    raise
                paper = dossier.get("paper_identity")
                paper_doi = paper.get("doi") if isinstance(paper, dict) else None
                doi = normalize_doi(paper_doi or entry.get("doi"))
                if paper_doi and normalize_doi(entry.get("doi")) != doi:
                    raise ContextualPilotError(
                        "export manifest/dossier DOI mismatch: %s" % dossier_path
                    )
                if doi in seen_dois:
                    raise ContextualPilotError(
                        "duplicate normalized DOI across dossier exports: %s" % doi
                    )
                seen_dois.add(doi)
                exclusions.append(
                    {
                        "doi": doi,
                        "path": str(dossier_path),
                        "file_sha256": actual_file_hash,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "reason_code": _dossier_refusal_reason_code(error),
                        "export_index": export_index,
                        "entry_index": entry_index,
                    }
                )
                export_record["excluded_dossier_count"] += 1
                continue
            doi = normalize_doi(dossier["paper_identity"]["doi"])
            if normalize_doi(entry.get("doi")) != doi:
                raise ContextualPilotError(
                    "export manifest/dossier DOI mismatch: %s" % dossier_path
                )
            if doi in seen_dois:
                raise ContextualPilotError(
                    "duplicate normalized DOI across dossier exports: %s" % doi
                )
            seen_dois.add(doi)
            recorded_tokens = entry.get("tokens")
            if recorded_tokens != validation["serialized_dossier_token_count"]:
                raise ContextualPilotError(
                    "export manifest/dossier token audit mismatch: %s" % dossier_path
                )
            paper = dossier["paper_identity"]
            paper_path = Path(paper["paper_path"]).expanduser().resolve()
            if not paper_path.is_file() or sha256_file(paper_path) != str(
                paper["paper_sha256"]
            ).lower():
                raise ContextualPilotError(
                    "dossier paper file checksum mismatch: %s" % paper_path
                )
            if (
                paper.get("paper_bytes") is not None
                and int(paper["paper_bytes"]) != paper_path.stat().st_size
            ):
                raise ContextualPilotError(
                    "dossier paper byte-size mismatch: %s" % paper_path
                )
            cohort = (
                "candidate"
                if supplied_candidate_keys(dossier)
                else "no_candidate"
            )
            if cohort == "no_candidate" and dossier.get("candidate_bundles"):
                raise ContextualPilotError(
                    "no-candidate dossier has nonempty candidate bundles"
                )
            export_record["accepted_dossier_count"] += 1
            dossiers.append(
                {
                    "doi": doi,
                    "cohort": cohort,
                    "export_index": export_index,
                    "entry_index": entry_index,
                    "source_path": dossier_path,
                    "source_file_sha256": actual_file_hash,
                    "dossier": dossier,
                    "dossier_content_sha256": validation["dossier_sha256"],
                    "paper_sha256": str(paper["paper_sha256"]).lower(),
                    "candidate_count": validation["candidate_count"],
                    "target_count": len(validation["target_group_ids"]),
                    "source_block_count": validation["source_block_count"],
                    "dossier_tokens": validation[
                        "serialized_dossier_token_count"
                    ],
                }
            )
    dossiers.sort(key=lambda item: item["doi"])
    exclusions.sort(key=lambda item: (item["doi"], item["path"]))
    return exports, dossiers, exclusions


def _cost_report(
    metadata,
    pricing,
    expected_output_tokens_per_request,
    max_completion_tokens,
):
    expected_output_tokens_per_request = int(expected_output_tokens_per_request)
    if expected_output_tokens_per_request < 0:
        raise ValueError("expected output tokens must be non-negative")
    if expected_output_tokens_per_request > max_completion_tokens:
        raise ValueError(
            "expected output tokens cannot exceed max completion tokens"
        )
    input_tokens = sum(
        item["token_estimate"]["estimated_body_tokens"] for item in metadata
    )
    request_tokens = sum(
        item["token_estimate"]["estimated_request_tokens"] for item in metadata
    )
    expected_output = len(metadata) * expected_output_tokens_per_request
    maximum_output = len(metadata) * max_completion_tokens
    input_cost = input_tokens * pricing["input_per_million_tokens_usd"] / 1e6
    cached_cost = (
        input_tokens * pricing["cached_input_per_million_tokens_usd"] / 1e6
    )
    expected_output_cost = (
        expected_output * pricing["output_per_million_tokens_usd"] / 1e6
    )
    maximum_output_cost = (
        maximum_output * pricing["output_per_million_tokens_usd"] / 1e6
    )
    return {
        "pricing": pricing,
        "input": {
            "estimated_body_tokens": input_tokens,
            "estimated_request_envelope_tokens": request_tokens,
            "estimated_uncached_cost_usd": input_cost,
            "estimated_fully_cached_cost_usd": cached_cost,
        },
        "output": {
            "expected_tokens_per_request": expected_output_tokens_per_request,
            "expected_tokens": expected_output,
            "maximum_tokens_per_request": max_completion_tokens,
            "maximum_tokens": maximum_output,
            "expected_cost_usd": expected_output_cost,
            "maximum_cost_usd": maximum_output_cost,
        },
        "totals": {
            "expected_uncached_usd": input_cost + expected_output_cost,
            "expected_fully_cached_usd": cached_cost + expected_output_cost,
            "maximum_uncached_usd": input_cost + maximum_output_cost,
            "maximum_fully_cached_usd": cached_cost + maximum_output_cost,
        },
    }


def prepare_contextual_pilot(
    dossier_export_dirs,
    run_dir,
    model=DEFAULT_CONTEXTUAL_PILOT_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    max_completion_tokens=DEFAULT_CONTEXTUAL_MAX_COMPLETION_TOKENS,
    input_price_per_million=0.0,
    output_price_per_million=0.0,
    cached_input_price_per_million=0.0,
    expected_output_tokens_per_request=DEFAULT_CONTEXTUAL_EXPECTED_OUTPUT_TOKENS,
    exclude_unpreparable_dossiers=False,
):
    """Verify exported dossiers and atomically publish an immutable pilot run."""
    if not dossier_export_dirs:
        raise ValueError("at least one dossier export directory is required")
    model = str(model or "").strip()
    if not model:
        raise ValueError("an exact contextual pilot model is required")
    max_completion_tokens = int(max_completion_tokens)
    if max_completion_tokens < 1:
        raise ValueError("max_completion_tokens must be positive")
    pricing = _pricing_record(
        input_price_per_million,
        output_price_per_million,
        cached_input_price_per_million,
    )
    exports, dossier_records, exclusions = _load_exports(
        dossier_export_dirs,
        exclude_unpreparable_dossiers=exclude_unpreparable_dossiers,
    )
    source_dossier_count = sum(
        record["dossier_count"] for record, _, _ in exports
    )
    if source_dossier_count != len(dossier_records) + len(exclusions):
        raise ContextualPilotError("contextual source dossier accounting mismatch")
    if not dossier_records:
        raise ContextualPilotError(
            "all exported contextual dossiers were excluded as unpreparable"
        )
    exclusion_fields = _exclusion_fields(
        exclude_unpreparable_dossiers,
        exclusions,
    )
    code = _code_identity()
    prepared = []
    for item in dossier_records:
        request, metadata = prepare_contextual_request(
            item["dossier"],
            model,
            max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        metadata.update(
            {
                "cohort": item["cohort"],
                "dossier_file": "dossiers/%s.json" % request["custom_id"],
                "dossier_file_sha256": item["source_file_sha256"],
                "paper_sha256": item["paper_sha256"],
                "candidate_count": item["candidate_count"],
                "target_count": item["target_count"],
                "source_block_count": item["source_block_count"],
                "dossier_serialized_tokens": item["dossier_tokens"],
                "token_estimate": _token_estimates(request),
                "export_index": item["export_index"],
            }
        )
        prepared.append((request, metadata, item))
    prepared.sort(key=lambda value: value[0]["custom_id"])
    custom_ids = [value[0]["custom_id"] for value in prepared]
    if len(custom_ids) != len(set(custom_ids)):
        raise ContextualPilotError("dossier exports produce duplicate custom IDs")

    preparation_identity = {
        "lifecycle_version": CONTEXTUAL_PILOT_LIFECYCLE_VERSION,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
        "prompt_version": CONTEXTUAL_PROMPT_VERSION,
        "prompt_sha256": contextual_system_prompt_sha256(),
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "schema_sha256": contextual_schema_sha256(),
        "code_sha256": code["sha256"],
        "export_manifest_sha256": [
            record["source_manifest_sha256"] for record, _, _ in exports
        ],
        "dossier_file_sha256": [
            item["source_file_sha256"] for item in dossier_records
        ],
        "pricing": pricing,
        "expected_output_tokens_per_request": int(
            expected_output_tokens_per_request
        ),
        **exclusion_fields,
    }
    destination = Path(run_dir).expanduser().resolve()
    staging = destination.parent / (".%s.preparing" % destination.name)
    if destination.exists():
        raise ContextualPilotError(
            "contextual pilot run directory already exists: %s" % destination
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        state_path = staging / "preparation_identity.json"
        recorded = _read_json(state_path) if state_path.is_file() else None
        if recorded != preparation_identity:
            raise ContextualPilotError(
                "existing contextual pilot staging identity mismatch: %s" % staging
            )
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        _atomic_json(staging / "preparation_identity.json", preparation_identity)
        cohort_manifest_exports = []
        for export_record, manifest_bytes, manifest in exports:
            copy_path = staging / export_record["manifest_copy"]
            _atomic_bytes(copy_path, manifest_bytes)
            copied = dict(export_record)
            copied["manifest_schema_version"] = manifest.get("schema_version")
            copied["selection_audit"] = manifest.get("selection_audit")
            cohort_manifest_exports.append(copied)
        requests = []
        metadata_rows = []
        cohort_dossiers = []
        for request, metadata, item in prepared:
            dossier_path = staging / metadata["dossier_file"]
            _atomic_bytes(dossier_path, item["source_path"].read_bytes())
            if sha256_file(dossier_path) != item["source_file_sha256"]:
                raise ContextualPilotError("copied dossier checksum mismatch")
            requests.append(request)
            metadata_rows.append(metadata)
            cohort_dossiers.append(
                {
                    "custom_id": request["custom_id"],
                    "doi": item["doi"],
                    "cohort": item["cohort"],
                    "dossier_file": metadata["dossier_file"],
                    "dossier_file_sha256": item["source_file_sha256"],
                    "dossier_content_sha256": item["dossier_content_sha256"],
                    "paper_sha256": item["paper_sha256"],
                    "candidate_count": item["candidate_count"],
                    "target_count": item["target_count"],
                }
            )
        _atomic_jsonl(
            staging / "requests.jsonl",
            requests,
            sort_key=lambda row: row["custom_id"],
        )
        _atomic_jsonl(
            staging / "request_metadata.jsonl",
            metadata_rows,
            sort_key=lambda row: row["custom_id"],
        )
        cohort_manifest = {
            "lifecycle_version": CONTEXTUAL_PILOT_LIFECYCLE_VERSION,
            "exports": cohort_manifest_exports,
            "dossiers": sorted(
                cohort_dossiers, key=lambda row: row["custom_id"]
            ),
            "cohort_counts": dict(
                sorted(Counter(row["cohort"] for row in cohort_dossiers).items())
            ),
            "doi_count": len(cohort_dossiers),
            "source_dossier_count": source_dossier_count,
            **exclusion_fields,
        }
        _atomic_json(staging / "cohort_manifest.json", cohort_manifest)
        cost_report = _cost_report(
            metadata_rows,
            pricing,
            int(expected_output_tokens_per_request),
            max_completion_tokens,
        )
        _atomic_json(staging / "cost_report.json", cost_report)
        immutable_paths = [
            "preparation_identity.json",
            "requests.jsonl",
            "request_metadata.jsonl",
            "cohort_manifest.json",
            "cost_report.json",
        ]
        immutable_paths.extend(
            record["manifest_copy"] for record in cohort_manifest_exports
        )
        immutable_paths.extend(
            row["dossier_file"] for row in cohort_dossiers
        )
        artifact_hashes = {
            path: sha256_file(staging / path) for path in sorted(immutable_paths)
        }
        environment = {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        config = {
            "lifecycle_version": CONTEXTUAL_PILOT_LIFECYCLE_VERSION,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
            "request_count": len(requests),
            "source_dossier_count": source_dossier_count,
            "cohort_counts": cohort_manifest["cohort_counts"],
            "prompt_version": CONTEXTUAL_PROMPT_VERSION,
            "prompt_sha256": contextual_system_prompt_sha256(),
            "schema_version": CONTEXTUAL_SCHEMA_VERSION,
            "schema_sha256": contextual_schema_sha256(),
            "code_files_sha256": code["files"],
            "code_sha256": code["sha256"],
            "environment": environment,
            "environment_sha256": _sha256_json(environment),
            "pricing": pricing,
            "cost_report": cost_report,
            "prepared_artifacts_sha256": artifact_hashes,
            **exclusion_fields,
        }
        _atomic_json(staging / "run_config.json", config)
        integrity = {
            "lifecycle_version": CONTEXTUAL_PILOT_LIFECYCLE_VERSION,
            "run_config_sha256": sha256_file(staging / "run_config.json"),
        }
        _atomic_json(staging / "pilot_integrity.json", integrity)
        os.replace(staging, destination)
        return config
    except Exception:
        raise


def _load_pilot(run_dir):
    root = Path(run_dir).expanduser().resolve()
    config = _read_json(root / "run_config.json")
    integrity = _read_json(root / "pilot_integrity.json")
    if config.get("lifecycle_version") != CONTEXTUAL_PILOT_LIFECYCLE_VERSION:
        raise ContextualPilotError("unsupported contextual pilot lifecycle")
    if sha256_file(root / "run_config.json") != integrity.get(
        "run_config_sha256"
    ):
        raise ContextualPilotError("contextual run_config.json checksum mismatch")
    for relative, expected in config.get("prepared_artifacts_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ContextualPilotError(
                "immutable contextual pilot artifact mismatch: %s" % path
            )
    if config.get("schema_sha256") != contextual_schema_sha256():
        raise ContextualPilotError("active contextual schema checksum changed")
    if config.get("prompt_sha256") != contextual_system_prompt_sha256():
        raise ContextualPilotError("active contextual prompt checksum changed")
    requests = _read_jsonl(root / "requests.jsonl")
    metadata_rows = _read_jsonl(root / "request_metadata.jsonl")
    requests_by_id = {}
    for request in requests:
        custom_id = str(request.get("custom_id") or "")
        if custom_id in requests_by_id:
            raise ContextualPilotError("duplicate prepared contextual custom_id")
        requests_by_id[custom_id] = request
    metadata = {}
    for row in metadata_rows:
        custom_id = str(row.get("custom_id") or "")
        if custom_id in metadata:
            raise ContextualPilotError("duplicate contextual request metadata")
        metadata[custom_id] = row
    if (
        set(requests_by_id) != set(metadata)
        or len(requests_by_id) != config.get("request_count")
    ):
        raise ContextualPilotError("prepared contextual request accounting mismatch")
    exclusions = _validate_exclusion_audit(config)
    if exclusions is not None:
        if config.get("source_dossier_count") != len(requests_by_id) + len(
            exclusions
        ):
            raise ContextualPilotError(
                "prepared contextual source/exclusion accounting mismatch"
            )
        cohort_manifest = _read_json(root / "cohort_manifest.json")
        for field in (
            "exclude_unpreparable_dossiers",
            "excluded_dossier_count",
            "excluded_dossiers_sha256",
            "excluded_dossiers",
            "source_dossier_count",
        ):
            if cohort_manifest.get(field) != config.get(field):
                raise ContextualPilotError(
                    "contextual cohort/config exclusion audit mismatch"
                )
        cohort_dossiers = cohort_manifest.get("dossiers")
        if (
            not isinstance(cohort_dossiers, list)
            or any(not isinstance(row, dict) for row in cohort_dossiers)
            or {row.get("custom_id") for row in cohort_dossiers}
            != set(requests_by_id)
            or cohort_manifest.get("doi_count") != len(requests_by_id)
        ):
            raise ContextualPilotError(
                "contextual cohort/request accounting mismatch"
            )
        excluded_dois = {record["doi"] for record in exclusions}
        requested_dois = {str(row.get("doi") or "") for row in metadata.values()}
        if excluded_dois & requested_dois:
            raise ContextualPilotError(
                "excluded contextual dossier was accidentally requested"
            )
    dossiers = {}
    for custom_id, row in metadata.items():
        dossier_path = root / row["dossier_file"]
        dossier = _read_json(dossier_path)
        if (
            sha256_file(dossier_path) != row["dossier_file_sha256"]
            or dossier["hashes"]["dossier_content_sha256"]
            != row["dossier_content_sha256"]
        ):
            raise ContextualPilotError(
                "contextual dossier metadata checksum mismatch: %s" % custom_id
            )
        dossiers[custom_id] = dossier
    return root, config, requests_by_id, metadata, dossiers


def _openai_client(api_key):
    try:
        OpenAI = importlib.import_module("openai").OpenAI
    except ImportError as error:
        raise RuntimeError(
            "context-pilot-run requires the optional openai package"
        ) from error
    return OpenAI(api_key=api_key, max_retries=0)


def _status_code(error):
    value = getattr(error, "status_code", None)
    if value is None:
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_transient_error(error):
    status = _status_code(error)
    if status in TRANSIENT_STATUS_CODES:
        return True
    return any(
        cls.__name__ in TRANSIENT_ERROR_NAMES
        for cls in type(error).__mro__
    )


def _sanitize_error_message(error, api_key=""):
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        "[REDACTED_API_KEY]",
        message,
    )
    return message[:4000]


def _scan_terminal_records(root, expected):
    raw = {}
    errors = {}
    for directory, destination in (
        (root / "raw_responses", raw),
        (root / "api_errors", errors),
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            record = _read_json(path)
            custom_id = str(record.get("custom_id") or "")
            if custom_id not in expected:
                raise ContextualPilotError(
                    "unexpected contextual terminal custom_id: %s" % custom_id
                )
            if path.name != _safe_response_name(custom_id):
                raise ContextualPilotError(
                    "contextual terminal filename/custom_id mismatch: %s" % path
                )
            if custom_id in destination:
                raise ContextualPilotError(
                    "duplicate contextual terminal record: %s" % custom_id
                )
            destination[custom_id] = record
    overlap = set(raw) & set(errors)
    if overlap:
        raise ContextualPilotError(
            "custom_id has both raw response and API error: %s"
            % sorted(overlap)[0]
        )
    return raw, errors


def _execute_one(
    custom_id,
    request,
    client,
    max_retries,
    retry_backoff_seconds,
    sleep,
    api_key,
):
    attempts = 0
    while True:
        attempts += 1
        try:
            response = client.chat.completions.create(**request["body"])
            return {
                "kind": "response",
                "record": {
                    "custom_id": custom_id,
                    "response": _object_dict(response),
                },
                "attempts": attempts,
            }
        except Exception as error:
            transient = _is_transient_error(error)
            if transient and attempts <= max_retries:
                sleep(retry_backoff_seconds * (2 ** (attempts - 1)))
                continue
            return {
                "kind": "api_error",
                "record": {
                    "custom_id": custom_id,
                    "error_type": type(error).__name__,
                    "status_code": _status_code(error),
                    "transient": transient,
                    "attempts": attempts,
                    "error": _sanitize_error_message(error, api_key),
                },
                "attempts": attempts,
            }


def _execution_progress(progress, total):
    if callable(progress):
        return progress
    mode = progress or "none"
    if mode == "auto":
        mode = "log" if not sys.stderr.isatty() else "log"
    if mode == "none":
        return lambda _value: None
    if mode != "log":
        raise ValueError("contextual execution progress must be auto, log, or none")
    return lambda value: print(
        "context pilot progress: terminal=%d/%d responses=%d api_errors=%d"
        % (
            value["terminal"],
            total,
            value["responses"],
            value["api_errors"],
        ),
        file=sys.stderr,
        flush=True,
    )


def execute_contextual_pilot(
    run_dir,
    client=None,
    api_key_path=None,
    workers=DEFAULT_CONTEXTUAL_WORKERS,
    max_retries=DEFAULT_CONTEXTUAL_MAX_RETRIES,
    retry_backoff_seconds=DEFAULT_CONTEXTUAL_RETRY_BACKOFF_SECONDS,
    limit=None,
    progress="none",
    _sleep=time.sleep,
):
    """Explicit paid action: execute pending requests with durable per-ID writes."""
    workers = int(workers)
    max_retries = int(max_retries)
    retry_backoff_seconds = float(retry_backoff_seconds)
    if workers < 1 or max_retries < 0 or retry_backoff_seconds < 0:
        raise ValueError("workers/retries/backoff are outside allowed ranges")
    if limit is not None and int(limit) < 1:
        raise ValueError("contextual execution limit must be positive")
    root, config, requests, _, _ = _load_pilot(run_dir)
    raw_dir = root / "raw_responses"
    error_dir = root / "api_errors"
    raw_dir.mkdir(exist_ok=True)
    error_dir.mkdir(exist_ok=True)
    raw, errors = _scan_terminal_records(root, set(requests))
    pending = sorted(set(requests) - set(raw) - set(errors))
    if limit is not None:
        pending = pending[: int(limit)]
    api_key = ""
    if client is None:
        api_key = load_api_key(config_path=api_key_path)
        client = _openai_client(api_key)
    report_progress = _execution_progress(progress, len(requests))
    lock = threading.Lock()
    executed_responses = 0
    executed_errors = 0
    retry_attempts = 0

    def persist(outcome):
        nonlocal executed_responses, executed_errors, retry_attempts
        record = outcome["record"]
        custom_id = record["custom_id"]
        directory = raw_dir if outcome["kind"] == "response" else error_dir
        _atomic_json(directory / _safe_response_name(custom_id), record)
        with lock:
            retry_attempts += max(0, outcome["attempts"] - 1)
            if outcome["kind"] == "response":
                raw[custom_id] = record
                executed_responses += 1
            else:
                errors[custom_id] = record
                executed_errors += 1
            report_progress(
                {
                    "terminal": len(raw) + len(errors),
                    "responses": len(raw),
                    "api_errors": len(errors),
                }
            )

    active_workers = min(workers, max(1, len(pending)))
    if pending:
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            pending_iterator = iter(pending)
            futures = {}
            for _ in range(min(len(pending), active_workers * 2)):
                custom_id = next(pending_iterator)
                futures[
                    executor.submit(
                        _execute_one,
                        custom_id,
                        requests[custom_id],
                        client,
                        max_retries,
                        retry_backoff_seconds,
                        _sleep,
                        api_key,
                    )
                ] = custom_id
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    persist(future.result())
                    try:
                        custom_id = next(pending_iterator)
                    except StopIteration:
                        continue
                    futures[
                        executor.submit(
                            _execute_one,
                            custom_id,
                            requests[custom_id],
                            client,
                            max_retries,
                            retry_backoff_seconds,
                            _sleep,
                            api_key,
                        )
                    ] = custom_id
    report = {
        "lifecycle_version": CONTEXTUAL_PILOT_LIFECYCLE_VERSION,
        "expected": len(requests),
        "terminal": len(raw) + len(errors),
        "successful_responses": len(raw),
        "api_errors": len(errors),
        "pending": len(requests) - len(raw) - len(errors),
        "executed_responses_now": executed_responses,
        "executed_api_errors_now": executed_errors,
        "skipped_terminal": (
            len(raw) + len(errors) - executed_responses - executed_errors
        ),
        "retry_attempts_now": retry_attempts,
        "worker_count": workers,
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "limit": int(limit) if limit is not None else None,
    }
    _atomic_json(root / "synchronous_execution_report.json", report)
    return report


def _response_content(record):
    response = record.get("response")
    if not isinstance(response, dict):
        raise ValueError("raw SDK record has no response object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("raw SDK response has no choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("raw SDK response has no string message content")
    return content


def _usage(response):
    usage = response.get("usage") if isinstance(response, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = int(
        usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    )
    output_tokens = int(
        usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    )
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    cached_tokens = int(details.get("cached_tokens", 0) or 0)
    cached_tokens = min(max(0, cached_tokens), max(0, input_tokens))
    return input_tokens, output_tokens, cached_tokens


def _technical_row(custom_id, metadata, error):
    return {
        "custom_id": custom_id,
        "doi": metadata[custom_id]["doi"],
        "cohort": metadata[custom_id]["cohort"],
        "technical_error": True,
        "error_phase": error["phase"],
        "error_code": error["error_code"],
        "error_type": error["error_type"],
        "error": error["error"],
    }


def _error_row(custom_id, phase, error, details=None):
    if isinstance(error, ContextualValidationError):
        code = error.code
        details = error.details
    else:
        code = type(error).__name__
    row = {
        "custom_id": custom_id,
        "phase": phase,
        "error_code": code,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if details:
        row["details"] = details
    return row


def _process_contextual_paper(custom_id, meta, dossier, raw_record, schema_sha256):
    """Parse, verify, tier, and audit one paper without side effects."""
    base = {
        "custom_id": custom_id,
        "doi": meta["doi"],
        "cohort": meta["cohort"],
    }
    try:
        content = _response_content(raw_record)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("model content is not a JSON object")
        validate_contextual_schema(parsed, schema_sha256)
    except Exception as error:
        return {
            "parsed": None,
            "verified": None,
            "tiered": None,
            "error": _error_row(custom_id, "schema", error),
            "audit_records": [],
        }
    parsed_row = dict(base, result=parsed)
    try:
        verified = verify_contextual_result_tolerant(
            parsed,
            dossier,
            expected_schema_sha256=schema_sha256,
        )
        tiered = derive_contextual_tiers_tolerant(verified)
    except Exception as error:
        return {
            "parsed": parsed_row,
            "verified": None,
            "tiered": None,
            "error": _error_row(custom_id, "verification", error),
            "audit_records": [],
        }
    verified_row = dict(base, result=verified)
    tiered_row = dict(base, technical_error=False, result=tiered)
    audit_records = audit_contextual_quotes(
        [verified_row],
        {custom_id: dossier},
    )["records"]
    return {
        "parsed": parsed_row,
        "verified": verified_row,
        "tiered": tiered_row,
        "error": None,
        "audit_records": audit_records,
    }


def _worker_error_outcome(custom_id, error, phase="worker"):
    return {
        "parsed": None,
        "verified": None,
        "tiered": None,
        "error": _error_row(custom_id, phase, error),
        "audit_records": [],
    }


def _terminate_process_pool(executor):
    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:
        terminate_workers()
        return
    processes = list(getattr(executor, "_processes", {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def _bounded_paper_outcomes(
    tasks,
    workers,
    timeout_seconds,
    worker_function=None,
):
    """Yield worker outcomes in input order with bounded process work."""
    if worker_function is None:
        worker_function = _process_contextual_paper
    source = iter(tasks)
    retries = deque()
    queued = deque()
    executor = None
    force_shutdown = False

    def next_task():
        if retries:
            return retries.popleft()
        return next(source)

    try:
        while True:
            if executor is None:
                executor = ProcessPoolExecutor(max_workers=workers)
            while len(queued) < (
                workers * CONTEXTUAL_BUILD_IN_FLIGHT_MULTIPLIER
            ):
                try:
                    task = next_task()
                except StopIteration:
                    break
                future = executor.submit(worker_function, *task)
                queued.append((task, future, time.monotonic()))
            if not queued:
                break
            task, future, submitted_at = queued.popleft()
            custom_id = task[0]
            remaining = max(
                0.0,
                timeout_seconds - (time.monotonic() - submitted_at),
            )
            try:
                outcome = future.result(timeout=remaining)
            except FutureTimeoutError:
                timeout = TimeoutError(
                    "contextual paper worker exceeded %s seconds"
                    % timeout_seconds
                )
                outcome = _worker_error_outcome(
                    custom_id,
                    timeout,
                    phase="worker_timeout",
                )
                retries.extend(item[0] for item in queued)
                queued.clear()
                _terminate_process_pool(executor)
                executor = None
            except Exception as error:
                outcome = _worker_error_outcome(custom_id, error)
            yield custom_id, outcome
    except BaseException:
        force_shutdown = True
        raise
    finally:
        if executor is not None:
            if force_shutdown:
                _terminate_process_pool(executor)
            else:
                executor.shutdown(wait=True, cancel_futures=True)


def _tiered_items(result):
    for field in (
        "state_claims",
        "event_claims",
        "excluded_candidates",
        "unassigned_candidates",
    ):
        for item in result.get(field) or []:
            yield field, item


def _cohort_template(request_count=0):
    return {
        "requests": request_count,
        "raw_responses": 0,
        "api_errors": 0,
        "schema_errors": 0,
        "verification_errors": 0,
        "missing_responses": 0,
        "schema_valid": 0,
        "verified": 0,
        "technical_errors": 0,
        "candidate_references_expected": 0,
        "candidate_references_verified": 0,
        "new_state_claims": 0,
        "new_event_claims": 0,
        "recovered_papers": 0,
        "tier_counts": {},
    }


def build_contextual_pilot(
    run_dir,
    output_dir=None,
    rebuild_identity=None,
    workers=DEFAULT_CONTEXTUAL_BUILD_WORKERS,
):
    """Build deterministic tolerant verification artifacts without API calls."""
    workers = _validate_build_workers(workers)
    progress_path = Path(run_dir).resolve() / "contextual_build_progress.json"
    build_started_at = time.monotonic()
    _write_build_progress(
        progress_path,
        "loading",
        0,
        0,
        workers,
        build_started_at,
        "verifying immutable run inputs",
    )
    root, config, requests, metadata, dossiers = _load_pilot(run_dir)
    output_root = Path(output_dir).resolve() if output_dir else root
    output_root.mkdir(parents=True, exist_ok=True)
    raw, api_errors = _scan_terminal_records(root, set(requests))
    total_requests = len(requests)
    processed_requests = 0
    progress_interval = max(1, min(100, total_requests))
    _write_build_progress(
        progress_path,
        "verifying",
        0,
        total_requests,
        workers,
        build_started_at,
        "parsing, quote verification, tiering, and quote audit",
    )

    def mark_processed(force=False):
        nonlocal processed_requests
        processed_requests += 1
        if (
            force
            or processed_requests == total_requests
            or processed_requests % progress_interval == 0
        ):
            _write_build_progress(
                progress_path,
                "verifying",
                processed_requests,
                total_requests,
                workers,
                build_started_at,
                "parsing, quote verification, tiering, and quote audit",
            )

    parsed_rows = []
    verified_rows = []
    tiered_rows = []
    validation_errors = []
    quote_audit_records = []
    aggregate_tiers = Counter()
    attribution_counts = Counter()
    quote_counts = Counter()
    finish_reasons = Counter()
    model_snapshots = Counter()
    usage_totals = Counter()
    candidate_expected = 0
    candidate_verified = 0
    candidate_accounted_once = 0
    candidate_missing = 0
    candidate_duplicate = 0
    candidate_unverified = 0
    candidate_unknown = 0
    new_state_claims = 0
    new_event_claims = 0
    no_candidate_recovered_papers = 0
    paper_usability = Counter()
    component_error_counts = Counter()
    claim_error_counts = Counter()
    component_warning_counts = Counter()
    event_direction_counts = Counter()
    event_regime_counts = Counter()
    event_condition_counts = Counter()
    evidence_error_papers = set()
    whole_response_error_count = 0
    cohort_breakdown = {
        cohort: _cohort_template(count)
        for cohort, count in sorted(config["cohort_counts"].items())
    }
    for breakdown in cohort_breakdown.values():
        breakdown.update(
            {
                "fully_clean": 0,
                "partially_usable": 0,
                "wholly_unusable": 0,
                "component_errors": 0,
                "component_warnings": 0,
                "candidate_references_accounted_once": 0,
                "candidate_references_missing": 0,
                "candidate_references_duplicate": 0,
                "candidate_references_unverified": 0,
            }
        )

    def paper_tasks():
        for task_custom_id in sorted(requests):
            if task_custom_id in api_errors or task_custom_id not in raw:
                continue
            yield (
                task_custom_id,
                metadata[task_custom_id],
                dossiers[task_custom_id],
                raw[task_custom_id],
                config["schema_sha256"],
            )

    worker_outcomes = iter(
        _bounded_paper_outcomes(
            paper_tasks(),
            workers,
            DEFAULT_CONTEXTUAL_BUILD_RECORD_TIMEOUT_SECONDS,
        )
    )
    for custom_id in sorted(requests):
        meta = metadata[custom_id]
        dossier = dossiers[custom_id]
        cohort = meta["cohort"]
        expected_candidates = len(supplied_candidate_keys(dossier))
        candidate_expected += expected_candidates
        cohort_breakdown[cohort]["candidate_references_expected"] += (
            expected_candidates
        )
        if custom_id in api_errors:
            record = api_errors[custom_id]
            cohort_breakdown[cohort]["api_errors"] += 1
            error = {
                "custom_id": custom_id,
                "phase": "api",
                "error_code": "api_error",
                "error_type": record.get("error_type", "APIError"),
                "error": record.get("error", "API request failed"),
                "details": {
                    "status_code": record.get("status_code"),
                    "transient": record.get("transient"),
                    "attempts": record.get("attempts"),
                },
            }
            validation_errors.append(error)
            tiered_rows.append(_technical_row(custom_id, metadata, error))
            cohort_breakdown[cohort]["technical_errors"] += 1
            whole_response_error_count += 1
            mark_processed()
            continue
        if custom_id not in raw:
            error = {
                "custom_id": custom_id,
                "phase": "response",
                "error_code": "missing_response",
                "error_type": "MissingResponse",
                "error": "no raw response or API-error record exists",
            }
            validation_errors.append(error)
            tiered_rows.append(_technical_row(custom_id, metadata, error))
            cohort_breakdown[cohort]["technical_errors"] += 1
            cohort_breakdown[cohort]["missing_responses"] += 1
            whole_response_error_count += 1
            mark_processed()
            continue
        cohort_breakdown[cohort]["raw_responses"] += 1
        response = raw[custom_id].get("response")
        input_tokens, output_tokens, cached_tokens = _usage(response)
        usage_totals["input_tokens"] += input_tokens
        usage_totals["output_tokens"] += output_tokens
        usage_totals["cached_input_tokens"] += cached_tokens
        if isinstance(response, dict):
            model_snapshots[str(response.get("model") or "unknown")] += 1
            choices = response.get("choices") or []
            finish_reason = (
                choices[0].get("finish_reason", "unknown")
                if choices and isinstance(choices[0], dict)
                else "unknown"
            )
            finish_reasons[str(finish_reason or "unknown")] += 1
        outcome_custom_id, outcome = next(worker_outcomes)
        if outcome_custom_id != custom_id:
            raise ContextualPilotError(
                "parallel contextual build order mismatch: %s != %s"
                % (outcome_custom_id, custom_id)
            )
        if outcome["parsed"] is not None:
            parsed_rows.append(outcome["parsed"])
            cohort_breakdown[cohort]["schema_valid"] += 1
        if outcome["error"] is not None:
            row = outcome["error"]
            validation_errors.append(row)
            tiered_rows.append(_technical_row(custom_id, metadata, row))
            cohort_breakdown[cohort]["technical_errors"] += 1
            if row["phase"] == "schema":
                cohort_breakdown[cohort]["schema_errors"] += 1
            else:
                cohort_breakdown[cohort]["verification_errors"] += 1
            whole_response_error_count += 1
            mark_processed()
            continue
        verified_rows.append(outcome["verified"])
        tiered_rows.append(outcome["tiered"])
        quote_audit_records.extend(outcome["audit_records"])
        verified = outcome["verified"]["result"]
        tiered = outcome["tiered"]["result"]
        cohort_breakdown[cohort]["verified"] += 1
        summary = verified["verification_summary"]
        usability = summary["paper_usability"]
        paper_usability[usability] += 1
        cohort_breakdown[cohort][usability] += 1
        cohort_breakdown[cohort]["component_errors"] += summary[
            "component_error_count"
        ]
        cohort_breakdown[cohort]["component_warnings"] += summary[
            "component_warning_count"
        ]
        accounting = verified["candidate_accounting"]
        candidate_verified += accounting["verified_exactly_once_count"]
        candidate_accounted_once += accounting["accounted_once_count"]
        candidate_missing += accounting["missing_count"]
        candidate_duplicate += accounting["duplicate_count"]
        candidate_unverified += accounting["unverified_count"]
        candidate_unknown += accounting["unknown_count"]
        cohort_breakdown[cohort]["candidate_references_verified"] += (
            accounting["verified_exactly_once_count"]
        )
        cohort_breakdown[cohort][
            "candidate_references_accounted_once"
        ] += accounting["accounted_once_count"]
        cohort_breakdown[cohort]["candidate_references_missing"] += accounting[
            "missing_count"
        ]
        cohort_breakdown[cohort]["candidate_references_duplicate"] += accounting[
            "duplicate_count"
        ]
        cohort_breakdown[cohort]["candidate_references_unverified"] += accounting[
            "unverified_count"
        ]
        for diagnostic in verified["component_diagnostics"]:
            row = {
                "custom_id": custom_id,
                "phase": "component_verification",
                "severity": diagnostic["severity"],
                "error_code": diagnostic["error_code"],
                "error_type": diagnostic["error_type"],
                "error": diagnostic["error"],
                "details": diagnostic["details"],
                "component_path": diagnostic["component_path"],
            }
            validation_errors.append(row)
            if diagnostic["severity"] == "error":
                component_error_counts[diagnostic["error_code"]] += 1
                if diagnostic["component_path"].startswith(
                    ("$.state_claims[", "$.event_claims[")
                ):
                    claim_error_counts[diagnostic["error_code"]] += 1
                if diagnostic["error_code"].startswith("evidence_"):
                    evidence_error_papers.add(custom_id)
            else:
                component_warning_counts[diagnostic["error_code"]] += 1
        if summary["component_error_count"]:
            cohort_breakdown[cohort]["verification_errors"] += 1
        verification_summary = verified["verification_summary"]
        quote_counts["literal_exact"] += verification_summary[
            "literal_exact_quotes"
        ]
        quote_counts["allowlisted_normalized_exact"] += verification_summary[
            "allowlisted_normalized_exact_quotes"
        ]
        quote_counts["invalid"] += verification_summary[
            "evidence_references_invalid"
        ]
        quote_counts["total"] += verification_summary[
            "evidence_references_total"
        ]
        current_tiers = tier_counts(tiered)
        aggregate_tiers.update(current_tiers)
        cohort_tiers = Counter(cohort_breakdown[cohort]["tier_counts"])
        cohort_tiers.update(current_tiers)
        cohort_breakdown[cohort]["tier_counts"] = dict(sorted(cohort_tiers.items()))
        for field in ("state_claims", "event_claims"):
            for claim in tiered[field]:
                attribution_counts[
                    claim["target_attribution"]["target_attribution_status"]
                ] += 1
                if not claim.get("candidate_references"):
                    if field == "state_claims":
                        new_state_claims += 1
                        cohort_breakdown[cohort]["new_state_claims"] += 1
                    else:
                        new_event_claims += 1
                        cohort_breakdown[cohort]["new_event_claims"] += 1
                if field == "event_claims":
                    event_direction_counts.update(claim["direction_processes"])
                    event_regime_counts[claim["dynamic_regime"]] += 1
                    event_condition_counts[
                        "condition_linked"
                        if claim["condition_linked"]
                        else "not_condition_linked"
                    ] += 1
        recovered = bool(tiered["state_claims"] or tiered["event_claims"])
        if recovered:
            cohort_breakdown[cohort]["recovered_papers"] += 1
            if cohort == "no_candidate":
                no_candidate_recovered_papers += 1
        mark_processed()

    for cohort in cohort_breakdown:
        cohort_breakdown[cohort]["tier_counts"] = dict(
            sorted(cohort_breakdown[cohort]["tier_counts"].items())
        )
    _write_build_progress(
        progress_path,
        "writing",
        total_requests,
        total_requests,
        workers,
        build_started_at,
        "writing deterministic artifacts and report",
    )
    _atomic_jsonl(
        output_root / "parsed_results.jsonl",
        parsed_rows,
        sort_key=lambda row: row["custom_id"],
    )
    _atomic_jsonl(
        output_root / "verified_results.jsonl",
        verified_rows,
        sort_key=lambda row: row["custom_id"],
    )
    _atomic_jsonl(
        output_root / "tiered_results.jsonl",
        tiered_rows,
        sort_key=lambda row: row["custom_id"],
    )
    _atomic_jsonl(
        output_root / "validation_errors.jsonl",
        validation_errors,
        sort_key=lambda row: row["custom_id"],
    )
    quote_audit = summarize_contextual_quote_records(quote_audit_records)
    _atomic_json(output_root / "quote_audit.json", quote_audit)
    _atomic_bytes(
        output_root / "quote_audit_examples.md",
        render_quote_audit_examples(quote_audit).encode("utf-8"),
    )
    pricing = config["pricing"]
    uncached_tokens = (
        usage_totals["input_tokens"] - usage_totals["cached_input_tokens"]
    )
    actual_cost = (
        uncached_tokens * pricing["input_per_million_tokens_usd"]
        + usage_totals["cached_input_tokens"]
        * pricing["cached_input_per_million_tokens_usd"]
        + usage_totals["output_tokens"]
        * pricing["output_per_million_tokens_usd"]
    ) / 1e6
    schema_errors = sum(
        row["phase"] == "schema" for row in validation_errors
    )
    verification_errors = sum(
        breakdown["verification_errors"]
        for breakdown in cohort_breakdown.values()
    )
    quote_valid = quote_counts["total"] - quote_counts["invalid"]
    build_code = _code_identity()
    report = {
        "lifecycle_version": CONTEXTUAL_PILOT_LIFECYCLE_VERSION,
        "verification_mode": "component_tolerant_v1",
        "parallel_build": _parallel_build_identity(workers),
        "rebuild_format_version": (
            CONTEXTUAL_TOLERANT_REBUILD_VERSION if rebuild_identity else None
        ),
        "rebuild_identity": rebuild_identity,
        "rebuild_relative_path": (
            "rebuilds/%s" % rebuild_identity["version_id"]
            if rebuild_identity
            else None
        ),
        "request_count": len(requests),
        "raw_response_count": len(raw),
        "technical_error_count": whole_response_error_count,
        "schema_valid_count": len(parsed_rows),
        "verified_result_count": len(verified_rows),
        "schema_error_count": schema_errors,
        "verification_error_count": verification_errors,
        "api_error_count": len(api_errors),
        "missing_response_count": sum(
            row["error_code"] == "missing_response"
            for row in validation_errors
        ),
        "candidate_accounting": {
            "expected_references": candidate_expected,
            "verified_exactly_once_references": candidate_verified,
            "accounted_once_references": candidate_accounted_once,
            "missing_reference_keys": candidate_missing,
            "duplicate_reference_keys": candidate_duplicate,
            "unverified_reference_keys": candidate_unverified,
            "unknown_reference_keys": candidate_unknown,
            "unverified_references_due_to_whole_response_errors": sum(
                len(supplied_candidate_keys(dossiers[row["custom_id"]]))
                for row in tiered_rows
                if row.get("technical_error")
            ),
            "papers_exact": sum(
                row["result"]["candidate_accounting"]["exact"]
                for row in verified_rows
            ),
        },
        "quote_verification": {
            "literal_exact": quote_counts["literal_exact"],
            "allowlisted_normalized_exact": quote_counts[
                "allowlisted_normalized_exact"
            ],
            "valid": quote_valid,
            "invalid": quote_counts["invalid"],
            "total": quote_counts["total"],
            "success_rate": (
                quote_valid / quote_counts["total"]
                if quote_counts["total"]
                else 1.0
            ),
            "fuzzy_accepted": 0,
            "verification_error_papers": len(evidence_error_papers),
        },
        "quote_audit": quote_audit["summary"],
        "paper_usability": dict(sorted(paper_usability.items())),
        "component_error_counts_by_code": dict(
            sorted(component_error_counts.items())
        ),
        "claim_error_counts_by_code": dict(sorted(claim_error_counts.items())),
        "component_warning_counts_by_code": dict(
            sorted(component_warning_counts.items())
        ),
        "attribution_counts": dict(sorted(attribution_counts.items())),
        "tier_counts": dict(sorted(aggregate_tiers.items())),
        "event_dimension_counts": {
            "direction_processes": dict(sorted(event_direction_counts.items())),
            "dynamic_regimes": dict(sorted(event_regime_counts.items())),
            "condition_linkage": dict(sorted(event_condition_counts.items())),
        },
        "new_claim_counts": {
            "state_claims": new_state_claims,
            "event_claims": new_event_claims,
            "total": new_state_claims + new_event_claims,
        },
        "no_candidate_recovered_paper_count": no_candidate_recovered_papers,
        "actual_usage": {
            "input_tokens": usage_totals["input_tokens"],
            "cached_input_tokens": usage_totals["cached_input_tokens"],
            "uncached_input_tokens": uncached_tokens,
            "output_tokens": usage_totals["output_tokens"],
        },
        "actual_cost_usd": actual_cost,
        "pricing": pricing,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "model_snapshots": dict(sorted(model_snapshots.items())),
        "requested_model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "preparation_code_sha256": config["code_sha256"],
        "build_code_sha256": build_code["sha256"],
        "build_code_files_sha256": build_code["files"],
        "cohort_breakdown": cohort_breakdown,
        "artifacts": {
            name: {
                "path": name,
                "sha256": sha256_file(output_root / name),
            }
            for name in (
                "parsed_results.jsonl",
                "verified_results.jsonl",
                "tiered_results.jsonl",
                "validation_errors.jsonl",
                "quote_audit.json",
                "quote_audit_examples.md",
            )
        },
        "raw_responses_preserved": True,
    }
    _atomic_json(output_root / "pilot_report.json", report)
    _write_build_progress(
        progress_path,
        "completed",
        total_requests,
        total_requests,
        workers,
        build_started_at,
        str(output_root),
    )
    return report


def _tolerant_rebuild_identity(root, workers):
    source_files = [
        root / "run_config.json",
        *(sorted((root / "raw_responses").glob("*.json"))),
        *(sorted((root / "api_errors").glob("*.json"))),
    ]
    return {
        "format_version": CONTEXTUAL_TOLERANT_REBUILD_VERSION,
        "source_run": str(root),
        "source_artifacts_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in source_files
        },
        "build_code_sha256": _code_identity()["sha256"],
        "parallel_build": _parallel_build_identity(workers),
    }


def rebuild_contextual_pilot(
    run_dir,
    workers=DEFAULT_CONTEXTUAL_BUILD_WORKERS,
):
    """Publish an immutable content-addressed tolerant rebuild offline."""
    workers = _validate_build_workers(workers)
    root = Path(run_dir).resolve()
    identity = _tolerant_rebuild_identity(root, workers)
    version_id = "tolerant-" + _sha256_json(identity)[:16]
    identity["version_id"] = version_id
    versions_root = root / "rebuilds"
    destination = versions_root / version_id
    if destination.exists():
        report_path = destination / "pilot_report.json"
        if not report_path.is_file():
            raise ContextualPilotError(
                "existing tolerant rebuild is incomplete: %s" % destination
            )
        report = _read_json(report_path)
        if report.get("rebuild_identity") != identity:
            raise ContextualPilotError(
                "existing tolerant rebuild identity mismatch: %s" % destination
            )
        for artifact in report.get("artifacts", {}).values():
            path = destination / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ContextualPilotError(
                    "existing tolerant rebuild artifact mismatch: %s" % path
                )
        return report
    versions_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=version_id + ".", dir=str(versions_root))
    )
    try:
        report = build_contextual_pilot(
            root,
            output_dir=staging,
            rebuild_identity=identity,
            workers=workers,
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


__all__ = [
    "CONTEXTUAL_PILOT_LIFECYCLE_VERSION",
    "CONTEXTUAL_PARALLEL_BUILD_VERSION",
    "CONTEXTUAL_TOLERANT_REBUILD_VERSION",
    "DEFAULT_CONTEXTUAL_BUILD_RECORD_TIMEOUT_SECONDS",
    "DEFAULT_CONTEXTUAL_BUILD_WORKERS",
    "DEFAULT_CONTEXTUAL_EXPECTED_OUTPUT_TOKENS",
    "DEFAULT_CONTEXTUAL_MAX_COMPLETION_TOKENS",
    "DEFAULT_CONTEXTUAL_PILOT_MODEL",
    "DEFAULT_CONTEXTUAL_WORKERS",
    "ContextualPilotError",
    "build_contextual_pilot",
    "execute_contextual_pilot",
    "prepare_contextual_pilot",
    "rebuild_contextual_pilot",
]
