"""Auditable, resumable OpenAI Batch lifecycle for literature extraction."""

import hashlib
import importlib
import json
import os
import platform
import signal
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path

from .doi import normalize_doi
from .extractor import prepare_group, verify_source_sha256
from .identity import IDENTITY_POLICY
from .hashing import sha256_file
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT
from .schema import OUTPUT_SCHEMA, SCHEMA_VERSION, schema_sha256
from .serialization import canonical_json, write_json, write_jsonl
from .validation import (
    build_layers,
    curate_result,
    validate_parsed_output,
    verify_result,
)


LIFECYCLE_VERSION = "openai_batch_lifecycle_v2"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_REQUESTS = 50000
DEFAULT_MAX_BYTES = 190 * 1024 * 1024
DEFAULT_PREPARE_WORKERS = 2
DEFAULT_EXPECTED_OUTPUT_TOKENS = 1000
DEFAULT_MAX_COMPLETION_TOKENS = 6000
DEFAULT_COLLECT_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_COLLECTION_TIMEOUT_SECONDS = 120.0
COLLECTION_CACHE_FORMAT_VERSION = "collection_layers_sqlite_v1"
COLLECTION_COMMIT_BATCH_SIZE = 16
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


class LifecycleError(RuntimeError):
    """Raised for unsafe or inconsistent lifecycle operations."""


class ValidationTimeoutError(BaseException):
    """Control-flow exception that cannot be swallowed by validation handlers."""


class OpenAIBatchClient:
    """Small adapter around the OpenAI SDK; lifecycle tests can use a fake."""

    def __init__(self, client=None):
        if client is None:
            try:
                OpenAI = importlib.import_module("openai").OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "API actions require the optional 'openai' package"
                ) from exc
            client = OpenAI()
        self.client = client

    def upload_file(self, path):
        with Path(path).open("rb") as handle:
            return _field(self.client.files.create(file=handle, purpose="batch"), "id")

    def create_batch(self, input_file_id, endpoint, completion_window, metadata):
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )
        return _object_dict(batch)

    def find_batch_by_metadata(self, metadata):
        """Find one previously created batch with all requested metadata."""
        matches = []
        after = None
        while True:
            arguments = {"limit": 100}
            if after:
                arguments["after"] = after
            page = self.client.batches.list(**arguments)
            batches = _field(page, "data", []) or []
            for batch in batches:
                record = _object_dict(batch)
                remote_metadata = _field(record, "metadata", {}) or {}
                if all(
                    str(_field(remote_metadata, key, "")) == str(value)
                    for key, value in metadata.items()
                ):
                    matches.append(record)
            if not _field(page, "has_more", False):
                break
            after = _field(batches[-1], "id") if batches else None
            if not after:
                raise LifecycleError("cannot paginate Batch metadata reconciliation")
        if len(matches) > 1:
            raise LifecycleError(
                "multiple remote batches match submission metadata: %s"
                % ",".join(sorted(str(_field(row, "id", "")) for row in matches))
            )
        return matches[0] if matches else None

    def retrieve_batch(self, batch_id):
        return _object_dict(self.client.batches.retrieve(batch_id))

    def download_file(self, file_id):
        response = self.client.files.content(file_id)
        if isinstance(response, bytes):
            return response
        if hasattr(response, "read"):
            return response.read()
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content
        raise TypeError("OpenAI file content response is not bytes-like")


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _object_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError("API object cannot be converted to a dictionary")


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path, strict=True):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                if strict:
                    raise ValueError("blank JSONL line at %s:%d" % (path, line_number))
                continue
            try:
                rows.append(json.loads(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "invalid JSONL at %s:%d: %s" % (path, line_number, exc)
                ) from exc
    return rows


def _load_canonical_manifest(path):
    rows = []
    previous_key = None
    seen_groups = set()
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                raise ValueError(
                    "manifest must be canonical nonblank newline-terminated JSONL"
                )
            line = raw[:-1]
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError("invalid manifest line %d" % line_number) from exc
            if line != canonical_json(row):
                raise ValueError("manifest line %d is not canonical JSON" % line_number)
            if not isinstance(row, dict):
                raise ValueError("manifest line %d is not an object" % line_number)
            group_id = str(row.get("group_id") or "")
            if not group_id or group_id in seen_groups:
                raise ValueError("missing or duplicate group_id at line %d" % line_number)
            key = (str(row.get("doi") or ""), str(row.get("uniprot_accession") or ""))
            if previous_key is not None and key < previous_key:
                raise ValueError("manifest rows are not canonically sorted")
            previous_key = key
            seen_groups.add(group_id)
            rows.append(row)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def _package_hashes():
    package_dir = Path(__file__).resolve().parent
    files = {}
    for path in sorted(package_dir.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        files[path.name] = sha256_file(path)
    return files, _sha256_json(files)


def _environment_record():
    record = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    return record, _sha256_json(record)


def _request_line(request):
    return (canonical_json(request) + "\n").encode("utf-8")


def _estimate_tokens(requests):
    characters = sum(
        len(message.get("content", ""))
        for request in requests
        for message in request["body"].get("messages", [])
    )
    try:
        import tiktoken
    except ImportError:
        return {
            "method": "ceil_characters_divided_by_4",
            "input_characters": characters,
            "estimated_input_tokens": (characters + 3) // 4,
        }
    model = requests[0]["body"].get("model", "") if requests else ""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    content_tokens = 0
    payload_tokens = 0
    for request in requests:
        body = request["body"]
        content_tokens += sum(
            len(encoding.encode(message.get("content", "")))
            for message in body.get("messages", [])
        )
        payload_tokens += len(
            encoding.encode(
                canonical_json(
                    {
                        "messages": body.get("messages", []),
                        "response_format": body.get("response_format", {}),
                    }
                )
            )
        )
    return {
        "method": "tiktoken_full_messages_and_response_schema_upper_estimate",
        "encoding": encoding.name,
        "input_characters": characters,
        "content_tokens": content_tokens,
        "estimated_input_tokens": payload_tokens,
    }


def build_cost_report(
    token_estimate,
    request_count,
    batch_input_price_per_million,
    batch_output_price_per_million,
    batch_cached_input_price_per_million,
    expected_output_tokens_per_request,
    max_completion_tokens,
):
    """Return explicit estimated/expected/upper-bound Batch token costs."""
    prices = {
        "batch_input_per_million_tokens_usd": float(batch_input_price_per_million),
        "batch_output_per_million_tokens_usd": float(batch_output_price_per_million),
        "batch_cached_input_per_million_tokens_usd": float(
            batch_cached_input_price_per_million
        ),
    }
    values = list(prices.values()) + [
        float(expected_output_tokens_per_request),
        float(max_completion_tokens),
    ]
    if any(value < 0 for value in values):
        raise ValueError("cost inputs must be non-negative")
    if expected_output_tokens_per_request > max_completion_tokens:
        raise ValueError("expected output tokens cannot exceed max completion tokens")
    input_tokens = int(token_estimate["estimated_input_tokens"])
    expected_output = int(request_count * expected_output_tokens_per_request)
    worst_output = int(request_count * max_completion_tokens)
    input_cost = input_tokens * prices["batch_input_per_million_tokens_usd"] / 1_000_000
    cached_cost = (
        input_tokens
        * prices["batch_cached_input_per_million_tokens_usd"]
        / 1_000_000
    )
    expected_output_cost = (
        expected_output * prices["batch_output_per_million_tokens_usd"] / 1_000_000
    )
    worst_output_cost = (
        worst_output * prices["batch_output_per_million_tokens_usd"] / 1_000_000
    )
    return {
        "pricing": prices,
        "input": {
            "method": token_estimate["method"],
            "encoding": token_estimate.get("encoding", ""),
            "characters": int(token_estimate["input_characters"]),
            "content_tokens": int(token_estimate.get("content_tokens", input_tokens)),
            "estimated_tokens": input_tokens,
            "estimated_uncached_cost_usd": input_cost,
            "estimated_fully_cached_cost_usd": cached_cost,
        },
        "output": {
            "expected_tokens_per_request": expected_output_tokens_per_request,
            "expected_tokens": expected_output,
            "expected_cost_usd": expected_output_cost,
            "max_completion_tokens_per_request": max_completion_tokens,
            "worst_case_tokens": worst_output,
            "worst_case_cost_usd": worst_output_cost,
        },
        "totals": {
            "expected_uncached_usd": input_cost + expected_output_cost,
            "expected_fully_cached_usd": cached_cost + expected_output_cost,
            "worst_case_uncached_usd": input_cost + worst_output_cost,
            "worst_case_fully_cached_usd": cached_cost + worst_output_cost,
        },
        "notes": [
            (
                "Input tokens use tokenizer output over canonical messages and response "
                "schema; provider framing may make billed tokens slightly lower."
                if str(token_estimate["method"]).startswith("tiktoken_")
                else "Input tokens are a character-based estimate, not tokenizer output."
            ),
            "Expected output is a planning assumption; worst-case output uses the configured per-request cap.",
            "Cached-input totals are an all-input-cached scenario, not a prediction of cache eligibility.",
        ],
    }


def _write_shards(run_dir, prepared, max_requests, max_bytes):
    if max_requests <= 0 or max_bytes <= 0:
        raise ValueError("shard limits must be positive")
    shards = []
    current = []
    current_bytes = 0
    for request, _ in prepared:
        line = _request_line(request)
        if len(line) > max_bytes:
            raise ValueError(
                "request %s is %d bytes, exceeding shard byte limit %d"
                % (request["custom_id"], len(line), max_bytes)
            )
        if current and (
            len(current) >= max_requests or current_bytes + len(line) > max_bytes
        ):
            shards.append((current, current_bytes))
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += len(line)
    if current:
        shards.append((current, current_bytes))

    records = []
    for index, (requests, serialized_bytes) in enumerate(shards):
        name = "shard-%05d" % index
        relative = Path("shards") / name / "requests.jsonl"
        path = Path(run_dir) / relative
        write_jsonl(path, requests)
        records.append(
            {
                "index": index,
                "name": name,
                "request_file": str(relative),
                "request_count": len(requests),
                "serialized_bytes": serialized_bytes,
                "request_sha256": sha256_file(path),
                "custom_ids": [request["custom_id"] for request in requests],
            }
        )
    return records


def _prepare_worker(arguments):
    if len(arguments) == 4:
        index, group, model, max_completion_tokens = arguments
        manifest_rows = [group]
    else:
        index, group, manifest_rows, model, max_completion_tokens = arguments
    request, metadata = prepare_group(
        group,
        model,
        max_completion_tokens=max_completion_tokens,
        verify_source=False,
        manifest_rows=manifest_rows,
    )
    return index, request, metadata


def _progress_reporter(progress, total):
    if callable(progress):
        return lambda completed: progress(completed, total), lambda: None
    mode = progress or "none"
    if mode == "auto":
        mode = "tqdm" if sys.stderr.isatty() else "log"
    if mode == "tqdm":
        try:
            tqdm = importlib.import_module("tqdm").tqdm
        except ImportError:
            mode = "log"
        else:
            bar = tqdm(total=total, desc="Preparing", unit="group")
            return lambda completed: bar.update(completed - bar.n), bar.close
    if mode == "log":
        interval = max(1, total // 100)

        def report(completed):
            if completed == total or completed % interval == 0:
                print(
                    "prepare progress: %d/%d (%.1f%%)"
                    % (completed, total, 100.0 * completed / max(1, total)),
                    file=sys.stderr,
                    flush=True,
                )

        return report, lambda: None
    if mode != "none":
        raise ValueError("progress must be auto, log, tqdm, none, or a callback")
    return lambda completed: None, lambda: None


def _verify_manifest_sources(groups):
    """Verify each distinct paper once before trusting worker-side parsing."""
    checksums = {}
    for group in groups:
        path = Path(group["paper_path"]).resolve()
        expected = str(group.get("paper_sha256") or "").lower()
        if not expected:
            raise ValueError("paper_sha256 is required for %s" % group["group_id"])
        if path not in checksums:
            checksums[path] = sha256_file(path)
        if checksums[path] != expected:
            raise ValueError("paper SHA-256 mismatch for group %s" % group["group_id"])
    return len(checksums)


def _open_prepare_cache(path):
    connection = sqlite3.connect(str(path))
    connection.execute(
        "CREATE TABLE IF NOT EXISTS prepared "
        "(item_index INTEGER PRIMARY KEY, request_json TEXT NOT NULL, "
        "metadata_json TEXT NOT NULL)"
    )
    return connection


def _prepare_items(
    groups,
    model,
    max_completion_tokens,
    workers,
    cache,
    progress,
    doi_rows_by_group=None,
):
    if workers <= 0:
        raise ValueError("workers must be positive")
    existing = {
        int(row[0]): (json.loads(row[1]), json.loads(row[2]))
        for row in cache.execute(
            "SELECT item_index, request_json, metadata_json FROM prepared"
        )
    }
    if any(index < 0 or index >= len(groups) for index in existing):
        raise LifecycleError("offline preparation cache contains an invalid index")
    pending = [index for index in range(len(groups)) if index not in existing]
    report, close_progress = _progress_reporter(progress, len(groups))
    completed = len(existing)
    report(completed)

    def persist(result):
        nonlocal completed
        index, request, metadata = result
        cache.execute(
            "INSERT OR REPLACE INTO prepared VALUES (?, ?, ?)",
            (index, canonical_json(request), canonical_json(metadata)),
        )
        cache.commit()
        existing[index] = (request, metadata)
        completed += 1
        report(completed)

    try:
        doi_rows_by_group = doi_rows_by_group or {
            str(group["group_id"]): [group] for group in groups
        }

        def arguments(index):
            group = groups[index]
            return (
                index,
                group,
                doi_rows_by_group[str(group["group_id"])],
                model,
                max_completion_tokens,
            )

        if workers == 1:
            for index in pending:
                persist(_prepare_worker(arguments(index)))
        elif pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                pending_iter = iter(pending)
                futures = {}
                for _ in range(min(len(pending), workers * 4)):
                    index = next(pending_iter)
                    future = executor.submit(
                        _prepare_worker,
                        arguments(index),
                    )
                    futures[future] = index
                while futures:
                    for future in as_completed(list(futures)):
                        futures.pop(future)
                        persist(future.result())
                        try:
                            index = next(pending_iter)
                        except StopIteration:
                            continue
                        replacement = executor.submit(
                            _prepare_worker,
                            arguments(index),
                        )
                        futures[replacement] = index
    finally:
        close_progress()
    return [existing[index] for index in range(len(groups))]


def prepare_run(
    manifest_path,
    run_dir,
    model=DEFAULT_MODEL,
    max_requests=DEFAULT_MAX_REQUESTS,
    max_bytes=DEFAULT_MAX_BYTES,
    seed=0,
    workers=DEFAULT_PREPARE_WORKERS,
    progress="none",
    batch_input_price_per_million=0.0,
    batch_output_price_per_million=0.0,
    batch_cached_input_price_per_million=0.0,
    expected_output_tokens_per_request=DEFAULT_EXPECTED_OUTPUT_TOKENS,
    max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
):
    """Resume offline staging as needed, then atomically publish an immutable run."""
    source = Path(manifest_path).resolve()
    destination = Path(run_dir).resolve()
    staging = destination.parent / (".%s.preparing" % destination.name)
    if destination.exists():
        raise LifecycleError("run directory already exists: %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        manifest_bytes = source.read_bytes()
        groups = _load_canonical_manifest(source)
        rows_by_doi = {}
        seen_units = set()
        for group in groups:
            doi = normalize_doi(group.get("doi"))
            unit = (doi, str(group.get("uniprot_accession") or "").upper())
            if unit in seen_units:
                raise ValueError(
                    "manifest has duplicate normalized DOI/UniProt request unit: %s/%s"
                    % unit
                )
            seen_units.add(unit)
            rows_by_doi.setdefault(doi, []).append(group)
        for doi in rows_by_doi:
            rows_by_doi[doi] = sorted(
                rows_by_doi[doi],
                key=lambda row: (
                    str(row.get("uniprot_accession") or "").upper(),
                    str(row.get("group_id") or ""),
                ),
            )
        doi_rows_by_group = {
            str(group["group_id"]): rows_by_doi[normalize_doi(group.get("doi"))]
            for group in groups
        }
        code_files, code_hash = _package_hashes()
        preparation_identity = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "schema_sha256": _sha256_json(OUTPUT_SCHEMA),
            "identity_policy_sha256": _sha256_json(IDENTITY_POLICY),
            "preparation_code_sha256": code_hash,
        }
        if staging.exists():
            state_path = staging / "preparation_state.json"
            staged_state = _read_json(state_path) if state_path.exists() else {}
            if staged_state.get("identity") != preparation_identity:
                raise LifecycleError(
                    "existing offline preparation staging does not match requested inputs: %s"
                    % staging
                )
        else:
            staging.mkdir()
            _atomic_bytes(staging / "manifest.jsonl", manifest_bytes)
            write_json(
                staging / "preparation_state.json",
                {"identity": preparation_identity, "created_at": _utc_now()},
            )
        if (staging / "manifest.jsonl").read_bytes() != manifest_bytes:
            raise LifecycleError("staged manifest does not match preparation input")
        preparation_started = time.perf_counter()
        unique_source_files = _verify_manifest_sources(groups)
        cache = _open_prepare_cache(staging / "preparation.sqlite3")
        try:
            prepared = _prepare_items(
                groups,
                model,
                max_completion_tokens,
                workers,
                cache,
                progress,
                doi_rows_by_group=doi_rows_by_group,
            )
        finally:
            cache.close()
        _, finished_code_hash = _package_hashes()
        if finished_code_hash != code_hash:
            raise LifecycleError(
                "preparation implementation changed during offline staging"
            )
        requests = [request for request, _ in prepared]
        metadata = [item for _, item in prepared]
        preparation_finished = time.perf_counter()
        custom_ids = [request["custom_id"] for request in requests]
        if len(custom_ids) != len(set(custom_ids)):
            raise ValueError("manifest produces duplicate custom IDs")

        write_jsonl(
            staging / "request_metadata.jsonl",
            metadata,
            sort_key=lambda row: row["custom_id"],
        )
        shards = _write_shards(staging, prepared, max_requests, max_bytes)
        environment, environment_hash = _environment_record()
        token_estimate = _estimate_tokens(requests)
        cost_report = build_cost_report(
            token_estimate,
            len(requests),
            batch_input_price_per_million,
            batch_output_price_per_million,
            batch_cached_input_price_per_million,
            expected_output_tokens_per_request,
            max_completion_tokens,
        )
        write_json(staging / "cost_report.json", cost_report)
        preparation_state = _read_json(staging / "preparation_state.json")
        stage_names = (
            "source_verification",
            "text_extraction",
            "identity_graph",
            "retrieval",
            "prompt_rendering",
            "total",
        )
        item_stage_totals = {
            stage: sum(
                float(item.get("preparation_elapsed_seconds", {}).get(stage, 0.0))
                for item in metadata
            )
            for stage in stage_names
        }
        preparation_timing = {
            "wall_elapsed_seconds": preparation_finished - preparation_started,
            "item_count": len(metadata),
            "item_stage_total_seconds": item_stage_totals,
            "mean_item_total_seconds": (
                item_stage_totals["total"] / len(metadata) if metadata else 0.0
            ),
            "max_item_total_seconds": max(
                (
                    float(
                        item.get("preparation_elapsed_seconds", {}).get(
                            "total", 0.0
                        )
                    )
                    for item in metadata
                ),
                default=0.0,
            ),
        }
        config = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "created_at": preparation_state["created_at"],
            "model": model,
            "seed": seed,
            "manifest_source": str(source),
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "manifest_copy_sha256": sha256_file(staging / "manifest.jsonl"),
            "request_metadata_sha256": sha256_file(
                staging / "request_metadata.jsonl"
            ),
            "cost_report_sha256": sha256_file(staging / "cost_report.json"),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": _sha256_json(OUTPUT_SCHEMA),
            "source_sha256": {
                item["group_id"]: item["source_sha256"] for item in metadata
            },
            "code_files_sha256": code_files,
            "code_sha256": code_hash,
            "environment": environment,
            "environment_sha256": environment_hash,
            "token_estimate": token_estimate,
            "cost_report": cost_report,
            "preparation_timing": preparation_timing,
            "preparation_code_sha256": code_hash,
            "unique_source_files_verified": unique_source_files,
            "limits": {
                "max_requests_per_shard": max_requests,
                "max_serialized_bytes_per_shard": max_bytes,
                "max_completion_tokens_per_request": max_completion_tokens,
            },
            "request_count": len(requests),
            "shards": shards,
        }
        write_json(staging / "run_config.json", config)
        state = {
            "lifecycle_version": LIFECYCLE_VERSION,
            "run_config_sha256": sha256_file(staging / "run_config.json"),
            "shards": {
                shard["name"]: {
                    "upload_id": None,
                    "batch_id": None,
                    "submission_intent": None,
                    "submitted_at": None,
                    "last_status": None,
                    "output_file_id": None,
                    "error_file_id": None,
                }
                for shard in shards
            },
        }
        write_json(staging / "batch_state.json", state)
        (staging / "preparation.sqlite3").unlink()
        (staging / "preparation_state.json").unlink()
        failed_marker = staging / "PREPARE_FAILED.json"
        if failed_marker.exists():
            failed_marker.unlink()
        os.replace(staging, destination)
        return config
    except Exception:
        if staging.exists():
            write_json(
                staging / "PREPARE_FAILED.json",
                {"failed_at": _utc_now(), "error": "%s" % sys.exc_info()[1]},
            )
        raise


def _load_run(run_dir):
    root = Path(run_dir).resolve()
    config = _read_json(root / "run_config.json")
    state = _read_json(root / "batch_state.json")
    expected = state.get("run_config_sha256")
    actual = sha256_file(root / "run_config.json")
    if expected != actual:
        raise LifecycleError("run_config.json checksum mismatch")
    if sha256_file(root / "manifest.jsonl") != config["manifest_copy_sha256"]:
        raise LifecycleError("manifest.jsonl checksum mismatch")
    if (
        sha256_file(root / "request_metadata.jsonl")
        != config["request_metadata_sha256"]
    ):
        raise LifecycleError("request_metadata.jsonl checksum mismatch")
    if "cost_report_sha256" in config:
        if sha256_file(root / "cost_report.json") != config["cost_report_sha256"]:
            raise LifecycleError("cost_report.json checksum mismatch")
    for shard in config["shards"]:
        request_path = root / shard["request_file"]
        if sha256_file(request_path) != shard["request_sha256"]:
            raise LifecycleError("prepared request checksum mismatch: %s" % request_path)
    return root, config, state


def submit_run(run_dir, client=None, completion_window="24h"):
    """Upload and submit shards with crash-safe intent and reconciliation."""
    root, config, state = _load_run(run_dir)
    client = client or OpenAIBatchClient()
    states = state["shards"]
    if all(states[shard["name"]].get("batch_id") for shard in config["shards"]):
        raise LifecycleError("all shards are already submitted")

    for shard in config["shards"]:
        shard_state = states[shard["name"]]
        if shard_state.get("batch_id"):
            continue
        if not shard_state.get("upload_id"):
            shard_state["upload_id"] = client.upload_file(root / shard["request_file"])
            write_json(root / "batch_state.json", state)
        expected_intent = {
            "submission_key": _sha256_json(
                {
                    "run_config_sha256": state["run_config_sha256"],
                    "shard": shard["name"],
                    "request_sha256": shard["request_sha256"],
                    "upload_id": shard_state["upload_id"],
                }
            )[:40],
            "endpoint": "/v1/chat/completions",
            "completion_window": completion_window,
            "upload_id": shard_state["upload_id"],
            "metadata": {
                "run": root.name[:64],
                "run_id": state["run_config_sha256"][:40],
                "shard": shard["name"],
                "lifecycle": LIFECYCLE_VERSION,
            },
        }
        expected_intent["metadata"]["submission_key"] = expected_intent[
            "submission_key"
        ]
        intent = shard_state.get("submission_intent")
        intent_preexisting = intent is not None
        if intent is None:
            intent = dict(expected_intent, intended_at=_utc_now())
            shard_state["submission_intent"] = intent
            # This durable write must precede the paid remote create call.
            write_json(root / "batch_state.json", state)
        else:
            for field in (
                "submission_key",
                "endpoint",
                "completion_window",
                "upload_id",
                "metadata",
            ):
                if intent.get(field) != expected_intent[field]:
                    raise LifecycleError(
                        "persisted submission intent changed for %s" % shard["name"]
                    )

        finder = getattr(client, "find_batch_by_metadata", None)
        if intent_preexisting and finder is None:
            raise LifecycleError(
                "cannot safely resume %s without Batch metadata reconciliation"
                % shard["name"]
            )
        batch = finder(intent["metadata"]) if finder else None
        if batch is not None:
            batch_id = _field(batch, "id")
            if not batch_id:
                raise LifecycleError("reconciled batch has no id")
            shard_state.update(
                {
                    "batch_id": batch_id,
                    "submitted_at": shard_state.get("submitted_at") or _utc_now(),
                    "reconciled_at": _utc_now(),
                    "last_status": _field(batch, "status"),
                    "output_file_id": _field(batch, "output_file_id"),
                    "error_file_id": _field(batch, "error_file_id"),
                }
            )
            write_json(root / "batch_state.json", state)
            continue
        batch = client.create_batch(
            shard_state["upload_id"],
            intent["endpoint"],
            completion_window,
            intent["metadata"],
        )
        batch_id = _field(batch, "id")
        if not batch_id:
            raise LifecycleError("batch creation returned no id")
        shard_state.update(
            {
                "batch_id": batch_id,
                "submitted_at": _utc_now(),
                "last_status": _field(batch, "status"),
                "output_file_id": _field(batch, "output_file_id"),
                "error_file_id": _field(batch, "error_file_id"),
            }
        )
        write_json(root / "batch_state.json", state)
    return state


def status_run(run_dir, client=None):
    """Refresh shard statuses and aggregate exact expected/completed/failed counts."""
    root, config, state = _load_run(run_dir)
    client = client or OpenAIBatchClient()
    summary = {
        "expected": sum(shard["request_count"] for shard in config["shards"]),
        "completed": 0,
        "failed": 0,
        "pending": 0,
        "unsubmitted": 0,
        "shards": [],
    }
    for shard in config["shards"]:
        shard_state = state["shards"][shard["name"]]
        if not shard_state.get("batch_id"):
            summary["unsubmitted"] += shard["request_count"]
            summary["shards"].append(
                {"name": shard["name"], "status": "unsubmitted", "expected": shard["request_count"]}
            )
            continue
        batch = client.retrieve_batch(shard_state["batch_id"])
        counts = _field(batch, "request_counts", {}) or {}
        completed = int(_field(counts, "completed", 0) or 0)
        failed = int(_field(counts, "failed", 0) or 0)
        pending = max(0, shard["request_count"] - completed - failed)
        status = _field(batch, "status", "unknown")
        summary["completed"] += completed
        summary["failed"] += failed
        summary["pending"] += pending
        shard_state.update(
            {
                "last_status": status,
                "status_checked_at": _utc_now(),
                "output_file_id": _field(batch, "output_file_id"),
                "error_file_id": _field(batch, "error_file_id"),
                "request_counts": {
                    "completed": completed,
                    "failed": failed,
                    "pending": pending,
                },
            }
        )
        summary["shards"].append(
            {
                "name": shard["name"],
                "status": status,
                "expected": shard["request_count"],
                "completed": completed,
                "failed": failed,
                "pending": pending,
            }
        )
    write_json(root / "batch_state.json", state)
    write_json(root / "status.json", summary)
    return summary


def _verify_sources(root, config):
    groups = _read_jsonl(root / "manifest.jsonl")
    for group in groups:
        expected = config["source_sha256"].get(str(group["group_id"]))
        try:
            actual = verify_source_sha256(group)
        except (OSError, ValueError) as exc:
            raise LifecycleError(
                "recorded source checksum mismatch: %s" % group["group_id"]
            ) from exc
        if actual != expected:
            raise LifecycleError("recorded source checksum mismatch: %s" % group["group_id"])
    return groups


def _recorded_schema_identity(config, metadata):
    """Resolve and verify one request's immutable schema contract."""
    config_version = config.get("schema_version")
    metadata_version = metadata.get("schema_version")
    if config_version and metadata_version and config_version != metadata_version:
        raise LifecycleError("run/request schema versions disagree")
    version = metadata_version or config_version
    if not version:
        raise LifecycleError("run/request metadata does not record a schema version")
    config_hash = config.get("schema_sha256")
    metadata_hash = metadata.get("schema_sha256")
    if config_hash and metadata_hash and config_hash != metadata_hash:
        raise LifecycleError("run/request schema checksums disagree")
    recorded_hash = metadata_hash or config_hash
    if not recorded_hash:
        raise LifecycleError("run/request metadata does not record a schema checksum")
    try:
        registered_hash = schema_sha256(version)
    except ValueError as exc:
        raise LifecycleError(str(exc)) from exc
    if recorded_hash != registered_hash:
        raise LifecycleError(
            "recorded schema checksum does not match registered %s" % version
        )
    return {"version": version, "sha256": recorded_hash}


def _download_once(client, file_id, destination):
    path = Path(destination)
    if path.exists():
        return
    _atomic_bytes(path, client.download_file(file_id))


def _response_content(row):
    response = row.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        raise ValueError("missing successful response")
    body = response.get("body")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing response choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("missing string message content")
    return content


def _read_raw_records(paths):
    records = []
    malformed = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    malformed.append({"file": str(path), "line": line_number, "error": "blank line"})
                    continue
                try:
                    value = json.loads(raw)
                except ValueError as exc:
                    malformed.append(
                        {"file": str(path), "line": line_number, "error": str(exc)}
                    )
                    continue
                if not isinstance(value, dict) or not isinstance(value.get("custom_id"), str):
                    malformed.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "error": "record lacks string custom_id",
                        }
                    )
                    continue
                records.append((value["custom_id"], value, str(path), line_number))
    return records, malformed


def _map_groups_to_custom_ids(groups, metadata_by_id):
    """Join manifest groups to request IDs in linear time."""
    custom_id_by_group = {}
    for custom_id, metadata in metadata_by_id.items():
        group_id = str(metadata.get("group_id") or "")
        if not group_id or group_id in custom_id_by_group:
            raise LifecycleError("request metadata/group mapping is not one-to-one")
        custom_id_by_group[group_id] = custom_id

    group_by_id = {}
    for group in groups:
        group_id = str(group.get("group_id") or "")
        custom_id = custom_id_by_group.get(group_id)
        if custom_id is None or custom_id in group_by_id:
            raise LifecycleError("request metadata/group mapping is not one-to-one")
        group_by_id[custom_id] = group
    if len(group_by_id) != len(metadata_by_id):
        raise LifecycleError("request metadata/group mapping is not one-to-one")
    return group_by_id


def _scan_raw_records(paths):
    """Account raw JSONL without retaining provider response payloads."""
    records = []
    malformed = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    malformed.append(
                        {"file": str(path), "line": line_number, "error": "blank line"}
                    )
                    continue
                try:
                    value = json.loads(raw)
                except ValueError as exc:
                    malformed.append(
                        {"file": str(path), "line": line_number, "error": str(exc)}
                    )
                    continue
                if not isinstance(value, dict) or not isinstance(
                    value.get("custom_id"), str
                ):
                    malformed.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "error": "record lacks string custom_id",
                        }
                    )
                    continue
                records.append((value["custom_id"], str(path), line_number))
    return records, malformed


def _iter_raw_records(paths):
    """Yield provider records lazily after exact accounting has succeeded."""
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if isinstance(value, dict) and isinstance(value.get("custom_id"), str):
                    yield value["custom_id"], value, str(path), line_number


def _collection_cache_identity(
    root,
    config,
    metadata_by_id,
    raw_paths,
    error_paths,
    record_timeout_seconds,
):
    """Bind cached validation rows to every input and implementation identity."""
    schema_contracts = {
        (identity["version"], identity["sha256"])
        for identity in (
            _recorded_schema_identity(config, metadata)
            for metadata in metadata_by_id.values()
        )
    }
    if len(schema_contracts) != 1:
        raise LifecycleError("a collected run must have one recorded schema contract")
    schema_version, recorded_schema_sha256 = next(iter(schema_contracts))
    identity_metadata = {
        custom_id: {
            "identity_graph_sha256": metadata.get(
                "identity_graph_sha256", "legacy_identity_grounding"
            ),
            "identity_policy_sha256": metadata.get(
                "identity_policy_sha256", "legacy_identity_grounding"
            ),
        }
        for custom_id, metadata in sorted(metadata_by_id.items())
    }
    raw_artifacts = []
    for kind, paths in (("output", raw_paths), ("api_error", error_paths)):
        for path in paths:
            resolved = Path(path).resolve()
            try:
                relative = str(resolved.relative_to(root))
            except ValueError:
                relative = str(resolved)
            raw_artifacts.append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": sha256_file(resolved),
                }
            )
    code_hashes = _validation_code_hashes()
    return {
        "format_version": COLLECTION_CACHE_FORMAT_VERSION,
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "recorded_manifest_sha256": config.get(
            "manifest_copy_sha256", config.get("manifest_sha256", "unrecorded")
        ),
        "request_metadata_sha256": sha256_file(root / "request_metadata.jsonl"),
        "recorded_request_metadata_sha256": config.get(
            "request_metadata_sha256", "unrecorded"
        ),
        "schema_version": schema_version,
        "schema_sha256": recorded_schema_sha256,
        "current_identity_policy_sha256": _sha256_json(IDENTITY_POLICY),
        "request_identity_metadata_sha256": _sha256_json(identity_metadata),
        "validation_module_sha256": code_hashes["validation_module_sha256"],
        "identity_module_sha256": code_hashes["identity_module_sha256"],
        "validator_code_sha256": code_hashes["validator_code_sha256"],
        "curator_code_sha256": code_hashes["curator_code_sha256"],
        "collection_module_sha256": sha256_file(Path(__file__).resolve()),
        "record_timeout_seconds": float(record_timeout_seconds),
        "raw_artifacts": sorted(
            raw_artifacts, key=lambda item: (item["kind"], item["path"])
        ),
    }


def _open_collection_cache(path, identity):
    """Open a durable cache, refusing partial or mismatched provenance."""
    connection = sqlite3.connect(str(path), timeout=60)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS cache_metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS collection_results ("
            "custom_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
            "parsed_json TEXT, verified_json TEXT, curated_json TEXT, "
            "error_json TEXT, source_file TEXT NOT NULL, "
            "source_line INTEGER NOT NULL, completed_at TEXT NOT NULL, "
            "CHECK (status IN ('valid', 'error', 'timeout')), "
            "CHECK ((status = 'valid' AND parsed_json IS NOT NULL "
            "AND verified_json IS NOT NULL AND curated_json IS NOT NULL "
            "AND error_json IS NULL) OR "
            "(status != 'valid' AND parsed_json IS NULL "
            "AND verified_json IS NULL AND curated_json IS NULL "
            "AND error_json IS NOT NULL)))"
        )
        identity_json = canonical_json(identity)
        identity_hash = _sha256_bytes(identity_json.encode("utf-8"))
        recorded = dict(
            connection.execute("SELECT key, value FROM cache_metadata").fetchall()
        )
        if recorded:
            if (
                recorded.get("format_version") != COLLECTION_CACHE_FORMAT_VERSION
                or recorded.get("identity_json") != identity_json
                or recorded.get("identity_sha256") != identity_hash
                or set(recorded)
                != {"format_version", "identity_json", "identity_sha256"}
            ):
                raise LifecycleError(
                    "collection cache identity mismatch: %s" % Path(path)
                )
        else:
            existing_results = connection.execute(
                "SELECT COUNT(*) FROM collection_results"
            ).fetchone()[0]
            if existing_results:
                raise LifecycleError(
                    "collection cache has results without an identity: %s"
                    % Path(path)
                )
            connection.executemany(
                "INSERT INTO cache_metadata (key, value) VALUES (?, ?)",
                (
                    ("format_version", COLLECTION_CACHE_FORMAT_VERSION),
                    ("identity_json", identity_json),
                    ("identity_sha256", identity_hash),
                ),
            )
            connection.commit()
        return connection, identity_hash
    except Exception:
        connection.close()
        raise


def _collection_timeout_handler(signum, frame):
    raise ValidationTimeoutError("collection record exceeded its wall-clock timeout")


def _collection_worker(arguments):
    """Build and canonically serialize one set of layers inside a worker."""
    (
        custom_id,
        row,
        source_file,
        line_number,
        metadata,
        group,
        schema_identity,
        record_timeout_seconds,
    ) = arguments
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, _collection_timeout_handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, record_timeout_seconds)
        try:
            layers = build_layers(
                _response_content(row),
                metadata["selected_passages"],
                group,
                identity_graph=metadata.get("identity_graph"),
                schema_version=schema_identity["version"],
                schema_sha256_value=schema_identity["sha256"],
            )
            base = {"custom_id": custom_id, "group_id": metadata["group_id"]}
            return {
                "custom_id": custom_id,
                "status": "valid",
                "parsed_json": canonical_json(dict(base, result=layers["parsed"])),
                "verified_json": canonical_json(
                    dict(base, result=layers["verified"])
                ),
                "curated_json": canonical_json(dict(base, result=layers["curated"])),
                "error_json": None,
                "source_file": source_file,
                "source_line": line_number,
            }
        except ValidationTimeoutError as exc:
            error = {
                "custom_id": custom_id,
                "file": source_file,
                "line": line_number,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return {
                "custom_id": custom_id,
                "status": "timeout",
                "parsed_json": None,
                "verified_json": None,
                "curated_json": None,
                "error_json": canonical_json(error),
                "source_file": source_file,
                "source_line": line_number,
            }
        except Exception as exc:
            error = {
                "custom_id": custom_id,
                "file": source_file,
                "line": line_number,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return {
                "custom_id": custom_id,
                "status": "error",
                "parsed_json": None,
                "verified_json": None,
                "curated_json": None,
                "error_json": canonical_json(error),
                "source_file": source_file,
                "source_line": line_number,
            }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _collection_future_error(arguments, exc):
    custom_id, _, source_file, line_number = arguments[:4]
    error = {
        "custom_id": custom_id,
        "file": source_file,
        "line": line_number,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    return {
        "custom_id": custom_id,
        "status": "timeout" if isinstance(exc, ValidationTimeoutError) else "error",
        "parsed_json": None,
        "verified_json": None,
        "curated_json": None,
        "error_json": canonical_json(error),
        "source_file": source_file,
        "source_line": line_number,
    }


def _run_collection_tasks(tasks, workers, max_in_flight, persist):
    """Run tasks while retaining only a fixed-size set of submitted futures."""
    if workers == 1:
        for arguments in tasks:
            persist(_collection_worker(arguments))
        return
    iterator = iter(tasks)
    executor = ProcessPoolExecutor(max_workers=workers)
    futures = {}
    try:
        for _ in range(max_in_flight):
            try:
                arguments = next(iterator)
            except StopIteration:
                break
            futures[executor.submit(_collection_worker, arguments)] = arguments
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                arguments = futures.pop(future)
                try:
                    result = future.result()
                except ValidationTimeoutError as exc:
                    result = _collection_future_error(arguments, exc)
                except Exception as exc:
                    result = _collection_future_error(arguments, exc)
                persist(result)
                try:
                    replacement = next(iterator)
                except StopIteration:
                    continue
                futures[
                    executor.submit(_collection_worker, replacement)
                ] = replacement
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _collection_progress_reporter(progress, total, cached, initial_errors):
    if callable(progress):
        callback = progress
    else:
        mode = progress or "none"
        if mode == "auto":
            mode = "log"
        if mode == "none":
            callback = lambda snapshot: None
        elif mode == "log":
            callback = lambda snapshot: print(
                "collection progress: completed=%d/%d cached=%d errors=%d (%.1f%%)"
                % (
                    snapshot["completed"],
                    snapshot["total"],
                    snapshot["cached"],
                    snapshot["errors"],
                    snapshot["percent"],
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            raise ValueError("collection progress must be auto, log, none, or a callback")
    interval = max(1, total // 100)
    last_reported = {"completed": None, "errors": None}

    def report(completed, errors, force=False):
        previous = last_reported["completed"]
        if completed == previous and errors == last_reported["errors"]:
            return
        if not force and previous is not None and completed < total:
            if completed // interval == previous // interval:
                return
        callback(
            {
                "completed": completed,
                "total": total,
                "cached": cached,
                "errors": errors,
                "percent": 100.0 * completed / max(1, total),
            }
        )
        last_reported["completed"] = completed
        last_reported["errors"] = errors

    return report


def _persist_collection_batch(connection, results):
    if not results:
        return
    connection.executemany(
        "INSERT OR REPLACE INTO collection_results "
        "(custom_id, status, parsed_json, verified_json, curated_json, "
        "error_json, source_file, source_line, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                result["custom_id"],
                result["status"],
                result["parsed_json"],
                result["verified_json"],
                result["curated_json"],
                result["error_json"],
                result["source_file"],
                result["source_line"],
                _utc_now(),
            )
            for result in results
        ],
    )
    connection.commit()


def _atomic_cache_jsonl(connection, destination, column, statuses):
    """Stream canonical cached rows to an atomically replaced JSONL file."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    placeholders = ",".join("?" for _ in statuses)
    query = (
        "SELECT %s FROM collection_results WHERE status IN (%s) "
        "ORDER BY custom_id" % (column, placeholders)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            cursor = connection.execute(query, tuple(statuses))
            while True:
                rows = cursor.fetchmany(256)
                if not rows:
                    break
                for (canonical_row,) in rows:
                    handle.write(canonical_row)
                    handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect_run(
    run_dir,
    client=None,
    workers=DEFAULT_COLLECT_WORKERS,
    record_timeout_seconds=DEFAULT_COLLECTION_TIMEOUT_SECONDS,
    progress="none",
):
    """Download raw artifacts unchanged, account IDs, and build result layers."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if record_timeout_seconds <= 0:
        raise ValueError("record timeout must be positive")
    root, config, state = _load_run(run_dir)
    groups = _verify_sources(root, config)
    accounting_path = root / "accounting.json"
    recorded_accounting = (
        _read_json(accounting_path) if accounting_path.exists() else None
    )
    reusable_accounting = bool(
        isinstance(recorded_accounting, dict)
        and recorded_accounting.get("exact") is True
    )
    active_client = client
    raw_paths = []
    error_paths = []
    for shard in config["shards"]:
        shard_state = state["shards"][shard["name"]]
        if not shard_state.get("batch_id"):
            raise LifecycleError("cannot collect unsubmitted shard %s" % shard["name"])
        shard_dir = root / "shards" / shard["name"]
        output_path = shard_dir / "raw_output.jsonl"
        error_path = shard_dir / "raw_errors.jsonl"
        output_id = shard_state.get("output_file_id")
        error_id = shard_state.get("error_file_id")
        local_artifacts_complete = (
            shard_state.get("last_status") in TERMINAL_STATUSES
            and (not output_id or output_path.is_file())
            and (not error_id or error_path.is_file())
            and (output_path.is_file() or error_path.is_file())
        )
        if reusable_accounting and local_artifacts_complete:
            if output_path.is_file():
                raw_paths.append(output_path)
            if error_path.is_file():
                error_paths.append(error_path)
            continue

        if active_client is None:
            active_client = OpenAIBatchClient()
        batch = active_client.retrieve_batch(shard_state["batch_id"])
        shard_state.update(
            {
                "last_status": _field(batch, "status"),
                "output_file_id": _field(batch, "output_file_id"),
                "error_file_id": _field(batch, "error_file_id"),
            }
        )
        output_id = shard_state.get("output_file_id")
        error_id = shard_state.get("error_file_id")
        if output_id:
            _download_once(active_client, output_id, output_path)
            raw_paths.append(output_path)
        if error_id:
            _download_once(active_client, error_id, error_path)
            error_paths.append(error_path)
        if _field(batch, "status") not in TERMINAL_STATUSES:
            raise LifecycleError("shard %s is not terminal" % shard["name"])
    write_json(root / "batch_state.json", state)

    output_records, malformed_output = _scan_raw_records(raw_paths)
    error_records, malformed_error = _scan_raw_records(error_paths)
    expected = {
        custom_id
        for shard in config["shards"]
        for custom_id in shard["custom_ids"]
    }
    all_ids = [item[0] for item in output_records + error_records]
    counts = Counter(all_ids)
    duplicates = sorted(custom_id for custom_id, count in counts.items() if count > 1)
    unexpected = sorted(set(all_ids) - expected)
    missing = sorted(expected - set(all_ids))
    accounting = {
        "expected": len(expected),
        "output_records": len(output_records),
        "error_records": len(error_records),
        "missing_custom_ids": missing,
        "duplicate_custom_ids": duplicates,
        "unexpected_custom_ids": unexpected,
        "malformed_output_records": malformed_output,
        "malformed_error_records": malformed_error,
        "exact": not (
            missing
            or duplicates
            or unexpected
            or malformed_output
            or malformed_error
        ),
    }
    if recorded_accounting != accounting:
        write_json(accounting_path, accounting)
    if not accounting["exact"]:
        raise LifecycleError("custom-ID accounting failed; see accounting.json")

    metadata_by_id = {
        item["custom_id"]: item for item in _read_jsonl(root / "request_metadata.jsonl")
    }
    group_by_id = _map_groups_to_custom_ids(groups, metadata_by_id)
    schema_by_id = {
        custom_id: _recorded_schema_identity(config, metadata)
        for custom_id, metadata in metadata_by_id.items()
    }
    cache_identity = _collection_cache_identity(
        root,
        config,
        metadata_by_id,
        raw_paths,
        error_paths,
        record_timeout_seconds,
    )
    cache_path = root / "collection_layers.sqlite3"
    cache, cache_identity_hash = _open_collection_cache(cache_path, cache_identity)
    try:
        cached_ids = {
            row[0]
            for row in cache.execute("SELECT custom_id FROM collection_results")
        }
        output_ids = {item[0] for item in output_records}
        if not cached_ids <= output_ids:
            raise LifecycleError("collection cache contains an unknown custom ID")
        counts = dict(
            cache.execute(
                "SELECT status, COUNT(*) FROM collection_results GROUP BY status"
            ).fetchall()
        )
        cached_count = sum(counts.values())
        completed = cached_count
        validation_errors = counts.get("error", 0) + counts.get("timeout", 0)
        report_progress = _collection_progress_reporter(
            progress, len(output_records), cached_count, validation_errors
        )
        report_progress(completed, validation_errors, force=True)

        def tasks():
            for custom_id, row, source_file, line_number in _iter_raw_records(
                raw_paths
            ):
                if custom_id in cached_ids:
                    continue
                yield (
                    custom_id,
                    row,
                    source_file,
                    line_number,
                    metadata_by_id[custom_id],
                    group_by_id[custom_id],
                    schema_by_id[custom_id],
                    float(record_timeout_seconds),
                )

        buffer = []

        def flush():
            nonlocal completed, validation_errors
            if not buffer:
                return
            persisted = list(buffer)
            _persist_collection_batch(cache, persisted)
            buffer.clear()
            completed += len(persisted)
            validation_errors += sum(
                result["status"] != "valid" for result in persisted
            )
            report_progress(completed, validation_errors)

        def persist(result):
            buffer.append(result)
            if len(buffer) >= COLLECTION_COMMIT_BATCH_SIZE:
                flush()

        active_workers = min(workers, max(1, len(output_records) - cached_count))
        max_in_flight = max(active_workers, active_workers * 4)
        try:
            _run_collection_tasks(tasks(), active_workers, max_in_flight, persist)
        finally:
            flush()
        report_progress(completed, validation_errors, force=True)
        if completed != len(output_records):
            raise LifecycleError(
                "collection cache is incomplete: %d/%d"
                % (completed, len(output_records))
            )

        finished_identity = _collection_cache_identity(
            root,
            config,
            metadata_by_id,
            raw_paths,
            error_paths,
            record_timeout_seconds,
        )
        if finished_identity != cache_identity:
            raise LifecycleError(
                "collection inputs or validation implementation changed during collection"
            )

        _atomic_cache_jsonl(
            cache, root / "parsed_results.jsonl", "parsed_json", ("valid",)
        )
        _atomic_cache_jsonl(
            cache, root / "verified_results.jsonl", "verified_json", ("valid",)
        )
        _atomic_cache_jsonl(
            cache, root / "curated_results.jsonl", "curated_json", ("valid",)
        )
        _atomic_cache_jsonl(
            cache,
            root / "validation_errors.jsonl",
            "error_json",
            ("error", "timeout"),
        )
        final_counts = dict(
            cache.execute(
                "SELECT status, COUNT(*) FROM collection_results GROUP BY status"
            ).fetchall()
        )
        schema_valid = final_counts.get("valid", 0)
        schema_invalid = final_counts.get("error", 0) + final_counts.get("timeout", 0)
        report = dict(
            accounting,
            schema_valid=schema_valid,
            schema_invalid=schema_invalid,
            validation_timeouts=final_counts.get("timeout", 0),
            API_error_records=len(error_records),
            worker_count=workers,
            max_in_flight=max_in_flight,
            cached_records_reused=cached_count,
            cache_identity=cache_identity,
            cache_identity_sha256=cache_identity_hash,
            collection_cache="collection_layers.sqlite3",
        )
        write_json(root / "collection_report.json", report)
        cache.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return report
    finally:
        cache.close()


REVALIDATION_FORMAT_VERSION = "immutable_revalidation_v1"


def _validation_code_hashes():
    """Hash validation implementations while retaining distinct layer identities."""
    module_sha256 = sha256_file(Path(__file__).with_name("validation.py"))
    identity_module_sha256 = sha256_file(Path(__file__).with_name("identity.py"))
    return {
        "validation_module_sha256": module_sha256,
        "identity_module_sha256": identity_module_sha256,
        "validator_code_sha256": _sha256_json(
            {
                "entrypoints": ["validate_parsed_output", "verify_result"],
                "validation_module_sha256": module_sha256,
                "identity_module_sha256": identity_module_sha256,
            }
        ),
        "curator_code_sha256": _sha256_json(
            {
                "entrypoints": ["curate_result"],
                "validation_module_sha256": module_sha256,
            }
        ),
    }


def _existing_revalidation_report(version_dir, identity):
    """Return a matching immutable version, rejecting partial or changed artifacts."""
    report_path = version_dir / "report.json"
    if not report_path.exists():
        raise LifecycleError("immutable revalidation version is incomplete: %s" % version_dir)
    report = _read_json(report_path)
    if (
        report.get("format_version") != REVALIDATION_FORMAT_VERSION
        or report.get("identity") != identity
        or report.get("version_id") != version_dir.name
    ):
        raise LifecycleError("immutable revalidation identity mismatch: %s" % version_dir)
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"verified", "curated"}:
        raise LifecycleError("immutable revalidation report is incomplete: %s" % version_dir)
    for name, artifact in artifacts.items():
        expected_path = (
            Path("revalidations") / version_dir.name / ("%s_results.jsonl" % name)
        )
        if artifact.get("path") != str(expected_path):
            raise LifecycleError(
                "immutable revalidation artifact path mismatch: %s" % version_dir
            )
        path = version_dir.parent.parent / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise LifecycleError("immutable revalidation artifact mismatch: %s" % path)
    return report


def revalidate_run(run_dir):
    """Create a content-addressed verified/curated version from parsed results."""
    root, config, state = _load_run(run_dir)
    groups = _verify_sources(root, config)
    metadata = {
        item["custom_id"]: item for item in _read_jsonl(root / "request_metadata.jsonl")
    }
    group_by_group_id = {str(group["group_id"]): group for group in groups}
    parsed_path = root / "parsed_results.jsonl"
    parsed_sha256 = sha256_file(parsed_path)
    code_hashes = _validation_code_hashes()
    source_checksums = {
        str(key): value
        for key, value in sorted(config.get("source_sha256", {}).items())
    }
    identity_metadata = {
        custom_id: {
            "identity_graph_sha256": item.get(
                "identity_graph_sha256", "legacy_identity_grounding"
            ),
            "identity_policy_sha256": item.get(
                "identity_policy_sha256", "legacy_identity_grounding"
            ),
        }
        for custom_id, item in sorted(metadata.items())
    }
    schema_metadata = {
        custom_id: _recorded_schema_identity(config, item)
        for custom_id, item in sorted(metadata.items())
    }
    schema_contracts = {
        (item["version"], item["sha256"]) for item in schema_metadata.values()
    }
    if len(schema_contracts) != 1:
        raise LifecycleError("a run must have one recorded schema contract")
    recorded_schema_version, recorded_schema_sha256 = next(iter(schema_contracts))
    prompt_metadata = {
        custom_id: {
            "prompt_version": item.get(
                "prompt_version", config.get("prompt_version", "unrecorded")
            ),
            "system_prompt_sha256": item.get(
                "system_prompt_sha256", config.get("prompt_sha256", "unrecorded")
            ),
            "user_prompt_sha256": item.get("user_prompt_sha256", "unrecorded"),
        }
        for custom_id, item in sorted(metadata.items())
    }
    identity_graph_hashes = {
        custom_id: item["identity_graph_sha256"]
        for custom_id, item in identity_metadata.items()
    }
    identity_policy_hashes = {
        custom_id: item["identity_policy_sha256"]
        for custom_id, item in identity_metadata.items()
    }
    identity = {
        "format_version": REVALIDATION_FORMAT_VERSION,
        "validator_code_sha256": code_hashes["validator_code_sha256"],
        "curator_code_sha256": code_hashes["curator_code_sha256"],
        "validation_code_sha256": _sha256_json(code_hashes),
        "schema_version": recorded_schema_version,
        "schema_sha256": recorded_schema_sha256,
        "prompt_version": config.get("prompt_version", "unrecorded"),
        "prompt_sha256": config.get("prompt_sha256", "unrecorded"),
        "request_prompt_metadata_sha256": _sha256_json(prompt_metadata),
        "source_checksums_sha256": _sha256_json(source_checksums),
        "identity_metadata_sha256": _sha256_json(identity_metadata),
        "identity_graph_hashes_sha256": _sha256_json(identity_graph_hashes),
        "identity_policy_hashes_sha256": _sha256_json(identity_policy_hashes),
        "parsed_results_sha256": parsed_sha256,
        "parsed_checksum": parsed_sha256,
    }
    version_id = _sha256_json(identity)
    versions_root = root / "revalidations"
    version_dir = versions_root / version_id
    if version_dir.exists():
        return _existing_revalidation_report(version_dir, identity)

    parsed_rows = _read_jsonl(parsed_path)
    verified_rows, curated_rows = [], []
    for row in parsed_rows:
        custom_id = row["custom_id"]
        group_id = str(row["group_id"])
        if custom_id not in metadata or group_id not in group_by_group_id:
            raise LifecycleError("parsed result has unknown identity: %s" % custom_id)
        schema_identity = schema_metadata[custom_id]
        validate_parsed_output(
            row["result"],
            schema_version=schema_identity["version"],
            expected_schema_sha256=schema_identity["sha256"],
        )
        passages = metadata[custom_id]["selected_passages"]
        verified = verify_result(
            row["result"],
            passages,
            group_by_group_id[group_id],
            identity_graph=metadata[custom_id].get("identity_graph"),
        )
        curated = curate_result(verified)
        base = {"custom_id": custom_id, "group_id": group_id}
        verified_rows.append(dict(base, result=verified))
        curated_rows.append(dict(base, result=curated))

    versions_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".%s." % version_id, dir=str(versions_root)
    ) as temporary:
        staging = Path(temporary)
        verified_path = staging / "verified_results.jsonl"
        curated_path = staging / "curated_results.jsonl"
        write_jsonl(verified_path, verified_rows, lambda row: row["custom_id"])
        write_jsonl(curated_path, curated_rows, lambda row: row["custom_id"])
        relative_dir = Path("revalidations") / version_id
        report = {
            "format_version": REVALIDATION_FORMAT_VERSION,
            "version_id": version_id,
            "validation_directory": str(relative_dir),
            "revalidated": len(parsed_rows),
            "source_checksums_verified": len(groups),
            "identity": identity,
            "code": code_hashes,
            "lineage": {
                "run_config": {
                    "path": "run_config.json",
                    "sha256": state["run_config_sha256"],
                },
                "parsed_results": {
                    "path": "parsed_results.jsonl",
                    "sha256": parsed_sha256,
                },
                "manifest": {
                    "path": "manifest.jsonl",
                    "sha256": config["manifest_copy_sha256"],
                },
                "request_metadata": {
                    "path": "request_metadata.jsonl",
                    "sha256": config["request_metadata_sha256"],
                },
                "prompt": {
                    "version": identity["prompt_version"],
                    "sha256": config.get("prompt_sha256", "unrecorded"),
                    "request_metadata_sha256": identity[
                        "request_prompt_metadata_sha256"
                    ],
                },
                "schema": {
                    "version": recorded_schema_version,
                    "sha256": identity["schema_sha256"],
                    "prepared_run_sha256": config.get(
                        "schema_sha256", "unrecorded"
                    ),
                },
                "sources": {
                    "checksums": source_checksums,
                    "sha256": identity["source_checksums_sha256"],
                },
                "identity_graphs": {
                    "sha256": identity["identity_graph_hashes_sha256"],
                    "per_request": identity_graph_hashes,
                },
                "identity_policies": {
                    "sha256": identity["identity_policy_hashes_sha256"],
                    "per_request": identity_policy_hashes,
                },
            },
            "artifacts": {
                "verified": {
                    "path": str(relative_dir / "verified_results.jsonl"),
                    "sha256": sha256_file(verified_path),
                },
                "curated": {
                    "path": str(relative_dir / "curated_results.jsonl"),
                    "sha256": sha256_file(curated_path),
                },
            },
        }
        write_json(staging / "report.json", report)
        try:
            os.rename(staging, version_dir)
        except OSError:
            if not version_dir.exists():
                raise
            return _existing_revalidation_report(version_dir, identity)
    return report
