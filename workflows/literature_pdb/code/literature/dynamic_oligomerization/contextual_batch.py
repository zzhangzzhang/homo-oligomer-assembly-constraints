"""Durable OpenAI Batch lifecycle for prepared contextual v6 runs."""

import hashlib
import importlib
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contextual_pilot import (
    _load_pilot,
    _safe_response_name,
    _scan_terminal_records,
)
from .lifecycle import OpenAIBatchClient
from .hashing import sha256_file
from .serialization import canonical_json
from .api_credentials import load_api_key


CONTEXTUAL_BATCH_LIFECYCLE_VERSION = "contextual_openai_batch_v1"
DEFAULT_CONTEXTUAL_BATCH_MAX_REQUESTS = 50_000
DEFAULT_CONTEXTUAL_BATCH_MAX_BYTES = 190 * 1024 * 1024
CONTEXTUAL_BATCH_ENDPOINT = "/v1/chat/completions"
CONTEXTUAL_BATCH_COMPLETION_WINDOW = "24h"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class ContextualBatchError(RuntimeError):
    """Raised when a contextual Batch artifact or transition is unsafe."""


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


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
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(str(destination.parent), directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path, value):
    _atomic_bytes(
        path,
        (canonical_json(value, indent=2) + "\n").encode("utf-8"),
    )


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise ContextualBatchError("cannot read JSON artifact %s: %s" % (path, error))
    if not isinstance(value, dict):
        raise ContextualBatchError("JSON artifact is not an object: %s" % path)
    return value


def _immutable_bytes(path, content):
    destination = Path(path)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise ContextualBatchError(
                "immutable contextual Batch artifact differs: %s" % destination
            )
        return
    _atomic_bytes(destination, content)


@contextmanager
def _batch_lock(root):
    try:
        import fcntl
    except ImportError:
        yield
        return
    path = Path(root) / ".context_batch.lock"
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _request_line(request):
    return (canonical_json(request) + "\n").encode("utf-8")


def _verify_contextual_requests(run_dir):
    root, pilot_config, requests, metadata, _ = _load_pilot(run_dir)
    expected_model = str(pilot_config.get("model") or "")
    expected_prompt = str(pilot_config.get("prompt_sha256") or "")
    expected_schema = str(pilot_config.get("schema_sha256") or "")
    if not expected_model or not expected_prompt or not expected_schema:
        raise ContextualBatchError(
            "prepared contextual run lacks model/prompt/schema identity"
        )
    for custom_id, request in requests.items():
        _safe_response_name(custom_id)
        if (
            request.get("custom_id") != custom_id
            or request.get("method") != "POST"
            or request.get("url") != CONTEXTUAL_BATCH_ENDPOINT
            or not isinstance(request.get("body"), dict)
        ):
            raise ContextualBatchError(
                "invalid contextual Batch request envelope: %s" % custom_id
            )
        body = request["body"]
        item = metadata[custom_id]
        messages = body.get("messages")
        system_prompt = (
            messages[0].get("content")
            if isinstance(messages, list)
            and messages
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
            else None
        )
        if (
            body.get("model") != expected_model
            or item.get("model") != expected_model
            or item.get("schema_sha256") != expected_schema
            or item.get("system_prompt_sha256") != expected_prompt
            or not isinstance(system_prompt, str)
            or _sha256_bytes(system_prompt.encode("utf-8")) != expected_prompt
            or item.get("request_body_sha256") != _sha256_json(body)
            or item.get("request_sha256") != _sha256_json(request)
        ):
            raise ContextualBatchError(
                "contextual request provenance mismatch: %s" % custom_id
            )
    return root, pilot_config, requests, metadata


def _partition_requests(requests, max_requests, max_bytes):
    shards = []
    current = []
    current_bytes = 0
    for custom_id in sorted(requests):
        request = requests[custom_id]
        line = _request_line(request)
        if len(line) > max_bytes:
            raise ContextualBatchError(
                "request %s is %d bytes, exceeding shard limit %d"
                % (custom_id, len(line), max_bytes)
            )
        if current and (
            len(current) >= max_requests or current_bytes + len(line) > max_bytes
        ):
            shards.append((current, current_bytes))
            current = []
            current_bytes = 0
        current.append((custom_id, request, line))
        current_bytes += len(line)
    if current:
        shards.append((current, current_bytes))
    return shards


def _initial_state(config_path, config):
    return {
        "lifecycle_version": CONTEXTUAL_BATCH_LIFECYCLE_VERSION,
        "batch_config_sha256": sha256_file(config_path),
        "batch_identity_sha256": config["batch_identity_sha256"],
        "shards": {
            shard["name"]: {
                "upload_id": None,
                "submission_intent": None,
                "batch_id": None,
                "submitted_at": None,
                "last_status": None,
                "request_counts": None,
                "output_file_id": None,
                "error_file_id": None,
                "raw_output": None,
                "raw_error": None,
            }
            for shard in config["shards"]
        },
    }


def prepare_contextual_batch(
    run_dir,
    max_requests=DEFAULT_CONTEXTUAL_BATCH_MAX_REQUESTS,
    max_bytes=DEFAULT_CONTEXTUAL_BATCH_MAX_BYTES,
):
    """Shard an immutable contextual pilot's already-prepared requests offline."""
    max_requests = int(max_requests)
    max_bytes = int(max_bytes)
    if max_requests < 1 or max_requests > DEFAULT_CONTEXTUAL_BATCH_MAX_REQUESTS:
        raise ValueError("max_requests must be between 1 and 50000")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    root, pilot_config, requests, metadata = _verify_contextual_requests(run_dir)
    if not requests:
        raise ContextualBatchError("prepared contextual run has no requests")
    config_path = root / "batch_config.json"
    state_path = root / "batch_state.json"
    with _batch_lock(root):
        if config_path.exists():
            existing_config = _read_json(config_path)
            limits = existing_config.get("limits", {})
            if (
                limits.get("max_requests_per_shard") != max_requests
                or limits.get("max_serialized_bytes_per_shard") != max_bytes
            ):
                raise ContextualBatchError(
                    "immutable batch_config.json was prepared with different limits"
                )
            if state_path.exists():
                _, config, _ = _load_contextual_batch(root)
                return config
        elif state_path.exists():
            raise ContextualBatchError(
                "batch_state.json exists without immutable batch_config.json"
            )

        partitions = _partition_requests(requests, max_requests, max_bytes)
        shard_records = []
        accounted_ids = []
        for index, (items, serialized_bytes) in enumerate(partitions):
            name = "shard-%05d" % index
            relative = Path("batch_shards") / name / "requests.jsonl"
            content = b"".join(item[2] for item in items)
            if len(content) != serialized_bytes:
                raise ContextualBatchError("serialized shard byte accounting failed")
            _immutable_bytes(root / relative, content)
            custom_ids = [item[0] for item in items]
            accounted_ids.extend(custom_ids)
            shard_records.append(
                {
                    "index": index,
                    "name": name,
                    "request_file": str(relative),
                    "request_count": len(items),
                    "serialized_bytes": serialized_bytes,
                    "request_sha256": _sha256_bytes(content),
                    "custom_ids": custom_ids,
                }
            )
        expected_ids = sorted(requests)
        if accounted_ids != expected_ids or len(set(accounted_ids)) != len(expected_ids):
            raise ContextualBatchError("contextual Batch shard accounting mismatch")

        model = pilot_config["model"]
        base_config = {
            "lifecycle_version": CONTEXTUAL_BATCH_LIFECYCLE_VERSION,
            "source_lifecycle_version": pilot_config["lifecycle_version"],
            "source_run_config_sha256": sha256_file(root / "run_config.json"),
            "source_pilot_integrity_sha256": sha256_file(
                root / "pilot_integrity.json"
            ),
            "source_requests_sha256": sha256_file(root / "requests.jsonl"),
            "source_request_metadata_sha256": sha256_file(
                root / "request_metadata.jsonl"
            ),
            "model": model,
            "model_sha256": _sha256_bytes(model.encode("utf-8")),
            "prompt_version": pilot_config["prompt_version"],
            "prompt_sha256": pilot_config["prompt_sha256"],
            "schema_version": pilot_config["schema_version"],
            "schema_sha256": pilot_config["schema_sha256"],
            "request_provenance_sha256": _sha256_json(
                {
                    custom_id: {
                        "request_sha256": metadata[custom_id]["request_sha256"],
                        "request_body_sha256": metadata[custom_id][
                            "request_body_sha256"
                        ],
                        "user_prompt_sha256": metadata[custom_id][
                            "user_prompt_sha256"
                        ],
                    }
                    for custom_id in expected_ids
                }
            ),
            "endpoint": CONTEXTUAL_BATCH_ENDPOINT,
            "completion_window": CONTEXTUAL_BATCH_COMPLETION_WINDOW,
            "request_count": len(expected_ids),
            "limits": {
                "max_requests_per_shard": max_requests,
                "max_serialized_bytes_per_shard": max_bytes,
            },
            "shards": shard_records,
        }
        config = dict(base_config)
        config["batch_identity_sha256"] = _sha256_json(base_config)
        _immutable_bytes(
            config_path,
            (canonical_json(config, indent=2) + "\n").encode("utf-8"),
        )
        _atomic_json(state_path, _initial_state(config_path, config))
        return config


def _load_contextual_batch(run_dir):
    root, pilot_config, requests, _ = _verify_contextual_requests(run_dir)
    config_path = root / "batch_config.json"
    state_path = root / "batch_state.json"
    config = _read_json(config_path)
    state = _read_json(state_path)
    if (
        config.get("lifecycle_version") != CONTEXTUAL_BATCH_LIFECYCLE_VERSION
        or state.get("lifecycle_version") != CONTEXTUAL_BATCH_LIFECYCLE_VERSION
    ):
        raise ContextualBatchError("unsupported contextual Batch lifecycle")
    if state.get("batch_config_sha256") != sha256_file(config_path):
        raise ContextualBatchError("batch_config.json checksum mismatch")
    base_config = dict(config)
    recorded_identity = base_config.pop("batch_identity_sha256", None)
    if (
        not recorded_identity
        or recorded_identity != _sha256_json(base_config)
        or state.get("batch_identity_sha256") != recorded_identity
    ):
        raise ContextualBatchError("contextual Batch identity mismatch")
    source_checks = {
        "source_run_config_sha256": root / "run_config.json",
        "source_pilot_integrity_sha256": root / "pilot_integrity.json",
        "source_requests_sha256": root / "requests.jsonl",
        "source_request_metadata_sha256": root / "request_metadata.jsonl",
    }
    for field, path in source_checks.items():
        if config.get(field) != sha256_file(path):
            raise ContextualBatchError(
                "prepared contextual source checksum mismatch: %s" % path
            )
    if (
        config.get("model") != pilot_config.get("model")
        or config.get("model_sha256")
        != _sha256_bytes(str(pilot_config.get("model") or "").encode("utf-8"))
        or config.get("prompt_sha256") != pilot_config.get("prompt_sha256")
        or config.get("schema_sha256") != pilot_config.get("schema_sha256")
        or config.get("endpoint") != CONTEXTUAL_BATCH_ENDPOINT
        or config.get("completion_window") != CONTEXTUAL_BATCH_COMPLETION_WINDOW
    ):
        raise ContextualBatchError("contextual Batch contract identity mismatch")
    shards = config.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContextualBatchError("batch_config.json has no shards")
    if set(state.get("shards", {})) != {shard.get("name") for shard in shards}:
        raise ContextualBatchError("batch state/config shard mismatch")
    accounted = []
    for expected_index, shard in enumerate(shards):
        if (
            shard.get("index") != expected_index
            or shard.get("name") != "shard-%05d" % expected_index
        ):
            raise ContextualBatchError("noncanonical contextual Batch shard ordering")
        path = root / shard["request_file"]
        if (
            not path.is_file()
            or path.stat().st_size != shard["serialized_bytes"]
            or sha256_file(path) != shard["request_sha256"]
        ):
            raise ContextualBatchError("prepared request checksum mismatch: %s" % path)
        rows = []
        with path.open(encoding="utf-8", newline="") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.endswith("\n") or not raw.strip():
                    raise ContextualBatchError(
                        "invalid shard JSONL at %s:%d" % (path, line_number)
                    )
                value = json.loads(raw)
                if raw[:-1] != canonical_json(value):
                    raise ContextualBatchError(
                        "noncanonical shard JSONL at %s:%d" % (path, line_number)
                    )
                rows.append(value)
        ids = [row.get("custom_id") for row in rows]
        if (
            ids != shard.get("custom_ids")
            or len(rows) != shard.get("request_count")
            or any(requests.get(custom_id) != row for custom_id, row in zip(ids, rows))
        ):
            raise ContextualBatchError("contextual Batch shard contents mismatch")
        accounted.extend(ids)
    if (
        accounted != sorted(requests)
        or len(accounted) != config.get("request_count")
        or len(set(accounted)) != len(accounted)
    ):
        raise ContextualBatchError("contextual Batch exact request accounting failed")
    return root, config, state


def _open_batch_client(api_key_file):
    api_key = load_api_key(config_path=api_key_file)
    try:
        OpenAI = importlib.import_module("openai").OpenAI
    except ImportError as error:
        raise RuntimeError(
            "contextual Batch API actions require the optional openai package"
        ) from error
    return OpenAIBatchClient(OpenAI(api_key=api_key, max_retries=0))


def _submission_summary(root, config, state):
    batch_ids = {
        shard["name"]: state["shards"][shard["name"]].get("batch_id")
        for shard in config["shards"]
    }
    return {
        "run_dir": str(root),
        "shard_count": len(config["shards"]),
        "submitted_shards": sum(value is not None for value in batch_ids.values()),
        "unsubmitted_shards": sum(value is None for value in batch_ids.values()),
        "batch_ids": batch_ids,
    }


def _validate_reconciled_batch(batch, intent):
    batch_id = _field(batch, "id")
    if not batch_id:
        raise ContextualBatchError("reconciled Batch has no id")
    metadata = _field(batch, "metadata", {}) or {}
    if any(
        str(_field(metadata, key, "")) != str(value)
        for key, value in intent["metadata"].items()
    ):
        raise ContextualBatchError("reconciled Batch metadata mismatch")
    checks = (
        ("input_file_id", intent["upload_id"]),
        ("endpoint", intent["endpoint"]),
        ("completion_window", intent["completion_window"]),
    )
    for field, expected in checks:
        actual = _field(batch, field)
        if actual is not None and actual != expected:
            raise ContextualBatchError("reconciled Batch %s mismatch" % field)
    return batch_id


def submit_contextual_batch(run_dir, client=None, api_key_file=None):
    """Upload and submit each shard using durable intent and reconciliation."""
    root, config, state = _load_contextual_batch(run_dir)
    with _batch_lock(root):
        root, config, state = _load_contextual_batch(root)
        if all(
            state["shards"][shard["name"]].get("batch_id")
            for shard in config["shards"]
        ):
            return _submission_summary(root, config, state)
        expected_ids = {
            custom_id
            for shard in config["shards"]
            for custom_id in shard["custom_ids"]
        }
        existing_responses, existing_errors = _scan_terminal_records(
            root, expected_ids
        )
        if existing_responses or existing_errors:
            raise ContextualBatchError(
                "refusing paid Batch submission because contextual terminal "
                "records already exist"
            )
        active_client = client or _open_batch_client(api_key_file)
        for shard in config["shards"]:
            shard_state = state["shards"][shard["name"]]
            if shard_state.get("batch_id"):
                continue
            if not shard_state.get("upload_id"):
                shard_state["upload_id"] = active_client.upload_file(
                    root / shard["request_file"]
                )
                _atomic_json(root / "batch_state.json", state)
            expected_intent = {
                "submission_key": _sha256_json(
                    {
                        "batch_identity_sha256": config["batch_identity_sha256"],
                        "shard": shard["name"],
                        "request_sha256": shard["request_sha256"],
                        "upload_id": shard_state["upload_id"],
                    }
                )[:40],
                "endpoint": CONTEXTUAL_BATCH_ENDPOINT,
                "completion_window": CONTEXTUAL_BATCH_COMPLETION_WINDOW,
                "upload_id": shard_state["upload_id"],
                "metadata": {
                    "run": root.name[:64],
                    "run_id": config["batch_identity_sha256"][:40],
                    "shard": shard["name"],
                    "lifecycle": CONTEXTUAL_BATCH_LIFECYCLE_VERSION,
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
                # This fsync-backed durable write must precede the paid create call.
                _atomic_json(root / "batch_state.json", state)
            else:
                for field in (
                    "submission_key",
                    "endpoint",
                    "completion_window",
                    "upload_id",
                    "metadata",
                ):
                    if intent.get(field) != expected_intent[field]:
                        raise ContextualBatchError(
                            "persisted submission intent changed for %s"
                            % shard["name"]
                        )
            finder = getattr(active_client, "find_batch_by_metadata", None)
            if intent_preexisting and not callable(finder):
                raise ContextualBatchError(
                    "cannot safely resume %s without Batch metadata reconciliation"
                    % shard["name"]
                )
            batch = finder(intent["metadata"]) if callable(finder) else None
            reconciled = batch is not None
            if batch is None:
                batch = active_client.create_batch(
                    shard_state["upload_id"],
                    intent["endpoint"],
                    intent["completion_window"],
                    intent["metadata"],
                )
            batch_id = _validate_reconciled_batch(batch, intent)
            shard_state.update(
                {
                    "batch_id": batch_id,
                    "submitted_at": shard_state.get("submitted_at") or _utc_now(),
                    "last_status": _field(batch, "status"),
                    "output_file_id": _field(batch, "output_file_id"),
                    "error_file_id": _field(batch, "error_file_id"),
                }
            )
            if reconciled:
                shard_state["reconciled_at"] = _utc_now()
            _atomic_json(root / "batch_state.json", state)
        return _submission_summary(root, config, state)


def status_contextual_batch(run_dir, client=None, api_key_file=None):
    """Refresh per-shard state and return exact aggregate Batch accounting."""
    root, config, state = _load_contextual_batch(run_dir)
    with _batch_lock(root):
        root, config, state = _load_contextual_batch(root)
        active_client = client
        summary = {
            "expected": config["request_count"],
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
                    {
                        "name": shard["name"],
                        "status": "unsubmitted",
                        "expected": shard["request_count"],
                        "completed": 0,
                        "failed": 0,
                        "pending": 0,
                        "unsubmitted": shard["request_count"],
                    }
                )
                continue
            if active_client is None:
                active_client = _open_batch_client(api_key_file)
            batch = active_client.retrieve_batch(shard_state["batch_id"])
            counts = _field(batch, "request_counts", {}) or {}
            completed = int(_field(counts, "completed", 0) or 0)
            failed = int(_field(counts, "failed", 0) or 0)
            if completed < 0 or failed < 0 or completed + failed > shard["request_count"]:
                raise ContextualBatchError(
                    "remote request counts exceed shard accounting: %s"
                    % shard["name"]
                )
            pending = shard["request_count"] - completed - failed
            status = str(_field(batch, "status", "unknown") or "unknown")
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
            summary["completed"] += completed
            summary["failed"] += failed
            summary["pending"] += pending
            summary["shards"].append(
                {
                    "name": shard["name"],
                    "batch_id": shard_state["batch_id"],
                    "status": status,
                    "terminal": status in TERMINAL_BATCH_STATUSES,
                    "expected": shard["request_count"],
                    "completed": completed,
                    "failed": failed,
                    "pending": pending,
                    "unsubmitted": 0,
                    "output_file_id": shard_state["output_file_id"],
                    "error_file_id": shard_state["error_file_id"],
                }
            )
        if (
            summary["completed"]
            + summary["failed"]
            + summary["pending"]
            + summary["unsubmitted"]
            != summary["expected"]
        ):
            raise ContextualBatchError("aggregate Batch status accounting failed")
        _atomic_json(root / "batch_state.json", state)
        _atomic_json(root / "batch_status.json", summary)
        return summary


def _ensure_download(client, file_id, path, recorded):
    destination = Path(path)
    if recorded:
        if recorded.get("file_id") != file_id or recorded.get("path") != str(
            destination.relative_to(destination.parents[2])
        ):
            raise ContextualBatchError(
                "recorded raw Batch artifact identity mismatch: %s" % destination
            )
        if (
            not destination.is_file()
            or destination.stat().st_size != recorded.get("bytes")
            or sha256_file(destination) != recorded.get("sha256")
        ):
            raise ContextualBatchError(
                "downloaded raw Batch artifact checksum mismatch: %s" % destination
            )
        return recorded
    if client is None:
        raise ContextualBatchError(
            "raw Batch artifact is not durably recorded: %s" % destination
        )
    content = client.download_file(file_id)
    if not isinstance(content, bytes):
        raise ContextualBatchError("downloaded Batch artifact is not bytes")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise ContextualBatchError(
                "unrecorded raw Batch artifact differs from remote: %s" % destination
            )
    else:
        _atomic_bytes(destination, content)
    root = destination.parents[2]
    return {
        "file_id": file_id,
        "path": str(destination.relative_to(root)),
        "bytes": len(content),
        "sha256": _sha256_bytes(content),
    }


def _parse_raw_jsonl(path, shard_name, kind, shard_ids):
    records = []
    malformed = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith("\n") or not raw.strip():
                malformed.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "error": "line must be nonblank and newline terminated",
                    }
                )
                continue
            try:
                value = json.loads(raw)
            except ValueError as error:
                malformed.append(
                    {"file": str(path), "line": line_number, "error": str(error)}
                )
                continue
            custom_id = value.get("custom_id") if isinstance(value, dict) else None
            if not isinstance(custom_id, str):
                malformed.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "error": "record lacks string custom_id",
                    }
                )
                continue
            records.append(
                {
                    "custom_id": custom_id,
                    "record": value,
                    "kind": kind,
                    "shard": shard_name,
                    "in_expected_shard": custom_id in shard_ids,
                }
            )
    return records, malformed


def _error_text(error):
    if isinstance(error, dict):
        return str(
            error.get("message")
            or error.get("code")
            or error.get("type")
            or canonical_json(error)
        )
    return str(error or "Batch request failed")


def _sanitize_error(value):
    return re.sub(
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        "[REDACTED_API_KEY]",
        str(value),
    )[:4000]


def _api_error_record(custom_id, row):
    response = row.get("response") if isinstance(row, dict) else None
    response = response if isinstance(response, dict) else {}
    status_code = response.get("status_code")
    body = response.get("body")
    body_error = body.get("error") if isinstance(body, dict) else None
    error = row.get("error") if isinstance(row, dict) else None
    detail = error or body_error or body or "Batch request failed"
    code = (
        detail.get("code") or detail.get("type")
        if isinstance(detail, dict)
        else None
    )
    try:
        numeric_status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        numeric_status = None
    return {
        "custom_id": custom_id,
        "error_type": str(code or "BatchRequestError"),
        "status_code": numeric_status,
        "transient": numeric_status in TRANSIENT_STATUS_CODES,
        "attempts": 1,
        "error": _sanitize_error(_error_text(detail)),
        "request_id": response.get("request_id"),
        "batch_error": _sanitize_error(
            canonical_json(detail)
            if isinstance(detail, (dict, list))
            else detail
        ),
    }


def _desired_terminal_record(item):
    custom_id = item["custom_id"]
    row = item["record"]
    if item["kind"] == "output":
        response = row.get("response") if isinstance(row, dict) else None
        status_code = response.get("status_code") if isinstance(response, dict) else None
        body = response.get("body") if isinstance(response, dict) else None
        if status_code == 200 and isinstance(body, dict) and not row.get("error"):
            return "response", {"custom_id": custom_id, "response": body}
    return "api_error", _api_error_record(custom_id, row)


def _usage_totals(records):
    totals = Counter()
    records_with_usage = 0
    for record in records:
        response = record.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            continue
        records_with_usage += 1
        totals["input_tokens"] += int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        totals["output_tokens"] += int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = usage.get("input_tokens_details")
        totals["cached_input_tokens"] += int(
            (details or {}).get("cached_tokens", 0) or 0
        )
    return {
        "records_with_usage": records_with_usage,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "cached_input_tokens": totals["cached_input_tokens"],
    }


def collect_contextual_batch(run_dir, client=None, api_key_file=None):
    """Collect terminal Batch files into contextual pilot-compatible records."""
    root, config, state = _load_contextual_batch(run_dir)
    with _batch_lock(root):
        root, config, state = _load_contextual_batch(root)
        active_client = client
        for shard in config["shards"]:
            shard_state = state["shards"][shard["name"]]
            if not shard_state.get("batch_id"):
                raise ContextualBatchError(
                    "cannot collect unsubmitted shard %s" % shard["name"]
                )
            if active_client is not None or shard_state.get(
                "last_status"
            ) not in TERMINAL_BATCH_STATUSES:
                if active_client is None:
                    active_client = _open_batch_client(api_key_file)
                batch = active_client.retrieve_batch(shard_state["batch_id"])
                shard_state.update(
                    {
                        "last_status": _field(batch, "status"),
                        "status_checked_at": _utc_now(),
                        "output_file_id": _field(batch, "output_file_id"),
                        "error_file_id": _field(batch, "error_file_id"),
                    }
                )
        _atomic_json(root / "batch_state.json", state)
        nonterminal = [
            shard["name"]
            for shard in config["shards"]
            if state["shards"][shard["name"]].get("last_status")
            not in TERMINAL_BATCH_STATUSES
        ]
        if nonterminal:
            raise ContextualBatchError(
                "cannot collect nonterminal shards: %s" % ",".join(nonterminal)
            )

        raw_paths = []
        for shard in config["shards"]:
            shard_state = state["shards"][shard["name"]]
            shard_dir = root / "batch_shards" / shard["name"]
            for file_field, state_field, filename, kind in (
                ("output_file_id", "raw_output", "raw_output.jsonl", "output"),
                ("error_file_id", "raw_error", "raw_error.jsonl", "error"),
            ):
                file_id = shard_state.get(file_field)
                if not file_id:
                    if shard_state.get(state_field):
                        raise ContextualBatchError(
                            "raw artifact recorded without remote file ID"
                        )
                    continue
                if shard_state.get(state_field) is None and active_client is None:
                    active_client = _open_batch_client(api_key_file)
                artifact = _ensure_download(
                    active_client,
                    file_id,
                    shard_dir / filename,
                    shard_state.get(state_field),
                )
                shard_state[state_field] = artifact
                _atomic_json(root / "batch_state.json", state)
                raw_paths.append(
                    (shard, kind, root / artifact["path"], artifact["sha256"])
                )

        parsed = []
        malformed = []
        for shard, kind, path, _ in raw_paths:
            rows, errors = _parse_raw_jsonl(
                path, shard["name"], kind, set(shard["custom_ids"])
            )
            parsed.extend(rows)
            malformed.extend(errors)
        expected = {
            custom_id
            for shard in config["shards"]
            for custom_id in shard["custom_ids"]
        }
        counts = Counter(item["custom_id"] for item in parsed)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        unexpected = sorted(set(counts) - expected)
        wrong_shard = sorted(
            {
                "%s:%s" % (item["shard"], item["custom_id"])
                for item in parsed
                if not item["in_expected_shard"]
            }
        )
        missing = sorted(expected - set(counts))
        accounting = {
            "expected": len(expected),
            "raw_output_records": sum(item["kind"] == "output" for item in parsed),
            "raw_error_records": sum(item["kind"] == "error" for item in parsed),
            "missing_custom_ids": missing,
            "duplicate_custom_ids": duplicates,
            "unexpected_custom_ids": unexpected,
            "wrong_shard_custom_ids": wrong_shard,
            "malformed_records": malformed,
            "exact": not (
                missing or duplicates or unexpected or wrong_shard or malformed
            ),
        }
        _atomic_json(root / "batch_accounting.json", accounting)
        if not accounting["exact"]:
            raise ContextualBatchError(
                "custom-ID accounting failed; see batch_accounting.json"
            )

        desired_responses = {}
        desired_errors = {}
        for item in parsed:
            kind, record = _desired_terminal_record(item)
            destination = desired_responses if kind == "response" else desired_errors
            destination[item["custom_id"]] = record
        existing_responses, existing_errors = _scan_terminal_records(root, expected)
        for custom_id, desired in desired_responses.items():
            if custom_id in existing_errors:
                raise ContextualBatchError(
                    "existing API error conflicts with Batch response: %s" % custom_id
                )
            if custom_id in existing_responses and existing_responses[custom_id] != desired:
                raise ContextualBatchError(
                    "existing terminal response differs: %s" % custom_id
                )
        for custom_id, desired in desired_errors.items():
            if custom_id in existing_responses:
                raise ContextualBatchError(
                    "existing response conflicts with Batch API error: %s" % custom_id
                )
            if custom_id in existing_errors and existing_errors[custom_id] != desired:
                raise ContextualBatchError(
                    "existing terminal API error differs: %s" % custom_id
                )
        for custom_id, desired in sorted(desired_responses.items()):
            if custom_id not in existing_responses:
                _atomic_json(
                    root / "raw_responses" / _safe_response_name(custom_id),
                    desired,
                )
        for custom_id, desired in sorted(desired_errors.items()):
            if custom_id not in existing_errors:
                _atomic_json(
                    root / "api_errors" / _safe_response_name(custom_id),
                    desired,
                )

        report = dict(
            accounting,
            successful_responses=len(desired_responses),
            api_errors=len(desired_errors),
            usage=_usage_totals(desired_responses.values()),
            raw_artifacts=[
                {
                    "shard": shard["name"],
                    "kind": kind,
                    "path": str(path.relative_to(root)),
                    "sha256": digest,
                }
                for shard, kind, path, digest in raw_paths
            ],
            terminal_records_preserved=True,
        )
        _atomic_json(root / "batch_collection_report.json", report)
        return report


__all__ = [
    "CONTEXTUAL_BATCH_COMPLETION_WINDOW",
    "CONTEXTUAL_BATCH_ENDPOINT",
    "CONTEXTUAL_BATCH_LIFECYCLE_VERSION",
    "DEFAULT_CONTEXTUAL_BATCH_MAX_BYTES",
    "DEFAULT_CONTEXTUAL_BATCH_MAX_REQUESTS",
    "ContextualBatchError",
    "collect_contextual_batch",
    "prepare_contextual_batch",
    "status_contextual_batch",
    "submit_contextual_batch",
]
