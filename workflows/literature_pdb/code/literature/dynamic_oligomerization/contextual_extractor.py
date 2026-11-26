"""Pure request construction for contextual paper-level v6 extraction.

No function in this module performs API, network, lifecycle, or filesystem I/O.
The caller must explicitly supply the model selected and frozen elsewhere.
"""

import hashlib
import re

from .contextual_prompt import (
    CONTEXTUAL_PROMPT_VERSION,
    CONTEXTUAL_SYSTEM_PROMPT,
    contextual_system_prompt_sha256,
    render_contextual_user_prompt,
)
from .contextual_schema import (
    CONTEXTUAL_OUTPUT_SCHEMA,
    CONTEXTUAL_SCHEMA_VERSION,
    contextual_schema_sha256,
)
from .doi import normalize_doi
from .serialization import canonical_json


SUPPORTED_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DEFAULT_REASONING_EFFORT = "medium"


def _sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def contextual_custom_id(doi, dossier_sha256):
    """Derive a stable batch ID from normalized DOI and dossier content hash."""
    normalized_doi = normalize_doi(doi)
    dossier_sha256 = str(dossier_sha256 or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", dossier_sha256):
        raise ValueError("dossier_sha256 must be a SHA-256 digest")
    digest = hashlib.sha256(
        (normalized_doi + "\n" + dossier_sha256).encode("utf-8")
    ).hexdigest()[:24]
    return "dynamic-paper-v6-" + digest


def make_contextual_request_body(
    dossier,
    model,
    max_completion_tokens,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
):
    """Build one deterministic OpenAI-compatible chat-completions body."""
    model = str(model or "").strip()
    if not model:
        raise ValueError("an exact model ID is required")
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of: %s"
            % ", ".join(SUPPORTED_REASONING_EFFORTS)
        )
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ValueError("max_completion_tokens must be a positive integer")
    user_prompt = render_contextual_user_prompt(dossier)
    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": [
            {"role": "system", "content": CONTEXTUAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": CONTEXTUAL_SCHEMA_VERSION,
                "strict": True,
                "schema": CONTEXTUAL_OUTPUT_SCHEMA,
            },
        },
        "max_completion_tokens": max_completion_tokens,
    }


def prepare_contextual_request(
    dossier,
    model,
    max_completion_tokens,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
):
    """Return a request record and deterministic provenance metadata."""
    body = make_contextual_request_body(
        dossier,
        model,
        max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    doi = normalize_doi(dossier["paper_identity"]["doi"])
    dossier_sha256 = str(dossier["hashes"]["dossier_content_sha256"]).lower()
    custom_id = contextual_custom_id(doi, dossier_sha256)
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    request_configuration = {
        "model": body["model"],
        "temperature": "model_default",
        "reasoning_effort": body["reasoning_effort"],
        "max_completion_tokens": body["max_completion_tokens"],
    }
    metadata = {
        "custom_id": custom_id,
        "doi": doi,
        "model": body["model"],
        "max_completion_tokens": body["max_completion_tokens"],
        "temperature": "model_default",
        "reasoning_effort": body["reasoning_effort"],
        "dossier_schema_version": dossier["dossier_schema_version"],
        "dossier_content_sha256": dossier_sha256,
        "dossier_hashes": dossier["hashes"],
        "context_audit": dossier["context_audit"],
        "token_audit": dossier["token_audit"],
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "prompt_version": CONTEXTUAL_PROMPT_VERSION,
        "schema_sha256": contextual_schema_sha256(),
        "system_prompt_sha256": contextual_system_prompt_sha256(),
        "user_prompt_sha256": _sha256_text(body["messages"][1]["content"]),
        "request_configuration_sha256": _sha256_json(request_configuration),
        "request_body_sha256": _sha256_json(body),
        "request_sha256": _sha256_json(request),
    }
    return request, metadata


__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "SUPPORTED_REASONING_EFFORTS",
    "contextual_custom_id",
    "make_contextual_request_body",
    "prepare_contextual_request",
]
