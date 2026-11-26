"""Pure request preparation for the canonical protein-level extractor.

This module intentionally contains no API submission or lifecycle operations.
"""

import hashlib
import json
import time
from pathlib import Path

from .identity import build_identity_graph, identity_graph_hashes
from .hashing import sha256_file
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt
from .schema import OUTPUT_SCHEMA, SCHEMA_VERSION, schema_sha256
from .text import extract_text, retrieve_source_blocks


def custom_id_for_group(group_id):
    """Return a stable request identifier derived only from canonical group_id."""
    value = str(group_id).strip()
    if not value:
        raise ValueError("group_id is required")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return "dynamic-protein-" + digest


def verify_source_sha256(group):
    path = Path(group["paper_path"])
    expected = str(group.get("paper_sha256") or "").lower()
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(
            "paper SHA-256 mismatch for group %s" % group.get("group_id", "")
        )
    return actual


def make_request_body(
    group,
    passages,
    model,
    max_completion_tokens=6000,
    identity_graph=None,
):
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_user_prompt(
                    group, passages, identity_graph=identity_graph
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_VERSION,
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            },
        },
        "max_completion_tokens": max_completion_tokens,
    }


def prepare_group(
    group,
    model,
    max_completion_tokens=6000,
    verify_source=True,
    manifest_rows=None,
):
    """Prepare one DOI-target request with DOI-level typed identity context."""
    started = time.perf_counter()
    source_sha256 = (
        verify_source_sha256(group)
        if verify_source
        else str(group.get("paper_sha256") or "").lower()
    )
    if not source_sha256:
        raise ValueError("paper_sha256 is required when source verification is disabled")
    verified = time.perf_counter()
    all_passages = extract_text(group["paper_path"])
    extracted = time.perf_counter()
    doi_rows = list(manifest_rows or [group])
    identity_graph = build_identity_graph(
        doi_rows,
        all_passages,
        doi=group["doi"],
        target_group_id=str(group["group_id"]),
    )
    identity_built = time.perf_counter()
    selected, retrieval_audit = retrieve_source_blocks(
        all_passages,
        group,
        identity_graph=identity_graph,
        return_audit=True,
    )
    retrieved = time.perf_counter()
    selected_by_source = {
        passage["source_id"]: passage["text"] for passage in selected
    }
    selected_by_base = {}
    selected_by_origin = {}
    for passage in selected:
        base_source_id = str(
            passage.get("base_source_id") or passage["source_id"]
        )
        selected_by_base.setdefault(base_source_id, []).append(passage)
        origin_source_id = str(
            passage.get("source_passage_id") or passage["source_id"]
        )
        selected_by_origin.setdefault(origin_source_id, []).append(passage)
    omitted_anchors = []
    for edge in identity_graph.get("edges") or []:
        source_id = str(edge.get("source_id") or "")
        anchor = str(edge.get("anchor_quote") or "")
        if source_id.startswith("manifest:"):
            continue
        candidate_windows = list(selected_by_base.get(source_id, []))
        candidate_windows.extend(
            passage
            for passage in selected_by_origin.get(source_id, [])
            if passage not in candidate_windows
        )
        if source_id in selected_by_source:
            candidate_windows.extend(
                passage
                for passage in selected
                if passage["source_id"] == source_id
                and passage not in candidate_windows
            )
        if not candidate_windows:
            omitted_anchors.append(
                {
                    "edge_id": edge["edge_id"],
                    "source_id": source_id,
                    "reason": "identity_anchor_source_not_retrievable",
                }
            )
        elif not any(anchor in passage["text"] for passage in candidate_windows):
            omitted_anchors.append(
                {
                    "edge_id": edge["edge_id"],
                    "source_id": source_id,
                    "reason": "identity_anchor_outside_selected_source_window",
                }
            )
    # Selected passages are already stored beside the graph. Avoid duplicating
    # full-paper text in graph metadata; validation hydrates this audit index
    # from selected passages before resolving claims.
    compact_identity_graph = json.loads(
        json.dumps(identity_graph, sort_keys=True, separators=(",", ":"))
    )
    compact_identity_graph.pop("source_text", None)
    hashes = identity_graph_hashes(compact_identity_graph)
    body = make_request_body(
        group,
        selected,
        model,
        max_completion_tokens,
        identity_graph=compact_identity_graph,
    )
    rendered = time.perf_counter()
    custom_id = custom_id_for_group(group["group_id"])
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    metadata = {
        "custom_id": custom_id,
        "group_id": str(group["group_id"]),
        "paper_path": str(Path(group["paper_path"])),
        "source_sha256": source_sha256,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "schema_sha256": schema_sha256(SCHEMA_VERSION),
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(
            body["messages"][1]["content"].encode("utf-8")
        ).hexdigest(),
        "selected_passages": selected,
        "identity_graph": compact_identity_graph,
        "identity_graph_sha256": hashes["identity_graph_sha256"],
        "identity_policy_sha256": hashes["identity_policy_sha256"],
        "identity_anchor_prompt_coverage": {
            "all_anchor_sources_supplied": not omitted_anchors,
            "omitted": omitted_anchors,
        },
        "bounded_term_audit": retrieval_audit,
        "preparation_elapsed_seconds": {
            "source_verification": verified - started,
            "text_extraction": extracted - verified,
            "identity_graph": identity_built - extracted,
            "retrieval": retrieved - identity_built,
            "prompt_rendering": rendered - retrieved,
            "total": rendered - started,
        },
    }
    return request, metadata
