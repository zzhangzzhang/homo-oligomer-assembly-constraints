"""Deterministic serialization helpers for extraction artifacts."""

import json
import os
import tempfile
from pathlib import Path


def canonical_json(value, indent=None):
    separators = (",", ":") if indent is None else None
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=separators,
    )


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value, indent=2))
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path, rows, sort_key=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(rows)
    if sort_key is not None:
        ordered.sort(key=sort_key)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(canonical_json(row) + "\n")
