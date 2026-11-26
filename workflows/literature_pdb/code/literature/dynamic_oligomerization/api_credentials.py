"""API credential loading shared by optional online literature stages."""

import os
from pathlib import Path


def load_api_key(environ=None, config_path=None):
    """Load an OpenAI API key from the environment or an explicit local file."""
    environ = os.environ if environ is None else environ
    key = str(environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    path = Path(config_path or "~/.config/openai/api_key").expanduser()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "OpenAI API key not found in OPENAI_API_KEY or %s" % path
        ) from exc
    if not key:
        raise RuntimeError("OpenAI API key file is empty: %s" % path)
    return key


__all__ = ["load_api_key"]
