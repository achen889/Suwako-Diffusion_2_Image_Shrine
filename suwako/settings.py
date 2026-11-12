"""Persist the user's explicitly saved studio configuration locally."""
from __future__ import annotations

import json

from .history import OUTPUT_ROOT

CONFIG_PATH = OUTPUT_ROOT / ".studio-configuration.json"


def load_last_configuration() -> dict | None:
    """Read a saved configuration, ignoring missing or malformed local state."""
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        return None
    return config


def save_last_configuration(config: dict) -> None:
    """Atomically save a configuration under the ignored outputs directory."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_PATH.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp.replace(CONFIG_PATH)
