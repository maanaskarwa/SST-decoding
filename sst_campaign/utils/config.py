"""Loads the SST campaign TOML configuration file."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml_config(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return tomllib.load(f)
