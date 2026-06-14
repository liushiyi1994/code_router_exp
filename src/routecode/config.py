from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def output_dir(config: dict[str, Any]) -> Path:
    out = Path(config.get("run", {}).get("output_dir", "results/demo"))
    out.mkdir(parents=True, exist_ok=True)
    return out
