from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    return data


def load_controlled_inputs(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else Path(".")
    inputs = config.get("inputs", {})
    for key in ("model_prices", "model_servers", "benchmark_sampling"):
        if key not in inputs:
            raise ValueError(f"Missing controlled input path: {key}")
    loaded = {
        "config": config,
        "prices": load_yaml(base_dir / inputs["model_prices"]),
        "servers": load_yaml(base_dir / inputs["model_servers"]),
        "benchmarks": load_yaml(base_dir / inputs["benchmark_sampling"]),
    }
    assert_no_claude_in_runnable_config(loaded)
    return loaded


def load_env_keys(path: str | Path) -> dict[str, bool]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    present: dict[str, bool] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        present[key.strip()] = bool(value.strip().strip('"').strip("'"))
    return present


def load_env_values(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def assert_no_claude_in_runnable_config(config_bundle: dict[str, Any]) -> None:
    """User requested no Claude/Anthropic in Phase 3 controlled execution configs."""

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return any(walk(k) or walk(v) for k, v in value.items())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        if isinstance(value, str):
            lowered = value.lower()
            return "claude" in lowered or "anthropic" in lowered
        return False

    checked = {
        "config": config_bundle.get("config", {}),
        "servers": config_bundle.get("servers", {}),
        "prices": config_bundle.get("prices", {}),
    }
    if walk(checked):
        raise ValueError("Runnable controlled configs must not include Claude/Anthropic models.")
