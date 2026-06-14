from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REQUESTED_BACKBONES = [
    "answerdotai/ModernBERT-base",
    "microsoft/deberta-v3-base",
]


def inspect_transformer_backbone_cache(
    cache_dir: str | Path,
    requested_model_ids: list[str] | None = None,
    max_runnable_gb: float = 2.0,
) -> pd.DataFrame:
    """Inspect local Hugging Face cache for encoder-backbone readiness.

    This intentionally reads metadata only. It does not import transformers,
    instantiate models, download files, or execute remote code.
    """

    root = Path(cache_dir).expanduser()
    requested = list(requested_model_ids or DEFAULT_REQUESTED_BACKBONES)
    cached = _cached_model_dirs(root)
    model_ids = sorted(set(requested) | set(cached))
    rows = [
        _row_for_model(model_id, cached.get(model_id), max_runnable_gb=max_runnable_gb)
        for model_id in model_ids
    ]
    return pd.DataFrame(rows)


def _cached_model_dirs(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    cached: dict[str, Path] = {}
    for path in sorted(root.glob("models--*")):
        if not path.is_dir():
            continue
        model_id = path.name.removeprefix("models--").replace("--", "/")
        cached[model_id] = path
    return cached


def _row_for_model(model_id: str, model_dir: Path | None, max_runnable_gb: float) -> dict[str, Any]:
    if model_dir is None:
        return {
            "model_id": model_id,
            "cache_status": "missing_local_cache",
            "runnable_as_encoder_baseline": False,
            "reason": "missing_local_cache",
            "architecture": "",
            "model_type": "",
            "hidden_size": "",
            "size_gb": 0.0,
            "local_path": "",
        }

    config_path = _latest_config_path(model_dir)
    size_gb = _directory_size_gb(model_dir)
    if config_path is None:
        return {
            "model_id": model_id,
            "cache_status": "cached",
            "runnable_as_encoder_baseline": False,
            "reason": "missing_transformer_config",
            "architecture": "",
            "model_type": "",
            "hidden_size": "",
            "size_gb": size_gb,
            "local_path": str(model_dir),
        }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architecture = ",".join(str(item) for item in architectures)
    model_type = str(config.get("model_type") or "")
    hidden_size = config.get("hidden_size", "")
    runnable, reason = _classify_backbone(architecture, model_type, size_gb, max_runnable_gb)
    return {
        "model_id": model_id,
        "cache_status": "cached",
        "runnable_as_encoder_baseline": runnable,
        "reason": reason,
        "architecture": architecture,
        "model_type": model_type,
        "hidden_size": hidden_size,
        "size_gb": size_gb,
        "local_path": str(config_path.parent),
    }


def _latest_config_path(model_dir: Path) -> Path | None:
    configs = sorted(model_dir.glob("snapshots/*/config.json"))
    if not configs:
        return None
    return configs[-1]


def _directory_size_gb(path: Path) -> float:
    total = 0
    seen: set[Path] = set()
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            total += resolved.stat().st_size
        except OSError:
            continue
    return total / (1024**3)


def _classify_backbone(
    architecture: str,
    model_type: str,
    size_gb: float,
    max_runnable_gb: float,
) -> tuple[bool, str]:
    arch_lower = architecture.lower()
    type_lower = model_type.lower()
    if "causallm" in arch_lower or "forconditionalgeneration" in arch_lower:
        return False, "causal_lm_not_lightweight_encoder"
    if any(marker in arch_lower or marker in type_lower for marker in ["diffusion", "flux", "vae"]):
        return False, "not_text_encoder_model"
    if size_gb > max_runnable_gb:
        return False, "cached_model_exceeds_size_budget"
    if any(marker in arch_lower or marker in type_lower for marker in ["bert", "deberta", "roberta", "e5", "mpnet"]):
        return True, "cached_encoder_candidate"
    return False, "unknown_text_backbone_type"
