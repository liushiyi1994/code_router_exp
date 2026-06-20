from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.transformer_backbones import DEFAULT_REQUESTED_BACKBONES, inspect_transformer_backbone_cache
from routecode.reporting import upsert_markdown_section


DEFAULT_CACHE_DIR = Path("~/.cache/huggingface/hub")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    backbone_config = config.get("transformer_backbones", {})
    cache_dir = Path(backbone_config.get("cache_dir", DEFAULT_CACHE_DIR)).expanduser()
    requested = list(backbone_config.get("requested_model_ids", DEFAULT_REQUESTED_BACKBONES))
    max_runnable_gb = float(backbone_config.get("max_runnable_gb", 2.0))
    table = inspect_transformer_backbone_cache(
        cache_dir,
        requested_model_ids=requested,
        max_runnable_gb=max_runnable_gb,
    )
    table.to_csv(out_dir / "table_transformer_backbone_readiness.csv", index=False)
    write_memo(out_dir, config_path, cache_dir, requested, max_runnable_gb, table)
    append_readme(out_dir, config_path, cache_dir, table)
    print(f"Wrote transformer backbone readiness outputs to {out_dir}")


def append_readme(out_dir: Path, config_path: str, cache_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Transformer Backbone Readiness"
    runnable_count = int(table["runnable_as_encoder_baseline"].sum()) if not table.empty else 0
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/13_transformer_backbone_readiness.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_transformer_backbone_readiness.csv`: cache-only transformer text-backbone readiness scan.",
        "- `phase_f_g_transformer_backbone_readiness_memo.md`: memo explaining why transformer embedding/direct-router baselines are or are not runnable locally.",
        "",
        f"Cache directory: `{cache_dir}`.",
        "",
        f"Runnable encoder candidates found: `{runnable_count}`. This scan performs no downloads and does not load model weights.",
        "",
        _markdown_table(_summary_table(table)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(
    out_dir: Path,
    config_path: str,
    cache_dir: Path,
    requested: list[str],
    max_runnable_gb: float,
    table: pd.DataFrame,
) -> None:
    runnable = table[table["runnable_as_encoder_baseline"] == True] if not table.empty else table
    requested_missing = (
        table[
            table["model_id"].isin(requested)
            & (table["runnable_as_encoder_baseline"] != True)
        ]
        if not table.empty
        else table
    )
    lines = [
        "# Phase F/G Transformer Backbone Readiness Memo",
        "",
        f"Command: `python experiments/13_transformer_backbone_readiness.py --config {config_path}`",
        "",
        f"Cache directory: `{cache_dir}`.",
        f"Requested text backbones: `{', '.join(requested)}`.",
        f"Runnable size budget: `{max_runnable_gb:.2f}` GB.",
        "",
        "This scan reads local Hugging Face cache metadata only. It performs no downloads, does not import transformer model classes, and does not load model weights.",
        "",
    ]
    if runnable.empty:
        lines.extend(
            [
                "No transformer embedding baseline was executed because no requested lightweight encoder checkpoint was available in the local cache.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "At least one cached lightweight encoder candidate is available. Transformer-embedding direct-router rows should be checked in `table_transformer_embedding_router.csv`; extraction must use `local_files_only=True` and the same query-id split.",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            _markdown_table(_summary_table(table)),
            "",
            "## Compatibility",
            "",
            "- This artifact is a readiness audit, not a routing metric table.",
            "- It does not satisfy the full requested predictor-type ablation until each claim-critical cached encoder is evaluated on the RouteCode split.",
            "- It moves the Research Flow forward by making available and missing transformer-backbone dependencies explicit and reproducible.",
            "",
            "## Next Step",
            "",
            _next_step_line(requested_missing),
            "",
        ]
    )
    (out_dir / "phase_f_g_transformer_backbone_readiness_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _next_step_line(missing: pd.DataFrame) -> str:
    if missing.empty:
        return "- Evaluate any additional claim-critical encoder backbones under the local-files-only transformer embedding router before making predictor-type claims."
    missing_ids = ", ".join(str(model_id) for model_id in missing["model_id"].tolist())
    return f"- Cache and evaluate the remaining requested encoder backbones under the local-files-only transformer embedding router: `{missing_ids}`."


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_id",
        "cache_status",
        "runnable_as_encoder_baseline",
        "reason",
        "architecture",
        "size_gb",
    ]
    existing = [column for column in columns if column in table.columns]
    return table[existing].sort_values(["cache_status", "model_id"])


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
