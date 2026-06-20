from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.external_baseline_assets import (
    build_external_baseline_assets,
    summarize_external_baseline_assets,
    write_external_baseline_assets,
)
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    asset_config = config.get("external_baseline_assets", {})
    threshold = float(asset_config.get("correctness_threshold", 0.5))
    assets = build_external_baseline_assets(
        prepared.matrices,
        prepared.embeddings,
        correctness_threshold=threshold,
    )
    written = write_external_baseline_assets(assets, out_dir)
    table = summarize_external_baseline_assets(assets)
    table.to_csv(out_dir / "table_external_baseline_assets.csv", index=False)
    write_memo(out_dir, config_path, written.asset_dir, table, assets.metadata)
    append_readme(out_dir, config_path, written.asset_dir, table)
    print(f"Wrote external baseline assets to {out_dir}")


def write_memo(
    out_dir: Path,
    config_path: str,
    asset_dir: Path,
    table: pd.DataFrame,
    metadata: dict,
) -> None:
    lines = [
        "# Phase E External Baseline Asset Memo",
        "",
        f"Command: `python experiments/26_external_baseline_assets.py --config {config_path}`",
        "",
        "This run writes split-aligned input assets for several upstream external baseline command paths. "
        "It makes no external API calls, trains no upstream models, and does not create metric rows.",
        "",
        "## Outputs",
        "",
        f"- Asset root: `{asset_dir}`",
        "- `frugalgpt_split_aligned/train.jsonl` and `test.jsonl`",
        "- `embedllm_assets/train.csv`, `test.csv`, `question_embeddings.pth`, and `question_embeddings_3584.pth`",
        "- `embedllm_assets/smoke_train.csv` and `smoke_test.csv` for bounded upstream KNN command-path smoke checks",
        "- `best_route_assets/train.jsonl`, `validation.jsonl`, and `test.jsonl`",
        "- `routerdc_assets/train.json`, `test.json`, and `final_eval.json`",
        "- `modelsat_assets/seed42/train.json`, `test.json`, `ood.json`, and `model_description.json`",
        "- `table_external_baseline_assets.csv`",
        "",
        "## Compatibility",
        "",
        f"- `split_aligned_with_routecode`: `{metadata['split_aligned_with_routecode']}`",
        f"- `routecode_metric_compatible`: `{metadata['routecode_metric_compatible']}`",
        f"- `official_upstream_result`: `{metadata['official_upstream_result']}`",
        f"- RouteCode splits exported: `{metadata['routecode_splits']}`",
        f"- Validation split used for upstream validation assets: `{metadata['validation_split']}`",
        "",
        _markdown_table(table),
        "",
        "## Remaining Gap",
        "",
        "These assets remove missing-input-file blockers for FrugalGPT, EmbedLLM, BEST-Route, RouterDC, and MODEL-SAT readiness checks. Remaining blockers are expected to be local checkpoints and Python dependency stacks until those environments are installed.",
        "",
    ]
    (out_dir / "phase_e_external_baseline_assets_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, asset_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Results\n"
    marker = "## External Baseline Assets"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/26_external_baseline_assets.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        f"- Asset root: `{asset_dir}`",
        "- `table_external_baseline_assets.csv`: asset counts and compatibility flags.",
        "- `embedllm_assets/question_embeddings_3584.pth`: padded upstream-compatible question embeddings for the EmbedLLM MF CLI.",
        "- `embedllm_assets/smoke_train.csv` and `smoke_test.csv`: bounded smoke inputs for the upstream EmbedLLM KNN CLI.",
        "- `phase_e_external_baseline_assets_memo.md`: memo explaining remaining blockers.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


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
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
