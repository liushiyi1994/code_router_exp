from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.graphrouter_assets import (
    build_graphrouter_assets,
    summarize_graphrouter_assets,
    write_graphrouter_assets,
)
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


ASSET_DIRNAME = "graphrouter_assets"
GRAPHROUTER_SOURCE = "data/raw/external/LLMRouterBench/baselines/GraphRouter"
GRAPHROUTER_REPO = "https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/GraphRouter"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    seed = int(config.get("run", {}).get("random_seed", 0))
    graph_config = config.get("graphrouter_assets", {})
    split_ratio = tuple(float(value) for value in graph_config.get("split_ratio", [0.7, 0.0, 0.3]))
    if len(split_ratio) != 3:
        raise ValueError("graphrouter_assets.split_ratio must contain three values")

    assets = build_graphrouter_assets(
        prepared.matrices,
        prepared.embeddings,
        seed=seed,
        split_ratio=split_ratio,
    )
    asset_dir = out_dir / ASSET_DIRNAME
    written = write_graphrouter_assets(assets, asset_dir)
    table = summarize_graphrouter_assets(assets)
    table.to_csv(out_dir / "table_graphrouter_assets.csv", index=False)
    write_memo(out_dir, config_path, written.asset_dir, written.config_path, table)
    append_readme(out_dir, config_path, written.asset_dir, table)
    print(f"Wrote GraphRouter-compatible assets to {asset_dir}")


def write_memo(
    out_dir: Path,
    config_path: str,
    asset_dir: Path,
    config_local_path: Path,
    table: pd.DataFrame,
) -> None:
    overall = table.set_index("split").loc["overall"]
    lines = [
        "# Phase E GraphRouter Asset Memo",
        "",
        f"Command: `python experiments/24_graphrouter_assets.py --config {config_path}`",
        "",
        "This run writes GraphRouter-compatible data-contract assets without API calls. "
        "It is not an upstream GraphRouter metric row because the unmodified upstream "
        "runner still requires its GNN dependency stack and performs its own internal split.",
        "",
        "## Outputs",
        "",
        f"- Asset directory: `{asset_dir}`",
        f"- Local upstream config: `{config_local_path}`",
        f"- Summary table: `{out_dir / 'table_graphrouter_assets.csv'}`",
        "",
        "## Summary",
        "",
        f"- Queries: `{int(overall['query_count'])}`",
        f"- Query-model rows: `{int(overall['row_count'])}`",
        f"- Models: `{int(overall['model_count'])}`",
        f"- Tasks: `{int(overall['task_count'])}`",
        "- RouteCode split labels are preserved in `routecode_split`; upstream GraphRouter does not consume that column unchanged.",
        f"- Source inspected: `{GRAPHROUTER_SOURCE}`; repo: {GRAPHROUTER_REPO}",
        "",
        _markdown_table(table),
    ]
    (out_dir / "phase_e_graphrouter_assets_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, asset_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Results\n"
    marker = "## GraphRouter Assets"
    section = [
        marker,
        "",
        "Command:",
        "",
        f"`python experiments/24_graphrouter_assets.py --config {config_path}`",
        "",
        "Outputs:",
        "",
        f"- `{asset_dir / 'router_data.csv'}`: GraphRouter router-data schema with RouteCode query splits retained as metadata.",
        f"- `{asset_dir / 'LLM_Descriptions.json'}`: model-description file for the upstream GraphRouter loader.",
        f"- `{asset_dir / 'llm_description_embedding.pkl'}`: deterministic model-description embedding matrix.",
        f"- `{asset_dir / 'config.local.yaml'}`: local upstream GraphRouter config with generated asset paths.",
        "- `table_graphrouter_assets.csv`: asset counts and compatibility flags.",
        "- `phase_e_graphrouter_assets_memo.md`: notes on compatibility and remaining blockers.",
        "",
        _markdown_table(table),
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, section), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
