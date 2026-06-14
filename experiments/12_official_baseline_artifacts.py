from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.external_baselines import load_official_routellm_artifacts
from routecode.reporting import upsert_markdown_section


DEFAULT_ROUTELLM_RESULTS = Path("data/raw/external/LLMRouterBench/baselines/RouteLLM/results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    results_dir = _results_dir(config)
    table = load_official_routellm_artifacts(results_dir)
    table.to_csv(out_dir / "table_official_external_artifacts.csv", index=False)
    write_memo(out_dir, config_path, results_dir, table)
    append_readme(out_dir, config_path, results_dir, table)
    print(f"Wrote official baseline artifact outputs to {out_dir}")


def _results_dir(config: dict) -> Path:
    configured = config.get("external_baselines", {}).get("official_routellm_results_dir")
    return Path(configured) if configured else DEFAULT_ROUTELLM_RESULTS


def append_readme(out_dir: Path, config_path: str, results_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Official External Baseline Artifacts"
    overall = _overall_summary(table)
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/12_official_baseline_artifacts.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_official_external_artifacts.csv`: parsed official upstream RouteLLM MF artifact rows from the local LLMRouterBench checkout.",
        "- `phase_e_official_baseline_artifacts_memo.md`: compatibility memo explaining why these artifacts are not RouteCode split-aligned metrics.",
        "",
        f"Source directory: `{results_dir}`.",
        "",
        "These rows are official upstream artifacts, but they are not RouteCode split-aligned and should not be ranked directly against `table_rate_distortion.csv`.",
        "",
        _markdown_table(overall),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, results_dir: Path, table: pd.DataFrame) -> None:
    overall = _overall_summary(table)
    datasets = sorted(name for name in table["dataset"].dropna().unique() if str(name))
    lines = [
        "# Phase E Official Baseline Artifact Memo",
        "",
        f"Command: `python experiments/12_official_baseline_artifacts.py --config {config_path}`",
        "",
        f"Source directory: `{results_dir}`.",
        "",
        "This memo records official RouteLLM MF artifacts found in the local LLMRouterBench checkout. These artifacts are useful for baseline inspection, dependency pinning, and novelty-boundary tracking, but they are not evaluated on the RouteCode train/test split or utility objective.",
        "",
        "## Compatibility",
        "",
        "- `split_aligned_with_routecode` is false for every row.",
        "- `routecode_metric_compatible` is false for every row.",
        "- Do not use these rows for direct method-ranking claims against RouteCode utility tables.",
        "",
        "## Overall Upstream MF Results",
        "",
        _markdown_table(overall),
        "",
        "## Dataset Coverage",
        "",
        ", ".join(datasets) if datasets else "No dataset-level rows found.",
        "",
        "## References Used",
        "",
        "- LLMRouterBench RouteLLM baseline artifacts: `data/raw/external/LLMRouterBench/baselines/RouteLLM/results`.",
        "- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm",
        "- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench",
        "",
        "## Remaining External-Baseline Gap",
        "",
        "- A split-aligned official RouteLLM reproduction still requires running the upstream router on the RouteCode train/test split with pinned embeddings/checkpoints.",
        "- GraphRouter, BEST-Route, and other external baselines still need pinned local commands before any direct ranking claim.",
        "",
    ]
    (out_dir / "phase_e_official_baseline_artifacts_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _overall_summary(table: pd.DataFrame) -> pd.DataFrame:
    overall = table[table["scope"] == "overall"].copy()
    columns = [
        "method",
        "seed",
        "total",
        "selection_accuracy",
        "routing_accuracy",
        "total_cost",
        "csv_selection_accuracy",
        "csv_total_cost",
    ]
    existing = [column for column in columns if column in overall.columns]
    return overall[existing].sort_values("seed")


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
