from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.benchmark_coverage import (
    build_broad_coverage_candidates,
    scan_llmrouterbench_coverage,
    summarize_dataset_coverage,
)
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    data_config = config.get("data", {})
    coverage_config = config.get("benchmark_coverage", {})
    model_counts = [int(value) for value in coverage_config.get("model_counts", [6, 10, 20, 32])]
    results_dir = data_config.get("results_dir")
    if not results_dir:
        raise ValueError("benchmark coverage requires data.results_dir")

    file_coverage = scan_llmrouterbench_coverage(results_dir)
    dataset_coverage = summarize_dataset_coverage(
        file_coverage,
        domain_map=data_config.get("domain_map"),
        taxonomy_map=data_config.get("task_taxonomy_map"),
    )
    candidates = build_broad_coverage_candidates(file_coverage, model_counts)

    file_coverage.to_csv(out_dir / "table_benchmark_file_coverage.csv", index=False)
    dataset_coverage.to_csv(out_dir / "table_benchmark_dataset_coverage.csv", index=False)
    candidates.to_csv(out_dir / "table_broad_coverage_candidates.csv", index=False)
    write_memo(out_dir, config_path, file_coverage, dataset_coverage, candidates)
    append_readme(out_dir, config_path, dataset_coverage, candidates)
    print(f"Wrote benchmark coverage audit outputs to {out_dir}")


def write_memo(
    out_dir: Path,
    config_path: str,
    file_coverage: pd.DataFrame,
    dataset_coverage: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    taxonomy_count = int(dataset_coverage["has_taxonomy"].sum()) if "has_taxonomy" in dataset_coverage else 0
    best = candidates.sort_values(["complete_query_count", "model_count"], ascending=[False, False]).head(1)
    lines = [
        "# Phase G Benchmark Coverage Memo",
        "",
        f"Command: `python experiments/20_benchmark_coverage.py --config {config_path}`",
        "",
        "This audit scans raw LLMRouterBench result JSON files before canonical schema validation. It does not run routers and makes no external API calls.",
        "",
        "## Coverage Summary",
        "",
        f"- Result files after latest-file filtering: `{len(file_coverage)}`.",
        f"- Datasets with local results: `{dataset_coverage['dataset'].nunique()}`.",
        f"- Models with local results: `{file_coverage['model_id'].nunique()}`.",
        f"- Datasets covered by configured taxonomy: `{taxonomy_count}`.",
        "",
        "## Candidate Complete Rectangles",
        "",
        _markdown_table(candidates),
        "",
        "## Readout",
        "",
    ]
    if best.empty:
        lines.append("- No complete rectangle candidates were produced.")
    else:
        row = best.iloc[0]
        lines.append(
            "- Largest candidate by complete query count uses "
            f"`{int(row['model_count'])}` models over `{int(row['dataset_count'])}` datasets "
            f"with `{int(row['complete_query_count'])}` complete queries."
        )
    lines.extend(
        [
            "- Use these candidates to choose larger real-data configs; do not infer routing performance from coverage alone.",
            "",
        ]
    )
    (out_dir / "phase_g_benchmark_coverage_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    config_path: str,
    dataset_coverage: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    readme_path = out_dir / "README.md"
    marker = "## Benchmark Coverage Audit"
    compact_datasets = dataset_coverage[
        ["dataset", "domain", "task_family", "task_subtype", "model_count", "total_model_records"]
    ].head(20)
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/20_benchmark_coverage.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_benchmark_file_coverage.csv`: latest raw result file coverage by dataset, split, and model.",
        "- `table_benchmark_dataset_coverage.csv`: dataset-level coverage and configured taxonomy status.",
        "- `table_broad_coverage_candidates.csv`: complete dataset/model rectangle candidates.",
        "- `phase_g_benchmark_coverage_memo.md`: benchmark coverage checkpoint memo.",
        "",
        "Candidate rectangles:",
        "",
        _markdown_table(candidates),
        "",
        "Dataset coverage sample:",
        "",
        _markdown_table(compact_datasets),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Benchmark Coverage Run\n"
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
    return str(value)


if __name__ == "__main__":
    main()
