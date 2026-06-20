from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir as config_output_dir
from routecode.local_eval.task_manifest import build_exact_task_manifest
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    run(config_path=args.config, output_dir=args.output_dir or None)


def run(*, config_path: str, output_dir: str | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    manifest_config = config.get("phase2_exact_task_manifest", {})
    out_dir = Path(output_dir) if output_dir else config_output_dir(config)
    prepared = prepare_from_config(config)
    datasets = [str(dataset) for dataset in manifest_config.get("datasets", ["aime", "math500"])]
    split = str(manifest_config.get("split", "test"))
    max_queries = int(manifest_config.get("max_queries", 200))
    manifest = build_exact_task_manifest(
        prepared.outcomes,
        datasets=datasets,
        split=split,
        max_queries=max_queries,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "local_exact_task_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_memo(out_dir, config_path, manifest, datasets, split, max_queries)
    append_readme(out_dir, config_path, manifest, datasets, split, max_queries)
    print(f"Wrote Phase 2 exact task manifest to {manifest_path}")
    return manifest


def write_memo(
    out_dir: Path,
    config_path: str,
    manifest: pd.DataFrame,
    datasets: list[str],
    split: str,
    max_queries: int,
) -> None:
    lines = [
        "# Phase 2 Exact Task Manifest",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/59_exact_task_manifest.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        "This manifest prepares exact-scored math tasks for true local Phase 2 runs. "
        "It uses RouteCode split assignments and excludes multiple-choice/code tasks until choices and sandboxed code evaluation are wired.",
        "",
        "Selection:",
        "",
        f"- Datasets requested: `{', '.join(datasets)}`.",
        f"- RouteCode split: `{split}`.",
        f"- Max queries: `{max_queries}`.",
        "",
        "Outputs:",
        "",
        "- `local_exact_task_manifest.csv`",
        "- `m10_exact_task_manifest_memo.md`",
        "",
        "Summary:",
        "",
        _markdown_table(_summary(manifest)),
        "",
    ]
    (out_dir / "m10_exact_task_manifest_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    config_path: str,
    manifest: pd.DataFrame,
    datasets: list[str],
    split: str,
    max_queries: int,
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Exact Task Manifest"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/59_exact_task_manifest.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        "This creates `local_exact_task_manifest.csv` for true local exact-scored math runs. "
        "It is a task substrate, not model-performance evidence.",
        "",
        "Selection:",
        "",
        f"- Datasets requested: `{', '.join(datasets)}`.",
        f"- RouteCode split: `{split}`.",
        f"- Max queries: `{max_queries}`.",
        "",
        "Outputs:",
        "",
        "- `local_exact_task_manifest.csv`",
        "- `m10_exact_task_manifest_memo.md`",
        "",
        _markdown_table(_summary(manifest)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    return (
        manifest.groupby(["dataset", "task_type", "routecode_split"], as_index=False)
        .agg(rows=("query_id", "count"), unique_queries=("query_id", "nunique"))
        .sort_values(["dataset", "task_type", "routecode_split"])
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
