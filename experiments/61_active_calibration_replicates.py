from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--source-table", default="")
    parser.add_argument("--output-dir", default="results/phase2")
    parser.add_argument("--max-holdout-models", type=int, default=2)
    parser.add_argument("--r-values", default="")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    r_values = _parse_int_list(args.r_values)
    seeds = _parse_int_list(args.seeds) or [0, 1, 2]
    run(
        config_path=args.config or None,
        source_table_path=args.source_table or None,
        output_dir=args.output_dir,
        max_holdout_models=args.max_holdout_models,
        r_values=r_values,
        seeds=seeds,
    )


def run(
    *,
    output_dir: str,
    config_path: str | None = None,
    source_table_path: str | None = None,
    max_holdout_models: int = 2,
    r_values: list[int] | None = None,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source_table_path:
        replicate_table = pd.read_csv(source_table_path)
        command = f"python experiments/61_active_calibration_replicates.py --source-table {source_table_path} --output-dir {output_dir}"
    elif config_path:
        seeds = seeds or [0, 1, 2]
        tables: list[pd.DataFrame] = []
        runner = _load_active_calibration_runner()
        for seed in seeds:
            table = runner.run_active_calibration_from_config(
                config_path=config_path,
                max_holdout_models=max_holdout_models,
                r_values=r_values,
                seed_override=seed,
            )
            table = table.copy()
            table.insert(0, "replicate_seed", int(seed))
            tables.append(table)
        replicate_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        command = _command(config_path, output_dir, max_holdout_models, r_values, seeds)
    else:
        replicate_table = pd.DataFrame(
            [
                {
                    "replicate_seed": "",
                    "method": "active_route_state_calibration",
                    "status": "blocked_missing_config_or_source_table",
                    "new_model_id": "",
                    "examples_per_label": "",
                    "new_model_evaluations": 0,
                    "mean_utility": pd.NA,
                    "recovered_gap_vs_oracle": pd.NA,
                }
            ]
        )
        command = f"python experiments/61_active_calibration_replicates.py --output-dir {output_dir}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_replicates(replicate_table)
    uniform_deltas = active_vs_uniform_deltas(replicate_table)
    random_deltas = active_vs_random_deltas(replicate_table)
    dataset_deltas = active_vs_dataset_deltas(replicate_table)
    embedding_deltas = active_vs_embedding_deltas(replicate_table)
    write_outputs(
        out_dir,
        replicate_table,
        summary,
        uniform_deltas,
        random_deltas,
        dataset_deltas,
        embedding_deltas,
        command,
    )
    print(f"Wrote active-calibration replicate rows to {out_dir / 'table_active_calibration_replicates.csv'}")
    print(f"Wrote active-calibration replicate summary to {out_dir / 'table_active_calibration_replicate_summary.csv'}")
    print(
        "Wrote active-vs-uniform replicate deltas to "
        f"{out_dir / 'table_active_calibration_active_vs_uniform_deltas.csv'}"
    )
    print(
        "Wrote active-vs-random replicate deltas to "
        f"{out_dir / 'table_active_calibration_active_vs_random_deltas.csv'}"
    )
    print(
        "Wrote active-vs-dataset replicate deltas to "
        f"{out_dir / 'table_active_calibration_active_vs_dataset_deltas.csv'}"
    )
    print(
        "Wrote active-vs-embedding replicate deltas to "
        f"{out_dir / 'table_active_calibration_active_vs_embedding_deltas.csv'}"
    )
    return replicate_table, summary, uniform_deltas, random_deltas, dataset_deltas, embedding_deltas


def summarize_replicates(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "mean_utility" not in table.columns:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ["method", "new_model_id", "examples_per_label"]
        if column in table.columns
    ]
    if not group_columns:
        return pd.DataFrame()
    numeric_table = table.copy()
    for column in ["mean_utility", "recovered_gap_vs_oracle", "new_model_evaluations"]:
        if column in numeric_table.columns:
            numeric_table[column] = pd.to_numeric(numeric_table[column], errors="coerce")
    aggregations: dict[str, tuple[str, str]] = {
        "replicates": ("mean_utility", "count"),
        "mean_utility_mean": ("mean_utility", "mean"),
        "mean_utility_std": ("mean_utility", "std"),
        "mean_utility_min": ("mean_utility", "min"),
        "mean_utility_max": ("mean_utility", "max"),
        "recovered_gap_vs_oracle_mean": ("recovered_gap_vs_oracle", "mean")
        if "recovered_gap_vs_oracle" in numeric_table.columns
        else ("mean_utility", "mean"),
    }
    if "new_model_evaluations" in numeric_table.columns:
        aggregations.update(
            {
                "new_model_evaluations_mean": ("new_model_evaluations", "mean"),
                "new_model_evaluations_min": ("new_model_evaluations", "min"),
                "new_model_evaluations_max": ("new_model_evaluations", "max"),
            }
        )
    summary = (
        numeric_table.dropna(subset=["mean_utility"])
        .groupby(group_columns, as_index=False)
        .agg(**aggregations)
    )
    for column in [
        "mean_utility_mean",
        "mean_utility_std",
        "mean_utility_min",
        "mean_utility_max",
        "recovered_gap_vs_oracle_mean",
        "new_model_evaluations_mean",
    ]:
        if column in summary.columns:
            summary[column] = summary[column].round(6)
    return summary.sort_values(group_columns).reset_index(drop=True)


def active_vs_uniform_deltas(table: pd.DataFrame) -> pd.DataFrame:
    return _active_vs_baseline_deltas(
        table,
        baseline_method="uniform_route_state_calibration",
        baseline_column="uniform_route_state_calibration",
        delta_column="active_minus_uniform_mean_utility",
    )


def active_vs_random_deltas(table: pd.DataFrame) -> pd.DataFrame:
    return _active_vs_baseline_deltas(
        table,
        baseline_method="random_route_state_calibration",
        baseline_column="random_route_state_calibration",
        delta_column="active_minus_random_mean_utility",
    )


def active_vs_dataset_deltas(table: pd.DataFrame) -> pd.DataFrame:
    return _active_vs_baseline_deltas(
        table,
        baseline_method="dataset_stratified_calibration",
        baseline_column="dataset_stratified_calibration",
        delta_column="active_minus_dataset_mean_utility",
    )


def active_vs_embedding_deltas(table: pd.DataFrame) -> pd.DataFrame:
    return _active_vs_baseline_deltas(
        table,
        baseline_method="embedding_cluster_calibration",
        baseline_column="embedding_cluster_calibration",
        delta_column="active_minus_embedding_mean_utility",
    )


def _active_vs_baseline_deltas(
    table: pd.DataFrame,
    *,
    baseline_method: str,
    baseline_column: str,
    delta_column: str,
) -> pd.DataFrame:
    required = {
        "replicate_seed",
        "method",
        "new_model_id",
        "examples_per_label",
        "new_model_evaluations",
        "mean_utility",
    }
    if table.empty or not required.issubset(table.columns):
        return pd.DataFrame()
    relevant = table[table["method"].isin(["active_route_state_calibration", baseline_method])].copy()
    if relevant.empty:
        return pd.DataFrame()
    relevant["mean_utility"] = pd.to_numeric(relevant["mean_utility"], errors="coerce")
    wide = relevant.pivot_table(
        index=["replicate_seed", "new_model_id", "examples_per_label", "new_model_evaluations"],
        columns="method",
        values="mean_utility",
        aggfunc="mean",
    ).reset_index()
    if "active_route_state_calibration" not in wide.columns or baseline_column not in wide.columns:
        return pd.DataFrame()
    wide[delta_column] = (wide["active_route_state_calibration"] - wide[baseline_column]).round(6)
    return wide.sort_values(["new_model_id", "examples_per_label", "new_model_evaluations", "replicate_seed"]).reset_index(
        drop=True
    )


def summarize_deltas(
    deltas: pd.DataFrame,
    *,
    delta_column: str = "active_minus_uniform_mean_utility",
) -> pd.DataFrame:
    if deltas.empty or delta_column not in deltas.columns:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ["new_model_id", "examples_per_label"]
        if column in deltas.columns
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "replicates": (delta_column, "count"),
        f"{delta_column}_mean": (delta_column, "mean"),
        f"{delta_column}_std": (delta_column, "std"),
        f"{delta_column}_min": (delta_column, "min"),
        f"{delta_column}_max": (delta_column, "max"),
    }
    if "new_model_evaluations" in deltas.columns:
        aggregations.update(
            {
                "new_model_evaluations_mean": ("new_model_evaluations", "mean"),
                "new_model_evaluations_min": ("new_model_evaluations", "min"),
                "new_model_evaluations_max": ("new_model_evaluations", "max"),
            }
        )
    summary = (
        deltas.groupby(group_columns, as_index=False)
        .agg(**aggregations)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    for column in [
        f"{delta_column}_mean",
        f"{delta_column}_std",
        f"{delta_column}_min",
        f"{delta_column}_max",
        "new_model_evaluations_mean",
    ]:
        if column in summary.columns:
            summary[column] = summary[column].round(6)
    return summary


def interpret_deltas(
    deltas: pd.DataFrame,
    *,
    delta_column: str = "active_minus_uniform_mean_utility",
    baseline_label: str = "uniform",
) -> str:
    if deltas.empty or delta_column not in deltas.columns:
        return "No paired active-vs-uniform rows were available."
    values = pd.to_numeric(deltas[delta_column], errors="coerce").dropna()
    if values.empty:
        return f"No numeric paired active-vs-{baseline_label} deltas were available."
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    tied = int((values == 0).sum())
    return (
        f"Across paired active-vs-{baseline_label} rows, active calibration has mean utility delta "
        f"`{values.mean():.4f}` over `{len(values)}` pairs "
        f"(`{positive}` positive, `{negative}` negative, `{tied}` tied)."
    )


def write_outputs(
    out_dir: Path,
    replicate_table: pd.DataFrame,
    summary: pd.DataFrame,
    uniform_deltas: pd.DataFrame,
    random_deltas: pd.DataFrame,
    dataset_deltas: pd.DataFrame,
    embedding_deltas: pd.DataFrame,
    command: str,
) -> None:
    replicate_table.to_csv(out_dir / "table_active_calibration_replicates.csv", index=False)
    summary.to_csv(out_dir / "table_active_calibration_replicate_summary.csv", index=False)
    uniform_deltas.to_csv(out_dir / "table_active_calibration_active_vs_uniform_deltas.csv", index=False)
    random_deltas.to_csv(out_dir / "table_active_calibration_active_vs_random_deltas.csv", index=False)
    dataset_deltas.to_csv(out_dir / "table_active_calibration_active_vs_dataset_deltas.csv", index=False)
    embedding_deltas.to_csv(out_dir / "table_active_calibration_active_vs_embedding_deltas.csv", index=False)
    write_memo(out_dir, summary, uniform_deltas, random_deltas, dataset_deltas, embedding_deltas, command)
    append_readme(out_dir, summary, uniform_deltas, random_deltas, dataset_deltas, embedding_deltas, command)


def write_memo(
    out_dir: Path,
    summary: pd.DataFrame,
    uniform_deltas: pd.DataFrame,
    random_deltas: pd.DataFrame,
    dataset_deltas: pd.DataFrame,
    embedding_deltas: pd.DataFrame,
    command: str,
) -> None:
    uniform_delta_summary = summarize_deltas(uniform_deltas, delta_column="active_minus_uniform_mean_utility")
    random_delta_summary = summarize_deltas(random_deltas, delta_column="active_minus_random_mean_utility")
    dataset_delta_summary = summarize_deltas(dataset_deltas, delta_column="active_minus_dataset_mean_utility")
    embedding_delta_summary = summarize_deltas(embedding_deltas, delta_column="active_minus_embedding_mean_utility")
    lines = [
        "# Phase 2 Active Calibration Replicates",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "This repeats the active new-model calibration comparison across seeds and held-out models.",
        "",
        "Outputs:",
        "",
        "- `table_active_calibration_replicates.csv`",
        "- `table_active_calibration_replicate_summary.csv`",
        "- `table_active_calibration_active_vs_uniform_deltas.csv`",
        "- `table_active_calibration_active_vs_random_deltas.csv`",
        "- `table_active_calibration_active_vs_dataset_deltas.csv`",
        "- `table_active_calibration_active_vs_embedding_deltas.csv`",
        "- `m6_active_calibration_replicates_memo.md`",
        "",
        "Replicate Summary:",
        "",
        _markdown_table(summary),
        "",
        "Active vs Uniform Deltas:",
        "",
        _markdown_table(uniform_delta_summary),
        "",
        "Active vs Random Deltas:",
        "",
        _markdown_table(random_delta_summary),
        "",
        "Active vs Dataset Deltas:",
        "",
        _markdown_table(dataset_delta_summary),
        "",
        "Active vs Embedding Deltas:",
        "",
        _markdown_table(embedding_delta_summary),
        "",
        "Interpretation:",
        "",
        interpret_deltas(uniform_deltas, delta_column="active_minus_uniform_mean_utility", baseline_label="uniform"),
        "",
        interpret_deltas(random_deltas, delta_column="active_minus_random_mean_utility", baseline_label="random"),
        "",
        interpret_deltas(dataset_deltas, delta_column="active_minus_dataset_mean_utility", baseline_label="dataset"),
        "",
        interpret_deltas(
            embedding_deltas,
            delta_column="active_minus_embedding_mean_utility",
            baseline_label="embedding",
        ),
        "",
    ]
    (out_dir / "m6_active_calibration_replicates_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    summary: pd.DataFrame,
    uniform_deltas: pd.DataFrame,
    random_deltas: pd.DataFrame,
    dataset_deltas: pd.DataFrame,
    embedding_deltas: pd.DataFrame,
    command: str,
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Active Calibration Replicates"
    uniform_delta_summary = summarize_deltas(uniform_deltas, delta_column="active_minus_uniform_mean_utility")
    random_delta_summary = summarize_deltas(random_deltas, delta_column="active_minus_random_mean_utility")
    dataset_delta_summary = summarize_deltas(dataset_deltas, delta_column="active_minus_dataset_mean_utility")
    embedding_delta_summary = summarize_deltas(embedding_deltas, delta_column="active_minus_embedding_mean_utility")
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "Outputs:",
        "",
        "- `table_active_calibration_replicates.csv`",
        "- `table_active_calibration_replicate_summary.csv`",
        "- `table_active_calibration_active_vs_uniform_deltas.csv`",
        "- `table_active_calibration_active_vs_random_deltas.csv`",
        "- `table_active_calibration_active_vs_dataset_deltas.csv`",
        "- `table_active_calibration_active_vs_embedding_deltas.csv`",
        "- `m6_active_calibration_replicates_memo.md`",
        "",
        "Replicate summary:",
        "",
        _markdown_table(summary),
        "",
        "Active vs uniform deltas:",
        "",
        _markdown_table(uniform_delta_summary),
        "",
        "Active vs random deltas:",
        "",
        _markdown_table(random_delta_summary),
        "",
        "Active vs dataset deltas:",
        "",
        _markdown_table(dataset_delta_summary),
        "",
        "Active vs embedding deltas:",
        "",
        _markdown_table(embedding_delta_summary),
        "",
        "Interpretation:",
        "",
        interpret_deltas(uniform_deltas, delta_column="active_minus_uniform_mean_utility", baseline_label="uniform"),
        "",
        interpret_deltas(random_deltas, delta_column="active_minus_random_mean_utility", baseline_label="random"),
        "",
        interpret_deltas(dataset_deltas, delta_column="active_minus_dataset_mean_utility", baseline_label="dataset"),
        "",
        interpret_deltas(
            embedding_deltas,
            delta_column="active_minus_embedding_mean_utility",
            baseline_label="embedding",
        ),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _load_active_calibration_runner():
    path = ROOT / "experiments" / "55_active_new_model_calibration.py"
    spec = importlib.util.spec_from_file_location("phase2_active_new_model_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load active calibration runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command(
    config_path: str,
    output_dir: str,
    max_holdout_models: int,
    r_values: list[int] | None,
    seeds: list[int],
) -> str:
    parts = [
        "python experiments/61_active_calibration_replicates.py",
        f"--config {config_path}",
        f"--output-dir {output_dir}",
        f"--max-holdout-models {max_holdout_models}",
        "--seeds " + ",".join(str(seed) for seed in seeds),
    ]
    if r_values:
        parts.append("--r-values " + ",".join(str(value) for value in r_values))
    return " ".join(parts)


def _parse_int_list(raw: str) -> list[int] | None:
    values = [int(value) for value in raw.split(",") if value]
    return values or None


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value: Any = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
