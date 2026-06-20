from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from routecode.config import load_config
from routecode.reporting import upsert_markdown_section


BASELINE_METHODS = [
    "uniform_route_state_calibration",
    "random_route_state_calibration",
    "dataset_stratified_calibration",
    "embedding_cluster_calibration",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--source-table", default="")
    parser.add_argument("--output-dir", default="results/phase2")
    parser.add_argument("--max-holdout-models", type=int, default=3)
    parser.add_argument("--k-values", default="")
    parser.add_argument("--alpha-values", default="")
    parser.add_argument("--r-values", default="1,4,8")
    parser.add_argument("--seeds", default="0,1")
    args = parser.parse_args()
    run(
        config_path=args.config or None,
        source_table_path=args.source_table or None,
        output_dir=args.output_dir,
        max_holdout_models=args.max_holdout_models,
        k_values=_parse_int_list(args.k_values),
        alpha_values=_parse_float_list(args.alpha_values),
        r_values=_parse_int_list(args.r_values),
        seeds=_parse_int_list(args.seeds) or [0, 1],
    )


def run(
    *,
    output_dir: str,
    config_path: str | None = None,
    source_table_path: str | None = None,
    max_holdout_models: int = 3,
    k_values: list[int] | None = None,
    alpha_values: list[float] | None = None,
    r_values: list[int] | None = None,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source_table_path:
        table = pd.read_csv(source_table_path)
        command = (
            "python experiments/62_active_calibration_sensitivity.py "
            f"--source-table {source_table_path} --output-dir {output_dir}"
        )
    elif config_path:
        table = run_sensitivity_from_config(
            config_path=config_path,
            output_dir=output_dir,
            max_holdout_models=max_holdout_models,
            k_values=k_values,
            alpha_values=alpha_values,
            r_values=r_values,
            seeds=seeds,
        )
        command = _command(config_path, output_dir, max_holdout_models, k_values, alpha_values, r_values, seeds or [0, 1])
    else:
        table = pd.DataFrame(
            [
                {
                    "sensitivity_name": "blocked_missing_config_or_source_table",
                    "sensitivity_k": pd.NA,
                    "sensitivity_alpha": pd.NA,
                    "replicate_seed": pd.NA,
                    "method": "active_route_state_calibration",
                    "status": "blocked_missing_config_or_source_table",
                    "new_model_id": "",
                    "examples_per_label": "",
                    "new_model_evaluations": 0,
                    "mean_utility": pd.NA,
                }
            ]
        )
        command = f"python experiments/62_active_calibration_sensitivity.py --output-dir {output_dir}"

    table = _ensure_sensitivity_columns(table)
    summary = summarize_sensitivity(table)
    deltas = summarize_active_deltas(table)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, table, summary, deltas, command)
    print(f"Wrote active-calibration sensitivity rows to {out_dir / 'table_active_calibration_sensitivity.csv'}")
    print(f"Wrote active-calibration sensitivity summary to {out_dir / 'table_active_calibration_sensitivity_summary.csv'}")
    print(f"Wrote active-calibration sensitivity deltas to {out_dir / 'table_active_calibration_sensitivity_deltas.csv'}")
    return table, summary, deltas


def run_sensitivity_from_config(
    *,
    config_path: str,
    output_dir: str,
    max_holdout_models: int,
    k_values: list[int] | None,
    alpha_values: list[float] | None,
    r_values: list[int] | None,
    seeds: list[int] | None,
) -> pd.DataFrame:
    config = load_config(config_path)
    calibration_config = config.get("new_model_calibration", {})
    d2_config = config.get("predictability_constrained", {})
    route_config = config.get("routecode", {})
    resolved_k_values = k_values or [
        int(calibration_config.get("k", d2_config.get("k", route_config.get("selected_k_for_cards", 16))))
    ]
    resolved_alpha_values = alpha_values or [float(calibration_config.get("alpha", d2_config.get("selected_alpha", 3.0)))]
    resolved_r_values = r_values or [int(value) for value in calibration_config.get("r_values", [1, 4, 8])]
    resolved_seeds = seeds or [int(config.get("run", {}).get("random_seed", 0))]
    runner = _load_active_calibration_runner()
    generated_config_dir = Path(output_dir) / "_active_calibration_sensitivity_configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []
    for k in resolved_k_values:
        for alpha in resolved_alpha_values:
            for seed in resolved_seeds:
                variant_name = _variant_name(k, alpha)
                variant_config = deepcopy(config)
                variant_config.setdefault("run", {})["random_seed"] = int(seed)
                variant_calibration = variant_config.setdefault("new_model_calibration", {})
                variant_calibration["k"] = int(k)
                variant_calibration["alpha"] = float(alpha)
                variant_calibration["r_values"] = [int(value) for value in resolved_r_values]
                variant_config_path = generated_config_dir / f"{variant_name}_seed_{seed}.yaml"
                variant_config_path.write_text(yaml.safe_dump(variant_config, sort_keys=False), encoding="utf-8")
                table = runner.run_active_calibration_from_config(
                    config_path=str(variant_config_path),
                    max_holdout_models=max_holdout_models,
                    r_values=resolved_r_values,
                    seed_override=int(seed),
                )
                table = table.copy()
                table.insert(0, "replicate_seed", int(seed))
                table.insert(0, "sensitivity_alpha", float(alpha))
                table.insert(0, "sensitivity_k", int(k))
                table.insert(0, "sensitivity_name", variant_name)
                tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def summarize_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "mean_utility" not in table.columns:
        return pd.DataFrame()
    numeric = table.copy()
    for column in ["mean_utility", "new_model_evaluations", "recovered_gap_vs_oracle"]:
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    group_columns = [
        column
        for column in [
            "sensitivity_name",
            "sensitivity_k",
            "sensitivity_alpha",
            "method",
            "new_model_id",
            "examples_per_label",
        ]
        if column in numeric.columns
    ]
    if not group_columns:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "replicates": ("mean_utility", "count"),
        "mean_utility_mean": ("mean_utility", "mean"),
        "mean_utility_std": ("mean_utility", "std"),
        "mean_utility_min": ("mean_utility", "min"),
        "mean_utility_max": ("mean_utility", "max"),
    }
    if "new_model_evaluations" in numeric.columns:
        aggregations.update(
            {
                "new_model_evaluations_mean": ("new_model_evaluations", "mean"),
                "new_model_evaluations_min": ("new_model_evaluations", "min"),
                "new_model_evaluations_max": ("new_model_evaluations", "max"),
            }
        )
    if "recovered_gap_vs_oracle" in numeric.columns:
        aggregations["recovered_gap_vs_oracle_mean"] = ("recovered_gap_vs_oracle", "mean")
    summary = numeric.dropna(subset=["mean_utility"]).groupby(group_columns, as_index=False).agg(**aggregations)
    return _round_numeric(summary).sort_values(group_columns).reset_index(drop=True)


def summarize_active_deltas(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sensitivity_name",
        "sensitivity_k",
        "sensitivity_alpha",
        "replicate_seed",
        "method",
        "new_model_id",
        "examples_per_label",
        "new_model_evaluations",
        "mean_utility",
    }
    if table.empty or not required.issubset(table.columns):
        return pd.DataFrame()
    numeric = table.copy()
    numeric["mean_utility"] = pd.to_numeric(numeric["mean_utility"], errors="coerce")
    index_columns = [
        "sensitivity_name",
        "sensitivity_k",
        "sensitivity_alpha",
        "replicate_seed",
        "new_model_id",
        "examples_per_label",
        "new_model_evaluations",
    ]
    rows: list[dict[str, Any]] = []
    for baseline in BASELINE_METHODS:
        relevant = numeric[numeric["method"].isin(["active_route_state_calibration", baseline])]
        if relevant.empty:
            continue
        wide = relevant.pivot_table(index=index_columns, columns="method", values="mean_utility", aggfunc="mean").reset_index()
        if "active_route_state_calibration" not in wide.columns or baseline not in wide.columns:
            continue
        wide["active_minus_baseline"] = wide["active_route_state_calibration"] - wide[baseline]
        for group_key, group in wide.groupby(["sensitivity_name", "sensitivity_k", "sensitivity_alpha"], sort=True):
            values = pd.to_numeric(group["active_minus_baseline"], errors="coerce").dropna()
            if values.empty:
                continue
            sensitivity_name, sensitivity_k, sensitivity_alpha = group_key
            rows.append(
                {
                    "sensitivity_name": sensitivity_name,
                    "sensitivity_k": sensitivity_k,
                    "sensitivity_alpha": sensitivity_alpha,
                    "baseline": baseline,
                    "paired_rows": int(len(values)),
                    "active_minus_baseline_mean": float(values.mean()),
                    "active_minus_baseline_std": float(values.std()) if len(values) > 1 else 0.0,
                    "active_minus_baseline_min": float(values.min()),
                    "active_minus_baseline_max": float(values.max()),
                    "positive": int((values > 0).sum()),
                    "negative": int((values < 0).sum()),
                    "tied": int((values == 0).sum()),
                }
            )
    return _round_numeric(pd.DataFrame(rows))


def write_outputs(
    out_dir: Path,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    command: str,
) -> None:
    table.to_csv(out_dir / "table_active_calibration_sensitivity.csv", index=False)
    summary.to_csv(out_dir / "table_active_calibration_sensitivity_summary.csv", index=False)
    deltas.to_csv(out_dir / "table_active_calibration_sensitivity_deltas.csv", index=False)
    write_memo(out_dir, summary, deltas, command)
    append_readme(out_dir, summary, deltas, command)


def write_memo(out_dir: Path, summary: pd.DataFrame, deltas: pd.DataFrame, command: str) -> None:
    lines = [
        "# Phase 2 Active Calibration Sensitivity",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "This sweeps active new-model calibration state-learning settings under matched evaluation budgets.",
        "",
        "Outputs:",
        "",
        "- `table_active_calibration_sensitivity.csv`",
        "- `table_active_calibration_sensitivity_summary.csv`",
        "- `table_active_calibration_sensitivity_deltas.csv`",
        "- `m7_active_calibration_sensitivity_memo.md`",
        "",
        "Sensitivity Delta Summary:",
        "",
        _markdown_table(deltas),
        "",
        "Best Active Rows:",
        "",
        _markdown_table(_best_active_rows(summary)),
        "",
        "Interpretation:",
        "",
        interpret_deltas(deltas),
        "",
    ]
    (out_dir / "m7_active_calibration_sensitivity_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, summary: pd.DataFrame, deltas: pd.DataFrame, command: str) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Active Calibration Sensitivity"
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
        "- `table_active_calibration_sensitivity.csv`",
        "- `table_active_calibration_sensitivity_summary.csv`",
        "- `table_active_calibration_sensitivity_deltas.csv`",
        "- `m7_active_calibration_sensitivity_memo.md`",
        "",
        "Sensitivity delta summary:",
        "",
        _markdown_table(deltas),
        "",
        "Best active rows:",
        "",
        _markdown_table(_best_active_rows(summary)),
        "",
        "Interpretation:",
        "",
        interpret_deltas(deltas),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def interpret_deltas(deltas: pd.DataFrame) -> str:
    if deltas.empty or "active_minus_baseline_mean" not in deltas.columns:
        return "No active-vs-baseline sensitivity deltas were available."
    random_rows = deltas[deltas["baseline"] == "random_route_state_calibration"]
    if random_rows.empty:
        values = pd.to_numeric(deltas["active_minus_baseline_mean"], errors="coerce").dropna()
        return f"Across available sensitivity cells, mean active-vs-baseline delta is `{values.mean():.4f}`."
    values = pd.to_numeric(random_rows["active_minus_baseline_mean"], errors="coerce").dropna()
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    tied = int((values == 0).sum())
    return (
        "Across active-vs-random sensitivity cells, active calibration has mean cell delta "
        f"`{values.mean():.4f}` over `{len(values)}` cells "
        f"(`{positive}` positive, `{negative}` negative, `{tied}` tied)."
    )


def _ensure_sensitivity_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "sensitivity_k" not in table.columns and "k" in table.columns:
        table["sensitivity_k"] = table["k"]
    if "sensitivity_name" not in table.columns:
        if "sensitivity_k" in table.columns:
            table["sensitivity_name"] = [
                _variant_name(k, table.get("sensitivity_alpha", pd.Series([pd.NA] * len(table))).iloc[idx])
                for idx, k in enumerate(table["sensitivity_k"])
            ]
        else:
            table["sensitivity_name"] = "unspecified"
    if "sensitivity_k" not in table.columns:
        table["sensitivity_k"] = pd.NA
    if "sensitivity_alpha" not in table.columns:
        table["sensitivity_alpha"] = pd.NA
    if "replicate_seed" not in table.columns:
        table["replicate_seed"] = 0
    return table


def _best_active_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "method" not in summary.columns or "mean_utility_mean" not in summary.columns:
        return pd.DataFrame()
    active = summary[summary["method"] == "active_route_state_calibration"].copy()
    if active.empty:
        return pd.DataFrame()
    columns = [
        column
        for column in [
            "sensitivity_name",
            "sensitivity_k",
            "sensitivity_alpha",
            "new_model_id",
            "examples_per_label",
            "replicates",
            "mean_utility_mean",
            "mean_utility_std",
            "new_model_evaluations_mean",
        ]
        if column in active.columns
    ]
    return active.sort_values("mean_utility_mean", ascending=False).head(12)[columns]


def _variant_name(k: Any, alpha: Any) -> str:
    alpha_text = str(alpha).replace(".", "p")
    return f"k_{k}_alpha_{alpha_text}"


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
    k_values: list[int] | None,
    alpha_values: list[float] | None,
    r_values: list[int] | None,
    seeds: list[int],
) -> str:
    parts = [
        "python experiments/62_active_calibration_sensitivity.py",
        f"--config {config_path}",
        f"--output-dir {output_dir}",
        f"--max-holdout-models {max_holdout_models}",
    ]
    if k_values:
        parts.append("--k-values " + ",".join(str(value) for value in k_values))
    if alpha_values:
        parts.append("--alpha-values " + ",".join(str(value) for value in alpha_values))
    if r_values:
        parts.append("--r-values " + ",".join(str(value) for value in r_values))
    parts.append("--seeds " + ",".join(str(seed) for seed in seeds))
    return " ".join(parts)


def _parse_int_list(raw: str) -> list[int] | None:
    values = [int(value) for value in raw.split(",") if value]
    return values or None


def _parse_float_list(raw: str) -> list[float] | None:
    values = [float(value) for value in raw.split(",") if value]
    return values or None


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].round(6)
    return rounded


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
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
