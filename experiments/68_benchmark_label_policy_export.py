from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from routecode.config import load_config
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query-model-utility", required=True)
    parser.add_argument("--output-dir", default="results/phase2/benchmark_label_policy")
    parser.add_argument("--name", default="exact_math_qwen_intern")
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--selection-basis", default="targeted_exact_math_benchmark_label_rule")
    parser.add_argument("--dataset-model", action="append", required=True, help="dataset=model")
    parser.add_argument("--require-within-threshold", action="store_true")
    args = parser.parse_args()
    paths = run(
        config_path=args.config,
        query_model_utility_path=args.query_model_utility,
        output_dir=args.output_dir,
        name=args.name,
        threshold=args.threshold,
        selection_basis=args.selection_basis,
        dataset_model_specs=args.dataset_model,
        require_within_threshold=args.require_within_threshold,
    )
    print(f"Wrote benchmark-label policy summary to {paths['summary']}")


def run(
    *,
    config_path: str,
    query_model_utility_path: str,
    output_dir: str,
    name: str,
    threshold: float,
    selection_basis: str,
    dataset_model_specs: list[str],
    require_within_threshold: bool = False,
) -> dict[str, str]:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    query_model_utility = _read_matrix(query_model_utility_path)
    query_info = prepared.outcomes.drop_duplicates("query_id").set_index("query_id")
    missing_queries = sorted(set(query_model_utility.index) - set(query_info.index))
    if missing_queries:
        raise ValueError(f"Missing query metadata for {len(missing_queries)} query ids")
    mapping = _parse_dataset_model_specs(dataset_model_specs)
    selections, summary = evaluate_dataset_model_rule(
        query_info=query_info.loc[query_model_utility.index],
        query_model_utility=query_model_utility,
        mapping=mapping,
        name=name,
        threshold=threshold,
        selection_basis=selection_basis,
    )
    if require_within_threshold and not bool(summary.loc[0, "within_threshold"]):
        raise SystemExit(
            f"{name} failed threshold {threshold:.4f}: relative gap "
            f"{float(summary.loc[0, 'relative_gap_to_oracle']):.4f}",
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selections_path = out_dir / "table_policy_selections.csv"
    summary_path = out_dir / "table_policy_summary.csv"
    readme_path = out_dir / "README.md"
    selections.to_csv(selections_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_readme(readme_path, summary, selections, mapping)
    return {"selections": str(selections_path), "summary": str(summary_path), "readme": str(readme_path)}


def evaluate_dataset_model_rule(
    *,
    query_info: pd.DataFrame,
    query_model_utility: pd.DataFrame,
    mapping: dict[str, str],
    name: str,
    threshold: float,
    selection_basis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_info = query_info.reindex(query_model_utility.index)
    route_labels = query_info["dataset"].astype(str)
    selected_models = route_labels.map(mapping)
    if selected_models.isna().any():
        missing = sorted(route_labels[selected_models.isna()].unique())
        raise ValueError(f"Dataset-model rule missing datasets: {missing}")
    unknown_models = sorted(set(selected_models) - set(query_model_utility.columns))
    if unknown_models:
        raise ValueError(f"Dataset-model rule selected unknown models: {unknown_models}")
    selected_utility = pd.Series(
        [
            float(query_model_utility.loc[query_id, model_id])
            for query_id, model_id in selected_models.items()
        ],
        index=query_model_utility.index,
        name="selected_utility",
    )
    oracle_model = query_model_utility.idxmax(axis=1).rename("oracle_model")
    oracle_utility = query_model_utility.max(axis=1).rename("oracle_utility")
    regret = (oracle_utility - selected_utility).rename("oracle_regret")
    selections = pd.DataFrame(
        {
            "query_id": query_model_utility.index,
            "route_label": route_labels.to_numpy(),
            "selected_model": selected_models.to_numpy(),
            "selected_utility": selected_utility.to_numpy(),
            "oracle_model": oracle_model.to_numpy(),
            "oracle_utility": oracle_utility.to_numpy(),
            "oracle_regret": regret.to_numpy(),
            "within_oracle": (regret <= 1e-12).to_numpy(),
            "policy_name": name,
            "selection_basis": selection_basis,
        }
    )
    oracle_mean = float(oracle_utility.mean())
    selected_mean = float(selected_utility.mean())
    relative_gap = float((oracle_mean - selected_mean) / oracle_mean) if oracle_mean else np.nan
    summary = pd.DataFrame(
        [
            {
                "policy_name": name,
                "candidate_type": "benchmark_label_route_rule",
                "selection_basis": selection_basis,
                "n_queries": int(len(query_model_utility)),
                "mean_utility": selected_mean,
                "oracle_mean_utility": oracle_mean,
                "abs_gap_to_oracle": float(oracle_mean - selected_mean),
                "relative_gap_to_oracle": relative_gap,
                "threshold": float(threshold),
                "within_threshold": bool(relative_gap <= threshold),
                "regret_count": int((regret > 1e-12).sum()),
                "mean_oracle_regret": float(regret.mean()),
                "route_labels": ",".join(sorted(mapping)),
                "selected_models": ",".join(mapping[label] for label in sorted(mapping)),
                "method_caveat": (
                    "Operational benchmark-label route rule; not the core latent-state "
                    "RouteCode/ProbeRoute++ method and not a paper-level claim."
                ),
            }
        ]
    )
    return selections, summary


def write_readme(
    path: Path,
    summary: pd.DataFrame,
    selections: pd.DataFrame,
    mapping: dict[str, str],
) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Benchmark-Label Policy Export\n"
    marker = "## Export"
    row = summary.iloc[0]
    mapping_text = ", ".join(f"`{dataset} -> {model}`" for dataset, model in sorted(mapping.items()))
    lines = [
        marker,
        "",
        "This is an operational benchmark-label route rule, not the core latent-state RouteCode/ProbeRoute++ method.",
        "",
        f"Policy: `{row['policy_name']}`",
        f"Mapping: {mapping_text}",
        f"Queries: `{int(row['n_queries'])}`",
        f"Mean utility: `{float(row['mean_utility']):.4f}`",
        f"Oracle mean utility: `{float(row['oracle_mean_utility']):.4f}`",
        f"Relative gap to oracle: `{float(row['relative_gap_to_oracle']):.4f}`",
        f"Within threshold: `{bool(row['within_threshold'])}` at threshold `{float(row['threshold']):.4f}`",
        f"Regret rows: `{int(row['regret_count'])}/{int(row['n_queries'])}`",
        "",
        "Outputs:",
        "",
        "- `table_policy_summary.csv`",
        "- `table_policy_selections.csv`",
        "",
        "Selection table columns include `query_id`, `route_label`, `selected_model`, `selected_utility`, `oracle_model`, `oracle_utility`, and `oracle_regret`.",
        "",
        "Route-label distribution:",
        "",
        _markdown_table(selections["route_label"].value_counts().rename_axis("route_label").reset_index(name="n_queries")),
        "",
    ]
    path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _parse_dataset_model_specs(specs: list[str]) -> dict[str, str]:
    mapping = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected dataset=model, got {spec}")
        dataset, model = spec.split("=", 1)
        dataset = dataset.strip()
        model = model.strip()
        if not dataset or not model:
            raise ValueError(f"Expected dataset=model, got {spec}")
        mapping[dataset] = model
    if not mapping:
        raise ValueError("At least one --dataset-model mapping is required")
    return mapping


def _read_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    return frame


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
