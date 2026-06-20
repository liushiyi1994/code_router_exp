from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.new_model_calibration import (
    budgeted_direct_oracle_labels,
    calibrate_new_model_by_label,
    fit_predict_budgeted_direct_router,
    sample_calibration_queries_per_label,
    selection_from_label_mapping,
)
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    seed = int(config.get("run", {}).get("random_seed", 0))
    bootstrap = config.get("bootstrap", {})
    route_config = config.get("routecode", {})
    d2_config = config.get("predictability_constrained", {})
    probe_config = config.get("stronger_direct_router_probe", {})

    k = int(probe_config.get("k", d2_config.get("k", route_config.get("selected_k_for_cards", 16))))
    alpha = float(probe_config.get("alpha", d2_config.get("selected_alpha", 3.0)))
    beta = float(probe_config.get("beta", d2_config.get("beta", 0.0)))
    r_values = [int(value) for value in probe_config.get("r_values", [8, 64])]
    max_holdout_models = int(probe_config.get("max_holdout_models", 2))
    direct_methods = [
        str(method)
        for method in probe_config.get(
            "direct_router_methods",
            ["logistic", "svm", "knn", "mlp", "gradient_boosting"],
        )
    ]
    max_iter = int(probe_config.get("max_iter", d2_config.get("max_iter", route_config.get("max_iter", 25))))
    refinement_iter = int(probe_config.get("refinement_iter", d2_config.get("refinement_iter", 10)))
    direct_max_iter = int(probe_config.get("direct_router_max_iter", 200))
    direct_knn_k = int(probe_config.get("direct_router_knn_k", config.get("routers", {}).get("knn_k", 15)))
    n_bootstrap = int(bootstrap.get("n_bootstrap", 100))
    ci = float(bootstrap.get("ci", 0.95))
    holdout_models = _holdout_models(probe_config, train.utility, max_holdout_models)

    best_single_full = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single_full).mean())
    full_oracle_selected = OracleRouter().predict(test.utility)
    full_oracle_mean = float(selected_values(test.utility, full_oracle_selected).mean())
    learned_reference_mean = baseline_mean

    rows: list[dict[str, Any]] = []
    for holdout_index, new_model_id in enumerate(holdout_models):
        base_models = [model for model in train.utility.columns if model != new_model_id]
        if new_model_id not in train.utility.columns or not base_models:
            continue
        train_base_utility = train.utility.loc[:, base_models]
        codebook = PredictabilityConstrainedRouteCode(
            n_labels=k,
            alpha=alpha,
            beta=beta,
            random_state=seed + holdout_index,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
        ).fit(train.query_info, train_base_utility, embeddings)
        if codebook.train_labels_ is None or codebook.label_utility_ is None or codebook.fallback_model is None:
            raise RuntimeError("D2 codebook failed to produce train labels")
        train_labels = codebook.train_labels_
        test_labels = codebook.predict_labels(embeddings.loc[test.utility.index])

        for r_index, examples_per_label in enumerate(r_values):
            calibration_ids = sample_calibration_queries_per_label(
                train_labels,
                examples_per_label=examples_per_label,
                seed=seed + 1000 * holdout_index + r_index,
            )
            label_calibration = calibrate_new_model_by_label(
                labels=train_labels,
                base_label_utility=codebook.label_utility_,
                full_utility=train.utility,
                new_model_id=new_model_id,
                calibration_query_ids=calibration_ids,
            )
            routecode_selected = selection_from_label_mapping(
                test_labels,
                label_calibration.label_to_model,
                codebook.fallback_model,
            )
            rows.append(
                _row(
                    method="routecode_label_calibration",
                    selected_models=routecode_selected,
                    test=test,
                    baseline_mean=baseline_mean,
                    learned_reference_mean=learned_reference_mean,
                    oracle_mean=full_oracle_mean,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    seed=seed + 2000 * holdout_index + r_index,
                    k=codebook.effective_labels,
                    labels=test_labels,
                    new_model_id=new_model_id,
                    examples_per_label=examples_per_label,
                    calibration_query_count=label_calibration.calibration_query_count,
                    direct_router_method="",
                )
            )

            direct_train_labels = budgeted_direct_oracle_labels(
                base_utility=train_base_utility,
                full_utility=train.utility,
                new_model_id=new_model_id,
                calibration_query_ids=calibration_ids,
            )
            for direct_index, direct_method in enumerate(direct_methods):
                direct_selected = fit_predict_budgeted_direct_router(
                    method=direct_method,
                    train_labels=direct_train_labels,
                    train_embeddings=embeddings.loc[train.utility.index],
                    test_embeddings=embeddings.loc[test.utility.index],
                    random_state=seed + holdout_index,
                    max_iter=direct_max_iter,
                    n_neighbors=direct_knn_k,
                )
                rows.append(
                    _row(
                        method=f"direct_retraining_budgeted_{direct_method}",
                        selected_models=direct_selected,
                        test=test,
                        baseline_mean=baseline_mean,
                        learned_reference_mean=learned_reference_mean,
                        oracle_mean=full_oracle_mean,
                        n_bootstrap=n_bootstrap,
                        ci=ci,
                        seed=seed + 3000 * holdout_index + 100 * direct_index + r_index,
                        new_model_id=new_model_id,
                        examples_per_label=examples_per_label,
                        calibration_query_count=label_calibration.calibration_query_count,
                        direct_router_method=direct_method,
                    )
                )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_stronger_direct_router_probe.csv", index=False)
    write_memo(out_dir, config_path, table, k, alpha, beta, max_holdout_models)
    append_readme(out_dir, config_path, table)
    print(f"Wrote stronger direct-router probe outputs to {out_dir}")


def _holdout_models(config: dict, utility: pd.DataFrame, max_holdout_models: int) -> list[str]:
    configured = config.get("holdout_models")
    if configured:
        return [str(model) for model in configured[:max_holdout_models]]
    winners = utility.idxmax(axis=1).value_counts()
    if winners.empty:
        return [str(model) for model in utility.columns[:max_holdout_models]]
    return [str(model) for model in winners.head(max_holdout_models).index]


def _row(
    *,
    method: str,
    selected_models: pd.Series,
    test,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    new_model_id: str,
    examples_per_label: int,
    calibration_query_count: int,
    direct_router_method: str,
    k: int | None = None,
    labels: pd.Series | None = None,
) -> dict[str, Any]:
    row = evaluate_selection(
        method=method,
        selected_models=selected_models,
        matrices=test,
        baseline_mean=baseline_mean,
        learned_reference_mean=learned_reference_mean,
        oracle_mean=oracle_mean,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        k=k,
        labels=labels,
    )
    row.update(
        {
            "probe_scope": "bounded_stronger_direct_router_probe",
            "new_model_id": new_model_id,
            "examples_per_label": int(examples_per_label),
            "calibration_query_count": int(calibration_query_count),
            "new_model_evaluations": int(calibration_query_count),
            "direct_router_method": direct_router_method,
        }
    )
    return row


def write_memo(
    out_dir: Path,
    config_path: str,
    table: pd.DataFrame,
    k: int,
    alpha: float,
    beta: float,
    max_holdout_models: int,
) -> None:
    summary = _summary_table(table)
    direct_rows = table[table["method"].astype(str).str.startswith("direct_retraining_budgeted_")]
    routecode_rows = table[table["method"].eq("routecode_label_calibration")]
    lines = [
        "# Phase E Stronger Direct-Router Probe Memo",
        "",
        f"Command: `python experiments/23_stronger_direct_router_probe.py --config {config_path}`",
        "",
        "This is a bounded stronger direct-router probe for the held-out/new-model calibration setting. It uses existing outcome tables and makes no external API calls.",
        "",
        f"Route labels: predictability-constrained RouteCode, K = {k}, alpha = {alpha:g}, beta = {beta:g}.",
        f"Hold-out models are capped at `{max_holdout_models}` by configuration.",
        "",
        "## Mean Across Probe Rows",
        "",
        _markdown_table(summary),
        "",
        "## Readout",
        "",
    ]
    if routecode_rows.empty:
        lines.append("- No RouteCode label-calibration rows were produced.")
    else:
        best_routecode = routecode_rows.sort_values("mean_utility", ascending=False).iloc[0]
        lines.append(
            "- Best RouteCode probe row: "
            f"`{best_routecode['new_model_id']}` at r `{int(best_routecode['examples_per_label'])}`, "
            f"mean utility `{float(best_routecode['mean_utility']):.4f}`."
        )
    if direct_rows.empty:
        lines.append("- No direct-router rows were produced.")
    else:
        best_direct = direct_rows.sort_values("mean_utility", ascending=False).iloc[0]
        methods = sorted(str(method) for method in direct_rows["direct_router_method"].dropna().unique() if method)
        lines.append("- Direct-router methods: `" + "`, `".join(methods) + "`.")
        lines.append(
            "- Best direct-router probe row: "
            f"`{best_direct['method']}` for `{best_direct['new_model_id']}` at r "
            f"`{int(best_direct['examples_per_label'])}`, mean utility `{float(best_direct['mean_utility']):.4f}`."
        )
    lines.extend(
        [
            "- This bounded probe extends the direct-router comparison beyond logistic/SVM/kNN, but it is not a full calibration sweep.",
            "",
        ]
    )
    (out_dir / "phase_e_stronger_direct_router_probe_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Stronger Direct-Router Probe\n"
    marker = "## Stronger Direct-Router Probe"
    compact = table[
        [
            "method",
            "new_model_id",
            "examples_per_label",
            "calibration_query_count",
            "mean_utility",
            "recovered_gap_vs_oracle",
        ]
    ].sort_values(["new_model_id", "method", "examples_per_label"]) if not table.empty else table
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/23_stronger_direct_router_probe.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_stronger_direct_router_probe.csv`: bounded held-out/new-model rows for RouteCode label calibration and stronger direct-router retraining baselines.",
        "- `phase_e_stronger_direct_router_probe_memo.md`: probe interpretation memo.",
        "",
        _markdown_table(compact),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    return (
        table.groupby(["method", "examples_per_label"], as_index=False)
        .agg(
            mean_utility=("mean_utility", "mean"),
            recovered_gap_vs_oracle=("recovered_gap_vs_oracle", "mean"),
            calibration_query_count=("calibration_query_count", "mean"),
        )
        .sort_values(["method", "examples_per_label"])
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
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
