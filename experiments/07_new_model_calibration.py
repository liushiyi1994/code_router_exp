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
from routecode.plots import save_transfer_calibration_curve
from routecode.predictors.classifiers import LogisticModelRouter
from routecode.reporting import upsert_markdown_section
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    seed = int(config.get("run", {}).get("random_seed", 0))
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    route_config = config.get("routecode", {})
    d2_config = config.get("predictability_constrained", {})
    calibration_config = config.get("new_model_calibration", {})

    k = int(calibration_config.get("k", d2_config.get("k", route_config.get("selected_k_for_cards", 16))))
    alpha = float(calibration_config.get("alpha", d2_config.get("selected_alpha", 3.0)))
    beta = float(calibration_config.get("beta", d2_config.get("beta", 0.0)))
    r_values = [int(value) for value in calibration_config.get("r_values", [1, 2, 4, 8, 16, 32, 64])]
    max_iter = int(calibration_config.get("max_iter", d2_config.get("max_iter", route_config.get("max_iter", 25))))
    refinement_iter = int(calibration_config.get("refinement_iter", d2_config.get("refinement_iter", 10)))
    direct_max_iter = int(calibration_config.get("direct_router_max_iter", 1000))
    direct_methods = [str(method) for method in calibration_config.get("direct_router_methods", ["logistic", "svm", "knn"])]
    direct_knn_k = int(calibration_config.get("direct_router_knn_k", config.get("routers", {}).get("knn_k", 15)))
    holdout_models = _holdout_models(calibration_config, train.utility)

    best_single_full = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single_full).mean())
    full_oracle_mean = float(test.utility.max(axis=1).mean())
    full_learned = LogisticModelRouter(random_state=seed).fit(train.query_info, train.utility, embeddings).predict(
        test.query_info,
        embeddings,
    )
    learned_reference_mean = max(
        baseline_mean,
        float(selected_values(test.utility, full_learned).mean()),
    )

    rows: list[dict[str, Any]] = []
    for holdout_index, new_model_id in enumerate(holdout_models):
        if new_model_id not in train.utility.columns:
            raise ValueError(f"Configured holdout model not found: {new_model_id}")
        base_models = [model for model in train.utility.columns if model != new_model_id]
        if not base_models:
            continue
        train_base_utility = train.utility.loc[:, base_models]
        codebook = PredictabilityConstrainedRouteCode(
            n_labels=k,
            alpha=alpha,
            beta=beta,
            random_state=seed,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
        ).fit(train.query_info, train_base_utility, embeddings)
        train_labels = codebook.train_labels_
        if train_labels is None or codebook.label_utility_ is None or codebook.fallback_model is None:
            raise RuntimeError("D2 codebook failed to produce train labels")
        test_labels = codebook.predict_labels(embeddings.loc[test.utility.index])

        base_selected = codebook.predict_from_labels(test_labels)
        rows.append(
            _row(
                method="routecode_no_new_model",
                selected_models=base_selected,
                test=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=full_oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 1000 * holdout_index,
                k=k,
                labels=test_labels,
                new_model_id=new_model_id,
                examples_per_label=0,
                calibration_query_count=0,
            )
        )

        full_oracle_selected = OracleRouter().predict(test.utility)
        rows.append(
            _row(
                method="query_oracle_with_new_model",
                selected_models=full_oracle_selected,
                test=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=full_oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 1000 * holdout_index + 1,
                new_model_id=new_model_id,
                examples_per_label="full",
                calibration_query_count=len(train.utility),
            )
        )

        always_new = pd.Series(new_model_id, index=test.utility.index, name="selected_model")
        rows.append(
            _row(
                method="new_model_always",
                selected_models=always_new,
                test=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=full_oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 1000 * holdout_index + 2,
                new_model_id=new_model_id,
                examples_per_label="always",
                calibration_query_count=0,
            )
        )

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
            routecode_row = _row(
                method="routecode_label_calibration",
                selected_models=routecode_selected,
                test=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=full_oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 2000 * holdout_index + r_index,
                k=k,
                labels=test_labels,
                new_model_id=new_model_id,
                examples_per_label=examples_per_label,
                calibration_query_count=label_calibration.calibration_query_count,
            )
            routecode_row["labels_switching_to_new_model"] = sum(
                1 for model in label_calibration.label_to_model.values() if model == new_model_id
            )
            rows.append(routecode_row)

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
                    random_state=seed,
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
                    )
                )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_new_model_integration.csv", index=False)
    save_transfer_calibration_curve(table, out_dir / "fig_transfer_calibration_curve.pdf")
    write_memo(out_dir, args.config, table, k, alpha, beta)
    append_readme(out_dir, args.config, table)
    print(f"Wrote new-model calibration outputs to {out_dir}")


def _holdout_models(config: dict, utility: pd.DataFrame) -> list[str]:
    configured = config.get("holdout_models")
    if configured:
        return [str(model) for model in configured]
    max_models = int(config.get("max_holdout_models", 2))
    winners = utility.idxmax(axis=1).value_counts()
    if winners.empty:
        return [str(model) for model in utility.columns[:max_models]]
    return [str(model) for model in winners.head(max_models).index]


def _row(
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
    examples_per_label: int | str,
    calibration_query_count: int,
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
            "new_model_id": new_model_id,
            "examples_per_label": examples_per_label,
            "calibration_query_count": calibration_query_count,
            "new_model_evaluations": calibration_query_count,
            "labels_switching_to_new_model": "",
        }
    )
    return row


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, k: int, alpha: float, beta: float) -> None:
    comparable = table[
        (table["method"] == "routecode_label_calibration")
        | table["method"].astype(str).str.startswith("direct_retraining_budgeted_")
    ].copy()
    summary = (
        comparable.groupby(["method", "examples_per_label"], as_index=False)
        .agg(
            mean_utility=("mean_utility", "mean"),
            recovered_gap_vs_oracle=("recovered_gap_vs_oracle", "mean"),
            calibration_query_count=("calibration_query_count", "mean"),
        )
        .sort_values(["method", "examples_per_label"])
    )
    best = comparable.sort_values("mean_utility", ascending=False).head(1)
    lines = [
        "# Phase D4/E5 New-Model Calibration Memo",
        "",
        f"Command: `python experiments/07_new_model_calibration.py --config {config_path}`",
        "",
        f"Route labels: predictability-constrained RouteCode, K = {k}, alpha = {alpha:g}, beta = {beta:g}.",
        "",
        "This is a simulated held-out-model calibration using existing outcome tables. It makes no external API calls.",
        "",
        "## Mean Across Held-Out Models",
        "",
        _markdown_table(summary),
        "",
        "## Current Readout",
        "",
    ]
    holdouts = sorted(str(model) for model in table["new_model_id"].dropna().unique())
    direct_methods = sorted(
        method.replace("direct_retraining_budgeted_", "")
        for method in table["method"].astype(str).unique()
        if method.startswith("direct_retraining_budgeted_")
    )
    lines.append("- Held-out/new models: `" + "`, `".join(holdouts) + "`.")
    lines.append("- Direct retraining baselines: `" + "`, `".join(direct_methods) + "`.")
    if best.empty:
        lines.append("- No calibration rows were produced.")
    else:
        row = best.iloc[0]
        lines.append(
            "- Best budgeted row: "
            f"`{row['method']}` for `{row['new_model_id']}` at r `{row['examples_per_label']}`, "
            f"mean utility `{float(row['mean_utility']):.4f}` with "
            f"`{int(row['calibration_query_count'])}` new-model evaluations."
        )
    lines.extend(
        [
            "- Interpret this as a sample-efficiency diagnostic only after comparing against the direct retraining curve.",
            "- A strong claim requires RouteCode to reach competitive utility with fewer new-model evaluations than direct retraining across held-out models.",
            "",
        ]
    )
    (out_dir / "phase_e5_new_model_calibration_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## New-Model Calibration"
    compact = table[
        (table["method"] == "routecode_label_calibration")
        | table["method"].astype(str).str.startswith("direct_retraining_budgeted_")
    ][
        [
            "method",
            "new_model_id",
            "examples_per_label",
            "calibration_query_count",
            "mean_utility",
            "recovered_gap_vs_oracle",
        ]
    ].sort_values(["new_model_id", "method", "examples_per_label"])
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/07_new_model_calibration.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_new_model_integration.csv`: held-out/new-model calibration sweep.",
        "- `fig_transfer_calibration_curve.pdf`: utility vs new-model calibration evaluations.",
        "- `phase_e5_new_model_calibration_memo.md`: D4/E5 checkpoint memo.",
        "",
        _markdown_table(compact),
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
    return str(value)


if __name__ == "__main__":
    main()
