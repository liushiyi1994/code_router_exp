from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.model_pool_transfer import (
    ModelPoolTransferScenario,
    build_model_pool_transfer_scenarios,
    fit_label_to_target_model,
    select_from_label_to_model,
)
from routecode.eval.new_model_calibration import fit_predict_budgeted_direct_router
from routecode.matrix import Matrices
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.knn import KNNRouter
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
    transfer_config = config.get("model_pool_transfer", {})
    d2_config = config.get("predictability_constrained", {})
    route_config = config.get("routecode", {})
    bootstrap = config.get("bootstrap", {})

    source_size = int(transfer_config.get("source_size", min(8, max(2, len(train.model_ids) // 2))))
    target_size = int(transfer_config.get("target_size", min(8, max(2, len(train.model_ids) - source_size))))
    source_sizes = [int(value) for value in transfer_config.get("source_sizes", [])] or None
    target_sizes = [int(value) for value in transfer_config.get("target_sizes", [])] or None
    k = int(transfer_config.get("k", d2_config.get("k", route_config.get("selected_k_for_cards", 16))))
    alpha = float(transfer_config.get("d2_alpha", d2_config.get("selected_alpha", 3.0)))
    beta = float(transfer_config.get("d2_beta", d2_config.get("beta", 0.0)))
    max_iter = int(transfer_config.get("max_iter", d2_config.get("max_iter", route_config.get("max_iter", 25))))
    refinement_iter = int(transfer_config.get("refinement_iter", d2_config.get("refinement_iter", 10)))
    direct_methods = [str(method) for method in transfer_config.get("direct_router_methods", ["logistic", "svm", "knn"])]
    direct_max_iter = int(transfer_config.get("direct_router_max_iter", 1000))
    direct_knn_k = int(transfer_config.get("direct_router_knn_k", config.get("routers", {}).get("knn_k", 15)))
    router_knn_k = int(config.get("routers", {}).get("knn_k", 15))
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))

    scenarios = build_model_pool_transfer_scenarios(
        train.utility,
        source_size=source_size,
        target_size=target_size,
        source_sizes=source_sizes,
        target_sizes=target_sizes,
    )
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        rows.extend(
            _scenario_rows(
                scenario=scenario,
                train=train,
                test=test,
                embeddings=embeddings,
                seed=seed + 100 * scenario_index,
                k=k,
                alpha=alpha,
                beta=beta,
                max_iter=max_iter,
                refinement_iter=refinement_iter,
                direct_methods=direct_methods,
                direct_max_iter=direct_max_iter,
                direct_knn_k=direct_knn_k,
                router_knn_k=router_knn_k,
                n_bootstrap=n_bootstrap,
                ci=ci,
            )
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_model_pool_transfer.csv", index=False)
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote model-pool transfer outputs to {out_dir}")


def _scenario_rows(
    *,
    scenario: ModelPoolTransferScenario,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    max_iter: int,
    refinement_iter: int,
    direct_methods: list[str],
    direct_max_iter: int,
    direct_knn_k: int,
    router_knn_k: int,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    source_train = _subset_matrices(train, scenario.source_models)
    target_train = _subset_matrices(train, scenario.target_models)
    target_test = _subset_matrices(test, scenario.target_models)

    best_selected = BestSingleRouter().fit(target_train.query_info, target_train.utility).predict(target_test.query_info)
    knn_selected = KNNRouter(router_knn_k).fit(target_train.query_info, target_train.utility, embeddings).predict(
        target_test.query_info,
        embeddings,
    )
    direct_selected: dict[str, pd.Series] = {}
    target_oracle_train_labels = target_train.utility.idxmax(axis=1).astype(str).rename("selected_model")
    for method in direct_methods:
        direct_selected[method] = fit_predict_budgeted_direct_router(
            method=method,
            train_labels=target_oracle_train_labels,
            train_embeddings=embeddings.loc[target_train.utility.index],
            test_embeddings=embeddings.loc[target_test.utility.index],
            random_state=seed,
            max_iter=direct_max_iter,
            n_neighbors=direct_knn_k,
        )

    source_codebook = PredictabilityConstrainedRouteCode(
        n_labels=k,
        alpha=alpha,
        beta=beta,
        random_state=seed,
        max_iter=max_iter,
        refinement_iter=refinement_iter,
    ).fit(source_train.query_info, source_train.utility, embeddings)
    if source_codebook.train_labels_ is None:
        raise RuntimeError("Source D2 codebook did not produce train labels")
    transfer_mapping, transfer_fallback = fit_label_to_target_model(
        source_codebook.train_labels_,
        target_train.utility,
        labels=range(source_codebook.effective_labels),
    )
    source_test_labels = source_codebook.predict_labels(embeddings.loc[target_test.utility.index])
    transfer_selected = select_from_label_to_model(source_test_labels, transfer_mapping, transfer_fallback)

    native_codebook = PredictabilityConstrainedRouteCode(
        n_labels=k,
        alpha=alpha,
        beta=beta,
        random_state=seed + 1,
        max_iter=max_iter,
        refinement_iter=refinement_iter,
    ).fit(target_train.query_info, target_train.utility, embeddings)
    native_test_labels = native_codebook.predict_labels(embeddings.loc[target_test.utility.index])
    native_selected = native_codebook.predict_from_labels(native_test_labels)

    baseline_mean = float(selected_values(target_test.utility, best_selected).mean())
    oracle_mean = float(target_test.utility.max(axis=1).mean())
    learned_reference_mean = max(
        [baseline_mean, float(selected_values(target_test.utility, knn_selected).mean())]
        + [float(selected_values(target_test.utility, selected).mean()) for selected in direct_selected.values()]
    )

    selected_rows: list[tuple[str, pd.Series, int | None, pd.Series | None]] = [
        ("target_best_single", best_selected, None, None),
        ("target_kNN", knn_selected, None, None),
    ]
    selected_rows.extend((f"target_direct_{method}", selected, None, None) for method, selected in direct_selected.items())
    selected_rows.extend(
        [
            ("target_d2_native", native_selected, native_codebook.effective_labels, native_test_labels),
            ("source_d2_label_transfer", transfer_selected, source_codebook.effective_labels, source_test_labels),
        ]
    )

    target_test_stats = _pool_stats(target_test.utility)
    rows = []
    for offset, (method, selected, row_k, labels) in enumerate(selected_rows):
        row = evaluate_selection(
            method=method,
            selected_models=selected,
            matrices=target_test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + offset,
            k=row_k,
            labels=labels,
        )
        row.update(
            {
                "transfer_scenario": scenario.name,
                "source_family": scenario.source_family,
                "source_model_count": len(scenario.source_models),
                "target_model_count": len(scenario.target_models),
                "source_models": ";".join(scenario.source_models),
                "target_models": ";".join(scenario.target_models),
                "source_target_overlap": len(set(scenario.source_models).intersection(scenario.target_models)),
                "d2_alpha": alpha if "d2" in method else "",
                "d2_beta": beta if "d2" in method else "",
                "train_source_oracle_gap": scenario.stats["source_oracle_gap"],
                "train_target_oracle_gap": scenario.stats["target_oracle_gap"],
                "target_oracle_gap": target_test_stats["oracle_gap"],
                "target_dominance_ratio": target_test_stats["dominance_ratio"],
                "target_winner_entropy": target_test_stats["winner_entropy"],
                "target_train_query_count": len(target_train.utility),
                "same_budget_as_direct_retraining": method in {"source_d2_label_transfer", "target_d2_native"}
                or method.startswith("target_direct_"),
            }
        )
        rows.append(row)
    return rows


def _subset_matrices(matrices: Matrices, models: list[str]) -> Matrices:
    return Matrices(
        quality=matrices.quality.loc[:, models],
        cost=matrices.cost.loc[:, models],
        utility=matrices.utility.loc[:, models],
        query_info=matrices.query_info,
        model_ids=models,
    )


def _pool_stats(utility: pd.DataFrame) -> dict[str, float]:
    best_single = float(utility.mean(axis=0).max())
    oracle = float(utility.max(axis=1).mean())
    winners = utility.idxmax(axis=1).astype(str)
    shares = winners.value_counts(normalize=True)
    entropy = float(-(shares * np.log2(shares)).sum()) if not shares.empty else 0.0
    return {
        "oracle_gap": oracle - best_single,
        "dominance_ratio": float(shares.max()) if not shares.empty else 0.0,
        "winner_entropy": entropy,
    }


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    summary = _summary_table(table)
    transfer_rows = table[table["method"].eq("source_d2_label_transfer")]
    direct_rows = table[table["method"].astype(str).str.startswith("target_direct_")]
    lines = [
        "# Phase F/G Held-Out Model-Pool Transfer Memo",
        "",
        f"Command: `python experiments/19_model_pool_transfer.py --config {config_path}`",
        "",
        "This run evaluates disjoint source and target model pools. Route labels are learned on the source pool, then remapped to target-pool models using target train utility only.",
        "",
        _markdown_table(summary),
        "",
        "## Transfer Readout",
        "",
        f"- Transfer scenarios: `{table['transfer_scenario'].nunique()}`.",
        f"- Source/target size pairs: {_size_pair_summary(table)}.",
        f"- Source/target overlap max: `{int(table['source_target_overlap'].max())}`.",
    ]
    if transfer_rows.empty:
        lines.append("- No transferred D2 rows were produced.")
    else:
        lines.append(
            "- Transferred D2 recovered-gap range: "
            f"`{transfer_rows['recovered_gap_vs_oracle'].min():.4f}` to "
            f"`{transfer_rows['recovered_gap_vs_oracle'].max():.4f}` "
            f"(mean `{transfer_rows['recovered_gap_vs_oracle'].mean():.4f}`)."
        )
    if not direct_rows.empty:
        lines.append(
            "- Direct retraining recovered-gap range: "
            f"`{direct_rows['recovered_gap_vs_oracle'].min():.4f}` to "
            f"`{direct_rows['recovered_gap_vs_oracle'].max():.4f}` "
            f"(mean `{direct_rows['recovered_gap_vs_oracle'].mean():.4f}`)."
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a held-out model-pool diagnostic. It does not prove transfer unless transferred labels are competitive with same-budget direct retraining across broader datasets and pools.",
            "",
        ]
    )
    (out_dir / "phase_f_g_model_pool_transfer_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    marker = "## Held-Out Model-Pool Transfer"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/19_model_pool_transfer.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_model_pool_transfer.csv`: disjoint source/target pool transfer rows for target baselines, native D2, and transferred source-D2 labels.",
        "- `phase_f_g_model_pool_transfer_memo.md`: held-out model-pool transfer checkpoint memo.",
        "",
        f"- Transfer scenarios: `{table['transfer_scenario'].nunique()}`.",
        f"- Source/target size pairs: {_size_pair_summary(table)}.",
        f"- Source/target overlap max: `{int(table['source_target_overlap'].max())}`.",
        "",
        _markdown_table(_summary_table(table)),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Model-Pool Transfer Run\n"
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transfer_scenario",
        "source_model_count",
        "target_model_count",
        "method",
        "mean_utility",
        "recovered_gap_vs_oracle",
    ]
    return table.loc[:, columns].sort_values(["transfer_scenario", "method"]).reset_index(drop=True)


def _size_pair_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "`none`"
    pairs = sorted(
        {
            (int(row.source_model_count), int(row.target_model_count))
            for row in table.loc[:, ["source_model_count", "target_model_count"]].itertuples(index=False)
        }
    )
    return ", ".join(f"`{source}x{target}`" for source, target in pairs)


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
