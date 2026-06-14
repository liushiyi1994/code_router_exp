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
from routecode.eval.model_pool_scale import ModelPoolScenario, build_model_pool_scale_scenarios, model_pool_stats
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
    scale_config = config.get("model_pool_scale", {})
    d2_config = config.get("predictability_constrained", {})
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    sizes = [int(value) for value in scale_config.get("sizes", [2, 4, 8, 16, len(train.model_ids)])]
    base_k = int(scale_config.get("k", d2_config.get("k", d2_config.get("selected_k_for_cards", 16))))
    alpha = float(scale_config.get("d2_alpha", d2_config.get("selected_alpha", 3.0)))
    beta = float(scale_config.get("d2_beta", d2_config.get("beta", 0.0)))

    scenarios = build_model_pool_scale_scenarios(train.utility, sizes)
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        subset_train = _subset_matrices(train, scenario.models)
        subset_test = _subset_matrices(test, scenario.models)
        rows.extend(
            _scenario_rows(
                scenario=scenario,
                train=subset_train,
                test=subset_test,
                embeddings=embeddings,
                seed=seed + scenario_index,
                k=min(base_k, len(scenario.models)),
                alpha=alpha,
                beta=beta,
                n_bootstrap=n_bootstrap,
                ci=ci,
            )
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_model_pool_scale.csv", index=False)
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote model-pool scale outputs to {out_dir}")


def _scenario_rows(
    *,
    scenario: ModelPoolScenario,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    seed: int,
    k: int,
    alpha: float,
    beta: float,
    n_bootstrap: int,
    ci: float,
) -> list[dict[str, Any]]:
    baseline_mean, learned_reference_mean, oracle_mean, knn_selected = _references(train, test, embeddings)
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    d2 = PredictabilityConstrainedRouteCode(k, alpha=alpha, beta=beta, random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    d2_labels = d2.predict_labels(embeddings.loc[test.utility.index])
    selected_rows = [
        ("best_single", best_single, None, None),
        ("kNN", knn_selected, None, None),
        ("d2_embedding_centroid", d2.predict_from_labels(d2_labels), k, d2_labels),
    ]
    train_stats = scenario.stats
    test_stats = model_pool_stats(test.utility, scenario.models)
    rows = []
    for offset, (method, selected, row_k, labels) in enumerate(selected_rows):
        row = evaluate_selection(
            method=method,
            selected_models=selected,
            matrices=test,
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
                "pool_family": scenario.family,
                "pool_name": scenario.name,
                "model_count": len(scenario.models),
                "models": ";".join(scenario.models),
                "d2_alpha": alpha if method == "d2_embedding_centroid" else "",
                "d2_beta": beta if method == "d2_embedding_centroid" else "",
                "train_best_single_model": train_stats["best_single_model"],
                "train_oracle_gap": train_stats["oracle_gap"],
                "train_dominance_ratio": train_stats["dominance_ratio"],
                "train_winner_entropy": train_stats["winner_entropy"],
                "test_best_single_model": test_stats["best_single_model"],
                "test_oracle_gap": test_stats["oracle_gap"],
                "test_dominance_ratio": test_stats["dominance_ratio"],
                "test_winner_entropy": test_stats["winner_entropy"],
            }
        )
        rows.append(row)
    return rows


def _references(train: Matrices, test: Matrices, embeddings: pd.DataFrame) -> tuple[float, float, float, pd.Series]:
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    oracle_mean = float(test.utility.max(axis=1).mean())
    knn_selected = KNNRouter(15).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, knn_selected).mean()))
    return baseline_mean, learned_reference_mean, oracle_mean, knn_selected


def _subset_matrices(matrices: Matrices, models: list[str]) -> Matrices:
    return Matrices(
        quality=matrices.quality.loc[:, models],
        cost=matrices.cost.loc[:, models],
        utility=matrices.utility.loc[:, models],
        query_info=matrices.query_info,
        model_ids=models,
    )


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    marker = "## Model-Pool Scale Robustness"
    summary = _summary_table(table)
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/18_model_pool_scale.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_model_pool_scale.csv`: top, complementary, dominated, and full model-pool rows for best-single, kNN, and D2.",
        "- `phase_f_g_model_pool_scale_memo.md`: model-pool scale/composition checkpoint memo.",
        "",
        _markdown_table(summary),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Model-Pool Scale Run\n"
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    summary = _summary_table(table)
    d2_rows = table[table["method"].eq("d2_embedding_centroid")]
    lines = [
        "# Phase F/G Model-Pool Scale Memo",
        "",
        f"Command: `python experiments/18_model_pool_scale.py --config {config_path}`",
        "",
        "This run evaluates top, complementary, and dominated model-pool scenarios using train-only pool construction. It extends the bounded model-pool sensitivity layer without changing the RouteCode method.",
        "",
        _markdown_table(summary),
        "",
        "## D2 Range",
        "",
        f"- D2 rows: `{len(d2_rows)}`.",
        f"- D2 recovered-gap range: `{d2_rows['recovered_gap_vs_oracle'].min():.4f}` to `{d2_rows['recovered_gap_vs_oracle'].max():.4f}`." if not d2_rows.empty else "- No D2 rows were produced.",
        f"- Model-count range: `{int(table['model_count'].min())}` to `{int(table['model_count'].max())}`.",
        "",
        "Interpretation: this is a robustness and diagnosis layer. It should not be used as a final model-pool transfer claim without additional held-out pool protocols and direct-router retraining comparisons.",
        "",
    ]
    (out_dir / "phase_f_g_model_pool_scale_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = ["pool_family", "pool_name", "model_count", "method", "mean_utility", "recovered_gap_vs_oracle"]
    return table.loc[:, columns].sort_values(["model_count", "pool_family", "pool_name", "method"]).reset_index(drop=True)


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
