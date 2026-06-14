from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.eval.predictor_diagnostics import (
    calibration_curve_table,
    expected_calibration_error,
    label_accuracy,
    utility_weighted_confusion,
)
from routecode.metrics import recovered_gap, selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_calibration_curve, save_utility_weighted_confusion
from routecode.predictors.classifiers import MLPRouteCodeLabelClassifier, RouteCodeLabelClassifier
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
    route_config = config.get("routecode", {})
    router_config = config.get("routers", {})
    route_k = int(route_config.get("selected_k_for_cards", 16))

    codebook = RouteCodeCodebook(
        route_k,
        random_state=seed,
        max_iter=int(route_config.get("max_iter", 25)),
    ).fit(train.query_info, train.utility, embeddings)

    oracle_code_labels = codebook.predict_utility_labels(test.utility)
    oracle_code_selected = codebook.predict_from_labels(oracle_code_labels)
    oracle_code_mean = float(selected_values(test.utility, oracle_code_selected).mean())
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    query_oracle_selected = OracleRouter().predict(test.utility)
    query_oracle_mean = float(selected_values(test.utility, query_oracle_selected).mean())

    predictions = _predictors(codebook, train, test, embeddings, seed, router_config)
    summary_rows = [
        _summary_row(
            "utility_oracle_labels",
            oracle_code_labels,
            pd.Series(1.0, index=oracle_code_labels.index, name="confidence"),
            oracle_code_labels,
            codebook,
            test.utility,
            baseline_mean,
            oracle_code_mean,
            query_oracle_mean,
        )
    ]
    confusion_rows = []
    calibration_rows = []
    for name, labels, confidence in predictions:
        summary_rows.append(
            _summary_row(
                name,
                labels,
                confidence,
                oracle_code_labels,
                codebook,
                test.utility,
                baseline_mean,
                oracle_code_mean,
                query_oracle_mean,
            )
        )
        confusion = utility_weighted_confusion(oracle_code_labels, labels, test.utility, codebook.label_to_model)
        confusion["predictor"] = name
        confusion_rows.append(confusion)
        correct = (labels.astype(int) == oracle_code_labels.astype(int)).astype(int)
        curve = calibration_curve_table(confidence, correct, n_bins=10)
        curve["predictor"] = name
        calibration_rows.append(curve)

    summary = pd.DataFrame(summary_rows)
    confusion_table = pd.concat(confusion_rows, ignore_index=True) if confusion_rows else pd.DataFrame()
    calibration_table = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()

    summary.to_csv(out_dir / "table_predictor_comparison.csv", index=False)
    confusion_table.to_csv(out_dir / "table_utility_weighted_confusion.csv", index=False)
    calibration_table.to_csv(out_dir / "table_calibration_curve.csv", index=False)
    best_deployable = (
        summary[summary["predictor"] != "utility_oracle_labels"]
        .sort_values("mean_utility", ascending=False)["predictor"]
        .iloc[0]
    )
    save_utility_weighted_confusion(
        confusion_table,
        out_dir / "fig_utility_weighted_confusion.pdf",
        predictor=str(best_deployable),
    )
    save_calibration_curve(calibration_table, out_dir / "fig_calibration_curve.pdf")
    append_readme(out_dir, args.config, summary)
    print(f"Wrote predictor diagnostics outputs to {out_dir}")


def _predictors(
    codebook: RouteCodeCodebook,
    train,
    test,
    embeddings: pd.DataFrame,
    seed: int,
    router_config: dict,
) -> list[tuple[str, pd.Series, pd.Series]]:
    test_embeddings = embeddings.loc[test.utility.index]
    predictions: list[tuple[str, pd.Series, pd.Series]] = []

    centroid_labels = codebook.predict_labels(test_embeddings)
    centroid_confidence = _embedding_centroid_confidence(codebook, test_embeddings)
    predictions.append(("embedding_centroid_assignment", centroid_labels, centroid_confidence))

    logistic = RouteCodeLabelClassifier(random_state=seed).fit(codebook, embeddings)
    logistic_labels = logistic.predict_labels(test_embeddings)
    predictions.append(("logistic_label_predictor", logistic_labels, logistic.predict_confidence(test_embeddings)))

    mlp = MLPRouteCodeLabelClassifier(random_state=seed, hidden_layer_sizes=(8,), max_iter=2000).fit(codebook, embeddings)
    mlp_labels = mlp.predict_labels(test_embeddings)
    predictions.append(("mlp_label_predictor", mlp_labels, mlp.predict_confidence(test_embeddings)))

    knn_k = min(int(router_config.get("knn_k", 15)), len(train.utility))
    train_embeddings = embeddings.loc[train.utility.index]
    knn = KNeighborsClassifier(n_neighbors=knn_k)
    knn.fit(train_embeddings.to_numpy(dtype=float), codebook.train_labels_.loc[train.utility.index].to_numpy(dtype=int))
    knn_labels = pd.Series(knn.predict(test_embeddings.to_numpy(dtype=float)), index=test.utility.index, name="route_label")
    knn_confidence = pd.Series(knn.predict_proba(test_embeddings.to_numpy(dtype=float)).max(axis=1), index=test.utility.index)
    predictions.append(("knn_label_predictor", knn_labels.astype(int), knn_confidence))
    return predictions


def _embedding_centroid_confidence(codebook: RouteCodeCodebook, embeddings: pd.DataFrame) -> pd.Series:
    if codebook.embedding_centroids_ is None:
        raise RuntimeError("Codebook must be fit before confidence is available")
    values = embeddings.to_numpy(dtype=float)
    centroids = codebook.embedding_centroids_.to_numpy(dtype=float)
    distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    logits = -distances
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return pd.Series(probabilities.max(axis=1), index=embeddings.index, name="confidence")


def _summary_row(
    name: str,
    labels: pd.Series,
    confidence: pd.Series,
    oracle_code_labels: pd.Series,
    codebook: RouteCodeCodebook,
    utility: pd.DataFrame,
    baseline_mean: float,
    oracle_code_mean: float,
    query_oracle_mean: float,
) -> dict:
    selected = codebook.predict_from_labels(labels)
    selected_utility = selected_values(utility, selected)
    correct = (labels.astype(int) == oracle_code_labels.astype(int)).astype(int)
    mean_utility = float(selected_utility.mean())
    return {
        "predictor": name,
        "label_accuracy": label_accuracy(oracle_code_labels, labels),
        "mean_confidence": float(confidence.mean()),
        "ece": expected_calibration_error(confidence, correct, n_bins=10),
        "mean_utility": mean_utility,
        "oracle_code_regret": float(oracle_code_mean - mean_utility),
        "recovered_gap_vs_oracle_code": recovered_gap(mean_utility, baseline_mean, oracle_code_mean),
        "recovered_gap_vs_query_oracle": recovered_gap(mean_utility, baseline_mean, query_oracle_mean),
    }


def append_readme(out_dir: Path, config_path: str, summary: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## RouteCode Predictor Diagnostics"
    compact = summary[
        [
            "predictor",
            "label_accuracy",
            "ece",
            "mean_utility",
            "oracle_code_regret",
            "recovered_gap_vs_query_oracle",
        ]
    ].sort_values("mean_utility", ascending=False)
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/05_predictor_diagnostics.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_predictor_comparison.csv`: oracle-code label accuracy, calibration, and routing utility by predictor.",
        "- `table_utility_weighted_confusion.csv`: label-confusion cells weighted by utility regret.",
        "- `table_calibration_curve.csv`: confidence-bin calibration data.",
        "- `fig_utility_weighted_confusion.pdf`: regret-weighted confusion heatmap for the best deployable predictor.",
        "- `fig_calibration_curve.pdf`: route-label predictor calibration curves.",
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
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
