from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_compression_ladder
from routecode.predictors.classifiers import (
    LogisticModelRouter,
    MLPModelRouter,
    MLPRouteCodeLabelClassifier,
    PredictedLabelLookupRouter,
    RouteCodeLabelClassifier,
    SVMModelRouter,
)
from routecode.routers.cluster_lookup import EmbeddingClusterRouter
from routecode.routers.dataset_lookup import DatasetLabelRouter, DatasetOracleRouter
from routecode.routers.knn import KNNRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.random import RandomRouter
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
    router_config = config.get("routers", {})
    route_config = config.get("routecode", {})
    cluster_k = int(router_config.get("embedding_clusters", 16))
    knn_k = int(router_config.get("knn_k", 15))
    route_k = int(route_config.get("selected_k_for_cards", 16))

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = selected_values(test.utility, best_single).mean()
    random_selected = RandomRouter(random_state=seed).fit(train.query_info, train.utility).predict(test.query_info)
    oracle_selected = OracleRouter().predict(test.utility)
    oracle_mean = test.utility.max(axis=1).mean()

    dataset_router = DatasetLabelRouter("dataset").fit(train.query_info, train.utility)
    dataset_selected = dataset_router.predict(test.query_info)
    dataset_oracle_selected = DatasetOracleRouter("dataset").fit(test.query_info, test.utility).predict(test.query_info)

    if "predicted_topic" in train.query_info.columns and "predicted_topic" in test.query_info.columns:
        topic_router = DatasetLabelRouter("predicted_topic").fit(train.query_info, train.utility)
        topic_selected = topic_router.predict(test.query_info)
        topic_labels = test.query_info["predicted_topic"]
        topic_k = test.query_info["predicted_topic"].nunique()
    else:
        topic_router = PredictedLabelLookupRouter("dataset", random_state=seed).fit(
            train.query_info,
            train.utility,
            embeddings,
        )
        topic_selected = topic_router.predict(test.query_info, embeddings)
        topic_labels = topic_router.predict_labels(embeddings.loc[test.utility.index])
        topic_k = int(topic_labels.nunique())

    cluster_router = EmbeddingClusterRouter(cluster_k, random_state=seed).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    cluster_labels = cluster_router.predict_labels(embeddings.loc[test.utility.index])
    cluster_selected = cluster_router.predict(test.query_info, embeddings)

    knn_router = KNNRouter(knn_k).fit(train.query_info, train.utility, embeddings)
    knn_selected = knn_router.predict(test.query_info, embeddings)

    logistic_router = LogisticModelRouter(random_state=seed).fit(train.query_info, train.utility, embeddings)
    logistic_selected = logistic_router.predict(test.query_info, embeddings)
    mlp_router = MLPModelRouter(random_state=seed, hidden_layer_sizes=(32,), max_iter=1000).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    mlp_selected = mlp_router.predict(test.query_info, embeddings)
    svm_router = SVMModelRouter(random_state=seed).fit(train.query_info, train.utility, embeddings)
    svm_selected = svm_router.predict(test.query_info, embeddings)

    routecode = RouteCodeCodebook(route_k, random_state=seed, max_iter=int(route_config.get("max_iter", 25))).fit(
        train.query_info,
        train.utility,
        embeddings,
    )
    route_labels = routecode.predict_utility_labels(test.utility)
    route_selected = routecode.predict_from_labels(route_labels)
    routecode_classifier = RouteCodeLabelClassifier(random_state=seed).fit(routecode, embeddings)
    predicted_route_labels = routecode_classifier.predict_labels(embeddings.loc[test.utility.index])
    predicted_route_selected = routecode_classifier.predict(test.query_info, embeddings)
    mlp_routecode_classifier = MLPRouteCodeLabelClassifier(
        random_state=seed,
        hidden_layer_sizes=(8,),
        max_iter=2000,
    ).fit(routecode, embeddings)
    mlp_predicted_route_labels = mlp_routecode_classifier.predict_labels(embeddings.loc[test.utility.index])
    mlp_predicted_route_selected = mlp_routecode_classifier.predict(test.query_info, embeddings)

    learned_reference_mean = max(
        selected_values(test.utility, knn_selected).mean(),
        selected_values(test.utility, logistic_selected).mean(),
        selected_values(test.utility, mlp_selected).mean(),
        selected_values(test.utility, svm_selected).mean(),
        selected_values(test.utility, predicted_route_selected).mean(),
        selected_values(test.utility, mlp_predicted_route_selected).mean(),
    )

    rows = []
    methods = [
        ("random", random_selected, random_selected, len(train.model_ids)),
        ("best_single", best_single, None, None),
        ("dataset_label_lookup", dataset_selected, test.query_info["dataset"], test.query_info["dataset"].nunique()),
        ("dataset_oracle", dataset_oracle_selected, test.query_info["dataset"], test.query_info["dataset"].nunique()),
        (
            "predicted_topic_lookup",
            topic_selected,
            topic_labels,
            topic_k,
        ),
        ("embedding_cluster_lookup", cluster_selected, cluster_labels, cluster_k),
        ("kNN", knn_selected, None, None),
        ("logistic_embedding_router", logistic_selected, None, None),
        ("mlp_embedding_router", mlp_selected, None, None),
        ("svm_embedding_router", svm_selected, None, None),
        ("routecode_oracle_labels", route_selected, route_labels, route_k),
        ("routecode_predicted_labels", predicted_route_selected, predicted_route_labels, route_k),
        ("routecode_mlp_predicted_labels", mlp_predicted_route_selected, mlp_predicted_route_labels, route_k),
        ("query_oracle", oracle_selected, oracle_selected, test.model_ids.__len__()),
    ]
    for method, selected, labels, k in methods:
        rows.append(
            evaluate_selection(
                method=method,
                selected_models=selected,
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
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_recovered_gap.csv", index=False)
    save_compression_ladder(table, out_dir / "fig_compression_ladder.pdf")

    leaky_dataset_selected = DatasetLabelRouter("dataset").fit(test.query_info, test.utility).predict(test.query_info)
    train_only_utility = selected_values(test.utility, dataset_selected).mean()
    leaky_utility = selected_values(test.utility, leaky_dataset_selected).mean()
    pd.DataFrame(
        [
            {
                "diagnostic": "dataset_label_lookup_test_fit_minus_train_fit",
                "train_only_mean_utility": float(train_only_utility),
                "leaky_test_fit_mean_utility": float(leaky_utility),
                "leakage_gap": float(leaky_utility - train_only_utility),
            }
        ]
    ).to_csv(out_dir / "table_leakage_gap.csv", index=False)
    print(f"Wrote compression ladder outputs to {out_dir}")


if __name__ == "__main__":
    main()
