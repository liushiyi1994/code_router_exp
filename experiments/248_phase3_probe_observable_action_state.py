from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from routecode.states.action_states import (
    FRONTIER_NEEDED,
    fit_action_state_policy,
    local_frontier_action_labels,
    selected_utility,
)
from routecode.states.probe_features_v2 import build_local_behavior_probe_features, numeric_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a probe-observable local/frontier RouteCode action state."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_probe_observable_action_state"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--text-classifiers", nargs="*", default=["knn", "logreg", "rf", "extratrees", "histgb"])
    parser.add_argument("--probe-classifiers", nargs="*", default=["logreg", "rf", "extratrees", "histgb"])
    parser.add_argument("--max-probe-rates", type=float, nargs="*", default=[0.10, 0.20, 0.30, 0.50, 0.75, 1.00])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    query_table = query_metadata(outputs)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean").dropna(axis=0)
    query_table = query_table[query_table["query_id"].isin(set(utility.index.astype(str)))].copy()
    local_models, frontier_models = model_families(outputs)
    labels = local_frontier_action_labels(utility, local_models=local_models, frontier_models=frontier_models)
    train_ids = ids_for_split(query_table, "train")
    policy = fit_action_state_policy(utility.reindex(train_ids).dropna(axis=0), labels.reindex(train_ids).dropna())

    embeddings = load_or_encode_embeddings(query_table, args.output_dir, args.embedding_model)
    probe_features = build_local_behavior_probe_features(outputs, local_models=local_models)
    probe_features = probe_features[probe_features["query_id"].isin(set(query_table["query_id"].astype(str)))].copy()
    views = design_matrices(probe_features, embeddings, train_ids=train_ids, seed=int(args.seed))

    rows: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []

    rows.extend(
        evaluate_baseline_rows(
            "best_single",
            labels,
            best_single_prediction(utility, train_ids, frontier_models),
            utility,
            policy,
            query_table,
            local_models,
            frontier_models,
        )
    )
    rows.extend(evaluate_prediction("true_action_state_oracle", labels, prediction_from_labels(labels), utility, policy, query_table, local_models, frontier_models))

    text_predictions: dict[str, pd.DataFrame] = {}
    for text_name in args.text_classifiers:
        pred = fit_predict_classifier(
            text_name,
            views["semantic"].reindex(train_ids),
            labels.reindex(train_ids),
            views["semantic"],
            seed=int(args.seed),
        )
        text_predictions[text_name] = pred
        rows.extend(
            evaluate_prediction(
                f"text_only::{text_name}",
                labels,
                pred,
                utility,
                policy,
                query_table,
                local_models,
                frontier_models,
            )
        )

    probe_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for view_name in ["probe_only", "semantic_pca8_plus_probe", "semantic_plus_probe"]:
        for probe_name in args.probe_classifiers:
            pred = fit_predict_classifier(
                probe_name,
                views[view_name].reindex(train_ids),
                labels.reindex(train_ids),
                views[view_name],
                seed=int(args.seed),
            )
            probe_predictions[(view_name, probe_name)] = pred
            rows.extend(
                evaluate_prediction(
                    f"always_probe::{view_name}::{probe_name}",
                    labels,
                    with_probe_rate(pred, 1.0),
                    utility,
                    policy,
                    query_table,
                    local_models,
                    frontier_models,
                )
            )

    for text_name, text_pred in text_predictions.items():
        for (view_name, probe_name), probe_pred in probe_predictions.items():
            for max_probe_rate in args.max_probe_rates:
                threshold = select_threshold_on_val(
                    text_pred,
                    probe_pred,
                    labels,
                    query_table,
                    max_probe_rate=float(max_probe_rate),
                )
                fused = fuse_predictions(text_pred, probe_pred, threshold)
                method = f"active_probe::{text_name}::{view_name}::{probe_name}::rate_{max_probe_rate:g}"
                rows.extend(
                    evaluate_prediction(
                        method,
                        labels,
                        fused,
                        utility,
                        policy,
                        query_table,
                        local_models,
                        frontier_models,
                        confidence_threshold=threshold,
                    )
                )
                if text_name in {"knn", "logreg"} and probe_name in {"histgb", "extratrees"}:
                    assignments.append(assignment_frame(method, labels, text_pred, probe_pred, fused, query_table))

    table = pd.DataFrame(rows).sort_values(["split", "state_accuracy", "mean_utility"], ascending=[True, False, False])
    table.to_csv(args.output_dir / "table_probe_observable_action_state.csv", index=False)
    probe_features.to_csv(args.output_dir / "table_probe_observable_action_features.csv", index=False)
    if assignments:
        pd.concat(assignments, ignore_index=True).to_csv(args.output_dir / "table_probe_observable_action_assignments.csv", index=False)
    write_readme(args, table, labels, policy)
    print(f"Wrote probe-observable action-state results to {args.output_dir}")
    print(table[table["split"].eq("test")].head(30).to_string(index=False))


def load_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce")
    outputs["normalized_remote_cost"] = pd.to_numeric(outputs["normalized_remote_cost"], errors="coerce").fillna(0.0)
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    return outputs.dropna(subset=["utility"])


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["query_id", "query_text", "split", "benchmark", "domain"] if col in outputs.columns]
    return outputs[cols].drop_duplicates("query_id").copy()


def model_families(outputs: pd.DataFrame) -> tuple[list[str], list[str]]:
    meta = outputs[["model_id", "is_local", "is_frontier"]].drop_duplicates("model_id")
    local = meta[meta["is_local"].astype(bool)]["model_id"].astype(str).tolist()
    frontier = meta[meta["is_frontier"].astype(bool)]["model_id"].astype(str).tolist()
    return local, frontier


def ids_for_split(query_table: pd.DataFrame, split: str) -> pd.Index:
    return pd.Index(query_table[query_table["split"].astype(str).eq(split)]["query_id"].astype(str), name="query_id")


def load_or_encode_embeddings(query_table: pd.DataFrame, output_dir: Path, model_name: str) -> pd.DataFrame:
    safe = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
    cache = output_dir / f"query_embeddings_{safe}_{len(query_table)}.npy"
    ids_path = output_dir / f"query_embedding_ids_{safe}_{len(query_table)}.csv"
    if cache.exists() and ids_path.exists():
        return pd.DataFrame(np.load(cache), index=pd.read_csv(ids_path)["query_id"].astype(str).tolist())
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name, local_files_only=True)
        arr = np.asarray(
            encoder.encode(
                query_table["query_text"].fillna("").astype(str).tolist(),
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        print(f"Embedding model unavailable ({exc}); falling back to TF-IDF/SVD.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        x = TfidfVectorizer(max_features=8192, ngram_range=(1, 2), stop_words="english").fit_transform(
            query_table["query_text"].fillna("").astype(str)
        )
        n_components = min(384, max(2, min(x.shape) - 1))
        arr = normalize(TruncatedSVD(n_components=n_components, random_state=17).fit_transform(x)).astype(np.float32)
    np.save(cache, arr)
    query_table[["query_id"]].to_csv(ids_path, index=False)
    return pd.DataFrame(arr, index=query_table["query_id"].astype(str).tolist())


def design_matrices(
    probe_features: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    train_ids: pd.Index,
    seed: int,
) -> dict[str, pd.DataFrame]:
    probe = probe_features.set_index("query_id")
    cols = numeric_feature_columns(probe_features)
    probe_x = probe[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    emb = embeddings.reindex(probe_x.index).fillna(0.0)
    emb.columns = [f"emb_{idx}" for idx in range(emb.shape[1])]
    train_emb = emb.reindex(train_ids).dropna(axis=0, how="any")
    n_components = min(8, emb.shape[1], max(1, len(train_emb) - 1))
    scaler = StandardScaler().fit(train_emb.to_numpy(dtype=float))
    pca = PCA(n_components=n_components, random_state=int(seed)).fit(scaler.transform(train_emb.to_numpy(dtype=float)))
    emb_pca = pd.DataFrame(
        pca.transform(scaler.transform(emb.to_numpy(dtype=float))),
        index=emb.index,
        columns=[f"emb_pca8_{idx}" for idx in range(n_components)],
    )
    return {
        "semantic": emb,
        "probe_only": probe_x,
        "semantic_pca8_plus_probe": pd.concat([emb_pca, probe_x], axis=1),
        "semantic_plus_probe": pd.concat([emb, probe_x], axis=1),
    }


def fit_predict_classifier(
    name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    all_x: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    train_y = train_y.reindex(train_x.index).dropna().astype(str)
    train_x = train_x.reindex(train_y.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_x = all_x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    encoder = LabelEncoder().fit(train_y.to_numpy(dtype=str))
    y = encoder.transform(train_y.to_numpy(dtype=str))
    clf = make_classifier(name, seed=seed)
    clf.fit(train_x.to_numpy(dtype=float), y)
    probs = clf.predict_proba(all_x.to_numpy(dtype=float))
    classes = encoder.inverse_transform(np.asarray(clf.classes_, dtype=int)).astype(str)
    return prediction_frame(all_x.index, classes, probs)


def make_classifier(name: str, *, seed: int):
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000, random_state=seed))
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=0.02, random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


def prediction_frame(index: pd.Index, classes: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(probs, index=index.astype(str), columns=[str(label) for label in classes])
    state_cols = [str(label) for label in classes]
    out["predicted_state"] = out[state_cols].idxmax(axis=1)
    out["confidence"] = out[state_cols].max(axis=1)
    out["used_probe"] = 0.0
    return out


def prediction_from_labels(labels: pd.Series) -> pd.DataFrame:
    classes = sorted(labels.astype(str).unique())
    probs = pd.DataFrame(0.0, index=labels.index.astype(str), columns=classes)
    for query_id, label in labels.astype(str).items():
        probs.loc[str(query_id), label] = 1.0
    probs["predicted_state"] = labels.astype(str)
    probs["confidence"] = 1.0
    probs["used_probe"] = 0.0
    return probs


def best_single_prediction(utility: pd.DataFrame, train_ids: pd.Index, frontier_models: list[str]) -> pd.DataFrame:
    model = str(utility.reindex(train_ids).mean(axis=0).sort_values(ascending=False).index[0])
    state = FRONTIER_NEEDED if model in set(frontier_models) else "local_enough"
    labels = pd.Series(state, index=utility.index.astype(str), name="predicted_state")
    out = prediction_from_labels(labels)
    out["selected_model"] = model
    return out


def with_probe_rate(pred: pd.DataFrame, rate: float) -> pd.DataFrame:
    out = pred.copy()
    out["used_probe"] = float(rate)
    return out


def select_threshold_on_val(
    text_pred: pd.DataFrame,
    probe_pred: pd.DataFrame,
    true_labels: pd.Series,
    query_table: pd.DataFrame,
    *,
    max_probe_rate: float,
) -> float:
    val_ids = ids_for_split(query_table, "val")
    aligned = pd.concat(
        [
            text_pred.reindex(val_ids)["confidence"].rename("confidence"),
            text_pred.reindex(val_ids)["predicted_state"].rename("text_predicted"),
            probe_pred.reindex(val_ids)["predicted_state"].rename("probe_predicted"),
            true_labels.reindex(val_ids).rename("true"),
        ],
        axis=1,
    ).dropna()
    best_threshold = 0.0
    best_accuracy = -1.0
    best_rate = 1.0
    for threshold in np.append(np.linspace(0.0, 1.0, 101), 1.01):
        probe = aligned["confidence"].astype(float).lt(float(threshold))
        probe_rate = float(probe.mean())
        if probe_rate > float(max_probe_rate) + 1e-12:
            continue
        final = aligned["text_predicted"].astype(str).copy()
        final.loc[probe] = aligned.loc[probe, "probe_predicted"].astype(str)
        accuracy = float(final.eq(aligned["true"].astype(str)).mean())
        if accuracy > best_accuracy or (np.isclose(accuracy, best_accuracy) and probe_rate < best_rate):
            best_accuracy = accuracy
            best_rate = probe_rate
            best_threshold = float(threshold)
    return best_threshold


def fuse_predictions(text_pred: pd.DataFrame, probe_pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = text_pred.copy()
    needs_probe = out["confidence"].astype(float).lt(float(threshold))
    out.loc[needs_probe, "predicted_state"] = probe_pred.reindex(out.index).loc[needs_probe, "predicted_state"].astype(str)
    out.loc[needs_probe, "confidence"] = probe_pred.reindex(out.index).loc[needs_probe, "confidence"].astype(float)
    out["used_probe"] = needs_probe.astype(float)
    return out


def evaluate_baseline_rows(
    method: str,
    labels: pd.Series,
    pred: pd.DataFrame,
    utility: pd.DataFrame,
    policy,
    query_table: pd.DataFrame,
    local_models: list[str],
    frontier_models: list[str],
) -> list[dict[str, Any]]:
    del policy
    selected_column = "selected_model" if "selected_model" in pred.columns else "predicted_state"
    selected = pd.Series(pred[selected_column].astype(str).iloc[0], index=utility.index.astype(str), name="selected_model")
    return metric_rows(method, labels, pred, utility, query_table, local_models, frontier_models, selected)


def evaluate_prediction(
    method: str,
    labels: pd.Series,
    pred: pd.DataFrame,
    utility: pd.DataFrame,
    policy,
    query_table: pd.DataFrame,
    local_models: list[str],
    frontier_models: list[str],
    *,
    confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    selected = policy.select(pred["predicted_state"].astype(str))
    rows = metric_rows(method, labels, pred, utility, query_table, local_models, frontier_models, selected)
    for row in rows:
        row["confidence_threshold"] = float(confidence_threshold) if confidence_threshold is not None else np.nan
    return rows


def metric_rows(
    method: str,
    labels: pd.Series,
    pred: pd.DataFrame,
    utility: pd.DataFrame,
    query_table: pd.DataFrame,
    local_models: list[str],
    frontier_models: list[str],
    selected: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    oracle_utility = utility.max(axis=1)
    selected_u = selected_utility(utility, selected)
    for split in ["train", "val", "test"]:
        ids = ids_for_split(query_table, split)
        aligned = pd.concat(
            [
                labels.reindex(ids).rename("true"),
                pred.reindex(ids)["predicted_state"].rename("predicted"),
                pred.reindex(ids)["confidence"].rename("confidence"),
                pred.reindex(ids)["used_probe"].rename("used_probe"),
                selected.reindex(ids).rename("selected_model"),
                selected_u.reindex(ids).rename("selected_utility"),
                oracle_utility.reindex(ids).rename("oracle_utility"),
            ],
            axis=1,
        ).dropna()
        precision, recall, f1, _ = precision_recall_fscore_support(
            aligned["true"].astype(str),
            aligned["predicted"].astype(str),
            labels=[FRONTIER_NEEDED],
            average="binary",
            pos_label=FRONTIER_NEEDED,
            zero_division=0,
        )
        rows.append(
            {
                "method": method,
                "split": split,
                "n_queries": int(len(aligned)),
                "state_accuracy": float(accuracy_score(aligned["true"], aligned["predicted"])) if len(aligned) else np.nan,
                "balanced_state_accuracy": float(balanced_accuracy_score(aligned["true"], aligned["predicted"])) if len(aligned) else np.nan,
                "frontier_precision": float(precision),
                "frontier_recall": float(recall),
                "frontier_f1": float(f1),
                "mean_utility": float(aligned["selected_utility"].mean()) if len(aligned) else np.nan,
                "oracle_utility": float(aligned["oracle_utility"].mean()) if len(aligned) else np.nan,
                "oracle_utility_ratio": float(aligned["selected_utility"].mean() / max(aligned["oracle_utility"].mean(), 1e-12)) if len(aligned) else np.nan,
                "oracle_regret": float((aligned["oracle_utility"] - aligned["selected_utility"]).mean()) if len(aligned) else np.nan,
                "frontier_call_rate": float(aligned["selected_model"].astype(str).isin(frontier_models).mean()) if len(aligned) else np.nan,
                "local_call_rate": float(aligned["selected_model"].astype(str).isin(local_models).mean()) if len(aligned) else np.nan,
                "actual_probe_rate": float(aligned["used_probe"].astype(float).mean()) if len(aligned) else np.nan,
                "mean_confidence": float(aligned["confidence"].astype(float).mean()) if len(aligned) else np.nan,
                "confidence_threshold": np.nan,
            }
        )
    return rows


def assignment_frame(
    method: str,
    labels: pd.Series,
    text_pred: pd.DataFrame,
    probe_pred: pd.DataFrame,
    fused: pd.DataFrame,
    query_table: pd.DataFrame,
) -> pd.DataFrame:
    ids = ids_for_split(query_table, "test")
    return pd.DataFrame(
        {
            "query_id": ids.astype(str),
            "method": method,
            "true_state": labels.reindex(ids).astype(str).to_numpy(),
            "text_state": text_pred.reindex(ids)["predicted_state"].astype(str).to_numpy(),
            "text_confidence": text_pred.reindex(ids)["confidence"].astype(float).to_numpy(),
            "probe_state": probe_pred.reindex(ids)["predicted_state"].astype(str).to_numpy(),
            "probe_confidence": probe_pred.reindex(ids)["confidence"].astype(float).to_numpy(),
            "final_state": fused.reindex(ids)["predicted_state"].astype(str).to_numpy(),
            "used_probe": fused.reindex(ids)["used_probe"].astype(float).to_numpy(),
        }
    )


def write_readme(args: argparse.Namespace, table: pd.DataFrame, labels: pd.Series, policy) -> None:
    test = table[table["split"].eq("test")].sort_values(["state_accuracy", "mean_utility"], ascending=[False, False])
    best_rows = "\n".join(
        "| {method} | {acc:.4f} | {bacc:.4f} | {utility:.4f} | {ratio:.4f} | {frontier:.4f} | {probe:.4f} |".format(
            method=row.method,
            acc=float(row.state_accuracy),
            bacc=float(row.balanced_state_accuracy),
            utility=float(row.mean_utility),
            ratio=float(row.oracle_utility_ratio),
            frontier=float(row.frontier_call_rate),
            probe=float(row.actual_probe_rate),
        )
        for row in test.head(20).itertuples(index=False)
    )
    target_hit = bool((test["state_accuracy"] >= 0.90).any())
    body = f"""# Probe-Observable Action State

This experiment evaluates a 1-bit RouteCode state learned from the cost-aware
utility matrix:

```text
local_enough vs frontier_needed
```

The state learner uses the utility matrix to assign train labels and build a
train-only state-to-model table. The deployable predictor uses query semantics
and cached local-model behavior only. No gold answer, quality score, utility
value, benchmark label, or frontier output is used as a predictor feature.

Command:

```bash
PYTHONPATH=src python experiments/248_phase3_probe_observable_action_state.py \\
  --outputs {args.outputs} \\
  --output-dir {args.output_dir} \\
  --embedding-model {args.embedding_model}
```

State counts:

```text
{labels.value_counts().to_string()}
```

Train state-to-model table:

```text
{policy.label_to_model}
```

Target hit: `{target_hit}`

## Best Test Rows

| method | state accuracy | balanced accuracy | mean utility | oracle utility ratio | frontier call rate | probe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{best_rows}

Artifacts:

- `table_probe_observable_action_state.csv`
- `table_probe_observable_action_features.csv`
- `table_probe_observable_action_assignments.csv`
"""
    (args.output_dir / "README.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
