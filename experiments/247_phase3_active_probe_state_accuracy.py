from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from routecode.eval.predictor_diagnostics import expected_calibration_error
from routecode.states.probe_features_v2 import (
    build_local_behavior_probe_features,
    numeric_feature_columns,
)
from routecode.states.utility_states_v2 import fit_utility_state_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text-semantic plus active-probe state prediction accuracy for RouteCode v2 states."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_routecode_v2_active_probe_accuracy"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--k-values", type=int, nargs="*", default=[2, 4, 6, 8, 16])
    parser.add_argument("--state-methods", nargs="*", default=["relative_kmeans", "calibration_refined"])
    parser.add_argument(
        "--text-classifiers",
        nargs="*",
        default=["knn", "logreg", "rf", "extratrees"],
    )
    parser.add_argument(
        "--probe-classifiers",
        nargs="*",
        default=["rf", "extratrees", "histgb", "logreg", "svc"],
    )
    parser.add_argument("--max-probe-rates", type=float, nargs="*", default=[0.10, 0.20, 0.30, 0.50, 0.75, 1.00])
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    query_table = query_metadata(outputs)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean").dropna(axis=0)
    query_table = query_table[query_table["query_id"].isin(set(utility.index.astype(str)))].copy()
    local_models, frontier_models = model_families(outputs)
    embeddings = load_or_encode_embeddings(query_table, args.output_dir, args.embedding_model)
    probe_features = build_local_behavior_probe_features(outputs, local_models=local_models)
    probe_features = probe_features[probe_features["query_id"].isin(set(query_table["query_id"].astype(str)))].copy()

    rows: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    for state_method in args.state_methods:
        for k in args.k_values:
            train_ids = ids_for_split(query_table, "train")
            model = fit_utility_state_model(
                utility.reindex(train_ids).dropna(axis=0),
                method=state_method,
                n_states=int(k),
                random_state=int(args.seed),
                local_models=tuple(local_models),
                frontier_models=tuple(frontier_models),
                calibration_eta=0.35,
                regret_gamma=0.35,
            )
            true_labels = assign_all_splits(model, utility, query_table)
            state_info = {
                "state_method": state_method,
                "k": int(k),
                "n_states": int(true_labels.nunique()),
            }
            text_predictions: dict[str, pd.DataFrame] = {}
            for text_name in args.text_classifiers:
                text_pred = fit_predict_classifier(
                    name=text_name,
                    train_x=embeddings.reindex(model.labels.index),
                    train_y=model.labels,
                    all_x=embeddings,
                    seed=int(args.seed),
                )
                text_predictions[text_name] = text_pred
                rows.extend(
                    evaluate_prediction(
                        true_labels,
                        text_pred,
                        query_table,
                        policy_kind="text_only",
                        text_classifier=text_name,
                        probe_classifier="none",
                        max_probe_rate=0.0,
                        **state_info,
                    )
                )

            probe_predictions: dict[tuple[str, str], pd.DataFrame] = {}
            for probe_view, probe_x in probe_design_matrices(probe_features, embeddings).items():
                for probe_name in args.probe_classifiers:
                    probe_pred = fit_predict_classifier(
                        name=probe_name,
                        train_x=probe_x.reindex(model.labels.index),
                        train_y=model.labels,
                        all_x=probe_x,
                        seed=int(args.seed),
                    )
                    probe_predictions[(probe_view, probe_name)] = probe_pred
                    rows.extend(
                        evaluate_prediction(
                            true_labels,
                            probe_pred,
                            query_table,
                            policy_kind=f"always_probe_{probe_view}",
                            text_classifier="none",
                            probe_classifier=probe_name,
                            max_probe_rate=1.0,
                            **state_info,
                        )
                    )

            for text_name, text_pred in text_predictions.items():
                for (probe_view, probe_name), probe_pred in probe_predictions.items():
                        for max_probe_rate in args.max_probe_rates:
                            threshold = select_threshold_on_val(
                                text_pred,
                                probe_pred,
                                true_labels,
                                query_table,
                                max_probe_rate=float(max_probe_rate),
                            )
                            fused = fuse_predictions(text_pred, probe_pred, threshold)
                            rows.extend(
                                evaluate_prediction(
                                    true_labels,
                                    fused,
                                    query_table,
                                    policy_kind=f"active_probe_{probe_view}",
                                    text_classifier=text_name,
                                    probe_classifier=probe_name,
                                    max_probe_rate=float(max_probe_rate),
                                    confidence_threshold=threshold,
                                    **state_info,
                                )
                            )
                            if state_method == args.state_methods[0] and int(k) in {2, 8, 16}:
                                assignments.append(
                                    assignment_frame(
                                        true_labels,
                                        text_pred,
                                        probe_pred,
                                        fused,
                                        query_table,
                                        state_method=state_method,
                                        k=int(k),
                                        text_classifier=text_name,
                                        probe_classifier=probe_name,
                                        probe_view=probe_view,
                                        max_probe_rate=float(max_probe_rate),
                                        threshold=threshold,
                                    )
                                )

    table = pd.DataFrame(rows).sort_values(["split", "state_accuracy"], ascending=[True, False])
    table.to_csv(args.output_dir / "table_active_probe_state_accuracy.csv", index=False)
    probe_features.to_csv(args.output_dir / "table_active_probe_features.csv", index=False)
    if assignments:
        pd.concat(assignments, ignore_index=True).to_csv(args.output_dir / "table_active_probe_assignments.csv", index=False)
    write_readme(args, table)
    print(f"Wrote active probe state accuracy results to {args.output_dir}")
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


def assign_all_splits(model, utility: pd.DataFrame, query_table: pd.DataFrame) -> pd.Series:
    labels = model.predict_from_utility(utility.reindex(query_table["query_id"].astype(str)).dropna(axis=0))
    labels.loc[model.labels.index] = model.labels
    return labels.rename("true_state")


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


def probe_design_matrices(probe_features: pd.DataFrame, embeddings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    probe = probe_features.set_index("query_id")
    cols = numeric_feature_columns(probe_features)
    probe_x = probe[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    emb = embeddings.reindex(probe_x.index).fillna(0.0)
    emb.columns = [f"emb_{idx}" for idx in range(emb.shape[1])]
    return {
        "probe_only": probe_x,
        "semantic_plus_probe": pd.concat([emb, probe_x], axis=1),
    }


def fit_predict_classifier(
    *,
    name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    all_x: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    train_y = train_y.reindex(train_x.index).dropna().astype(str)
    train_x = train_x.reindex(train_y.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    all_x = all_x.reindex(all_x.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    encoder = LabelEncoder().fit(train_y.to_numpy(dtype=str))
    y = encoder.transform(train_y.to_numpy(dtype=str))
    if len(encoder.classes_) == 1:
        probs = np.ones((len(all_x), 1), dtype=float)
        return prediction_frame(all_x.index, encoder.classes_, probs)
    clf = make_classifier(name, seed=seed, n_classes=len(encoder.classes_))
    clf.fit(train_x.to_numpy(dtype=float), y)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(all_x.to_numpy(dtype=float))
        classes = encoder.inverse_transform(np.asarray(clf.classes_, dtype=int))
    else:
        pred = clf.predict(all_x.to_numpy(dtype=float))
        classes = encoder.classes_
        probs = np.zeros((len(all_x), len(classes)), dtype=float)
        probs[np.arange(len(all_x)), pred] = 1.0
    return prediction_frame(all_x.index, classes.astype(str), probs)


def make_classifier(name: str, *, seed: int, n_classes: int):
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000, random_state=seed),
        )
    if name == "svc":
        return make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.05,
            l2_regularization=0.02,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


def prediction_frame(index: pd.Index, classes: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(probs, index=index.astype(str), columns=[str(label) for label in classes])
    out["predicted_state"] = out.idxmax(axis=1)
    out["confidence"] = out[[str(label) for label in classes]].max(axis=1)
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
            text_pred.reindex(val_ids)["predicted_state"].rename("predicted"),
            probe_pred.reindex(val_ids)["predicted_state"].rename("probe_predicted"),
            true_labels.reindex(val_ids).rename("true"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return 1.0
    best_threshold = 0.0
    best_accuracy = -1.0
    best_probe_rate = 1.0
    for threshold in np.append(np.linspace(0.0, 1.0, 101), 1.01):
        probe = aligned["confidence"].astype(float).lt(float(threshold))
        probe_rate = float(probe.mean())
        if probe_rate > float(max_probe_rate) + 1e-12:
            continue
        final = aligned["predicted"].astype(str).copy()
        final.loc[probe] = aligned.loc[probe, "probe_predicted"].astype(str)
        acc = float(final.eq(aligned["true"].astype(str)).mean())
        if acc > best_accuracy or (np.isclose(acc, best_accuracy) and probe_rate < best_probe_rate):
            best_accuracy = acc
            best_probe_rate = probe_rate
            best_threshold = float(threshold)
    return best_threshold


def fuse_predictions(text_pred: pd.DataFrame, probe_pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = text_pred.copy()
    needs_probe = out["confidence"].astype(float).lt(float(threshold))
    out.loc[needs_probe, "predicted_state"] = probe_pred.reindex(out.index).loc[needs_probe, "predicted_state"].astype(str)
    out.loc[needs_probe, "confidence"] = probe_pred.reindex(out.index).loc[needs_probe, "confidence"].astype(float)
    out["used_probe"] = needs_probe.astype(float)
    return out


def evaluate_prediction(
    true_labels: pd.Series,
    pred: pd.DataFrame,
    query_table: pd.DataFrame,
    *,
    state_method: str,
    k: int,
    n_states: int,
    policy_kind: str,
    text_classifier: str,
    probe_classifier: str,
    max_probe_rate: float,
    confidence_threshold: float | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "val", "test"]:
        split_ids = ids_for_split(query_table, split)
        if policy_kind.startswith("always_probe"):
            used_probe = pd.Series(1.0, index=split_ids, name="used_probe")
        else:
            used_probe = pred.reindex(split_ids).get("used_probe", pd.Series(0.0, index=split_ids)).rename("used_probe")
        aligned = pd.concat(
            [
                true_labels.reindex(split_ids).rename("true"),
                pred.reindex(split_ids)["predicted_state"].rename("predicted"),
                pred.reindex(split_ids)["confidence"].rename("confidence"),
                used_probe,
            ],
            axis=1,
        ).dropna()
        correct = aligned["true"].astype(str).eq(aligned["predicted"].astype(str))
        rows.append(
            {
                "state_method": state_method,
                "k": int(k),
                "n_states": int(n_states),
                "split": split,
                "policy_kind": policy_kind,
                "text_classifier": text_classifier,
                "probe_classifier": probe_classifier,
                "max_probe_rate": float(max_probe_rate),
                "confidence_threshold": float(confidence_threshold) if confidence_threshold is not None else np.nan,
                "n_queries": int(len(aligned)),
                "state_accuracy": float(correct.mean()) if len(aligned) else np.nan,
                "adjusted_rand": float(adjusted_rand_score(aligned["true"], aligned["predicted"])) if len(aligned) else np.nan,
                "mean_confidence": float(aligned["confidence"].astype(float).mean()) if len(aligned) else np.nan,
                "ece": expected_calibration_error(aligned["confidence"].astype(float), correct.astype(float), n_bins=10),
                "actual_probe_rate": float(aligned["used_probe"].astype(float).mean()) if len(aligned) else 0.0,
            }
        )
    return rows


def assignment_frame(
    true_labels: pd.Series,
    text_pred: pd.DataFrame,
    probe_pred: pd.DataFrame,
    fused: pd.DataFrame,
    query_table: pd.DataFrame,
    *,
    state_method: str,
    k: int,
    text_classifier: str,
    probe_classifier: str,
    probe_view: str,
    max_probe_rate: float,
    threshold: float,
) -> pd.DataFrame:
    ids = ids_for_split(query_table, "test")
    return pd.DataFrame(
        {
            "query_id": ids.astype(str),
            "state_method": state_method,
            "k": int(k),
            "text_classifier": text_classifier,
            "probe_classifier": probe_classifier,
            "probe_view": probe_view,
            "max_probe_rate": float(max_probe_rate),
            "confidence_threshold": float(threshold),
            "true_state": true_labels.reindex(ids).astype(str).to_numpy(),
            "text_state": text_pred.reindex(ids)["predicted_state"].astype(str).to_numpy(),
            "text_confidence": text_pred.reindex(ids)["confidence"].astype(float).to_numpy(),
            "probe_state": probe_pred.reindex(ids)["predicted_state"].astype(str).to_numpy(),
            "probe_confidence": probe_pred.reindex(ids)["confidence"].astype(float).to_numpy(),
            "final_state": fused.reindex(ids)["predicted_state"].astype(str).to_numpy(),
            "used_probe": fused.reindex(ids).get("used_probe", pd.Series(0.0, index=ids)).astype(float).to_numpy(),
        }
    )


def write_readme(args: argparse.Namespace, table: pd.DataFrame) -> None:
    test = table[table["split"].eq("test")].sort_values("state_accuracy", ascending=False)
    best = test.head(20)
    lines = "\n".join(
        "| {policy} | {k} | {text} | {probe} | {rate:.2f} | {acc:.4f} | {probe_rate:.4f} | {ari:.4f} |".format(
            policy=row.policy_kind,
            k=int(row.k),
            text=row.text_classifier,
            probe=row.probe_classifier,
            rate=float(row.max_probe_rate),
            acc=float(row.state_accuracy),
            probe_rate=float(row.actual_probe_rate),
            ari=float(row.adjusted_rand),
        )
        for row in best.itertuples(index=False)
    )
    target_hit = bool((test["state_accuracy"] >= 0.90).any())
    body = f"""# Active Probe State Accuracy

This experiment tests whether RouteCode v2 state prediction can exceed 90%
held-out state accuracy with:

```text
query text semantics -> low-confidence trigger -> local behavior active probe -> final state
```

No frontier outputs, gold answers, `quality_score`, or utility values are used
as predictor features. Probe features come from cached local-model behavior only.

Command:

```bash
PYTHONPATH=src python experiments/247_phase3_active_probe_state_accuracy.py \\
  --outputs {args.outputs} \\
  --output-dir {args.output_dir} \\
  --embedding-model {args.embedding_model}
```

Target hit: `{target_hit}`

## Best Test Rows

| policy | K | text classifier | probe classifier | max probe rate | state accuracy | actual probe rate | adjusted rand |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
{lines}

Artifacts:

- `table_active_probe_state_accuracy.csv`
- `table_active_probe_features.csv`
- `table_active_probe_assignments.csv`
"""
    (args.output_dir / "README.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
