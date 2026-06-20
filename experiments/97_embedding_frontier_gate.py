from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
FRONTIER_MODELS = ["gemini-3.5-flash", "gpt-5.5"]
GEMINI = "gemini-3.5-flash"
GPT = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding-based frontier-needed gate for expanded local pool.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/expanded_local_pool_qwen14/query_table_expanded_local_pool.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/embedding_frontier_gate")
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def add_targets(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    local_quality = table[[f"{model_id}_quality" for model_id in LOCAL_MODELS]].copy()
    local_quality.columns = LOCAL_MODELS
    frontier_quality = table[[f"{model_id}_quality" for model_id in FRONTIER_MODELS]].copy()
    frontier_quality.columns = FRONTIER_MODELS
    local_utility = local_quality.copy()
    frontier_utility = table[[f"{GEMINI}_utility_selected_cost", f"{GPT}_utility_selected_cost"]].copy()
    frontier_utility.columns = FRONTIER_MODELS

    table["local_oracle_model"] = local_quality.idxmax(axis=1)
    table["frontier_quality_oracle_model"] = frontier_quality.idxmax(axis=1)
    table["frontier_utility_oracle_model"] = frontier_utility.idxmax(axis=1)
    table["cost_oracle_model_expanded"] = pd.concat([local_utility, frontier_utility], axis=1).idxmax(axis=1)
    table["frontier_only_needed"] = (
        frontier_quality.max(axis=1).gt(local_quality.max(axis=1))
        & frontier_quality.max(axis=1).gt(0.5)
    ).astype(int)
    table["expanded_cost_oracle_utility"] = pd.concat([local_utility, frontier_utility], axis=1).max(axis=1)
    return table


def text_inputs(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    query_text = table["query_text"].fillna("").astype(str).tolist()
    local_answer_text = []
    for _, row in table.iterrows():
        parts = [str(row.get("query_text", ""))]
        for model_id in LOCAL_MODELS:
            answer_col = f"{model_id}_answer_norm" if f"{model_id}_answer_norm" in table.columns else f"{model_id}_answer"
            parts.append(f"{model_id}: {row.get(answer_col, '')}")
        local_answer_text.append("\n".join(parts))
    return query_text, local_answer_text


def embed_texts(model_name: str, texts: list[str], cache_path: Path, batch_size: int) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    values = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_path, values)
    return values


def numeric_features(table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        "local_max_vote",
        "local_ensemble_votes",
    ]
    columns += [f"{model_id}_answer_len" for model_id in LOCAL_MODELS if f"{model_id}_answer_len" in table]
    columns += [column for column in table.columns if column.startswith("agree__")]
    columns += [
        column
        for column in ["qwen8_4b_agree", "qwen8_06b_agree", "small_pair_agree", "all_three_agree"]
        if column in table
    ]
    columns = list(dict.fromkeys(columns))
    frame = pd.DataFrame(index=table.index)
    for column in columns:
        if table[column].dtype == bool:
            frame[column] = table[column].astype(float)
        else:
            frame[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    dataset_dummies = pd.get_dummies(table["dataset"].fillna("unknown"), prefix="dataset", dtype=float)
    frame = pd.concat([frame, dataset_dummies], axis=1)
    return frame.to_numpy(dtype=float), list(frame.columns)


def build_feature_matrices(table: pd.DataFrame, output_dir: Path, embedding_models: Iterable[str], batch_size: int) -> dict[str, np.ndarray]:
    query_text, local_answer_text = text_inputs(table)
    numeric, _ = numeric_features(table)
    matrices = {"numeric_only": numeric}
    safe_names = []
    for model_name in embedding_models:
        safe = model_name.replace("/", "__").replace("-", "_").replace(".", "_")
        safe_names.append(safe)
        query_emb = embed_texts(model_name, query_text, output_dir / f"emb_{safe}_query.npy", batch_size)
        local_emb = embed_texts(model_name, local_answer_text, output_dir / f"emb_{safe}_query_local.npy", batch_size)
        matrices[f"{safe}_query"] = np.hstack([query_emb, numeric])
        matrices[f"{safe}_query_local"] = np.hstack([local_emb, numeric])
        matrices[f"{safe}_combined"] = np.hstack([query_emb, local_emb, np.abs(query_emb - local_emb), numeric])
    if safe_names:
        combined = [matrices[f"{safe}_query"][:, :-numeric.shape[1]] for safe in safe_names]
        matrices["all_embeddings_numeric"] = np.hstack(combined + [numeric])
    return matrices


def classifiers() -> dict[str, object]:
    return {
        "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)),
        "linear_svc": make_pipeline(StandardScaler(with_mean=False), SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
        "rf": RandomForestClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1, class_weight="balanced", random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
    }


def fit_predict_proba(clf: object, x_train: np.ndarray, y_train: pd.Series, x_eval: np.ndarray) -> np.ndarray:
    clf.fit(x_train, y_train)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x_eval)
        classes = list(clf.classes_)  # type: ignore[attr-defined]
        return proba[:, classes.index(1)]
    decision = clf.decision_function(x_eval)  # type: ignore[attr-defined]
    return 1.0 / (1.0 + np.exp(-decision))


def fit_predict_classes(clf: object, x_train: np.ndarray, y_train: pd.Series, x_eval: np.ndarray) -> np.ndarray:
    clf.fit(x_train, y_train)
    return clf.predict(x_eval)


def evaluate_actions(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    lambda_cost: float,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        if action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            cost = 0.0
            gpt = False
            gemini = False
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            cost = float(row[f"{GEMINI}_cost"])
            gpt = False
            gemini = True
        elif action == GPT:
            quality = float(row[f"{GPT}_quality"])
            cost = float(row[f"{GPT}_cost"])
            gpt = True
            gemini = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
    all_gpt_cost = max(float(frame[f"{GPT}_cost"].sum()), 1e-12)
    normalized_cost = float(np.sum(costs) / all_gpt_cost)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(mean_quality - lambda_cost * normalized_cost)
    oracle_utility = float(frame["expanded_cost_oracle_utility"].mean())
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_expanded_oracle": float(frame["expanded_quality_oracle"].mean() - mean_quality),
        "utility_ratio_to_expanded_cost_oracle": float(mean_utility / oracle_utility) if oracle_utility else np.nan,
        "normalized_remote_cost_vs_all_gpt": normalized_cost,
        "frontier_call_rate": float(np.mean([a or b for a, b in zip(gpt_calls, gemini_calls)])),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "action_counts": json.dumps({str(key): int(value) for key, value in actions.value_counts().to_dict().items()}),
    }


def route_rows(
    frame: pd.DataFrame,
    local_actions: pd.Series,
    frontier_actions: pd.Series,
    frontier_score: pd.Series,
    *,
    threshold: float | None = None,
    budget_rate: float | None = None,
) -> pd.Series:
    actions = local_actions.copy()
    if threshold is not None:
        chosen = frontier_score[frontier_score.ge(threshold)].index
    elif budget_rate is not None:
        budget = int(np.floor(budget_rate * len(frame)))
        chosen = frontier_score.sort_values(ascending=False).head(budget).index if budget > 0 else []
    else:
        chosen = []
    actions.loc[chosen] = frontier_actions.loc[chosen]
    return actions


def evaluate_matrix(table: pd.DataFrame, x: np.ndarray, matrix_name: str, lambda_cost: float) -> pd.DataFrame:
    train_mask = table["split"].eq("train").to_numpy()
    rows: list[dict[str, object]] = []
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    budget_rates = [0.25, 0.30, 0.35, 0.40]
    for clf_name, clf in classifiers().items():
        try:
            frontier_prob = pd.Series(
                fit_predict_proba(clf, x[train_mask], table.loc[train_mask, "frontier_only_needed"], x),
                index=table.index,
            )
        except Exception as exc:
            rows.append({"method": f"{matrix_name}_{clf_name}", "split": "error", "error": repr(exc)})
            continue
        local_actions = pd.Series(
            fit_predict_classes(classifiers()["extra_trees"], x[train_mask], table.loc[train_mask, "local_oracle_model"], x),
            index=table.index,
        )
        frontier_actions = pd.Series(
            fit_predict_classes(classifiers()["extra_trees"], x[train_mask], table.loc[train_mask, "frontier_utility_oracle_model"], x),
            index=table.index,
        )
        for split, frame in table.groupby("split", sort=False):
            if split == "train":
                continue
            for threshold in thresholds:
                actions = route_rows(
                    frame,
                    local_actions.loc[frame.index],
                    frontier_actions.loc[frame.index],
                    frontier_prob.loc[frame.index],
                    threshold=threshold,
                )
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"{matrix_name}_{clf_name}_threshold{threshold:.2f}",
                    lambda_cost=lambda_cost,
                )
                row["matrix"] = matrix_name
                row["classifier"] = clf_name
                row["threshold"] = threshold
                row["budget_rate"] = np.nan
                rows.append(row)
            for budget_rate in budget_rates:
                actions = route_rows(
                    frame,
                    local_actions.loc[frame.index],
                    frontier_actions.loc[frame.index],
                    frontier_prob.loc[frame.index],
                    budget_rate=budget_rate,
                )
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"{matrix_name}_{clf_name}_budget{budget_rate:.2f}",
                    lambda_cost=lambda_cost,
                )
                row["matrix"] = matrix_name
                row["classifier"] = clf_name
                row["threshold"] = np.nan
                row["budget_rate"] = budget_rate
                rows.append(row)
    return pd.DataFrame(rows)


def select_rows(rows: pd.DataFrame, quality_gap_target: float, frontier_rate_target: float) -> pd.DataFrame:
    rows = rows[rows["split"].isin(["val", "test"])].copy()
    val = rows[rows["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_expanded_oracle"].le(quality_gap_target)
        & val["frontier_call_rate"].le(frontier_rate_target)
    ].copy()
    if feasible.empty:
        feasible = val.copy()
        feasible["selection_status"] = "no_validation_feasible_row"
    else:
        feasible["selection_status"] = "validation_feasible"
    feasible = feasible.sort_values(
        ["selection_status", "utility_ratio_to_expanded_cost_oracle", "mean_quality"],
        ascending=[True, False, False],
    )
    selected = feasible.head(10)
    test = rows[rows["split"].eq("test")]
    return selected.merge(test, on="method", how="left", suffixes=("_val", "_test"))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    table = add_targets(table)
    matrices = build_feature_matrices(table, output_dir, args.embedding_models, args.batch_size)
    all_rows = []
    for matrix_name, matrix in matrices.items():
        all_rows.append(evaluate_matrix(table, matrix, matrix_name, args.lambda_cost))
    rows = pd.concat(all_rows, ignore_index=True)
    selected = select_rows(rows, args.quality_gap_target, args.frontier_rate_target)

    table_path = output_dir / "query_table_embedding_frontier_gate.csv"
    rows_path = output_dir / "table_embedding_frontier_gate.csv"
    selected_path = output_dir / "table_embedding_frontier_gate_selected.csv"
    memo_path = output_dir / "EMBEDDING_FRONTIER_GATE_MEMO.md"
    table.to_csv(table_path, index=False)
    rows.to_csv(rows_path, index=False)
    selected.to_csv(selected_path, index=False)

    display_cols = [
        "method",
        "selection_status",
        "mean_quality_val",
        "quality_gap_to_expanded_oracle_val",
        "utility_ratio_to_expanded_cost_oracle_val",
        "normalized_remote_cost_vs_all_gpt_val",
        "frontier_call_rate_val",
        "mean_quality_test",
        "quality_gap_to_expanded_oracle_test",
        "utility_ratio_to_expanded_cost_oracle_test",
        "normalized_remote_cost_vs_all_gpt_test",
        "frontier_call_rate_test",
        "gpt_call_rate_test",
    ]
    best_test = rows[rows["split"].eq("test")].sort_values(
        ["utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False
    ).head(15)
    best_cap = rows[rows["split"].eq("test") & rows["frontier_call_rate"].le(args.frontier_rate_target + 1e-12)].sort_values(
        ["mean_quality", "utility_ratio_to_expanded_cost_oracle"], ascending=False
    ).head(15)
    memo = [
        "# Embedding Frontier Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Embedding models: `{', '.join(args.embedding_models)}` loaded with `local_files_only=True`.",
        "This experiment uses local query/local-answer embeddings plus local-only numeric/agreement features. It trains on train only, selects on validation, and reports held-out test rows.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected[[col for col in display_cols if col in selected.columns]]),
        "",
        "## Best Held-Out Test Rows By Utility",
        "",
        markdown_table(best_test),
        "",
        "## Best Held-Out Test Rows Under Frontier Cap",
        "",
        markdown_table(best_cap),
        "",
        "## Files",
        "",
        f"- `{table_path}`",
        f"- `{rows_path}`",
        f"- `{selected_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
