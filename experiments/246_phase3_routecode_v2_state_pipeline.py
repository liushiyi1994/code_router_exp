from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.eval.predictor_diagnostics import expected_calibration_error
from routecode.states.utility_states_v2 import (
    EmbeddingStatePredictor,
    confidence_trigger_mask,
    fit_utility_state_model,
    select_confidence_threshold,
    state_policy,
    state_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RouteCode v2 utility-state and query-to-state pipeline.")
    parser.add_argument(
        "--broad-outputs",
        type=Path,
        default=Path("results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet"),
    )
    parser.add_argument(
        "--new-outputs",
        type=Path,
        default=Path("results/phase3_new_benchmark_live/live_smoke_qwen4_gpt_15/model_outputs.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_routecode_v2_state_pipeline"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k-values", type=int, nargs="*", default=[16, 24])
    parser.add_argument(
        "--state-methods",
        nargs="*",
        default=["raw_kmeans", "relative_kmeans", "two_stage_relative_kmeans", "calibration_refined"],
    )
    parser.add_argument("--predictors", nargs="*", default=["knn", "mlp"])
    parser.add_argument("--active-label-budgets", type=int, nargs="*", default=[64, 128, 256, 492])
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-active-rate", type=float, default=0.30)
    parser.add_argument("--ood-percentile", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_broad_outputs(args.broad_outputs, lambda_cost=float(args.lambda_cost))
    query_table = query_metadata(outputs)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean").dropna(axis=0)
    query_table = align_query_table(query_table, utility.index.astype(str))
    outputs = outputs[outputs["query_id"].isin(set(query_table["query_id"].astype(str)))].copy()
    local_models, frontier_models = model_families(outputs)
    embeddings = load_or_encode_embeddings(query_table, args.output_dir, args.embedding_model, prefix="broad100")

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    state_cards: list[pd.DataFrame] = []
    new_rows: list[dict[str, Any]] = []
    new_assignments: list[pd.DataFrame] = []
    active_learning_rows: list[dict[str, Any]] = []

    for k in args.k_values:
        for state_method in args.state_methods:
            train_ids = query_ids_for_split(query_table, "train")
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
            state_utility, state_variance = state_tables(utility.reindex(model.labels.index), model.labels)
            label_to_model = state_utility.idxmax(axis=1).astype(str).to_dict()
            fallback_model = str(utility.reindex(model.labels.index).mean(axis=0).sort_values(ascending=False).index[0])
            state_cards.append(build_state_cards(model, true_labels, query_table, state_utility, state_variance))
            assignments.append(
                true_labels.reset_index()
                .rename(columns={"index": "query_id", "state_label": "group_id"})
                .merge(query_table[["query_id", "split", "benchmark", "domain"]], on="query_id", how="left")
                .assign(state_method=state_method, k=int(k), assignment_kind="utility_state_label")
            )

            for split in ["val", "test"]:
                split_ids = query_ids_for_split(query_table, split)
                true_selection = state_policy(true_labels.reindex(split_ids), label_to_model, fallback_model)
                rows.append(
                    evaluate_selection(
                        outputs,
                        true_selection,
                        split=split,
                        method=f"{state_method}_k{k}_diagnostic_true_state",
                        state_method=state_method,
                        predictor="oracle_state",
                        k=int(k),
                        confidence_threshold=np.nan,
                        active_probe_rate=0.0,
                        policy_kind="diagnostic_true_state",
                    )
                )

            for predictor_kind in args.predictors:
                predictor = EmbeddingStatePredictor(
                    kind=predictor_kind,
                    n_neighbors=15,
                    hidden_layer_sizes=(128, 64),
                    max_iter=300,
                    random_state=int(args.seed),
                ).fit(embeddings.reindex(model.labels.index), model.labels)
                pred = predictor.predict(embeddings)
                val_ids = query_ids_for_split(query_table, "val")
                threshold = select_confidence_threshold(
                    pred.confidence.reindex(val_ids),
                    pred.labels.reindex(val_ids),
                    true_labels.reindex(val_ids),
                    max_probe_rate=float(args.max_active_rate),
                )
                for split in ["val", "test"]:
                    split_ids = query_ids_for_split(query_table, split)
                    split_pred = pred.labels.reindex(split_ids)
                    split_conf = pred.confidence.reindex(split_ids)
                    split_true = true_labels.reindex(split_ids)
                    probe = confidence_trigger_mask(split_conf, threshold)
                    diagnostics.append(
                        prediction_diagnostic_row(
                            state_method=state_method,
                            predictor=predictor_kind,
                            split=split,
                            k=int(k),
                            true_labels=split_true,
                            predicted_labels=split_pred,
                            confidence=split_conf,
                            threshold=threshold,
                        )
                    )

                    plain_selection = state_policy(split_pred, label_to_model, fallback_model)
                    rows.append(
                        evaluate_selection(
                            outputs,
                            plain_selection,
                            split=split,
                            method=f"{state_method}_k{k}_{predictor_kind}_plain",
                            state_method=state_method,
                            predictor=predictor_kind,
                            k=int(k),
                            confidence_threshold=threshold,
                            active_probe_rate=0.0,
                            policy_kind="plain_predicted_state",
                        )
                    )

                    fallback_selection = plain_selection.copy()
                    fallback_selection.loc[probe[probe].index] = fallback_model
                    rows.append(
                        evaluate_selection(
                            outputs,
                            fallback_selection,
                            split=split,
                            method=f"{state_method}_k{k}_{predictor_kind}_lowconf_fallback",
                            state_method=state_method,
                            predictor=predictor_kind,
                            k=int(k),
                            confidence_threshold=threshold,
                            active_probe_rate=float(probe.mean()),
                            policy_kind="low_confidence_fallback",
                        )
                    )

                    active_labels = split_pred.copy()
                    active_labels.loc[probe[probe].index] = split_true.loc[probe[probe].index]
                    active_selection = state_policy(active_labels, label_to_model, fallback_model)
                    rows.append(
                        evaluate_selection(
                            outputs,
                            active_selection,
                            split=split,
                            method=f"{state_method}_k{k}_{predictor_kind}_active_state_reveal",
                            state_method=state_method,
                            predictor=predictor_kind,
                            k=int(k),
                            confidence_threshold=threshold,
                            active_probe_rate=float(probe.mean()),
                            policy_kind="diagnostic_active_state_reveal",
                        )
                    )

                if args.new_outputs.exists():
                    new_eval, new_assign = evaluate_new_benchmark(
                        args,
                        model,
                        predictor,
                        query_table,
                        embeddings,
                        outputs,
                        state_method=state_method,
                        predictor_kind=predictor_kind,
                        k=int(k),
                        threshold=threshold,
                    )
                    new_rows.extend(new_eval)
                    new_assignments.append(new_assign)

                if state_method == "relative_kmeans" and int(k) == min(args.k_values):
                    active_learning_rows.extend(
                        run_active_query_state_learning(
                            args,
                            outputs,
                            embeddings,
                            true_labels,
                            query_table,
                            label_to_model,
                            fallback_model,
                            state_method=state_method,
                            predictor_kind=predictor_kind,
                            k=int(k),
                        )
                    )

    policy_table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    diagnostics_table = pd.DataFrame(diagnostics).sort_values(["split", "state_method", "k", "predictor"])
    assignment_table = pd.concat(assignments, ignore_index=True)
    state_card_table = pd.concat(state_cards, ignore_index=True)
    new_policy = pd.DataFrame(new_rows).sort_values(["mean_utility", "mean_quality"], ascending=False)
    new_assignment_table = pd.concat(new_assignments, ignore_index=True) if new_assignments else pd.DataFrame()
    active_learning = pd.DataFrame(active_learning_rows)

    policy_table.to_csv(args.output_dir / "table_v2_state_policy.csv", index=False)
    diagnostics_table.to_csv(args.output_dir / "table_v2_query_state_predictor_diagnostics.csv", index=False)
    assignment_table.to_csv(args.output_dir / "table_v2_state_assignments.csv", index=False)
    state_card_table.to_csv(args.output_dir / "table_v2_state_cards.csv", index=False)
    if not new_policy.empty:
        new_policy.to_csv(args.output_dir / "table_v2_new_benchmark_policy.csv", index=False)
        new_assignment_table.to_csv(args.output_dir / "table_v2_new_benchmark_assignments.csv", index=False)
    if not active_learning.empty:
        active_learning = active_learning.sort_values(["predictor", "budget", "strategy"])
        active_learning.to_csv(args.output_dir / "table_v2_active_query_state_learning.csv", index=False)
    write_readme(args, policy_table, diagnostics_table, new_policy, active_learning)

    print(f"Wrote RouteCode v2 state pipeline to {args.output_dir}")
    print(policy_table[policy_table["split"].eq("test")].head(20).to_string(index=False))
    if not new_policy.empty:
        print("\nNew benchmark:")
        print(new_policy.head(20).to_string(index=False))


def load_broad_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    outputs["benchmark"] = outputs["benchmark"].astype(str)
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce")
    outputs["normalized_remote_cost"] = pd.to_numeric(outputs["normalized_remote_cost"], errors="coerce").fillna(0.0)
    outputs["cost_total_usd"] = pd.to_numeric(outputs["cost_total_usd"], errors="coerce").fillna(0.0)
    outputs["latency_s"] = pd.to_numeric(outputs["latency_s"], errors="coerce").fillna(0.0)
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    return outputs.dropna(subset=["quality_score", "utility"])


def load_new_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce")
    outputs["cost_total_usd"] = pd.to_numeric(outputs["cost_total_usd"], errors="coerce").fillna(0.0)
    outputs["latency_s"] = pd.to_numeric(outputs["latency_s"], errors="coerce").fillna(0.0)
    gpt_cost = outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()) if not gpt_cost.empty else float(outputs["cost_total_usd"].max()), 1e-12)
    outputs["normalized_remote_cost"] = outputs["cost_total_usd"] / cost_norm
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    outputs["split"] = "new_benchmark"
    return outputs.dropna(subset=["quality_score", "utility"])


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    cols = ["query_id", "query_text", "split", "benchmark", "domain", "metric"]
    present = [col for col in cols if col in outputs.columns]
    table = outputs[present].drop_duplicates("query_id").copy()
    table["query_id"] = table["query_id"].astype(str)
    return table


def align_query_table(query_table: pd.DataFrame, query_ids: pd.Index) -> pd.DataFrame:
    return query_table.set_index("query_id").reindex(query_ids.astype(str)).reset_index().rename(columns={"index": "query_id"})


def model_families(outputs: pd.DataFrame) -> tuple[list[str], list[str]]:
    meta = outputs[["model_id", "is_local", "is_frontier"]].drop_duplicates("model_id")
    local = meta[meta["is_local"].astype(bool)]["model_id"].astype(str).tolist()
    frontier = meta[meta["is_frontier"].astype(bool)]["model_id"].astype(str).tolist()
    if not local:
        local = sorted([model for model in outputs["model_id"].astype(str).unique() if "local" in model or "deterministic" in model])
    if not frontier:
        frontier = sorted([model for model in outputs["model_id"].astype(str).unique() if model not in set(local)])
    return local, frontier


def query_ids_for_split(query_table: pd.DataFrame, split: str) -> pd.Index:
    return pd.Index(query_table[query_table["split"].astype(str).eq(split)]["query_id"].astype(str), name="query_id")


def assign_all_splits(model, utility: pd.DataFrame, query_table: pd.DataFrame) -> pd.Series:
    labels = model.predict_from_utility(utility.reindex(query_table["query_id"].astype(str)).dropna(axis=0))
    labels.loc[model.labels.index] = model.labels
    return labels.rename("state_label")


def load_or_encode_embeddings(query_table: pd.DataFrame, output_dir: Path, model_name: str, *, prefix: str) -> pd.DataFrame:
    cache_path = output_dir / f"{prefix}_embeddings_{safe_name(model_name)}_{len(query_table)}.npy"
    ids_path = output_dir / f"{prefix}_embedding_query_ids.csv"
    if cache_path.exists() and ids_path.exists():
        arr = np.load(cache_path)
        ids = pd.read_csv(ids_path)["query_id"].astype(str).tolist()
        return pd.DataFrame(arr, index=ids)
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

        tfidf = TfidfVectorizer(min_df=1, max_features=4096, ngram_range=(1, 2), stop_words="english")
        x = tfidf.fit_transform(query_table["query_text"].fillna("").astype(str))
        n_components = min(256, max(2, min(x.shape) - 1))
        arr = normalize(TruncatedSVD(n_components=n_components, random_state=17).fit_transform(x)).astype(np.float32)
    np.save(cache_path, arr)
    query_table[["query_id"]].to_csv(ids_path, index=False)
    return pd.DataFrame(arr, index=query_table["query_id"].astype(str).tolist())


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def evaluate_selection(
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    state_method: str,
    predictor: str,
    k: int,
    confidence_threshold: float,
    active_probe_rate: float,
    policy_kind: str,
) -> dict[str, Any]:
    split_outputs = outputs[outputs["split"].astype(str).eq(split)].copy()
    selected_frame = selected.rename("model_id").reset_index().rename(columns={"index": "query_id"})
    selected_frame["query_id"] = selected_frame["query_id"].astype(str)
    selected_frame["model_id"] = selected_frame["model_id"].astype(str)
    rows = selected_frame.merge(split_outputs, on=["query_id", "model_id"], how="inner")
    oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()].copy()
    return {
        "method": method,
        "split": split,
        "state_method": state_method,
        "predictor": predictor,
        "k": int(k),
        "policy_kind": policy_kind,
        "n_queries": int(rows["query_id"].nunique()),
        "mean_quality": float(rows["quality_score"].mean()),
        "mean_utility": float(rows["utility"].mean()),
        "oracle_mean_utility": float(oracle["utility"].mean()),
        "utility_gap_to_oracle": float(oracle["utility"].mean() - rows["utility"].mean()),
        "oracle_utility_ratio": float(rows["utility"].mean() / oracle["utility"].mean()) if abs(float(oracle["utility"].mean())) > 1e-12 else np.nan,
        "remote_cost_total_usd": float(rows["cost_total_usd"].sum()),
        "frontier_call_rate": float(rows.get("is_frontier", pd.Series(False, index=rows.index)).astype(bool).mean()),
        "mean_latency_s": float(rows["latency_s"].mean()),
        "confidence_threshold": float(confidence_threshold) if pd.notna(confidence_threshold) else np.nan,
        "active_probe_rate": float(active_probe_rate),
        "selected_models_json": json.dumps(rows["model_id"].astype(str).value_counts().to_dict(), sort_keys=True),
    }


def prediction_diagnostic_row(
    *,
    state_method: str,
    predictor: str,
    split: str,
    k: int,
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    confidence: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    aligned = pd.concat(
        [
            true_labels.rename("true").astype(str),
            predicted_labels.rename("predicted").astype(str),
            confidence.rename("confidence").astype(float),
        ],
        axis=1,
        join="inner",
    ).dropna()
    correct = aligned["true"].eq(aligned["predicted"])
    probe = aligned["confidence"].lt(float(threshold))
    covered = aligned[~probe]
    return {
        "state_method": state_method,
        "predictor": predictor,
        "split": split,
        "k": int(k),
        "n_queries": int(len(aligned)),
        "state_accuracy": float(correct.mean()) if len(aligned) else np.nan,
        "mean_confidence": float(aligned["confidence"].mean()) if len(aligned) else np.nan,
        "ece": expected_calibration_error(aligned["confidence"], correct.astype(float), n_bins=10),
        "confidence_threshold": float(threshold),
        "active_probe_rate": float(probe.mean()) if len(aligned) else np.nan,
        "covered_state_accuracy": float(covered["true"].eq(covered["predicted"]).mean()) if not covered.empty else np.nan,
        "covered_queries": int(len(covered)),
    }


def build_state_cards(model, labels: pd.Series, query_table: pd.DataFrame, state_utility: pd.DataFrame, state_variance: pd.DataFrame) -> pd.DataFrame:
    meta = query_table.set_index("query_id").reindex(labels.index.astype(str))
    rows = []
    for state, state_labels in labels.groupby(labels.astype(str)):
        ids = state_labels.index.astype(str)
        bench = meta.reindex(ids)["benchmark"].astype(str).value_counts()
        domain = meta.reindex(ids)["domain"].astype(str).value_counts()
        utilities = state_utility.loc[state].astype(float).sort_values(ascending=False)
        variance = state_variance.loc[state].astype(float)
        rows.append(
            {
                "state_method": model.method,
                "k": int(model.n_states),
                "state_label": state,
                "n_queries": int(len(ids)),
                "best_model": str(utilities.index[0]),
                "best_utility": float(utilities.iloc[0]),
                "second_model": str(utilities.index[1]) if len(utilities) > 1 else "",
                "margin": float(utilities.iloc[0] - utilities.iloc[1]) if len(utilities) > 1 else float(utilities.iloc[0]),
                "mean_utility_variance": float(variance.mean()),
                "top_benchmark": str(bench.index[0]) if not bench.empty else "",
                "top_benchmark_frac": float(bench.iloc[0] / len(ids)) if not bench.empty else np.nan,
                "top_domain": str(domain.index[0]) if not domain.empty else "",
                "top_domain_frac": float(domain.iloc[0] / len(ids)) if not domain.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_active_query_state_learning(
    args: argparse.Namespace,
    outputs: pd.DataFrame,
    embeddings: pd.DataFrame,
    true_labels: pd.Series,
    query_table: pd.DataFrame,
    label_to_model: dict[str, str],
    fallback_model: str,
    *,
    state_method: str,
    predictor_kind: str,
    k: int,
) -> list[dict[str, Any]]:
    train_ids = query_ids_for_split(query_table, "train")
    test_ids = query_ids_for_split(query_table, "test")
    train_labels = true_labels.reindex(train_ids).dropna().astype(str)
    test_labels = true_labels.reindex(test_ids).dropna().astype(str)
    rng = np.random.default_rng(int(args.seed) + int(k) + (0 if predictor_kind == "knn" else 1000))
    rows: list[dict[str, Any]] = []
    for strategy in ["random", "active_low_confidence"]:
        for budget in args.active_label_budgets:
            labeled_ids = active_label_ids(
                embeddings,
                train_labels,
                budget=int(budget),
                strategy=strategy,
                predictor_kind=predictor_kind,
                rng=rng,
                seed=int(args.seed),
            )
            predictor = EmbeddingStatePredictor(
                kind=predictor_kind,
                n_neighbors=15,
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                random_state=int(args.seed),
            ).fit(embeddings.reindex(labeled_ids), train_labels.reindex(labeled_ids))
            pred = predictor.predict(embeddings.reindex(test_labels.index))
            selected = state_policy(pred.labels, label_to_model, fallback_model)
            eval_row = evaluate_selection(
                outputs,
                selected,
                split="test",
                method=f"{state_method}_k{k}_{predictor_kind}_{strategy}_active_labels_b{len(labeled_ids)}",
                state_method=state_method,
                predictor=predictor_kind,
                k=int(k),
                confidence_threshold=np.nan,
                active_probe_rate=0.0,
                policy_kind=f"active_query_state_learning_{strategy}",
            )
            eval_row.update(
                {
                    "strategy": strategy,
                    "budget": int(budget),
                    "labeled_queries": int(len(labeled_ids)),
                    "state_accuracy": float(pred.labels.astype(str).eq(test_labels.astype(str)).mean()),
                    "mean_confidence": float(pred.confidence.mean()),
                }
            )
            rows.append(eval_row)
    return rows


def active_label_ids(
    embeddings: pd.DataFrame,
    labels: pd.Series,
    *,
    budget: int,
    strategy: str,
    predictor_kind: str,
    rng: np.random.Generator,
    seed: int,
) -> pd.Index:
    labels = labels.dropna().astype(str)
    budget = min(max(int(budget), 0), len(labels))
    if budget == 0:
        return pd.Index([], name=labels.index.name)
    labeled: list[str] = []
    for state in sorted(labels.unique()):
        candidates = labels.index[labels.eq(state)].astype(str).to_numpy()
        if len(candidates) == 0:
            continue
        labeled.append(str(rng.choice(candidates)))
        if len(labeled) >= budget:
            return pd.Index(labeled, name=labels.index.name)
    while len(labeled) < budget:
        unlabeled = pd.Index([query_id for query_id in labels.index.astype(str) if query_id not in set(labeled)])
        if unlabeled.empty:
            break
        take = min(16, budget - len(labeled), len(unlabeled))
        if strategy == "random" or len(set(labels.reindex(labeled))) < 2:
            chosen = rng.choice(unlabeled.to_numpy(dtype=object), size=take, replace=False).astype(str).tolist()
        elif strategy == "active_low_confidence":
            predictor = EmbeddingStatePredictor(
                kind=predictor_kind,
                n_neighbors=15,
                hidden_layer_sizes=(128, 64),
                max_iter=200,
                random_state=int(seed),
            ).fit(embeddings.reindex(labeled), labels.reindex(labeled))
            pred = predictor.predict(embeddings.reindex(unlabeled))
            chosen = pred.confidence.sort_values(ascending=True).head(take).index.astype(str).tolist()
        else:
            raise ValueError(f"Unknown active label strategy: {strategy}")
        labeled.extend([query_id for query_id in chosen if query_id not in set(labeled)])
    return pd.Index(labeled[:budget], name=labels.index.name)


def evaluate_new_benchmark(
    args: argparse.Namespace,
    model,
    predictor: EmbeddingStatePredictor,
    broad_query_table: pd.DataFrame,
    broad_embeddings: pd.DataFrame,
    broad_outputs: pd.DataFrame,
    *,
    state_method: str,
    predictor_kind: str,
    k: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    new_outputs = load_new_outputs(args.new_outputs, lambda_cost=float(args.lambda_cost))
    new_query_table = query_metadata(new_outputs)
    new_embeddings = load_or_encode_embeddings(new_query_table, args.output_dir, args.embedding_model, prefix="new")
    pred = predictor.predict(new_embeddings)
    common_models = sorted(set(new_outputs["model_id"].astype(str)).intersection(set(broad_outputs["model_id"].astype(str))))
    train_ids = pd.Index(model.labels.index.astype(str))
    train_common = (
        broad_outputs[
            broad_outputs["query_id"].isin(set(train_ids))
            & broad_outputs["model_id"].astype(str).isin(common_models)
        ]
        .pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
        .dropna(axis=0)
    )
    train_labels = model.labels.reindex(train_common.index).dropna().astype(str)
    common_state_utility, _ = state_tables(train_common.reindex(train_labels.index), train_labels)
    label_to_model = common_state_utility.idxmax(axis=1).astype(str).to_dict()
    fallback_model = str(train_common.mean(axis=0).sort_values(ascending=False).index[0])
    plain = state_policy(pred.labels, label_to_model, fallback_model)
    semantic_gate = semantic_ood_gate(
        broad_query_table,
        broad_embeddings,
        model.labels,
        new_embeddings,
        percentile=float(args.ood_percentile),
    )
    low_conf = confidence_trigger_mask(pred.confidence, threshold)
    gated = plain.copy()
    preferred_remote = "gpt-5.5" if "gpt-5.5" in common_models else fallback_model
    gated.loc[(semantic_gate | low_conf).reindex(gated.index).fillna(True)] = preferred_remote

    rows = [
        evaluate_new_selection(new_outputs, plain, state_method, predictor_kind, k, "plain_predicted_state", threshold, float(low_conf.mean()), semantic_gate),
        evaluate_new_selection(new_outputs, gated, state_method, predictor_kind, k, "semantic_or_lowconf_remote_gate", threshold, float((semantic_gate | low_conf).mean()), semantic_gate),
    ]
    assignment = pd.DataFrame(
        {
            "query_id": pred.labels.index.astype(str),
            "state_method": state_method,
            "predictor": predictor_kind,
            "k": int(k),
            "predicted_state": pred.labels.to_numpy(dtype=str),
            "state_confidence": pred.confidence.to_numpy(dtype=float),
            "semantic_ood_gate": semantic_gate.reindex(pred.labels.index).fillna(True).to_numpy(dtype=bool),
            "low_confidence_gate": low_conf.reindex(pred.labels.index).fillna(True).to_numpy(dtype=bool),
            "plain_model": plain.reindex(pred.labels.index).astype(str).to_numpy(),
            "gated_model": gated.reindex(pred.labels.index).astype(str).to_numpy(),
        }
    )
    return rows, assignment


def semantic_ood_gate(
    query_table: pd.DataFrame,
    embeddings: pd.DataFrame,
    train_labels: pd.Series,
    new_embeddings: pd.DataFrame,
    *,
    percentile: float,
) -> pd.Series:
    train_emb = embeddings.reindex(train_labels.index).dropna(axis=0, how="any")
    labels = train_labels.reindex(train_emb.index).astype(str)
    centroids = []
    state_ids = []
    for state, state_labels in labels.groupby(labels):
        state_ids.append(str(state))
        centroids.append(train_emb.reindex(state_labels.index).mean(axis=0).to_numpy(dtype=float))
    centroid_matrix = normalize_rows(np.vstack(centroids))
    train_sims = train_emb.to_numpy(dtype=float) @ centroid_matrix.T
    train_max = train_sims.max(axis=1)
    threshold = float(np.quantile(train_max, percentile))
    new_sims = new_embeddings.to_numpy(dtype=float) @ centroid_matrix.T
    max_sim = new_sims.max(axis=1)
    return pd.Series(max_sim < threshold, index=new_embeddings.index, name="semantic_ood_gate")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def evaluate_new_selection(
    outputs: pd.DataFrame,
    selected: pd.Series,
    state_method: str,
    predictor: str,
    k: int,
    policy_kind: str,
    threshold: float,
    active_rate: float,
    semantic_gate: pd.Series,
) -> dict[str, Any]:
    selected_frame = selected.rename("model_id").reset_index().rename(columns={"index": "query_id"})
    selected_frame["query_id"] = selected_frame["query_id"].astype(str)
    selected_frame["model_id"] = selected_frame["model_id"].astype(str)
    rows = selected_frame.merge(outputs, on=["query_id", "model_id"], how="inner")
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()].copy()
    return {
        "method": f"{state_method}_k{k}_{predictor}_{policy_kind}",
        "split": "new_benchmark",
        "state_method": state_method,
        "predictor": predictor,
        "k": int(k),
        "policy_kind": policy_kind,
        "n_queries": int(rows["query_id"].nunique()),
        "mean_quality": float(rows["quality_score"].mean()),
        "mean_utility": float(rows["utility"].mean()),
        "oracle_mean_utility": float(oracle["utility"].mean()),
        "utility_gap_to_oracle": float(oracle["utility"].mean() - rows["utility"].mean()),
        "oracle_utility_ratio": float(rows["utility"].mean() / oracle["utility"].mean()) if abs(float(oracle["utility"].mean())) > 1e-12 else np.nan,
        "remote_cost_total_usd": float(rows["cost_total_usd"].sum()),
        "frontier_call_rate": float(rows.get("is_frontier", pd.Series(False, index=rows.index)).astype(bool).mean()),
        "mean_latency_s": float(rows["latency_s"].mean()),
        "confidence_threshold": float(threshold),
        "active_probe_rate": float(active_rate),
        "semantic_ood_rate": float(semantic_gate.mean()),
        "selected_models_json": json.dumps(rows["model_id"].astype(str).value_counts().to_dict(), sort_keys=True),
    }


def write_readme(
    args: argparse.Namespace,
    policy: pd.DataFrame,
    diagnostics: pd.DataFrame,
    new_policy: pd.DataFrame,
    active_learning: pd.DataFrame,
) -> None:
    test_top = policy[policy["split"].eq("test")].sort_values("mean_utility", ascending=False).head(12)
    test_lines = "\n".join(
        f"| {row.method} | {row.mean_quality:.4f} | {row.mean_utility:.4f} | {row.oracle_utility_ratio:.4f} | "
        f"{row.active_probe_rate:.4f} | {row.selected_models_json} |"
        for row in test_top.itertuples(index=False)
    )
    diag_top = diagnostics[diagnostics["split"].eq("test")].sort_values("state_accuracy", ascending=False).head(12)
    diag_lines = "\n".join(
        f"| {row.state_method} | {int(row.k)} | {row.predictor} | {row.state_accuracy:.4f} | "
        f"{row.mean_confidence:.4f} | {row.ece:.4f} | {row.active_probe_rate:.4f} | {row.covered_state_accuracy:.4f} |"
        for row in diag_top.itertuples(index=False)
    )
    if new_policy.empty:
        new_lines = "| not run | | | | | |"
    else:
        new_lines = "\n".join(
            f"| {row.method} | {row.mean_quality:.4f} | {row.mean_utility:.4f} | {row.oracle_utility_ratio:.4f} | "
            f"{row.active_probe_rate:.4f} | {row.selected_models_json} |"
            for row in new_policy.head(12).itertuples(index=False)
        )
    if active_learning.empty:
        active_lines = "| not run | | | | | | |"
    else:
        active_top = active_learning.sort_values(["budget", "mean_utility"], ascending=[True, False])
        active_lines = "\n".join(
            f"| {row.predictor} | {row.strategy} | {int(row.budget)} | {row.state_accuracy:.4f} | "
            f"{row.mean_utility:.4f} | {row.oracle_utility_ratio:.4f} | {row.selected_models_json} |"
            for row in active_top.itertuples(index=False)
        )
    body = f"""# RouteCode V2 State Pipeline

This run implements the split proposed in the Phase 3 direction change:

```text
Model 1: utility matrix -> latent states + state-to-model table
Model 2: query text embedding -> p(state | query)
```

State learner variants:

- `raw_kmeans`
- `relative_kmeans`
- `two_stage_relative_kmeans`
- `calibration_refined`

Query-to-state predictors:

- KNN over local query embeddings
- MLP over local query embeddings

Low-confidence predictions trigger either:

- deployable fallback to the train-best model; or
- diagnostic active-state reveal, an upper bound for what active probing could recover.

## Top Test Policies

| method | quality | utility | oracle utility ratio | active/probe rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
{test_lines}

## Query-to-State Diagnostics

| state method | K | predictor | state accuracy | mean confidence | ECE | probe rate | covered accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
{diag_lines}

## New Benchmark Smoke

| method | quality | utility | oracle utility ratio | gate rate | selected models |
| --- | ---: | ---: | ---: | ---: | --- |
{new_lines}

## Active Query-State Labeling Simulation

This is train-only active learning for the query-to-state classifier. It starts
with one labeled query per state, then either samples random additional state
labels or samples low-confidence training queries. Evaluation is on held-out
Broad100 test queries.

| predictor | strategy | labeled queries | state accuracy | utility | oracle utility ratio | selected models |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{active_lines}

## Artifacts

- `table_v2_state_policy.csv`
- `table_v2_query_state_predictor_diagnostics.csv`
- `table_v2_state_assignments.csv`
- `table_v2_state_cards.csv`
- `table_v2_new_benchmark_policy.csv`
- `table_v2_new_benchmark_assignments.csv`
- `table_v2_active_query_state_learning.csv`

Command:

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py
```
"""
    (args.output_dir / "README.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
