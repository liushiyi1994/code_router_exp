from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ID_COLS = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}
CONFIDENCE_COLS = [
    "local_valid_count",
    "local_missing_count",
    "local_unique_answer_count",
    "local_vote_frac",
    "local_vote_margin",
    "local_vote_entropy",
    "local_all_agree",
    "small_valid_count",
    "small_unique_answer_count",
    "medium_valid_count",
    "medium_unique_answer_count",
    "small_medium_agree",
    "small_medium_disagree",
    "q4_q8_agree",
    "q4_q14_agree",
    "q8_q14_agree",
    "q14_q32_agree",
    "q32_sc_agree",
    "sc_vote_frac",
    "sc_vote_margin",
    "sc_vote_entropy",
    "sc_all_samples_agree",
    "q4_lp_logprob_mean",
    "q4_lp_logprob_min",
    "q4_lp_logprob_margin_mean",
    "q4_lp_logprob_margin_min",
    "q4_lp_logprob_first_token_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether utility states are semantically coherent and aligned with local confidence."
    )
    parser.add_argument(
        "--broad-outputs",
        type=Path,
        default=Path("results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet"),
    )
    parser.add_argument(
        "--broad-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--new-outputs",
        type=Path,
        default=Path("results/phase3_new_benchmark_live/live_smoke_qwen4_gpt_15/model_outputs.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_state_semantic_confidence_audit"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k-values", type=int, nargs="*", default=[16, 24])
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(args.broad_outputs)
    query_table = query_metadata(outputs)
    features = load_feature_table(args.broad_features, query_table)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean").dropna(axis=0)
    query_table = query_table[query_table["query_id"].isin(utility.index.astype(str))].copy()
    query_table = query_table.set_index("query_id").loc[utility.index.astype(str)].reset_index()
    features = features.set_index("query_id").reindex(query_table["query_id"].astype(str)).reset_index()
    embeddings = load_or_encode_embeddings(query_table, args.output_dir, args.embedding_model)

    all_rows: list[pd.DataFrame] = []
    semantic_rows: list[dict[str, Any]] = []
    state_card_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    local_quality = local_quality_table(outputs)

    new_outputs = load_new_outputs(args.new_outputs)
    new_query_table = query_metadata(new_outputs)
    new_embeddings = load_or_encode_embeddings(new_query_table, args.output_dir, args.embedding_model, prefix="new")
    new_features = build_new_feature_table(new_outputs, features)

    for k in args.k_values:
        state_assignments, state_model, scaler = utility_state_assignments(
            utility, query_table, k=int(k), seed=int(args.seed)
        )
        all_rows.append(state_assignments)
        group_specs = [
            state_assignments.assign(group_method=f"utility_state_k{k}"),
            random_size_matched_groups(state_assignments, k=int(k), seed=int(args.seed)),
            benchmark_groups(query_table),
            embedding_cluster_groups(query_table, embeddings, k=int(k), seed=int(args.seed)),
        ]
        groups = pd.concat(group_specs, ignore_index=True)
        for method, method_groups in groups.groupby("group_method", sort=False):
            for split in ["train", "val", "test", "all"]:
                ids = split_ids(query_table, split)
                frame = method_groups[method_groups["query_id"].isin(ids)].copy()
                if frame.empty:
                    continue
                semantic_rows.append(
                    semantic_summary(frame, embeddings, query_table, method=str(method), split=split)
                )

        state_card_rows.extend(
            utility_state_cards(
                state_assignments,
                embeddings,
                query_table,
                features,
                local_quality,
                k=int(k),
            )
        )
        confidence_rows.extend(confidence_separability(state_assignments, features, k=int(k), split="all"))
        confidence_rows.extend(
            confidence_separability(
                state_assignments[state_assignments["split"].eq("train")].copy(),
                features,
                k=int(k),
                split="train",
            )
        )
        new_rows.extend(
            new_benchmark_semantic_scores(
                state_assignments,
                query_table,
                embeddings,
                new_query_table,
                new_embeddings,
                new_features,
                k=int(k),
            )
        )

    assignments = pd.concat(all_rows, ignore_index=True)
    semantic_table = (
        pd.DataFrame(semantic_rows)
        .drop_duplicates(subset=["group_method", "split", "n_queries", "n_groups"])
        .sort_values(["k", "split", "group_method"])
    )
    state_cards = pd.DataFrame(state_card_rows).sort_values(["k", "state_id"])
    confidence_table = pd.DataFrame(confidence_rows).sort_values(
        ["k", "split", "eta_squared"], ascending=[True, True, False]
    )
    new_table = pd.DataFrame(new_rows).sort_values(["k", "benchmark", "query_id"])

    assignments.to_csv(args.output_dir / "table_utility_state_assignments.csv", index=False)
    semantic_table.to_csv(args.output_dir / "table_state_semantic_coherence.csv", index=False)
    state_cards.to_csv(args.output_dir / "table_utility_state_cards.csv", index=False)
    confidence_table.to_csv(args.output_dir / "table_state_confidence_separability.csv", index=False)
    new_table.to_csv(args.output_dir / "table_new_benchmark_semantic_gate.csv", index=False)
    write_readme(args.output_dir, args, semantic_table, confidence_table, new_table)

    print(f"Wrote semantic/confidence state audit to {args.output_dir}")
    print(semantic_table[semantic_table["split"].eq("test")].to_string(index=False))
    print("\nTop confidence separability rows:")
    print(confidence_table.head(20).to_string(index=False))
    print("\nNew benchmark semantic gate:")
    print(new_table[["k", "query_id", "benchmark", "nearest_state", "max_state_cosine", "train_percentile"]].to_string(index=False))


def load_outputs(path: Path) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["utility"] = pd.to_numeric(outputs["utility"], errors="coerce")
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce")
    return outputs.dropna(subset=["utility"])


def load_new_outputs(path: Path) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce")
    outputs["cost_total_usd"] = pd.to_numeric(outputs["cost_total_usd"], errors="coerce").fillna(0.0)
    outputs["latency_s"] = pd.to_numeric(outputs["latency_s"], errors="coerce").fillna(0.0)
    return outputs.dropna(subset=["quality_score"])


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    cols = ["query_id", "query_text", "split", "benchmark", "domain", "metric"]
    present = [col for col in cols if col in outputs.columns]
    table = outputs[present].drop_duplicates("query_id").copy()
    table["query_id"] = table["query_id"].astype(str)
    if "split" not in table.columns:
        table["split"] = "new_benchmark"
    return table


def load_feature_table(path: Path, query_table: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(path).copy()
    features["query_id"] = features["query_id"].astype(str)
    work = query_table[["query_id", "query_text", "split", "benchmark", "domain"]].merge(
        features, on=["query_id", "query_text", "split", "benchmark", "domain"], how="left"
    )
    for col in work.columns:
        if col in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return work


def build_new_feature_table(new_outputs: pd.DataFrame, feature_template: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in feature_template.columns if col not in ID_COLS and pd.api.types.is_numeric_dtype(feature_template[col])]
    qwen = new_outputs[new_outputs["model_id"].eq("qwen3-4b-local")].set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id, frame in new_outputs.groupby("query_id", sort=False):
        first = frame.iloc[0]
        query_text = str(first.get("query_text", ""))
        metric = str(first.get("metric", ""))
        row: dict[str, Any] = {
            "query_id": str(query_id),
            "query_text": query_text,
            "split": "new_benchmark",
            "benchmark": str(first.get("benchmark", "")),
            "domain": str(first.get("domain", "")),
            "metric": metric,
        }
        for col in numeric_cols:
            row[col] = 0.0
        row["query_chars"] = float(len(query_text))
        row["query_words"] = float(len(query_text.split()))
        row["is_multiple_choice_prompt"] = float(metric == "multiple_choice")
        row["is_exact_answer_prompt"] = float(metric in {"exact_final_answer", "exact_ordered", "short_answer"})
        if str(query_id) in qwen.index:
            qrow = qwen.loc[str(query_id)]
            if isinstance(qrow, pd.DataFrame):
                qrow = qrow.iloc[0]
            parsed = str(qrow.get("parsed_answer", ""))
            valid = float(parsed != "")
            for key, value in {
                "local_valid_count": valid,
                "local_missing_count": 1.0 - valid,
                "local_unique_answer_count": valid,
                "local_top_vote_count": valid,
                "local_vote_frac": valid,
                "local_vote_margin": valid,
                "local_vote_entropy": 0.0,
                "local_all_agree": valid,
                "answer_chars_mean": float(len(parsed)),
                "output_tokens_mean": float(qrow.get("output_tokens", 0.0) or 0.0),
                "qwen4b_valid": valid,
                "qwen4b_answer_chars": float(len(parsed)),
                "qwen4b_status_success": valid,
                "qwen4b_output_tokens": float(qrow.get("output_tokens", 0.0) or 0.0),
                "qwen4b_latency_s": float(qrow.get("latency_s", 0.0) or 0.0),
            }.items():
                if key in row:
                    row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_or_encode_embeddings(
    query_table: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    *,
    prefix: str = "broad100",
) -> np.ndarray:
    cache_path = output_dir / f"{prefix}_embeddings_{safe_name(model_name)}_{len(query_table)}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=True)
        values = model.encode(
            query_table["query_text"].fillna("").astype(str).tolist(),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(values, dtype=np.float32)
    except Exception as exc:
        print(f"Embedding model unavailable ({exc}); falling back to TF-IDF/SVD features.")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        tfidf = TfidfVectorizer(min_df=1, max_features=4096, ngram_range=(1, 2), stop_words="english")
        x = tfidf.fit_transform(query_table["query_text"].fillna("").astype(str))
        n_components = min(256, max(2, min(x.shape) - 1))
        arr = TruncatedSVD(n_components=n_components, random_state=17).fit_transform(x)
        arr = normalize(arr).astype(np.float32)
    np.save(cache_path, arr)
    pd.DataFrame({"query_id": query_table["query_id"].astype(str)}).to_csv(
        output_dir / f"{prefix}_embedding_query_ids.csv", index=False
    )
    return arr


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def utility_state_assignments(
    utility: pd.DataFrame,
    query_table: pd.DataFrame,
    *,
    k: int,
    seed: int,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    train_ids = query_table[query_table["split"].astype(str).eq("train")]["query_id"].astype(str)
    train_matrix = utility.reindex(train_ids).dropna(axis=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_matrix.to_numpy(dtype=float))
    model = KMeans(n_clusters=k, random_state=seed, n_init=30).fit(x_train)
    labels = model.predict(scaler.transform(utility.to_numpy(dtype=float)))
    meta = query_table.set_index("query_id").reindex(utility.index.astype(str))
    frame = pd.DataFrame(
        {
            "query_id": utility.index.astype(str),
            "split": meta["split"].astype(str).to_numpy(),
            "benchmark": meta["benchmark"].astype(str).to_numpy(),
            "domain": meta["domain"].astype(str).to_numpy(),
            "group_method": f"utility_state_k{k}",
            "group_id": [f"u{int(label):02d}" for label in labels],
            "k": int(k),
        }
    )
    return frame, model, scaler


def random_size_matched_groups(assignments: pd.DataFrame, *, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + k)
    frame = assignments.copy()
    labels = frame["group_id"].to_numpy().copy()
    rng.shuffle(labels)
    frame["group_method"] = f"random_size_matched_k{k}"
    frame["group_id"] = labels
    return frame


def benchmark_groups(query_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": query_table["query_id"].astype(str),
            "split": query_table["split"].astype(str),
            "benchmark": query_table["benchmark"].astype(str),
            "domain": query_table["domain"].astype(str),
            "group_method": "benchmark_label",
            "group_id": query_table["benchmark"].astype(str),
            "k": query_table["benchmark"].nunique(),
        }
    )


def embedding_cluster_groups(query_table: pd.DataFrame, embeddings: np.ndarray, *, k: int, seed: int) -> pd.DataFrame:
    train_mask = query_table["split"].astype(str).eq("train").to_numpy()
    model = KMeans(n_clusters=k, random_state=seed, n_init=30).fit(embeddings[train_mask])
    labels = model.predict(embeddings)
    return pd.DataFrame(
        {
            "query_id": query_table["query_id"].astype(str),
            "split": query_table["split"].astype(str),
            "benchmark": query_table["benchmark"].astype(str),
            "domain": query_table["domain"].astype(str),
            "group_method": f"embedding_cluster_k{k}",
            "group_id": [f"e{int(label):02d}" for label in labels],
            "k": int(k),
        }
    )


def split_ids(query_table: pd.DataFrame, split: str) -> set[str]:
    if split == "all":
        return set(query_table["query_id"].astype(str))
    return set(query_table[query_table["split"].astype(str).eq(split)]["query_id"].astype(str))


def semantic_summary(
    groups: pd.DataFrame,
    embeddings: np.ndarray,
    query_table: pd.DataFrame,
    *,
    method: str,
    split: str,
) -> dict[str, Any]:
    index = {query_id: i for i, query_id in enumerate(query_table["query_id"].astype(str))}
    pairwise_values: list[float] = []
    centroid_values: list[float] = []
    weighted_benchmark_purity = 0.0
    total = 0
    usable_groups = 0
    for _, group in groups.groupby("group_id", sort=False):
        positions = [index[str(query_id)] for query_id in group["query_id"].astype(str) if str(query_id) in index]
        if not positions:
            continue
        emb = embeddings[positions]
        centroid = normalize_rows(emb.mean(axis=0, keepdims=True))[0]
        centroid_values.extend((emb @ centroid).astype(float).tolist())
        counts = group["benchmark"].astype(str).value_counts()
        weighted_benchmark_purity += float(counts.iloc[0])
        total += int(len(group))
        if len(positions) >= 2:
            sim = emb @ emb.T
            pairwise_values.append(float((sim.sum() - len(positions)) / (len(positions) * (len(positions) - 1))))
            usable_groups += 1
    return {
        "group_method": method,
        "split": split,
        "k": parse_k(method),
        "n_queries": int(len(groups)),
        "n_groups": int(groups["group_id"].nunique()),
        "usable_pairwise_groups": int(usable_groups),
        "mean_group_pairwise_cosine": float(np.mean(pairwise_values)) if pairwise_values else np.nan,
        "mean_query_to_group_centroid_cosine": float(np.mean(centroid_values)) if centroid_values else np.nan,
        "weighted_benchmark_purity": weighted_benchmark_purity / max(total, 1),
    }


def parse_k(method: str) -> int:
    for token in method.split("_"):
        if token.startswith("k") and token[1:].isdigit():
            return int(token[1:])
    return -1


def utility_state_cards(
    assignments: pd.DataFrame,
    embeddings: np.ndarray,
    query_table: pd.DataFrame,
    features: pd.DataFrame,
    local_quality: pd.DataFrame,
    *,
    k: int,
) -> list[dict[str, Any]]:
    index = {query_id: i for i, query_id in enumerate(query_table["query_id"].astype(str))}
    feature_indexed = features.set_index("query_id")
    quality_indexed = local_quality.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for state_id, group in assignments.groupby("group_id", sort=True):
        positions = [index[str(query_id)] for query_id in group["query_id"].astype(str) if str(query_id) in index]
        emb = embeddings[positions]
        centroid = normalize_rows(emb.mean(axis=0, keepdims=True))[0]
        centroid_cos = (emb @ centroid).astype(float)
        benchmark_counts = group["benchmark"].astype(str).value_counts()
        domain_counts = group["domain"].astype(str).value_counts()
        query_ids = group["query_id"].astype(str).tolist()
        feature_slice = feature_indexed.reindex(query_ids)
        quality_slice = quality_indexed.reindex(query_ids)
        row = {
            "k": int(k),
            "state_id": state_id,
            "n_queries": int(len(group)),
            "train_queries": int(group["split"].astype(str).eq("train").sum()),
            "val_queries": int(group["split"].astype(str).eq("val").sum()),
            "test_queries": int(group["split"].astype(str).eq("test").sum()),
            "mean_query_to_centroid_cosine": float(np.mean(centroid_cos)) if len(centroid_cos) else np.nan,
            "top_benchmark": str(benchmark_counts.index[0]) if not benchmark_counts.empty else "",
            "top_benchmark_frac": float(benchmark_counts.iloc[0] / len(group)) if not benchmark_counts.empty else np.nan,
            "top_domain": str(domain_counts.index[0]) if not domain_counts.empty else "",
            "top_domain_frac": float(domain_counts.iloc[0] / len(group)) if not domain_counts.empty else np.nan,
            "mean_qwen4b_quality": float(quality_slice["qwen4b_quality"].mean()),
            "qwen4b_success_rate": float((quality_slice["qwen4b_quality"] > 0).mean()),
        }
        for col in CONFIDENCE_COLS:
            if col in feature_slice.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(feature_slice[col], errors="coerce").mean())
        rows.append(row)
    return rows


def confidence_separability(assignments: pd.DataFrame, features: pd.DataFrame, *, k: int, split: str) -> list[dict[str, Any]]:
    work = assignments[["query_id", "group_id"]].merge(features, on="query_id", how="left")
    rows: list[dict[str, Any]] = []
    for col in CONFIDENCE_COLS:
        if col not in work.columns:
            continue
        values = pd.to_numeric(work[col], errors="coerce")
        if values.notna().sum() < 2:
            continue
        overall = float(values.mean())
        total_ss = float(((values - overall) ** 2).sum())
        if total_ss <= 1e-12:
            eta = 0.0
        else:
            means = work.assign(value=values).groupby("group_id")["value"].agg(["mean", "count"])
            between_ss = float((means["count"] * (means["mean"] - overall) ** 2).sum())
            eta = between_ss / total_ss
        rows.append(
            {
                "k": int(k),
                "split": split,
                "feature": col,
                "mean_value": overall,
                "eta_squared": float(eta),
                "n_queries": int(values.notna().sum()),
                "n_states": int(work["group_id"].nunique()),
            }
        )
    return rows


def new_benchmark_semantic_scores(
    state_assignments: pd.DataFrame,
    query_table: pd.DataFrame,
    embeddings: np.ndarray,
    new_query_table: pd.DataFrame,
    new_embeddings: np.ndarray,
    new_features: pd.DataFrame,
    *,
    k: int,
) -> list[dict[str, Any]]:
    index = {query_id: i for i, query_id in enumerate(query_table["query_id"].astype(str))}
    train_assignments = state_assignments[state_assignments["split"].astype(str).eq("train")].copy()
    centroids = []
    state_ids = []
    for state_id, group in train_assignments.groupby("group_id", sort=True):
        positions = [index[str(query_id)] for query_id in group["query_id"].astype(str) if str(query_id) in index]
        if not positions:
            continue
        centroids.append(embeddings[positions].mean(axis=0))
        state_ids.append(str(state_id))
    centroid_matrix = normalize_rows(np.vstack(centroids))
    train_max = embeddings[[index[str(query_id)] for query_id in train_assignments["query_id"].astype(str)]] @ centroid_matrix.T
    train_max_values = train_max.max(axis=1)
    sim = new_embeddings @ centroid_matrix.T
    new_feature_index = new_features.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for i, row in new_query_table.reset_index(drop=True).iterrows():
        values = sim[i]
        best = int(np.argmax(values))
        query_id = str(row["query_id"])
        max_sim = float(values[best])
        percentile = float((train_max_values <= max_sim).mean())
        feature_row = new_feature_index.loc[query_id] if query_id in new_feature_index.index else pd.Series(dtype=float)
        rows.append(
            {
                "k": int(k),
                "query_id": query_id,
                "benchmark": str(row.get("benchmark", "")),
                "domain": str(row.get("domain", "")),
                "nearest_state": state_ids[best],
                "max_state_cosine": max_sim,
                "train_percentile": percentile,
                "below_train_p10": bool(percentile < 0.10),
                "local_vote_frac": float(feature_row.get("local_vote_frac", np.nan)),
                "local_vote_entropy": float(feature_row.get("local_vote_entropy", np.nan)),
                "local_all_agree": float(feature_row.get("local_all_agree", np.nan)),
                "qwen4b_output_tokens": float(feature_row.get("qwen4b_output_tokens", np.nan)),
                "qwen4b_answer_chars": float(feature_row.get("qwen4b_answer_chars", np.nan)),
            }
        )
    return rows


def local_quality_table(outputs: pd.DataFrame) -> pd.DataFrame:
    local = outputs[outputs["model_id"].astype(str).eq("qwen3-4b-local")].copy()
    if local.empty:
        return pd.DataFrame({"query_id": outputs["query_id"].drop_duplicates().astype(str), "qwen4b_quality": np.nan})
    return local.groupby("query_id", as_index=False).agg(qwen4b_quality=("quality_score", "mean"))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    semantic: pd.DataFrame,
    confidence: pd.DataFrame,
    new_table: pd.DataFrame,
) -> None:
    test_semantic = semantic[semantic["split"].eq("test")].copy()
    semantic_lines = "\n".join(
        f"| {row.group_method} | {int(row.n_queries)} | {int(row.n_groups)} | "
        f"{row.mean_group_pairwise_cosine:.4f} | {row.mean_query_to_group_centroid_cosine:.4f} | "
        f"{row.weighted_benchmark_purity:.4f} |"
        for row in test_semantic.itertuples(index=False)
    )
    top_conf = confidence.head(12)
    confidence_lines = "\n".join(
        f"| {row.k} | {row.split} | {row.feature} | {row.mean_value:.4f} | {row.eta_squared:.4f} |"
        for row in top_conf.itertuples(index=False)
    )
    new_summary = (
        new_table.groupby("k", as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            nearest_states=("nearest_state", "nunique"),
            mean_max_state_cosine=("max_state_cosine", "mean"),
            min_train_percentile=("train_percentile", "min"),
            below_train_p10_rate=("below_train_p10", "mean"),
        )
        .sort_values("k")
    )
    new_lines = "\n".join(
        f"| {int(row.k)} | {int(row.n_queries)} | {int(row.nearest_states)} | "
        f"{row.mean_max_state_cosine:.4f} | {row.min_train_percentile:.4f} | {row.below_train_p10_rate:.4f} |"
        for row in new_summary.itertuples(index=False)
    )
    body = f"""# Phase 3 State Semantic / Confidence Audit

This audit checks the mechanism proposed after the frozen-state transfer failure:

```text
query semantic similarity + local confidence
  -> decide whether state assignment is safe
  -> then assign to a RouteCode state/action
```

Inputs:

- Broad100 utility/output table: `{args.broad_outputs}`
- Broad100 probe features: `{args.broad_features}`
- New benchmark smoke outputs: `{args.new_outputs}`
- Embedding model: `{args.embedding_model}` loaded locally, with TF-IDF fallback.

## Test-Split Semantic Coherence

Higher cosine means queries in the same group are semantically closer.

| group method | queries | groups | pairwise cosine | query-centroid cosine | benchmark purity |
| --- | ---: | ---: | ---: | ---: | ---: |
{semantic_lines}

## Confidence Separability

`eta_squared` is the share of variance in a local-confidence feature explained by
utility state ID. Higher means that feature differs clearly by state.

| K | split | feature | mean | eta_squared |
| ---: | --- | --- | ---: | ---: |
{confidence_lines}

## New Benchmark Semantic Gate Summary

`train_percentile` compares each new query's nearest-state cosine to Broad100
train queries. Very low values are OOD warnings.

| K | new queries | nearest states used | mean nearest cosine | min train percentile | below train p10 rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
{new_lines}

## Files

- `table_state_semantic_coherence.csv`
- `table_state_confidence_separability.csv`
- `table_utility_state_cards.csv`
- `table_utility_state_assignments.csv`
- `table_new_benchmark_semantic_gate.csv`
"""
    (output_dir / "README.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
