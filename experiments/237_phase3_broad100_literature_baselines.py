from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
FRONTIER_MODELS = {"gpt-5.5", "gemini-3.5-flash", "gemini-3.5-flash-strong-solve"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cached Broad100 adapters for the selected literature baselines.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "literature_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(Path(config["inputs"]["broad100_outputs"]), float(config["method"]["lambda_cost"]))
    query_table = query_metadata(outputs)
    features = build_text_features(query_table, seed=int(args.seed))

    choices: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    for frame, summary, status in [
        run_routellm_pairwise(outputs, query_table, features, seed=int(args.seed)),
        run_llmrouter_knn(outputs, query_table, features),
        run_avengerspro_cluster(outputs, query_table, features, seed=int(args.seed)),
    ]:
        choices.append(frame)
        summaries.append(summary)
        statuses.append(status)

    choices_df = pd.concat(choices, ignore_index=True)
    choices_df.to_csv(out_dir / "table_broad100_literature_baseline_choices.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "table_broad100_literature_baselines.csv", index=False)
    pd.DataFrame(statuses).to_csv(out_dir / "table_broad100_literature_baseline_status.csv", index=False)
    write_figure(out_dir / "fig_broad100_literature_baselines.pdf", pd.DataFrame(summaries))
    write_memo(out_dir / "BROAD100_LITERATURE_BASELINES_MEMO.md", summaries, statuses, config)
    print(f"Wrote cached Broad100 literature baseline adapters to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_outputs(path: Path, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    outputs["benchmark"] = outputs["benchmark"].astype(str)
    outputs["quality_score"] = outputs["quality_score"].astype(float)
    outputs["normalized_remote_cost"] = outputs["normalized_remote_cost"].astype(float)
    outputs["cost_total_usd"] = outputs["cost_total_usd"].astype(float)
    outputs["latency_s"] = outputs["latency_s"].astype(float)
    outputs["utility"] = outputs["quality_score"] - lambda_cost * outputs["normalized_remote_cost"]
    return outputs


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    return (
        outputs[["query_id", "query_text", "split", "benchmark"]]
        .drop_duplicates("query_id")
        .sort_values(["split", "benchmark", "query_id"])
        .reset_index(drop=True)
    )


def build_text_features(query_table: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    train = query_table[query_table["split"].eq("train")].copy()
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2)
    x_train_sparse = vectorizer.fit_transform(train["query_text"].fillna("").astype(str))
    n_components = max(2, min(64, x_train_sparse.shape[1] - 1, x_train_sparse.shape[0] - 1))
    reducer = make_pipeline(TruncatedSVD(n_components=n_components, random_state=seed), Normalizer(copy=False))
    reducer.fit(x_train_sparse)
    matrices = {}
    ids = {}
    for split, frame in query_table.groupby("split", sort=False):
        x = reducer.transform(vectorizer.transform(frame["query_text"].fillna("").astype(str)))
        matrices[str(split)] = np.asarray(x, dtype=float)
        ids[str(split)] = frame["query_id"].astype(str).to_numpy()
    return {"vectorizer": vectorizer, "reducer": reducer, "x": matrices, "ids": ids}


def run_routellm_pairwise(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    features: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_means = outputs[outputs["split"].eq("train")].groupby("model_id")["utility"].mean().sort_values(ascending=False)
    local_models = [model for model in train_means.index if model not in FRONTIER_MODELS and model != "deterministic_math_tool"]
    strong_model = str(train_means.index[0])
    weak_model = str(local_models[0]) if local_models else str(train_means.index[-1])
    matrix = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
    train_ids = features["ids"]["train"]
    y = (matrix.reindex(train_ids)[strong_model] > matrix.reindex(train_ids)[weak_model]).astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        probabilities = {split: np.ones(len(features["ids"][split])) for split in ["val", "test"]}
        c_value = None
    else:
        c_value = 1.0
        clf = LogisticRegression(C=c_value, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed)
        clf.fit(features["x"]["train"], y)
        probabilities = {
            split: clf.predict_proba(features["x"][split])[:, 1]
            for split in ["val", "test"]
        }
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = select_pairwise_threshold(outputs, features["ids"]["val"], probabilities["val"], strong_model, weak_model, thresholds)
    val_choices = pairwise_choices(query_table, features["ids"]["val"], probabilities["val"], best_thr, strong_model, weak_model, "val")
    test_choices = pairwise_choices(query_table, features["ids"]["test"], probabilities["test"], best_thr, strong_model, weak_model, "test")
    choices = pd.concat([val_choices, test_choices], ignore_index=True)
    summary = summarize_choices(
        outputs,
        test_choices,
        method="routellm_pairwise_mf_adapter",
        family="routellm",
        notes=f"RouteLLM-style two-model adapter; strong={strong_model}; weak={weak_model}; threshold={best_thr:.2f}",
    )
    summary.update({"selected_threshold": float(best_thr), "strong_model": strong_model, "weak_model": weak_model, "classifier_c": c_value})
    status = baseline_status(
        "routellm_mf",
        "RouteLLM pairwise MF/logistic adapter",
        "broad100_adapter_executed",
        "https://github.com/lm-sys/routellm",
        "data/raw/external/routellm",
        "Official RouteLLM is two-model routing; this cached adapter routes between train-best strong and train-best local weak action.",
    )
    return choices, summary, status


def select_pairwise_threshold(
    outputs: pd.DataFrame,
    query_ids: np.ndarray,
    probabilities: np.ndarray,
    strong_model: str,
    weak_model: str,
    thresholds: np.ndarray,
) -> float:
    best_thr = float(thresholds[0])
    best_u = -np.inf
    lookup = outputs[outputs["split"].eq("val")].set_index(["query_id", "model_id"])
    for threshold in thresholds:
        selected = np.where(probabilities >= threshold, strong_model, weak_model)
        values = []
        for qid, model_id in zip(query_ids, selected, strict=False):
            key = (str(qid), str(model_id))
            if key in lookup.index:
                values.append(float(lookup.loc[key, "utility"]))
        mean_u = float(np.mean(values)) if values else -np.inf
        if mean_u > best_u:
            best_u = mean_u
            best_thr = float(threshold)
    return best_thr


def pairwise_choices(
    query_table: pd.DataFrame,
    query_ids: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    strong_model: str,
    weak_model: str,
    split: str,
) -> pd.DataFrame:
    selected = np.where(probabilities >= threshold, strong_model, weak_model)
    meta = query_table.set_index("query_id").reindex(query_ids)
    return pd.DataFrame(
        {
            "method": "routellm_pairwise_mf_adapter",
            "method_role": "literature_baseline",
            "baseline": "routellm_mf",
            "split": split,
            "query_id": query_ids.astype(str),
            "benchmark": meta["benchmark"].astype(str).to_numpy(),
            "model_id": selected.astype(str),
        }
    )


def run_llmrouter_knn(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    features: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_utility = outputs[outputs["split"].eq("train")].pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
    train_ids = features["ids"]["train"]
    train_utility = train_utility.reindex(train_ids)
    candidates = [3, 7, 15, 31, 63]
    val_rows = []
    for k in candidates:
        val_rows.append((k, evaluate_knn_split(outputs, query_table, features, train_utility, k=k, split="val")))
    best_k = max(val_rows, key=lambda item: item[1]["mean_utility"])[0]
    val_choices = knn_choices(outputs, query_table, features, train_utility, k=best_k, split="val")
    test_choices = knn_choices(outputs, query_table, features, train_utility, k=best_k, split="test")
    choices = pd.concat([val_choices, test_choices], ignore_index=True)
    summary = summarize_choices(
        outputs,
        test_choices,
        method=f"llmrouter_knn_fallback_k{best_k}",
        family="llmrouter_graphrouter_fallback",
        notes="LLMRouter fallback kNN adapter over cached Broad100 text features; used because GraphRouter has no direct cached-Broad100 adapter.",
    )
    summary["selected_k"] = int(best_k)
    status = baseline_status(
        "graphrouter",
        "LLMRouter kNN fallback for GraphRouter slot",
        "fallback_broad100_adapter_executed",
        "https://github.com/ulab-uiuc/LLMRouter",
        "data/raw/external/LLMRouter",
        "GraphRouter native cached-Broad100 adapter not available; LLMRouter kNN fallback was run per fallback policy.",
    )
    return choices, summary, status


def evaluate_knn_split(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    features: dict[str, Any],
    train_utility: pd.DataFrame,
    *,
    k: int,
    split: str,
) -> dict[str, Any]:
    choices = knn_choices(outputs, query_table, features, train_utility, k=k, split=split)
    return summarize_choices(outputs, choices, method=f"tmp_knn{k}", family="tmp", notes="")


def knn_choices(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    features: dict[str, Any],
    train_utility: pd.DataFrame,
    *,
    k: int,
    split: str,
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=min(k, len(features["x"]["train"])), metric="cosine")
    nn.fit(features["x"]["train"])
    _, indices = nn.kneighbors(features["x"][split])
    models = train_utility.columns.astype(str).tolist()
    selected = []
    for neighbors in indices:
        means = train_utility.iloc[neighbors][models].mean(axis=0)
        selected.append(str(means.sort_values(ascending=False).index[0]))
    query_ids = features["ids"][split].astype(str)
    meta = query_table.set_index("query_id").reindex(query_ids)
    return pd.DataFrame(
        {
            "method": f"llmrouter_knn_fallback_k{k}",
            "method_role": "literature_baseline",
            "baseline": "graphrouter",
            "split": split,
            "query_id": query_ids,
            "benchmark": meta["benchmark"].astype(str).to_numpy(),
            "model_id": selected,
        }
    )


def run_avengerspro_cluster(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    features: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    train_utility = outputs[outputs["split"].eq("train")].pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
    train_ids = features["ids"]["train"]
    train_utility = train_utility.reindex(train_ids)
    candidate_k = [4, 8, 16, 32]
    candidate_rows = []
    fitted: dict[int, KMeans] = {}
    for k in candidate_k:
        model = KMeans(n_clusters=min(k, len(train_ids)), random_state=seed, n_init=20)
        model.fit(features["x"]["train"])
        fitted[k] = model
        choices = avengers_choices(outputs, query_table, features, train_utility, model, split="val")
        candidate_rows.append((k, summarize_choices(outputs, choices, method=f"tmp_avengers{k}", family="tmp", notes="")))
    best_k = max(candidate_rows, key=lambda item: item[1]["mean_utility"])[0]
    model = fitted[best_k]
    val_choices = avengers_choices(outputs, query_table, features, train_utility, model, split="val")
    test_choices = avengers_choices(outputs, query_table, features, train_utility, model, split="test")
    choices = pd.concat([val_choices, test_choices], ignore_index=True)
    method = f"avengerspro_cluster_adapter_k{best_k}"
    choices["method"] = method
    summary = summarize_choices(
        outputs,
        test_choices.assign(method=method),
        method=method,
        family="avengerspro",
        notes="Avengers-Pro-style cluster router over cached Broad100 text features and train utility rankings.",
    )
    summary["selected_k"] = int(best_k)
    status = baseline_status(
        "avengerspro",
        "Avengers-Pro cluster adapter",
        "broad100_adapter_executed",
        "https://github.com/ZhangYiqun018/AvengersPro",
        "data/raw/external/LLMRouterBench/baselines/AvengersPro",
        "Cached no-API implementation of the released cluster-routing contract.",
    )
    return choices, summary, status


def avengers_choices(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    features: dict[str, Any],
    train_utility: pd.DataFrame,
    clusterer: KMeans,
    *,
    split: str,
) -> pd.DataFrame:
    train_labels = clusterer.predict(features["x"]["train"])
    cluster_best = {}
    for label in sorted(set(train_labels.tolist())):
        member = train_utility.iloc[np.where(train_labels == label)[0]]
        cluster_best[int(label)] = str(member.mean(axis=0).sort_values(ascending=False).index[0])
    test_labels = clusterer.predict(features["x"][split])
    selected = [cluster_best[int(label)] for label in test_labels]
    query_ids = features["ids"][split].astype(str)
    meta = query_table.set_index("query_id").reindex(query_ids)
    return pd.DataFrame(
        {
            "method": f"avengerspro_cluster_adapter_k{clusterer.n_clusters}",
            "method_role": "literature_baseline",
            "baseline": "avengerspro",
            "split": split,
            "query_id": query_ids,
            "benchmark": meta["benchmark"].astype(str).to_numpy(),
            "model_id": selected,
        }
    )


def summarize_choices(outputs: pd.DataFrame, choices: pd.DataFrame, *, method: str, family: str, notes: str) -> dict[str, Any]:
    split = str(choices["split"].iloc[0]) if "split" in choices else "test"
    target = outputs[outputs["split"].eq(split)].copy()
    oracle = target.sort_values(["query_id", "utility"], ascending=[True, False]).groupby("query_id").head(1)
    lookup = target.set_index(["query_id", "model_id"], drop=False)
    rows = []
    for row in choices.to_dict("records"):
        key = (str(row["query_id"]), str(row["model_id"]))
        if key in lookup.index:
            rows.append(lookup.loc[key])
    selected = pd.DataFrame(rows)
    frontier = selected["model_id"].isin(FRONTIER_MODELS) if not selected.empty else pd.Series(dtype=bool)
    mean_utility = float(selected["utility"].mean()) if not selected.empty else float("nan")
    oracle_utility = float(oracle["utility"].mean())
    return {
        "method": method,
        "baseline_family": family,
        "split": split,
        "n_queries": int(selected["query_id"].nunique()) if not selected.empty else 0,
        "mean_quality": float(selected["quality_score"].mean()) if not selected.empty else float("nan"),
        "mean_utility": mean_utility,
        "oracle_mean_utility": oracle_utility,
        "oracle_utility_ratio": mean_utility / max(oracle_utility, 1e-12),
        "utility_gap_to_oracle": oracle_utility - mean_utility,
        "frontier_call_rate": float(frontier.mean()) if len(frontier) else float("nan"),
        "remote_cost_per_1k_queries": float(selected["cost_total_usd"].sum(skipna=True) / max(len(selected), 1) * 1000.0)
        if not selected.empty
        else float("nan"),
        "selected_actions_json": json.dumps(selected["model_id"].astype(str).value_counts().sort_index().to_dict(), sort_keys=True)
        if not selected.empty
        else "{}",
        "notes": notes,
    }


def baseline_status(
    baseline: str,
    description: str,
    status: str,
    repo_url: str,
    repo_path: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "description": description,
        "status": status,
        "broad100_final_eval_included": True,
        "repo_url": repo_url,
        "repo_path": repo_path,
        "commit": git_commit(Path(repo_path)),
        "command": "python experiments/237_phase3_broad100_literature_baselines.py --config configs/probecode_final_eval.yaml",
        "notes": notes,
    }


def git_commit(path: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=False, text=True, capture_output=True)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def write_figure(path: Path, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot = summary.sort_values("mean_utility", ascending=True)
    ax.barh(plot["method"], plot["mean_utility"], color="#426b69")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Cached Broad100 Literature Baseline Adapters")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(path: Path, summaries: list[dict[str, Any]], statuses: list[dict[str, Any]], config: dict[str, Any]) -> None:
    lines = [
        "# Broad100 Literature Baseline Adapters",
        "",
        "This run evaluates the three selected literature-baseline slots on the final cached Broad100 matrix.",
        "No provider, vLLM, or external embedding calls are made.",
        "",
        "## Input",
        "",
        f"- Outcome matrix: `{config['inputs']['broad100_outputs']}`",
        "",
        "## Results",
        "",
    ]
    for row in sorted(summaries, key=lambda item: item["mean_utility"], reverse=True):
        lines.append(
            f"- `{row['method']}`: utility `{float(row['mean_utility']):.4f}`, "
            f"quality `{float(row['mean_quality']):.4f}`, frontier rate `{float(row['frontier_call_rate']):.4f}`"
        )
    lines.extend(["", "## Adapter Status", ""])
    for row in statuses:
        lines.append(f"- `{row['baseline']}`: `{row['status']}`; commit `{row['commit']}`; {row['notes']}")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "These are cached Broad100 adapters. RouteLLM is represented as a two-model pairwise adapter; "
            "GraphRouter uses the documented LLMRouter kNN fallback because a native cached-Broad100 GraphRouter adapter is not available.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

