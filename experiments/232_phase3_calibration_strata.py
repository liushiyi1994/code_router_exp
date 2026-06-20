from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
EPSILON_STABLE = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether RouteCode states are useful calibration strata.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "calibration_strata"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(Path(config["inputs"]["broad100_outputs"]), float(config["method"]["lambda_cost"]))
    query_table = query_metadata(outputs)
    groups = build_groups(outputs, query_table, config, k=int(args.k), seed=int(args.seed))

    variance, group_detail = state_variance(outputs, groups)
    estimation = estimation_error(outputs, groups)
    best_model = best_model_accuracy(outputs, groups)

    variance.to_csv(out_dir / "table_state_variance.csv", index=False)
    estimation.to_csv(out_dir / "table_state_estimation_error.csv", index=False)
    best_model.to_csv(out_dir / "table_state_best_model_accuracy.csv", index=False)
    group_detail.to_csv(out_dir / "table_state_group_details.csv", index=False)
    write_variance_figure(out_dir / "fig_state_variance.pdf", variance)
    write_memo(out_dir / "CALIBRATION_STRATA_MEMO.md", variance, estimation, best_model, config)
    print(f"Wrote calibration-strata experiment to {out_dir}")


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
    outputs["utility"] = outputs["quality_score"] - lambda_cost * outputs["normalized_remote_cost"].astype(float)
    return outputs


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    cols = ["query_id", "query_text", "split", "benchmark", "domain"]
    return outputs[cols].drop_duplicates("query_id").sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def build_groups(outputs: pd.DataFrame, query_table: pd.DataFrame, config: dict[str, Any], *, k: int, seed: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    frames.append(random_groups(query_table, k=k, seed=seed))
    frames.append(label_groups(query_table, "benchmark_label", "benchmark"))
    frames.append(text_cluster_groups(query_table, k=k, seed=seed))
    frames.append(utility_cluster_groups(outputs, query_table, k=k, seed=seed))
    frames.append(routecode_groups(config, query_table, calibration_aware=False))
    frames.append(routecode_groups(config, query_table, calibration_aware=True))
    groups = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    groups["group_id"] = groups["group_id"].astype(str)
    return groups


def random_groups(query_table: pd.DataFrame, *, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = query_table[["query_id", "split", "benchmark"]].copy()
    work["group_method"] = f"random_k{k}"
    work["group_id"] = [f"r{int(value):02d}" for value in rng.integers(0, k, size=len(work))]
    work["deployability"] = "random_baseline"
    return work


def label_groups(query_table: pd.DataFrame, method: str, label_col: str) -> pd.DataFrame:
    work = query_table[["query_id", "split", "benchmark"]].copy()
    work["group_method"] = method
    work["group_id"] = query_table[label_col].fillna("unknown").astype(str)
    work["deployability"] = "diagnostic_label"
    return work


def text_cluster_groups(query_table: pd.DataFrame, *, k: int, seed: int) -> pd.DataFrame:
    train = query_table[query_table["split"].eq("train")].copy()
    if train.empty:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train["query_text"].fillna("").astype(str))
    n_components = max(2, min(32, x_train.shape[1] - 1, x_train.shape[0] - 1))
    if n_components < 2:
        return pd.DataFrame()
    pipe = make_pipeline(TruncatedSVD(n_components=n_components, random_state=seed), Normalizer(copy=False))
    z_train = pipe.fit_transform(x_train)
    model = KMeans(n_clusters=min(k, len(train)), random_state=seed, n_init=20)
    model.fit(z_train)
    rows = []
    for split, frame in query_table.groupby("split", sort=False):
        x = vectorizer.transform(frame["query_text"].fillna("").astype(str))
        z = pipe.transform(x)
        labels = model.predict(z)
        rows.append(
            pd.DataFrame(
                {
                    "query_id": frame["query_id"].astype(str).to_numpy(),
                    "split": split,
                    "benchmark": frame["benchmark"].astype(str).to_numpy(),
                    "group_method": f"text_cluster_k{k}",
                    "group_id": [f"t{int(label):02d}" for label in labels],
                    "deployability": "train_fit_text_cluster",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def utility_cluster_groups(outputs: pd.DataFrame, query_table: pd.DataFrame, *, k: int, seed: int) -> pd.DataFrame:
    matrix = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
    matrix = matrix.dropna(axis=0)
    train_ids = query_table[query_table["split"].eq("train")]["query_id"].astype(str)
    train_matrix = matrix.reindex(train_ids).dropna(axis=0)
    if len(train_matrix) < k:
        return pd.DataFrame()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_matrix.to_numpy(dtype=float))
    model = KMeans(n_clusters=k, random_state=seed, n_init=30)
    model.fit(x_train)
    rows = []
    for split, frame in query_table.groupby("split", sort=False):
        split_matrix = matrix.reindex(frame["query_id"].astype(str)).dropna(axis=0)
        if split_matrix.empty:
            continue
        labels = model.predict(scaler.transform(split_matrix.to_numpy(dtype=float)))
        meta = frame.set_index("query_id").reindex(split_matrix.index)
        rows.append(
            pd.DataFrame(
                {
                    "query_id": split_matrix.index.astype(str),
                    "split": split,
                    "benchmark": meta["benchmark"].astype(str).to_numpy(),
                    "group_method": f"utility_cluster_k{k}_diagnostic",
                    "group_id": [f"u{int(label):02d}" for label in labels],
                    "deployability": "diagnostic_uses_hidden_utility_vectors",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def routecode_groups(config: dict[str, Any], query_table: pd.DataFrame, *, calibration_aware: bool) -> pd.DataFrame:
    path = Path(config["inputs"]["broad100_learned_verifiability_assignments"])
    if not path.exists():
        return pd.DataFrame()
    assignments = pd.read_csv(path)
    assignments["query_id"] = assignments["query_id"].astype(str)
    method = str(config["method"]["compact_state_method"])
    assignments = assignments[assignments["method"].astype(str).eq(method)].copy()
    if assignments.empty:
        return pd.DataFrame()
    cols = ["query_id", "split", "benchmark", "probe_state", "need_large", "pred_tool_available"]
    assignments = assignments[cols].drop_duplicates("query_id").copy()
    if calibration_aware:
        assignments["group_method"] = f"calibration_aware_routecode_state_k{int(assignments['probe_state'].nunique())}"
        assignments["group_id"] = (
            "z"
            + assignments["probe_state"].astype(int).astype(str).str.zfill(2)
            + "_large"
            + assignments["need_large"].astype(int).astype(str)
            + "_tool"
            + assignments["pred_tool_available"].astype(int).astype(str)
        )
        assignments["deployability"] = "learned_probe_state_plus_observable_bits"
    else:
        assignments["group_method"] = f"routecode_state_k{int(assignments['probe_state'].nunique())}"
        assignments["group_id"] = "z" + assignments["probe_state"].astype(int).astype(str).str.zfill(2)
        assignments["deployability"] = "learned_probe_state"
    valid_ids = set(query_table["query_id"].astype(str))
    assignments = assignments[assignments["query_id"].isin(valid_ids)]
    return assignments[["query_id", "split", "benchmark", "group_method", "group_id", "deployability"]].copy()


def state_variance(outputs: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = outputs.merge(groups, on=["query_id", "split", "benchmark"], how="inner")
    detail = (
        work.groupby(["group_method", "deployability", "split", "group_id", "model_id"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            mean_utility=("utility", "mean"),
            utility_variance=("utility", "var"),
            quality_variance=("quality_score", "var"),
        )
        .fillna({"utility_variance": 0.0, "quality_variance": 0.0})
    )
    detail["ci_width_utility_95"] = 2.0 * 1.96 * np.sqrt(detail["utility_variance"].clip(lower=0.0) / detail["n_queries"].clip(lower=1))
    detail["samples_needed_for_0p03"] = np.ceil((1.96 * np.sqrt(detail["utility_variance"].clip(lower=0.0)) / EPSILON_STABLE) ** 2)

    rows = []
    for keys, frame in detail.groupby(["group_method", "deployability", "split"]):
        method, deployability, split = keys
        weights = frame["n_queries"].to_numpy(dtype=float)
        rows.append(
            {
                "group_method": method,
                "deployability": deployability,
                "split": split,
                "n_groups": int(frame["group_id"].nunique()),
                "n_group_model_cells": int(len(frame)),
                "weighted_utility_variance": weighted_mean(frame["utility_variance"], weights),
                "weighted_quality_variance": weighted_mean(frame["quality_variance"], weights),
                "mean_ci_width_utility_95": float(frame["ci_width_utility_95"].mean()),
                "traffic_weighted_ci_width_utility_95": weighted_mean(frame["ci_width_utility_95"], weights),
                "median_samples_needed_for_0p03": float(frame["samples_needed_for_0p03"].median()),
                "traffic_weighted_samples_needed_for_0p03": weighted_mean(frame["samples_needed_for_0p03"], weights),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "weighted_utility_variance"]), detail


def estimation_error(outputs: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    work = outputs.merge(groups, on=["query_id", "split", "benchmark"], how="inner")
    means = (
        work.groupby(["group_method", "deployability", "split", "group_id", "model_id"], as_index=False)
        .agg(n_queries=("query_id", "nunique"), mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"))
    )
    train = means[means["split"].eq("train")].rename(
        columns={"mean_utility": "train_mean_utility", "mean_quality": "train_mean_quality", "n_queries": "train_n_queries"}
    )
    test = means[means["split"].eq("test")].rename(
        columns={"mean_utility": "test_mean_utility", "mean_quality": "test_mean_quality", "n_queries": "test_n_queries"}
    )
    joined = test.merge(
        train[["group_method", "group_id", "model_id", "train_mean_utility", "train_mean_quality", "train_n_queries"]],
        on=["group_method", "group_id", "model_id"],
        how="left",
    ).dropna(subset=["train_mean_utility"])
    joined["abs_utility_estimation_error"] = (joined["test_mean_utility"] - joined["train_mean_utility"]).abs()
    joined["abs_quality_estimation_error"] = (joined["test_mean_quality"] - joined["train_mean_quality"]).abs()
    rows = []
    for keys, frame in joined.groupby(["group_method", "deployability"]):
        method, deployability = keys
        weights = frame["test_n_queries"].to_numpy(dtype=float)
        rows.append(
            {
                "group_method": method,
                "deployability": deployability,
                "n_cells": int(len(frame)),
                "weighted_abs_utility_estimation_error": weighted_mean(frame["abs_utility_estimation_error"], weights),
                "weighted_abs_quality_estimation_error": weighted_mean(frame["abs_quality_estimation_error"], weights),
                "median_abs_utility_estimation_error": float(frame["abs_utility_estimation_error"].median()),
                "mean_train_queries_per_cell": float(frame["train_n_queries"].mean()),
                "mean_test_queries_per_cell": float(frame["test_n_queries"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_abs_utility_estimation_error")


def best_model_accuracy(outputs: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    work = outputs.merge(groups, on=["query_id", "split", "benchmark"], how="inner")
    means = (
        work.groupby(["group_method", "deployability", "split", "group_id", "model_id"], as_index=False)
        .agg(n_queries=("query_id", "nunique"), mean_utility=("utility", "mean"))
    )
    idx = means.groupby(["group_method", "deployability", "split", "group_id"])["mean_utility"].idxmax()
    best = means.loc[idx].rename(columns={"model_id": "best_model", "mean_utility": "best_mean_utility"})
    train = best[best["split"].eq("train")].rename(columns={"best_model": "train_best_model"})
    test = best[best["split"].eq("test")].rename(columns={"best_model": "test_best_model", "n_queries": "test_n_queries"})
    joined = test.merge(
        train[["group_method", "group_id", "train_best_model"]],
        on=["group_method", "group_id"],
        how="left",
    ).dropna(subset=["train_best_model"])
    joined["best_model_match"] = joined["train_best_model"].astype(str).eq(joined["test_best_model"].astype(str))
    rows = []
    for keys, frame in joined.groupby(["group_method", "deployability"]):
        method, deployability = keys
        weights = frame["test_n_queries"].to_numpy(dtype=float)
        rows.append(
            {
                "group_method": method,
                "deployability": deployability,
                "n_test_groups": int(len(frame)),
                "best_model_identification_accuracy": float(frame["best_model_match"].mean()),
                "traffic_weighted_best_model_identification_accuracy": weighted_mean(frame["best_model_match"].astype(float), weights),
            }
        )
    return pd.DataFrame(rows).sort_values("traffic_weighted_best_model_identification_accuracy", ascending=False)


def weighted_mean(values: pd.Series | np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return float(np.nan)
    return float(np.average(values, weights=weights))


def write_variance_figure(path: Path, variance: pd.DataFrame) -> None:
    plot = variance[variance["split"].eq("test")].sort_values("weighted_utility_variance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot["group_method"], plot["weighted_utility_variance"], color="#426b69")
    ax.set_xlabel("Traffic-weighted within-state utility variance")
    ax.set_title("Calibration Strata Stability On Held-Out Test")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(path: Path, variance: pd.DataFrame, estimation: pd.DataFrame, best_model: pd.DataFrame, config: dict[str, Any]) -> None:
    test_var = variance[variance["split"].eq("test")].sort_values("weighted_utility_variance")
    lines = [
        "# Calibration Strata Experiment",
        "",
        "This no-call experiment tests whether learned states are useful groups for estimating model utility.",
        "",
        "## Inputs",
        "",
        f"- Outcome matrix: `{config['inputs']['broad100_outputs']}`",
        f"- RouteCode assignments: `{config['inputs']['broad100_learned_verifiability_assignments']}`",
        f"- Compact state method: `{config['method']['compact_state_method']}`",
        "",
        "## Test Within-State Utility Variance",
        "",
    ]
    for row in test_var.to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: variance `{float(row['weighted_utility_variance']):.4f}`, "
            f"groups `{int(row['n_groups'])}`"
        )
    lines.extend(["", "## New-Model Estimation Error Proxy", ""])
    for row in estimation.head(8).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: weighted abs utility error "
            f"`{float(row['weighted_abs_utility_estimation_error']):.4f}`"
        )
    lines.extend(["", "## Best-Model Identification", ""])
    for row in best_model.head(8).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: traffic-weighted match "
            f"`{float(row['traffic_weighted_best_model_identification_accuracy']):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "`utility_cluster_*` is diagnostic because assigning validation/test queries to a utility cluster uses hidden outcome vectors. "
            "The deployable learned-state rows are the RouteCode/probe-state rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

