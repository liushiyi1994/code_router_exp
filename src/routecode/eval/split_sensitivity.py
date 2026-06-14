from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from routecode.data.splits import split_by_query


def leave_one_group_split(
    outcomes: pd.DataFrame,
    group_column: str,
    holdout_value: str,
    seed: int,
    val_frac_of_train: float = 0.20,
) -> pd.DataFrame:
    if group_column not in outcomes.columns:
        raise ValueError(f"Missing group column: {group_column}")
    result = outcomes.copy()
    holdout_mask = result[group_column].astype(str) == str(holdout_value)
    result.loc[holdout_mask, "split"] = "test"
    train_val = result.loc[~holdout_mask].drop(columns=["split"], errors="ignore")
    if train_val["query_id"].nunique() < 2:
        raise ValueError(f"Not enough non-holdout queries for {group_column}={holdout_value}")
    train_val_split = split_by_query(
        train_val,
        train_frac=1.0 - val_frac_of_train,
        val_frac=val_frac_of_train,
        test_frac=0.0,
        seed=seed,
    )
    train_val_split.loc[train_val_split["split"] == "test", "split"] = "val"
    result.loc[~holdout_mask, "split"] = train_val_split["split"].to_numpy()
    return result


def domain_homogeneous_split(
    outcomes: pd.DataFrame,
    domain_value: str,
    seed: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> pd.DataFrame:
    subset = outcomes[outcomes["domain"].astype(str) == str(domain_value)].copy()
    if subset["query_id"].nunique() < 3:
        raise ValueError(f"Not enough queries for homogeneous domain split: {domain_value}")
    return split_by_query(subset, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed)


def cluster_heldout_split(
    outcomes: pd.DataFrame,
    embeddings: pd.DataFrame,
    n_clusters: int,
    heldout_cluster: int,
    seed: int,
    val_frac_of_train: float = 0.20,
) -> tuple[pd.DataFrame, int]:
    query_ids = outcomes["query_id"].drop_duplicates().tolist()
    aligned = embeddings.loc[query_ids]
    effective_clusters = max(2, min(int(n_clusters), len(aligned)))
    kmeans = KMeans(n_clusters=effective_clusters, random_state=seed, n_init=10)
    labels = pd.Series(kmeans.fit_predict(aligned.to_numpy()), index=aligned.index, name="cluster")
    heldout = int(heldout_cluster) % effective_clusters
    query_to_cluster = labels.to_dict()
    result = outcomes.copy()
    result["_cluster"] = result["query_id"].map(query_to_cluster)
    result.loc[result["_cluster"] == heldout, "split"] = "test"
    train_val = result[result["_cluster"] != heldout].drop(columns=["split"], errors="ignore")
    if train_val["query_id"].nunique() < 2:
        raise ValueError(f"Cluster holdout leaves too few train/val queries: {heldout}")
    train_val_split = split_by_query(
        train_val.drop(columns=["_cluster"]),
        train_frac=1.0 - val_frac_of_train,
        val_frac=val_frac_of_train,
        test_frac=0.0,
        seed=seed,
    )
    train_val_split.loc[train_val_split["split"] == "test", "split"] = "val"
    result.loc[result["_cluster"] != heldout, "split"] = train_val_split["split"].to_numpy()
    return result.drop(columns=["_cluster"]), heldout


def ranking_correlation(reference: pd.DataFrame, comparison: pd.DataFrame) -> float:
    merged = reference[["method", "mean_utility"]].merge(
        comparison[["method", "mean_utility"]],
        on="method",
        suffixes=("_reference", "_comparison"),
    )
    if len(merged) < 2:
        return float("nan")
    ref_rank = merged["mean_utility_reference"].rank(ascending=False, method="average")
    cmp_rank = merged["mean_utility_comparison"].rank(ascending=False, method="average")
    return float(ref_rank.corr(cmp_rank, method="pearson"))


def compression_rate_to_reach(
    table: pd.DataFrame,
    threshold: float = 0.80,
    method_contains: str = "routecode_predicted",
) -> float:
    candidates = table[
        table["method"].astype(str).str.contains(method_contains)
        & table["K"].notna()
        & (table["recovered_gap_vs_learned"] >= threshold)
    ].sort_values("rate_log2K")
    if candidates.empty:
        return float("nan")
    return float(candidates.iloc[0]["rate_log2K"])
