from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
FRONTIER_MODELS = {"gpt-5.5", "gemini-3.5-flash", "gemini-3.5-flash-strong-solve"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate new-model onboarding from cached Broad100 outcomes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--budgets", type=int, nargs="*", default=[20, 40, 80, 160, 320, 640])
    parser.add_argument("--seeds", type=int, nargs="*", default=[17, 18, 19])
    parser.add_argument("--k", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "new_model_onboarding"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(Path(config["inputs"]["broad100_outputs"]), float(config["method"]["lambda_cost"]))
    query_table = query_metadata(outputs)
    groups = build_groups(outputs, query_table, config, k=int(args.k), seed=int(config.get("seed", 17)))
    features = load_probe_features(Path(config["inputs"]["broad100_probe_features"]), query_table)

    rows = run_onboarding(outputs, groups, features, budgets=args.budgets, seeds=args.seeds)
    table = pd.DataFrame(rows).sort_values(["heldout_model", "budget", "mean_utility"], ascending=[True, True, False])
    by_type = summarize_by_model_type(table)
    acquisition = summarize_acquisition(table)

    table.to_csv(out_dir / "table_new_model_onboarding.csv", index=False)
    by_type.to_csv(out_dir / "table_onboarding_by_model_type.csv", index=False)
    acquisition.to_csv(out_dir / "table_acquisition_ablation.csv", index=False)
    write_figures(out_dir, table)
    write_memo(out_dir / "NEW_MODEL_ONBOARDING_MEMO.md", table, by_type, config)
    print(f"Wrote simulated new-model onboarding experiment to {out_dir}")


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
        outputs[["query_id", "query_text", "split", "benchmark", "domain"]]
        .drop_duplicates("query_id")
        .sort_values(["split", "benchmark", "query_id"])
        .reset_index(drop=True)
    )


def build_groups(outputs: pd.DataFrame, query_table: pd.DataFrame, config: dict[str, Any], *, k: int, seed: int) -> dict[str, pd.DataFrame]:
    return {
        "dataset_stratified_calibration": label_groups(query_table, "benchmark"),
        "embedding_cluster_calibration": text_cluster_groups(query_table, k=k, seed=seed),
        "route_state_calibration": routecode_groups(config, query_table, calibration_aware=False),
        "calibration_aware_route_state": routecode_groups(config, query_table, calibration_aware=True),
    }


def label_groups(query_table: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = query_table[["query_id", "split", "benchmark"]].copy()
    work["group_id"] = query_table[label_col].fillna("unknown").astype(str)
    return work


def text_cluster_groups(query_table: pd.DataFrame, *, k: int, seed: int) -> pd.DataFrame:
    train = query_table[query_table["split"].eq("train")].copy()
    vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train["query_text"].fillna("").astype(str))
    n_components = max(2, min(32, x_train.shape[1] - 1, x_train.shape[0] - 1))
    pipe = make_pipeline(TruncatedSVD(n_components=n_components, random_state=seed), Normalizer(copy=False))
    z_train = pipe.fit_transform(x_train)
    clusterer = KMeans(n_clusters=min(k, len(train)), random_state=seed, n_init=20)
    clusterer.fit(z_train)
    rows = []
    for split, frame in query_table.groupby("split", sort=False):
        z = pipe.transform(vectorizer.transform(frame["query_text"].fillna("").astype(str)))
        labels = clusterer.predict(z)
        rows.append(
            pd.DataFrame(
                {
                    "query_id": frame["query_id"].astype(str).to_numpy(),
                    "split": split,
                    "benchmark": frame["benchmark"].astype(str).to_numpy(),
                    "group_id": [f"t{int(label):02d}" for label in labels],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def routecode_groups(config: dict[str, Any], query_table: pd.DataFrame, *, calibration_aware: bool) -> pd.DataFrame:
    path = Path(config["inputs"]["broad100_learned_verifiability_assignments"])
    assignments = pd.read_csv(path)
    assignments["query_id"] = assignments["query_id"].astype(str)
    assignments = assignments[assignments["method"].astype(str).eq(str(config["method"]["compact_state_method"]))].copy()
    assignments = assignments.drop_duplicates("query_id")
    if calibration_aware:
        assignments["group_id"] = (
            "z"
            + assignments["probe_state"].astype(int).astype(str).str.zfill(2)
            + "_large"
            + assignments["need_large"].astype(int).astype(str)
            + "_tool"
            + assignments["pred_tool_available"].astype(int).astype(str)
        )
    else:
        assignments["group_id"] = "z" + assignments["probe_state"].astype(int).astype(str).str.zfill(2)
    valid = query_table[["query_id"]]
    return (
        assignments[["query_id", "split", "benchmark", "group_id"]]
        .merge(valid, on="query_id", how="inner")
        .drop_duplicates("query_id")
    )


def load_probe_features(path: Path, query_table: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(path)
    features["query_id"] = features["query_id"].astype(str)
    id_cols = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}
    numeric_cols = [col for col in features.columns if col not in id_cols and pd.api.types.is_numeric_dtype(features[col])]
    work = features[["query_id", "split", "benchmark", *numeric_cols]].copy()
    work[numeric_cols] = work[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return query_table[["query_id", "split", "benchmark"]].merge(work, on=["query_id", "split", "benchmark"], how="left").fillna(0.0)


def run_onboarding(
    outputs: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    *,
    budgets: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    models = sorted(outputs["model_id"].unique())
    rows: list[dict[str, Any]] = []
    route_groups = groups["route_state_calibration"]
    for heldout_model in models:
        train_ids_all = set(
            outputs[(outputs["split"].eq("train")) & (outputs["model_id"].eq(heldout_model))]["query_id"].astype(str)
        )
        full_refs = {
            "route_state": evaluate_group_calibration(
                outputs,
                route_groups,
                heldout_model=heldout_model,
                sampled_query_ids=train_ids_all,
                method="full_route_state_calibration",
                budget=-1,
                seed=-1,
                acquisition="all_train",
                training_time_s=0.0,
            ),
            "dataset": evaluate_group_calibration(
                outputs,
                groups["dataset_stratified_calibration"],
                heldout_model=heldout_model,
                sampled_query_ids=train_ids_all,
                method="full_dataset_stratified_calibration",
                budget=-1,
                seed=-1,
                acquisition="all_train",
                training_time_s=0.0,
            ),
            "embedding": evaluate_group_calibration(
                outputs,
                groups["embedding_cluster_calibration"],
                heldout_model=heldout_model,
                sampled_query_ids=train_ids_all,
                method="full_embedding_cluster_calibration",
                budget=-1,
                seed=-1,
                acquisition="all_train",
                training_time_s=0.0,
            ),
            "calibration_aware": evaluate_group_calibration(
                outputs,
                groups["calibration_aware_route_state"],
                heldout_model=heldout_model,
                sampled_query_ids=train_ids_all,
                method="full_calibration_aware_route_state",
                budget=-1,
                seed=-1,
                acquisition="all_train",
                training_time_s=0.0,
            ),
        }
        direct_full = evaluate_direct_regressor(
            outputs,
            features,
            heldout_model=heldout_model,
            budget=len(train_ids_all),
            seed=-1,
            method="full_direct_probe_regressor_retrain",
        )
        direct_full["budget"] = -1
        direct_full["n_new_model_evals"] = len(train_ids_all)
        full_refs["direct"] = direct_full
        for full_reference in full_refs.values():
            full_utility = full_reference["mean_utility"]
            rows.append({**full_reference, "full_calibration_mean_utility": full_utility, "regret_to_full_calibration": 0.0})
        for budget in budgets:
            for seed in seeds:
                for method_name, group_frame, acquisition in [
                    ("random_query_route_state_calibration", route_groups, "random_query"),
                    ("dataset_stratified_calibration", groups["dataset_stratified_calibration"], "uniform_group"),
                    ("embedding_cluster_calibration", groups["embedding_cluster_calibration"], "uniform_group"),
                    ("uniform_route_state_calibration", route_groups, "uniform_group"),
                    ("active_route_state_calibration", route_groups, "active_margin"),
                    ("calibration_aware_route_state", groups["calibration_aware_route_state"], "active_margin"),
                ]:
                    sampled = sample_calibration_queries(outputs, group_frame, heldout_model, budget=budget, seed=seed, acquisition=acquisition)
                    row = evaluate_group_calibration(
                        outputs,
                        group_frame,
                        heldout_model=heldout_model,
                        sampled_query_ids=sampled,
                        method=method_name,
                        budget=budget,
                        seed=seed,
                        acquisition=acquisition,
                        training_time_s=0.0,
                    )
                    ref_key = reference_key(method_name)
                    full_utility = full_refs[ref_key]["mean_utility"]
                    row["full_calibration_mean_utility"] = full_utility
                    row["regret_to_full_calibration"] = full_utility - row["mean_utility"]
                    rows.append(row)
                direct = evaluate_direct_regressor(outputs, features, heldout_model=heldout_model, budget=budget, seed=seed)
                full_utility = full_refs["direct"]["mean_utility"]
                direct["full_calibration_mean_utility"] = full_utility
                direct["regret_to_full_calibration"] = full_utility - direct["mean_utility"]
                rows.append(direct)
    return rows


def reference_key(method_name: str) -> str:
    if method_name == "dataset_stratified_calibration":
        return "dataset"
    if method_name == "embedding_cluster_calibration":
        return "embedding"
    if method_name == "calibration_aware_route_state":
        return "calibration_aware"
    return "route_state"


def sample_calibration_queries(
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    heldout_model: str,
    *,
    budget: int,
    seed: int,
    acquisition: str,
) -> set[str]:
    train_ids = outputs[(outputs["split"].eq("train")) & (outputs["model_id"].eq(heldout_model))]["query_id"].astype(str).unique()
    train_ids = np.asarray(sorted(train_ids), dtype=object)
    if budget >= len(train_ids):
        return set(map(str, train_ids))
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    group_train = groups[groups["split"].eq("train") & groups["query_id"].isin(train_ids)].copy()
    if acquisition == "random_query":
        return set(map(str, rng.choice(train_ids, size=max(1, budget), replace=False)))
    if acquisition == "active_margin":
        weights = active_group_weights(outputs, group_train, heldout_model)
        group_train = group_train.merge(weights, on="group_id", how="left")
        query_weights = group_train["group_weight"].fillna(1.0).to_numpy(dtype=float)
        query_weights = query_weights / np.maximum(group_train.groupby("group_id")["query_id"].transform("count").to_numpy(dtype=float), 1.0)
        query_weights = query_weights / query_weights.sum()
        return set(map(str, rng.choice(group_train["query_id"].to_numpy(dtype=object), size=max(1, budget), replace=False, p=query_weights)))
    return sample_uniform_by_group(group_train, budget=budget, rng=rng)


def active_group_weights(outputs: pd.DataFrame, group_train: pd.DataFrame, heldout_model: str) -> pd.DataFrame:
    remaining = outputs[outputs["split"].eq("train") & ~outputs["model_id"].eq(heldout_model)].merge(
        group_train[["query_id", "group_id"]], on="query_id", how="inner"
    )
    means = remaining.groupby(["group_id", "model_id"], as_index=False)["utility"].mean()
    ranks = means.sort_values(["group_id", "utility"], ascending=[True, False]).groupby("group_id").head(2)
    rows = []
    for group_id, frame in ranks.groupby("group_id"):
        values = frame["utility"].to_numpy(dtype=float)
        margin = float(values[0] - values[1]) if len(values) > 1 else 1.0
        traffic = int(group_train[group_train["group_id"].eq(group_id)]["query_id"].nunique())
        rows.append({"group_id": group_id, "group_weight": traffic / max(margin + 0.02, 0.02), "existing_margin": margin})
    return pd.DataFrame(rows)


def sample_uniform_by_group(group_train: pd.DataFrame, *, budget: int, rng: np.random.Generator) -> set[str]:
    selected: list[str] = []
    groups = [frame.copy() for _, frame in group_train.groupby("group_id")]
    rng.shuffle(groups)
    remaining = max(1, budget)
    while remaining > 0 and groups:
        progressed = False
        for frame in groups:
            available = [qid for qid in frame["query_id"].astype(str).tolist() if qid not in selected]
            if not available:
                continue
            selected.append(str(rng.choice(available)))
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return set(selected)


def evaluate_group_calibration(
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    heldout_model: str,
    sampled_query_ids: set[str],
    method: str,
    budget: int,
    seed: int,
    acquisition: str,
    training_time_s: float,
) -> dict[str, Any]:
    train_groups = groups[groups["split"].eq("train")].copy()
    test_groups = groups[groups["split"].eq("test")].copy()
    train = outputs[outputs["split"].eq("train")].merge(train_groups[["query_id", "group_id"]], on="query_id", how="inner")
    test = outputs[outputs["split"].eq("test")].merge(test_groups[["query_id", "group_id"]], on="query_id", how="inner")

    remaining = train[~train["model_id"].eq(heldout_model)].copy()
    existing_means = remaining.groupby(["group_id", "model_id"], as_index=False)["utility"].mean()
    idx = existing_means.groupby("group_id")["utility"].idxmax()
    existing_best = existing_means.loc[idx].rename(columns={"model_id": "existing_best_model", "utility": "existing_best_utility"})

    heldout_train = train[train["model_id"].eq(heldout_model) & train["query_id"].isin(sampled_query_ids)].copy()
    heldout_group = heldout_train.groupby("group_id", as_index=False)["utility"].mean().rename(columns={"utility": "heldout_estimated_utility"})
    global_estimate = float(heldout_train["utility"].mean()) if not heldout_train.empty else -np.inf
    action_table = existing_best.merge(heldout_group, on="group_id", how="left")
    action_table["heldout_estimated_utility"] = action_table["heldout_estimated_utility"].fillna(global_estimate)
    action_table["selected_model"] = np.where(
        action_table["heldout_estimated_utility"] > action_table["existing_best_utility"],
        heldout_model,
        action_table["existing_best_model"],
    )

    selected = select_test_rows(test, test_groups, action_table, heldout_model)
    return summarize_selection(
        selected,
        outputs,
        heldout_model=heldout_model,
        method=method,
        budget=budget,
        seed=seed,
        acquisition=acquisition,
        n_new_model_evals=len(sampled_query_ids),
        training_time_s=training_time_s,
        group_count=int(groups["group_id"].nunique()),
    )


def select_test_rows(test: pd.DataFrame, test_groups: pd.DataFrame, action_table: pd.DataFrame, heldout_model: str) -> pd.DataFrame:
    selected_models = test_groups[["query_id", "group_id"]].merge(action_table[["group_id", "selected_model"]], on="group_id", how="left")
    fallback = action_table["selected_model"].mode()
    selected_models["selected_model"] = selected_models["selected_model"].fillna(str(fallback.iloc[0]) if not fallback.empty else heldout_model)
    lookup = test.set_index(["query_id", "model_id"], drop=False)
    rows = []
    for row in selected_models.to_dict("records"):
        key = (str(row["query_id"]), str(row["selected_model"]))
        if key in lookup.index:
            rows.append(lookup.loc[key])
    return pd.DataFrame(rows).reset_index(drop=True)


def evaluate_direct_regressor(
    outputs: pd.DataFrame,
    features: pd.DataFrame,
    *,
    heldout_model: str,
    budget: int,
    seed: int,
    method: str = "direct_probe_regressor_retrain",
) -> dict[str, Any]:
    start = time.perf_counter()
    train_features = features[features["split"].eq("train")].drop_duplicates("query_id").set_index("query_id")
    test_features = features[features["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    feature_cols = [col for col in train_features.columns if col not in {"split", "benchmark"} and pd.api.types.is_numeric_dtype(train_features[col])]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features[feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test_features[feature_cols].to_numpy(dtype=float))
    train_ids = train_features.index.astype(str).to_numpy()
    test_ids = test_features.index.astype(str).to_numpy()
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    sampled = set(map(str, rng.choice(train_ids, size=min(max(1, budget), len(train_ids)), replace=False)))

    utility_lookup = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
    models = sorted(outputs["model_id"].unique())
    pred = pd.DataFrame(index=test_ids)
    for model_id in models:
        y = utility_lookup.reindex(train_ids)[model_id].to_numpy(dtype=float)
        if model_id == heldout_model:
            mask = np.asarray([qid in sampled for qid in train_ids], dtype=bool)
        else:
            mask = np.isfinite(y)
        mask = mask & np.isfinite(y)
        y_fit = y[mask]
        if len(y_fit) < 2 or np.nanstd(y_fit) < 1e-12:
            pred[model_id] = float(np.nanmean(y_fit)) if len(y_fit) else -np.inf
            continue
        model = Ridge(alpha=10.0)
        model.fit(x_train[mask], y_fit)
        pred[model_id] = model.predict(x_test)
    selected_model = pred.idxmax(axis=1)
    test = outputs[outputs["split"].eq("test")].set_index(["query_id", "model_id"], drop=False)
    rows = []
    for qid, model_id in selected_model.items():
        key = (str(qid), str(model_id))
        if key in test.index:
            rows.append(test.loc[key])
    selected = pd.DataFrame(rows).reset_index(drop=True)
    return summarize_selection(
        selected,
        outputs,
        heldout_model=heldout_model,
        method=method,
        budget=budget,
        seed=seed,
        acquisition="direct_regressor",
        n_new_model_evals=len(sampled),
        training_time_s=time.perf_counter() - start,
        group_count=0,
    )


def summarize_selection(
    selected: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    heldout_model: str,
    method: str,
    budget: int,
    seed: int,
    acquisition: str,
    n_new_model_evals: int,
    training_time_s: float,
    group_count: int,
) -> dict[str, Any]:
    oracle = outputs[outputs["split"].eq("test")].sort_values(["query_id", "utility"], ascending=[True, False]).groupby("query_id").head(1)
    oracle_mean_utility = float(oracle["utility"].mean())
    selected = selected.copy()
    frontier = selected["model_id"].astype(str).isin(FRONTIER_MODELS)
    return {
        "heldout_model": heldout_model,
        "heldout_model_type": model_type(heldout_model),
        "method": method,
        "budget": int(budget),
        "seed": int(seed),
        "acquisition": acquisition,
        "n_new_model_evals": int(n_new_model_evals),
        "group_count": int(group_count),
        "n_test_queries": int(selected["query_id"].nunique()),
        "mean_quality": float(selected["quality_score"].mean()),
        "mean_utility": float(selected["utility"].mean()),
        "query_oracle_mean_utility": oracle_mean_utility,
        "query_oracle_regret": oracle_mean_utility - float(selected["utility"].mean()),
        "new_model_selection_rate": float(selected["model_id"].astype(str).eq(heldout_model).mean()),
        "frontier_call_rate": float(frontier.mean()),
        "remote_cost_per_1k_queries": float(selected["cost_total_usd"].sum(skipna=True) / max(len(selected), 1) * 1000.0),
        "mean_normalized_cost": float(selected["normalized_remote_cost"].mean()),
        "training_time_s": float(training_time_s),
    }


def model_type(model_id: str) -> str:
    if model_id == "deterministic_math_tool":
        return "verifiable_action"
    if model_id in FRONTIER_MODELS:
        return "frontier"
    if "32b" in model_id:
        return "strong_local"
    if "14b" in model_id or "8b" in model_id:
        return "medium_local"
    return "cheap_local"


def summarize_by_model_type(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table[table["budget"].ge(0)]
        .groupby(["heldout_model_type", "method", "budget"], as_index=False)
        .agg(
            mean_utility=("mean_utility", "mean"),
            mean_quality=("mean_quality", "mean"),
            regret_to_full_calibration=("regret_to_full_calibration", "mean"),
            n_new_model_evals=("n_new_model_evals", "mean"),
            new_model_selection_rate=("new_model_selection_rate", "mean"),
            training_time_s=("training_time_s", "mean"),
        )
        .sort_values(["heldout_model_type", "budget", "mean_utility"], ascending=[True, True, False])
    )


def summarize_acquisition(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table[table["method"].isin(["uniform_route_state_calibration", "active_route_state_calibration", "calibration_aware_route_state"])]
        .groupby(["method", "budget"], as_index=False)
        .agg(
            mean_utility=("mean_utility", "mean"),
            regret_to_full_calibration=("regret_to_full_calibration", "mean"),
            n_new_model_evals=("n_new_model_evals", "mean"),
        )
        .sort_values(["budget", "mean_utility"], ascending=[True, False])
    )


def write_figures(out_dir: Path, table: pd.DataFrame) -> None:
    plot = (
        table[table["budget"].ge(0)]
        .groupby(["method", "budget"], as_index=False)["mean_utility"]
        .mean()
        .sort_values(["method", "budget"])
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, frame in plot.groupby("method"):
        if method in {
            "random_query_route_state_calibration",
            "dataset_stratified_calibration",
            "embedding_cluster_calibration",
            "uniform_route_state_calibration",
            "active_route_state_calibration",
            "direct_probe_regressor_retrain",
        }:
            ax.plot(frame["budget"], frame["mean_utility"], marker="o", label=method)
    ax.set_xlabel("New-model calibration evaluations")
    ax.set_ylabel("Mean utility after onboarding")
    ax.set_title("Simulated New-Model Onboarding")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_utility_vs_calibration_budget.pdf")
    plt.close(fig)

    quality = table[table["budget"].ge(0)].groupby(["method", "budget"], as_index=False)["mean_quality"].mean()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, frame in quality.groupby("method"):
        if method in {
            "random_query_route_state_calibration",
            "dataset_stratified_calibration",
            "embedding_cluster_calibration",
            "uniform_route_state_calibration",
            "active_route_state_calibration",
            "direct_probe_regressor_retrain",
        }:
            ax.plot(frame["budget"], frame["mean_quality"], marker="o", label=method)
    ax.set_xlabel("New-model calibration evaluations")
    ax.set_ylabel("Mean quality after onboarding")
    ax.set_title("Simulated New-Model Onboarding Quality")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_quality_vs_calibration_budget.pdf")
    plt.close(fig)


def write_memo(path: Path, table: pd.DataFrame, by_type: pd.DataFrame, config: dict[str, Any]) -> None:
    final_budget = int(table[table["budget"].ge(0)]["budget"].max())
    top = (
        table[table["budget"].eq(final_budget)]
        .groupby("method", as_index=False)
        .agg(mean_utility=("mean_utility", "mean"), regret=("regret_to_full_calibration", "mean"), evals=("n_new_model_evals", "mean"))
        .sort_values("mean_utility", ascending=False)
    )
    lines = [
        "# Simulated New-Model Onboarding",
        "",
        "This cache-only experiment treats each cached action as a held-out new model/action.",
        "",
        "## Inputs",
        "",
        f"- Outcome matrix: `{config['inputs']['broad100_outputs']}`",
        f"- Probe features: `{config['inputs']['broad100_probe_features']}`",
        f"- RouteCode state method: `{config['method']['compact_state_method']}`",
        "",
        f"## Mean Results At Budget {final_budget}",
        "",
    ]
    for row in top.to_dict("records"):
        lines.append(
            f"- `{row['method']}`: utility `{float(row['mean_utility']):.4f}`, "
            f"regret to full `{float(row['regret']):.4f}`, evals `{float(row['evals']):.1f}`"
        )
    lines.extend(["", "## Model-Type Coverage", ""])
    for row in by_type[by_type["budget"].eq(final_budget)].head(20).to_dict("records"):
        lines.append(
            f"- `{row['heldout_model_type']}` / `{row['method']}`: utility `{float(row['mean_utility']):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is simulated from cached outcomes. It measures sample efficiency for state-table calibration, not live API deployment cost.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
