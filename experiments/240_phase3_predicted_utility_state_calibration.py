from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
CALIBRATION_HELPERS = Path("experiments/232_phase3_calibration_strata.py")
ONBOARDING_HELPERS = Path("experiments/233_phase3_new_model_onboarding.py")
ID_COLS = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deployable predicted utility states as calibration strata "
            "and new-model onboarding groups."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k-values", type=int, nargs="*", default=[6, 8, 16, 24])
    parser.add_argument("--classifiers", nargs="*", default=["rf", "extratrees"])
    parser.add_argument("--feature-views", nargs="*", default=["probe_only", "probe_plus_benchmark"])
    parser.add_argument("--budgets", type=int, nargs="*", default=[20, 40, 80, 160, 320])
    parser.add_argument("--seeds", type=int, nargs="*", default=[17, 18, 19])
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = args.output_dir or Path(config["outputs"]["root"]) / "predicted_utility_states"
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_module("routecode_phase3_calibration_helpers", CALIBRATION_HELPERS)
    onboarding = load_module("routecode_phase3_onboarding_helpers", ONBOARDING_HELPERS)

    outputs = calibration.load_outputs(Path(config["inputs"]["broad100_outputs"]), float(config["method"]["lambda_cost"]))
    query_table = calibration.query_metadata(outputs)
    feature_table = load_feature_table(Path(config["inputs"]["broad100_probe_features"]), query_table)

    baseline_groups = build_baseline_groups(calibration, outputs, query_table, config, seed=int(args.seed))
    predicted_groups, prediction_diagnostics = build_predicted_state_groups(
        outputs,
        query_table,
        feature_table,
        k_values=args.k_values,
        classifier_names=args.classifiers,
        feature_views=args.feature_views,
        seed=int(args.seed),
    )
    all_groups = pd.concat([baseline_groups, predicted_groups], ignore_index=True)

    variance, group_detail = calibration.state_variance(outputs, all_groups)
    estimation = calibration.estimation_error(outputs, all_groups)
    best_model = calibration.best_model_accuracy(outputs, all_groups)
    validation_estimation = split_estimation_error(outputs, all_groups, target_split="val")
    selected_strata_method = select_method_on_validation(variance)
    selected_onboarding_method = select_onboarding_method(validation_estimation)
    selected_groups = predicted_groups[predicted_groups["group_method"].eq(selected_onboarding_method)].copy()

    onboarding_rows = run_predicted_state_onboarding(
        onboarding,
        outputs,
        feature_table,
        selected_groups,
        budgets=args.budgets,
        seeds=args.seeds,
    )
    onboarding_table = pd.DataFrame(onboarding_rows).sort_values(
        ["heldout_model", "budget", "mean_utility"], ascending=[True, True, False]
    )
    claims = build_claims(variance, onboarding_table, selected_strata_method, selected_onboarding_method)

    prediction_diagnostics.to_csv(out_dir / "table_predicted_state_diagnostics.csv", index=False)
    predicted_groups.to_csv(out_dir / "table_predicted_state_assignments.csv", index=False)
    variance.to_csv(out_dir / "table_predicted_state_variance.csv", index=False)
    group_detail.to_csv(out_dir / "table_predicted_state_group_details.csv", index=False)
    estimation.to_csv(out_dir / "table_predicted_state_estimation_error.csv", index=False)
    validation_estimation.to_csv(out_dir / "table_predicted_state_validation_estimation_error.csv", index=False)
    best_model.to_csv(out_dir / "table_predicted_state_best_model_accuracy.csv", index=False)
    onboarding_table.to_csv(out_dir / "table_predicted_state_onboarding.csv", index=False)
    claims.to_csv(out_dir / "table_predicted_state_claims.csv", index=False)
    write_figures(out_dir, variance, onboarding_table, selected_strata_method)
    write_memo(
        out_dir / "PREDICTED_UTILITY_STATE_MEMO.md",
        config,
        selected_strata_method,
        selected_onboarding_method,
        prediction_diagnostics,
        variance,
        estimation,
        validation_estimation,
        best_model,
        onboarding_table,
        claims,
    )
    print(f"Wrote predicted utility-state calibration experiment to {out_dir}")
    print(f"Selected strata state method on validation variance: {selected_strata_method}")
    print(f"Selected onboarding state method on validation estimation error: {selected_onboarding_method}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_feature_table(path: Path, query_table: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(path)
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


def build_baseline_groups(
    calibration: Any,
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    frames = [
        calibration.random_groups(query_table, k=8, seed=seed),
        calibration.label_groups(query_table, "benchmark_label", "benchmark"),
        calibration.text_cluster_groups(query_table, k=8, seed=seed),
        calibration.utility_cluster_groups(outputs, query_table, k=8, seed=seed),
        calibration.routecode_groups(config, query_table, calibration_aware=False),
        calibration.routecode_groups(config, query_table, calibration_aware=True),
    ]
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def build_predicted_state_groups(
    outputs: pd.DataFrame,
    query_table: pd.DataFrame,
    feature_table: pd.DataFrame,
    *,
    k_values: list[int],
    classifier_names: list[str],
    feature_views: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean").dropna(axis=0)
    train_ids = query_table[query_table["split"].eq("train")]["query_id"].astype(str)
    train_matrix = matrix.reindex(train_ids).dropna(axis=0)
    if train_matrix.empty:
        raise ValueError("No complete train utility rows available for state learning.")

    rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for k in k_values:
        if len(train_matrix) < k:
            continue
        scaler = StandardScaler()
        train_utility = scaler.fit_transform(train_matrix.to_numpy(dtype=float))
        clusterer = ExtraTreesSafeKMeans(k=k, seed=seed)
        train_cluster = clusterer.fit_predict(train_utility)
        oracle_assignments = assign_utility_clusters(matrix, query_table, scaler, clusterer)
        train_labels = pd.Series(train_cluster, index=train_matrix.index.astype(str), name="utility_state")

        for feature_view in feature_views:
            x_all, feature_cols = design_matrix(feature_table, feature_view)
            x_train = x_all.reindex(train_matrix.index.astype(str)).fillna(0.0)
            for classifier_name in classifier_names:
                clf = make_classifier(classifier_name, seed=seed)
                clf.fit(x_train.to_numpy(dtype=float), train_labels.to_numpy(dtype=int))
                pred = pd.Series(
                    clf.predict(x_all.to_numpy(dtype=float)).astype(int),
                    index=x_all.index.astype(str),
                    name="predicted_state",
                )
                method = f"predicted_utility_state_{classifier_name}_{feature_view}_k{k}"
                frame = query_table[["query_id", "split", "benchmark"]].copy()
                frame["query_id"] = frame["query_id"].astype(str)
                frame["group_method"] = method
                frame["group_id"] = "p" + frame["query_id"].map(pred).astype(int).astype(str).str.zfill(2)
                frame["deployability"] = "train_utility_clusters_predicted_from_observable_probe_features"
                rows.append(frame)

                joined = oracle_assignments.merge(
                    pd.DataFrame({"query_id": pred.index, "predicted_state": pred.to_numpy(dtype=int)}),
                    on="query_id",
                    how="inner",
                )
                for split, split_frame in joined.groupby("split", sort=False):
                    diagnostics.append(
                        {
                            "group_method": method,
                            "split": split,
                            "k": int(k),
                            "classifier": classifier_name,
                            "feature_view": feature_view,
                            "n_queries": int(len(split_frame)),
                            "n_features": int(len(feature_cols)),
                            "oracle_cluster_accuracy": float(
                                accuracy_score(split_frame["oracle_state"], split_frame["predicted_state"])
                            ),
                            "oracle_cluster_adjusted_rand": float(
                                adjusted_rand_score(split_frame["oracle_state"], split_frame["predicted_state"])
                            ),
                        }
                    )
    if not rows:
        raise ValueError("No predicted utility-state groups were created.")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


class ExtraTreesSafeKMeans:
    """Small wrapper so the script can keep deterministic KMeans behavior local."""

    def __init__(self, *, k: int, seed: int) -> None:
        from sklearn.cluster import KMeans

        self.model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=30)

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.fit_predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


def assign_utility_clusters(
    matrix: pd.DataFrame,
    query_table: pd.DataFrame,
    scaler: StandardScaler,
    clusterer: ExtraTreesSafeKMeans,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split, frame in query_table.groupby("split", sort=False):
        split_matrix = matrix.reindex(frame["query_id"].astype(str)).dropna(axis=0)
        if split_matrix.empty:
            continue
        labels = clusterer.predict(scaler.transform(split_matrix.to_numpy(dtype=float)))
        rows.append(pd.DataFrame({"query_id": split_matrix.index.astype(str), "split": split, "oracle_state": labels}))
    return pd.concat(rows, ignore_index=True)


def design_matrix(feature_table: pd.DataFrame, feature_view: str) -> tuple[pd.DataFrame, list[str]]:
    work = feature_table.copy()
    numeric_cols = [col for col in work.columns if col not in ID_COLS and pd.api.types.is_numeric_dtype(work[col])]
    x = work[["query_id", *numeric_cols]].copy()
    if feature_view in {"probe_plus_benchmark", "probe_plus_metadata"}:
        one_hot = pd.get_dummies(work[["benchmark", "domain"]].fillna("unknown").astype(str), prefix=["bench", "domain"])
        x = pd.concat([x, one_hot], axis=1)
    elif feature_view != "probe_only":
        raise ValueError(f"Unknown feature view: {feature_view}")
    x = x.set_index("query_id")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, list(x.columns)


def make_classifier(name: str, *, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


def select_method_on_validation(variance: pd.DataFrame) -> str:
    candidates = variance[
        variance["split"].astype(str).eq("val")
        & variance["group_method"].astype(str).str.startswith("predicted_utility_state_")
    ].copy()
    if candidates.empty:
        raise ValueError("No validation rows for predicted utility states.")
    selected = candidates.sort_values(["weighted_utility_variance", "n_groups"], ascending=[True, True]).iloc[0]
    return str(selected["group_method"])


def split_estimation_error(outputs: pd.DataFrame, groups: pd.DataFrame, *, target_split: str) -> pd.DataFrame:
    work = outputs.merge(groups, on=["query_id", "split", "benchmark"], how="inner")
    means = (
        work.groupby(["group_method", "deployability", "split", "group_id", "model_id"], as_index=False)
        .agg(n_queries=("query_id", "nunique"), mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"))
    )
    train = means[means["split"].eq("train")].rename(
        columns={"mean_utility": "train_mean_utility", "mean_quality": "train_mean_quality", "n_queries": "train_n_queries"}
    )
    target = means[means["split"].eq(target_split)].rename(
        columns={
            "mean_utility": f"{target_split}_mean_utility",
            "mean_quality": f"{target_split}_mean_quality",
            "n_queries": f"{target_split}_n_queries",
        }
    )
    joined = target.merge(
        train[["group_method", "group_id", "model_id", "train_mean_utility", "train_mean_quality", "train_n_queries"]],
        on=["group_method", "group_id", "model_id"],
        how="left",
    ).dropna(subset=["train_mean_utility"])
    joined["abs_utility_estimation_error"] = (
        joined[f"{target_split}_mean_utility"] - joined["train_mean_utility"]
    ).abs()
    joined["abs_quality_estimation_error"] = (
        joined[f"{target_split}_mean_quality"] - joined["train_mean_quality"]
    ).abs()
    rows: list[dict[str, Any]] = []
    for keys, frame in joined.groupby(["group_method", "deployability"]):
        method, deployability = keys
        weights = frame[f"{target_split}_n_queries"].to_numpy(dtype=float)
        rows.append(
            {
                "group_method": method,
                "deployability": deployability,
                "target_split": target_split,
                "n_cells": int(len(frame)),
                "weighted_abs_utility_estimation_error": weighted_mean(
                    frame["abs_utility_estimation_error"], weights
                ),
                "weighted_abs_quality_estimation_error": weighted_mean(
                    frame["abs_quality_estimation_error"], weights
                ),
                "median_abs_utility_estimation_error": float(frame["abs_utility_estimation_error"].median()),
                "mean_train_queries_per_cell": float(frame["train_n_queries"].mean()),
                f"mean_{target_split}_queries_per_cell": float(frame[f"{target_split}_n_queries"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_abs_utility_estimation_error")


def weighted_mean(values: pd.Series | np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return float(np.nan)
    return float(np.average(values, weights=weights))


def select_onboarding_method(validation_estimation: pd.DataFrame) -> str:
    candidates = validation_estimation[
        validation_estimation["group_method"].astype(str).str.startswith("predicted_utility_state_")
    ].copy()
    if candidates.empty:
        raise ValueError("No validation estimation rows for predicted utility states.")
    selected = candidates.sort_values(["weighted_abs_utility_estimation_error", "n_cells"], ascending=[True, True]).iloc[0]
    return str(selected["group_method"])


def run_predicted_state_onboarding(
    onboarding: Any,
    outputs: pd.DataFrame,
    feature_table: pd.DataFrame,
    selected_groups: pd.DataFrame,
    *,
    budgets: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    if selected_groups.empty:
        raise ValueError("Selected predicted state group table is empty.")
    features = onboarding_ready_features(feature_table)
    models = sorted(outputs["model_id"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for heldout_model in models:
        train_ids_all = set(
            outputs[
                outputs["split"].eq("train")
                & outputs["model_id"].astype(str).eq(heldout_model)
                & outputs["status"].astype(str).eq("success")
            ]["query_id"].astype(str)
        )
        full = onboarding.evaluate_group_calibration(
            outputs,
            selected_groups,
            heldout_model=heldout_model,
            sampled_query_ids=train_ids_all,
            method="full_predicted_utility_state_calibration",
            budget=-1,
            seed=-1,
            acquisition="all_train",
            training_time_s=0.0,
        )
        rows.append({**full, "full_calibration_mean_utility": full["mean_utility"], "regret_to_full_calibration": 0.0})
        direct_full = onboarding.evaluate_direct_regressor(
            outputs,
            features,
            heldout_model=heldout_model,
            budget=len(train_ids_all),
            seed=-1,
            method="full_direct_probe_regressor_retrain",
        )
        rows.append(
            {
                **direct_full,
                "budget": -1,
                "n_new_model_evals": len(train_ids_all),
                "full_calibration_mean_utility": direct_full["mean_utility"],
                "regret_to_full_calibration": 0.0,
            }
        )

        for budget in budgets:
            for seed in seeds:
                sampled_random = onboarding.sample_calibration_queries(
                    outputs,
                    selected_groups,
                    heldout_model,
                    budget=budget,
                    seed=seed,
                    acquisition="random_query",
                )
                sampled_uniform = onboarding.sample_calibration_queries(
                    outputs,
                    selected_groups,
                    heldout_model,
                    budget=budget,
                    seed=seed,
                    acquisition="uniform_group",
                )
                sampled_active = sample_adaptive_value_queries(
                    outputs,
                    selected_groups,
                    heldout_model,
                    budget=budget,
                    seed=seed,
                )
                for method, acquisition, sampled in [
                    ("random_query_predicted_utility_state", "random_query", sampled_random),
                    ("uniform_predicted_utility_state", "uniform_group", sampled_uniform),
                    ("active_predicted_utility_state", "adaptive_value", sampled_active),
                ]:
                    row = onboarding.evaluate_group_calibration(
                        outputs,
                        selected_groups,
                        heldout_model=heldout_model,
                        sampled_query_ids=sampled,
                        method=method,
                        budget=budget,
                        seed=seed,
                        acquisition=acquisition,
                        training_time_s=0.0,
                    )
                    row["full_calibration_mean_utility"] = full["mean_utility"]
                    row["regret_to_full_calibration"] = full["mean_utility"] - row["mean_utility"]
                    rows.append(row)
                start = time.perf_counter()
                direct = onboarding.evaluate_direct_regressor(
                    outputs,
                    features,
                    heldout_model=heldout_model,
                    budget=budget,
                    seed=seed,
                )
                direct["training_time_s"] = float(direct.get("training_time_s", 0.0)) + (time.perf_counter() - start)
                direct["full_calibration_mean_utility"] = direct_full["mean_utility"]
                direct["regret_to_full_calibration"] = direct_full["mean_utility"] - direct["mean_utility"]
                rows.append(direct)
    return rows


def onboarding_ready_features(feature_table: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in feature_table.columns if col not in ID_COLS and pd.api.types.is_numeric_dtype(feature_table[col])]
    cols = ["query_id", "split", "benchmark", *numeric_cols]
    out = feature_table[cols].copy()
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def sample_adaptive_value_queries(
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    heldout_model: str,
    *,
    budget: int,
    seed: int,
) -> set[str]:
    train_ids = sorted(
        outputs[
            outputs["split"].eq("train")
            & outputs["model_id"].astype(str).eq(heldout_model)
            & outputs["status"].astype(str).eq("success")
        ]["query_id"].astype(str).unique()
    )
    if budget >= len(train_ids):
        return set(train_ids)
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    group_train = groups[groups["split"].eq("train") & groups["query_id"].isin(train_ids)].copy()
    group_train["query_id"] = group_train["query_id"].astype(str)

    train = outputs[outputs["split"].eq("train")].merge(group_train[["query_id", "group_id"]], on="query_id", how="inner")
    existing = train[~train["model_id"].astype(str).eq(heldout_model)].copy()
    existing_best = existing.groupby(["group_id", "model_id"], as_index=False)["utility"].mean()
    existing_best = existing_best.loc[existing_best.groupby("group_id")["utility"].idxmax()]
    existing_best = existing_best.set_index("group_id")["utility"].to_dict()

    heldout = train[train["model_id"].astype(str).eq(heldout_model)].set_index("query_id")["utility"].astype(float).to_dict()
    by_group = {
        str(group_id): frame["query_id"].astype(str).drop_duplicates().tolist()
        for group_id, frame in group_train.groupby("group_id")
    }
    traffic = {group_id: len(qids) for group_id, qids in by_group.items()}
    global_std = float(np.nanstd(list(heldout.values()))) if heldout else 0.25
    global_std = max(global_std, 0.05)

    selected: set[str] = set()
    group_order = sorted(by_group, key=lambda gid: traffic.get(gid, 0), reverse=True)
    for group_id in group_order:
        if len(selected) >= min(budget, len(group_order)):
            break
        choices = [qid for qid in by_group[group_id] if qid in heldout and qid not in selected]
        if choices:
            selected.add(str(rng.choice(choices)))

    while len(selected) < budget:
        best_group = None
        best_score = -np.inf
        for group_id, qids in by_group.items():
            available = [qid for qid in qids if qid in heldout and qid not in selected]
            if not available:
                continue
            sampled_values = np.asarray([heldout[qid] for qid in qids if qid in selected and qid in heldout], dtype=float)
            n = len(sampled_values)
            if n == 0:
                mean = float(np.nanmean(list(heldout.values()))) if heldout else -np.inf
                std = global_std
            elif n == 1:
                mean = float(sampled_values.mean())
                std = global_std
            else:
                mean = float(sampled_values.mean())
                std = max(float(sampled_values.std(ddof=1)), 0.03)
            current_best = float(existing_best.get(group_id, 0.0))
            gap = abs(mean - current_best)
            uncertainty = std / np.sqrt(n + 1.0)
            score = traffic.get(group_id, 1) * (uncertainty + 0.02) / (gap + 0.05)
            if score > best_score:
                best_score = score
                best_group = group_id
        if best_group is None:
            break
        choices = [qid for qid in by_group[best_group] if qid in heldout and qid not in selected]
        selected.add(str(rng.choice(choices)))
    return selected


def build_claims(
    variance: pd.DataFrame,
    onboarding_table: pd.DataFrame,
    selected_strata_method: str,
    selected_onboarding_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    test = variance[variance["split"].astype(str).eq("test")].copy()
    selected_var = test[test["group_method"].eq(selected_strata_method)]
    benchmark = test[test["group_method"].eq("benchmark_label")]
    text = test[test["group_method"].astype(str).str.startswith("text_cluster")]
    if not selected_var.empty and not benchmark.empty and not text.empty:
        selected_value = float(selected_var.iloc[0]["weighted_utility_variance"])
        simple_value = float(pd.concat([benchmark, text])["weighted_utility_variance"].min())
        status = "supported_on_cached_broad100" if selected_value < simple_value else "not_supported_on_cached_broad100"
        rows.append(
            {
                "claim_id": "predicted_states_as_calibration_strata",
                "status": status,
                "evidence": (
                    f"selected={selected_strata_method};test_variance={selected_value:.4f};"
                    f"best_label_or_text={simple_value:.4f}"
                ),
                "caveat": "State predictor is trained on Broad100 train outcomes and observable cached probe features; no fresh model calls.",
            }
        )
    budgeted = onboarding_table[onboarding_table["budget"].ge(0)].copy()
    if not budgeted.empty:
        max_budget = int(budgeted["budget"].max())
        active = budgeted[(budgeted["budget"].eq(max_budget)) & budgeted["method"].eq("active_predicted_utility_state")]
        random = budgeted[(budgeted["budget"].eq(max_budget)) & budgeted["method"].eq("random_query_predicted_utility_state")]
        uniform = budgeted[(budgeted["budget"].eq(max_budget)) & budgeted["method"].eq("uniform_predicted_utility_state")]
        if not active.empty and not random.empty and not uniform.empty:
            active_u = float(active["mean_utility"].mean())
            random_u = float(random["mean_utility"].mean())
            uniform_u = float(uniform["mean_utility"].mean())
            margin = active_u - max(random_u, uniform_u)
            if margin > 0.005:
                active_status = "supported_on_cached_broad100"
            elif margin > 0.0:
                active_status = "weakly_supported_on_cached_broad100"
            else:
                active_status = "partial_or_not_supported"
            rows.append(
                {
                    "claim_id": "active_acquisition_advantage",
                    "status": active_status,
                    "evidence": (
                        f"selected={selected_onboarding_method};budget={max_budget};active={active_u:.4f};"
                        f"random={random_u:.4f};uniform={uniform_u:.4f};margin={margin:.4f}"
                    ),
                    "caveat": "Simulated held-out-model onboarding from cached outcomes; active uses acquired train labels only.",
                }
            )
        direct = budgeted[(budgeted["budget"].eq(max_budget)) & budgeted["method"].eq("direct_probe_regressor_retrain")]
        state_rows = budgeted[
            budgeted["budget"].eq(max_budget)
            & budgeted["method"].isin(
                [
                    "active_predicted_utility_state",
                    "random_query_predicted_utility_state",
                    "uniform_predicted_utility_state",
                ]
            )
        ]
        if not state_rows.empty and not direct.empty:
            best_state = float(state_rows.groupby("method")["mean_utility"].mean().max())
            direct_u = float(direct["mean_utility"].mean())
            status = "supported_on_cached_broad100" if best_state > direct_u else "not_supported_on_cached_broad100"
            rows.append(
                {
                    "claim_id": "predicted_state_new_model_onboarding",
                    "status": status,
                    "evidence": (
                        f"selected={selected_onboarding_method};budget={max_budget};"
                        f"best_state={best_state:.4f};direct_retrain_proxy={direct_u:.4f};"
                        f"state_minus_direct={best_state - direct_u:.4f}"
                    ),
                    "caveat": "This supports state-based calibration, not a strong active-acquisition superiority claim.",
                }
            )
    return pd.DataFrame(rows)


def write_figures(out_dir: Path, variance: pd.DataFrame, onboarding_table: pd.DataFrame, selected_method: str) -> None:
    plot = variance[
        variance["split"].astype(str).eq("test")
        & (
            variance["group_method"].astype(str).isin(["benchmark_label", selected_method])
            | variance["group_method"].astype(str).str.startswith("text_cluster")
            | variance["group_method"].astype(str).str.startswith("routecode_state")
            | variance["group_method"].astype(str).str.startswith("calibration_aware_routecode")
        )
    ].sort_values("weighted_utility_variance")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(plot["group_method"], plot["weighted_utility_variance"], color="#3f6f6a")
    ax.set_xlabel("Traffic-weighted within-state utility variance")
    ax.set_title("Predicted Utility States As Calibration Strata")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_predicted_state_variance.pdf")
    plt.close(fig)

    budgeted = onboarding_table[onboarding_table["budget"].ge(0)]
    if budgeted.empty:
        return
    summary = (
        budgeted.groupby(["method", "budget"], as_index=False)["mean_utility"]
        .mean()
        .sort_values(["method", "budget"])
    )
    keep = {
        "random_query_predicted_utility_state",
        "uniform_predicted_utility_state",
        "active_predicted_utility_state",
        "direct_probe_regressor_retrain",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, frame in summary.groupby("method"):
        if method in keep:
            ax.plot(frame["budget"], frame["mean_utility"], marker="o", label=method)
    ax.set_xlabel("New-model calibration evaluations")
    ax.set_ylabel("Mean utility after onboarding")
    ax.set_title("Predicted Utility-State Onboarding")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_predicted_state_onboarding.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    config: dict[str, Any],
    selected_strata_method: str,
    selected_onboarding_method: str,
    diagnostics: pd.DataFrame,
    variance: pd.DataFrame,
    estimation: pd.DataFrame,
    validation_estimation: pd.DataFrame,
    best_model: pd.DataFrame,
    onboarding_table: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    test_var = variance[variance["split"].astype(str).eq("test")].sort_values("weighted_utility_variance")
    selected_var = test_var[test_var["group_method"].eq(selected_strata_method)]
    val_diag = diagnostics[diagnostics["split"].astype(str).eq("val")].copy()
    selected_diag = diagnostics[diagnostics["group_method"].eq(selected_onboarding_method)].copy()
    max_budget = int(onboarding_table[onboarding_table["budget"].ge(0)]["budget"].max()) if not onboarding_table.empty else -1
    onboarding_summary = (
        onboarding_table[onboarding_table["budget"].eq(max_budget)]
        .groupby("method", as_index=False)
        .agg(mean_utility=("mean_utility", "mean"), mean_quality=("mean_quality", "mean"), evals=("n_new_model_evals", "mean"))
        .sort_values("mean_utility", ascending=False)
    )
    lines = [
        "# Predicted Utility-State Calibration Experiment",
        "",
        "This experiment tests whether RouteCode states become better calibration strata when the state is learned from utility vectors on train and then predicted from cheap observable probe features.",
        "",
        "## Commands",
        "",
        "- `PYTHONPATH=src python experiments/240_phase3_predicted_utility_state_calibration.py --config configs/probecode_final_eval.yaml`",
        "",
        "## Inputs",
        "",
        f"- Outcome matrix: `{config['inputs']['broad100_outputs']}`",
        f"- Probe features: `{config['inputs']['broad100_probe_features']}`",
        "- No fresh model calls are made by this script.",
        "",
        "## Method",
        "",
        "1. Learn KMeans states from train query utility vectors only.",
        "2. Train a RandomForest or ExtraTrees predictor from observable probe features to those train utility states.",
        "3. Select the deployable strata variant using validation within-state utility variance.",
        "4. Select the onboarding variant using train-to-validation utility estimation error.",
        "5. Report held-out test strata quality and simulated held-out-model onboarding.",
        "",
        f"Selected strata method: `{selected_strata_method}`.",
        f"Selected onboarding method: `{selected_onboarding_method}`.",
        "",
    ]
    if not selected_diag.empty:
        lines.extend(["## State Prediction Diagnostics", ""])
        for row in selected_diag.sort_values("split").to_dict("records"):
            lines.append(
                f"- `{row['split']}`: adjusted Rand `{float(row['oracle_cluster_adjusted_rand']):.4f}`, "
                f"raw cluster accuracy `{float(row['oracle_cluster_accuracy']):.4f}`"
            )
        lines.append("")
    if not val_diag.empty:
        best_val = val_diag.sort_values("oracle_cluster_adjusted_rand", ascending=False).head(5)
        lines.extend(["Top validation state-prediction variants by adjusted Rand:", ""])
        for row in best_val.to_dict("records"):
            lines.append(
                f"- `{row['group_method']}`: ARI `{float(row['oracle_cluster_adjusted_rand']):.4f}`"
            )
        lines.append("")
    lines.extend(["## Validation Onboarding-State Selection", ""])
    for row in validation_estimation[
        validation_estimation["group_method"].astype(str).str.startswith("predicted_utility_state_")
    ].head(8).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: train-to-val utility estimation error "
            f"`{float(row['weighted_abs_utility_estimation_error']):.4f}`"
        )
    lines.append("")
    lines.extend(["## Test Calibration-Strata Results", ""])
    for row in test_var.head(12).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: utility variance `{float(row['weighted_utility_variance']):.4f}`, "
            f"groups `{int(row['n_groups'])}`"
        )
    lines.append("")
    if not selected_var.empty:
        row = selected_var.iloc[0]
        lines.append(
            f"Selected predicted state test variance: `{float(row['weighted_utility_variance']):.4f}`."
        )
        lines.append("")
    lines.extend(["## Estimation And Best-Model Checks", ""])
    for row in estimation.head(8).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: weighted abs utility estimation error "
            f"`{float(row['weighted_abs_utility_estimation_error']):.4f}`"
        )
    lines.append("")
    for row in best_model.head(8).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: traffic-weighted best-model match "
            f"`{float(row['traffic_weighted_best_model_identification_accuracy']):.4f}`"
        )
    lines.extend(["", f"## Onboarding Results At Budget {max_budget}", ""])
    for row in onboarding_summary.to_dict("records"):
        lines.append(
            f"- `{row['method']}`: utility `{float(row['mean_utility']):.4f}`, "
            f"quality `{float(row['mean_quality']):.4f}`, evals `{float(row['evals']):.1f}`"
        )
    lines.extend(["", "## Claim Status", ""])
    for row in claims.to_dict("records"):
        lines.append(f"- `{row['claim_id']}`: `{row['status']}`; {row['evidence']}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is cached Broad100 evidence, not a fresh GPT/Gemini calibration deployment.",
            "- Utility-cluster labels are learned from train outcomes, but validation/test assignments here are predicted from observable cached probe features.",
            "- The onboarding method is selected on validation estimation error, not test onboarding utility.",
            "- If the active acquisition row only narrowly beats random/uniform, the claim should be written as weak evidence, not a strong 3-5x sample-efficiency result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
