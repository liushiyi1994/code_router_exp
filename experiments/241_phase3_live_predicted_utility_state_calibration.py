from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
DEFAULT_LIVE_OUTPUTS = Path("results/controlled/live_broad100_stage0/model_outputs.parquet")
DEFAULT_FEATURES = Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv")
PREDICTED_STATE_HELPERS = Path("experiments/240_phase3_predicted_utility_state_calibration.py")
CALIBRATION_HELPERS = Path("experiments/232_phase3_calibration_strata.py")
ONBOARDING_HELPERS = Path("experiments/233_phase3_new_model_onboarding.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predicted utility-state calibration claims on the live Broad100 Stage0 matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--live-outputs", type=Path, default=DEFAULT_LIVE_OUTPUTS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-model-coverage", type=float, default=0.94)
    parser.add_argument("--k-values", type=int, nargs="*", default=[4, 6, 8, 12, 16])
    parser.add_argument("--budgets", type=int, nargs="*", default=[20, 40, 80, 160, 320])
    parser.add_argument("--seeds", type=int, nargs="*", default=[17, 18, 19])
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = args.output_dir or Path(config["outputs"]["root"]) / "live_predicted_utility_states"
    out_dir.mkdir(parents=True, exist_ok=True)

    helper = load_module("routecode_predicted_state_helpers", PREDICTED_STATE_HELPERS)
    calibration = load_module("routecode_phase3_calibration_helpers_live", CALIBRATION_HELPERS)
    onboarding = load_module("routecode_phase3_onboarding_helpers_live", ONBOARDING_HELPERS)

    features = pd.read_csv(args.features)
    live_outputs, coverage = prepare_live_outputs(
        args.live_outputs,
        features,
        lambda_cost=float(config["method"]["lambda_cost"]),
        min_model_coverage=float(args.min_model_coverage),
    )
    query_table = calibration.query_metadata(live_outputs)
    feature_table = helper.load_feature_table(args.features, query_table)

    baseline_groups = build_live_baselines(calibration, live_outputs, query_table, seed=int(args.seed))
    predicted_groups, diagnostics = helper.build_predicted_state_groups(
        live_outputs,
        query_table,
        feature_table,
        k_values=args.k_values,
        classifier_names=["rf", "extratrees"],
        feature_views=["probe_only", "probe_plus_benchmark"],
        seed=int(args.seed),
    )
    all_groups = pd.concat([baseline_groups, predicted_groups], ignore_index=True)
    variance, group_detail = calibration.state_variance(live_outputs, all_groups)
    estimation = calibration.estimation_error(live_outputs, all_groups)
    best_model = calibration.best_model_accuracy(live_outputs, all_groups)
    validation_estimation = helper.split_estimation_error(live_outputs, all_groups, target_split="val")

    selected_strata = helper.select_method_on_validation(variance)
    selected_onboarding = helper.select_onboarding_method(validation_estimation)
    selected_groups = predicted_groups[predicted_groups["group_method"].eq(selected_onboarding)].copy()
    onboarding_rows, active_validation = run_live_predicted_state_onboarding(
        onboarding,
        helper,
        live_outputs,
        feature_table,
        selected_groups,
        budgets=args.budgets,
        seeds=args.seeds,
    )
    onboarding_table = pd.DataFrame(onboarding_rows).sort_values(
        ["heldout_model", "budget", "mean_utility"], ascending=[True, True, False]
    )
    frontier_validation = summarize_frontier_onboarding(
        onboarding,
        helper,
        live_outputs,
        feature_table,
        selected_groups,
        active_validation,
        budgets=args.budgets,
        seeds=args.seeds,
        validation=True,
    )
    selected_frontier_budget = select_frontier_budget_on_validation(frontier_validation)
    frontier_test = summarize_frontier_onboarding(
        onboarding,
        helper,
        live_outputs,
        feature_table,
        selected_groups,
        active_validation,
        budgets=args.budgets,
        seeds=args.seeds,
        validation=False,
    )
    frontier_budget_efficiency = summarize_frontier_budget_efficiency(frontier_test, selected_frontier_budget)
    claims = build_live_claims(
        variance,
        onboarding_table,
        selected_strata,
        selected_onboarding,
        frontier_validation,
        frontier_test,
        frontier_budget_efficiency,
        selected_frontier_budget,
    )

    live_outputs.to_parquet(out_dir / "live_outputs_with_splits_and_utility.parquet", index=False)
    coverage.to_csv(out_dir / "table_live_model_coverage.csv", index=False)
    diagnostics.to_csv(out_dir / "table_live_predicted_state_diagnostics.csv", index=False)
    active_validation.to_csv(out_dir / "table_live_active_acquisition_validation.csv", index=False)
    frontier_validation.to_csv(out_dir / "table_live_frontier_onboarding_validation.csv", index=False)
    frontier_test.to_csv(out_dir / "table_live_frontier_onboarding_test.csv", index=False)
    frontier_budget_efficiency.to_csv(out_dir / "table_live_frontier_budget_efficiency.csv", index=False)
    predicted_groups.to_csv(out_dir / "table_live_predicted_state_assignments.csv", index=False)
    variance.to_csv(out_dir / "table_live_predicted_state_variance.csv", index=False)
    group_detail.to_csv(out_dir / "table_live_predicted_state_group_details.csv", index=False)
    estimation.to_csv(out_dir / "table_live_predicted_state_estimation_error.csv", index=False)
    validation_estimation.to_csv(out_dir / "table_live_predicted_state_validation_estimation_error.csv", index=False)
    best_model.to_csv(out_dir / "table_live_predicted_state_best_model_accuracy.csv", index=False)
    onboarding_table.to_csv(out_dir / "table_live_predicted_state_onboarding.csv", index=False)
    claims.to_csv(out_dir / "table_live_predicted_state_claims.csv", index=False)
    helper.write_figures(out_dir, variance, onboarding_table, selected_strata)
    write_live_memo(
        out_dir / "LIVE_PREDICTED_UTILITY_STATE_MEMO.md",
        args,
        config,
        coverage,
        selected_strata,
        selected_onboarding,
        variance,
        validation_estimation,
        active_validation,
        frontier_validation,
        frontier_test,
        frontier_budget_efficiency,
        selected_frontier_budget,
        onboarding_table,
        claims,
    )
    print(f"Wrote live predicted utility-state experiment to {out_dir}")
    print(f"Selected live strata state: {selected_strata}")
    print(f"Selected live onboarding state: {selected_onboarding}")


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


def prepare_live_outputs(
    live_path: Path,
    features: pd.DataFrame,
    *,
    lambda_cost: float,
    min_model_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    live = pd.read_parquet(live_path).copy()
    split_cols = features[["query_id", "split", "benchmark", "domain", "query_text"]].drop_duplicates("query_id")
    split_cols["query_id"] = split_cols["query_id"].astype(str)
    live["query_id"] = live["query_id"].astype(str)
    live = live.drop(columns=[col for col in ["split"] if col in live.columns])
    live = live.merge(split_cols, on="query_id", how="inner", suffixes=("", "_feature"))
    for col in ["benchmark", "domain", "query_text"]:
        feature_col = f"{col}_feature"
        if feature_col in live.columns:
            live[col] = live[feature_col].combine_first(live.get(col))
            live = live.drop(columns=[feature_col])
    live = live[live["status"].astype(str).eq("success")].copy()
    live["model_id"] = live["model_id"].astype(str)
    live["quality_score"] = pd.to_numeric(live["quality_score"], errors="coerce")
    live["cost_total_usd"] = pd.to_numeric(live["cost_total_usd"], errors="coerce").fillna(0.0)
    live = live.dropna(subset=["quality_score"])

    n_queries = live["query_id"].nunique()
    coverage = (
        live.groupby(["model_id", "provider"], as_index=False)
        .agg(
            successful_queries=("query_id", "nunique"),
            mean_quality=("quality_score", "mean"),
            total_cost_usd=("cost_total_usd", "sum"),
            mean_latency_s=("latency_s", "mean"),
        )
        .sort_values(["successful_queries", "mean_quality"], ascending=[False, False])
    )
    coverage["coverage_rate"] = coverage["successful_queries"] / max(n_queries, 1)
    selected_models = coverage[coverage["coverage_rate"].ge(min_model_coverage)]["model_id"].astype(str).tolist()
    if "gpt-5.5" not in selected_models and "gpt-5.5" in set(live["model_id"]):
        selected_models.append("gpt-5.5")
    if len(selected_models) < 3:
        raise ValueError(f"Not enough live models pass coverage {min_model_coverage}: {selected_models}")
    live = live[live["model_id"].isin(selected_models)].copy()
    gpt_cost = live[live["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = float(gpt_cost.mean()) if not gpt_cost.empty else float(live["cost_total_usd"].max())
    cost_norm = max(cost_norm, 1e-12)
    live["normalized_remote_cost"] = live["cost_total_usd"] / cost_norm
    live["utility"] = live["quality_score"] - float(lambda_cost) * live["normalized_remote_cost"]
    coverage["selected_for_live_claim_eval"] = coverage["model_id"].astype(str).isin(selected_models)
    coverage["cost_normalization_usd"] = cost_norm
    return live, coverage


def build_live_baselines(calibration: Any, outputs: pd.DataFrame, query_table: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    frames = [
        calibration.random_groups(query_table, k=8, seed=seed),
        calibration.label_groups(query_table, "benchmark_label", "benchmark"),
        calibration.text_cluster_groups(query_table, k=8, seed=seed),
        calibration.utility_cluster_groups(outputs, query_table, k=8, seed=seed),
    ]
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def build_live_claims(
    variance: pd.DataFrame,
    onboarding: pd.DataFrame,
    selected_strata: str,
    selected_onboarding: str,
    frontier_validation: pd.DataFrame,
    frontier_test: pd.DataFrame,
    frontier_budget_efficiency: pd.DataFrame,
    selected_frontier_budget: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    test = variance[variance["split"].astype(str).eq("test")].copy()
    selected = test[test["group_method"].eq(selected_strata)]
    simple = test[
        test["group_method"].astype(str).eq("benchmark_label")
        | test["group_method"].astype(str).str.startswith("text_cluster")
    ]
    if not selected.empty and not simple.empty:
        selected_var = float(selected.iloc[0]["weighted_utility_variance"])
        simple_var = float(simple["weighted_utility_variance"].min())
        rows.append(
            {
                "claim_id": "live_predicted_states_as_calibration_strata",
                "status": "supported_on_live_broad100_stage0" if selected_var < simple_var else "not_supported_on_live_broad100_stage0",
                "evidence": f"selected={selected_strata};test_variance={selected_var:.4f};best_label_or_text={simple_var:.4f}",
                "caveat": "Uses live/cached Stage0 GPT, Gemini, and local vLLM outcomes; state predictor uses cached observable probe features.",
            }
        )
    budgeted = onboarding[onboarding["budget"].ge(0)].copy()
    if not budgeted.empty:
        max_budget = int(budgeted["budget"].max())
        rows_at_budget = budgeted[budgeted["budget"].eq(max_budget)]
        active = mean_for(rows_at_budget, "active_predicted_utility_state")
        random = mean_for(rows_at_budget, "random_query_predicted_utility_state")
        uniform = mean_for(rows_at_budget, "uniform_predicted_utility_state")
        direct = mean_for(rows_at_budget, "direct_probe_regressor_retrain")
        best_state = max(active, random, uniform)
        active_status = "supported_on_live_broad100_stage0" if best_state > direct else "not_supported_on_live_broad100_stage0"
        if best_state > direct and best_state - direct < 0.005:
            active_status = "weakly_supported_on_live_broad100_stage0"
        rows.append(
            {
                "claim_id": "live_predicted_state_new_model_onboarding",
                "status": active_status,
                "evidence": (
                    f"selected={selected_onboarding};budget={max_budget};best_state={best_state:.4f};"
                    f"direct_retrain_proxy={direct:.4f};state_minus_direct={best_state - direct:.4f}"
                ),
                "caveat": "Supports state-based calibration on the live Stage0 matrix; not a proof of active acquisition superiority.",
            }
        )
        active_margin = active - max(random, uniform)
        if active_margin > 0.005:
            status = "supported_on_live_broad100_stage0"
        elif active_margin > 0.0:
            status = "weakly_supported_on_live_broad100_stage0"
        else:
            status = "not_supported_on_live_broad100_stage0"
        rows.append(
            {
                "claim_id": "live_active_acquisition_advantage",
                "status": status,
                "evidence": (
                    f"selected={selected_onboarding};budget={max_budget};active={active:.4f};"
                    f"random={random:.4f};uniform={uniform:.4f};margin={active_margin:.4f}"
                ),
                "caveat": "Active acquisition is evaluated separately from whether states are good calibration strata.",
            }
        )
    if selected_frontier_budget is not None and not frontier_test.empty:
        test_row = frontier_test[frontier_test["budget"].eq(int(selected_frontier_budget))]
        val_row = frontier_validation[frontier_validation["budget"].eq(int(selected_frontier_budget))]
        if not test_row.empty and not val_row.empty:
            row = test_row.iloc[0]
            margin = float(row["active_minus_best_competitor"])
            if margin > 0.005:
                status = "supported_on_live_broad100_stage0"
            elif margin > 0.0:
                status = "weakly_supported_on_live_broad100_stage0"
            else:
                status = "not_supported_on_live_broad100_stage0"
            rows.append(
                {
                    "claim_id": "live_frontier_active_onboarding_low_budget",
                    "status": status,
                    "evidence": (
                        f"validation_selected_budget={selected_frontier_budget};"
                        f"test_active={float(row['active_mean_utility']):.4f};"
                        f"test_best_competitor={float(row['best_competitor_mean_utility']):.4f};"
                        f"margin={margin:.4f};heldout_models=gpt-5.5,gemini-3.5-flash"
                    ),
                    "caveat": "This is a frontier-model onboarding slice, not an all-model average claim.",
                }
            )
    if not frontier_budget_efficiency.empty:
        direct = frontier_budget_efficiency[frontier_budget_efficiency["competitor"].eq("direct")]
        random = frontier_budget_efficiency[frontier_budget_efficiency["competitor"].eq("random")]
        if not direct.empty and not random.empty:
            direct_ratio = float(direct.iloc[0]["eval_reduction_lower_bound"])
            random_ratio = float(random.iloc[0]["eval_reduction_lower_bound"])
            status = (
                "supported_on_live_broad100_stage0"
                if direct_ratio >= 3.0 and random_ratio >= 3.0
                else "not_supported_on_live_broad100_stage0"
            )
            rows.append(
                {
                    "claim_id": "live_frontier_budget_efficiency",
                    "status": status,
                    "evidence": (
                        f"active_budget={int(direct.iloc[0]['active_budget'])};"
                        f"direct_eval_reduction_lower_bound={direct_ratio:.1f}x;"
                        f"random_eval_reduction_lower_bound={random_ratio:.1f}x;"
                        f"target_active_utility={float(direct.iloc[0]['active_mean_utility']):.4f}"
                    ),
                    "caveat": "Budget efficiency is measured on the validation-selected GPT/Gemini frontier slice.",
                }
            )
    return pd.DataFrame(rows)


def summarize_frontier_onboarding(
    onboarding: Any,
    helper: Any,
    outputs: pd.DataFrame,
    feature_table: pd.DataFrame,
    selected_groups: pd.DataFrame,
    active_validation: pd.DataFrame,
    *,
    budgets: list[int],
    seeds: list[int],
    validation: bool,
) -> pd.DataFrame:
    frontier_models = ["gemini-3.5-flash", "gpt-5.5"]
    active_by_budget = {
        int(row["budget"]): str(row["selected_acquisition"])
        for row in active_validation[active_validation["is_selected"].astype(bool)].to_dict("records")
    }
    eval_outputs = relabel_val_as_test(outputs) if validation else outputs
    eval_groups = relabel_val_as_test(selected_groups) if validation else selected_groups
    eval_features = helper.onboarding_ready_features(relabel_val_as_test(feature_table) if validation else feature_table)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        active_name = active_by_budget.get(int(budget), "traffic_active")
        method_rows: list[dict[str, Any]] = []
        for heldout_model in frontier_models:
            if heldout_model not in set(eval_outputs["model_id"].astype(str)):
                continue
            for seed in seeds:
                samples = {
                    "active": sample_active_acquisition(
                        active_name,
                        onboarding,
                        helper,
                        eval_outputs,
                        eval_groups,
                        heldout_model,
                        budget=budget,
                        seed=seed,
                    ),
                    "uniform": onboarding.sample_calibration_queries(
                        eval_outputs,
                        eval_groups,
                        heldout_model,
                        budget=budget,
                        seed=seed,
                        acquisition="uniform_group",
                    ),
                    "random": onboarding.sample_calibration_queries(
                        eval_outputs,
                        eval_groups,
                        heldout_model,
                        budget=budget,
                        seed=seed,
                        acquisition="random_query",
                    ),
                }
                for method, sampled in samples.items():
                    row = onboarding.evaluate_group_calibration(
                        eval_outputs,
                        eval_groups,
                        heldout_model=heldout_model,
                        sampled_query_ids=sampled,
                        method=method,
                        budget=budget,
                        seed=seed,
                        acquisition=method,
                        training_time_s=0.0,
                    )
                    method_rows.append(row)
                direct = onboarding.evaluate_direct_regressor(
                    eval_outputs,
                    eval_features,
                    heldout_model=heldout_model,
                    budget=budget,
                    seed=seed,
                )
                direct["method"] = "direct"
                method_rows.append(direct)
        method_table = pd.DataFrame(method_rows)
        means = method_table.groupby("method", as_index=False).agg(
            mean_utility=("mean_utility", "mean"),
            mean_quality=("mean_quality", "mean"),
            new_model_selection_rate=("new_model_selection_rate", "mean"),
        )
        values = {str(row["method"]): float(row["mean_utility"]) for row in means.to_dict("records")}
        qualities = {str(row["method"]): float(row["mean_quality"]) for row in means.to_dict("records")}
        rates = {str(row["method"]): float(row["new_model_selection_rate"]) for row in means.to_dict("records")}
        competitors = [values.get("uniform", float("-inf")), values.get("random", float("-inf")), values.get("direct", float("-inf"))]
        best_competitor = max(competitors)
        rows.append(
            {
                "split": "validation" if validation else "test",
                "budget": int(budget),
                "selected_active_acquisition": active_name,
                "active_mean_utility": values.get("active", float("nan")),
                "uniform_mean_utility": values.get("uniform", float("nan")),
                "random_mean_utility": values.get("random", float("nan")),
                "direct_mean_utility": values.get("direct", float("nan")),
                "best_competitor_mean_utility": best_competitor,
                "active_minus_best_competitor": values.get("active", float("nan")) - best_competitor,
                "active_mean_quality": qualities.get("active", float("nan")),
                "uniform_mean_quality": qualities.get("uniform", float("nan")),
                "random_mean_quality": qualities.get("random", float("nan")),
                "direct_mean_quality": qualities.get("direct", float("nan")),
                "active_new_model_selection_rate": rates.get("active", float("nan")),
                "uniform_new_model_selection_rate": rates.get("uniform", float("nan")),
                "random_new_model_selection_rate": rates.get("random", float("nan")),
                "direct_new_model_selection_rate": rates.get("direct", float("nan")),
            }
        )
    return pd.DataFrame(rows).sort_values(["budget"])


def select_frontier_budget_on_validation(frontier_validation: pd.DataFrame) -> int | None:
    if frontier_validation.empty:
        return None
    selected = frontier_validation.sort_values(
        ["active_minus_best_competitor", "active_mean_utility", "budget"], ascending=[False, False, True]
    ).head(1)
    if selected.empty:
        return None
    return int(selected.iloc[0]["budget"])


def summarize_frontier_budget_efficiency(
    frontier_test: pd.DataFrame,
    selected_frontier_budget: int | None,
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    if selected_frontier_budget is None or frontier_test.empty:
        return pd.DataFrame()
    selected = frontier_test[frontier_test["budget"].eq(int(selected_frontier_budget))]
    if selected.empty:
        return pd.DataFrame()
    active_utility = float(selected.iloc[0]["active_mean_utility"])
    max_budget = int(frontier_test["budget"].max())
    rows: list[dict[str, Any]] = []
    for competitor, col in [
        ("uniform", "uniform_mean_utility"),
        ("random", "random_mean_utility"),
        ("direct", "direct_mean_utility"),
    ]:
        matches = frontier_test[frontier_test[col].astype(float).ge(active_utility - tolerance)].sort_values("budget")
        matched = not matches.empty
        competitor_budget = int(matches.iloc[0]["budget"]) if matched else max_budget
        rows.append(
            {
                "competitor": competitor,
                "active_budget": int(selected_frontier_budget),
                "active_mean_utility": active_utility,
                "competitor_match_budget": competitor_budget if matched else np.nan,
                "competitor_best_tested_budget": max_budget,
                "competitor_utility_at_best_tested_budget": float(
                    frontier_test[frontier_test["budget"].eq(max_budget)].iloc[0][col]
                ),
                "matched_active_utility": bool(matched),
                "eval_reduction_lower_bound": float(competitor_budget / max(int(selected_frontier_budget), 1)),
            }
        )
    return pd.DataFrame(rows)


def run_live_predicted_state_onboarding(
    onboarding: Any,
    helper: Any,
    outputs: pd.DataFrame,
    feature_table: pd.DataFrame,
    selected_groups: pd.DataFrame,
    *,
    budgets: list[int],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    active_validation = select_active_acquisitions_on_validation(
        onboarding, helper, outputs, selected_groups, budgets=budgets, seeds=seeds
    )
    active_by_budget = {
        int(row["budget"]): str(row["selected_acquisition"])
        for row in active_validation[active_validation["is_selected"].astype(bool)].to_dict("records")
    }
    features = helper.onboarding_ready_features(feature_table)
    rows: list[dict[str, Any]] = []
    for heldout_model in sorted(outputs["model_id"].astype(str).unique()):
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
            active_name = active_by_budget.get(int(budget), "traffic_active")
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
                sampled_active = sample_active_acquisition(
                    active_name,
                    onboarding,
                    helper,
                    outputs,
                    selected_groups,
                    heldout_model,
                    budget=budget,
                    seed=seed,
                )
                for method, acquisition, sampled in [
                    ("random_query_predicted_utility_state", "random_query", sampled_random),
                    ("uniform_predicted_utility_state", "uniform_group", sampled_uniform),
                    ("active_predicted_utility_state", f"validation_selected:{active_name}", sampled_active),
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
                direct = onboarding.evaluate_direct_regressor(
                    outputs,
                    features,
                    heldout_model=heldout_model,
                    budget=budget,
                    seed=seed,
                )
                direct["full_calibration_mean_utility"] = direct_full["mean_utility"]
                direct["regret_to_full_calibration"] = direct_full["mean_utility"] - direct["mean_utility"]
                rows.append(direct)
    return rows, active_validation


def select_active_acquisitions_on_validation(
    onboarding: Any,
    helper: Any,
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    budgets: list[int],
    seeds: list[int],
) -> pd.DataFrame:
    outputs_val = relabel_val_as_test(outputs)
    groups_val = relabel_val_as_test(groups)
    candidate_names = ["active_margin", "adaptive_value", "traffic_active", "pilot_gain_active"]
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for name in candidate_names:
            utilities: list[float] = []
            qualities: list[float] = []
            for heldout_model in sorted(outputs_val["model_id"].astype(str).unique()):
                for seed in seeds:
                    sampled = sample_active_acquisition(
                        name,
                        onboarding,
                        helper,
                        outputs_val,
                        groups_val,
                        heldout_model,
                        budget=budget,
                        seed=seed,
                    )
                    row = onboarding.evaluate_group_calibration(
                        outputs_val,
                        groups_val,
                        heldout_model=heldout_model,
                        sampled_query_ids=sampled,
                        method=name,
                        budget=budget,
                        seed=seed,
                        acquisition=name,
                        training_time_s=0.0,
                    )
                    utilities.append(float(row["mean_utility"]))
                    qualities.append(float(row["mean_quality"]))
            rows.append(
                {
                    "budget": int(budget),
                    "candidate_acquisition": name,
                    "validation_mean_utility": float(np.mean(utilities)) if utilities else float("nan"),
                    "validation_mean_quality": float(np.mean(qualities)) if qualities else float("nan"),
                }
            )
    table = pd.DataFrame(rows)
    selected = (
        table.sort_values(["budget", "validation_mean_utility"], ascending=[True, False])
        .groupby("budget")
        .head(1)[["budget", "candidate_acquisition"]]
        .rename(columns={"candidate_acquisition": "selected_acquisition"})
    )
    table = table.merge(selected, on="budget", how="left")
    table["is_selected"] = table["candidate_acquisition"].astype(str).eq(table["selected_acquisition"].astype(str))
    return table.sort_values(["budget", "validation_mean_utility"], ascending=[True, False])


def relabel_val_as_test(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[frame["split"].astype(str).isin(["train", "val"])].copy()
    work["split"] = work["split"].astype(str).replace({"val": "test"})
    return work


def sample_active_acquisition(
    name: str,
    onboarding: Any,
    helper: Any,
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    heldout_model: str,
    *,
    budget: int,
    seed: int,
) -> set[str]:
    if name == "active_margin":
        return onboarding.sample_calibration_queries(
            outputs, groups, heldout_model, budget=budget, seed=seed, acquisition="active_margin"
        )
    if name == "adaptive_value":
        return helper.sample_adaptive_value_queries(outputs, groups, heldout_model, budget=budget, seed=seed)
    if name == "traffic_active":
        return sample_traffic_weighted(outputs, groups, heldout_model, budget=budget, seed=seed)
    if name == "pilot_gain_active":
        return sample_pilot_gain(outputs, groups, heldout_model, budget=budget, seed=seed)
    raise ValueError(f"Unknown active acquisition: {name}")


def sample_traffic_weighted(
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
    group_train = groups[groups["split"].eq("train") & groups["query_id"].isin(train_ids)].copy()
    by_group = {
        str(group_id): frame["query_id"].astype(str).drop_duplicates().tolist()
        for group_id, frame in group_train.groupby("group_id")
    }
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    group_ids = list(by_group)
    weights = np.asarray([len(by_group[group_id]) for group_id in group_ids], dtype=float)
    probs = weights / weights.sum()
    selected: list[str] = []
    while len(selected) < min(budget, len(train_ids)):
        group_id = str(rng.choice(group_ids, p=probs))
        available = [query_id for query_id in by_group[group_id] if query_id not in selected]
        if not available:
            available = [query_id for query_id in train_ids if query_id not in selected]
        if not available:
            break
        selected.append(str(rng.choice(available)))
    return set(selected)


def sample_pilot_gain(
    outputs: pd.DataFrame,
    groups: pd.DataFrame,
    heldout_model: str,
    *,
    budget: int,
    seed: int,
    pilot_per_group: int = 4,
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
    group_train = groups[groups["split"].eq("train") & groups["query_id"].isin(train_ids)].copy()
    by_group = {
        str(group_id): frame["query_id"].astype(str).drop_duplicates().tolist()
        for group_id, frame in group_train.groupby("group_id")
    }
    heldout_utility = (
        outputs[outputs["split"].eq("train") & outputs["model_id"].astype(str).eq(heldout_model)]
        .set_index("query_id")["utility"]
        .astype(float)
        .to_dict()
    )
    existing = outputs[outputs["split"].eq("train") & ~outputs["model_id"].astype(str).eq(heldout_model)].merge(
        group_train[["query_id", "group_id"]], on="query_id", how="inner"
    )
    existing_means = existing.groupby(["group_id", "model_id"], as_index=False)["utility"].mean()
    existing_best = existing_means.loc[existing_means.groupby("group_id")["utility"].idxmax()]
    existing_best = existing_best.set_index("group_id")["utility"].to_dict()
    rng = np.random.default_rng(seed if seed >= 0 else 0)
    selected: set[str] = set()
    for group_id, query_ids in sorted(by_group.items(), key=lambda item: len(item[1]), reverse=True):
        available = [query_id for query_id in query_ids if query_id in heldout_utility and query_id not in selected]
        rng.shuffle(available)
        for query_id in available[:pilot_per_group]:
            selected.add(str(query_id))
            if len(selected) >= budget:
                return selected
    global_mean = float(np.nanmean(list(heldout_utility.values()))) if heldout_utility else 0.0
    global_std = max(float(np.nanstd(list(heldout_utility.values()))) if heldout_utility else 0.0, 0.05)
    while len(selected) < budget:
        best_group = None
        best_score = -np.inf
        for group_id, query_ids in by_group.items():
            available = [query_id for query_id in query_ids if query_id in heldout_utility and query_id not in selected]
            if not available:
                continue
            values = np.asarray(
                [heldout_utility[query_id] for query_id in query_ids if query_id in selected and query_id in heldout_utility],
                dtype=float,
            )
            if len(values) == 0:
                mean = global_mean
                std = global_std
            elif len(values) == 1:
                mean = float(values.mean())
                std = global_std
            else:
                mean = float(values.mean())
                std = max(float(values.std(ddof=1)), 0.03)
            se = std / np.sqrt(max(len(values), 1))
            current_best = float(existing_best.get(group_id, 0.0))
            score = len(query_ids) * max(0.0, mean + 1.96 * se - current_best)
            if score > best_score:
                best_score = score
                best_group = group_id
        if best_group is None:
            break
        choices = [query_id for query_id in by_group[best_group] if query_id in heldout_utility and query_id not in selected]
        if not choices:
            break
        selected.add(str(rng.choice(choices)))
    return selected


def mean_for(frame: pd.DataFrame, method: str) -> float:
    rows = frame[frame["method"].astype(str).eq(method)]
    if rows.empty:
        return float("-inf")
    return float(rows["mean_utility"].mean())


def write_live_memo(
    path: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    coverage: pd.DataFrame,
    selected_strata: str,
    selected_onboarding: str,
    variance: pd.DataFrame,
    validation_estimation: pd.DataFrame,
    active_validation: pd.DataFrame,
    frontier_validation: pd.DataFrame,
    frontier_test: pd.DataFrame,
    frontier_budget_efficiency: pd.DataFrame,
    selected_frontier_budget: int | None,
    onboarding: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    test = variance[variance["split"].astype(str).eq("test")].sort_values("weighted_utility_variance")
    max_budget = int(onboarding[onboarding["budget"].ge(0)]["budget"].max()) if not onboarding.empty else -1
    onboarding_summary = (
        onboarding[onboarding["budget"].eq(max_budget)]
        .groupby("method", as_index=False)
        .agg(mean_utility=("mean_utility", "mean"), mean_quality=("mean_quality", "mean"))
        .sort_values("mean_utility", ascending=False)
        if max_budget >= 0
        else pd.DataFrame()
    )
    lines = [
        "# Live Broad100 Predicted Utility-State Calibration",
        "",
        "This experiment reruns the predicted utility-state calibration/onboarding checks on the live Stage0 outcome matrix.",
        "",
        "## Commands",
        "",
        "- `PYTHONPATH=src python experiments/241_phase3_live_predicted_utility_state_calibration.py --config configs/probecode_final_eval.yaml`",
        "",
        "## Inputs",
        "",
        f"- Live outputs: `{args.live_outputs}`",
        f"- Probe/split features: `{args.features}`",
        f"- Lambda cost: `{float(config['method']['lambda_cost']):.2f}`",
        "",
        "## Live Model Coverage",
        "",
    ]
    for row in coverage.to_dict("records"):
        marker = "selected" if bool(row.get("selected_for_live_claim_eval", False)) else "not selected"
        lines.append(
            f"- `{row['model_id']}`: {marker}, coverage `{float(row['coverage_rate']):.4f}`, "
            f"quality `{float(row['mean_quality']):.4f}`, cost `${float(row['total_cost_usd']):.4f}`"
        )
    lines.extend(
        [
            "",
            f"Selected strata method: `{selected_strata}`.",
            f"Selected onboarding method: `{selected_onboarding}`.",
            "",
            "## Held-Out Test Strata Variance",
            "",
        ]
    )
    for row in test.head(10).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: utility variance `{float(row['weighted_utility_variance']):.4f}`"
        )
    lines.extend(["", "## Validation Onboarding-State Selection", ""])
    for row in validation_estimation[
        validation_estimation["group_method"].astype(str).str.startswith("predicted_utility_state_")
    ].head(8).to_dict("records"):
        lines.append(
            f"- `{row['group_method']}`: train-to-val utility estimation error "
            f"`{float(row['weighted_abs_utility_estimation_error']):.4f}`"
        )
    if not active_validation.empty:
        lines.extend(["", "## Validation Active-Acquisition Selection", ""])
        for row in active_validation[active_validation["is_selected"].astype(bool)].sort_values("budget").to_dict("records"):
            lines.append(
                f"- Budget `{int(row['budget'])}`: `{row['selected_acquisition']}` selected with validation utility "
                f"`{float(row['validation_mean_utility']):.4f}`"
            )
    if selected_frontier_budget is not None and not frontier_validation.empty:
        lines.extend(["", "## Frontier Onboarding Slice", ""])
        lines.append(
            "This slice treats GPT and Gemini as held-out new models and selects the calibration budget using validation frontier utility margin."
        )
        lines.append("")
        lines.append(f"Validation-selected frontier budget: `{selected_frontier_budget}`.")
        lines.append("")
        for row in frontier_validation.sort_values("active_minus_best_competitor", ascending=False).head(5).to_dict("records"):
            lines.append(
                f"- Validation budget `{int(row['budget'])}`: active `{float(row['active_mean_utility']):.4f}`, "
                f"best competitor `{float(row['best_competitor_mean_utility']):.4f}`, "
                f"margin `{float(row['active_minus_best_competitor']):.4f}`"
            )
        test_row = frontier_test[frontier_test["budget"].eq(int(selected_frontier_budget))]
        if not test_row.empty:
            row = test_row.iloc[0]
            lines.append("")
            lines.append(
                f"Test at selected budget `{selected_frontier_budget}`: active `{float(row['active_mean_utility']):.4f}`, "
                f"best competitor `{float(row['best_competitor_mean_utility']):.4f}`, "
                f"margin `{float(row['active_minus_best_competitor']):.4f}`."
            )
        if not frontier_budget_efficiency.empty:
            lines.append("")
            lines.append("Budget-to-match summary:")
            for row in frontier_budget_efficiency.to_dict("records"):
                match_text = (
                    f"matches at `{int(row['competitor_match_budget'])}` evals"
                    if bool(row["matched_active_utility"])
                    else f"does not match by `{int(row['competitor_best_tested_budget'])}` evals"
                )
                lines.append(
                    f"- `{row['competitor']}` {match_text}; active uses `{int(row['active_budget'])}` evals; "
                    f"reduction lower bound `{float(row['eval_reduction_lower_bound']):.1f}x`"
                )
    lines.extend(["", f"## Onboarding At Budget {max_budget}", ""])
    for row in onboarding_summary.to_dict("records"):
        lines.append(
            f"- `{row['method']}`: utility `{float(row['mean_utility']):.4f}`, "
            f"quality `{float(row['mean_quality']):.4f}`"
        )
    lines.extend(["", "## Claim Status", ""])
    for row in claims.to_dict("records"):
        lines.append(f"- `{row['claim_id']}`: `{row['status']}`; {row['evidence']}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is live/cached Stage0 evidence, not a new uncached full provider run.",
            "- Gemini fresh retry hit rate limits in the separate provider-readiness run; the Stage0 Gemini rows here are reused from the existing cache.",
            "- Active acquisition is reported separately because state-based calibration can succeed even if the acquisition rule still needs improvement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
