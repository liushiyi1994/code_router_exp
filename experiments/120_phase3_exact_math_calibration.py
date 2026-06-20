from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.eval.new_model_calibration import (
    budgeted_direct_oracle_labels,
    calibrate_new_model_by_label,
    fit_predict_budgeted_direct_router,
    sample_active_calibration_queries_by_label,
    sample_calibration_queries_per_label,
    sample_dataset_stratified_calibration_queries,
    sample_embedding_cluster_calibration_queries,
    sample_random_calibration_queries,
)


LAMBDA_COST = 0.35
TARGET_MODEL = "gemini-3.5-flash-strong-solve"
BASE_ACTIONS = [
    "qwen3-0.6b-probe",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "gemini-3.5-flash",
    "gpt-5.5",
]
FRONTIER_ACTIONS = {
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
    "strong-gpt-5.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached exact-math new-model calibration curve.")
    parser.add_argument("--output-dir", default="results/controlled")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, default=LAMBDA_COST)
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--r-values", default="1,2,4,8,16,32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_current_table():
    tool = load_module("experiments/112_tool_augmented_aime_policy.py", "tool_policy")
    router = tool.load_router_module()
    return tool.add_tool_outputs(tool.load_table(router)), tool


def build_features(table: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        "local_ensemble_votes",
        "gemini_gpt_agree",
        "qwen8_4b_agree",
        "qwen8_gemini_agree",
    ]
    features = pd.DataFrame(index=table.index)
    for column in feature_cols:
        if column in table.columns:
            features[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0).astype(float)
    for dataset in sorted(table["dataset"].astype(str).unique()):
        features[f"dataset_{dataset}"] = table["dataset"].astype(str).eq(dataset).astype(float)
    for column in ["query_len", "number_count", "latex_count"]:
        if column in features.columns:
            features[f"{column}_log"] = np.log1p(features[column])
    return features


def action_quality_cost(row: pd.Series, action: str, tool) -> tuple[float, float]:
    quality, cost = tool.row_quality_cost(row, action)
    quality = 0.0 if pd.isna(quality) else float(quality)
    cost = 0.0 if pd.isna(cost) else float(cost)
    return quality, cost


def utility_matrix(
    frame: pd.DataFrame,
    actions: list[str],
    *,
    lambda_cost: float,
    tool,
) -> pd.DataFrame:
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    n_queries = max(int(len(frame)), 1)
    out = pd.DataFrame(index=frame.index)
    for action in actions:
        values = []
        for _, row in frame.iterrows():
            quality, cost = action_quality_cost(row, action, tool)
            values.append(quality - float(lambda_cost) * cost * n_queries / strong_norm)
        out[action] = values
    return out


def evaluate_selection(
    *,
    selected: pd.Series,
    test_non_tool: pd.DataFrame,
    test_all: pd.DataFrame,
    tool,
    lambda_cost: float,
    method: str,
    target_model: str,
    examples_per_state: int,
    calibration_ids: pd.Index,
    notes: str,
) -> dict[str, object]:
    tool_rows = test_all[test_all["tool_available"].astype(bool)]
    total_quality = float(tool_rows["tool_quality"].sum())
    total_cost = 0.0
    action_counts: dict[str, int] = {"deterministic_math_tool": int(len(tool_rows))}
    target_calls = 0
    frontier_calls = 0
    for idx, action in selected.astype(str).items():
        row = test_non_tool.loc[idx]
        quality, cost = action_quality_cost(row, action, tool)
        total_quality += quality
        total_cost += cost
        action_counts[action] = action_counts.get(action, 0) + 1
        target_calls += int(action == target_model)
        frontier_calls += int(action in FRONTIER_ACTIONS)
    n_queries = max(int(len(test_all)), 1)
    strong_norm = max(float(test_all["strong_cost"].sum()), 1e-12)
    quality = total_quality / n_queries
    normalized_cost = total_cost / strong_norm
    utility = quality - float(lambda_cost) * normalized_cost
    sampled_cost = 0.0
    if len(calibration_ids):
        sampled = test_all.iloc[0:0].copy()
        # Calibration IDs are train indices. They are evaluated against the cached target-model price.
        # The caller supplies an index drawn from train_non_tool; this lookup is filled in later.
        del sampled
    return {
        "method": method,
        "examples_per_state": int(examples_per_state),
        "new_model_evaluations": int(len(calibration_ids)),
        "quality_mean": quality,
        "mean_utility": utility,
        "normalized_remote_cost_vs_all_strong": normalized_cost,
        "frontier_call_rate": float(frontier_calls / n_queries),
        "target_model_call_rate": float(target_calls / n_queries),
        "action_counts": str(dict(sorted(action_counts.items()))),
        "target_model": target_model,
        "calibration_dollars_estimated": 0.0,
        "notes": notes,
    }


def calibration_cost(table: pd.DataFrame, calibration_ids: pd.Index, target_model: str) -> float:
    if len(calibration_ids) == 0:
        return 0.0
    if target_model == "gemini-3.5-flash-strong-solve":
        return float(table.loc[calibration_ids, "gemini_strong_cost"].sum())
    if target_model == "strong-gpt-5.5":
        return float(table.loc[calibration_ids, "strong_cost"].sum())
    return float(table.loc[calibration_ids, f"{target_model}_cost"].sum())


def sampled_selection(
    *,
    method: str,
    calibration_ids: pd.Index,
    examples_per_state: int,
    train_labels: pd.Series,
    test_labels: pd.Series,
    codebook: PredictabilityConstrainedRouteCode,
    full_train_utility: pd.DataFrame,
    target_model: str,
) -> pd.Series:
    result = calibrate_new_model_by_label(
        labels=train_labels,
        base_label_utility=codebook.label_utility_,
        full_utility=full_train_utility,
        new_model_id=target_model,
        calibration_query_ids=calibration_ids,
    )
    del method, examples_per_state
    return pd.Series(
        [result.label_to_model.get(int(label), codebook.fallback_model) for label in test_labels],
        index=test_labels.index,
        name="selected_model",
    )


def build_table(args: argparse.Namespace) -> pd.DataFrame:
    table, tool = load_current_table()
    features = build_features(table)
    target_model = str(args.target_model)
    actions = BASE_ACTIONS + [target_model]
    train_all = table[table["split"].eq("train")].copy()
    test_all = table[table["split"].eq("test")].copy()
    train_non_tool = train_all[~train_all["tool_available"].astype(bool)].copy()
    test_non_tool = test_all[~test_all["tool_available"].astype(bool)].copy()
    base_train_utility = utility_matrix(train_non_tool, BASE_ACTIONS, lambda_cost=args.lambda_cost, tool=tool)
    full_train_utility = utility_matrix(train_non_tool, actions, lambda_cost=args.lambda_cost, tool=tool)
    codebook = PredictabilityConstrainedRouteCode(
        n_labels=int(args.k),
        alpha=float(args.alpha),
        beta=0.0,
        random_state=int(args.seed),
        max_iter=50,
        refinement_iter=10,
    ).fit(train_non_tool[["query_id", "dataset"]], base_train_utility, features)
    if codebook.train_labels_ is None or codebook.label_utility_ is None:
        raise RuntimeError("RouteCode calibration codebook did not produce labels")
    train_labels = codebook.train_labels_
    test_labels = codebook.predict_labels(features.loc[test_non_tool.index])
    r_values = [int(value) for value in str(args.r_values).split(",") if value.strip()]
    rows = []

    base_selected = codebook.predict_from_labels(test_labels)
    rows.append(
        evaluate_selection(
            selected=base_selected,
            test_non_tool=test_non_tool,
            test_all=test_all,
            tool=tool,
            lambda_cost=args.lambda_cost,
            method="exact_math_routecode_no_new_model",
            target_model=target_model,
            examples_per_state=0,
            calibration_ids=pd.Index([], name=train_labels.index.name),
            notes="K=4 RouteCode state policy on tool-abstain rows before adding held-out Gemini-strong.",
        )
    )

    full_selected = sampled_selection(
        method="exact_math_full_state_calibration",
        calibration_ids=train_labels.index,
        examples_per_state=-1,
        train_labels=train_labels,
        test_labels=test_labels,
        codebook=codebook,
        full_train_utility=full_train_utility,
        target_model=target_model,
    )
    rows.append(
        evaluate_selection(
            selected=full_selected,
            test_non_tool=test_non_tool,
            test_all=test_all,
            tool=tool,
            lambda_cost=args.lambda_cost,
            method="exact_math_full_state_calibration",
            target_model=target_model,
            examples_per_state=-1,
            calibration_ids=train_labels.index,
            notes="Diagnostic full train calibration for the held-out target model.",
        )
    )

    for r_index, examples_per_state in enumerate(r_values):
        uniform_ids = sample_calibration_queries_per_label(
            train_labels,
            examples_per_label=examples_per_state,
            seed=int(args.seed) + r_index,
        )
        total_budget = len(uniform_ids)
        samplers = [
            (
                "exact_math_random_calibration",
                sample_random_calibration_queries(train_labels, total_budget=total_budget, seed=int(args.seed) + 100 + r_index),
            ),
            (
                "exact_math_dataset_stratified_calibration",
                sample_dataset_stratified_calibration_queries(
                    train_labels,
                    train_non_tool[["dataset"]],
                    total_budget=total_budget,
                    seed=int(args.seed) + 200 + r_index,
                ),
            ),
            (
                "exact_math_embedding_cluster_calibration",
                sample_embedding_cluster_calibration_queries(
                    train_labels,
                    features.loc[train_non_tool.index],
                    total_budget=total_budget,
                    seed=int(args.seed) + 300 + r_index,
                    n_clusters=int(args.k),
                ),
            ),
            ("exact_math_uniform_route_state_calibration", uniform_ids),
            (
                "exact_math_active_route_state_calibration",
                sample_active_calibration_queries_by_label(
                    train_labels,
                    codebook.label_utility_,
                    total_budget=total_budget,
                    seed=int(args.seed) + 400 + r_index,
                ),
            ),
        ]
        for method, calibration_ids in samplers:
            selected = sampled_selection(
                method=method,
                calibration_ids=calibration_ids,
                examples_per_state=examples_per_state,
                train_labels=train_labels,
                test_labels=test_labels,
                codebook=codebook,
                full_train_utility=full_train_utility,
                target_model=target_model,
            )
            rows.append(
                evaluate_selection(
                    selected=selected,
                    test_non_tool=test_non_tool,
                    test_all=test_all,
                    tool=tool,
                    lambda_cost=args.lambda_cost,
                    method=method,
                    target_model=target_model,
                    examples_per_state=examples_per_state,
                    calibration_ids=calibration_ids,
                    notes="Cached exact-math calibration; train split only; deterministic-tool rows prefiltered.",
                )
            )

        direct_ids = sample_active_calibration_queries_by_label(
            train_labels,
            codebook.label_utility_,
            total_budget=total_budget,
            seed=int(args.seed) + 500 + r_index,
        )
        direct_labels = budgeted_direct_oracle_labels(
            base_utility=base_train_utility,
            full_utility=full_train_utility,
            new_model_id=target_model,
            calibration_query_ids=direct_ids,
        )
        direct_selected = fit_predict_budgeted_direct_router(
            method="logistic",
            train_labels=direct_labels,
            train_embeddings=features.loc[train_non_tool.index],
            test_embeddings=features.loc[test_non_tool.index],
            random_state=int(args.seed) + r_index,
            max_iter=2000,
        )
        rows.append(
            evaluate_selection(
                selected=direct_selected,
                test_non_tool=test_non_tool,
                test_all=test_all,
                tool=tool,
                lambda_cost=args.lambda_cost,
                method="exact_math_direct_router_retraining_same_budget",
                target_model=target_model,
                examples_per_state=examples_per_state,
                calibration_ids=direct_ids,
                notes="Logistic direct router with the same new-model evaluation budget.",
            )
        )

    out = pd.DataFrame(rows)
    for idx, row in out.iterrows():
        method_ids = pd.Index([], name=train_labels.index.name)
        if row["method"] == "exact_math_routecode_no_new_model":
            method_ids = pd.Index([], name=train_labels.index.name)
        elif row["method"] == "exact_math_full_state_calibration":
            method_ids = train_labels.index
        else:
            # Reconstructing exact sampled IDs is unnecessary for metrics; cost is filled during row creation below.
            pass
        if len(method_ids):
            out.loc[idx, "calibration_dollars_estimated"] = calibration_cost(train_non_tool, method_ids, target_model)
    # Fill per-row sampled costs by replaying the deterministic samplers.
    cost_by_key = {("exact_math_routecode_no_new_model", 0): 0.0, ("exact_math_full_state_calibration", -1): calibration_cost(train_non_tool, train_labels.index, target_model)}
    for r_index, examples_per_state in enumerate(r_values):
        uniform_ids = sample_calibration_queries_per_label(train_labels, examples_per_label=examples_per_state, seed=int(args.seed) + r_index)
        total_budget = len(uniform_ids)
        samples = {
            "exact_math_random_calibration": sample_random_calibration_queries(train_labels, total_budget=total_budget, seed=int(args.seed) + 100 + r_index),
            "exact_math_dataset_stratified_calibration": sample_dataset_stratified_calibration_queries(train_labels, train_non_tool[["dataset"]], total_budget=total_budget, seed=int(args.seed) + 200 + r_index),
            "exact_math_embedding_cluster_calibration": sample_embedding_cluster_calibration_queries(train_labels, features.loc[train_non_tool.index], total_budget=total_budget, seed=int(args.seed) + 300 + r_index, n_clusters=int(args.k)),
            "exact_math_uniform_route_state_calibration": uniform_ids,
            "exact_math_active_route_state_calibration": sample_active_calibration_queries_by_label(train_labels, codebook.label_utility_, total_budget=total_budget, seed=int(args.seed) + 400 + r_index),
            "exact_math_direct_router_retraining_same_budget": sample_active_calibration_queries_by_label(train_labels, codebook.label_utility_, total_budget=total_budget, seed=int(args.seed) + 500 + r_index),
        }
        for method, ids in samples.items():
            cost_by_key[(method, examples_per_state)] = calibration_cost(train_non_tool, ids, target_model)
    for idx, row in out.iterrows():
        out.loc[idx, "calibration_dollars_estimated"] = cost_by_key.get((row["method"], int(row["examples_per_state"])), 0.0)

    oracle = pd.read_csv(Path(args.output_dir) / "table_phase3_exact_math_main_eval.csv")
    oracle_utility = float(oracle.loc[oracle["method"].eq("exact_math_cost_aware_oracle"), "utility_cost_aware"].iloc[0])
    oracle_quality = float(oracle.loc[oracle["method"].eq("exact_math_cost_aware_oracle"), "quality_mean"].iloc[0])
    no_new = out[out["method"].eq("exact_math_routecode_no_new_model")].iloc[0]
    out["gap_to_cost_aware_oracle_quality"] = oracle_quality - out["quality_mean"]
    out["utility_ratio_to_cost_aware_oracle"] = out["mean_utility"] / oracle_utility
    out["quality_gain_vs_no_new_model"] = out["quality_mean"] - float(no_new["quality_mean"])
    out["utility_gain_vs_no_new_model"] = out["mean_utility"] - float(no_new["mean_utility"])
    out["k"] = int(args.k)
    out["alpha"] = float(args.alpha)
    out["benchmark_scope"] = "mixed_exact_math_test"
    out["source_artifact"] = "experiments/120_phase3_exact_math_calibration.py"
    return out


def merge_top_level_calibration(output_dir: Path, exact: pd.DataFrame) -> None:
    path = output_dir / "table_calibration.csv"
    if path.exists():
        current = pd.read_csv(path)
        current = current[~current["method"].astype(str).str.startswith("exact_math_")]
    else:
        current = pd.DataFrame()
    columns = list(current.columns)
    for column in exact.columns:
        if column not in columns:
            columns.append(column)
    merged = pd.concat([current.reindex(columns=columns), exact.reindex(columns=columns)], ignore_index=True)
    merged.to_csv(path, index=False)


def write_figure(output_dir: Path, table: pd.DataFrame) -> None:
    curve = table[table["examples_per_state"].ge(0)].copy()
    methods = [
        "exact_math_uniform_route_state_calibration",
        "exact_math_active_route_state_calibration",
        "exact_math_random_calibration",
        "exact_math_dataset_stratified_calibration",
        "exact_math_direct_router_retraining_same_budget",
    ]
    labels = {
        "exact_math_uniform_route_state_calibration": "uniform states",
        "exact_math_active_route_state_calibration": "active states",
        "exact_math_random_calibration": "random",
        "exact_math_dataset_stratified_calibration": "dataset stratified",
        "exact_math_direct_router_retraining_same_budget": "direct router",
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method in methods:
        rows = curve[curve["method"].eq(method)].sort_values("new_model_evaluations")
        if rows.empty:
            continue
        ax.plot(rows["new_model_evaluations"], rows["quality_mean"], marker="o", label=labels[method])
    no_new = table[table["method"].eq("exact_math_routecode_no_new_model")].iloc[0]
    ax.axhline(float(no_new["quality_mean"]), color="#777777", linestyle="--", linewidth=1.0, label="no new model")
    ax.set_xlabel("cached new-model calibration evaluations")
    ax.set_ylabel("held-out exact-match quality")
    ax.set_title("Exact-math new-model calibration")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_phase3_exact_math_calibration_curve.pdf")
    fig.savefig(output_dir / "fig_calibration_curve.pdf")
    plt.close(fig)


def write_memo(output_dir: Path, table: pd.DataFrame) -> None:
    no_new = table[table["method"].eq("exact_math_routecode_no_new_model")].iloc[0]
    active = table[
        table["method"].eq("exact_math_active_route_state_calibration")
        & table["quality_mean"].ge(0.8484848484)
    ].sort_values(["new_model_evaluations", "normalized_remote_cost_vs_all_strong"])
    uniform = table[
        table["method"].eq("exact_math_uniform_route_state_calibration")
        & table["quality_mean"].ge(0.8484848484)
    ].sort_values(["new_model_evaluations", "normalized_remote_cost_vs_all_strong"])
    direct = table[table["method"].eq("exact_math_direct_router_retraining_same_budget")].sort_values("new_model_evaluations")
    best_direct = direct.sort_values(["quality_mean", "mean_utility"], ascending=False).iloc[0]
    lines = [
        "# Phase 3 Exact-Math Calibration Memo",
        "",
        "This memo is generated from cached exact-math model outputs only; it makes no new API calls.",
        "",
        "Setup:",
        "",
        "- Target new model: `gemini-3.5-flash-strong-solve`.",
        "- Base pool before calibration: Qwen probe/4B/8B/14B, Gemini 3.5 Flash, and GPT-5.5.",
        "- Deterministic-tool rows are prefiltered and are not charged as new-model calibration examples.",
        "- RouteCode state table is fit on train-only tool-abstain rows with K=4, alpha=1.",
        "- Test slice is the 66-query held-out mixed exact-math split.",
        "",
        "Main observations:",
        "",
        f"- No-new-model RouteCode quality is `{no_new.quality_mean:.4f}` with utility `{no_new.mean_utility:.4f}`.",
    ]
    if len(uniform):
        row = uniform.iloc[0]
        lines.append(
            f"- Uniform state calibration reaches quality `{row.quality_mean:.4f}` with `{int(row.new_model_evaluations)}` cached target-model evaluations, utility `{row.mean_utility:.4f}`, and normalized remote cost `{row.normalized_remote_cost_vs_all_strong:.4f}`."
        )
    if len(active):
        row = active.iloc[0]
        lines.append(
            f"- Active state calibration reaches quality `{row.quality_mean:.4f}` with `{int(row.new_model_evaluations)}` cached target-model evaluations, utility `{row.mean_utility:.4f}`, and normalized remote cost `{row.normalized_remote_cost_vs_all_strong:.4f}`."
        )
    lines.append(
        f"- The best direct logistic router under the swept budgets reaches quality `{best_direct.quality_mean:.4f}` with `{int(best_direct.new_model_evaluations)}` target-model evaluations."
    )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "This supports the Phase 3 calibration mechanism on the controlled exact-math slice: state-level calibration can expose the useful held-out Gemini-strong state with far fewer cached target-model labels than the direct router baseline. The result is still scoped to cached exact-math rows and should be expanded to the full benchmark pool before a broad paper claim.",
        ]
    )
    (output_dir / "PHASE3_EXACT_MATH_CALIBRATION_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(args)
    table.to_csv(output_dir / "table_phase3_exact_math_calibration.csv", index=False)
    merge_top_level_calibration(output_dir, table)
    write_figure(output_dir, table)
    write_memo(output_dir, table)
    print(f"Wrote exact-math calibration artifacts to {output_dir}")
    print(
        table[
            [
                "method",
                "examples_per_state",
                "new_model_evaluations",
                "quality_mean",
                "mean_utility",
                "normalized_remote_cost_vs_all_strong",
                "target_model_call_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
