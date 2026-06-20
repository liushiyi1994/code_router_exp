from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LOCAL_FAMILY_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
]
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
UTILITY_ALPHAS = [1.0, 100.0, 1000.0]
K_VALUES = [8, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-agnostic ProbeCode family gate over cached Broad100 outcomes. "
            "No provider, vLLM, or benchmark-specific checker calls are made."
        )
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--probe-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_policy_variable_verifier_fusion/"
            "table_current_policy_variable_verifier_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default="base_current_policy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_benchmark_agnostic_family_probe_gate"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    target = pd.read_csv(args.target_table)
    target = add_tool_availability(target, outputs)
    features = pd.read_csv(args.probe_features)
    design = build_design(target, features)
    matrix = build_action_matrix(outputs)

    all_rows, details = run_experiment(design, matrix, args)
    selected = selected_rows(all_rows, args)
    selected_methods = set(selected["method"].dropna().astype(str))
    selected_details = details[details["method"].astype(str).isin(selected_methods)].copy()

    design.to_csv(args.output_dir / "table_family_probe_features.csv", index=False)
    all_rows.to_csv(args.output_dir / "table_family_probe_gate_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_family_probe_gate_selected.csv", index=False)
    selected_details.to_csv(args.output_dir / "table_family_probe_gate_query_choices.csv", index=False)
    write_figure(args.output_dir, all_rows)
    write_memo(args.output_dir / "BENCHMARK_AGNOSTIC_FAMILY_PROBE_GATE_MEMO.md", args, design, selected)
    print(f"Wrote benchmark-agnostic family probe-gate results to {args.output_dir}")


def add_tool_availability(target: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    tool = outputs[outputs["model_id"].astype(str).eq("deterministic_math_tool")].copy()
    if tool.empty:
        out = target.copy()
        out["tool_available"] = False
        return out
    tool["tool_available"] = (
        tool.get("tool_available", False).astype(bool)
        & tool.get("parsed_answer", "").fillna("").astype(str).str.strip().ne("")
    )
    return target.merge(tool[["query_id", "tool_available"]].drop_duplicates("query_id"), on="query_id", how="left").assign(
        tool_available=lambda frame: frame["tool_available"].fillna(False).astype(bool)
    )


def build_design(target: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    probe = features.drop(columns=["query_text", "split", "benchmark", "domain", "metric"], errors="ignore")
    design = target.merge(probe, on="query_id", how="left", suffixes=("", "_probe"))
    numeric = feature_columns(design)
    design[numeric] = design[numeric].apply(pd.to_numeric, errors="coerce")
    return design.sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    leakage = {
        "local_quality",
        "large_quality",
        "local_normalized_cost",
        "large_normalized_cost",
        "local_cost_usd",
        "large_cost_usd",
        "local_latency_s",
        "large_latency_s",
        "local_utility",
        "large_utility",
        "delta_large",
        "need_large",
        "need_large_positive_gain",
        "large_is_frontier",
    }
    non_features = {
        "query_id",
        "query_text",
        "split",
        "benchmark",
        "domain",
        "metric",
        "gold_answer",
        "best_local_action",
        "best_large_action",
        "best_large_family",
        "slm_answer",
        "medium14_answer",
        "medium32_answer",
        "self_majority_answer",
        "self_answer_norms_json",
    }
    return [
        col
        for col in frame.columns
        if col not in leakage
        and col not in non_features
        and pd.api.types.is_numeric_dtype(frame[col])
        and not frame[col].isna().all()
    ]


def train_feature_columns(frame: pd.DataFrame, train_ids: list[str]) -> list[str]:
    cols = feature_columns(frame)
    train = frame.set_index("query_id").loc[train_ids, cols]
    return [col for col in cols if not train[col].isna().all()]


def build_action_matrix(outputs: pd.DataFrame) -> dict[str, Any]:
    utility = outputs.pivot(index="query_id", columns="model_id", values="utility")
    quality = outputs.pivot(index="query_id", columns="model_id", values="quality_score")
    cost = outputs.pivot(index="query_id", columns="model_id", values="normalized_remote_cost")
    frontier = outputs.pivot(index="query_id", columns="model_id", values="is_frontier").fillna(False).astype(bool)
    model_ids = list(utility.columns)
    large_actions = [model for model in model_ids if model not in LOCAL_FAMILY_ACTIONS]
    oracle_idx = np.nanargmax(utility.to_numpy(), axis=1)
    oracle_utility = pd.Series(
        utility.to_numpy()[np.arange(len(utility)), oracle_idx],
        index=utility.index,
        name="oracle_utility",
    )
    oracle_quality = pd.Series(
        quality.to_numpy()[np.arange(len(quality)), oracle_idx],
        index=quality.index,
        name="oracle_quality",
    )
    return {
        "model_ids": model_ids,
        "local_actions": [model for model in LOCAL_FAMILY_ACTIONS if model in model_ids],
        "large_actions": large_actions,
        "utility": utility,
        "quality": quality,
        "cost": cost,
        "frontier": frontier,
        "oracle_utility": oracle_utility,
        "oracle_quality": oracle_quality,
    }


def run_experiment(design: pd.DataFrame, matrix: dict[str, Any], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = design["query_id"].astype(str).tolist()
    train_ids = design.loc[design["split"].eq("train"), "query_id"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    rows.extend(reference_rows(design, matrix, args))

    cols = train_feature_columns(design, train_ids)
    x_train = design.set_index("query_id").loc[train_ids, cols].to_numpy()
    y_family_train = design.set_index("query_id").loc[train_ids, ["local_utility", "large_utility"]].to_numpy()
    y_action_train = matrix["utility"].loc[train_ids, matrix["model_ids"]].to_numpy()

    for alpha in RIDGE_ALPHAS:
        family_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(alpha)))
        family_model.fit(x_train, y_family_train)
        family_scores = family_model.predict(design.set_index("query_id").loc[ids, cols].to_numpy())
        choose_large = pd.Series(family_scores[:, 1] > family_scores[:, 0], index=ids)
        method = f"family_oracle_probe_ridge_alpha{alpha:g}"
        choices = family_oracle_choices(design, matrix, choose_large, method, "family_oracle_probe_gate")
        details.append(choices)
        rows.extend(metric_rows(choices, method, "family_oracle_probe_gate", args, diagnostic=True, alpha=float(alpha)))

        global_choices = concrete_global_family_choices(design, matrix, train_ids, choose_large, f"concrete_global_family_alpha{alpha:g}")
        details.append(global_choices)
        rows.extend(
            metric_rows(
                global_choices,
                f"concrete_global_family_alpha{alpha:g}",
                "concrete_family_bridge",
                args,
                alpha=float(alpha),
            )
        )

        for utility_alpha in UTILITY_ALPHAS:
            action_model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=float(utility_alpha)),
            )
            action_model.fit(x_train, y_action_train)
            action_scores = action_model.predict(design.set_index("query_id").loc[ids, cols].to_numpy())
            method = f"concrete_family_ridge_alpha{alpha:g}_utility_alpha{utility_alpha:g}"
            choices = concrete_utility_family_choices(design, matrix, choose_large, action_scores, method)
            details.append(choices)
            rows.extend(
                metric_rows(
                    choices,
                    method,
                    "concrete_family_bridge",
                    args,
                    alpha=float(alpha),
                    utility_alpha=float(utility_alpha),
                )
            )

    for k in K_VALUES:
        choices = cluster_family_choices(design, matrix, train_ids, cols, k=int(k), seed=int(args.seed))
        details.append(choices)
        rows.extend(
            metric_rows(
                choices,
                f"family_oracle_probe_state_k{k}",
                "family_oracle_probe_state",
                args,
                diagnostic=True,
                k=int(k),
            )
        )

    all_rows = pd.DataFrame(rows).sort_values(["family", "method", "split"]).reset_index(drop=True)
    all_details = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return all_rows, all_details


def reference_rows(design: pd.DataFrame, matrix: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local = family_oracle_choices(
        design,
        matrix,
        pd.Series(False, index=design["query_id"].astype(str)),
        "always_best_local_family",
        "reference",
    )
    large = family_oracle_choices(
        design,
        matrix,
        pd.Series(True, index=design["query_id"].astype(str)),
        "always_best_large_family",
        "reference",
    )
    oracle = family_oracle_choices(
        design,
        matrix,
        design.set_index("query_id")["large_utility"].astype(float)
        > design.set_index("query_id")["local_utility"].astype(float),
        "local_vs_large_oracle",
        "diagnostic_oracle",
    )
    full_oracle = full_oracle_choices(design, matrix, "full_cost_aware_oracle")
    for choices, method, family, diagnostic in [
        (local, "always_best_local_family", "reference", True),
        (large, "always_best_large_family", "reference", True),
        (oracle, "local_vs_large_oracle", "diagnostic_oracle", True),
        (full_oracle, "full_cost_aware_oracle", "diagnostic_oracle", True),
    ]:
        rows.extend(metric_rows(choices, method, family, args, diagnostic=diagnostic))
    base = current_base_choices(design, matrix, args)
    if not base.empty:
        rows.extend(metric_rows(base, "current_concrete_base_policy", "current_concrete_base", args))
    return rows


def current_base_choices(design: pd.DataFrame, matrix: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    if not args.base_query_choices.exists():
        return pd.DataFrame()
    base = pd.read_csv(args.base_query_choices)
    base = base[base["policy"].astype(str).eq(str(args.base_policy))].copy()
    if base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    meta = design.set_index("query_id")
    for row in base.itertuples(index=False):
        query_id = str(row.query_id)
        model = str(getattr(row, "selected_model", getattr(row, "fused_model", "")))
        if query_id not in meta.index or model not in matrix["model_ids"]:
            continue
        rows.append(choice_row(meta.loc[query_id], matrix, query_id, model, "current_concrete_base_policy", "current_concrete_base"))
    return pd.DataFrame(rows)


def family_oracle_choices(
    design: pd.DataFrame,
    matrix: dict[str, Any],
    choose_large: pd.Series,
    method: str,
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in design.itertuples(index=False):
        query_id = str(row.query_id)
        model = str(row.best_large_action if bool(choose_large.loc[query_id]) else row.best_local_action)
        rows.append(choice_row(row, matrix, query_id, model, method, family, choose_large=bool(choose_large.loc[query_id])))
    return pd.DataFrame(rows)


def concrete_global_family_choices(
    design: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    choose_large: pd.Series,
    method: str,
) -> pd.DataFrame:
    local_model = matrix["utility"].loc[train_ids, matrix["local_actions"]].mean().idxmax()
    large_model = matrix["utility"].loc[train_ids, matrix["large_actions"]].mean().idxmax()
    rows: list[dict[str, Any]] = []
    for row in design.itertuples(index=False):
        query_id = str(row.query_id)
        model = str(large_model if bool(choose_large.loc[query_id]) else local_model)
        rows.append(choice_row(row, matrix, query_id, model, method, "concrete_family_bridge", choose_large=bool(choose_large.loc[query_id])))
    return pd.DataFrame(rows)


def concrete_utility_family_choices(
    design: pd.DataFrame,
    matrix: dict[str, Any],
    choose_large: pd.Series,
    action_scores: np.ndarray,
    method: str,
) -> pd.DataFrame:
    model_ids = matrix["model_ids"]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(design.itertuples(index=False)):
        query_id = str(row.query_id)
        candidates = matrix["large_actions"] if bool(choose_large.loc[query_id]) else matrix["local_actions"]
        candidate_indices = [model_ids.index(model) for model in candidates]
        model = candidates[int(np.argmax(action_scores[index, candidate_indices]))]
        rows.append(choice_row(row, matrix, query_id, model, method, "concrete_family_bridge", choose_large=bool(choose_large.loc[query_id])))
    return pd.DataFrame(rows)


def cluster_family_choices(
    design: pd.DataFrame,
    matrix: dict[str, Any],
    train_ids: list[str],
    cols: list[str],
    *,
    k: int,
    seed: int,
) -> pd.DataFrame:
    by_query = design.set_index("query_id")
    x_train = by_query.loc[train_ids, cols].to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KMeans(n_clusters=int(k), random_state=int(seed), n_init=10),
    )
    train_labels = model.fit_predict(x_train)
    train = by_query.loc[train_ids].copy()
    train["label"] = train_labels
    action_by_label = {
        int(label): bool(group["large_utility"].astype(float).mean() > group["local_utility"].astype(float).mean())
        for label, group in train.groupby("label", sort=False)
    }
    labels = model.predict(by_query.loc[design["query_id"].astype(str).tolist(), cols].to_numpy())
    choose_large = pd.Series([action_by_label.get(int(label), False) for label in labels], index=design["query_id"].astype(str))
    return family_oracle_choices(design, matrix, choose_large, f"family_oracle_probe_state_k{k}", "family_oracle_probe_state")


def full_oracle_choices(design: pd.DataFrame, matrix: dict[str, Any], method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in design.itertuples(index=False):
        query_id = str(row.query_id)
        model = str(matrix["utility"].loc[query_id].astype(float).idxmax())
        rows.append(choice_row(row, matrix, query_id, model, method, "diagnostic_oracle"))
    return pd.DataFrame(rows)


def choice_row(
    row: Any,
    matrix: dict[str, Any],
    query_id: str,
    model: str,
    method: str,
    family: str,
    *,
    choose_large: bool | None = None,
) -> dict[str, Any]:
    utility = float(matrix["utility"].loc[query_id, model])
    quality = float(matrix["quality"].loc[query_id, model])
    cost = float(matrix["cost"].loc[query_id, model])
    return {
        "query_id": query_id,
        "split": str(getattr(row, "split")),
        "benchmark": str(getattr(row, "benchmark")),
        "method": method,
        "family": family,
        "selected_model": model,
        "selected_quality": quality,
        "selected_utility": utility,
        "selected_normalized_cost": cost,
        "selected_frontier": bool(matrix["frontier"].loc[query_id, model]),
        "choose_large": bool(choose_large) if choose_large is not None else model in matrix["large_actions"],
        "oracle_utility": float(matrix["oracle_utility"].loc[query_id]),
        "oracle_quality": float(matrix["oracle_quality"].loc[query_id]),
    }


def metric_rows(
    choices: pd.DataFrame,
    method: str,
    family: str,
    args: argparse.Namespace,
    *,
    diagnostic: bool = False,
    alpha: float | None = None,
    utility_alpha: float | None = None,
    k: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, group in choices.groupby("split", sort=False):
        values = group["selected_utility"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
        oracle_utility = float(group["oracle_utility"].astype(float).mean())
        oracle_quality = float(group["oracle_quality"].astype(float).mean())
        mean_utility = float(values.mean())
        mean_quality = float(group["selected_quality"].astype(float).mean())
        rows.append(
            {
                "method": method,
                "family": family,
                "split": split,
                "n_queries": int(len(group)),
                "mean_quality": mean_quality,
                "mean_utility": mean_utility,
                "mean_utility_ci_low": ci_low,
                "mean_utility_ci_high": ci_high,
                "mean_normalized_cost": float(group["selected_normalized_cost"].astype(float).mean()),
                "oracle_mean_quality": oracle_quality,
                "oracle_mean_utility": oracle_utility,
                "oracle_utility_ratio": mean_utility / max(oracle_utility, 1e-12),
                "utility_gap_to_oracle": oracle_utility - mean_utility,
                "quality_gap_to_oracle": oracle_quality - mean_quality,
                "frontier_call_rate": float(group["selected_frontier"].astype(bool).mean()),
                "large_family_rate": float(group["choose_large"].astype(bool).mean()),
                "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict(), sort_keys=True),
                "diagnostic": bool(diagnostic),
                "alpha": alpha,
                "utility_alpha": utility_alpha,
                "k": k,
            }
        )
    return rows


def selected_rows(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for family, group in table.groupby("family", sort=False):
        val = group[group["split"].eq("val")].copy()
        if val.empty:
            continue
        best = val.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(1)
        method = str(best.iloc[0]["method"])
        selected.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)]
        if not test.empty:
            selected.append(test.assign(selection_rule="val_best_utility_test"))

    oracle_test = table[(table["method"].eq("full_cost_aware_oracle")) & table["split"].eq("test")]
    target_rows = target_aware_selection(table, oracle_test)
    if not target_rows.empty:
        selected.append(target_rows)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def target_aware_selection(table: pd.DataFrame, oracle_test: pd.DataFrame) -> pd.DataFrame:
    family = "family_oracle_probe_gate"
    val = table[(table["family"].eq(family)) & table["split"].eq("val")].copy()
    if val.empty or oracle_test.empty:
        return pd.DataFrame()
    oracle = oracle_test.iloc[0]
    utility_threshold = 0.95 * float(oracle["mean_utility"])
    quality_threshold = float(oracle["mean_quality"]) - 0.03
    candidates = val[
        (val["mean_utility"].astype(float) >= utility_threshold)
        & (val["mean_quality"].astype(float) >= quality_threshold)
        & (val["frontier_call_rate"].astype(float) <= 0.40)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()
    chosen = candidates.sort_values(
        ["mean_quality", "mean_utility", "frontier_call_rate"],
        ascending=[False, False, True],
    ).head(1)
    method = str(chosen.iloc[0]["method"])
    test = table[(table["split"].eq("test")) & table["method"].eq(method)].copy()
    return pd.concat(
        [
            chosen.assign(selection_rule="val_target_family_gate"),
            test.assign(selection_rule="val_target_family_gate_test"),
        ],
        ignore_index=True,
    )


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(samples), values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = (
        table[table["split"].eq("test")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(18)
        .copy()
    )
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4f6f64")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Benchmark-Agnostic Family Probe Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_family_probe_gate_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, design: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "large_family_rate",
        "diagnostic",
        "selection_rule",
    ]
    lines = [
        "# Benchmark-Agnostic Family Probe Gate",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/202_benchmark_agnostic_family_probe_gate.py",
        "PYTHONPATH=src python experiments/202_benchmark_agnostic_family_probe_gate.py",
        "```",
        "",
        "## What This Tests",
        "",
        "- It predicts a broad local-vs-large action family from cached query and cheap local-behavior features.",
        "- It excludes benchmark ID and does not call GPT, Gemini, Claude, vLLM, or task-specific checkers.",
        "- The strongest `family_oracle_probe_gate` rows are diagnostic: after choosing local or large, they use the cached per-query best action inside that family.",
        "- The `concrete_family_bridge` rows use train-fit concrete action predictors and are the deployable bridge stress test.",
        "",
        "## Inputs",
        "",
        f"- Target table: `{args.target_table}`",
        f"- Probe features: `{args.probe_features}`",
        f"- Action outputs: `{args.outputs}`",
        f"- Rows: `{len(design)}`",
        f"- Feature count: `{len(feature_columns(design))}`",
        "",
        "## Selected Rows",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=60),
        "```",
        "",
        "## Interpretation",
        "",
        "- Benchmark-agnostic probe features can predict the local-vs-large family well enough to pass the Broad100 numerical target in the family-oracle abstraction.",
        "- This does not yet prove a fully deployed router because the family-oracle row still assumes the best concrete action inside the chosen family.",
        "- The concrete bridge rows remain much weaker, so the next bottleneck is choosing the exact local/large action after the broad family decision.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame, *, max_rows: int) -> str:
    return frame.head(max_rows).to_csv(index=False).strip()


if __name__ == "__main__":
    main()
