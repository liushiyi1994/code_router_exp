from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOCAL_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
LOCAL_LLM_ACTIONS = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
STRONG_OR_FRONTIER = {
    "gpt-5.5",
    "gemini-3.5-flash",
    "gemini-3.5-flash-strong-solve",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-safe gain gate over cached local consensus features.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--benchmark-composed-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_tool_aware_benchmark_composed_policy/"
            "table_tool_aware_benchmark_composed_choices.csv"
        ),
    )
    parser.add_argument("--benchmark-composed-method", default="tool_aware_benchmark_composed_eps0.01_recall_then_quality")
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_local_safe_gain_gate"))
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_183")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_183")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_183")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_183")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    rows_by_query = exp177.rows_by_query_map(outputs)
    base_choices = build_base_choices(exp177, exp172, exp175, outputs, target, rows_by_query)
    feature_frame = build_local_safe_features(base_choices, target, outputs, rows_by_query)
    policy_table, query_choices = evaluate_gain_gates(
        feature_frame,
        outputs,
        target,
        exp172,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = selected_rows(policy_table, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))

    feature_frame.to_csv(args.output_dir / "table_local_safe_gain_features.csv", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_local_safe_gain_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_local_safe_gain_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_local_safe_gain_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "LOCAL_SAFE_GAIN_GATE_MEMO.md", args, feature_frame, policy_table, selected)
    print(f"Wrote local-safe gain gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_base_choices(exp177, exp172, exp175, outputs: pd.DataFrame, target: pd.DataFrame, rows_by_query: dict[str, Any]) -> pd.DataFrame:
    priors = exp172.fit_train_priors(outputs)
    feature_frame, cat_cols, num_cols = exp177.build_feature_frame(outputs, target)
    config = exp177.RankerConfig("hgb_l1", "gate_rank_localplus", 0.25)
    pipe = exp177.make_pipeline(config.model_name, cat_cols, num_cols)
    train = feature_frame[feature_frame["split"].astype(str).eq("train")]
    pipe.fit(train[cat_cols + num_cols], train["quality_score"].astype(float))
    frames: list[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        candidates = feature_frame[feature_frame["split"].astype(str).eq(split)].copy()
        candidates = exp177.add_predictions(candidates, pipe, config.cost_penalty, cat_cols, num_cols)
        frame = target[target["split"].astype(str).eq(split)].copy()
        selected = exp177.select_actions(frame, candidates, config, rows_by_query, priors, exp172=exp172, exp175=exp175)
        selected["split"] = split
        frames.append(selected.rename(columns={"model_id": "base_model_id"}))
    return pd.concat(frames, ignore_index=True)


def build_local_safe_features(
    base_choices: pd.DataFrame,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    train_outputs = outputs[outputs["split"].astype(str).eq("train")]
    benchmark_model_prior = train_outputs.groupby(["benchmark", "model_id"])["utility"].mean().to_dict()
    global_model_prior = train_outputs.groupby("model_id")["utility"].mean().to_dict()
    signal_cols = [
        "query_id",
        "benchmark",
        "domain",
        "metric",
        "need_large",
        "benchmark_composed_need_large",
        "self_vote_frac",
        "self_vote_margin",
        "self_vote_entropy",
        "signal_constrained_plus_cached_mean_risk",
        "signal_constrained_plus_cached_max_risk",
    ]
    merged = base_choices.merge(target[[column for column in signal_cols if column in target.columns]], on="query_id", how="left")
    rows: list[dict[str, Any]] = []
    for item in merged.itertuples(index=False):
        query_id = str(item.query_id)
        benchmark = str(item.benchmark)
        base_model = str(item.base_model_id)
        actions = rows_by_query[query_id]
        base_row = actions[base_model]
        consensus = consensus_features(
            query_id,
            benchmark,
            rows_by_query,
            benchmark_model_prior,
            global_model_prior,
        )
        row = item._asdict()
        row.update(consensus)
        row.update(
            {
                "base_quality": float(base_row.get("quality_score", 0.0) or 0.0),
                "base_utility": float(base_row.get("utility", 0.0) or 0.0),
                "base_norm_cost": float(base_row.get("normalized_remote_cost", 0.0) or 0.0),
                "base_is_frontier": bool(base_row.get("is_frontier", False)),
                "base_is_strong": base_model in STRONG_OR_FRONTIER,
                "gain_consensus_vs_base": float(consensus["consensus_utility"]) - float(base_row.get("utility", 0.0) or 0.0),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in numeric_feature_columns():
        frame[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
    return frame


def consensus_features(
    query_id: str,
    benchmark: str,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    benchmark_model_prior: dict[tuple[str, str], float],
    global_model_prior: dict[str, float],
) -> dict[str, Any]:
    actions = rows_by_query[query_id]
    answer_support: dict[str, int] = {}
    answer_models: dict[str, list[str]] = {}
    valid = 0
    for model_id in LOCAL_ACTIONS:
        if model_id not in actions:
            continue
        answer = normalize_answer(actions[model_id].get("parsed_answer", ""))
        if not answer:
            continue
        valid += 1
        answer_support[answer] = answer_support.get(answer, 0) + 1
        answer_models.setdefault(answer, []).append(model_id)
    if not answer_support:
        empty = {
            "consensus_model": "",
            "consensus_answer": "",
            "consensus_support": 0,
            "consensus_frac": 0.0,
            "local_valid": 0,
            "local_unique": 0,
            "consensus_entropy": 0.0,
            "consensus_prior": 0.0,
            "consensus_norm_cost": 0.0,
            "consensus_quality": 0.0,
            "consensus_utility": 0.0,
        }
        empty.update({f"has_{model_id}": 0 for model_id in LOCAL_LLM_ACTIONS})
        return empty
    counts = np.asarray(list(answer_support.values()), dtype=float)
    probs = counts / max(float(counts.sum()), 1e-12)
    entropy = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())
    best_answer = sorted(
        answer_support,
        key=lambda answer: (
            answer_support[answer],
            max(model_prior(benchmark_model_prior, global_model_prior, benchmark, model) for model in answer_models[answer]),
        ),
        reverse=True,
    )[0]
    models = answer_models[best_answer]
    best_model = sorted(
        models,
        key=lambda model: (
            model_prior(benchmark_model_prior, global_model_prior, benchmark, model),
            -float(actions[model].get("normalized_remote_cost", 0.0) or 0.0),
        ),
        reverse=True,
    )[0]
    action = actions[best_model]
    result = {
        "consensus_model": best_model,
        "consensus_answer": best_answer,
        "consensus_support": int(answer_support[best_answer]),
        "consensus_frac": float(answer_support[best_answer] / max(valid, 1)),
        "local_valid": int(valid),
        "local_unique": int(len(answer_support)),
        "consensus_entropy": entropy,
        "consensus_prior": model_prior(benchmark_model_prior, global_model_prior, benchmark, best_model),
        "consensus_norm_cost": float(action.get("normalized_remote_cost", 0.0) or 0.0),
        "consensus_quality": float(action.get("quality_score", 0.0) or 0.0),
        "consensus_utility": float(action.get("utility", 0.0) or 0.0),
    }
    result.update({f"has_{model_id}": int(model_id in models) for model_id in LOCAL_LLM_ACTIONS})
    return result


def model_prior(
    benchmark_model_prior: dict[tuple[str, str], float],
    global_model_prior: dict[str, float],
    benchmark: str,
    model_id: str,
) -> float:
    return float(benchmark_model_prior.get((benchmark, model_id), global_model_prior.get(model_id, 0.0)))


def evaluate_gain_gates(
    feature_frame: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = categorical_feature_columns()
    num_cols = numeric_feature_columns()
    train = feature_frame[feature_frame["split"].astype(str).eq("train")].copy()
    predictors = fit_predictors(train, cat_cols, num_cols)
    scored = feature_frame.copy()
    for name, pipe in predictors.items():
        scored[f"pred_{name}"] = pipe.predict(scored[cat_cols + num_cols])

    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    methods = [("base_candidate_ranker", "reference", "", math.nan)]
    for pred_col in [f"pred_{name}" for name in predictors]:
        thresholds = threshold_grid(scored[scored["split"].astype(str).eq("val")][pred_col])
        for threshold in thresholds:
            methods.append((f"{pred_col}_thr{threshold:.4f}", "local_safe_gain_gate", pred_col, float(threshold)))
    methods.append(("diagnostic_oracle_base_vs_consensus", "diagnostic_oracle", "oracle", math.nan))

    for method, family, pred_col, threshold in methods:
        for split in ["val", "test"]:
            split_frame = scored[scored["split"].astype(str).eq(split)].copy()
            choices = choose_actions(split_frame, pred_col=pred_col, threshold=threshold, family=family)
            selected_rows = choices[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            target_split = target[target["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                method,
                family,
                split,
                selected_rows,
                outputs,
                target=target_split,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            row.update(
                {
                    "predictor": pred_col,
                    "threshold": threshold,
                    "override_rate": float(choices["overrode_base"].mean()) if not choices.empty else 0.0,
                    "consensus_available_rate": float(choices["consensus_available"].mean()) if not choices.empty else 0.0,
                    "local_probe_call_rate": 1.0,
                }
            )
            rows.append(row)
            if split == "test":
                details.append(
                    selected_rows[
                        [
                            "query_id",
                            "query_text",
                            "benchmark",
                            "metric",
                            "model_id",
                            "quality_score",
                            "utility",
                            "normalized_remote_cost",
                            "is_frontier",
                            "parsed_answer",
                        ]
                    ]
                    .merge(
                        choices[
                            [
                                "query_id",
                                "base_model_id",
                                "consensus_model",
                                "consensus_support",
                                "consensus_frac",
                                "predicted_gain",
                                "overrode_base",
                            ]
                        ],
                        on="query_id",
                        how="left",
                    )
                    .assign(method=method, family=family)
                )
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    return table, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def fit_predictors(train: pd.DataFrame, cat_cols: list[str], num_cols: list[str]) -> dict[str, Pipeline]:
    predictors: dict[str, Pipeline] = {}
    model_specs = {
        "ridge": Ridge(alpha=1.0),
        "ridge10": Ridge(alpha=10.0),
        "hgb": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=10,
            l2_regularization=0.1,
            random_state=17,
        ),
        "rf": RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=5, random_state=17, n_jobs=-1),
        "et": ExtraTreesRegressor(n_estimators=250, max_depth=7, min_samples_leaf=4, random_state=17, n_jobs=-1),
    }
    for name, model in model_specs.items():
        pipe = Pipeline([("pre", preprocessor(cat_cols, num_cols)), ("model", model)])
        pipe.fit(train[cat_cols + num_cols], train["gain_consensus_vs_base"].astype(float))
        predictors[name] = pipe
    return predictors


def preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), num_cols),
        ]
    )


def threshold_grid(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return [0.0]
    quantiles = np.quantile(numeric.to_numpy(dtype=float), np.linspace(0.0, 1.0, 81))
    anchors = np.asarray([-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5])
    return sorted({float(x) for x in np.round(np.concatenate([quantiles, anchors]), 6)})


def choose_actions(frame: pd.DataFrame, *, pred_col: str, threshold: float, family: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        base_model = str(item.base_model_id)
        consensus_model = str(item.consensus_model)
        selected = base_model
        predicted_gain = 0.0
        if family == "diagnostic_oracle":
            if float(item.consensus_utility) > float(item.base_utility):
                selected = consensus_model
                predicted_gain = float(item.gain_consensus_vs_base)
        elif family != "reference":
            predicted_gain = float(getattr(item, pred_col))
            if consensus_model and predicted_gain >= float(threshold):
                selected = consensus_model
        rows.append(
            {
                "query_id": str(item.query_id),
                "model_id": selected,
                "base_model_id": base_model,
                "consensus_model": consensus_model,
                "consensus_support": int(item.consensus_support),
                "consensus_frac": float(item.consensus_frac),
                "predicted_gain": predicted_gain,
                "overrode_base": selected != base_model,
                "consensus_available": bool(consensus_model),
            }
        )
    return pd.DataFrame(rows)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility"]:
        val = table[table["split"].eq("val") & table["family"].eq("local_safe_gain_gate")].copy()
        if val.empty:
            continue
        best = val.sort_values([objective, "frontier_call_rate", "override_rate"], ascending=[False, True, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule=f"val_best_{objective}"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule=f"val_best_{objective}_test"))
        capped = val[val["frontier_call_rate"] <= 0.30].copy()
        if not capped.empty:
            best_cap = capped.sort_values([objective, "frontier_call_rate"], ascending=[False, True]).head(1)
            cap_method = str(best_cap.iloc[0]["method"])
            rows.append(best_cap.assign(selection_rule=f"val_best_{objective}_frontier_le_0.30"))
            rows.append(
                table[table["split"].eq("test") & table["method"].eq(cap_method)].copy().assign(
                    selection_rule=f"val_best_{objective}_frontier_le_0.30_test"
                )
            )
    reference = table[table["split"].eq("test") & table["family"].eq("reference")]
    if not reference.empty:
        rows.append(reference.assign(selection_rule="reference_test"))
    diagnostic = table[table["split"].eq("test") & table["family"].eq("diagnostic_oracle")]
    if not diagnostic.empty:
        rows.append(diagnostic.assign(selection_rule="diagnostic_oracle_test"))
    top_test = (
        table[table["split"].eq("test") & table["family"].ne("diagnostic_oracle")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(12)
    )
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    selected = selected.drop_duplicates(["selection_rule", "method", "split"])
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(
        table[["method", "split", "_utility_values"]],
        on=["method", "split"],
        how="left",
    )
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed).drop(
        columns=["_utility_values"], errors="ignore"
    )


def categorical_feature_columns() -> list[str]:
    return ["benchmark", "domain", "metric", "base_model_id", "consensus_model"]


def numeric_feature_columns() -> list[str]:
    return [
        "need_large",
        "benchmark_composed_need_large",
        "self_vote_frac",
        "self_vote_margin",
        "self_vote_entropy",
        "signal_constrained_plus_cached_mean_risk",
        "signal_constrained_plus_cached_max_risk",
        "base_norm_cost",
        "base_is_frontier",
        "base_is_strong",
        "consensus_support",
        "consensus_frac",
        "local_valid",
        "local_unique",
        "consensus_entropy",
        "consensus_prior",
        "consensus_norm_cost",
        *[f"has_{model_id}" for model_id in LOCAL_LLM_ACTIONS],
    ]


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    return " ".join(text.replace("$", "").replace("\\boxed", "").replace("{", "").replace("}", "").split())


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4f7a88")
    ax.set_xlabel("Held-out test selected-action utility")
    ax.set_title("Local-Safe Gain Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_local_safe_gain_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, features: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "override_rate",
        "consensus_available_rate",
        "local_probe_call_rate",
    ]
    lines = [
        "# Local-Safe Gain Gate",
        "",
        "This cached experiment trains a train-only gain predictor for replacing the practical candidate-ranker action with a local consensus action.",
        "It makes no GPT, Gemini, Claude, vLLM, or local model calls; it reuses cached local outputs as probe evidence.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/183_local_safe_gain_gate.py",
        (
            "PYTHONPATH=src python experiments/183_local_safe_gain_gate.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Feature Summary",
        "",
        markdown_table(
            features.groupby("split", as_index=False).agg(
                n_queries=("query_id", "nunique"),
                mean_consensus_support=("consensus_support", "mean"),
                mean_consensus_frac=("consensus_frac", "mean"),
                mean_true_consensus_gain=("gain_consensus_vs_base", "mean"),
                positive_gain_rate=("gain_consensus_vs_base", lambda s: float((s > 0).mean())),
            )
        ),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]),
        "",
        "## Best Held-Out Non-Oracle Rows",
        "",
        markdown_table(
            table[table["split"].eq("test") & table["family"].ne("diagnostic_oracle")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)
            .head(12)[[column for column in cols if column in table.columns]]
        ),
        "",
        "## Interpretation",
        "",
        "- This is a partial positive for cheap local evidence: it suppresses some expensive actions while preserving quality.",
        "- The diagnostic oracle between the base action and local consensus remains below the full cost-aware oracle, so local consensus alone cannot reach the target.",
        "- The next branch needs stronger task-specific correctness checks, especially for GPQA and MMLUPro.",
        "",
        "## Artifacts",
        "",
        f"- Feature table: `{args.output_dir / 'table_local_safe_gain_features.csv'}`",
        f"- All policy rows: `{args.output_dir / 'table_local_safe_gain_policy_all.csv'}`",
        f"- Selected policy rows: `{args.output_dir / 'table_local_safe_gain_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_local_safe_gain_query_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_local_safe_gain_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
