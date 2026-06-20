from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TOOL = "deterministic_math_tool"
LOCAL_POOL = ("qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local")
LOCAL_PLUS_POOL = (
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
)
LARGE_POOL = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
ALL_POOL = tuple(dict.fromkeys((TOOL, *LOCAL_PLUS_POOL, "gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve")))
CODE_BENCHMARKS = {"humaneval", "mbpp"}


@dataclass(frozen=True)
class RankerConfig:
    model_name: str
    action_mode: str
    cost_penalty: float

    @property
    def method(self) -> str:
        return f"{self.model_name}_{self.action_mode}_pen{self.cost_penalty:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only candidate correctness ranker for broad100 action identity.")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_candidate_correctness_ranker_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cv-folds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    priors = exp172.fit_train_priors(outputs)
    feature_frame, cat_cols, num_cols = build_feature_frame(outputs, target)
    configs = candidate_configs()
    cv_table = cross_validate_configs(
        configs,
        feature_frame,
        target,
        outputs,
        priors,
        exp172=exp172,
        exp175=exp175,
        cat_cols=cat_cols,
        num_cols=num_cols,
        lambda_cost=float(args.lambda_cost),
        folds=int(args.cv_folds),
        seed=int(args.seed),
    )
    policy_table_internal, details = evaluate_configs(
        configs,
        feature_frame,
        target,
        outputs,
        priors,
        exp172=exp172,
        exp175=exp175,
        cat_cols=cat_cols,
        num_cols=num_cols,
        lambda_cost=float(args.lambda_cost),
        cv_table=cv_table,
    )
    selected = selected_rows(policy_table_internal, cv_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed), exp172=exp172)
    policy_table = exp172.add_bootstrap_ci(policy_table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    policy_table = policy_table.drop(columns=["_utility_values"], errors="ignore")
    policy_table.to_csv(args.output_dir / "table_candidate_ranker_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_candidate_ranker_policy_selected.csv", index=False)
    cv_table.to_csv(args.output_dir / "table_candidate_ranker_cv.csv", index=False)
    details.to_csv(args.output_dir / "table_candidate_ranker_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "CANDIDATE_CORRECTNESS_RANKER_MEMO.md", args, cv_table, policy_table, selected)
    print(f"Wrote candidate correctness ranker policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def candidate_configs() -> list[RankerConfig]:
    configs: list[RankerConfig] = []
    for model_name in ["et", "rf", "hgb_l1", "hgb_l2"]:
        for action_mode in ["all_rank", "gate_rank_localplus", "localplus_rank_large_prior"]:
            for cost_penalty in [0.25, 0.35, 0.50, 0.75]:
                configs.append(RankerConfig(model_name=model_name, action_mode=action_mode, cost_penalty=float(cost_penalty)))
    return configs


def build_feature_frame(outputs: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    signal_cols = [
        "query_id",
        "need_large",
        "benchmark_composed_need_large",
        "signal_combined_mean_risk",
        "signal_combined_max_risk",
        "signal_constrained_yesno_local_evidence_risk",
        "signal_constrained_yesno_query_only_risk",
        "signal_semantic_uncertainty",
        "signal_early_rollout_instability",
        "signal_slm_medium_divergence",
        "self_vote_frac",
        "self_vote_entropy",
    ]
    frame = outputs.merge(target[[col for col in signal_cols if col in target.columns]], on="query_id", how="left")
    frame["answer_norm_ranker"] = frame["parsed_answer"].map(normalized_answer)
    frame["answer_len"] = frame["answer_norm_ranker"].str.len().fillna(0).astype(float)
    frame["answer_empty"] = frame["answer_norm_ranker"].eq("").astype(int)
    frame["answer_numeric"] = frame["answer_norm_ranker"].str.fullmatch(r"[-+]?\d+(\.\d+)?").fillna(False).astype(int)

    local_counts = (
        frame[frame["model_id"].isin(LOCAL_PLUS_POOL)]
        .groupby(["query_id", "answer_norm_ranker"])
        .size()
        .rename("local_support")
        .reset_index()
    )
    frame = frame.merge(local_counts, on=["query_id", "answer_norm_ranker"], how="left")
    frame["local_support"] = frame["local_support"].fillna(0.0)
    frame["local_support_frac"] = frame["local_support"] / max(len(LOCAL_PLUS_POOL), 1)

    frame["is_tool"] = frame["model_id"].eq(TOOL).astype(int)
    frame["tool_available_bool"] = frame.get("tool_available", False).fillna(False).astype(bool).astype(int)
    frame["is_frontier_bool"] = frame.get("is_frontier", False).fillna(False).astype(bool).astype(int)
    frame["is_local_bool"] = frame.get("is_local", False).fillna(False).astype(bool).astype(int)
    frame["public_test_pass"] = (
        frame["benchmark"].isin(CODE_BENCHMARKS) & frame["parsed_answer"].astype(str).str.lower().eq("passed")
    ).astype(int)

    train = frame[frame["split"].astype(str).eq("train")].copy()
    by_benchmark = (
        train.groupby(["benchmark", "model_id"], as_index=False)
        .agg(
            train_model_benchmark_quality=("quality_score", "mean"),
            train_model_benchmark_utility=("utility", "mean"),
        )
    )
    by_model = (
        train.groupby("model_id", as_index=False)
        .agg(
            train_model_quality=("quality_score", "mean"),
            train_model_utility=("utility", "mean"),
        )
    )
    frame = frame.merge(by_benchmark, on=["benchmark", "model_id"], how="left").merge(by_model, on="model_id", how="left")
    for col in ["train_model_benchmark_quality", "train_model_benchmark_utility", "train_model_quality", "train_model_utility"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(float(train["quality_score"].mean()))

    cat_cols = ["benchmark", "model_id", "metric"]
    num_cols = [
        "normalized_remote_cost",
        "latency_s",
        "input_tokens",
        "output_tokens",
        "answer_len",
        "answer_empty",
        "answer_numeric",
        "local_support",
        "local_support_frac",
        "is_tool",
        "tool_available_bool",
        "is_frontier_bool",
        "is_local_bool",
        "public_test_pass",
        "train_model_benchmark_quality",
        "train_model_benchmark_utility",
        "train_model_quality",
        "train_model_utility",
    ]
    for col in signal_cols:
        if col not in {"query_id", "need_large", "benchmark_composed_need_large"} and col in frame.columns:
            num_cols.append(col)
    for col in num_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame, cat_cols, num_cols


def normalized_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"nan", "none", "no_code"} or text.startswith("failed"):
        return ""
    return text


def make_model(name: str):
    if name == "et":
        return ExtraTreesRegressor(n_estimators=250, max_depth=9, min_samples_leaf=3, random_state=17, n_jobs=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=4, random_state=17, n_jobs=-1)
    if name == "hgb_l1":
        return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.04, l2_regularization=1.0, max_leaf_nodes=12, random_state=17)
    if name == "hgb_l2":
        return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, l2_regularization=0.1, max_leaf_nodes=15, random_state=17)
    raise ValueError(f"Unknown ranker model: {name}")


def make_pipeline(model_name: str, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), num_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", make_model(model_name))])


def cross_validate_configs(
    configs: list[RankerConfig],
    feature_frame: pd.DataFrame,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
    *,
    exp172,
    exp175,
    cat_cols: list[str],
    num_cols: list[str],
    lambda_cost: float,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    query_ids = target[target["split"].astype(str).eq("train")]["query_id"].astype(str).unique()
    kfold = KFold(n_splits=max(2, int(folds)), shuffle=True, random_state=int(seed))
    rows: list[dict[str, Any]] = []
    rows_by_query = rows_by_query_map(outputs)
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    for config in configs:
        utilities: list[float] = []
        qualities: list[float] = []
        for train_idx, holdout_idx in kfold.split(query_ids):
            fit_ids = set(query_ids[train_idx])
            holdout_ids = set(query_ids[holdout_idx])
            fit = feature_frame[feature_frame["query_id"].astype(str).isin(fit_ids)]
            holdout = feature_frame[feature_frame["query_id"].astype(str).isin(holdout_ids)].copy()
            pipe = make_pipeline(config.model_name, cat_cols, num_cols)
            pipe.fit(fit[cat_cols + num_cols], fit["quality_score"].astype(float))
            holdout = add_predictions(holdout, pipe, config.cost_penalty, cat_cols, num_cols)
            frame = target[target["query_id"].astype(str).isin(holdout_ids)].copy()
            selected = select_actions(frame, holdout, config, rows_by_query, priors, exp172=exp172, exp175=exp175)
            selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq("train")].copy()
            row = exp172.evaluate_selected_rows(
                config.method,
                "candidate_ranker_cv",
                "train",
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            utilities.append(float(row["mean_utility"]))
            qualities.append(float(row["mean_quality"]))
        rows.append(
            {
                "method": config.method,
                "model_name": config.model_name,
                "action_mode": config.action_mode,
                "cost_penalty": config.cost_penalty,
                "cv_mean_utility": float(np.mean(utilities)),
                "cv_std_utility": float(np.std(utilities)),
                "cv_mean_quality": float(np.mean(qualities)),
            }
        )
    return pd.DataFrame(rows).sort_values(["cv_mean_utility", "cv_std_utility"], ascending=[False, True])


def evaluate_configs(
    configs: list[RankerConfig],
    feature_frame: pd.DataFrame,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
    *,
    exp172,
    exp175,
    cat_cols: list[str],
    num_cols: list[str],
    lambda_cost: float,
    cv_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_by_query = rows_by_query_map(outputs)
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    train = feature_frame[feature_frame["split"].astype(str).eq("train")]
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    cv_meta = cv_table.set_index("method").to_dict("index")
    for config in configs:
        pipe = make_pipeline(config.model_name, cat_cols, num_cols)
        pipe.fit(train[cat_cols + num_cols], train["quality_score"].astype(float))
        for split in ["val", "test"]:
            candidates = feature_frame[feature_frame["split"].astype(str).eq(split)].copy()
            candidates = add_predictions(candidates, pipe, config.cost_penalty, cat_cols, num_cols)
            frame = target[target["split"].astype(str).eq(split)].copy()
            selected = select_actions(frame, candidates, config, rows_by_query, priors, exp172=exp172, exp175=exp175)
            selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                config.method,
                "candidate_ranker",
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            row.update(cv_meta.get(config.method, {}))
            row.update(candidate_generation_cost_columns(config, selected, outputs, selected_rows, lambda_cost=lambda_cost))
            rows.append(row)
            if split == "test" and config.method in detail_methods(cv_table):
                detail = selected_rows[
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
                ].copy()
                detail["method"] = config.method
                detail_frames.append(detail)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False]), (
        pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    )


def rows_by_query_map(outputs: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs.groupby("query_id", sort=False)}


def add_predictions(frame: pd.DataFrame, pipe: Pipeline, penalty: float, cat_cols: list[str], num_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["predicted_quality"] = np.clip(pipe.predict(out[cat_cols + num_cols]), 0.0, 1.0)
    out["predicted_utility"] = out["predicted_quality"] - float(penalty) * out["normalized_remote_cost"].astype(float)
    return out


def select_actions(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    config: RankerConfig,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    priors: dict[str, Any],
    *,
    exp172,
    exp175,
) -> pd.DataFrame:
    candidate_by_query = {str(query_id): group for query_id, group in candidates.groupby("query_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        query_id = str(row["query_id"])
        actions = rows_by_query[query_id]
        benchmark = str(row.get("benchmark", ""))
        model_id = exp172.tool_action(actions)
        if not model_id and benchmark in CODE_BENCHMARKS:
            model_id = exp175.public_test_choice(row, actions, exp172, priors)
        if not model_id:
            pool = pool_for(row, config, priors, actions, exp172)
            if pool:
                query_candidates = candidate_by_query[query_id]
                query_candidates = query_candidates[query_candidates["model_id"].isin(pool)].copy()
                query_candidates = query_candidates[
                    query_candidates["model_id"].map(lambda model: exp172.is_action_available(actions, str(model)))
                ]
                if not query_candidates.empty:
                    model_id = str(
                        query_candidates.sort_values(["predicted_utility", "predicted_quality"], ascending=False)
                        .iloc[0]["model_id"]
                    )
        if not model_id or not exp172.is_action_available(actions, str(model_id)):
            model_id = exp172.first_available(actions, exp172.ALL_ACTIONS)
        rows.append({"query_id": query_id, "model_id": str(model_id)})
    return pd.DataFrame(rows)


def pool_for(row: pd.Series, config: RankerConfig, priors: dict[str, Any], actions: dict[str, dict[str, Any]], exp172) -> tuple[str, ...]:
    if config.action_mode == "all_rank":
        return ALL_POOL
    if config.action_mode == "gate_rank_localplus":
        return LARGE_POOL if bool(row.get("benchmark_composed_need_large", False)) else LOCAL_PLUS_POOL
    if config.action_mode == "localplus_rank_large_prior":
        if bool(row.get("benchmark_composed_need_large", False)):
            return (exp172.choose_prior_action(row, actions, priors, LARGE_POOL, scope="benchmark"),)
        return LOCAL_PLUS_POOL
    raise ValueError(f"Unknown action mode: {config.action_mode}")


def candidate_generation_cost_columns(
    config: RankerConfig,
    selected: pd.DataFrame,
    outputs: pd.DataFrame,
    selected_rows: pd.DataFrame,
    *,
    lambda_cost: float,
) -> dict[str, Any]:
    gpt_cost = max(
        float(outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )
    selected_cost = float(selected_rows["normalized_remote_cost"].mean())
    if config.action_mode == "all_rank":
        frontier_cost = (
            outputs[outputs["model_id"].isin(["gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve"])]
            .pivot_table(index="query_id", values="cost_total_usd", aggfunc="sum")
            .reindex(selected["query_id"].astype(str))
            .fillna(0.0)["cost_total_usd"]
        )
    else:
        frontier_cost = pd.Series(0.0, index=selected["query_id"].astype(str))
    generation_norm = float((frontier_cost / gpt_cost).mean())
    mean_quality = float(selected_rows["quality_score"].mean())
    utility_with_generation = mean_quality - float(lambda_cost) * (selected_cost + generation_norm)
    return {
        "candidate_generation_norm_cost_mean": generation_norm,
        "utility_with_candidate_generation_cost": utility_with_generation,
    }


def selected_rows(table: pd.DataFrame, cv_table: pd.DataFrame, *, bootstrap_samples: int, seed: int, exp172) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not cv_table.empty:
        method = str(cv_table.iloc[0]["method"])
        rows.append(table[table["method"].eq(method) & table["split"].eq("val")].copy().assign(selection_rule="train_group_cv_best_val"))
        rows.append(table[table["method"].eq(method) & table["split"].eq("test")].copy().assign(selection_rule="train_group_cv_best_test"))
    val = table[table["split"].eq("val")].sort_values(["mean_utility", "normalized_cost_mean"], ascending=[False, True])
    if not val.empty:
        method = str(val.iloc[0]["method"])
        rows.append(val.head(1).copy().assign(selection_rule="val_best_utility"))
        rows.append(table[table["method"].eq(method) & table["split"].eq("test")].copy().assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    with_values = table[["method", "split", "_utility_values"]]
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(with_values, on=["method", "split"], how="left")
    selected = exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)
    return selected.drop(columns=["_utility_values"], errors="ignore")


def detail_methods(cv_table: pd.DataFrame) -> set[str]:
    out = set(cv_table.head(3)["method"].astype(str).tolist())
    return out


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5a6d8c")
    ax.set_xlabel("Held-out test selected-solver utility")
    ax.set_title("Candidate Correctness Ranker")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_candidate_ranker_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, cv_table: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "candidate_generation_norm_cost_mean",
        "utility_with_candidate_generation_cost",
        "selection_rule",
    ]
    lines = [
        "# Candidate Correctness Ranker Policy",
        "",
        "This cached experiment trains a small candidate-level correctness regressor on train rows only. It is meant to attack concrete action identity, not only upward-routing detection. It makes no GPT, Gemini, Claude, vLLM, or local model calls.",
        "",
        "Important cost caveat: `all_rank` modes rank over frontier candidate answers too. Their selected-solver utility is not a deployable cost claim unless the frontier candidate-generation cost is also charged.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/177_candidate_correctness_ranker_policy.py",
        (
            "PYTHONPATH=src python experiments/177_candidate_correctness_ranker_policy.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Train Group-CV Ranking",
        "",
        markdown_table(cv_table.head(12)),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[col for col in cols if col in selected.columns]]),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)[[col for col in cols if col in table.columns]]),
        "",
        "## Interpretation",
        "",
        "- Candidate-level ranking is a partial positive: it improves concrete action identity over the earlier deployed-action bridge.",
        "- It still does not reach the full cost-aware oracle target.",
        "- The highest selected-solver rows require observing frontier candidate answers, so full deployment cost must include those candidate calls.",
        "",
        "## Artifacts",
        "",
        f"- CV table: `{args.output_dir / 'table_candidate_ranker_cv.csv'}`",
        f"- All policy table: `{args.output_dir / 'table_candidate_ranker_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_candidate_ranker_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_candidate_ranker_query_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_candidate_ranker_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
