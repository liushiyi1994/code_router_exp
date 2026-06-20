from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_EVIDENCE_POOL = (
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
)
LOCAL_NO_TOOL_POOL = tuple(model for model in LOCAL_EVIDENCE_POOL if model != "deterministic_math_tool")
STRONG_OR_FRONTIER_ACTIONS = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
SUPPORT_SIGNALS = (
    "signal_combined_mean_risk",
    "signal_combined_max_risk",
    "signal_constrained_yesno_local_evidence_risk",
    "signal_constrained_yesno_mean_risk",
    "signal_constrained_plus_cached_mean_risk",
    "signal_constrained_plus_cached_max_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-calibrated answer-support policy for deployed action identity.")
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
        default=Path("results/controlled/broad100_answer_support_action_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171")
    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(target, args.benchmark_composed_choices, args.benchmark_composed_method, exp171)
    priors = exp172.fit_train_priors(outputs)
    support_model, support_train = fit_support_model(outputs, exp172)
    table_internal, details = evaluate_support_library(
        target,
        outputs,
        exp172=exp172,
        support_model=support_model,
        priors=priors,
        lambda_cost=float(args.lambda_cost),
    )
    selected = exp172.validation_selected_rows(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = exp172.add_bootstrap_ci(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = table.drop(columns=["_utility_values"], errors="ignore")
    table.to_csv(args.output_dir / "table_answer_support_action_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_answer_support_action_policy_selected.csv", index=False)
    details.to_csv(args.output_dir / "table_answer_support_action_policy_query_choices.csv", index=False)
    support_train.to_csv(args.output_dir / "table_answer_support_train_features.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "ANSWER_SUPPORT_ACTION_POLICY_MEMO.md", args, table, selected, support_train, exp172)
    print(f"Wrote answer-support action policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_support_model(outputs: pd.DataFrame, exp172) -> tuple[dict[str, Any], pd.DataFrame]:
    train = outputs[outputs["split"].astype(str).eq("train")].copy()
    rows: list[dict[str, Any]] = []
    for _, group in train.groupby("query_id", sort=False):
        actions = group.set_index("model_id").to_dict("index")
        benchmark = str(group.iloc[0].get("benchmark", ""))
        domain = str(group.iloc[0].get("domain", ""))
        metric = str(group.iloc[0].get("metric", ""))
        counts = answer_counts(actions, exp172, LOCAL_EVIDENCE_POOL)
        top_count = max(counts.values()) if counts else 0
        n_answers = max(sum(counts.values()), 1)
        tool_answer = action_answer(actions, "deterministic_math_tool", exp172)
        self_answer = action_answer(actions, "qwen3-32b-awq-selfconsistency-n3-local", exp172)
        qwen32_answer = action_answer(actions, "qwen3-32b-awq-local", exp172)
        for model_id in LOCAL_EVIDENCE_POOL:
            if not exp172.is_action_available(actions, model_id):
                continue
            answer = action_answer(actions, model_id, exp172)
            if not answer:
                continue
            support_count = int(counts.get(answer, 0))
            rows.append(
                {
                    "query_id": str(group.iloc[0]["query_id"]),
                    "benchmark": benchmark,
                    "domain": domain,
                    "metric": metric,
                    "model_id": model_id,
                    "answer_norm": answer,
                    "support_count": support_count,
                    "support_frac": support_count / n_answers,
                    "is_top_group": bool(support_count == top_count and top_count > 0),
                    "has_tool_support": bool(tool_answer and answer == tool_answer),
                    "has_self_support": bool(self_answer and answer == self_answer),
                    "has_qwen32_support": bool(qwen32_answer and answer == qwen32_answer),
                    "unique_answer_count": int(len(counts)),
                    "quality": float(actions[model_id].get("quality_score", 0.0) or 0.0),
                    "utility": float(actions[model_id].get("utility", 0.0) or 0.0),
                }
            )
    features = pd.DataFrame(rows)
    if features.empty:
        return {"global_mean": 0.0, "tables": []}, features
    global_mean = float(features["utility"].mean())
    table_specs = [
        ("bench_model_support_tool_top", ["benchmark", "model_id", "support_count", "has_tool_support", "is_top_group"]),
        ("bench_support_tool_top", ["benchmark", "support_count", "has_tool_support", "is_top_group"]),
        ("bench_support_top", ["benchmark", "support_count", "is_top_group"]),
        ("bench_model", ["benchmark", "model_id"]),
        ("model_support_top", ["model_id", "support_count", "is_top_group"]),
        ("support_tool_top", ["support_count", "has_tool_support", "is_top_group"]),
        ("support_top", ["support_count", "is_top_group"]),
        ("model", ["model_id"]),
        ("benchmark", ["benchmark"]),
    ]
    tables = []
    for name, keys in table_specs:
        table = (
            features.groupby(keys, dropna=False)
            .agg(n=("query_id", "size"), mean_utility=("utility", "mean"), mean_quality=("quality", "mean"))
            .reset_index()
        )
        lookup = {tuple(row[key] for key in keys): row for _, row in table.iterrows()}
        tables.append((name, keys, lookup))
    return {"global_mean": global_mean, "tables": tables}, features


def evaluate_support_library(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    val = target[target["split"].astype(str).eq("val")]
    local_scores = [
        local_support_choice(row, rows_by_query[str(row["query_id"])], exp172, support_model, priors)["score"]
        for _, row in val.iterrows()
        if str(row["query_id"]) in rows_by_query
    ]
    local_thresholds = candidate_thresholds(np.asarray(local_scores, dtype=float))
    policy_specs = support_policy_specs(val, local_thresholds, exp172=exp172, support_model=support_model, priors=priors)
    policy_specs.update(
        benchmark_support_policy_specs(
            val,
            rows_by_query,
            local_thresholds,
            exp172=exp172,
            support_model=support_model,
            priors=priors,
        )
    )
    baseline_specs = {
        "full_cost_aware_oracle": ("diagnostic_oracle", lambda row, actions: exp172.full_oracle_selector("utility")(row, actions)),
        "full_quality_oracle": ("diagnostic_oracle", lambda row, actions: exp172.full_oracle_selector("quality_score")(row, actions)),
        "support_local_only": (
            "answer_support_policy",
            lambda row, actions: local_support_choice(row, actions, exp172, support_model, priors)["model_id"],
        ),
        "support_oracle_need_large_gate": (
            "diagnostic_policy",
            lambda row, actions: large_or_support(row, actions, exp172, support_model, priors, oracle_gate=True),
        ),
        "support_171_gate_large_prior": (
            "answer_support_policy",
            lambda row, actions: large_or_support(row, actions, exp172, support_model, priors, gate_column="benchmark_composed_need_large"),
        ),
    }
    all_specs = {**baseline_specs, **policy_specs}
    for split in ["val", "test"]:
        frame = target[target["split"].astype(str).eq(split)].copy()
        for method, (family, selector) in all_specs.items():
            selected = select_actions(frame, selector, rows_by_query, exp172, support_model, priors)
            selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                method,
                family,
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            support_scores = selected["support_score"].to_numpy(dtype=float) if "support_score" in selected else np.asarray([])
            row["mean_support_score"] = float(np.mean(support_scores)) if support_scores.size else np.nan
            rows.append(row)
            if split == "test" and method in detail_methods():
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
                detail["method"] = method
                detail["family"] = family
                detail = detail.merge(
                    selected[["query_id", "support_model_id", "support_score", "support_answer", "support_count"]],
                    on="query_id",
                    how="left",
                )
                detail_frames.append(detail)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return table, details


def support_policy_specs(
    val: pd.DataFrame,
    local_thresholds: np.ndarray,
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
) -> dict[str, tuple[str, Any]]:
    specs: dict[str, tuple[str, Any]] = {}
    for threshold in local_thresholds:
        threshold = float(threshold)
        name = f"support_score_ge{threshold:.4g}_else_large_prior"
        specs[name] = ("answer_support_threshold", support_threshold_selector(threshold, exp172=exp172, support_model=support_model, priors=priors))
        name = f"support_171_gate_score_ge{threshold:.4g}"
        specs[name] = (
            "answer_support_threshold",
            support_threshold_selector(threshold, gate_column="benchmark_composed_need_large", exp172=exp172, support_model=support_model, priors=priors),
        )
    for signal in [s for s in SUPPORT_SIGNALS if s in val.columns]:
        for risk_threshold in candidate_thresholds(val[signal].to_numpy(dtype=float))[::5]:
            for support_threshold in local_thresholds[::5]:
                name = f"support_{signal}_risk{risk_threshold:.4g}_score{support_threshold:.4g}"
                specs[name] = (
                    "answer_support_threshold",
                    support_threshold_selector(
                        float(support_threshold),
                        risk_signal=signal,
                        risk_threshold=float(risk_threshold),
                        exp172=exp172,
                        support_model=support_model,
                        priors=priors,
                    ),
                )
    return specs


def benchmark_support_policy_specs(
    val: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    local_thresholds: np.ndarray,
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
) -> dict[str, tuple[str, Any]]:
    specs: dict[str, tuple[str, Any]] = {}
    for use_gate, name in [(False, "benchmark_support_else_large"), (True, "benchmark_support_171_gate")]:
        for epsilon in [0.0, 0.005, 0.01, 0.02, 0.04]:
            for tie_break in ["utility", "frontier_light", "quality"]:
                thresholds = choose_threshold_by_benchmark(
                    val,
                    rows_by_query,
                    local_thresholds,
                    exp172=exp172,
                    support_model=support_model,
                    priors=priors,
                    use_gate=use_gate,
                    epsilon=epsilon,
                    tie_break=tie_break,
                )
                method = f"{name}_eps{epsilon:g}_{tie_break}"
                specs[method] = (
                    "answer_support_benchmark_threshold",
                    benchmark_support_selector(thresholds, exp172=exp172, support_model=support_model, priors=priors, use_gate=use_gate),
                )
    return specs


def choose_threshold_by_benchmark(
    val: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    thresholds: np.ndarray,
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
    use_gate: bool,
    epsilon: float,
    tie_break: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for benchmark, frame in val.groupby("benchmark", sort=False):
        candidate_rows: list[dict[str, float]] = []
        for threshold in thresholds:
            utilities: list[float] = []
            qualities: list[float] = []
            frontier_flags: list[float] = []
            for _, row in frame.iterrows():
                actions = rows_by_query[str(row["query_id"])]
                model_id = threshold_decision(
                    row,
                    actions,
                    float(threshold),
                    exp172=exp172,
                    support_model=support_model,
                    priors=priors,
                    use_gate=use_gate,
                )
                action = actions.get(model_id, {})
                utilities.append(as_float(action.get("utility", 0.0)))
                qualities.append(as_float(action.get("quality_score", 0.0)))
                frontier_flags.append(float(bool(action.get("is_frontier", False)) or model_id in STRONG_OR_FRONTIER_ACTIONS))
            candidate_rows.append(
                {
                    "threshold": float(threshold),
                    "mean_utility": float(np.mean(utilities)) if utilities else -np.inf,
                    "mean_quality": float(np.mean(qualities)) if qualities else -np.inf,
                    "frontier_rate": float(np.mean(frontier_flags)) if frontier_flags else 1.0,
                }
            )
        table = pd.DataFrame(candidate_rows)
        if table.empty:
            out[str(benchmark)] = float(thresholds[0]) if len(thresholds) else 0.0
            continue
        best_utility = float(table["mean_utility"].max())
        eligible = table[table["mean_utility"] >= best_utility - float(epsilon)].copy()
        if tie_break == "frontier_light":
            eligible = eligible.sort_values(["frontier_rate", "mean_quality", "mean_utility"], ascending=[True, False, False])
        elif tie_break == "quality":
            eligible = eligible.sort_values(["mean_quality", "mean_utility", "frontier_rate"], ascending=[False, False, True])
        else:
            eligible = eligible.sort_values(["mean_utility", "mean_quality", "frontier_rate"], ascending=[False, False, True])
        out[str(benchmark)] = float(eligible.iloc[0]["threshold"])
    return out


def benchmark_support_selector(
    thresholds: dict[str, float],
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
    use_gate: bool,
):
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        threshold = float(thresholds.get(str(row.get("benchmark", "")), 0.0))
        return threshold_decision(row, actions, threshold, exp172=exp172, support_model=support_model, priors=priors, use_gate=use_gate)

    return select


def threshold_decision(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    threshold: float,
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
    use_gate: bool,
) -> str:
    choice = local_support_choice(row, actions, exp172, support_model, priors)
    use_large = choice["score"] < float(threshold)
    if use_gate:
        use_large = use_large or bool(row.get("benchmark_composed_need_large", False))
    if use_large:
        return exp172.choose_prior_action(row, actions, priors, STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
    return str(choice["model_id"])


def support_threshold_selector(
    threshold: float,
    *,
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
    gate_column: str | None = None,
    risk_signal: str | None = None,
    risk_threshold: float = 0.0,
):
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        choice = local_support_choice(row, actions, exp172, support_model, priors)
        use_large = choice["score"] < threshold
        if gate_column is not None:
            use_large = use_large or bool(row.get(gate_column, False))
        if risk_signal is not None:
            use_large = use_large or as_float(row.get(risk_signal, -np.inf), default=-np.inf) >= risk_threshold
        if use_large:
            return exp172.choose_prior_action(row, actions, priors, STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
        return str(choice["model_id"])

    return select


def large_or_support(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
    *,
    oracle_gate: bool = False,
    gate_column: str | None = None,
) -> str:
    use_large = bool(row.get("need_large", False)) if oracle_gate else bool(row.get(gate_column or "", False))
    if use_large:
        return exp172.choose_prior_action(row, actions, priors, STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
    return str(local_support_choice(row, actions, exp172, support_model, priors)["model_id"])


def select_actions(
    frame: pd.DataFrame,
    selector,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        query_id = str(row["query_id"])
        actions = rows_by_query[query_id]
        model_id = selector(row, actions)
        if not exp172.is_action_available(actions, model_id):
            model_id = exp172.first_available(actions, tuple(actions))
        support_choice = local_support_choice(row, actions, exp172, support_model, priors)
        rows.append(
            {
                "query_id": query_id,
                "model_id": str(model_id),
                "support_model_id": str(support_choice["model_id"]),
                "support_score": float(support_choice["score"]),
                "support_answer": str(support_choice["answer"]),
                "support_count": int(support_choice["support_count"]),
            }
        )
    return pd.DataFrame(rows)


def local_support_choice(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    exp172,
    support_model: dict[str, Any],
    priors: dict[str, Any],
) -> dict[str, Any]:
    counts = answer_counts(actions, exp172, LOCAL_EVIDENCE_POOL)
    top_count = max(counts.values()) if counts else 0
    n_answers = max(sum(counts.values()), 1)
    ranking = exp172.prior_rank_index(row, priors, LOCAL_NO_TOOL_POOL)
    candidates: list[tuple[float, float, float, float, str, str, int]] = []
    tool_answer = action_answer(actions, "deterministic_math_tool", exp172)
    self_answer = action_answer(actions, "qwen3-32b-awq-selfconsistency-n3-local", exp172)
    qwen32_answer = action_answer(actions, "qwen3-32b-awq-local", exp172)
    for model_id in LOCAL_EVIDENCE_POOL:
        if not exp172.is_action_available(actions, model_id):
            continue
        answer = action_answer(actions, model_id, exp172)
        if not answer:
            continue
        support_count = int(counts.get(answer, 0))
        features = {
            "benchmark": str(row.get("benchmark", "")),
            "model_id": model_id,
            "support_count": support_count,
            "has_tool_support": bool(tool_answer and answer == tool_answer),
            "has_self_support": bool(self_answer and answer == self_answer),
            "has_qwen32_support": bool(qwen32_answer and answer == qwen32_answer),
            "is_top_group": bool(support_count == top_count and top_count > 0),
        }
        expected = lookup_support_score(support_model, features)
        prior = 1.0 if model_id == "deterministic_math_tool" else float(ranking.get(model_id, 0.0))
        support_frac = support_count / n_answers
        score = expected + 0.04 * prior + 0.04 * support_frac + (0.03 if features["has_tool_support"] else 0.0)
        quality = as_float(actions[model_id].get("quality_score", 0.0))
        cost = as_float(actions[model_id].get("normalized_remote_cost", 0.0))
        candidates.append((score, quality, support_frac, -cost, model_id, answer, support_count))
    if candidates:
        score, quality, support_frac, neg_cost, model_id, answer, support_count = sorted(candidates, reverse=True)[0]
        return {
            "model_id": model_id,
            "score": float(score),
            "answer": answer,
            "support_count": int(support_count),
            "support_frac": float(support_frac),
            "quality": float(quality),
        }
    fallback = exp172.choose_prior_action(row, actions, priors, LOCAL_NO_TOOL_POOL, scope="benchmark")
    return {"model_id": fallback, "score": 0.0, "answer": "", "support_count": 0, "support_frac": 0.0, "quality": 0.0}


def lookup_support_score(model: dict[str, Any], features: dict[str, Any]) -> float:
    global_mean = float(model.get("global_mean", 0.0))
    for _, keys, lookup in model.get("tables", []):
        key = tuple(features.get(key_name) for key_name in keys)
        row = lookup.get(key)
        if row is None:
            continue
        n = float(row.get("n", 0.0) or 0.0)
        mean = float(row.get("mean_utility", global_mean) or global_mean)
        return (n * mean + 4.0 * global_mean) / (n + 4.0)
    return global_mean


def answer_counts(actions: dict[str, dict[str, Any]], exp172, pool: tuple[str, ...]) -> Counter[str]:
    values: list[str] = []
    for model_id in pool:
        if not exp172.is_action_available(actions, model_id):
            continue
        answer = action_answer(actions, model_id, exp172)
        if answer:
            values.append(answer)
    return Counter(values)


def action_answer(actions: dict[str, dict[str, Any]], model_id: str, exp172) -> str:
    if model_id not in actions:
        return ""
    return str(actions[model_id].get("answer_norm", "") or exp172.normalized_answer(actions[model_id].get("parsed_answer", "")))


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([0.0])
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, 21)))


def as_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def detail_methods() -> set[str]:
    return {
        "full_cost_aware_oracle",
        "support_local_only",
        "support_171_gate_large_prior",
        "support_oracle_need_large_gate",
    }


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#567063")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Answer-Support Action Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_answer_support_action_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    support_train: pd.DataFrame,
    exp172,
) -> None:
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
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "utility_gap_to_oracle",
        "quality_gap_to_oracle",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "mean_support_score",
        "selection_rule",
    ]
    selected_cols = [column for column in cols if column in selected.columns]
    test = table[table["split"].eq("test")].copy()
    full_oracle = test[test["method"].eq("full_cost_aware_oracle")].head(1)
    deployable_test = test[test["family"].isin(["answer_support_policy", "answer_support_threshold"])]
    best_deployable = deployable_test.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1)
    support_summary = (
        support_train.groupby(["benchmark", "support_count"], as_index=False)
        .agg(n=("query_id", "size"), mean_utility=("utility", "mean"), mean_quality=("quality", "mean"))
        .sort_values(["benchmark", "support_count"])
    )
    lines = [
        "# Answer-Support Action Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/174_answer_support_action_policy.py",
        (
            "PYTHONPATH=src python experiments/174_answer_support_action_policy.py "
            f"--target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## What This Tests",
        "",
        "- Cached-only deployed-action selector; no GPT, Gemini, Claude, vLLM, or other model calls.",
        "- Fits train-only reliability tables for local answer groups.",
        "- Uses validation-selected thresholds to decide whether to trust the supported local answer group or escalate to the train-best large action.",
        "- This targets the action-identity gap from experiments 172 and 173.",
        "",
        "## Selected Rows",
        "",
        "```csv",
        exp172.compact_csv(selected[selected_cols], max_rows=80),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        exp172.compact_csv(test.sort_values(["mean_utility", "mean_quality"], ascending=False)[[c for c in cols if c in test.columns]], max_rows=40),
        "```",
        "",
        "## Target Check",
        "",
        *exp172.target_check_lines(full_oracle, best_deployable),
        "",
        "## Train Support Summary",
        "",
        "```csv",
        exp172.compact_csv(support_summary, max_rows=80),
        "```",
        "",
        "## Interpretation",
        "",
        "- If validation-selected rows do not beat 172, local answer support is insufficient as an action-identity probe.",
        "- Diagnostic rows remain diagnostic only; deployable rows must be selected on validation.",
        "",
        "## Artifacts",
        "",
        f"- All policy table: `{args.output_dir / 'table_answer_support_action_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_answer_support_action_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_answer_support_action_policy_query_choices.csv'}`",
        f"- Train support features: `{args.output_dir / 'table_answer_support_train_features.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_answer_support_action_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
