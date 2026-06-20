from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import normalize_answer


LOCAL_ACTIONS = (
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
)
LARGE_ACTIONS = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
FRONTIER_ACTIONS = {
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
}
SELECTED_POLICY = "local_majority_scopegsm8k_votes2_if_base_frontier_cheapest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-call residual audit for concrete action identity after early-signal probe pilots."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--current-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_local_consensus_cost_suppression_audit/"
            "table_local_consensus_cost_suppression_query_choices.csv"
        ),
    )
    parser.add_argument(
        "--early-targets",
        type=Path,
        default=Path("results/controlled/broad100_slm_llm_early_signal_probe_pilot/table_slm_llm_oracle_targets.csv"),
    )
    parser.add_argument("--policy", default=SELECTED_POLICY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_residual_action_identity_audit"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(args.outputs)
    choices = load_policy_choices(args.current_choices, args.policy)
    early_targets = load_optional_csv(args.early_targets)
    detail = build_query_audit(outputs, choices, early_targets)
    benchmark = summarize_by_benchmark(detail)
    confusion = summarize_action_confusion(detail)
    ceilings, ceiling_choices = evaluate_evidence_ceilings(detail, outputs)

    detail.to_csv(args.output_dir / "table_residual_action_identity_queries.csv", index=False)
    benchmark.to_csv(args.output_dir / "table_residual_by_benchmark.csv", index=False)
    confusion.to_csv(args.output_dir / "table_action_confusion.csv", index=False)
    ceilings.to_csv(args.output_dir / "table_evidence_ceilings.csv", index=False)
    ceiling_choices.to_csv(args.output_dir / "table_evidence_ceiling_query_choices.csv", index=False)
    write_memo(args.output_dir / "RESIDUAL_ACTION_IDENTITY_AUDIT_MEMO.md", args, detail, benchmark, confusion, ceilings)
    print(f"Wrote residual action-identity audit to {args.output_dir}")


def load_outputs(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "query_id",
        "model_id",
        "split",
        "benchmark",
        "query_text",
        "parsed_answer",
        "quality_score",
        "utility",
        "normalized_remote_cost",
        "cost_total_usd",
        "latency_s",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = frame.copy()
    out["query_id"] = out["query_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["answer_norm"] = out["parsed_answer"].map(norm_answer)
    return out


def load_policy_choices(path: Path, policy: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "policy" not in frame.columns:
        raise ValueError(f"{path} has no policy column")
    out = frame[frame["policy"].astype(str).eq(policy)].copy()
    if out.empty:
        available = sorted(frame["policy"].astype(str).unique())
        raise ValueError(f"Policy {policy!r} not found. Available policies: {available}")
    out["query_id"] = out["query_id"].astype(str)
    return out


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "query_id" in frame.columns:
        frame["query_id"] = frame["query_id"].astype(str)
    return frame


def build_query_audit(outputs: pd.DataFrame, choices: pd.DataFrame, early_targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_query = {str(query_id): group.copy() for query_id, group in outputs.groupby("query_id", sort=False)}
    early_by_query = (
        early_targets.drop_duplicates("query_id").set_index("query_id").to_dict("index")
        if not early_targets.empty and "query_id" in early_targets.columns
        else {}
    )

    selected_cols = [
        "query_id",
        "split",
        "benchmark",
        "query_text",
        "gold_answer",
        "metric",
        "selected_model",
        "selected_quality",
        "selected_utility",
        "selected_frontier",
        "normalized_remote_cost",
        "cost_total_usd",
        "changed",
        "probe_used",
        "support_source",
        "support_model",
        "support_count",
        "support_confidence",
        "oracle_utility",
        "oracle_quality",
    ]
    for _, choice in choices[selected_cols].iterrows():
        query_id = str(choice["query_id"])
        group = by_query.get(query_id)
        if group is None or group.empty:
            continue
        oracle = best_row(group)
        best_local = best_row(group[group["model_id"].isin(LOCAL_ACTIONS)])
        best_large = best_row(group[group["model_id"].isin(LARGE_ACTIONS)])
        selected_model = str(choice["selected_model"])
        selected_row = first_model_row(group, selected_model)
        if selected_row is None:
            selected_answer = ""
            selected_norm_cost = float(choice.get("normalized_remote_cost", np.nan))
            selected_cost_usd = float(choice.get("cost_total_usd", np.nan))
            selected_latency = float("nan")
        else:
            selected_answer = str(selected_row["answer_norm"])
            selected_norm_cost = float(selected_row["normalized_remote_cost"])
            selected_cost_usd = float(selected_row["cost_total_usd"])
            selected_latency = float(selected_row["latency_s"])

        local_group = group[group["model_id"].isin(LOCAL_ACTIONS)].copy()
        local_counts = answer_counts(local_group)
        oracle_answer = str(oracle["answer_norm"])
        oracle_model = str(oracle["model_id"])
        selected_is_local = selected_model in LOCAL_ACTIONS
        selected_is_large = selected_model in LARGE_ACTIONS
        oracle_is_local = oracle_model in LOCAL_ACTIONS
        oracle_is_large = oracle_model in LARGE_ACTIONS
        selected_is_frontier = selected_model in FRONTIER_ACTIONS
        oracle_is_frontier = oracle_model in FRONTIER_ACTIONS
        residual = float(oracle["utility"]) - float(choice["selected_utility"])
        delta_large = float(best_large["utility"]) - float(best_local["utility"])
        category = miss_category(
            selected_model=selected_model,
            oracle_model=oracle_model,
            selected_answer=selected_answer,
            oracle_answer=oracle_answer,
            selected_is_frontier=selected_is_frontier,
            oracle_is_frontier=oracle_is_frontier,
            selected_is_local=selected_is_local,
            oracle_is_local=oracle_is_local,
        )
        early = early_by_query.get(query_id, {})
        rows.append(
            {
                "query_id": query_id,
                "split": str(choice["split"]),
                "benchmark": str(choice["benchmark"]),
                "metric": str(choice.get("metric", "")),
                "query_text": str(choice["query_text"]),
                "gold_answer": str(choice.get("gold_answer", "")),
                "selected_model": selected_model,
                "selected_quality": float(choice["selected_quality"]),
                "selected_utility": float(choice["selected_utility"]),
                "selected_normalized_cost": selected_norm_cost,
                "selected_cost_usd": selected_cost_usd,
                "selected_latency_s": selected_latency,
                "selected_is_local": bool(selected_is_local),
                "selected_is_large": bool(selected_is_large),
                "selected_is_frontier": bool(selected_is_frontier),
                "selected_answer_norm": selected_answer,
                "oracle_model": oracle_model,
                "oracle_quality": float(oracle["quality_score"]),
                "oracle_utility": float(oracle["utility"]),
                "oracle_normalized_cost": float(oracle["normalized_remote_cost"]),
                "oracle_cost_usd": float(oracle["cost_total_usd"]),
                "oracle_latency_s": float(oracle["latency_s"]),
                "oracle_is_local": bool(oracle_is_local),
                "oracle_is_large": bool(oracle_is_large),
                "oracle_is_frontier": bool(oracle_is_frontier),
                "oracle_answer_norm": oracle_answer,
                "residual_utility": residual,
                "residual_quality": float(oracle["quality_score"]) - float(choice["selected_quality"]),
                "best_local_action": str(best_local["model_id"]),
                "best_large_action": str(best_large["model_id"]),
                "local_utility": float(best_local["utility"]),
                "large_utility": float(best_large["utility"]),
                "local_quality": float(best_local["quality_score"]),
                "large_quality": float(best_large["quality_score"]),
                "delta_large": delta_large,
                "need_large": bool(delta_large > 1e-12),
                "oracle_answer_local_support": int(local_counts.get(oracle_answer, 0)) if oracle_answer else 0,
                "selected_answer_local_support": int(local_counts.get(selected_answer, 0)) if selected_answer else 0,
                "local_unique_answer_count": int(sum(1 for ans in local_counts if ans)),
                "local_answer_counts_json": json.dumps(local_counts, sort_keys=True),
                "any_local_matches_oracle_answer": bool(oracle_answer and local_counts.get(oracle_answer, 0) > 0),
                "any_local_matches_selected_answer": bool(selected_answer and local_counts.get(selected_answer, 0) > 0),
                "selected_matches_oracle_action": bool(selected_model == oracle_model),
                "selected_matches_oracle_answer": bool(selected_answer and selected_answer == oracle_answer),
                "miss_category": category,
                "changed_by_current_policy": bool(choice.get("changed", False)),
                "probe_used_by_current_policy": bool(choice.get("probe_used", False)),
                "support_source": str(choice.get("support_source", "")),
                "support_model": str(choice.get("support_model", "")),
                "support_count": safe_float(choice.get("support_count", 0.0)),
                "support_confidence": safe_float(choice.get("support_confidence", 0.0)),
                "signal_query_answerability_risk": safe_float(early.get("signal_query_answerability_risk", np.nan)),
                "signal_early_rollout_instability": safe_float(early.get("signal_early_rollout_instability", np.nan)),
                "signal_slm_medium_divergence": safe_float(early.get("signal_slm_medium_divergence", np.nan)),
                "signal_semantic_uncertainty": safe_float(early.get("signal_semantic_uncertainty", np.nan)),
                "signal_combined_mean_risk": safe_float(early.get("signal_combined_mean_risk", np.nan)),
                "signal_combined_max_risk": safe_float(early.get("signal_combined_max_risk", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def summarize_by_benchmark(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, benchmark), group in detail.groupby(["split", "benchmark"], sort=True):
        rows.append(
            {
                "split": split,
                "benchmark": benchmark,
                "n_queries": int(len(group)),
                "selected_mean_quality": float(group["selected_quality"].mean()),
                "selected_mean_utility": float(group["selected_utility"].mean()),
                "oracle_mean_quality": float(group["oracle_quality"].mean()),
                "oracle_mean_utility": float(group["oracle_utility"].mean()),
                "mean_residual_utility": float(group["residual_utility"].mean()),
                "total_residual_utility": float(group["residual_utility"].sum()),
                "selected_frontier_rate": float(group["selected_is_frontier"].mean()),
                "oracle_frontier_rate": float(group["oracle_is_frontier"].mean()),
                "need_large_rate": float(group["need_large"].mean()),
                "same_action_rate": float(group["selected_matches_oracle_action"].mean()),
                "same_answer_rate": float(group["selected_matches_oracle_answer"].mean()),
                "oracle_answer_local_support_rate": float(group["any_local_matches_oracle_answer"].mean()),
                "mean_oracle_answer_local_support": float(group["oracle_answer_local_support"].mean()),
                "miss_categories_json": json.dumps(group["miss_category"].value_counts().sort_index().to_dict()),
                "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict()),
                "oracle_models_json": json.dumps(group["oracle_model"].value_counts().sort_index().to_dict()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "total_residual_utility"], ascending=[True, False])


def summarize_action_confusion(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detail.groupby(["split", "selected_model", "oracle_model"], sort=True)
        .agg(
            n_queries=("query_id", "size"),
            mean_selected_utility=("selected_utility", "mean"),
            mean_oracle_utility=("oracle_utility", "mean"),
            mean_residual_utility=("residual_utility", "mean"),
            total_residual_utility=("residual_utility", "sum"),
            same_answer_rate=("selected_matches_oracle_answer", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["split", "total_residual_utility"], ascending=[True, False])


def evaluate_evidence_ceilings(detail: pd.DataFrame, outputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_query = {str(query_id): group.copy() for query_id, group in outputs.groupby("query_id", sort=False)}
    choice_rows: list[dict[str, Any]] = []
    methods = [
        "current_selected",
        "query_oracle",
        "local_vs_large_oracle",
        "oracle_answer_local_equivalence_ceiling",
        "selected_frontier_answer_local_equivalence_ceiling",
        "family_known_action_oracle",
    ]
    for _, row in detail.iterrows():
        query_id = str(row["query_id"])
        group = by_query[query_id]
        method_rows = {
            "current_selected": model_choice_from_name(group, row["selected_model"]),
            "query_oracle": best_row(group),
            "local_vs_large_oracle": best_row(
                pd.DataFrame(
                    [
                        best_row(group[group["model_id"].isin(LOCAL_ACTIONS)]),
                        best_row(group[group["model_id"].isin(LARGE_ACTIONS)]),
                    ]
                )
            ),
            "oracle_answer_local_equivalence_ceiling": oracle_answer_local_equivalence(row, group),
            "selected_frontier_answer_local_equivalence_ceiling": selected_frontier_answer_local_equivalence(row, group),
            "family_known_action_oracle": family_known_action_oracle(row, group),
        }
        for method in methods:
            selected = method_rows[method]
            choice_rows.append(
                {
                    "query_id": query_id,
                    "split": str(row["split"]),
                    "benchmark": str(row["benchmark"]),
                    "method": method,
                    "selected_model": str(selected["model_id"]),
                    "quality": float(selected["quality_score"]),
                    "utility": float(selected["utility"]),
                    "normalized_cost": float(selected["normalized_remote_cost"]),
                    "cost_usd": float(selected["cost_total_usd"]),
                    "latency_s": float(selected["latency_s"]),
                    "frontier": bool(str(selected["model_id"]) in FRONTIER_ACTIONS),
                    "changed_from_current": bool(str(selected["model_id"]) != str(row["selected_model"])),
                    "oracle_utility": float(row["oracle_utility"]),
                }
            )
    choice_frame = pd.DataFrame(choice_rows)
    summary_rows: list[dict[str, Any]] = []
    for (split, method), group in choice_frame.groupby(["split", "method"], sort=True):
        oracle_mean = float(group["oracle_utility"].mean())
        summary_rows.append(
            {
                "split": split,
                "method": method,
                "n_queries": int(len(group)),
                "mean_quality": float(group["quality"].mean()),
                "mean_utility": float(group["utility"].mean()),
                "oracle_mean_utility": oracle_mean,
                "oracle_utility_ratio": float(group["utility"].mean() / max(oracle_mean, 1e-12)),
                "utility_gap_to_oracle": float(oracle_mean - group["utility"].mean()),
                "normalized_cost_mean": float(group["normalized_cost"].mean()),
                "remote_cost_total_usd": float(group["cost_usd"].sum()),
                "mean_latency_s": float(group["latency_s"].mean()),
                "p95_latency_s": float(group["latency_s"].quantile(0.95)),
                "frontier_call_rate": float(group["frontier"].mean()),
                "changed_rate": float(group["changed_from_current"].mean()),
                "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    return summary, choice_frame.sort_values(["split", "method", "benchmark", "query_id"]).reset_index(drop=True)


def best_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("Cannot choose best row from an empty frame")
    return frame.sort_values(
        ["utility", "quality_score", "normalized_remote_cost", "model_id"],
        ascending=[False, False, True, True],
    ).iloc[0].copy()


def first_model_row(group: pd.DataFrame, model_id: str) -> pd.Series | None:
    rows = group[group["model_id"].astype(str).eq(str(model_id))]
    if rows.empty:
        return None
    return rows.iloc[0].copy()


def model_choice_from_name(group: pd.DataFrame, model_id: str) -> pd.Series:
    row = first_model_row(group, str(model_id))
    if row is None:
        return best_row(group)
    return row


def oracle_answer_local_equivalence(row: pd.Series, group: pd.DataFrame) -> pd.Series:
    oracle_answer = str(row["oracle_answer_norm"])
    if oracle_answer:
        local_matches = group[group["model_id"].isin(LOCAL_ACTIONS) & group["answer_norm"].astype(str).eq(oracle_answer)]
        if not local_matches.empty:
            return best_row(local_matches)
    return model_choice_from_name(group, str(row["selected_model"]))


def selected_frontier_answer_local_equivalence(row: pd.Series, group: pd.DataFrame) -> pd.Series:
    selected = model_choice_from_name(group, str(row["selected_model"]))
    selected_answer = str(selected["answer_norm"])
    if bool(row["selected_is_frontier"]) and selected_answer:
        local_matches = group[group["model_id"].isin(LOCAL_ACTIONS) & group["answer_norm"].astype(str).eq(selected_answer)]
        if not local_matches.empty:
            return best_row(local_matches)
    return selected


def family_known_action_oracle(row: pd.Series, group: pd.DataFrame) -> pd.Series:
    if bool(row["need_large"]):
        return best_row(group[group["model_id"].isin(LARGE_ACTIONS)])
    return best_row(group[group["model_id"].isin(LOCAL_ACTIONS)])


def answer_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(str(ans) for ans in frame["answer_norm"].tolist() if str(ans))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def miss_category(
    *,
    selected_model: str,
    oracle_model: str,
    selected_answer: str,
    oracle_answer: str,
    selected_is_frontier: bool,
    oracle_is_frontier: bool,
    selected_is_local: bool,
    oracle_is_local: bool,
) -> str:
    if selected_model == oracle_model:
        return "matched_oracle_action"
    if selected_answer and oracle_answer and selected_answer == oracle_answer:
        return "same_answer_wrong_cost_or_action"
    if selected_is_frontier and oracle_is_local:
        return "over_escalated_frontier_when_oracle_local"
    if (not selected_is_frontier) and oracle_is_frontier:
        return "missed_frontier_oracle"
    if selected_is_local and oracle_is_local:
        return "wrong_local_action"
    if (not selected_is_local) and (not oracle_is_local):
        return "wrong_large_action"
    return "wrong_family_or_action"


def norm_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return normalize_answer(str(value))


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_memo(
    path: Path,
    args: argparse.Namespace,
    detail: pd.DataFrame,
    benchmark: pd.DataFrame,
    confusion: pd.DataFrame,
    ceilings: pd.DataFrame,
) -> None:
    selected_policy = str(args.policy)
    test_detail = detail[detail["split"].eq("test")]
    val_detail = detail[detail["split"].eq("val")]
    test_ceilings = ceilings[ceilings["split"].eq("test")]
    val_ceilings = ceilings[ceilings["split"].eq("val")]
    current_test = metric_row(test_ceilings, "current_selected")
    oracle_test = metric_row(test_ceilings, "query_oracle")
    family_test = metric_row(test_ceilings, "family_known_action_oracle")
    local_eq_test = metric_row(test_ceilings, "oracle_answer_local_equivalence_ceiling")
    frontier_eq_test = metric_row(test_ceilings, "selected_frontier_answer_local_equivalence_ceiling")
    current_val = metric_row(val_ceilings, "current_selected")
    top_bench = (
        benchmark[benchmark["split"].eq("test")]
        .sort_values("total_residual_utility", ascending=False)
        .head(6)
    )
    top_confusion = (
        confusion[confusion["split"].eq("test")]
        .sort_values("total_residual_utility", ascending=False)
        .head(8)
    )
    lines: list[str] = [
        "# Residual Action-Identity Audit",
        "",
        "Status: no-call diagnostic audit over cached broad100 validation/test outputs. This is not a new solved router.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/198_residual_action_identity_audit.py",
        "```",
        "",
        "## Inputs",
        "",
        f"- Cached action matrix: `{args.outputs}`",
        f"- Current validation-selected concrete policy table: `{args.current_choices}`",
        f"- Current audited policy: `{selected_policy}`",
        f"- Cached early-signal target table: `{args.early_targets}`",
        "- No GPT, Gemini, Claude, local generation, or vLLM serving calls were made.",
        "",
        "## Benchmark And Model Scope",
        "",
        "- Slice: broad100 validation/test rows from the controlled LLMRouterBench-derived manifest.",
        "- Held-out test queries: "
        f"`{len(test_detail)}` across `{test_detail['benchmark'].nunique()}` benchmarks.",
        "- Validation queries: "
        f"`{len(val_detail)}` across `{val_detail['benchmark'].nunique()}` benchmarks.",
        "- Local actions: "
        + ", ".join(f"`{item}`" for item in LOCAL_ACTIONS),
        "- Large/strong actions: "
        + ", ".join(f"`{item}`" for item in LARGE_ACTIONS),
        "- Frontier/API families in cached action matrix: GPT-family via `gpt-5.5`, Gemini-family via `gemini-3.5-flash` and `gemini-3.5-flash-strong-solve`; no Claude rows in this run.",
        "",
        "## Main Held-Out Numbers",
        "",
        f"- Current selected policy utility: `{current_test['mean_utility']:.6f}`; quality: `{current_test['mean_quality']:.6f}`; frontier-call rate: `{current_test['frontier_call_rate']:.6f}`.",
        f"- Query/action oracle utility: `{oracle_test['mean_utility']:.6f}`; quality: `{oracle_test['mean_quality']:.6f}`; frontier-call rate: `{oracle_test['frontier_call_rate']:.6f}`.",
        f"- Gap to oracle: `{current_test['utility_gap_to_oracle']:.6f}` utility and `{oracle_test['mean_quality'] - current_test['mean_quality']:.6f}` quality.",
        f"- Oracle utility ratio: `{current_test['oracle_utility_ratio']:.6f}`.",
        f"- Validation utility for the same policy: `{current_val['mean_utility']:.6f}`.",
        "",
        "## Diagnostic Ceilings",
        "",
        f"- If the router knew local-vs-large family and also had oracle concrete action identity inside that family, held-out utility would be `{family_test['mean_utility']:.6f}`.",
        f"- If a perfect pre-call local-equivalence signal knew when a local answer matched the oracle answer, held-out utility would be `{local_eq_test['mean_utility']:.6f}`.",
        f"- If it only suppressed frontier calls when a local answer matched the selected frontier answer post hoc, held-out utility would be `{frontier_eq_test['mean_utility']:.6f}`.",
        "",
        "Interpretation: the remaining loss is not solved by a binary larger-action gate. A large share is concrete action identity: which local/large action, or which answer, should be trusted.",
        "",
        "## Residual Concentration By Benchmark",
        "",
    ]
    table_md(
        top_bench,
        [
            "benchmark",
            "n_queries",
            "selected_mean_utility",
            "oracle_mean_utility",
            "total_residual_utility",
            "same_action_rate",
            "same_answer_rate",
            "oracle_answer_local_support_rate",
        ],
        lines,
    )
    lines.extend(
        [
            "",
            "## Largest Action Confusions",
            "",
        ]
    )
    table_md(
        top_confusion,
        [
            "selected_model",
            "oracle_model",
            "n_queries",
            "total_residual_utility",
            "same_answer_rate",
        ],
        lines,
    )
    lines.extend(
        [
            "",
            "## Probe Observations",
            "",
            "- The cached early-signal threshold pilot already built the requested oracle target table and signals for query-only answerability, rollout instability, SLM-vs-medium divergence, and semantic uncertainty.",
            "- Its validation-selected threshold underperformed the later concrete policy on held-out test, so those signals are not enough as standalone threshold routers.",
            "- The current policy's local-majority probe mostly reduces frontier calls, but only changes a few test rows and leaves a large oracle gap.",
            "- The useful post-hoc ceiling remains answer equivalence: knowing when a cheaper local answer is equivalent to the chosen/oracle answer would help, but the first plain local LLM verifier did not predict that reliably.",
            "",
            "## Next Recommended Probe",
            "",
            "Build a task-specific evidence-backed checker rather than another generic confidence threshold:",
            "",
            "1. For exact math, use symbolic/calculator verification and answer-consistency checks over candidate derivations.",
            "2. For GPQA/MMLUPro, use option-elimination or evidence-demanding MCQ verification with strict abstention.",
            "3. Treat verifier output as an action-identity signal, not a final answer unless the provider/local call is already paid and costed as the selected action.",
            "",
            "## Artifacts",
            "",
            f"- `{args.output_dir / 'table_residual_action_identity_queries.csv'}`",
            f"- `{args.output_dir / 'table_residual_by_benchmark.csv'}`",
            f"- `{args.output_dir / 'table_action_confusion.csv'}`",
            f"- `{args.output_dir / 'table_evidence_ceilings.csv'}`",
            f"- `{args.output_dir / 'table_evidence_ceiling_query_choices.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_row(frame: pd.DataFrame, method: str) -> pd.Series:
    row = frame[frame["method"].eq(method)]
    if row.empty:
        raise ValueError(f"Missing metric row for {method}")
    return row.iloc[0]


def table_md(frame: pd.DataFrame, columns: list[str], lines: list[str]) -> None:
    if frame.empty:
        lines.append("_No rows._")
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
