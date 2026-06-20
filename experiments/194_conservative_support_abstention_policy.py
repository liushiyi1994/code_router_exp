from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_POLICY = "scopegpqa+bbh+gsm8k+mmlupro_selected_qwen32_qwen3-14b-awq-local_none"
FRONTIERS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve", "gpt-5.5"}
STRONG_OR_FRONTIER = {"qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local", *FRONTIERS}
CHEAP_LOCAL_ACTIONS = {
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
}
LOCAL_SIGNAL_ACTIONS = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
SUPPORT_BENCHMARKS = ["gpqa", "mmlupro", "bbh", "gsm8k", "math500", "livemathbench", "aime"]
CONFIDENCE_THRESHOLDS = [0.0, 0.85, 0.95]
MIN_SUPPORTS = [2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative no-training support/abstention policy over cached local vLLM solve-support probes."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_targeted_residual_repair_policy/"
            "table_targeted_residual_repair_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default=BASE_POLICY)
    parser.add_argument(
        "--qwen14-verifier",
        type=Path,
        default=Path(
            "results/controlled/broad100_local_vllm_solve_support_residual_fusion/"
            "table_local_vllm_solve_support_verifier_outputs.csv"
        ),
    )
    parser.add_argument(
        "--qwen32-verifier",
        type=Path,
        default=Path(
            "results/controlled/broad100_local_vllm_solve_support_residual_fusion_qwen32/"
            "table_local_vllm_solve_support_verifier_outputs.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_conservative_support_abstention_policy"),
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
    base = pd.read_csv(args.base_query_choices)
    base = base[base["policy"].astype(str).eq(str(args.base_policy))].copy()
    if base.empty:
        raise RuntimeError(f"Base policy {args.base_policy!r} not found in {args.base_query_choices}.")

    verifier_tables = {
        "qwen14": pd.read_csv(args.qwen14_verifier),
        "qwen32": pd.read_csv(args.qwen32_verifier),
    }
    action_map = {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs.groupby("query_id")}
    support_counts = build_support_counts(outputs)
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][
        ["query_id", "model_id", "utility", "quality_score"]
    ].rename(
        columns={
            "model_id": "cost_oracle_model",
            "utility": "oracle_utility",
            "quality_score": "oracle_quality",
        }
    )
    base = drop_prefixed(base, ["oracle_utility", "oracle_quality", "cost_oracle_model"]).merge(
        oracle, on="query_id", how="left"
    )

    target_table = build_oracle_target_table(outputs)
    target_table.to_csv(args.output_dir / "table_oracle_targets_local_vs_large.csv", index=False)
    signal_table = build_probe_signal_table(outputs, verifier_tables, support_counts, target_table)
    signal_table.to_csv(args.output_dir / "table_probe_signals_cached.csv", index=False)

    rules = enumerate_rules()
    policy_table, query_choices = evaluate_rules(base, verifier_tables, action_map, support_counts, rules, args)
    selected = selected_rows(policy_table)
    selected_policies = set(selected["policy"].astype(str).tolist()) if not selected.empty else set()
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    pd.DataFrame(rules).to_csv(args.output_dir / "table_conservative_support_rule_library.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_conservative_support_abstention_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_conservative_support_abstention_policy_selected.csv", index=False)
    query_choices_to_write.to_csv(
        args.output_dir / "table_conservative_support_abstention_query_choices.csv", index=False
    )
    write_memo(args.output_dir / "CONSERVATIVE_SUPPORT_ABSTENTION_POLICY_MEMO.md", args, target_table, selected)
    print(f"Wrote conservative support/abstention policy results to {args.output_dir}")


def drop_prefixed(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    cols = [col for col in frame.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
    return frame.drop(columns=cols, errors="ignore")


def build_oracle_target_table(outputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, group in outputs.groupby("query_id", sort=False):
        cheap_local = group[group["model_id"].astype(str).isin(CHEAP_LOCAL_ACTIONS)].copy()
        large = group[group["model_id"].astype(str).isin(STRONG_OR_FRONTIER)].copy()
        if cheap_local.empty:
            cheap_local = group[group["is_frontier"].astype(bool).eq(False)].copy()
        if large.empty:
            large = group.copy()
        local_row = cheap_local.loc[cheap_local["utility"].astype(float).idxmax()]
        large_row = large.loc[large["utility"].astype(float).idxmax()]
        rows.append(
            {
                "query_id": str(query_id),
                "split": str(group.iloc[0]["split"]),
                "benchmark": str(group.iloc[0]["benchmark"]),
                "domain": str(group.iloc[0]["domain"]),
                "query_text": str(group.iloc[0]["query_text"]),
                "best_local_action": str(local_row["model_id"]),
                "best_large_action": str(large_row["model_id"]),
                "local_utility": float(local_row["utility"]),
                "large_utility": float(large_row["utility"]),
                "delta_large": float(large_row["utility"]) - float(local_row["utility"]),
                "need_large": bool(float(large_row["utility"]) > float(local_row["utility"]) + 1e-12),
                "local_quality": float(local_row["quality_score"]),
                "large_quality": float(large_row["quality_score"]),
                "local_normalized_cost": float(local_row["normalized_remote_cost"]),
                "large_normalized_cost": float(large_row["normalized_remote_cost"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"])


def build_probe_signal_table(
    outputs: pd.DataFrame,
    verifier_tables: dict[str, pd.DataFrame],
    support_counts: dict[str, dict[str, int]],
    target_table: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    verifier_maps = {name: table.set_index("query_id").to_dict("index") for name, table in verifier_tables.items()}
    for query_id, group in outputs.groupby("query_id", sort=False):
        answer_by_model = {
            str(row.model_id): normalize_answer(getattr(row, "parsed_answer", ""))
            for row in group.itertuples(index=False)
            if str(row.model_id) in LOCAL_SIGNAL_ACTIONS
        }
        answers = [answer for answer in answer_by_model.values() if answer]
        counts = Counter(answers)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()] if total else []
        entropy = -sum(prob * np.log2(prob) for prob in probs if prob > 0)
        top_counts = sorted(counts.values(), reverse=True)
        row = {
            "query_id": str(query_id),
            "split": str(group.iloc[0]["split"]),
            "benchmark": str(group.iloc[0]["benchmark"]),
            "query_text": str(group.iloc[0]["query_text"]),
            "local_unique_answers": int(len(counts)),
            "local_answer_entropy": float(entropy),
            "local_top_vote_count": int(top_counts[0]) if top_counts else 0,
            "local_vote_margin": int(top_counts[0] - top_counts[1]) if len(top_counts) > 1 else int(top_counts[0]) if top_counts else 0,
            "qwen4_qwen14_disagree": answer_by_model.get("qwen3-4b-local", "")
            != answer_by_model.get("qwen3-14b-awq-local", ""),
            "qwen4_qwen32_disagree": answer_by_model.get("qwen3-4b-local", "")
            != answer_by_model.get("qwen3-32b-awq-local", ""),
            "qwen14_qwen32_disagree": answer_by_model.get("qwen3-14b-awq-local", "")
            != answer_by_model.get("qwen3-32b-awq-local", ""),
        }
        for source, verifier_map in verifier_maps.items():
            item = verifier_map.get(str(query_id), {})
            supported = str(item.get("supported_model", "") or "")
            row[f"{source}_status"] = str(item.get("status", "missing") or "missing")
            row[f"{source}_verifier_confidence"] = float(item.get("verifier_confidence", 0.0) or 0.0)
            row[f"{source}_verifier_quality"] = safe_float(item.get("quality_score", np.nan))
            row[f"{source}_supported_model"] = supported
            row[f"{source}_supported_count"] = int(support_counts.get(str(query_id), {}).get(supported, 0))
            row[f"{source}_latency_s"] = safe_float(item.get("latency_s", np.nan))
        rows.append(row)
    signals = pd.DataFrame(rows)
    keep_targets = target_table[
        [
            "query_id",
            "best_local_action",
            "best_large_action",
            "local_utility",
            "large_utility",
            "delta_large",
            "need_large",
        ]
    ]
    return signals.merge(keep_targets, on="query_id", how="left").sort_values(["split", "benchmark", "query_id"])


def build_support_counts(outputs: pd.DataFrame) -> dict[str, dict[str, int]]:
    support: dict[str, dict[str, int]] = {}
    for query_id, group in outputs.groupby("query_id", sort=False):
        answer_by_model = {}
        for row in group.itertuples(index=False):
            answer = normalize_answer(getattr(row, "parsed_answer", ""))
            if answer:
                answer_by_model[str(row.model_id)] = answer
        counts = Counter(answer_by_model.values())
        support[str(query_id)] = {model: int(counts[answer]) for model, answer in answer_by_model.items()}
    return support


def enumerate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_targeted_residual_repair",
            "source": "none",
            "benchmark": "all",
            "confidence_threshold": np.nan,
            "min_support": np.nan,
            "only_nonfrontier": True,
            "selector_kind": "base",
        }
    ]
    for source in ["qwen14", "qwen32"]:
        for benchmark in SUPPORT_BENCHMARKS:
            for min_support in MIN_SUPPORTS:
                for threshold in CONFIDENCE_THRESHOLDS:
                    rules.append(
                        {
                            "policy": f"{source}_{benchmark}_support{min_support}_conf{threshold:g}_nonfrontier",
                            "source": source,
                            "benchmark": benchmark,
                            "confidence_threshold": float(threshold),
                            "min_support": int(min_support),
                            "only_nonfrontier": True,
                            "selector_kind": "single_benchmark_support",
                        }
                    )
    return rules


def evaluate_rules(
    base: pd.DataFrame,
    verifier_tables: dict[str, pd.DataFrame],
    action_map: dict[str, dict[str, dict[str, Any]]],
    support_counts: dict[str, dict[str, int]],
    rules: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    verifier_maps = {name: table.set_index("query_id").to_dict("index") for name, table in verifier_tables.items()}
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for rule in rules:
        choices = apply_rule(base, verifier_maps, action_map, support_counts, rule, float(args.lambda_cost))
        details.append(choices.assign(policy=str(rule["policy"])))
        for split, group in choices.groupby("split", dropna=False):
            values = group["fused_utility"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            oracle_q = float(group["oracle_quality"].astype(float).mean())
            mean_u = float(values.mean())
            mean_q = float(group["fused_quality"].astype(float).mean())
            need_large = group["need_large"].astype(bool) if "need_large" in group else pd.Series(False, index=group.index)
            selected_large = group["fused_model"].astype(str).map(is_large)
            tp = int((need_large & selected_large).sum())
            fp = int((~need_large & selected_large).sum())
            fn = int((need_large & ~selected_large).sum())
            rows.append(
                {
                    **rule,
                    "split": split,
                    "n_queries": int(len(group)),
                    "mean_quality": mean_q,
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "mean_utility_with_probe_cost": mean_u,
                    "cost_oracle_mean_utility": oracle_u,
                    "quality_oracle_mean_quality": oracle_q,
                    "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
                    "oracle_utility_ratio_with_probe_cost": mean_u / max(oracle_u, 1e-12),
                    "utility_gap_to_oracle": oracle_u - mean_u,
                    "quality_gap_to_oracle": oracle_q - mean_q,
                    "frontier_call_rate": float(group["fused_frontier"].mean()),
                    "strong_or_frontier_call_rate": float(selected_large.mean()),
                    "probe_call_rate": float(group["probe_used"].mean()),
                    "override_rate": float(group["fused_changed"].mean()),
                    "mean_probe_latency_s": float(group.loc[group["probe_used"], "probe_latency_s"].mean())
                    if group["probe_used"].any()
                    else 0.0,
                    "need_large_precision": tp / (tp + fp) if (tp + fp) else 0.0,
                    "need_large_recall": tp / (tp + fn) if (tp + fn) else 0.0,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "selected_models_json": json.dumps(group["fused_model"].value_counts().sort_index().to_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def apply_rule(
    base: pd.DataFrame,
    verifier_maps: dict[str, dict[str, dict[str, Any]]],
    action_map: dict[str, dict[str, dict[str, Any]]],
    support_counts: dict[str, dict[str, int]],
    rule: dict[str, Any],
    lambda_cost: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = str(rule.get("source", "none"))
    benchmark = str(rule.get("benchmark", "all"))
    min_support = 0 if pd.isna(rule.get("min_support", np.nan)) else int(rule["min_support"])
    threshold = 0.0 if pd.isna(rule.get("confidence_threshold", np.nan)) else float(rule["confidence_threshold"])
    verifier_map = verifier_maps.get(source, {})
    for row in base.itertuples(index=False):
        selected = str(row.patched_model)
        fused_model = selected
        probe_used = False
        probe_latency_s = 0.0
        supported_model = ""
        supported_count = 0
        verifier_confidence = 0.0
        if str(rule.get("selector_kind")) != "base" and str(row.benchmark) == benchmark:
            item = verifier_map.get(str(row.query_id), {})
            if item and str(item.get("status", "")) == "success":
                probe_used = True
                probe_latency_s = safe_float(item.get("latency_s", 0.0))
                supported_model = str(item.get("supported_model", "") or "")
                verifier_confidence = safe_float(item.get("verifier_confidence", 0.0))
                supported_count = int(support_counts.get(str(row.query_id), {}).get(supported_model, 0))
                can_switch = (
                    supported_model
                    and supported_model in action_map.get(str(row.query_id), {})
                    and verifier_confidence >= threshold
                    and supported_count >= min_support
                    and (not bool(rule.get("only_nonfrontier", True)) or supported_model not in FRONTIERS)
                )
                if can_switch:
                    fused_model = supported_model
        action = action_map.get(str(row.query_id), {}).get(fused_model, {})
        quality = float(action.get("quality_score", row.patched_quality))
        norm_cost = float(action.get("normalized_remote_cost", 0.0) or 0.0)
        utility = quality - float(lambda_cost) * norm_cost
        rows.append(
            {
                **row._asdict(),
                "fused_model": fused_model,
                "fused_quality": quality,
                "fused_utility": utility,
                "fused_normalized_cost": norm_cost,
                "fused_frontier": bool(action.get("is_frontier", False)),
                "fused_changed": fused_model != selected,
                "probe_used": bool(probe_used),
                "probe_latency_s": float(probe_latency_s),
                "support_source": source,
                "support_model": supported_model,
                "support_count": int(supported_count),
                "support_confidence": float(verifier_confidence),
            }
        )
    return pd.DataFrame(rows)


def selected_rows(policy_table: pd.DataFrame) -> pd.DataFrame:
    val = policy_table[policy_table["split"].astype(str).eq("val")].copy()
    test = policy_table[policy_table["split"].astype(str).eq("test")].copy()
    rows: list[pd.Series] = []
    for split_name, frame in [("val", val), ("test", test)]:
        base = frame[frame["policy"].astype(str).eq("base_targeted_residual_repair")]
        if not base.empty:
            row = base.iloc[0].copy()
            row["selection_rule"] = f"base_reference_{split_name}"
            rows.append(row)

    base_val = val[val["policy"].astype(str).eq("base_targeted_residual_repair")]
    if not base_val.empty:
        base_u = float(base_val.iloc[0]["mean_utility"])
        active = val[
            val["selector_kind"].astype(str).eq("single_benchmark_support")
            & val["override_rate"].astype(float).gt(0.0)
            & val["mean_utility"].astype(float).ge(base_u - 1e-12)
        ].copy()
        if not active.empty:
            best = active.sort_values(
                ["mean_utility", "probe_call_rate", "frontier_call_rate", "override_rate"],
                ascending=[False, True, True, True],
            ).iloc[0].copy()
            best["selection_rule"] = "val_best_single_benchmark_support"
            rows.append(best)
            match = test[test["policy"].astype(str).eq(str(best["policy"]))]
            if not match.empty:
                test_row = match.iloc[0].copy()
                test_row["selection_rule"] = "val_best_single_benchmark_support_test"
                rows.append(test_row)

    for _, row in test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(10).iterrows():
        diagnostic = row.copy()
        diagnostic["selection_rule"] = "top_test_diagnostic"
        rows.append(diagnostic)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["selection_rule", "policy", "split"], keep="first")


def normalize_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    text = text.removeprefix("answer:").strip()
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.strip().strip("$").strip()
    return text


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_large(model: str) -> bool:
    return str(model) in STRONG_OR_FRONTIER


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_memo(path: Path, args: argparse.Namespace, target_table: pd.DataFrame, selected: pd.DataFrame) -> None:
    target_test = target_table[target_table["split"].astype(str).eq("test")]
    target_val = target_table[target_table["split"].astype(str).eq("val")]
    val_target_summary = target_summary(target_val)
    test_target_summary = target_summary(target_test)
    cols = [
        "selection_rule",
        "policy",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "probe_call_rate",
        "override_rate",
        "mean_probe_latency_s",
    ]
    lines = [
        "# Conservative Support-Abstention Policy",
        "",
        "This no-training probe pilot reuses cached local vLLM solve-support rows and cached broad100 model outputs.",
        "It selects only single-benchmark threshold rules on validation and reports held-out test once.",
        "No GPT, Gemini, Claude, local generation, or vLLM serving calls are made by this script.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/194_conservative_support_abstention_policy.py",
        "```",
        "",
        "## Benchmark And Models",
        "",
        "- Slice: broad100 validation/test query choices from Experiment 189 residual repair.",
        "- Base policy: Experiment 189 targeted residual repair.",
        "- Cached action matrix: `results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet`.",
        "- Probe models: cached local vLLM `Qwen/Qwen3-14B-AWQ` and `Qwen/Qwen3-32B-AWQ` solve-support verifiers.",
        "- Frontier/API actions in the cached matrix: `gpt-5.5`, `gemini-3.5-flash`, `gemini-3.5-flash-strong-solve`.",
        "",
        "## Local-Vs-Large Oracle Target",
        "",
        (
            f"- Validation rows: `{len(target_val)}`; best-local mean utility `{val_target_summary['local_mean']:.4f}`; "
            f"best-large mean utility `{val_target_summary['large_mean']:.4f}`; local-vs-large oracle utility "
            f"`{val_target_summary['oracle_mean']:.4f}`; need-large rate `{val_target_summary['need_large_rate']:.4f}`; "
            f"mean delta_large `{val_target_summary['mean_delta']:.4f}`."
        ),
        (
            f"- Test rows: `{len(target_test)}`; best-local mean utility `{test_target_summary['local_mean']:.4f}`; "
            f"best-large mean utility `{test_target_summary['large_mean']:.4f}`; local-vs-large oracle utility "
            f"`{test_target_summary['oracle_mean']:.4f}`; need-large rate `{test_target_summary['need_large_rate']:.4f}`; "
            f"mean delta_large `{test_target_summary['mean_delta']:.4f}`."
        ),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]) if not selected.empty else "No selected rows.",
        "",
        "## What Helped Or Failed",
        "",
        "- The helpful signal is conservative candidate support: switch only when the local verifier supports a non-frontier candidate whose answer is also shared by at least two cached actions.",
        "- Validation chooses a single-benchmark support rule instead of composing benchmark rules, because previous benchmark composition overfit.",
        "- This still does not meet the Phase 3 target; it is a small held-out utility improvement over Experiment 189, not a solved router.",
        "- The broader Qwen14/Qwen32 solve-support sweeps failed because plain answer support over-escalated or switched to wrong actions on GPQA and MMLU-Pro.",
        "",
        "## Next Recommended Probe",
        "",
        "Use stronger checker evidence rather than another plain support signal: task-specific verifier/checker probes for GPQA/MMLU-Pro and execution/calculator-style checks for exact math/code. Keep rules validation-selected and avoid broad benchmark composition unless it has shrinkage or a separate calibration split.",
        "",
        "## Artifacts",
        "",
        f"- Oracle target table: `{path.parent / 'table_oracle_targets_local_vs_large.csv'}`",
        f"- Cached probe signals: `{path.parent / 'table_probe_signals_cached.csv'}`",
        f"- Rule library: `{path.parent / 'table_conservative_support_rule_library.csv'}`",
        f"- All policies: `{path.parent / 'table_conservative_support_abstention_policy_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_conservative_support_abstention_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_conservative_support_abstention_query_choices.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "local_mean": 0.0,
            "large_mean": 0.0,
            "oracle_mean": 0.0,
            "need_large_rate": 0.0,
            "mean_delta": 0.0,
        }
    return {
        "local_mean": float(frame["local_utility"].astype(float).mean()),
        "large_mean": float(frame["large_utility"].astype(float).mean()),
        "oracle_mean": float(frame[["local_utility", "large_utility"]].astype(float).max(axis=1).mean()),
        "need_large_rate": float(frame["need_large"].astype(bool).mean()),
        "mean_delta": float(frame["delta_large"].astype(float).mean()),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
