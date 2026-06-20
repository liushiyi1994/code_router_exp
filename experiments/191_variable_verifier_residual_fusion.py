from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_POLICY = "scopegpqa+bbh+gsm8k+mmlupro_selected_qwen32_qwen3-14b-awq-local_none"
FRONTIERS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve", "gpt-5.5"}
STRONG_OR_FRONTIER = {"qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local", *FRONTIERS}
THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]
MODES = ["always", "if_changed_false", "if_selected_large", "if_selected_qwen32", "if_not_frontier"]
SCOPES = [("gpqa",), ("mmlupro",), ("gpqa", "mmlupro")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse variable-option MCQ verifier support with the targeted residual repair policy."
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_targeted_residual_repair_policy/"
            "table_targeted_residual_repair_query_choices.csv"
        ),
    )
    parser.add_argument(
        "--base-policy",
        default=BASE_POLICY,
    )
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path(
            "results/controlled/broad100_variable_option_mcq_verifier_policy/"
            "table_variable_option_mcq_verifier_outputs.csv"
        ),
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
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_variable_verifier_residual_fusion"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--reliable-quality-threshold", type=float, default=0.85)
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
    verifier = pd.read_csv(args.verifier)
    action_map = {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs.groupby("query_id")}
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][["query_id", "utility", "quality_score"]].rename(
        columns={"utility": "oracle_utility", "quality_score": "oracle_quality"}
    )
    base = drop_prefixed(base, ["oracle_utility", "oracle_quality"]).merge(oracle, on="query_id", how="left")
    verifier_diag = verifier_diagnostics(verifier)
    rules = enumerate_rules()
    policy_table, query_choices = evaluate_rules(base, verifier, outputs, action_map, rules, args)
    reliable_raw = build_reliable_policy(
        policy_table,
        verifier_diag,
        base,
        metric="mean_utility",
        quality_threshold=float(args.reliable_quality_threshold),
    )
    reliable_costed = build_reliable_policy(
        policy_table,
        verifier_diag,
        base,
        metric="mean_utility_with_probe_cost",
        quality_threshold=float(args.reliable_quality_threshold),
    )
    if reliable_raw:
        rules.append(reliable_raw)
    if reliable_costed:
        rules.append(reliable_costed)
    if reliable_raw or reliable_costed:
        policy_table, query_choices = evaluate_rules(base, verifier, outputs, action_map, rules, args)
    selected = selected_rows(policy_table, args)
    selected_policies = set(selected["policy"].astype(str).tolist()) if not selected.empty else set()
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    verifier_diag.to_csv(args.output_dir / "table_variable_verifier_fusion_verifier_diagnostics.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output_dir / "table_variable_verifier_fusion_rule_library.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_variable_verifier_fusion_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_variable_verifier_fusion_policy_selected.csv", index=False)
    query_choices_to_write.to_csv(args.output_dir / "table_variable_verifier_fusion_query_choices.csv", index=False)
    write_memo(args.output_dir / "VARIABLE_VERIFIER_RESIDUAL_FUSION_MEMO.md", args, verifier_diag, selected)
    print(f"Wrote variable-verifier residual fusion results to {args.output_dir}")


def drop_prefixed(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    cols = [col for col in frame.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
    return frame.drop(columns=cols, errors="ignore")


def verifier_diagnostics(verifier: pd.DataFrame) -> pd.DataFrame:
    return (
        verifier.groupby(["benchmark", "split"], as_index=False)
        .agg(
            n=("query_id", "count"),
            verifier_quality=("quality_score", "mean"),
            verifier_confidence=("verifier_confidence", "mean"),
            verifier_cost_total=("cost_total_usd", "sum"),
        )
        .sort_values(["split", "benchmark"])
    )


def enumerate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_targeted_residual_repair",
            "scope_json": "[]",
            "threshold": np.nan,
            "mode": "base",
            "selector_kind": "base",
        }
    ]
    for scope in SCOPES:
        for threshold in THRESHOLDS:
            for mode in MODES:
                rules.append(
                    {
                        "policy": f"scope{'+'.join(scope)}_thr{threshold:g}_{mode}",
                        "scope_json": json.dumps(scope),
                        "threshold": float(threshold),
                        "mode": mode,
                        "selector_kind": "grid",
                    }
                )
    return rules


def build_reliable_policy(
    policy_table: pd.DataFrame,
    verifier_diag: pd.DataFrame,
    base: pd.DataFrame,
    *,
    metric: str,
    quality_threshold: float,
) -> dict[str, Any] | None:
    val_base = policy_table[policy_table["split"].eq("val") & policy_table["policy"].eq("base_targeted_residual_repair")]
    if val_base.empty:
        return None
    base_value = float(val_base.iloc[0][metric])
    val_diag = verifier_diag[verifier_diag["split"].eq("val")]
    reliable = set(
        val_diag[val_diag["verifier_quality"].astype(float).ge(float(quality_threshold))]["benchmark"].astype(str).tolist()
    )
    if not reliable:
        return None
    chosen: dict[str, str] = {}
    for benchmark in sorted(reliable):
        candidates = policy_table[
            policy_table["split"].eq("val")
            & policy_table["policy"].str.startswith(f"scope{benchmark}_")
        ].copy()
        candidates = candidates[candidates[metric].astype(float).gt(base_value + 1e-12)]
        if candidates.empty:
            continue
        best = candidates.sort_values([metric, "probe_call_rate", "frontier_call_rate"], ascending=[False, True, True]).iloc[0]
        chosen[benchmark] = str(best["policy"])
    if not chosen:
        return {
            "policy": f"reliable_{metric}_no_active_benchmark",
            "scope_json": json.dumps({}),
            "threshold": np.nan,
            "mode": "composed",
            "selector_kind": f"reliable_{metric}",
            "mapping_json": json.dumps({}, sort_keys=True),
        }
    return {
        "policy": f"reliable_{metric}_benchmark_support",
        "scope_json": json.dumps(tuple(sorted(chosen))),
        "threshold": np.nan,
        "mode": "composed",
        "selector_kind": f"reliable_{metric}",
        "mapping_json": json.dumps(chosen, sort_keys=True),
    }


def evaluate_rules(
    base: pd.DataFrame,
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    rules: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    verifier_map = verifier.set_index("query_id").to_dict("index")
    gpt_cost = mean_gpt_cost(outputs)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for rule in rules:
        choices = apply_rule(base, verifier_map, action_map, rule, gpt_cost, float(args.lambda_cost))
        details.append(choices.assign(policy=str(rule["policy"])))
        for split, group in choices.groupby("split", dropna=False):
            values = group["fused_utility"].astype(float).to_numpy()
            cost_values = group["fused_utility_with_probe_cost"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            oracle_q = float(group["oracle_quality"].astype(float).mean())
            mean_u = float(values.mean())
            mean_cost_u = float(cost_values.mean())
            rows.append(
                {
                    **rule,
                    "split": split,
                    "n_queries": len(group),
                    "mean_quality": float(group["fused_quality"].astype(float).mean()),
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "mean_utility_with_probe_cost": mean_cost_u,
                    "cost_oracle_mean_utility": oracle_u,
                    "quality_oracle_mean_quality": oracle_q,
                    "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
                    "oracle_utility_ratio_with_probe_cost": mean_cost_u / max(oracle_u, 1e-12),
                    "quality_gap_to_oracle": oracle_q - float(group["fused_quality"].astype(float).mean()),
                    "frontier_call_rate": float(group["fused_frontier"].mean()),
                    "strong_or_frontier_call_rate": float(group["fused_model"].astype(str).map(is_large).mean()),
                    "probe_call_rate": float(group["probe_used"].mean()),
                    "override_rate": float(group["fused_changed"].mean()),
                    "extra_probe_norm_cost_mean": float(group["probe_norm_cost"].mean()),
                    "selected_models_json": json.dumps(group["fused_model"].value_counts().sort_index().to_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def apply_rule(
    base: pd.DataFrame,
    verifier_map: dict[str, dict[str, Any]],
    action_map: dict[str, dict[str, dict[str, Any]]],
    rule: dict[str, Any],
    gpt_cost: float,
    lambda_cost: float,
) -> pd.DataFrame:
    if str(rule.get("mode")) == "composed":
        mapping = json.loads(str(rule.get("mapping_json", "{}")))
        rows = []
        for row in base.itertuples(index=False):
            benchmark = str(row.benchmark)
            if benchmark in mapping:
                chosen = apply_rule(pd.DataFrame([row._asdict()]), verifier_map, action_map, lookup_rule(mapping[benchmark]), gpt_cost, lambda_cost)
                rows.extend(chosen.to_dict("records"))
            else:
                rows.extend(apply_base_row(row, action_map, probe_cost=0.0, lambda_cost=lambda_cost).to_dict("records"))
        return pd.DataFrame(rows)
    scope = set(json.loads(str(rule.get("scope_json", "[]"))))
    threshold = 0.0 if pd.isna(rule.get("threshold", np.nan)) else float(rule["threshold"])
    mode = str(rule.get("mode", "base"))
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        model = str(row.patched_model)
        probe_used = False
        probe_cost = 0.0
        if mode != "base" and str(row.benchmark) in scope:
            item = verifier_map.get(str(row.query_id), {})
            if item and str(item.get("status", "")) == "success":
                probe_used = True
                probe_cost = float(item.get("cost_total_usd", 0.0) or 0.0) / max(gpt_cost, 1e-12)
                supported = str(item.get("supported_model", "") or "")
                confidence = float(item.get("verifier_confidence", 0.0) or 0.0)
                if confidence >= threshold and supported.upper() != "NONE" and supported in action_map.get(str(row.query_id), {}):
                    if mode == "always":
                        model = supported
                    elif mode == "if_changed_false" and not bool(row.changed):
                        model = supported
                    elif mode == "if_selected_large" and is_large(model):
                        model = supported
                    elif mode == "if_selected_qwen32" and model == "qwen3-32b-awq-local":
                        model = supported
                    elif mode == "if_not_frontier" and model not in FRONTIERS:
                        model = supported
        rows.extend(apply_base_row(row, action_map, model=model, probe_used=probe_used, probe_cost=probe_cost, lambda_cost=lambda_cost).to_dict("records"))
    return pd.DataFrame(rows)


def lookup_rule(policy: str) -> dict[str, Any]:
    if policy == "base_targeted_residual_repair":
        return {"policy": policy, "scope_json": "[]", "threshold": np.nan, "mode": "base"}
    prefix, mode = policy.rsplit("_", 1)
    prefix, threshold_text = prefix.rsplit("_thr", 1)
    scope = tuple(prefix.removeprefix("scope").split("+"))
    return {"policy": policy, "scope_json": json.dumps(scope), "threshold": float(threshold_text), "mode": mode}


def apply_base_row(
    row: Any,
    action_map: dict[str, dict[str, dict[str, Any]]],
    *,
    model: str | None = None,
    probe_used: bool = False,
    probe_cost: float = 0.0,
    lambda_cost: float,
) -> pd.DataFrame:
    selected = str(row.patched_model if model is None else model)
    action = action_map.get(str(row.query_id), {}).get(selected, {})
    quality = float(action.get("quality_score", row.patched_quality))
    base_cost = float(action.get("normalized_remote_cost", 0.0) or 0.0)
    utility = quality - float(lambda_cost) * base_cost
    return pd.DataFrame(
        [
            {
                **row._asdict(),
                "fused_model": selected,
                "fused_quality": quality,
                "fused_utility": utility,
                "fused_utility_with_probe_cost": quality - float(lambda_cost) * (base_cost + float(probe_cost)),
                "fused_frontier": bool(action.get("is_frontier", False)),
                "fused_changed": selected != str(row.patched_model),
                "probe_used": bool(probe_used),
                "probe_norm_cost": float(probe_cost),
            }
        ]
    )


def selected_rows(policy_table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    val = policy_table[policy_table["split"].eq("val")].copy()
    test = policy_table[policy_table["split"].eq("test")].copy()
    rows: list[pd.Series] = []
    selectors = {
        "val_best_mean_utility": ("mean_utility", val),
        "val_best_probe_cost_utility": ("mean_utility_with_probe_cost", val),
    }
    for rule, (metric, frame) in selectors.items():
        if frame.empty:
            continue
        best = frame.sort_values([metric, "probe_call_rate", "frontier_call_rate"], ascending=[False, True, True]).iloc[0]
        best = best.copy()
        best["selection_rule"] = rule
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = f"{rule}_test"
            rows.append(test_row)
    for policy in ["reliable_mean_utility_benchmark_support", "reliable_mean_utility_with_probe_cost_benchmark_support", "reliable_mean_utility_with_probe_cost_no_active_benchmark"]:
        for split, frame in [("val", val), ("test", test)]:
            match = frame[frame["policy"].astype(str).eq(policy)]
            if not match.empty:
                row = match.iloc[0].copy()
                row["selection_rule"] = f"{policy}_{split}"
                rows.append(row)
    top_test = test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(8)
    for _, row in top_test.iterrows():
        row = row.copy()
        row["selection_rule"] = "top_test_diagnostic"
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(["selection_rule", "policy", "split"], keep="first")


def mean_gpt_cost(outputs: pd.DataFrame) -> float:
    gpt = outputs[outputs["model_id"].astype(str).eq("gpt-5.5")]
    return max(float(gpt.groupby("query_id")["cost_total_usd"].mean().mean()), 1e-12)


def is_large(model: str) -> bool:
    return str(model) in STRONG_OR_FRONTIER


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_memo(path: Path, args: argparse.Namespace, verifier_diag: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "policy",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_probe_cost",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "frontier_call_rate",
        "probe_call_rate",
        "override_rate",
    ]
    lines = [
        "# Variable-Verifier Residual Fusion",
        "",
        "This is a no-new-call fusion of Experiment 190 variable-option verifier support with Experiment 189 residual repair.",
        "It reports both raw selected-action utility and utility after charging GPT verifier probe cost.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/191_variable_verifier_residual_fusion.py",
        "```",
        "",
        "## Verifier Diagnostics",
        "",
        markdown_table(verifier_diag),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[cols]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- A real deployable improvement must beat Experiment 189's `0.7736` held-out utility and remain useful after charging verifier probe cost.",
        "- Reliable raw benchmark support is allowed to select only benchmarks with validation verifier quality above the configured threshold.",
        "- Probe-cost selection is the stricter route-time metric because GPT verifier calls are paid probes.",
        "",
        "## Artifacts",
        "",
        f"- Verifier diagnostics: `{path.parent / 'table_variable_verifier_fusion_verifier_diagnostics.csv'}`",
        f"- Rule library: `{path.parent / 'table_variable_verifier_fusion_rule_library.csv'}`",
        f"- All policies: `{path.parent / 'table_variable_verifier_fusion_policy_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_variable_verifier_fusion_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_variable_verifier_fusion_query_choices.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
