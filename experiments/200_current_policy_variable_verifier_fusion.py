from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_POLICY = "local_majority_scopegsm8k_votes2_if_base_frontier_cheapest"
FRONTIERS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve", "gpt-5.5"}
THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]
SCOPES = [("gpqa",), ("mmlupro",), ("gpqa", "mmlupro")]
MODES = [
    "always",
    "if_base_frontier",
    "if_base_not_frontier",
    "if_supported_not_frontier",
    "if_base_frontier_supported_not_frontier",
    "if_different",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse cached variable-option MCQ verifier support with the current best "
            "deployable-ish broad100 policy. No new provider or vLLM calls are made."
        )
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_local_consensus_cost_suppression_audit/"
            "table_local_consensus_cost_suppression_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default=BASE_POLICY)
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
        default=Path("results/controlled/broad100_current_policy_variable_verifier_fusion"),
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
    action_map = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id")
    }
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][
        ["query_id", "utility", "quality_score"]
    ].rename(columns={"utility": "oracle_utility", "quality_score": "oracle_quality"})

    base = pd.read_csv(args.base_query_choices)
    base = base[base["policy"].astype(str).eq(str(args.base_policy))].copy()
    if base.empty:
        raise RuntimeError(f"Base policy {args.base_policy!r} not found in {args.base_query_choices}.")
    base = drop_prefixed(base, ["oracle_utility", "oracle_quality"]).merge(oracle, on="query_id", how="left")

    verifier = pd.read_csv(args.verifier)
    verifier_diag = verifier_diagnostics(verifier)
    rules = enumerate_rules()
    policy_table, query_choices = evaluate_rules(base, verifier, action_map, rules, args, outputs)

    reliable_raw = build_reliable_policy(
        policy_table,
        verifier_diag,
        metric="mean_utility",
        quality_threshold=float(args.reliable_quality_threshold),
    )
    reliable_costed = build_reliable_policy(
        policy_table,
        verifier_diag,
        metric="mean_utility_with_probe_cost",
        quality_threshold=float(args.reliable_quality_threshold),
    )
    for rule in [reliable_raw, reliable_costed]:
        if rule is not None:
            rules.append(rule)
    if reliable_raw is not None or reliable_costed is not None:
        policy_table, query_choices = evaluate_rules(base, verifier, action_map, rules, args, outputs)

    selected = selected_rows(policy_table)
    selected_policies = set(selected["policy"].astype(str).tolist())
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    verifier_diag.to_csv(args.output_dir / "table_current_policy_variable_verifier_diagnostics.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output_dir / "table_current_policy_variable_verifier_rules.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_current_policy_variable_verifier_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_current_policy_variable_verifier_selected.csv", index=False)
    query_choices_to_write.to_csv(
        args.output_dir / "table_current_policy_variable_verifier_query_choices.csv", index=False
    )
    write_memo(args.output_dir / "CURRENT_POLICY_VARIABLE_VERIFIER_FUSION_MEMO.md", args, verifier_diag, selected)
    print(f"Wrote current-policy variable-verifier fusion results to {args.output_dir}")


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
            verifier_cost_total_usd=("cost_total_usd", "sum"),
        )
        .sort_values(["split", "benchmark"])
    )


def enumerate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_current_policy",
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
    *,
    metric: str,
    quality_threshold: float,
) -> dict[str, Any] | None:
    val = policy_table[policy_table["split"].astype(str).eq("val")].copy()
    base = val[val["policy"].astype(str).eq("base_current_policy")]
    if base.empty:
        return None
    base_value = float(base.iloc[0][metric])
    reliable = set(
        verifier_diag[
            verifier_diag["split"].astype(str).eq("val")
            & verifier_diag["verifier_quality"].astype(float).ge(float(quality_threshold))
        ]["benchmark"].astype(str)
    )
    if not reliable:
        return {
            "policy": f"reliable_{metric}_no_active_benchmark",
            "scope_json": json.dumps({}),
            "threshold": np.nan,
            "mode": "composed",
            "selector_kind": f"reliable_{metric}",
            "mapping_json": json.dumps({}, sort_keys=True),
        }

    chosen: dict[str, str] = {}
    for benchmark in sorted(reliable):
        candidates = val[val["policy"].astype(str).str.startswith(f"scope{benchmark}_")].copy()
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
    action_map: dict[str, dict[str, dict[str, Any]]],
    rules: list[dict[str, Any]],
    args: argparse.Namespace,
    outputs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    verifier_map = verifier.set_index("query_id").to_dict("index")
    gpt_cost = mean_gpt_cost(outputs)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for rule in rules:
        choices = apply_rule(base, verifier_map, action_map, rule, gpt_cost, float(args.lambda_cost))
        details.append(choices.assign(policy=str(rule["policy"])))
        for split, group in choices.groupby("split", dropna=False):
            raw_values = group["fused_utility"].astype(float).to_numpy()
            probe_values = group["fused_utility_with_probe_cost"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci(raw_values, int(args.bootstrap_samples), int(args.seed))
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            oracle_q = float(group["oracle_quality"].astype(float).mean())
            mean_u = float(raw_values.mean())
            mean_q = float(group["fused_quality"].astype(float).mean())
            mean_probe_u = float(probe_values.mean())
            rows.append(
                {
                    **rule,
                    "split": split,
                    "n_queries": int(len(group)),
                    "mean_quality": mean_q,
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "mean_utility_with_probe_cost": mean_probe_u,
                    "cost_oracle_mean_utility": oracle_u,
                    "quality_oracle_mean_quality": oracle_q,
                    "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
                    "oracle_utility_ratio_with_probe_cost": mean_probe_u / max(oracle_u, 1e-12),
                    "utility_gap_to_oracle": oracle_u - mean_u,
                    "quality_gap_to_oracle": oracle_q - mean_q,
                    "frontier_call_rate": float(group["fused_frontier"].mean()),
                    "probe_call_rate": float(group["probe_used"].mean()),
                    "override_rate": float(group["fused_changed"].mean()),
                    "extra_probe_norm_cost_mean": float(group["probe_norm_cost"].mean()),
                    "selected_models_json": json.dumps(
                        group["fused_model"].value_counts().sort_index().to_dict(), sort_keys=True
                    ),
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
        rows: list[dict[str, Any]] = []
        for row in base.itertuples(index=False):
            benchmark = str(row.benchmark)
            if benchmark in mapping:
                chosen = apply_rule(
                    pd.DataFrame([row._asdict()]),
                    verifier_map,
                    action_map,
                    lookup_rule(mapping[benchmark]),
                    gpt_cost,
                    lambda_cost,
                )
                rows.extend(chosen.to_dict("records"))
            else:
                rows.append(apply_base_row(row, action_map, lambda_cost=lambda_cost))
        return pd.DataFrame(rows)

    scope = set(json.loads(str(rule.get("scope_json", "[]"))))
    threshold = 0.0 if pd.isna(rule.get("threshold", np.nan)) else float(rule["threshold"])
    mode = str(rule.get("mode", "base"))
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        selected = str(row.selected_model)
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
                        selected = supported
                    elif mode == "if_base_frontier" and bool(row.selected_frontier):
                        selected = supported
                    elif mode == "if_base_not_frontier" and not bool(row.selected_frontier):
                        selected = supported
                    elif mode == "if_supported_not_frontier" and supported not in FRONTIERS:
                        selected = supported
                    elif (
                        mode == "if_base_frontier_supported_not_frontier"
                        and bool(row.selected_frontier)
                        and supported not in FRONTIERS
                    ):
                        selected = supported
                    elif mode == "if_different" and supported != str(row.selected_model):
                        selected = supported
        rows.append(
            apply_base_row(
                row,
                action_map,
                model=selected,
                probe_used=probe_used,
                probe_cost=probe_cost,
                lambda_cost=lambda_cost,
            )
        )
    return pd.DataFrame(rows)


def lookup_rule(policy: str) -> dict[str, Any]:
    if policy == "base_current_policy":
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
) -> dict[str, Any]:
    selected = str(row.selected_model if model is None else model)
    action = action_map.get(str(row.query_id), {}).get(selected, {})
    quality = float(action.get("quality_score", row.selected_quality))
    base_cost = float(action.get("normalized_remote_cost", 0.0) or 0.0)
    utility = quality - float(lambda_cost) * base_cost
    return {
        **row._asdict(),
        "fused_model": selected,
        "fused_quality": quality,
        "fused_utility": utility,
        "fused_utility_with_probe_cost": quality - float(lambda_cost) * (base_cost + float(probe_cost)),
        "fused_frontier": bool(action.get("is_frontier", False)),
        "fused_changed": selected != str(row.selected_model),
        "probe_used": bool(probe_used),
        "probe_norm_cost": float(probe_cost),
    }


def selected_rows(policy_table: pd.DataFrame) -> pd.DataFrame:
    val = policy_table[policy_table["split"].astype(str).eq("val")].copy()
    test = policy_table[policy_table["split"].astype(str).eq("test")].copy()
    rows: list[pd.Series] = []

    for split, frame in [("val", val), ("test", test)]:
        base = frame[frame["policy"].astype(str).eq("base_current_policy")]
        if not base.empty:
            row = base.iloc[0].copy()
            row["selection_rule"] = f"base_reference_{split}"
            rows.append(row)

    for selection_rule, metric in [
        ("val_best_mean_utility", "mean_utility"),
        ("val_best_probe_cost_utility", "mean_utility_with_probe_cost"),
    ]:
        if val.empty:
            continue
        candidates = val[val["policy"].astype(str).ne("base_current_policy")].copy()
        if candidates.empty:
            continue
        best = candidates.sort_values([metric, "probe_call_rate", "frontier_call_rate"], ascending=[False, True, True]).iloc[0]
        best = best.copy()
        best["selection_rule"] = selection_rule
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = f"{selection_rule}_test"
            rows.append(test_row)

    for policy in [
        "reliable_mean_utility_benchmark_support",
        "reliable_mean_utility_with_probe_cost_benchmark_support",
        "reliable_mean_utility_no_active_benchmark",
        "reliable_mean_utility_with_probe_cost_no_active_benchmark",
    ]:
        for split, frame in [("val", val), ("test", test)]:
            match = frame[frame["policy"].astype(str).eq(policy)]
            if not match.empty:
                row = match.iloc[0].copy()
                row["selection_rule"] = f"{policy}_{split}"
                rows.append(row)

    for _, row in test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(10).iterrows():
        diagnostic = row.copy()
        diagnostic["selection_rule"] = "top_test_diagnostic"
        rows.append(diagnostic)

    return pd.DataFrame(rows).drop_duplicates(["selection_rule", "policy", "split"], keep="first")


def mean_gpt_cost(outputs: pd.DataFrame) -> float:
    gpt = outputs[outputs["model_id"].astype(str).eq("gpt-5.5")]
    if gpt.empty:
        return 1.0
    return max(float(gpt.groupby("query_id")["cost_total_usd"].mean().mean()), 1e-12)


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
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "frontier_call_rate",
        "probe_call_rate",
        "override_rate",
    ]
    present_cols = [column for column in cols if column in selected.columns]
    lines = [
        "# Current-Policy Variable-Verifier Fusion",
        "",
        "This is a no-new-call replay that adds cached GPT-5.5 variable-option MCQ verifier support to the current",
        f"best broad100 policy: `{args.base_policy}`.",
        "",
        "The point is not to claim a solved router. It tests whether benchmark-scoped answer support can repair",
        "the residual action-identity mistakes without training a new router. Raw selected-action utility and",
        "probe-cost-adjusted utility are both reported.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/200_current_policy_variable_verifier_fusion.py",
        "```",
        "",
        "## Verifier Diagnostics",
        "",
        markdown_table(verifier_diag),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[present_cols]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- Validation-only utility selection overfits toward GPQA support; GPQA verifier quality is lower and test utility falls.",
        "- The reliability-constrained selector admits benchmarks with validation verifier quality above the configured threshold.",
        "- With the default threshold, that keeps MMLUPro and rejects GPQA. This helps raw held-out utility but still misses the 95% oracle target.",
        "- Probe-cost-adjusted utility is lower because the cached GPT verifier is an expensive route-time probe.",
        "",
        "## Artifacts",
        "",
        f"- Diagnostics: `{path.parent / 'table_current_policy_variable_verifier_diagnostics.csv'}`",
        f"- Rule library: `{path.parent / 'table_current_policy_variable_verifier_rules.csv'}`",
        f"- All policies: `{path.parent / 'table_current_policy_variable_verifier_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_current_policy_variable_verifier_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_current_policy_variable_verifier_query_choices.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
