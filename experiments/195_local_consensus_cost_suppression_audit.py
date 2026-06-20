from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_POLICY = "qwen14_bbh_support2_conf0_nonfrontier"
LOCAL_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
EXACT_MATH_BENCHMARKS = ("gsm8k", "math500", "livemathbench")
EXTENDED_EXACT_MATH_BENCHMARKS = ("aime", "gsm8k", "math500", "livemathbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit deployable local-consensus and diagnostic same-answer cost suppression."
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
            "results/controlled/broad100_conservative_support_abstention_policy/"
            "table_conservative_support_abstention_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default=BASE_POLICY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_local_consensus_cost_suppression_audit"),
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
    action_map = {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs.groupby("query_id")}
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][
        ["query_id", "utility", "quality_score"]
    ].rename(columns={"utility": "oracle_utility", "quality_score": "oracle_quality"})
    base = drop_prefixed(base, ["oracle_utility", "oracle_quality"]).merge(oracle, on="query_id", how="left")

    rules = enumerate_rules()
    policy_table, query_choices = evaluate_rules(base, action_map, rules, args)
    selected = selected_rows(policy_table)
    selected_policies = set(selected["policy"].astype(str).tolist())
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    pd.DataFrame(rules).to_csv(args.output_dir / "table_local_consensus_cost_suppression_rules.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_local_consensus_cost_suppression_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_local_consensus_cost_suppression_selected.csv", index=False)
    query_choices_to_write.to_csv(args.output_dir / "table_local_consensus_cost_suppression_query_choices.csv", index=False)
    write_memo(args.output_dir / "LOCAL_CONSENSUS_COST_SUPPRESSION_MEMO.md", args, selected)
    print(f"Wrote local consensus cost-suppression audit to {args.output_dir}")


def drop_prefixed(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    cols = [col for col in frame.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
    return frame.drop(columns=cols, errors="ignore")


def enumerate_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_exp194",
            "family": "reference",
            "scope_json": "[]",
            "mode": "base",
            "min_votes": np.nan,
            "order": "none",
            "deployable": True,
        }
    ]
    scopes = [
        ("gsm8k",),
        ("math500",),
        ("livemathbench",),
        ("math500", "livemathbench"),
        EXACT_MATH_BENCHMARKS,
        EXTENDED_EXACT_MATH_BENCHMARKS,
        ("gpqa",),
        ("mmlupro",),
    ]
    for scope in scopes:
        for min_votes in [2, 3, 4, 5, 6]:
            for condition in ["always", "if_base_frontier", "if_base_not_frontier"]:
                for order in ["cheapest", "strongest"]:
                    rules.append(
                        {
                            "policy": f"local_majority_scope{'+'.join(scope)}_votes{min_votes}_{condition}_{order}",
                            "family": "deployable_local_consensus",
                            "scope_json": json.dumps(scope),
                            "mode": condition,
                            "min_votes": int(min_votes),
                            "order": order,
                            "deployable": True,
                        }
                    )
    for scope in [
        ("math500",),
        ("livemathbench",),
        ("math500", "livemathbench"),
        EXACT_MATH_BENCHMARKS,
        EXTENDED_EXACT_MATH_BENCHMARKS,
    ]:
        for condition in ["always", "if_base_frontier"]:
            rules.append(
                {
                    "policy": f"diagnostic_same_answer_scope{'+'.join(scope)}_{condition}",
                    "family": "diagnostic_posthoc_same_answer",
                    "scope_json": json.dumps(scope),
                    "mode": condition,
                    "min_votes": np.nan,
                    "order": "cheapest",
                    "deployable": False,
                }
            )
    return rules


def evaluate_rules(
    base: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    rules: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for rule in rules:
        choices = apply_rule(base, action_map, rule, float(args.lambda_cost))
        details.append(choices.assign(policy=str(rule["policy"])))
        for split, group in choices.groupby("split", dropna=False):
            values = group["selected_utility"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            oracle_q = float(group["oracle_quality"].astype(float).mean())
            mean_u = float(values.mean())
            mean_q = float(group["selected_quality"].astype(float).mean())
            rows.append(
                {
                    **rule,
                    "split": split,
                    "n_queries": int(len(group)),
                    "mean_quality": mean_q,
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "cost_oracle_mean_utility": oracle_u,
                    "quality_oracle_mean_quality": oracle_q,
                    "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
                    "utility_gap_to_oracle": oracle_u - mean_u,
                    "quality_gap_to_oracle": oracle_q - mean_q,
                    "frontier_call_rate": float(group["selected_frontier"].mean()),
                    "probe_call_rate": float(group["probe_used"].mean()),
                    "changed_rate": float(group["changed"].mean()),
                    "local_consensus_available_rate": float(group["local_consensus_available"].mean()),
                    "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def apply_rule(
    base: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    rule: dict[str, Any],
    lambda_cost: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scope = set(json.loads(str(rule.get("scope_json", "[]"))))
    mode = str(rule.get("mode", "base"))
    order = str(rule.get("order", "cheapest"))
    min_votes = 0 if pd.isna(rule.get("min_votes", np.nan)) else int(rule["min_votes"])
    for row in base.itertuples(index=False):
        query_id = str(row.query_id)
        base_model = str(row.fused_model)
        selected = base_model
        probe_used = False
        consensus_available = False
        if mode != "base" and str(row.benchmark) in scope and mode_applies(row, mode):
            if bool(rule.get("deployable", True)):
                probe_used = True
                candidate = local_majority_action(action_map.get(query_id, {}), min_votes=min_votes, order=order)
                consensus_available = bool(candidate)
                if candidate:
                    selected = candidate
            else:
                candidate = cheapest_local_same_answer(action_map.get(query_id, {}), base_model)
                consensus_available = bool(candidate)
                if candidate:
                    selected = candidate
        action = action_map.get(query_id, {}).get(selected, {})
        quality = float(action.get("quality_score", row.fused_quality))
        norm_cost = float(action.get("normalized_remote_cost", 0.0) or 0.0)
        rows.append(
            {
                **row._asdict(),
                "selected_model": selected,
                "selected_quality": quality,
                "selected_utility": quality - float(lambda_cost) * norm_cost,
                "selected_frontier": bool(action.get("is_frontier", False)),
                "changed": selected != base_model,
                "probe_used": bool(probe_used),
                "local_consensus_available": bool(consensus_available),
            }
        )
    return pd.DataFrame(rows)


def mode_applies(row: Any, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "if_base_frontier":
        return bool(row.fused_frontier)
    if mode == "if_base_not_frontier":
        return not bool(row.fused_frontier)
    return False


def local_majority_action(actions: dict[str, dict[str, Any]], *, min_votes: int, order: str) -> str:
    answers = []
    answer_by_model = {}
    for model_id in LOCAL_ACTIONS:
        item = actions.get(model_id)
        if not item:
            continue
        answer = normalize_answer(item.get("parsed_answer", ""))
        if not answer:
            continue
        answer_by_model[model_id] = answer
        answers.append(answer)
    if not answers:
        return ""
    counts = Counter(answers)
    answer, count = counts.most_common(1)[0]
    if count < int(min_votes):
        return ""
    model_order = LOCAL_ACTIONS if order == "cheapest" else list(reversed(LOCAL_ACTIONS))
    for model_id in model_order:
        if answer_by_model.get(model_id) == answer:
            return model_id
    return ""


def cheapest_local_same_answer(actions: dict[str, dict[str, Any]], base_model: str) -> str:
    base = actions.get(base_model)
    if not base:
        return ""
    base_answer = normalize_answer(base.get("parsed_answer", ""))
    if not base_answer:
        return ""
    for model_id in LOCAL_ACTIONS:
        item = actions.get(model_id)
        if item and normalize_answer(item.get("parsed_answer", "")) == base_answer:
            return model_id
    return ""


def selected_rows(policy_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    val = policy_table[policy_table["split"].astype(str).eq("val")].copy()
    test = policy_table[policy_table["split"].astype(str).eq("test")].copy()
    for split, frame in [("val", val), ("test", test)]:
        base = frame[frame["policy"].astype(str).eq("base_exp194")]
        if not base.empty:
            row = base.iloc[0].copy()
            row["selection_rule"] = f"base_reference_{split}"
            rows.append(row)

    deployable = val[val["deployable"].astype(bool) & val["changed_rate"].astype(float).gt(0.0)].copy()
    if not deployable.empty:
        best = deployable.sort_values(
            ["mean_utility", "probe_call_rate", "frontier_call_rate", "changed_rate"],
            ascending=[False, True, True, True],
        ).iloc[0].copy()
        best["selection_rule"] = "val_best_deployable_local_consensus"
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = "val_best_deployable_local_consensus_test"
            rows.append(test_row)

    diagnostic = val[(~val["deployable"].astype(bool)) & val["changed_rate"].astype(float).gt(0.0)].copy()
    if not diagnostic.empty:
        best_diag = diagnostic.sort_values(
            ["mean_utility", "frontier_call_rate", "changed_rate"],
            ascending=[False, True, True],
        ).iloc[0].copy()
        best_diag["selection_rule"] = "val_best_diagnostic_same_answer"
        rows.append(best_diag)
        match = test[test["policy"].astype(str).eq(str(best_diag["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = "val_best_diagnostic_same_answer_test"
            rows.append(test_row)

    for _, row in test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(8).iterrows():
        diagnostic_row = row.copy()
        diagnostic_row["selection_rule"] = "top_test_diagnostic"
        rows.append(diagnostic_row)
    return pd.DataFrame(rows).drop_duplicates(["selection_rule", "policy", "split"], keep="first")


def normalize_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    return text.removeprefix("answer:").strip().strip("$").strip()


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_memo(path: Path, args: argparse.Namespace, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "policy",
        "family",
        "deployable",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "probe_call_rate",
        "changed_rate",
    ]
    lines = [
        "# Local Consensus Cost-Suppression Audit",
        "",
        "This no-call experiment tests whether local agreement can suppress unnecessary remote calls after Experiment 194.",
        "It reports deployable local-majority rules separately from a diagnostic post-hoc same-answer upper bound.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/195_local_consensus_cost_suppression_audit.py",
        "```",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- Deployable local-majority consensus uses only local candidate answers before a frontier/API call.",
        "- Diagnostic same-answer suppression uses the selected frontier answer as the anchor, so it is not a deployable pre-call router.",
        "- If the diagnostic row beats deployable consensus, the missing signal is a cheap way to know that the local answer would match the remote answer before paying for the remote call.",
        "- No GPT, Gemini, Claude, local generation, or vLLM serving calls are made by this script.",
        "",
        "## Artifacts",
        "",
        f"- All policies: `{path.parent / 'table_local_consensus_cost_suppression_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_local_consensus_cost_suppression_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_local_consensus_cost_suppression_query_choices.csv'}`",
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
