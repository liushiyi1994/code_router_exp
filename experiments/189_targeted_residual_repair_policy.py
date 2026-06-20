from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_METHOD = "current_verifier_switch_conf0.85_pred_rf_thr-0.0288"
LOCAL_METHOD = "pred_rf_thr-0.0288"
LOCAL_MODELS = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
REPLACEMENTS = [
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gemini-3.5-flash-strong-solve",
    "gpt-5.5",
    "local_majority",
    "local_majority_cheapest",
]
FRONTIERS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve", "gpt-5.5"}
STRONG_OR_FRONTIER = {"qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local", *FRONTIERS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted residual repair rules over cached current-action verifier choices.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_action_verifier_qwen14b/"
            "table_current_action_verifier_features.csv"
        ),
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_action_verifier_qwen14b/"
            "table_current_action_verifier_probe.csv"
        ),
    )
    parser.add_argument(
        "--oracle-targets",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_action_verifier_qwen14b/"
            "table_current_action_oracle_targets.csv"
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
        default=Path("results/controlled/broad100_targeted_residual_repair_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--near-best-eps", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp187 = load_module("experiments/187_current_action_verifier_vllm.py", "current_verifier_187_for_189")
    exp183 = load_module("experiments/183_local_safe_gain_gate.py", "local_safe_183_for_189")
    features = pd.read_csv(args.features)
    probe = pd.read_csv(args.probe)
    targets = pd.read_csv(args.oracle_targets)
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    base = reconstruct_base_choices(features, probe, outputs, targets, exp187, exp183)
    action_map = {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs.groupby("query_id")}
    oracle = query_oracle(outputs)
    base = base.merge(oracle, on="query_id", how="left")
    base["regret"] = base["oracle_utility"] - base["utility"].astype(float)
    residual = residual_by_benchmark(base)
    rules = enumerate_rules(residual)
    all_rows, query_rows = evaluate_rules(base, action_map, rules, args)
    selected = selected_rows(all_rows, args)
    selected_policies = set(selected["policy"].astype(str).tolist()) if not selected.empty else set()
    query_rows_to_write = query_rows[query_rows["policy"].astype(str).isin(selected_policies)].copy()

    residual.to_csv(args.output_dir / "table_targeted_residual_repair_residuals.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output_dir / "table_targeted_residual_repair_rule_library.csv", index=False)
    all_rows.to_csv(args.output_dir / "table_targeted_residual_repair_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_targeted_residual_repair_policy_selected.csv", index=False)
    query_rows_to_write.to_csv(args.output_dir / "table_targeted_residual_repair_query_choices.csv", index=False)
    write_memo(args.output_dir / "TARGETED_RESIDUAL_REPAIR_POLICY_MEMO.md", args, residual, selected)
    print(f"Wrote targeted residual repair policies to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reconstruct_base_choices(features: pd.DataFrame, probe: pd.DataFrame, outputs: pd.DataFrame, targets: pd.DataFrame, exp187, exp183) -> pd.DataFrame:
    available = {str(query_id): set(group["model_id"].astype(str)) for query_id, group in outputs.groupby("query_id", sort=False)}
    probe_map = probe.set_index(["query_id", "method"]).to_dict("index")
    spec = {
        "method": BASE_METHOD,
        "family": "current_action_verifier_switch",
        "kind": "switch",
        "local_method": LOCAL_METHOD,
        "pred_col": "pred_rf",
        "local_threshold": -0.0288,
        "confidence": 0.85,
    }
    frames: list[pd.DataFrame] = []
    target_cols = targets[["query_id", "split", "need_large"]].drop_duplicates("query_id")
    for split in ["val", "test"]:
        split_features = features[features["split"].astype(str).eq(split)].copy()
        choices = exp187.choose_policy_actions(split_features, spec, exp183, probe_map, available)
        selected = choices[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
        selected = selected[selected["split"].astype(str).eq(split)].copy()
        selected = selected.merge(choices.drop(columns=["model_id"]), on="query_id", how="left")
        selected = selected.rename(columns={"model_id": "selected_model_id"})
        selected = selected.merge(target_cols, on=["query_id", "split"], how="left")
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def query_oracle(outputs: pd.DataFrame) -> pd.DataFrame:
    idx = outputs.groupby("query_id")["utility"].idxmax()
    return outputs.loc[idx, ["query_id", "utility", "quality_score"]].rename(
        columns={"utility": "oracle_utility", "quality_score": "oracle_quality"}
    )


def residual_by_benchmark(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, benchmark), group in base.groupby(["split", "benchmark"], dropna=False):
        rows.append(
            {
                "split": split,
                "benchmark": benchmark,
                "n_queries": len(group),
                "mean_utility": group["utility"].astype(float).mean(),
                "oracle_utility": group["oracle_utility"].astype(float).mean(),
                "mean_regret": group["regret"].astype(float).mean(),
                "total_regret": group["regret"].astype(float).sum(),
                "frontier_call_rate": group["is_frontier"].astype(bool).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "total_regret"], ascending=[True, False])


def enumerate_rules(residual: pd.DataFrame) -> list[dict[str, Any]]:
    val_residual = residual[residual["split"].eq("val")].copy()
    ordered = val_residual.sort_values("total_regret", ascending=False)["benchmark"].astype(str).tolist()
    scopes: list[tuple[str, ...]] = []
    for size in [1, 2, 3, 4]:
        if len(ordered) >= size:
            scopes.append(tuple(ordered[:size]))
    scopes.extend(
        [
            ("gpqa",),
            ("mmlupro",),
            ("bbh",),
            ("math500",),
            ("livemathbench",),
            ("gpqa", "mmlupro"),
            ("gpqa", "mmlupro", "bbh"),
            ("gpqa", "mmlupro", "bbh", "math500"),
        ]
    )
    scopes = sorted(set(scopes))
    conditions: list[tuple[str, float | None]] = [
        ("all", None),
        ("selected_qwen32", None),
        ("selected_strong_solve", None),
        ("selected_frontier", None),
        ("selected_large", None),
    ]
    for condition in ["verifier_escalate", "verifier_switch", "verifier_not_accept", "qwen32_and_not_accept"]:
        for threshold in [0.0, 0.5, 0.7, 0.85]:
            conditions.append((condition, threshold))
    residual_map = val_residual.set_index("benchmark")["total_regret"].to_dict()
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_current_verifier",
            "scope_json": "[]",
            "condition": "base",
            "threshold": np.nan,
            "replacement": "none",
            "validation_residual_coverage": 0.0,
        }
    ]
    for scope in scopes:
        coverage = float(sum(float(residual_map.get(benchmark, 0.0)) for benchmark in scope))
        for condition, threshold in conditions:
            for replacement in REPLACEMENTS:
                policy = f"scope{'+'.join(scope)}_{condition}_{replacement}_{threshold if threshold is not None else 'none'}"
                rules.append(
                    {
                        "policy": policy,
                        "scope_json": json.dumps(scope),
                        "condition": condition,
                        "threshold": np.nan if threshold is None else float(threshold),
                        "replacement": replacement,
                        "validation_residual_coverage": coverage,
                    }
                )
    return rules


def evaluate_rules(
    base: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    rules: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    query_frames: list[pd.DataFrame] = []
    for rule in rules:
        choices = apply_rule(base, action_map, rule)
        query_frames.append(choices.assign(policy=str(rule["policy"])))
        for split, group in choices.groupby("split", dropna=False):
            values = group["patched_utility"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            oracle_q = float(group["oracle_quality"].astype(float).mean())
            mean_u = float(np.mean(values))
            mean_q = float(group["patched_quality"].astype(float).mean())
            tp = int(((group["need_large"].astype(bool)) & (group["patched_model"].astype(str).map(is_large))).sum())
            fp = int(((~group["need_large"].astype(bool)) & (group["patched_model"].astype(str).map(is_large))).sum())
            fn = int(((group["need_large"].astype(bool)) & (~group["patched_model"].astype(str).map(is_large))).sum())
            summary_rows.append(
                {
                    **rule,
                    "split": split,
                    "n_queries": len(group),
                    "mean_quality": mean_q,
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "cost_oracle_mean_utility": oracle_u,
                    "quality_oracle_mean_quality": oracle_q,
                    "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
                    "utility_gap_to_oracle": oracle_u - mean_u,
                    "quality_gap_to_oracle": oracle_q - mean_q,
                    "frontier_call_rate": float(group["patched_frontier"].mean()),
                    "strong_or_frontier_call_rate": float(group["patched_model"].astype(str).map(is_large).mean()),
                    "changed_rate": float(group["changed"].mean()),
                    "need_large_precision": tp / (tp + fp) if (tp + fp) else 0.0,
                    "need_large_recall": tp / (tp + fn) if (tp + fn) else 0.0,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "selected_models_json": json.dumps(group["patched_model"].value_counts().sort_index().to_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(query_frames, ignore_index=True)


def apply_rule(base: pd.DataFrame, action_map: dict[str, dict[str, dict[str, Any]]], rule: dict[str, Any]) -> pd.DataFrame:
    if str(rule["policy"]) == "base_current_verifier":
        out = base.copy()
        out["patched_model"] = out["selected_model_id"].astype(str)
        out["patched_quality"] = out["quality_score"].astype(float)
        out["patched_utility"] = out["utility"].astype(float)
        out["patched_frontier"] = out["is_frontier"].astype(bool)
        out["changed"] = False
        return out
    scope = set(json.loads(str(rule["scope_json"])))
    condition = str(rule["condition"])
    threshold = None if pd.isna(rule["threshold"]) else float(rule["threshold"])
    replacement = str(rule["replacement"])
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        selected = str(row.selected_model_id)
        should_patch = str(row.benchmark) in scope and condition_holds(row, condition, threshold)
        patched = choose_replacement(row, action_map.get(str(row.query_id), {}), replacement) if should_patch else selected
        action = action_map.get(str(row.query_id), {}).get(patched, action_map.get(str(row.query_id), {}).get(selected, {}))
        rows.append(
            {
                **row._asdict(),
                "patched_model": patched,
                "patched_quality": float(action.get("quality_score", row.quality_score)),
                "patched_utility": float(action.get("utility", row.utility)),
                "patched_frontier": bool(action.get("is_frontier", row.is_frontier)),
                "changed": patched != selected,
            }
        )
    return pd.DataFrame(rows)


def condition_holds(row: Any, condition: str, threshold: float | None) -> bool:
    selected = str(row.selected_model_id)
    verdict = str(getattr(row, "verifier_verdict", ""))
    confidence = float(getattr(row, "verifier_confidence", 0.0) or 0.0)
    threshold = 0.0 if threshold is None else threshold
    if condition == "all":
        return True
    if condition == "selected_qwen32":
        return selected == "qwen3-32b-awq-local"
    if condition == "selected_strong_solve":
        return selected == "gemini-3.5-flash-strong-solve"
    if condition == "selected_frontier":
        return selected in FRONTIERS
    if condition == "selected_large":
        return selected in STRONG_OR_FRONTIER
    if condition == "verifier_escalate":
        return verdict == "escalate" and confidence >= threshold
    if condition == "verifier_switch":
        return verdict == "switch" and confidence >= threshold
    if condition == "verifier_not_accept":
        return verdict != "accept" and confidence >= threshold
    if condition == "qwen32_and_not_accept":
        return selected == "qwen3-32b-awq-local" and verdict != "accept" and confidence >= threshold
    return False


def choose_replacement(row: Any, actions: dict[str, dict[str, Any]], replacement: str) -> str:
    selected = str(row.selected_model_id)
    if replacement in actions:
        return replacement
    if replacement.startswith("local_majority"):
        answers = {model: str(actions.get(model, {}).get("parsed_answer", "")) for model in LOCAL_MODELS if model in actions}
        counts = pd.Series(list(answers.values())).value_counts()
        if counts.empty or int(counts.iloc[0]) < 2:
            return selected
        answer = str(counts.index[0])
        order = (
            ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local"]
            if replacement == "local_majority_cheapest"
            else ["qwen3-14b-awq-local", "qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local", "qwen3-8b-local", "qwen3-4b-local"]
        )
        for model in order:
            if answers.get(model) == answer:
                return model
    return selected


def selected_rows(all_rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    val = all_rows[all_rows["split"].eq("val")].copy()
    test = all_rows[all_rows["split"].eq("test")].copy()
    rows: list[pd.Series] = []
    base_val = val[val["policy"].eq("base_current_verifier")].iloc[0]
    best = val.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).iloc[0]
    selectors = {"val_best_utility": best}
    active = val[val["changed_rate"].astype(float).ge(0.01)].copy()
    near = active[active["mean_utility"].astype(float).ge(float(best["mean_utility"]) - float(args.near_best_eps))].copy()
    if not near.empty:
        selectors["val_near_best_residual_coverage"] = near.sort_values(
            ["validation_residual_coverage", "mean_utility", "frontier_call_rate"],
            ascending=[False, False, True],
        ).iloc[0]
    no_harm = active[active["mean_utility"].astype(float).ge(float(base_val["mean_utility"]) - 1e-12)].copy()
    if not no_harm.empty:
        selectors["val_no_harm_residual_coverage"] = no_harm.sort_values(
            ["validation_residual_coverage", "mean_utility", "frontier_call_rate"],
            ascending=[False, False, True],
        ).iloc[0]
    for rule, row in selectors.items():
        row = row.copy()
        row["selection_rule"] = rule
        rows.append(row)
        match = test[test["policy"].astype(str).eq(str(row["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = f"{rule}_test"
            rows.append(test_row)
    for _, row in test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(12).iterrows():
        row = row.copy()
        row["selection_rule"] = "top_test_diagnostic"
        rows.append(row)
    return pd.DataFrame(rows)


def is_large(model: str) -> bool:
    return str(model) in STRONG_OR_FRONTIER


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_memo(path: Path, args: argparse.Namespace, residual: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Targeted Residual Repair Policy",
        "",
        "This is a no-new-call residual-repair sweep over cached Experiment 187 choices.",
        "Rules are selected using validation only; test-picked rows are diagnostic.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/189_targeted_residual_repair_policy.py",
        "```",
        "",
        "## Validation Residuals",
        "",
        markdown_table(
            residual[residual["split"].eq("val")][
                ["benchmark", "n_queries", "mean_utility", "oracle_utility", "mean_regret", "total_regret", "frontier_call_rate"]
            ]
        ),
        "",
        "## Selected Rows",
        "",
        markdown_table(
            selected[
                [
                    "selection_rule",
                    "policy",
                    "split",
                    "n_queries",
                    "mean_quality",
                    "mean_utility",
                    "oracle_utility_ratio",
                    "frontier_call_rate",
                    "changed_rate",
                    "validation_residual_coverage",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "- A deployable improvement must beat Experiment 187's held-out utility `0.7678` without using test-picked rules.",
        "- Rules with no validation activation are diagnostic only, even when they improve held-out test.",
        "- This branch tests whether simple residual action repair can close the broad100 gap before spending on another verifier.",
        "",
        "## Artifacts",
        "",
        f"- Residuals: `{path.parent / 'table_targeted_residual_repair_residuals.csv'}`",
        f"- Rule library: `{path.parent / 'table_targeted_residual_repair_rule_library.csv'}`",
        f"- All policies: `{path.parent / 'table_targeted_residual_repair_policy_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_targeted_residual_repair_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_targeted_residual_repair_query_choices.csv'}`",
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
