from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCAL_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
CHEAP_LOCAL_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
]
LARGE_ACTIONS = [
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gemini-3.5-flash-strong-solve",
    "gpt-5.5",
]
DEFAULT_GATE_METHOD = "tool_aware_benchmark_composed_eps0.01_recall_then_quality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep concrete action mappings under the strong tool-aware local-vs-large target gate."
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
        "--target-table",
        type=Path,
        default=Path(
            "results/controlled/broad100_constrained_yesno_probe_qwen14b/"
            "table_constrained_yesno_targets.csv"
        ),
    )
    parser.add_argument(
        "--benchmark-composed-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_tool_aware_benchmark_composed_policy/"
            "table_tool_aware_benchmark_composed_choices.csv"
        ),
    )
    parser.add_argument("--gate-method", default=DEFAULT_GATE_METHOD)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_target_gate_concrete_bridge_sweep"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["answer_norm"] = outputs["parsed_answer"].map(normalize_answer)
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )

    target = build_gate_target(args, outputs)
    row_map = {
        (str(row.query_id), str(row.model_id)): row
        for row in outputs.itertuples(index=False)
    }
    output_by_query = {str(query_id): group.copy() for query_id, group in outputs.groupby("query_id", sort=False)}
    priors = fit_train_priors(outputs)
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][
        ["query_id", "utility", "quality_score"]
    ].rename(columns={"utility": "oracle_utility", "quality_score": "oracle_quality"})

    policies = enumerate_policies()
    all_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for policy in policies:
        for split in ["val", "test"]:
            summary, details = evaluate_policy(
                target[target["split"].astype(str).eq(split)].copy(),
                row_map,
                output_by_query,
                priors,
                oracle,
                policy,
                split,
                int(args.bootstrap_samples),
                int(args.seed),
            )
            all_rows.append(summary)
            if split == "test":
                detail_frames.append(details)

    table = pd.DataFrame(all_rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    selected = selected_rows(table, args)
    selected_policies = set(selected["policy"].astype(str))
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    selected_details = details[details["policy"].astype(str).isin(selected_policies)].copy()

    table.to_csv(args.output_dir / "table_target_gate_concrete_bridge_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_target_gate_concrete_bridge_selected.csv", index=False)
    selected_details.to_csv(args.output_dir / "table_target_gate_concrete_bridge_query_choices.csv", index=False)
    write_memo(args.output_dir / "TARGET_GATE_CONCRETE_BRIDGE_SWEEP_MEMO.md", args, selected, table)
    print(f"Wrote target-gate concrete bridge sweep to {args.output_dir}")


def build_gate_target(args: argparse.Namespace, outputs: pd.DataFrame) -> pd.DataFrame:
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "exp171_for_197")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "exp172_for_197")
    target = pd.read_csv(args.target_table)
    prepared = exp172.prepare_outputs(outputs)
    target = exp171.add_tool_availability(target, prepared)
    return exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        str(args.gate_method),
        exp171,
    )


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fit_train_priors(outputs: pd.DataFrame) -> dict[str, dict[str, str]]:
    train = outputs[outputs["split"].astype(str).eq("train")].copy()
    return {
        "local_utility": fit_prior(train, CHEAP_LOCAL_ACTIONS, "utility"),
        "local_quality": fit_prior(train, CHEAP_LOCAL_ACTIONS, "quality_score"),
        "large_utility": fit_prior(train, LARGE_ACTIONS, "utility"),
        "large_quality": fit_prior(train, LARGE_ACTIONS, "quality_score"),
    }


def fit_prior(train: pd.DataFrame, pool: list[str], metric: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for benchmark, group in train[train["model_id"].isin(pool)].groupby("benchmark"):
        means = group.groupby("model_id")[metric].mean().sort_values(ascending=False)
        if not means.empty:
            out[str(benchmark)] = str(means.index[0])
    return out


def enumerate_policies() -> list[dict[str, str]]:
    local_modes = [
        "prior_utility",
        "prior_quality",
        "local_consensus_cheapest",
        "local_consensus_strongest",
        "cheap_consensus_cheapest",
        "cheap_consensus_strongest",
        "deterministic_math_tool",
        "qwen3-4b-local",
        "qwen3-8b-local",
        "qwen3-14b-awq-local",
    ]
    large_modes = [
        "prior_utility",
        "prior_quality",
        "large_consensus_cheapest",
        "large_consensus_strongest",
        *LARGE_ACTIONS,
    ]
    return [
        {
            "policy": f"target_gate_local_{local_mode}_large_{large_mode}",
            "local_mode": local_mode,
            "large_mode": large_mode,
        }
        for local_mode in local_modes
        for large_mode in large_modes
    ]


def evaluate_policy(
    target: pd.DataFrame,
    row_map: dict[tuple[str, str], Any],
    output_by_query: dict[str, pd.DataFrame],
    priors: dict[str, dict[str, str]],
    oracle: pd.DataFrame,
    policy: dict[str, str],
    split: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in target.itertuples(index=False):
        query_id = str(row.query_id)
        benchmark = str(row.benchmark)
        use_large = bool(getattr(row, "benchmark_composed_need_large"))
        selected = (
            choose_large_action(query_id, benchmark, output_by_query, priors, policy["large_mode"])
            if use_large
            else choose_local_action(query_id, benchmark, output_by_query, priors, policy["local_mode"])
        )
        output = row_map.get((query_id, selected))
        if output is None:
            continue
        rows.append(
            {
                "query_id": query_id,
                "query_text": str(row.query_text),
                "benchmark": benchmark,
                "split": split,
                "policy": policy["policy"],
                "local_mode": policy["local_mode"],
                "large_mode": policy["large_mode"],
                "gate_need_large": use_large,
                "selected_model": selected,
                "selected_quality": float(output.quality_score),
                "selected_utility": float(output.utility),
                "selected_frontier": bool(output.is_frontier),
                "selected_normalized_cost": float(output.normalized_remote_cost),
                "selected_latency_s": float(output.latency_s),
            }
        )
    details = pd.DataFrame(rows).merge(oracle, on="query_id", how="left")
    values = details["selected_utility"].astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(values, bootstrap_samples, seed)
    oracle_u = float(details["oracle_utility"].mean())
    oracle_q = float(details["oracle_quality"].mean())
    mean_u = float(details["selected_utility"].mean())
    mean_q = float(details["selected_quality"].mean())
    return (
        {
            **policy,
            "split": split,
            "n_queries": int(len(details)),
            "mean_quality": mean_q,
            "mean_utility": mean_u,
            "mean_utility_ci_low": ci_low,
            "mean_utility_ci_high": ci_high,
            "cost_oracle_mean_utility": oracle_u,
            "quality_oracle_mean_quality": oracle_q,
            "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
            "utility_gap_to_oracle": oracle_u - mean_u,
            "quality_gap_to_oracle": oracle_q - mean_q,
            "frontier_call_rate": float(details["selected_frontier"].mean()),
            "large_gate_rate": float(details["gate_need_large"].mean()),
            "normalized_cost_mean": float(details["selected_normalized_cost"].mean()),
            "p95_latency_s": float(details["selected_latency_s"].quantile(0.95)),
            "selected_models_json": json.dumps(
                details["selected_model"].value_counts().sort_index().to_dict(),
                sort_keys=True,
            ),
        },
        details,
    )


def choose_local_action(
    query_id: str,
    benchmark: str,
    output_by_query: dict[str, pd.DataFrame],
    priors: dict[str, dict[str, str]],
    mode: str,
) -> str:
    if mode == "prior_utility":
        return priors["local_utility"].get(benchmark, "qwen3-4b-local")
    if mode == "prior_quality":
        return priors["local_quality"].get(benchmark, "qwen3-4b-local")
    if mode in {"local_consensus_cheapest", "local_consensus_strongest"}:
        return consensus_action(query_id, output_by_query, LOCAL_ACTIONS, mode.endswith("cheapest")) or priors[
            "local_utility"
        ].get(benchmark, "qwen3-4b-local")
    if mode in {"cheap_consensus_cheapest", "cheap_consensus_strongest"}:
        return consensus_action(query_id, output_by_query, CHEAP_LOCAL_ACTIONS, mode.endswith("cheapest")) or priors[
            "local_utility"
        ].get(benchmark, "qwen3-4b-local")
    return mode


def choose_large_action(
    query_id: str,
    benchmark: str,
    output_by_query: dict[str, pd.DataFrame],
    priors: dict[str, dict[str, str]],
    mode: str,
) -> str:
    if mode == "prior_utility":
        return priors["large_utility"].get(benchmark, "qwen3-32b-awq-local")
    if mode == "prior_quality":
        return priors["large_quality"].get(benchmark, "qwen3-32b-awq-local")
    if mode in {"large_consensus_cheapest", "large_consensus_strongest"}:
        return consensus_action(query_id, output_by_query, LARGE_ACTIONS, mode.endswith("cheapest")) or priors[
            "large_utility"
        ].get(benchmark, "qwen3-32b-awq-local")
    return mode


def consensus_action(
    query_id: str,
    output_by_query: dict[str, pd.DataFrame],
    pool: list[str],
    cheapest: bool,
) -> str:
    group = output_by_query.get(query_id)
    if group is None or group.empty:
        return ""
    answer_groups: dict[str, list[str]] = defaultdict(list)
    for row in group[group["model_id"].isin(pool)].itertuples(index=False):
        answer = str(row.answer_norm)
        if answer:
            answer_groups[answer].append(str(row.model_id))
    best: tuple[tuple[int, int, int], list[str]] | None = None
    for models in answer_groups.values():
        models = sorted(set(models), key=lambda model: pool.index(model))
        if len(models) < 2:
            continue
        score = (
            len(models),
            int("deterministic_math_tool" in models),
            int("qwen3-32b-awq-selfconsistency-n3-local" in models),
        )
        if best is None or score > best[0]:
            best = (score, models)
    if best is None:
        return ""
    return best[1][0] if cheapest else best[1][-1]


def selected_rows(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[pd.Series] = []
    val = table[table["split"].astype(str).eq("val")].copy()
    test = table[table["split"].astype(str).eq("test")].copy()
    if not val.empty:
        best = val.sort_values(["mean_utility", "mean_quality", "frontier_call_rate"], ascending=[False, False, True]).iloc[0].copy()
        best["selection_rule"] = "val_best_utility"
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            row = match.iloc[0].copy()
            row["selection_rule"] = "val_best_utility_test"
            rows.append(row)
    oracle_u = float(test["cost_oracle_mean_utility"].iloc[0]) if not test.empty else float("nan")
    oracle_q = float(test["quality_oracle_mean_quality"].iloc[0]) if not test.empty else float("nan")
    val_oracle_u = float(val["cost_oracle_mean_utility"].iloc[0]) if not val.empty else float("nan")
    val_oracle_q = float(val["quality_oracle_mean_quality"].iloc[0]) if not val.empty else float("nan")
    target_val = val[
        val["mean_utility"].astype(float).ge(0.95 * val_oracle_u)
        & val["mean_quality"].astype(float).ge(val_oracle_q - 0.03)
    ].copy()
    if not target_val.empty:
        best = target_val.sort_values(["frontier_call_rate", "mean_utility"], ascending=[True, False]).iloc[0].copy()
        best["selection_rule"] = "val_target_pass_low_frontier"
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            row = match.iloc[0].copy()
            row["selection_rule"] = "val_target_pass_low_frontier_test"
            rows.append(row)
    top = test.sort_values(["mean_utility", "mean_quality"], ascending=[False, False]).head(5)
    for _, row in top.iterrows():
        diagnostic = row.copy()
        diagnostic["selection_rule"] = "top_test_diagnostic"
        rows.append(diagnostic)
    selected = pd.DataFrame(rows)
    if not selected.empty:
        selected["test_target_utility_threshold"] = 0.95 * oracle_u
        selected["test_target_quality_threshold"] = oracle_q - 0.03
        selected["within_95pct_oracle_utility"] = selected["mean_utility"].astype(float).ge(0.95 * oracle_u)
        selected["within_3pt_oracle_quality"] = selected["mean_quality"].astype(float).ge(oracle_q - 0.03)
    return selected.drop_duplicates(["selection_rule", "policy", "split"], keep="first")


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


def write_memo(path: Path, args: argparse.Namespace, selected: pd.DataFrame, table: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "policy",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "large_gate_rate",
        "within_95pct_oracle_utility",
        "within_3pt_oracle_quality",
    ]
    lines = [
        "# Target-Gate Concrete Bridge Sweep",
        "",
        "This no-call experiment keeps the strong tool-aware local-vs-large gate fixed, then sweeps concrete local and large action mappings.",
        "It tests whether the local-vs-large target result survives when the router must choose an actual model/action.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/197_target_gate_concrete_bridge_sweep.py",
        "```",
        "",
        f"- Gate method: `{args.gate_method}`",
        "- No GPT, Gemini, Claude, local generation, or vLLM calls are made.",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- The local-vs-large gate does not transfer through these simple concrete action mappings.",
        "- The remaining bottleneck is concrete action identity after deciding that a larger action is needed.",
        "- This supports focusing next on a reliable action chooser or evidence-backed answer adjudicator, not another broad escalation gate.",
        "",
        "## Artifacts",
        "",
        f"- All policies: `{path.parent / 'table_target_gate_concrete_bridge_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_target_gate_concrete_bridge_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_target_gate_concrete_bridge_query_choices.csv'}`",
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
