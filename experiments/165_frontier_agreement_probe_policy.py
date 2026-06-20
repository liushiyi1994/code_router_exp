from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_BASE_RULE = ("all", "base", 0.50, 3, "math_mmlupro", "self", 1.00, 1, None)
LOCAL_PRIORITY = [
    "qwen3-32b-awq-local",
    "qwen3-14b-awq-local",
    SELF_MODEL_ID,
    "qwen3-4b-local",
    "qwen3-8b-local",
]
AGREE_SETS = {
    "stronglocals": ("qwen3-14b-awq-local", "qwen3-32b-awq-local"),
    "alllocals": tuple(LOCAL_PRIORITY),
    "q32": ("qwen3-32b-awq-local",),
    "q14": ("qwen3-14b-awq-local",),
}
BENCHMARK_SETS = {
    "all": ("aime", "bbh", "gpqa", "gsm8k", "humaneval", "livemathbench", "math500", "mbpp", "mmlupro"),
    "stress": ("gpqa", "mmlupro", "math500", "livemathbench"),
    "gpqa_mmlu": ("gpqa", "mmlupro"),
    "math": ("math500", "livemathbench", "aime", "gsm8k"),
}
TRIGGERS = ("all", "low_vote", "not_unanimous", "pairwise_strong", "pairwise_self", "pairwise_base_or_self")
FALLBACKS = ("base", "probe", "strong")
PROBE_MODELS = ("gemini-3.5-flash", "gpt-5.5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frontier answer-agreement probe policy with probe cost accounted.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path("results/controlled/broad100_embedding_self_action_gate/embedding_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_frontier_agreement_probe_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--val-tie-eps", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_gate")
    calibrated = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_gate")
    pairwise = load_module("experiments/162_pairwise_action_ranker.py", "pairwise_action_ranker")
    residual = load_module("experiments/163_residual_confidence_rule_policy.py", "residual_confidence_rule")

    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    context = residual.build_context(
        package,
        self_gate,
        calibrated,
        pairwise,
        outputs,
        probe,
        embedding_cache_dir=args.embedding_cache_dir,
        self_model_id=SELF_MODEL_ID,
        max_features=int(args.max_features),
    )
    normalized_outputs = normalize_outputs(outputs)
    table = run_agreement_grid(
        residual,
        context,
        normalized_outputs,
        lambda_cost=float(args.lambda_cost),
    )
    selected = validation_selected_rows(table, eps=float(args.val_tie_eps))
    table.to_csv(args.output_dir / "table_frontier_agreement_probe_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_frontier_agreement_probe_policy_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "FRONTIER_AGREEMENT_PROBE_POLICY_MEMO.md", args, table, selected)
    print(f"Wrote frontier agreement-probe policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_outputs(outputs: pd.DataFrame) -> pd.DataFrame:
    out = outputs.copy()
    if "split" not in out.columns:
        for candidate in ["split_y", "split_x"]:
            if candidate in out.columns:
                out["split"] = out[candidate].astype(str)
                break
    if "rank_in_benchmark" not in out.columns:
        for candidate in ["rank_in_benchmark_y", "rank_in_benchmark_x"]:
            if candidate in out.columns:
                out["rank_in_benchmark"] = out[candidate]
                break
    return out.drop_duplicates(["query_id", "model_id"], keep="last")


def run_agreement_grid(
    residual,
    context: dict[str, Any],
    outputs: pd.DataFrame,
    *,
    lambda_cost: float,
) -> pd.DataFrame:
    by_query_model = outputs.set_index(["query_id", "model_id"])
    base_actions = {
        split: residual.apply_rule(context, split, context["base_actions"][split], DEFAULT_BASE_RULE)
        for split in ["val", "test"]
    }
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            evaluate_selected_models(
                context,
                by_query_model,
                split,
                baseline_models(context, split, base_actions[split]),
                np.zeros(len(base_actions[split]), dtype=bool),
                probe_model="",
                method="residual_rule_baseline",
                family="baseline",
                lambda_cost=lambda_cost,
            )
        )
    for probe_model in PROBE_MODELS:
        for benchmark_set_name, benchmark_set in BENCHMARK_SETS.items():
            for trigger in TRIGGERS:
                for agree_set_name, agree_set in AGREE_SETS.items():
                    for fallback in FALLBACKS:
                        method = (
                            f"{probe_model}_agreement_probe_{benchmark_set_name}_{trigger}"
                            f"_{agree_set_name}_{fallback}"
                        )
                        for split in ["val", "test"]:
                            selected, probe_called = apply_agreement_policy(
                                context,
                                by_query_model,
                                split,
                                base_actions[split],
                                probe_model=probe_model,
                                benchmark_set=benchmark_set,
                                trigger=trigger,
                                agree_set=set(agree_set),
                                fallback=fallback,
                            )
                            row = evaluate_selected_models(
                                context,
                                by_query_model,
                                split,
                                selected,
                                probe_called,
                                probe_model=probe_model,
                                method=method,
                                family=f"{probe_model}_agreement_probe",
                                lambda_cost=lambda_cost,
                            )
                            row.update(
                                {
                                    "probe_model": probe_model,
                                    "benchmark_set": benchmark_set_name,
                                    "trigger": trigger,
                                    "agree_set": agree_set_name,
                                    "fallback": fallback,
                                }
                            )
                            rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def baseline_models(context: dict[str, Any], split: str, actions: np.ndarray) -> np.ndarray:
    metrics = context["metrics"][split]
    idx = np.arange(len(actions))
    return metrics["selected_models"][idx, actions].astype(str)


def apply_agreement_policy(
    context: dict[str, Any],
    by_query_model: pd.DataFrame,
    split: str,
    actions: np.ndarray,
    *,
    probe_model: str,
    benchmark_set: tuple[str, ...],
    trigger: str,
    agree_set: set[str],
    fallback: str,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = context["arrays"][split]
    query_ids = arrays["query_id"].astype(str)
    selected = baseline_models(context, split, actions).copy()
    probe_called = np.zeros(len(query_ids), dtype=bool)
    for idx, query_id in enumerate(query_ids):
        if arrays["benchmark"][idx] not in benchmark_set:
            continue
        if not trigger_matches(trigger, actions[idx], arrays, idx):
            continue
        probe_called[idx] = True
        probe_answer = normalized_answer(by_query_model.loc[(query_id, probe_model), "parsed_answer"])
        agreed_model = ""
        if probe_answer:
            for local_model in LOCAL_PRIORITY:
                if local_model not in agree_set or (query_id, local_model) not in by_query_model.index:
                    continue
                local_answer = normalized_answer(by_query_model.loc[(query_id, local_model), "parsed_answer"])
                if local_answer == probe_answer:
                    agreed_model = local_model
                    break
        if agreed_model:
            selected[idx] = agreed_model
        elif fallback == "probe":
            selected[idx] = probe_model
        elif fallback == "strong":
            selected[idx] = STRONG_MODEL_ID
    return selected, probe_called


def trigger_matches(trigger: str, action: int, arrays: dict[str, np.ndarray], idx: int) -> bool:
    if trigger == "all":
        return True
    if trigger == "low_vote":
        return bool(arrays["vote_frac"][idx] <= 0.67)
    if trigger == "not_unanimous":
        return bool(arrays["vote_frac"][idx] < 1.0)
    if trigger == "pairwise_strong":
        return int(action) == 2
    if trigger == "pairwise_self":
        return int(action) == 1
    if trigger == "pairwise_base_or_self":
        return int(action) in {0, 1}
    raise ValueError(trigger)


def evaluate_selected_models(
    context: dict[str, Any],
    by_query_model: pd.DataFrame,
    split: str,
    selected_models: np.ndarray,
    probe_called: np.ndarray,
    *,
    probe_model: str,
    method: str,
    family: str,
    lambda_cost: float,
) -> dict[str, Any]:
    query_ids = context["arrays"][split]["query_id"].astype(str)
    qualities: list[float] = []
    utilities: list[float] = []
    norm_costs: list[float] = []
    usd_costs: list[float] = []
    latencies: list[float] = []
    selected_frontier: list[bool] = []
    remote_calls: list[float] = []
    for query_id, selected_model, called in zip(query_ids, selected_models, probe_called):
        row = by_query_model.loc[(query_id, selected_model)]
        quality = float(row["quality_score"])
        norm_cost = float(row["normalized_remote_cost"])
        usd_cost = float(row["cost_total_usd"])
        latency = float(row["latency_s"])
        call_count = 1.0 if bool(row["is_frontier"]) else 0.0
        if called and probe_model and selected_model != probe_model:
            probe_row = by_query_model.loc[(query_id, probe_model)]
            norm_cost += float(probe_row["normalized_remote_cost"])
            usd_cost += float(probe_row["cost_total_usd"])
            latency += float(probe_row["latency_s"])
            call_count += 1.0
        qualities.append(quality)
        utilities.append(quality - float(lambda_cost) * norm_cost)
        norm_costs.append(norm_cost)
        usd_costs.append(usd_cost)
        latencies.append(latency)
        selected_frontier.append(bool(row["is_frontier"]))
        remote_calls.append(call_count)
    model_counts = pd.Series(selected_models).value_counts().sort_index().to_dict()
    oracle_stats = context["oracle_stats"][split]
    return {
        "method": method,
        "family": family,
        "split": split,
        "n_queries": int(len(query_ids)),
        "mean_quality": float(np.mean(qualities)),
        "mean_utility": float(np.mean(utilities)),
        "quality_oracle_mean_quality": float(np.mean(oracle_stats["quality"])),
        "cost_oracle_mean_utility": float(np.mean(oracle_stats["utility"])),
        "quality_gap_to_oracle": float(np.mean(oracle_stats["quality"]) - np.mean(qualities)),
        "utility_gap_to_oracle": float(np.mean(oracle_stats["utility"]) - np.mean(utilities)),
        "oracle_utility_ratio": float(np.mean(utilities) / np.mean(oracle_stats["utility"])),
        "normalized_remote_cost_mean": float(np.mean(norm_costs)),
        "remote_cost_total_usd": float(np.sum(usd_costs)),
        "selected_frontier_rate": float(np.mean(selected_frontier)),
        "frontier_call_rate": float(np.mean(remote_calls)),
        "probe_rate": float(np.mean(probe_called)),
        "strong_call_rate": float(np.mean(selected_models == STRONG_MODEL_ID)),
        "self_action_rate": float(np.mean(selected_models == SELF_MODEL_ID)),
        "mean_latency_s": float(np.mean(latencies)),
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
        "selected_models_json": json.dumps({str(key): int(value) for key, value in model_counts.items()}, sort_keys=True),
    }


def validation_selected_rows(table: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.head(1).copy()
        best_method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="strict_val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(best_method)]
        if not test.empty:
            rows.append(test.head(1).assign(selection_rule="strict_val_best_utility_test"))
        threshold = float(best.iloc[0]["mean_utility"]) - float(eps)
        near = val[val["mean_utility"] >= threshold].sort_values(
            ["normalized_remote_cost_mean", "probe_rate", "mean_utility", "mean_quality"],
            ascending=[True, True, False, False],
        )
        if not near.empty:
            cost_best = near.head(1).copy()
            cost_method = str(cost_best.iloc[0]["method"])
            rows.append(cost_best.assign(selection_rule=f"val_near_best_cost_tiebreak_eps{eps:g}"))
            cost_test = group[group["split"].eq("test") & group["method"].eq(cost_method)]
            if not cost_test.empty:
                rows.append(cost_test.head(1).assign(selection_rule=f"val_near_best_cost_tiebreak_eps{eps:g}_test"))
    diagnostic = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    if not diagnostic.empty:
        rows.append(diagnostic.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def normalized_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#586f86")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Frontier Agreement Probe Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_frontier_agreement_probe_policy.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "normalized_remote_cost_mean",
        "frontier_call_rate",
        "probe_rate",
        "strong_call_rate",
        "self_action_rate",
        "probe_model",
        "benchmark_set",
        "trigger",
        "agree_set",
        "fallback",
        "selection_rule",
    ]
    lines = [
        "# Frontier Agreement Probe Policy",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Self-consistency probe table: `{args.probe_table}`.",
        "",
        "This cached-only evaluator tests whether a sparse Gemini/GPT agreement check can make the base/self/strong action router more observable. It counts the frontier probe cost whenever the probe is called and is not reused as the final selected model.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=48),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(
            table[table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)[[column for column in cols if column in table.columns]],
            max_rows=32,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- Frontier/local answer agreement is a real correctness signal, but it must pay for the frontier probe that reveals the agreement.",
        "- If validation-selected held-out utility is below the residual-rule baseline, this is a negative result for frontier agreement probing under the current cost objective.",
        "- Top held-out diagnostic rows are not deployable claims because they are selected with test labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
