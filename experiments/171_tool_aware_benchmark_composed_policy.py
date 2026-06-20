from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUERY_SIGNAL = "signal_constrained_yesno_query_only_risk"
EVIDENCE_SIGNAL = "signal_constrained_yesno_local_evidence_risk"
CACHED_SIGNAL = "signal_constrained_plus_cached_mean_risk"
TOOL_MODEL_ID = "deterministic_math_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tool-aware benchmark-composed non-training YES/NO policy.")
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
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_tool_aware_benchmark_composed_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166")
    target = pd.read_csv(args.target_table)
    target = add_tool_availability(target, pd.read_parquet(args.outputs))
    table, choices, tool_summary = evaluate_compositions(rc166, target, lambda_cost=float(args.lambda_cost))
    selected = selected_rows(table, rc166, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table.to_csv(args.output_dir / "table_tool_aware_benchmark_composed_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_tool_aware_benchmark_composed_policy_selected.csv", index=False)
    choices.to_csv(args.output_dir / "table_tool_aware_benchmark_composed_choices.csv", index=False)
    tool_summary.to_csv(args.output_dir / "table_tool_availability_summary.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(
        args.output_dir / "TOOL_AWARE_BENCHMARK_COMPOSED_POLICY_MEMO.md",
        args,
        table,
        selected,
        choices,
        tool_summary,
    )
    print(f"Wrote tool-aware benchmark-composed policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def add_tool_availability(target: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    tool_rows = outputs[outputs["model_id"].astype(str).eq(TOOL_MODEL_ID)].copy()
    if tool_rows.empty:
        out = target.copy()
        out["tool_available"] = False
        return out
    tool_rows["tool_available_route_signal"] = (
        tool_rows.get("tool_available", False).astype(bool)
        & tool_rows.get("parsed_answer", "").fillna("").astype(str).str.strip().ne("")
    )
    tool = tool_rows[["query_id", "tool_available_route_signal"]].drop_duplicates("query_id")
    out = target.merge(tool, on="query_id", how="left")
    out["tool_available"] = out["tool_available_route_signal"].fillna(False).astype(bool)
    return out.drop(columns=["tool_available_route_signal"], errors="ignore")


def evaluate_compositions(rc166, target: pd.DataFrame, *, lambda_cost: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    policy_fns = candidate_policy_functions()
    val = target[target["split"].eq("val")].copy()
    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(frame, split=split, lambda_cost=lambda_cost))

    for epsilon in [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]:
        for tie_break in ["utility", "quality", "quality_then_large", "large_then_quality", "recall_then_quality"]:
            method = f"tool_aware_benchmark_composed_eps{epsilon:g}_{tie_break}"
            selected_by_benchmark: dict[str, str] = {}
            for benchmark in sorted(val["benchmark"].dropna().astype(str).unique()):
                source = val[val["benchmark"].astype(str).eq(benchmark)].copy()
                scored = []
                for policy_name, policy_fn in policy_fns.items():
                    choose = policy_fn(source)
                    row = rc166.evaluate_decision(
                        source,
                        choose,
                        split="val_source",
                        method=policy_name,
                        family="candidate",
                        lambda_cost=lambda_cost,
                    )
                    scored.append(row)
                scored_frame = pd.DataFrame(scored)
                chosen = choose_candidate(scored_frame, epsilon=epsilon, tie_break=tie_break)
                selected_by_benchmark[benchmark] = chosen
                choice_row = scored_frame[scored_frame["method"].eq(chosen)].head(1).to_dict("records")[0]
                choices.append(
                    {
                        "method": method,
                        "benchmark": benchmark,
                        "chosen_policy": chosen,
                        "epsilon": float(epsilon),
                        "tie_break": tie_break,
                        "val_policy_utility": float(choice_row["mean_utility"]),
                        "val_policy_quality": float(choice_row["mean_quality"]),
                        "val_large_call_rate": float(choice_row["large_call_rate"]),
                        "val_frontier_call_rate": float(choice_row["frontier_call_rate"]),
                        "all_candidate_scores_json": json.dumps(
                            scored_frame[
                                [
                                    "method",
                                    "mean_utility",
                                    "mean_quality",
                                    "large_call_rate",
                                    "frontier_call_rate",
                                    "need_large_recall",
                                ]
                            ].to_dict("records"),
                            sort_keys=True,
                        ),
                    }
                )
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].copy()
                choose_large = compose_decision(frame, selected_by_benchmark, policy_fns)
                row = rc166.evaluate_decision(
                    frame,
                    choose_large,
                    split=split,
                    method=method,
                    family="tool_aware_benchmark_composed",
                    lambda_cost=lambda_cost,
                )
                row.update({"epsilon": float(epsilon), "tie_break": tie_break})
                rows.append(row)

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    tool_summary = summarize_tool_availability(target)
    return table, pd.DataFrame(choices), tool_summary


def candidate_policy_functions() -> dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    base: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "always_local": lambda frame: np.zeros(len(frame), dtype=bool),
        "always_large": lambda frame: np.ones(len(frame), dtype=bool),
        "and_q0.287_e0.179": lambda frame: (frame[QUERY_SIGNAL].to_numpy(dtype=float) >= 0.287)
        & (frame[EVIDENCE_SIGNAL].to_numpy(dtype=float) >= 0.179),
        "cap_q0.60": lambda frame: cap_decision(frame, QUERY_SIGNAL, 0.60),
        "cap_e0.75": lambda frame: cap_decision(frame, EVIDENCE_SIGNAL, 0.75),
        "cap_c0.25": lambda frame: cap_decision(frame, CACHED_SIGNAL, 0.25),
        "cap_c0.50": lambda frame: cap_decision(frame, CACHED_SIGNAL, 0.50),
        "cap_c0.75": lambda frame: cap_decision(frame, CACHED_SIGNAL, 0.75),
    }
    out = dict(base)
    for name, fn in base.items():
        out[f"tool_{name}"] = tool_suppressed(fn)
    return out


def tool_suppressed(policy_fn: Callable[[pd.DataFrame], np.ndarray]) -> Callable[[pd.DataFrame], np.ndarray]:
    def decide(frame: pd.DataFrame) -> np.ndarray:
        choose = policy_fn(frame).astype(bool)
        if "tool_available" not in frame:
            return choose
        return np.where(frame["tool_available"].to_numpy(dtype=bool), False, choose)

    return decide


def cap_decision(frame: pd.DataFrame, signal: str, cap: float) -> np.ndarray:
    choose = np.zeros(len(frame), dtype=bool)
    if frame.empty:
        return choose
    scores = frame[signal].to_numpy(dtype=float)
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
    k = max(1, int(np.floor(float(cap) * len(frame))))
    choose[order[:k]] = True
    return choose


def choose_candidate(scored: pd.DataFrame, *, epsilon: float, tie_break: str) -> str:
    best_utility = float(scored["mean_utility"].max())
    near = scored[scored["mean_utility"] >= best_utility - float(epsilon)].copy()
    if tie_break == "utility":
        sort_cols = ["mean_utility", "mean_quality", "large_call_rate"]
        ascending = [False, False, True]
    elif tie_break == "quality":
        sort_cols = ["mean_quality", "mean_utility", "large_call_rate"]
        ascending = [False, False, True]
    elif tie_break == "quality_then_large":
        sort_cols = ["mean_quality", "large_call_rate", "mean_utility"]
        ascending = [False, False, False]
    elif tie_break == "large_then_quality":
        sort_cols = ["large_call_rate", "mean_quality", "mean_utility"]
        ascending = [False, False, False]
    elif tie_break == "recall_then_quality":
        sort_cols = ["need_large_recall", "mean_quality", "mean_utility"]
        ascending = [False, False, False]
    else:
        raise ValueError(tie_break)
    chosen = near.sort_values(sort_cols, ascending=ascending).iloc[0]
    return str(chosen["method"])


def compose_decision(
    frame: pd.DataFrame,
    selected_by_benchmark: dict[str, str],
    policy_fns: dict[str, Callable[[pd.DataFrame], np.ndarray]],
) -> np.ndarray:
    choose = np.zeros(len(frame), dtype=bool)
    benchmarks = frame["benchmark"].astype(str).to_numpy()
    for benchmark, policy_name in selected_by_benchmark.items():
        positions = np.where(benchmarks == benchmark)[0]
        if positions.size == 0:
            continue
        sub = frame.iloc[positions].copy()
        choose[positions] = policy_fns[policy_name](sub)
    return choose


def summarize_tool_availability(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    out["tool_available"] = out.get("tool_available", False).astype(bool)
    return (
        out.groupby(["split", "benchmark", "tool_available"], as_index=False)
        .agg(
            n=("query_id", "size"),
            local_quality=("local_quality", "mean"),
            local_utility=("local_utility", "mean"),
            large_quality=("large_quality", "mean"),
            large_utility=("large_utility", "mean"),
            need_large_rate=("need_large", "mean"),
        )
        .sort_values(["split", "benchmark", "tool_available"])
    )


def selected_rows(table: pd.DataFrame, rc166, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    selected = rc166.validation_selected_rows(table, bootstrap_samples=bootstrap_samples, seed=seed)
    extra: list[pd.DataFrame] = []
    family = "tool_aware_benchmark_composed"
    group = table[table["family"].eq(family)].copy()
    val = group[group["split"].eq("val")].copy()
    if not val.empty:
        val["quality_gap_to_local_large_oracle"] = val["local_large_oracle_mean_quality"] - val["mean_quality"]
        target_candidates = val[
            (val["oracle_utility_ratio"] >= 0.95)
            & (val["quality_gap_to_local_large_oracle"] <= 0.03)
            & (val["frontier_call_rate"] <= 0.40)
        ].copy()
        if target_candidates.empty:
            target_candidates = val
        chosen = target_candidates.sort_values(
            [
                "mean_quality",
                "need_large_recall",
                "mean_utility",
                "frontier_call_rate",
                "large_call_rate",
            ],
            ascending=[False, False, False, True, True],
        ).head(1)
        method = str(chosen.iloc[0]["method"])
        extra.append(chosen.drop(columns=["quality_gap_to_local_large_oracle"]).assign(selection_rule="val_target_quality_recall"))
        test = table[table["split"].eq("test") & table["method"].eq(method)]
        if not test.empty:
            extra.append(test.assign(selection_rule="val_target_quality_recall_test"))
    if extra:
        extra_frame = pd.concat(extra, ignore_index=True)
        extra_frame = rc166.add_bootstrap_ci(extra_frame, bootstrap_samples=bootstrap_samples, seed=seed)
        extra_frame = extra_frame.drop(columns=["_utility_values"], errors="ignore")
        selected = pd.concat([selected, extra_frame], ignore_index=True)
    return selected


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4f6f64")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Tool-Aware Benchmark-Composed YES/NO Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tool_aware_benchmark_composed_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    choices: pd.DataFrame,
    tool_summary: pd.DataFrame,
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
        "recovered_gap_vs_local",
        "large_call_rate",
        "frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "selection_rule",
    ]
    selected_methods = set(selected["method"].dropna().astype(str).unique()) if not selected.empty else set()
    gate_lines = target_gate_lines(selected)
    lines = [
        "# Tool-Aware Benchmark-Composed YES/NO Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/171_tool_aware_benchmark_composed_policy.py",
        (
            "PYTHONPATH=src python experiments/171_tool_aware_benchmark_composed_policy.py "
            f"--target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "- This composes fixed non-training policies per benchmark, selected on validation only.",
        "- It adds one route-time signal: whether the deterministic exact-math tool produced a non-empty answer.",
        "- Tool-aware candidate policies force the local side when that deterministic tool is available.",
        "- The script makes no GPT, Gemini, Claude, or vLLM calls.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=56),
        "```",
        "",
        "## Target Gate Check",
        "",
        *gate_lines,
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(
            table[table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)[[column for column in cols if column in table.columns]],
            max_rows=36,
        ),
        "```",
        "",
        "## Selected Benchmark Policies",
        "",
        "```csv",
        compact_csv(
            choices[choices["method"].isin(selected_methods)]
            .sort_values(["method", "benchmark"])
            .drop(columns=["all_candidate_scores_json"], errors="ignore"),
            max_rows=140,
        ),
        "```",
        "",
        "## Tool Availability Summary",
        "",
        "```csv",
        compact_csv(tool_summary, max_rows=80),
        "```",
        "",
        "## Interpretation",
        "",
        "- The improvement comes from suppressing upward routing when a deterministic exact-math tool is available.",
        "- This is not a trained-router result and does not use held-out labels for assignment.",
        "- The result is still a local-vs-large diagnostic: choosing local means the target-table local side, not a fully deployed local model selector.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_gate_lines(selected: pd.DataFrame) -> list[str]:
    if selected.empty:
        return ["No selected rows available."]
    oracle_rows = selected[(selected["method"].eq("oracle_local_vs_large_gate")) & (selected["split"].eq("test"))]
    target_rows = selected[selected["selection_rule"].eq("val_target_quality_recall_test")]
    if oracle_rows.empty or target_rows.empty:
        return ["Target-aware or oracle rows are missing from the selected table."]
    oracle = oracle_rows.iloc[0]
    target = target_rows.iloc[0]
    utility_threshold = 0.95 * float(oracle["mean_utility"])
    quality_threshold = float(oracle["mean_quality"]) - 0.03
    utility_pass = float(target["mean_utility"]) >= utility_threshold
    quality_pass = float(target["mean_quality"]) >= quality_threshold
    frontier_pass = float(target["frontier_call_rate"]) <= 0.40
    return [
        f"- Held-out local-vs-large oracle utility: `{float(oracle['mean_utility']):.4f}`; 95% target: `{utility_threshold:.4f}`.",
        f"- Held-out local-vs-large oracle quality: `{float(oracle['mean_quality']):.4f}`; within-3-point target: `{quality_threshold:.4f}`.",
        f"- Target-aware selected utility: `{float(target['mean_utility']):.4f}`; pass: `{str(utility_pass)}`.",
        f"- Target-aware selected quality: `{float(target['mean_quality']):.4f}`; pass: `{str(quality_pass)}`.",
        f"- Target-aware selected frontier-call rate: `{float(target['frontier_call_rate']):.4f}`; pass <=0.40: `{str(frontier_pass)}`.",
        "- Caveat: this gate is for the local-vs-large diagnostic abstraction, not a full deployed multi-action router.",
    ]


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
