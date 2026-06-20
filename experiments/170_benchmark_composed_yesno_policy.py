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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose benchmark-level non-training YES/NO policies selected on validation.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_benchmark_composed_yesno_policy"),
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
    table, choices = evaluate_compositions(rc166, target, lambda_cost=float(args.lambda_cost))
    selected = rc166.validation_selected_rows(table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table.to_csv(args.output_dir / "table_benchmark_composed_yesno_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_benchmark_composed_yesno_policy_selected.csv", index=False)
    choices.to_csv(args.output_dir / "table_benchmark_composed_yesno_choices.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "BENCHMARK_COMPOSED_YESNO_POLICY_MEMO.md", args, table, selected, choices)
    print(f"Wrote benchmark-composed YES/NO policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_compositions(rc166, target: pd.DataFrame, *, lambda_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    policy_fns = candidate_policy_functions()
    val = target[target["split"].eq("val")].copy()
    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(frame, split=split, lambda_cost=lambda_cost))

    for epsilon in [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]:
        for tie_break in ["utility", "quality", "quality_then_large", "large_then_quality", "recall_then_quality"]:
            method = f"benchmark_composed_eps{epsilon:g}_{tie_break}"
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
                    family="benchmark_composed",
                    lambda_cost=lambda_cost,
                )
                row.update({"epsilon": float(epsilon), "tie_break": tie_break})
                rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False]), pd.DataFrame(choices)


def candidate_policy_functions() -> dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    return {
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


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#76625f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Benchmark-Composed YES/NO Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_composed_yesno_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, choices: pd.DataFrame) -> None:
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
    lines = [
        "# Benchmark-Composed YES/NO Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/170_benchmark_composed_yesno_policy.py",
        (
            "PYTHONPATH=src python experiments/170_benchmark_composed_yesno_policy.py "
            f"--target-table {args.target_table} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "- This composes a small set of fixed non-training policies per benchmark.",
        "- Benchmark choices are selected on validation only, including near-best utility tolerances and deterministic tie-breaks.",
        "- No GPT, Gemini, Claude, or vLLM calls are made by this script.",
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
        "## Selected Benchmark Policies",
        "",
        "```csv",
        compact_csv(
            choices[choices["method"].isin(selected["method"].dropna().astype(str).unique())]
            .sort_values(["method", "benchmark"])
            .drop(columns=["all_candidate_scores_json"], errors="ignore"),
            max_rows=120,
        ),
        "```",
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
