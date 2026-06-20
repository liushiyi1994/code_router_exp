from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SIGNALS = (
    "signal_constrained_yesno_query_only_risk",
    "signal_constrained_yesno_local_evidence_risk",
    "signal_constrained_plus_cached_mean_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark-aware non-training policies over constrained YES/NO scores.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_benchmark_aware_yesno_policy"),
    )
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166")
    target = pd.read_csv(args.target_table)
    signals = [item.strip() for item in str(args.signals).split(",") if item.strip()]
    missing = [signal for signal in signals if signal not in target.columns]
    if missing:
        raise ValueError(f"Missing signal columns: {missing}")
    table, learned_rules = evaluate_benchmark_policies(rc166, target, signals=signals, lambda_cost=float(args.lambda_cost))
    selected = rc166.validation_selected_rows(table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table.to_csv(args.output_dir / "table_benchmark_aware_yesno_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_benchmark_aware_yesno_policy_selected.csv", index=False)
    learned_rules.to_csv(args.output_dir / "table_benchmark_aware_yesno_rules.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "BENCHMARK_AWARE_YESNO_POLICY_MEMO.md", args, table, selected, learned_rules)
    print(f"Wrote benchmark-aware YES/NO policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_benchmark_policies(
    rc166,
    target: pd.DataFrame,
    *,
    signals: list[str],
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(frame, split=split, lambda_cost=lambda_cost))

    train = target[target["split"].eq("train")].copy()
    val = target[target["split"].eq("val")].copy()
    for signal in signals:
        for family in ["benchmark_threshold_train", "benchmark_threshold_train_val_blend"]:
            thresholds = {}
            rule_rows = []
            for benchmark in sorted(target["benchmark"].dropna().astype(str).unique()):
                source = train[train["benchmark"].astype(str).eq(benchmark)]
                if family == "benchmark_threshold_train_val_blend":
                    source = pd.concat([source, val[val["benchmark"].astype(str).eq(benchmark)]], ignore_index=True)
                if source.empty or not source[signal].notna().any():
                    source = train
                threshold, direction, source_utility = best_threshold_for_source(rc166, source, signal, lambda_cost=lambda_cost)
                thresholds[benchmark] = (threshold, direction)
                rule_rows.append(
                    {
                        "method": f"{family}_{short_signal(signal)}",
                        "family": family,
                        "signal": signal,
                        "benchmark": benchmark,
                        "direction": direction,
                        "threshold": threshold,
                        "source_mean_utility": source_utility,
                        "source_n": int(len(source)),
                    }
                )
            rules.extend(rule_rows)
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].copy()
                choose_large = benchmark_threshold_decision(frame, signal, thresholds)
                row = rc166.evaluate_decision(
                    frame,
                    choose_large,
                    split=split,
                    method=f"{family}_{short_signal(signal)}",
                    family=family,
                    lambda_cost=lambda_cost,
                )
                row.update({"signal": signal})
                rows.append(row)

    for signal in signals:
        for cap in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]:
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].copy()
                choose_large = benchmark_cap_decision(frame, signal, cap)
                row = rc166.evaluate_decision(
                    frame,
                    choose_large,
                    split=split,
                    method=f"benchmark_cap_{short_signal(signal)}_{cap:g}",
                    family="benchmark_cap",
                    lambda_cost=lambda_cost,
                )
                row.update({"signal": signal, "cap": float(cap)})
                rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False]), pd.DataFrame(rules)


def best_threshold_for_source(rc166, source: pd.DataFrame, signal: str, *, lambda_cost: float) -> tuple[float, str, float]:
    best: tuple[float, str, float] | None = None
    values = pd.to_numeric(source[signal], errors="coerce")
    thresholds = np.unique(np.quantile(values.dropna().to_numpy(dtype=float), np.linspace(0.0, 1.0, 31))) if values.notna().any() else np.asarray([0.0])
    for direction in ["high", "low"]:
        for threshold in thresholds:
            choose_large = rc166.threshold_decision(values.to_numpy(dtype=float), float(threshold), direction)
            row = rc166.evaluate_decision(source, choose_large, split="source", method="source", family="source", lambda_cost=lambda_cost)
            score = (float(row["mean_utility"]), -float(row["normalized_cost_mean"]), -float(row["large_call_rate"]))
            if best is None or score > (best[2], 0.0, 0.0):
                best = (float(threshold), direction, float(row["mean_utility"]))
    assert best is not None
    return best


def benchmark_threshold_decision(frame: pd.DataFrame, signal: str, thresholds: dict[str, tuple[float, str]]) -> np.ndarray:
    choose = np.zeros(len(frame), dtype=bool)
    values = frame[signal].to_numpy(dtype=float)
    for index, benchmark in enumerate(frame["benchmark"].astype(str).to_numpy()):
        threshold, direction = thresholds.get(benchmark, (float("inf"), "high"))
        value = values[index]
        if not np.isfinite(value):
            choose[index] = False
        elif direction == "high":
            choose[index] = bool(value >= threshold)
        elif direction == "low":
            choose[index] = bool(value <= threshold)
        else:
            raise ValueError(direction)
    return choose


def benchmark_cap_decision(frame: pd.DataFrame, signal: str, cap: float) -> np.ndarray:
    choose = np.zeros(len(frame), dtype=bool)
    for _, group in frame.groupby("benchmark", sort=False):
        indices = group.index.to_numpy()
        scores = group[signal].to_numpy(dtype=float)
        order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
        k = max(1, int(np.floor(float(cap) * len(group))))
        choose[frame.index.get_indexer(indices[order[:k]])] = True
    return choose


def short_signal(signal: str) -> str:
    replacements = {
        "signal_constrained_yesno_query_only_risk": "q",
        "signal_constrained_yesno_local_evidence_risk": "e",
        "signal_constrained_plus_cached_mean_risk": "c",
    }
    return replacements.get(signal, signal.replace("signal_", "")[:8])


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#706752")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Benchmark-Aware YES/NO Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_aware_yesno_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, rules: pd.DataFrame) -> None:
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
        "# Benchmark-Aware YES/NO Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/169_benchmark_aware_yesno_policy.py",
        (
            "PYTHONPATH=src python experiments/169_benchmark_aware_yesno_policy.py "
            f"--target-table {args.target_table} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "- This is a threshold/cap policy over cached constrained YES/NO scores.",
        "- It makes no GPT, Gemini, Claude, or vLLM calls; train/val/test constrained scores must already be cached.",
        "- Benchmark-aware thresholds are fit on train or train+validation and selected/reporting follows validation to held-out test.",
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
        "## Learned Rule Preview",
        "",
        "```csv",
        compact_csv(rules, max_rows=80),
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
