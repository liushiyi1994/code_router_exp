from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate non-training combo policies over constrained YES/NO probe scores.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_combo_policy"),
    )
    parser.add_argument("--query-signal", default="signal_constrained_yesno_query_only_risk")
    parser.add_argument("--evidence-signal", default="signal_constrained_yesno_local_evidence_risk")
    parser.add_argument("--cached-signal", default="signal_constrained_plus_cached_mean_risk")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166")
    target = pd.read_csv(args.target_table)
    table = evaluate_combo_policies(rc166, target, args)
    selected = rc166.validation_selected_rows(
        table,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    table.to_csv(args.output_dir / "table_constrained_yesno_combo_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_constrained_yesno_combo_policy_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "CONSTRAINED_YESNO_COMBO_POLICY_MEMO.md", args, table, selected)
    print(f"Wrote constrained YES/NO combo policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_combo_policies(rc166, target: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    query_signal = str(args.query_signal)
    evidence_signal = str(args.evidence_signal)
    cached_signal = str(args.cached_signal)
    for signal in [query_signal, evidence_signal, cached_signal]:
        if signal not in target.columns:
            raise ValueError(f"Missing signal column: {signal}")

    for split in ["val", "test"]:
        split_frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(split_frame, split=split, lambda_cost=float(args.lambda_cost)))

    val = target[target["split"].eq("val")].copy()
    q_thresholds = quantile_thresholds(val[query_signal])
    e_thresholds = quantile_thresholds(val[evidence_signal])

    for q_threshold in q_thresholds:
        for e_threshold in e_thresholds:
            for mode in ["and", "or"]:
                for split in ["val", "test"]:
                    frame = target[target["split"].eq(split)].copy()
                    query_high = frame[query_signal].to_numpy(dtype=float) >= float(q_threshold)
                    evidence_high = frame[evidence_signal].to_numpy(dtype=float) >= float(e_threshold)
                    choose_large = query_high & evidence_high if mode == "and" else query_high | evidence_high
                    row = rc166.evaluate_decision(
                        frame,
                        choose_large,
                        split=split,
                        method=f"{mode}_{short_signal(query_signal)}{q_threshold:.4g}_{short_signal(evidence_signal)}{e_threshold:.4g}",
                        family="two_signal_combo",
                        lambda_cost=float(args.lambda_cost),
                    )
                    row.update(
                        {
                            "combo_mode": mode,
                            "query_signal": query_signal,
                            "evidence_signal": evidence_signal,
                            "query_threshold": float(q_threshold),
                            "evidence_threshold": float(e_threshold),
                        }
                    )
                    rows.append(row)

    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        score_name = f"_weighted_score_{alpha:g}"
        target[score_name] = alpha * target[query_signal].astype(float) + (1.0 - alpha) * target[evidence_signal].astype(float)
        for threshold in quantile_thresholds(target[target["split"].eq("val")][score_name], n=41):
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].copy()
                choose_large = frame[score_name].to_numpy(dtype=float) >= float(threshold)
                row = rc166.evaluate_decision(
                    frame,
                    choose_large,
                    split=split,
                    method=f"weighted_alpha{alpha:g}_thr{threshold:.4g}",
                    family="weighted_combo",
                    lambda_cost=float(args.lambda_cost),
                )
                row.update({"alpha": float(alpha), "threshold": float(threshold)})
                rows.append(row)

    for signal in [query_signal, evidence_signal, cached_signal]:
        for cap in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]:
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].copy()
                scores = frame[signal].to_numpy(dtype=float)
                order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
                k = max(1, int(np.floor(float(cap) * len(frame))))
                choose_large = np.zeros(len(frame), dtype=bool)
                choose_large[order[:k]] = True
                row = rc166.evaluate_decision(
                    frame,
                    choose_large,
                    split=split,
                    method=f"cap_{short_signal(signal)}_{cap:g}",
                    family="cap_combo",
                    lambda_cost=float(args.lambda_cost),
                )
                row.update({"signal": signal, "cap": float(cap)})
                rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def quantile_thresholds(values: pd.Series, *, n: int = 21) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return np.asarray([0.0])
    return np.unique(np.quantile(array, np.linspace(0.0, 1.0, n)))


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
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#6b7658")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Constrained YES/NO Combo Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_constrained_yesno_combo_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
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
        "# Constrained YES/NO Combo Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/168_constrained_yesno_combo_policy.py",
        (
            "PYTHONPATH=src python experiments/168_constrained_yesno_combo_policy.py "
            f"--target-table {args.target_table} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "- This is a non-training policy search over cached constrained YES/NO probe scores.",
        "- It makes no GPT, Gemini, Claude, or local vLLM calls; the vLLM probe cache comes from experiment 167.",
        "- Thresholds and caps are selected on validation and reported on held-out test.",
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
        "- The key deployable row is the validation-selected held-out test row, not the top test diagnostic rows.",
        "- Beating always-large is useful progress, but the target remains the local-vs-large cost-aware oracle.",
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
