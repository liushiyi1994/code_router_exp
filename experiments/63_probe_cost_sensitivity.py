from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from routecode.probes.policies import (
    blocked_policy_table,
    default_policy_set,
    evaluate_proberoute_policies,
    select_models_from_belief,
)
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-beliefs", default="")
    parser.add_argument("--after-beliefs", default="")
    parser.add_argument("--state-model-utility", default="")
    parser.add_argument("--query-model-utility", default="")
    parser.add_argument("--probe-cost", default="")
    parser.add_argument("--predicted-gain", default="")
    parser.add_argument("--probe-cost-multipliers", default="0,0.5,1,2,5,10,50,100")
    parser.add_argument("--output-dir", default="results/phase2")
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        before_beliefs_path=args.before_beliefs or None,
        after_beliefs_path=args.after_beliefs or None,
        state_model_utility_path=args.state_model_utility or None,
        query_model_utility_path=args.query_model_utility or None,
        probe_cost_path=args.probe_cost or None,
        predicted_gain_path=args.predicted_gain or None,
        probe_cost_multipliers=_parse_float_list(args.probe_cost_multipliers) or [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    )


def run(
    *,
    output_dir: str,
    before_beliefs_path: str | None = None,
    after_beliefs_path: str | None = None,
    state_model_utility_path: str | None = None,
    query_model_utility_path: str | None = None,
    probe_cost_path: str | None = None,
    predicted_gain_path: str | None = None,
    probe_cost_multipliers: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    multipliers = probe_cost_multipliers or [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    if not all([before_beliefs_path, after_beliefs_path, state_model_utility_path, query_model_utility_path]):
        table = blocked_policy_table(
            "blocked_missing_policy_inputs",
            "Missing aligned before/after beliefs, state-model utility, or query-model utility.",
        )
        table.insert(0, "probe_cost_multiplier", pd.NA)
    else:
        table = run_sensitivity(
            before_beliefs_path=before_beliefs_path,
            after_beliefs_path=after_beliefs_path,
            state_model_utility_path=state_model_utility_path,
            query_model_utility_path=query_model_utility_path,
            probe_cost_path=probe_cost_path,
            predicted_gain_path=predicted_gain_path,
            probe_cost_multipliers=multipliers,
        )
    summary = summarize_cost_sensitivity(table)
    command = _command(
        out_dir,
        before_beliefs_path=before_beliefs_path,
        after_beliefs_path=after_beliefs_path,
        state_model_utility_path=state_model_utility_path,
        query_model_utility_path=query_model_utility_path,
        probe_cost_path=probe_cost_path,
        predicted_gain_path=predicted_gain_path,
        probe_cost_multipliers=multipliers,
    )
    write_outputs(out_dir, table, summary, command)
    print(f"Wrote probe-cost sensitivity table to {out_dir / 'table_probe_cost_sensitivity.csv'}")
    print(f"Wrote probe-cost sensitivity summary to {out_dir / 'table_probe_cost_sensitivity_summary.csv'}")
    print(f"Wrote probe-cost sensitivity figure to {out_dir / 'fig_probe_cost_sensitivity.pdf'}")
    return table, summary


def run_sensitivity(
    *,
    before_beliefs_path: str,
    after_beliefs_path: str,
    state_model_utility_path: str,
    query_model_utility_path: str,
    probe_cost_path: str | None,
    predicted_gain_path: str | None,
    probe_cost_multipliers: list[float],
) -> pd.DataFrame:
    before_beliefs = _read_matrix(before_beliefs_path)
    after_beliefs = _read_matrix(after_beliefs_path)
    state_model_utility = _read_matrix(state_model_utility_path)
    query_model_utility = _read_matrix(query_model_utility_path)
    base_probe_cost = _read_optional_series(probe_cost_path)
    predicted_gain = _read_optional_series(predicted_gain_path)
    baseline_mean = _selected_utility_mean(
        query_model_utility,
        select_models_from_belief(before_beliefs, state_model_utility),
    )
    oracle_reference = float(query_model_utility.max(axis=1).mean())

    tables: list[pd.DataFrame] = []
    for multiplier in probe_cost_multipliers:
        scaled_cost = None
        if base_probe_cost is not None:
            scaled_cost = base_probe_cost.astype(float) * float(multiplier)
        table = evaluate_proberoute_policies(
            policies=default_policy_set(),
            before_beliefs=before_beliefs,
            after_beliefs=after_beliefs,
            state_model_utility=state_model_utility,
            query_model_utility=query_model_utility,
            probe_cost=scaled_cost,
            predicted_gain=predicted_gain,
            baseline_mean_utility=baseline_mean,
            oracle_reference_mean_utility=oracle_reference,
        )
        table.insert(0, "probe_cost_multiplier", float(multiplier))
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def summarize_cost_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "probe_cost_multiplier" not in table.columns or "mean_net_utility" not in table.columns:
        return pd.DataFrame()
    executed = table[table["status"].eq("executed")].copy()
    if executed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for multiplier, group in executed.groupby("probe_cost_multiplier", sort=True):
        by_policy = group.set_index("policy")
        best = group.sort_values("mean_net_utility", ascending=False).iloc[0]
        row: dict[str, Any] = {
            "probe_cost_multiplier": float(multiplier),
            "n_queries": int(best["n_queries"]),
            "best_policy_by_mean_net_utility": str(best["policy"]),
            "best_mean_net_utility": float(best["mean_net_utility"]),
        }
        for policy in [
            "never_probe",
            "always_probe",
            "entropy_threshold",
            "margin_threshold",
            "voi_probe",
            "oracle_probe",
        ]:
            if policy in by_policy.index:
                row[f"{policy}_mean_net_utility"] = float(by_policy.loc[policy, "mean_net_utility"])
                row[f"{policy}_fraction_probed"] = float(by_policy.loc[policy, "fraction_probed"])
                row[f"{policy}_gap_closed"] = float(by_policy.loc[policy, "observability_gap_closed"])
        if {"voi_probe", "never_probe"}.issubset(by_policy.index):
            row["voi_minus_never_mean_net_utility"] = float(
                by_policy.loc["voi_probe", "mean_net_utility"] - by_policy.loc["never_probe", "mean_net_utility"]
            )
        threshold_values = [
            float(by_policy.loc[policy, "mean_net_utility"])
            for policy in ["entropy_threshold", "margin_threshold"]
            if policy in by_policy.index
        ]
        if "voi_probe" in by_policy.index and threshold_values:
            row["voi_minus_best_threshold_mean_net_utility"] = float(
                by_policy.loc["voi_probe", "mean_net_utility"] - max(threshold_values)
            )
        rows.append(row)
    return _round_numeric(pd.DataFrame(rows))


def write_outputs(out_dir: Path, table: pd.DataFrame, summary: pd.DataFrame, command: str) -> None:
    table.to_csv(out_dir / "table_probe_cost_sensitivity.csv", index=False)
    summary.to_csv(out_dir / "table_probe_cost_sensitivity_summary.csv", index=False)
    write_figure(table, out_dir / "fig_probe_cost_sensitivity.pdf")
    write_memo(out_dir, table, summary, command)
    append_readme(out_dir, summary, command)


def write_figure(table: pd.DataFrame, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    executed = table[table["status"].eq("executed")].copy()
    if executed.empty:
        ax.text(0.5, 0.5, "Probe-cost sensitivity blocked", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
    else:
        for policy, group in executed.groupby("policy", sort=True):
            group = group.sort_values("probe_cost_multiplier")
            ax.plot(
                group["probe_cost_multiplier"].astype(float),
                group["mean_net_utility"].astype(float),
                marker="o",
                label=str(policy),
            )
        ax.set_xlabel("Probe cost multiplier")
        ax.set_ylabel("Mean net utility")
        ax.set_title("ProbeRoute++ Probe-Cost Sensitivity")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_memo(out_dir: Path, table: pd.DataFrame, summary: pd.DataFrame, command: str) -> None:
    lines = [
        "# Phase 2 Probe Cost Sensitivity",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "This sweeps probe-cost multipliers for the same state-mediated ProbeRoute++ policy inputs.",
        "",
        "Outputs:",
        "",
        "- `table_probe_cost_sensitivity.csv`",
        "- `table_probe_cost_sensitivity_summary.csv`",
        "- `fig_probe_cost_sensitivity.pdf`",
        "- `m7_probe_cost_sensitivity_memo.md`",
        "",
        "Summary:",
        "",
        _markdown_table(summary),
        "",
        "Policy Rows:",
        "",
        _markdown_table(table),
        "",
        "Interpretation:",
        "",
        interpret_summary(summary),
        "",
    ]
    (out_dir / "m7_probe_cost_sensitivity_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, summary: pd.DataFrame, command: str) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Probe Cost Sensitivity"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "Outputs:",
        "",
        "- `table_probe_cost_sensitivity.csv`",
        "- `table_probe_cost_sensitivity_summary.csv`",
        "- `fig_probe_cost_sensitivity.pdf`",
        "- `m7_probe_cost_sensitivity_memo.md`",
        "",
        "Summary:",
        "",
        _markdown_table(summary),
        "",
        "Interpretation:",
        "",
        interpret_summary(summary),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def interpret_summary(summary: pd.DataFrame) -> str:
    if summary.empty or "voi_minus_best_threshold_mean_net_utility" not in summary.columns:
        return "No executable probe-cost sensitivity rows were available."
    values = pd.to_numeric(summary["voi_minus_best_threshold_mean_net_utility"], errors="coerce").dropna()
    if values.empty:
        return "No numeric VOI-vs-threshold probe-cost sensitivity deltas were available."
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    tied = int((values == 0).sum())
    return (
        "Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta "
        f"`{values.mean():.4f}` over `{len(values)}` settings "
        f"(`{positive}` positive, `{negative}` negative, `{tied}` tied)."
    )


def _read_matrix(path: str) -> pd.DataFrame:
    frame = _read_table(path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    if "state_label" in frame.columns:
        return frame.set_index("state_label")
    return frame


def _read_optional_series(path: str | None) -> pd.Series | None:
    if not path:
        return None
    frame = _read_table(path)
    if "query_id" in frame.columns:
        value_columns = [column for column in frame.columns if column != "query_id"]
        if len(value_columns) != 1:
            raise ValueError(f"Expected one value column in {path}")
        return frame.set_index("query_id")[value_columns[0]]
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one value column in {path}")
    return frame.iloc[:, 0]


def _read_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def _selected_utility_mean(query_model_utility: pd.DataFrame, selected_models: pd.Series) -> float:
    return float(
        pd.Series(
            [query_model_utility.loc[query_id, model_id] for query_id, model_id in selected_models.items()],
            index=selected_models.index,
        ).mean()
    )


def _command(
    out_dir: Path,
    *,
    before_beliefs_path: str | None,
    after_beliefs_path: str | None,
    state_model_utility_path: str | None,
    query_model_utility_path: str | None,
    probe_cost_path: str | None,
    predicted_gain_path: str | None,
    probe_cost_multipliers: list[float],
) -> str:
    parts = ["python experiments/63_probe_cost_sensitivity.py", f"--output-dir {out_dir}"]
    path_flags = [
        ("--before-beliefs", before_beliefs_path),
        ("--after-beliefs", after_beliefs_path),
        ("--state-model-utility", state_model_utility_path),
        ("--query-model-utility", query_model_utility_path),
        ("--probe-cost", probe_cost_path),
        ("--predicted-gain", predicted_gain_path),
    ]
    for flag, path in path_flags:
        if path:
            parts.append(f"{flag} {path}")
    parts.append("--probe-cost-multipliers " + ",".join(str(value) for value in probe_cost_multipliers))
    return " ".join(parts)


def _parse_float_list(raw: str) -> list[float] | None:
    values = [float(value) for value in raw.split(",") if value]
    return values or None


def _round_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].round(6)
    return rounded


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
