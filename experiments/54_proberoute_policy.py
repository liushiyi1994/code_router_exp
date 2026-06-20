from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not all([before_beliefs_path, after_beliefs_path, state_model_utility_path, query_model_utility_path]):
        table = blocked_policy_table(
            "blocked_missing_policy_inputs",
            "Missing aligned before/after beliefs, state-model utility, or query-model utility.",
        )
    else:
        before_beliefs = _read_matrix(before_beliefs_path)
        after_beliefs = _read_matrix(after_beliefs_path)
        state_model_utility = _read_matrix(state_model_utility_path)
        query_model_utility = _read_matrix(query_model_utility_path)
        baseline_mean = _selected_utility_mean(
            query_model_utility,
            select_models_from_belief(before_beliefs, state_model_utility),
        )
        oracle_reference = float(query_model_utility.max(axis=1).mean())
        table = evaluate_proberoute_policies(
            policies=default_policy_set(),
            before_beliefs=before_beliefs,
            after_beliefs=after_beliefs,
            state_model_utility=state_model_utility,
            query_model_utility=query_model_utility,
            probe_cost=_read_optional_series(probe_cost_path),
            predicted_gain=_read_optional_series(predicted_gain_path),
            baseline_mean_utility=baseline_mean,
            oracle_reference_mean_utility=oracle_reference,
        )

    table_path = out_dir / "table_proberoute_policy.csv"
    figure_path = out_dir / "fig_gap_closed_vs_probe_cost.pdf"
    table.to_csv(table_path, index=False)
    write_policy_figure(table, figure_path)
    write_memo(
        out_dir,
        table,
        before_beliefs_path=before_beliefs_path,
        after_beliefs_path=after_beliefs_path,
        state_model_utility_path=state_model_utility_path,
        query_model_utility_path=query_model_utility_path,
        probe_cost_path=probe_cost_path,
        predicted_gain_path=predicted_gain_path,
    )
    append_readme(out_dir, table)
    print(f"Wrote Phase 2 ProbeRoute++ policy table to {table_path}")
    print(f"Wrote Phase 2 ProbeRoute++ policy figure to {figure_path}")
    return table


def write_policy_figure(table: pd.DataFrame, figure_path: Path) -> None:
    executed = table[table["status"].eq("executed")].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if executed.empty:
        ax.text(
            0.5,
            0.5,
            "ProbeRoute++ policy evaluation blocked\n(no aligned beliefs/utilities)",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
    else:
        ax.scatter(
            executed["mean_probe_cost_proxy"].astype(float),
            executed["observability_gap_closed"].astype(float),
            s=60,
            color="#59A14F",
        )
        for _, row in executed.iterrows():
            ax.annotate(
                str(row["policy"]),
                (float(row["mean_probe_cost_proxy"]), float(row["observability_gap_closed"])),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean probe cost proxy")
        ax.set_ylabel("Observability gap closed")
        ax.set_title("ProbeRoute++ Gap Closed vs Probe Cost")
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_memo(out_dir: Path, table: pd.DataFrame, **paths: str | None) -> None:
    lines = [
        "# Phase 2 ProbeRoute++ Policy",
        "",
        "Command:",
        "",
        "```bash",
        _command(out_dir, **paths),
        "```",
        "",
        _status_sentence(table),
        "",
        "Outputs:",
        "",
        "- `table_proberoute_policy.csv`",
        "- `fig_gap_closed_vs_probe_cost.pdf`",
        "- `m5_proberoute_policy_memo.md`",
        "",
        "Summary:",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "m5_proberoute_policy_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 ProbeRoute++ Policy"
    lines = [
        marker,
        "",
        _status_sentence(table),
        "",
        "Outputs:",
        "",
        "- `table_proberoute_policy.csv`",
        "- `fig_gap_closed_vs_probe_cost.pdf`",
        "- `m5_proberoute_policy_memo.md`",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


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


def _status_sentence(table: pd.DataFrame) -> str:
    statuses = set(table["status"].astype(str))
    if statuses == {"executed"}:
        return "M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting."
    return (
        "M5 currently cannot support ProbeRoute++ policy claims because aligned before/after beliefs "
        "and state/query utility tables are not available."
    )


def _command(out_dir: Path, **paths: str | None) -> str:
    parts = ["python experiments/54_proberoute_policy.py", f"--output-dir {out_dir}"]
    flag_for_key = {
        "before_beliefs_path": "--before-beliefs",
        "after_beliefs_path": "--after-beliefs",
        "state_model_utility_path": "--state-model-utility",
        "query_model_utility_path": "--query-model-utility",
        "probe_cost_path": "--probe-cost",
        "predicted_gain_path": "--predicted-gain",
    }
    for key, flag in flag_for_key.items():
        if paths.get(key):
            parts.append(f"{flag} {paths[key]}")
    return " ".join(parts)


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
