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

from routecode.probes.signal_analysis import analyze_probe_signal
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-features", default="results/phase2/probe_features.parquet")
    parser.add_argument("--state-targets", default="")
    parser.add_argument("--query-features", default="")
    parser.add_argument("--output-dir", default="results/phase2")
    args = parser.parse_args()
    run(
        probe_features_path=args.probe_features,
        output_dir=args.output_dir,
        state_targets_path=args.state_targets or None,
        query_features_path=args.query_features or None,
    )


def run(
    *,
    probe_features_path: str,
    output_dir: str,
    state_targets_path: str | None = None,
    query_features_path: str | None = None,
) -> pd.DataFrame:
    probe_features = pd.read_parquet(probe_features_path)
    state_targets = _read_optional_table(state_targets_path)
    query_features = _read_optional_table(query_features_path)
    table = analyze_probe_signal(
        probe_features=probe_features,
        state_targets=state_targets,
        query_features=query_features,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "table_probe_signal_analysis.csv"
    figure_path = out_dir / "fig_probe_signal_gain.pdf"
    table.to_csv(table_path, index=False)
    write_probe_signal_figure(table, figure_path)
    write_memo(out_dir, probe_features_path, state_targets_path, query_features_path, table)
    append_readme(out_dir, probe_features_path, state_targets_path, query_features_path, table)
    print(f"Wrote Phase 2 probe signal analysis to {table_path}")
    print(f"Wrote Phase 2 probe signal figure to {figure_path}")
    return table


def write_probe_signal_figure(table: pd.DataFrame, figure_path: Path) -> None:
    executed = table[table["status"].eq("executed")].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if executed.empty:
        ax.text(
            0.5,
            0.5,
            "Probe-signal analysis blocked\n(no aligned route-state targets)",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
    else:
        labels = [str(value).replace("_state_predictor", "").replace("_", "\n") for value in executed["method"]]
        ax.bar(labels, executed["state_prediction_accuracy"].astype(float), color="#4C78A8")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("State prediction accuracy")
        ax.set_title("Probe Signal Gain")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def write_memo(
    out_dir: Path,
    probe_features_path: str,
    state_targets_path: str | None,
    query_features_path: str | None,
    table: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 2 Probe Signal Analysis",
        "",
        "Command:",
        "",
        "```bash",
        _command(probe_features_path, out_dir, state_targets_path, query_features_path),
        "```",
        "",
        _status_sentence(table),
        "",
        "Outputs:",
        "",
        "- `table_probe_signal_analysis.csv`",
        "- `fig_probe_signal_gain.pdf`",
        "- `m4_probe_signal_analysis_memo.md`",
        "",
        "Summary:",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "m4_probe_signal_analysis_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    probe_features_path: str,
    state_targets_path: str | None,
    query_features_path: str | None,
    table: pd.DataFrame,
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Probe Signal Analysis"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        _command(probe_features_path, out_dir, state_targets_path, query_features_path),
        "```",
        "",
        _status_sentence(table),
        "",
        "Outputs:",
        "",
        "- `table_probe_signal_analysis.csv`",
        "- `fig_probe_signal_gain.pdf`",
        "- `m4_probe_signal_analysis_memo.md`",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _read_optional_table(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def _status_sentence(table: pd.DataFrame) -> str:
    statuses = set(table["status"].astype(str))
    if statuses == {"executed"}:
        return "M4 executed on aligned probe features and route-state targets."
    return (
        "M4 currently cannot support probe-signal claims because the available probe features "
        "do not have aligned route-state targets for evaluation."
    )


def _command(
    probe_features_path: str,
    out_dir: Path,
    state_targets_path: str | None,
    query_features_path: str | None,
) -> str:
    parts = [
        "python experiments/53_probe_signal_analysis.py",
        f"--probe-features {probe_features_path}",
        f"--output-dir {out_dir}",
    ]
    if state_targets_path:
        parts.append(f"--state-targets {state_targets_path}")
    if query_features_path:
        parts.append(f"--query-features {query_features_path}")
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
