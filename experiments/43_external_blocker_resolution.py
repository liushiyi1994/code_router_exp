from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.eval.external_blockers import summarize_external_blockers
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readiness-table",
        action="append",
        dest="readiness_tables",
        help="Path to a table_external_command_readiness.csv file. May be repeated.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    tables = args.readiness_tables or [
        str(ROOT / "results/llmrouterbench_pilot/table_external_command_readiness.csv"),
        str(ROOT / "results/llmrouterbench_broad20/table_external_command_readiness.csv"),
    ]
    run(tables, args.output_dir)


def run(readiness_table_paths: list[str] | list[Path], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readiness_tables = {
        _infer_run_name(Path(path)): pd.read_csv(path) for path in readiness_table_paths if Path(path).exists()
    }
    summary = summarize_external_blockers(readiness_tables)
    summary.to_csv(out_dir / "table_external_blocker_resolution.csv", index=False)
    write_memo(out_dir, readiness_table_paths, summary)
    append_readme(out_dir, readiness_table_paths, summary)
    print(f"Wrote external blocker resolution outputs to {out_dir}")


def write_memo(out_dir: Path, readiness_table_paths: list[str] | list[Path], summary: pd.DataFrame) -> None:
    checkpoint_gated = _checkpoint_gated_count(summary)
    module_only = _module_only_count(summary)
    lines = [
        "# Phase E External Blocker Resolution Memo",
        "",
        "This memo aggregates blocked exact-command readiness rows across RouteCode runs. It performs no downloads, installs, or external API calls.",
        "",
        "Inputs:",
        "",
        *[f"- `{path}`" for path in readiness_table_paths],
        "",
        f"Blocked rows: `{len(summary)}`.",
        f"Checkpoint-gated blocked rows: `{checkpoint_gated}`.",
        f"Module-only blocked rows: `{module_only}`.",
        "",
        "## Blockers",
        "",
        _markdown_table(summary),
        "",
        "## Interpretation",
        "",
        *_interpretation_lines(summary),
        "",
    ]
    (out_dir / "phase_e_external_blocker_resolution_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, readiness_table_paths: list[str] | list[Path], summary: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## External Blocker Resolution"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        "python experiments/43_external_blocker_resolution.py",
        "```",
        "",
        "Inputs:",
        "",
        *[f"- `{path}`" for path in readiness_table_paths],
        "",
        "Outputs:",
        "",
        "- `table_external_blocker_resolution.csv`: blocked external-command rows grouped across runs with missing modules, checkpoints, local assets, service requirements, and next actions.",
        "- `phase_e_external_blocker_resolution_memo.md`: interpretation memo for the unresolved Phase E blockers.",
        "",
        f"Blocked rows: `{len(summary)}`.",
        f"Checkpoint-gated blocked rows: `{_checkpoint_gated_count(summary)}`.",
        "",
        _markdown_table(summary),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _infer_run_name(path: Path) -> str:
    return path.parent.name or path.stem


def _checkpoint_gated_count(summary: pd.DataFrame) -> int:
    if summary.empty or "missing_checkpoints" not in summary.columns:
        return 0
    return int(summary["missing_checkpoints"].fillna("").astype(str).ne("").sum())


def _module_only_count(summary: pd.DataFrame) -> int:
    if summary.empty:
        return 0
    modules = summary.get("missing_modules", pd.Series("", index=summary.index)).fillna("").astype(str)
    checkpoints = summary.get("missing_checkpoints", pd.Series("", index=summary.index)).fillna("").astype(str)
    assets = summary.get("missing_assets", pd.Series("", index=summary.index)).fillna("").astype(str)
    service = summary.get("service_requirements", pd.Series("", index=summary.index)).fillna("").astype(str)
    return int((modules.ne("") & checkpoints.eq("") & assets.eq("") & service.eq("")).sum())


def _interpretation_lines(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["- No blocked external-command rows are present in the input readiness tables."]
    lines = [
        "- Rows with `missing_checkpoints` require local checkpoint/model assets before they can become runnable; installing Python packages alone is insufficient.",
        "- Rows with only `missing_modules` can be advanced locally by installing the listed modules, subject to compatibility with the current Python environment.",
        "- Rows with service requirements should use cached/local embeddings to preserve the no-external-API constraint.",
    ]
    blocked_ids = ", ".join(f"`{item}`" for item in summary["check_id"].astype(str))
    lines.append(f"- Current unresolved rows: {blocked_ids}.")
    return lines


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
