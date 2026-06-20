from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.eval.phase_e_coverage import audit_phase_e_baseline_coverage, required_coverage_complete
from routecode.reporting import upsert_markdown_section


TABLE_PATHS = {
    "recovered_gap": "table_recovered_gap.csv",
    "routability": "table_routability.csv",
    "routellm_mf": "table_routellm_mf_split_aligned.csv",
    "llmrouter_library": "table_llmrouter_library_adapters.csv",
    "graphrouter": "table_graphrouter_split_aligned.csv",
    "avengerspro": "table_avengerspro_upstream_metric.csv",
    "cost_quality": "table_cost_quality_frontier.csv",
    "readiness": "table_external_command_readiness.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("results/llmrouterbench_pilot"))
    args = parser.parse_args()
    run(args.result_dir)


def run(result_dir: str | Path) -> None:
    out_dir = Path(result_dir)
    tables = _load_tables(out_dir)
    coverage = audit_phase_e_baseline_coverage(tables)
    coverage.to_csv(out_dir / "table_phase_e_baseline_coverage.csv", index=False)
    write_memo(out_dir, coverage)
    append_readme(out_dir, coverage)
    print(f"Wrote Phase E baseline coverage outputs to {out_dir}")


def _load_tables(result_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for key, filename in TABLE_PATHS.items():
        path = result_dir / filename
        tables[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def write_memo(out_dir: Path, coverage: pd.DataFrame) -> None:
    complete = required_coverage_complete(coverage)
    lines = [
        "# Phase E Baseline Coverage Memo",
        "",
        f"Command: `python experiments/44_phase_e_baseline_coverage.py --result-dir {out_dir}`",
        "",
        "This memo maps the Phase E baseline list in `Research Flow.md` to existing run artifacts. Extra checkpoint-heavy external methods are documented separately and do not replace the required baseline coverage evidence.",
        "",
        f"Required/conditional baseline coverage complete: `{complete}`.",
        "",
        "## Coverage",
        "",
        _markdown_table(coverage),
        "",
    ]
    (out_dir / "phase_e_baseline_coverage_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, coverage: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## Phase E Baseline Coverage"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/44_phase_e_baseline_coverage.py --result-dir {out_dir}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_phase_e_baseline_coverage.csv`: Research Flow Phase E baseline coverage audit.",
        "- `phase_e_baseline_coverage_memo.md`: memo distinguishing required/conditional baseline coverage from optional checkpoint-gated external rows.",
        "",
        f"Required/conditional baseline coverage complete: `{required_coverage_complete(coverage)}`.",
        "",
        _markdown_table(coverage),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


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
        return "" if pd.isna(value) else f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
