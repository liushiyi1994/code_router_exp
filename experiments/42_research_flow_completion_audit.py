from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.eval.research_flow_completion import audit_research_flow_completion
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.root, args.output_dir)


def run(root: str | Path, output_dir: str | Path) -> None:
    root_path = Path(root)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root_path / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table = audit_research_flow_completion(root_path)
    table.to_csv(out_dir / "table_research_flow_completion.csv", index=False)
    write_memo(out_dir, root_path, table)
    append_readme(out_dir, root_path, table)
    print(f"Wrote Research Flow completion audit outputs to {out_dir}")


def write_memo(out_dir: Path, root: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Research Flow Completion Audit",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/42_research_flow_completion_audit.py --root {root} --output-dir {out_dir}",
        "```",
        "",
        "This audit checks the explicit phases in `Research Flow.md` against current result artifacts. It is a completion audit, not a claim that the full project is complete.",
        "",
        "## Phase Status",
        "",
        _markdown_table(table),
        "",
        "## Interpretation",
        "",
        *_interpretation_lines(table),
        "",
    ]
    (out_dir / "phase_h_research_flow_completion_audit.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, root: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Results\n"
    marker = "## Research Flow Completion Audit"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/42_research_flow_completion_audit.py --root {root} --output-dir {out_dir}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_research_flow_completion.csv`: phase-by-phase completion evidence.",
        "- `phase_h_research_flow_completion_audit.md`: completion audit memo.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _interpretation_lines(table: pd.DataFrame) -> list[str]:
    counts = table["status"].value_counts().sort_index()
    lines = [f"- `{status}` phases: `{count}`." for status, count in counts.items()]
    unfinished = table[~table["status"].isin(["complete", "deferred"])]
    if unfinished.empty:
        lines.append("- All required phases are either complete or explicitly deferred.")
    else:
        lines.append(
            "- Remaining non-complete phases: "
            + ", ".join(f"`{row['phase_id']}` ({row['status']})" for _, row in unfinished.iterrows())
            + "."
        )
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
