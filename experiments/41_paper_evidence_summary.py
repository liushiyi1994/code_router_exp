from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.eval.paper_evidence import build_paper_evidence_summary
from routecode.reporting import upsert_markdown_section


DEFAULT_READINESS_TABLES = [
    Path("results/llmrouterbench_pilot/table_external_command_readiness.csv"),
    Path("results/llmrouterbench_broad20/table_external_command_readiness.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--readiness-table", action="append", type=Path, default=[])
    parser.add_argument("--paper-notes", type=Path, default=Path("paper_notes.md"))
    args = parser.parse_args()
    readiness_tables = args.readiness_table or [path for path in DEFAULT_READINESS_TABLES if path.exists()]
    run(args.output_dir, readiness_tables, args.paper_notes)


def run(output_dir: str | Path, readiness_table_paths: list[str | Path], paper_notes_path: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    claims_path = out_dir / "table_claim_status_global.csv"
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    readiness_tables = {}
    readiness_paths = {}
    for path_like in readiness_table_paths:
        path = Path(path_like)
        if not path.exists():
            continue
        run_id = path.parent.name
        readiness_tables[run_id] = pd.read_csv(path)
        readiness_paths[run_id] = path
    summary = build_paper_evidence_summary(claims, readiness_tables, readiness_paths=readiness_paths)
    summary.to_csv(out_dir / "table_paper_evidence_summary.csv", index=False)
    write_memo(out_dir, readiness_table_paths, summary)
    append_readme(out_dir, readiness_table_paths, summary, Path(paper_notes_path))
    completion_path = out_dir / "table_research_flow_completion.csv"
    completion = pd.read_csv(completion_path) if completion_path.exists() else pd.DataFrame()
    write_paper_notes(Path(paper_notes_path), summary, completion)
    print(f"Wrote paper evidence summary outputs to {out_dir}")


def write_memo(out_dir: Path, readiness_table_paths: list[str | Path], summary: pd.DataFrame) -> None:
    lines = [
        "# Phase H Paper Evidence Summary",
        "",
        "Command:",
        "",
        "```bash",
        _command(out_dir, readiness_table_paths),
        "```",
        "",
        "This memo translates current claim gates and external-readiness evidence into conservative paper-positioning notes. It does not upgrade diagnostic evidence into final paper claims.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
    ]
    (out_dir / "phase_h_paper_evidence_summary.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    readiness_table_paths: list[str | Path],
    summary: pd.DataFrame,
    paper_notes_path: Path,
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Results\n"
    marker = "## Paper Evidence Summary"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        _command(out_dir, readiness_table_paths),
        "```",
        "",
        "Outputs:",
        "",
        "- `table_paper_evidence_summary.csv`: paper-facing claim and baseline posture table.",
        "- `phase_h_paper_evidence_summary.md`: conservative paper-positioning memo.",
        f"- `{_relative_to(out_dir, paper_notes_path)}`: root paper notes generated from the same evidence table.",
        "",
        _markdown_table(summary),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_paper_notes(path: Path, summary: pd.DataFrame, completion: pd.DataFrame | None = None) -> None:
    rows = summary.set_index(["section", "item"]) if not summary.empty else pd.DataFrame()
    direction = _row(rows, "paper_direction", "recommended_framing")
    claim_rows = summary[summary["section"] == "claim"] if "section" in summary.columns else pd.DataFrame()
    external_rows = summary[summary["section"] == "external_baselines"] if "section" in summary.columns else pd.DataFrame()
    blockers = external_rows[external_rows["status"] == "blocked"] if not external_rows.empty else pd.DataFrame()
    lines = [
        "# RouteCode Paper Notes",
        "",
        "Last updated: 2026-06-15",
        "",
        "These notes are generated from current Phase H claim gates and external-baseline readiness artifacts. They are not a paper draft and should not be read as final claims.",
        "",
        "## Recommended Framing",
        "",
        f"- Status: `{direction.get('status', '')}`.",
        f"- {direction.get('interpretation', '')}",
        "",
        "## Claim Posture",
        "",
        _markdown_table(claim_rows[["item", "status", "key_value", "interpretation"]] if not claim_rows.empty else claim_rows),
        "",
        "## External Baseline Posture",
        "",
        _markdown_table(external_rows[["item", "status", "key_value", "interpretation"]] if not external_rows.empty else external_rows),
        "",
        "## Research Flow Completion",
        "",
        *_research_flow_lines(completion if completion is not None else pd.DataFrame()),
        "",
        "## Remaining Blockers",
        "",
    ]
    if blockers.empty:
        lines.append("- No blocked external-baseline rows in the supplied readiness tables.")
    else:
        lines.extend(
            f"- `{row['item']}`: {row['interpretation']}"
            for _, row in blockers.iterrows()
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not frame the project as saving router tokens.",
            "- Do not claim that small inferred route labels recover most routing performance unless the pre-committed threshold is met with confidence intervals.",
            "- Keep calibration, transfer, and benchmark-diagnosis claims diagnostic until broader external-baseline and robustness coverage is available.",
            "- Keep adaptive refinement deferred unless a stronger deployable residual-risk gate appears.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _research_flow_lines(completion: pd.DataFrame) -> list[str]:
    if completion.empty or "status" not in completion.columns:
        return [
            "Evidence: `results/table_research_flow_completion.csv` and `results/phase_h_research_flow_completion_audit.md`.",
            "",
            "- Research Flow completion audit has not been generated yet.",
        ]
    counts = completion["status"].value_counts().to_dict()
    lines = [
        "Evidence: `results/table_research_flow_completion.csv` and `results/phase_h_research_flow_completion_audit.md`.",
        "",
        f"- Complete phases: `{int(counts.get('complete', 0))}`.",
        _status_line(completion, "deferred", "Deferred phases"),
        _status_line(completion, "blocked", "Blocked phases"),
        _status_line(completion, "incomplete", "Incomplete phases"),
    ]
    unfinished = completion[~completion["status"].isin(["complete", "deferred"])]
    if unfinished.empty:
        lines.append("- Current non-complete reasons: none.")
    else:
        reasons = "; ".join(
            f"{row['phase_id']}: {row.get('notes', '')}"
            for _, row in unfinished.iterrows()
        )
        lines.append(f"- Current non-complete reasons: {reasons}.")
    return lines


def _status_line(completion: pd.DataFrame, status: str, label: str) -> str:
    rows = completion[completion["status"] == status]
    if rows.empty:
        return f"- {label}: `0`."
    phase_ids = ", ".join(f"`{item}`" for item in rows["phase_id"].astype(str))
    return f"- {label}: `{len(rows)}` ({phase_ids})."


def _row(rows: pd.DataFrame, section: str, item: str) -> dict[str, object]:
    if rows.empty or (section, item) not in rows.index:
        return {}
    return rows.loc[(section, item)].to_dict()


def _command(out_dir: Path, readiness_table_paths: list[str | Path]) -> str:
    parts = ["python experiments/41_paper_evidence_summary.py", "--output-dir", str(out_dir)]
    for path in readiness_table_paths:
        parts.extend(["--readiness-table", str(Path(path))])
    parts.extend(["--paper-notes", "paper_notes.md"])
    return " ".join(parts)


def _relative_to(out_dir: Path, path: Path) -> str:
    return os.path.relpath(path, start=out_dir)


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
