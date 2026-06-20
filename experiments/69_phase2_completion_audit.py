from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.eval.phase2_completion import audit_phase2_completion
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2"))
    args = parser.parse_args()
    run(root=args.root, output_dir=args.output_dir)


def run(*, root: str | Path, output_dir: str | Path) -> dict[str, str]:
    root_path = Path(root)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root_path / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table = audit_phase2_completion(root_path, out_dir)
    table_path = out_dir / "table_phase2_completion_audit.csv"
    memo_path = out_dir / "phase2_completion_audit.md"
    report_path = out_dir / "PHASE2_EVIDENCE_REPORT.md"
    table.to_csv(table_path, index=False)
    write_memo(memo_path, root_path, out_dir, table)
    append_report(report_path, root_path, out_dir, table)
    print(f"Wrote Phase 2 completion audit to {table_path}")
    return {"table": str(table_path), "memo": str(memo_path), "report": str(report_path)}


def write_memo(path: Path, root: Path, out_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Phase 2 Completion Audit",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/69_phase2_completion_audit.py --root {root} --output-dir {out_dir}",
        "```",
        "",
        "This audit checks the explicit `CODEX_GOAL_PHASE2.md` requirements against current files and result tables. It does not convert unsupported claims into supported claims.",
        "",
        "## Status Summary",
        "",
        _status_summary_table(table),
        "",
        "## Non-Complete Items",
        "",
        _markdown_table(_non_complete(table)),
        "",
        "## Full Audit",
        "",
        _markdown_table(table),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_report(path: Path, root: Path, out_dir: Path, table: pd.DataFrame) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Phase 2 Evidence Report\n"
    marker = "## Phase 2 Completion Audit"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/69_phase2_completion_audit.py --root {root} --output-dir {out_dir}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_phase2_completion_audit.csv`",
        "- `phase2_completion_audit.md`",
        "",
        "Status summary:",
        "",
        _status_summary_table(table),
        "",
        "Completed user gates:",
        "",
        _markdown_table(_complete_user_gates(table)),
        "",
        "Target-rate RouteCode policy artifacts:",
        "",
        "- `routecode_target_rate_policy_inputs_vllm_all200/true_probe_before_beliefs.csv`",
        "- `routecode_target_rate_policy_inputs_vllm_all200/true_probe_state_model_utility.csv`",
        "- `routecode_target_rate_policy_inputs_vllm_all200/routecode_target_rate_policy_input_metadata.json`",
        "- `routecode_target_rate_policy_vllm_all200/table_proberoute_policy.csv`",
        "- `routecode_target_rate_policy_vllm_all200/fig_gap_closed_vs_probe_cost.pdf`",
        "",
        "Current non-complete items:",
        "",
        _markdown_table(_non_complete(table)),
        "",
        "Interpretation: Phase 2 has the required artifact plumbing, including the 200-query two-local-model vLLM path, and the evidence report answers the definition-of-done questions. The best deployable latent-state policy artifact is now within 3% of oracle through the predeclared K=32 target-rate RouteCode path. The strict true-probe/VOI policy and the minimum-validation-gap RouteCode selector still miss, so this supports a working target-rate system but not a cheap-probe or VOI success claim. The exported benchmark-label policy remains an operational fallback rather than a core RouteCode/ProbeRoute++ success.",
        "",
    ]
    path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _non_complete(table: pd.DataFrame) -> pd.DataFrame:
    statuses = {"partial", "not_supported", "missing", "operational_fallback", "blocked_readiness"}
    cols = ["requirement_id", "category", "status", "metric", "notes", "evidence_paths"]
    return table[table["status"].isin(statuses)][cols]


def _complete_user_gates(table: pd.DataFrame) -> pd.DataFrame:
    cols = ["requirement_id", "category", "status", "metric", "notes", "evidence_paths"]
    return table[table["category"].eq("user_gate") & table["status"].eq("complete")][cols]


def _status_summary_table(table: pd.DataFrame) -> str:
    summary = (
        table.groupby(["category", "status"], dropna=False)
        .size()
        .reset_index(name="n_requirements")
        .sort_values(["category", "status"])
    )
    return _markdown_table(summary)


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
