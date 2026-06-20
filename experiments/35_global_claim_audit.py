from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.eval.global_claim_audit import audit_global_claims
from routecode.reporting import upsert_markdown_section


DEFAULT_RESULT_DIRS = [
    Path("results/llmrouterbench_pilot"),
    Path("results/llmrouterbench_broad10"),
    Path("results/llmrouterbench_broad20"),
    Path("results/llmrouterbench_scale20"),
    Path("results/llmrouterbench_32model"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    result_dirs = args.result_dir or [path for path in DEFAULT_RESULT_DIRS if path.exists()]
    run(result_dirs, args.output_dir)


def run(result_dirs: list[str | Path], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run, summary = audit_global_claims(result_dirs)
    per_run.to_csv(out_dir / "table_claim_status_by_run.csv", index=False)
    summary.to_csv(out_dir / "table_claim_status_global.csv", index=False)
    write_memo(out_dir, result_dirs, summary)
    append_readme(out_dir, result_dirs, summary)
    print(f"Wrote global Phase H claim audit outputs to {out_dir}")


def write_memo(out_dir: Path, result_dirs: list[str | Path], summary: pd.DataFrame) -> None:
    lines = [
        "# Global Phase H Claim Audit",
        "",
        "Command:",
        "",
        "```bash",
        _command(result_dirs, out_dir),
        "```",
        "",
        "This memo aggregates per-run Phase H claim gates across the supplied result directories. It is intentionally conservative: contradictory nonmissing evidence becomes `mixed_evidence`, and missing run evidence is counted but does not by itself create support.",
        "",
        "Result directories:",
        "",
        *[f"- `{Path(path)}`" for path in result_dirs],
        "",
        "## Global Claim Status",
        "",
        _markdown_table(summary),
        "",
    ]
    (out_dir / "phase_h_global_claim_status_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, result_dirs: list[str | Path], summary: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Results\n"
    marker = "## Global Claim Audit"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        _command(result_dirs, out_dir),
        "```",
        "",
        "Outputs:",
        "",
        "- `table_claim_status_by_run.csv`: per-run claim gates.",
        "- `table_claim_status_global.csv`: conservative cross-run claim status.",
        "- `phase_h_global_claim_status_memo.md`: global claim-gate memo.",
        "",
        _markdown_table(summary),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _command(result_dirs: list[str | Path], out_dir: Path) -> str:
    parts = ["python experiments/35_global_claim_audit.py"]
    for result_dir in result_dirs:
        parts.extend(["--result-dir", str(Path(result_dir))])
    parts.extend(["--output-dir", str(out_dir)])
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
