from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.claim_audit import audit_claims
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    table = audit_claims(out_dir)
    table.to_csv(out_dir / "table_claim_status.csv", index=False)
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote Phase H claim audit outputs to {out_dir}")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    lines = [
        "# Phase H Claim Status Memo",
        "",
        f"Command: `python experiments/34_claim_audit.py --config {config_path}`",
        "",
        "This memo converts the current result artifacts into claim-level gates. Do not use unsupported claims in paper text or README summaries.",
        "",
        "## Claim Status",
        "",
        _markdown_table(table),
        "",
        "## Interpretation",
        "",
        "- `supported` means the configured threshold is met by current artifacts.",
        "- `diagnostic_supported` means the evidence supports a diagnostic framing, not a broad paper claim.",
        "- `diagnostic_alive` means the claim remains worth testing but needs broader coverage or stronger baselines.",
        "- `not_supported` means current evidence argues against the claim or is below the declared threshold.",
        "- `missing_evidence` means required result artifacts are absent or too weak to verify the claim.",
        "",
    ]
    (out_dir / "phase_h_claim_status_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Phase H Claim Audit"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/34_claim_audit.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_claim_status.csv`: claim-level status, metric, threshold, and evidence pointers.",
        "- `phase_h_claim_status_memo.md`: interpretation memo for supported, diagnostic, unsupported, and missing-evidence claims.",
        "",
        _markdown_table(table),
        "",
    ]
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
