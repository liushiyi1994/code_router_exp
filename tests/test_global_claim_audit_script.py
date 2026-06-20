from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "35_global_claim_audit.py"
    spec = importlib.util.spec_from_file_location("global_claim_audit_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_claim_status(path: Path, value: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "claim_id": "small_inferred_labels",
                "claim": "Small inferred route labels recover most routing performance.",
                "status": "not_supported",
                "primary_metric": "best_inferred_recovered_gap_vs_oracle",
                "primary_value": value,
                "threshold": ">=0.85",
                "evidence": path.name,
                "interpretation": "no",
            }
        ]
    ).to_csv(path / "table_claim_status.csv", index=False)


def test_global_claim_audit_script_writes_global_tables_memo_and_readme(tmp_path):
    module = _load_script()
    _write_claim_status(tmp_path / "pilot", 0.34)
    _write_claim_status(tmp_path / "broad20", 0.09)
    out_dir = tmp_path / "results"

    module.run([tmp_path / "pilot", tmp_path / "broad20"], out_dir)

    per_run_path = out_dir / "table_claim_status_by_run.csv"
    summary_path = out_dir / "table_claim_status_global.csv"
    memo_path = out_dir / "phase_h_global_claim_status_memo.md"
    readme_path = out_dir / "README.md"
    assert per_run_path.exists()
    assert summary_path.exists()
    assert memo_path.exists()
    assert readme_path.exists()

    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "claim_id"] == "small_inferred_labels"
    assert summary.loc[0, "global_status"] == "not_supported"
    assert "Global Phase H Claim Audit" in memo_path.read_text(encoding="utf-8")
    assert "## Global Claim Audit" in readme_path.read_text(encoding="utf-8")
