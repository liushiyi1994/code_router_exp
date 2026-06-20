from __future__ import annotations

from pathlib import Path

import pandas as pd

from routecode.eval.global_claim_audit import aggregate_claim_tables, audit_global_claims


def _write_claim_status(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "table_claim_status.csv", index=False)


def test_aggregate_claim_tables_keeps_unsupported_claims_unsupported(tmp_path):
    _write_claim_status(
        tmp_path / "pilot",
        [
            {
                "claim_id": "small_inferred_labels",
                "claim": "Small inferred route labels recover most routing performance.",
                "status": "not_supported",
                "primary_metric": "best_inferred_recovered_gap_vs_oracle",
                "primary_value": 0.34,
                "threshold": ">=0.85",
                "evidence": "pilot",
                "interpretation": "no",
            },
            {
                "claim_id": "new_model_calibration",
                "claim": "New models can be integrated with fewer calibration examples than direct retraining.",
                "status": "diagnostic_alive",
                "primary_metric": "mean_matched_routecode_minus_direct_recovered_gap",
                "primary_value": 0.20,
                "threshold": ">0",
                "evidence": "pilot",
                "interpretation": "alive",
            },
        ],
    )
    _write_claim_status(
        tmp_path / "broad20",
        [
            {
                "claim_id": "small_inferred_labels",
                "claim": "Small inferred route labels recover most routing performance.",
                "status": "not_supported",
                "primary_metric": "best_inferred_recovered_gap_vs_oracle",
                "primary_value": 0.09,
                "threshold": ">=0.85",
                "evidence": "broad",
                "interpretation": "no",
            },
            {
                "claim_id": "new_model_calibration",
                "claim": "New models can be integrated with fewer calibration examples than direct retraining.",
                "status": "diagnostic_alive",
                "primary_metric": "mean_matched_routecode_minus_direct_recovered_gap",
                "primary_value": 0.70,
                "threshold": ">0",
                "evidence": "broad",
                "interpretation": "alive",
            },
        ],
    )

    per_run, summary = audit_global_claims([tmp_path / "pilot", tmp_path / "broad20"])
    by_claim = summary.set_index("claim_id")

    assert set(per_run["result_id"]) == {"pilot", "broad20"}
    assert by_claim.loc["small_inferred_labels", "global_status"] == "not_supported"
    assert by_claim.loc["small_inferred_labels", "best_primary_value"] == 0.34
    assert by_claim.loc["new_model_calibration", "global_status"] == "diagnostic_alive"
    assert by_claim.loc["new_model_calibration", "best_primary_value"] == 0.70


def test_aggregate_claim_tables_marks_conflicting_nonmissing_evidence_as_mixed():
    per_run = pd.DataFrame(
        [
            {
                "result_id": "a",
                "claim_id": "model_pool_transfer",
                "claim": "Route labels transfer across model pools better than same-budget direct retraining.",
                "status": "diagnostic_alive",
                "primary_metric": "delta",
                "primary_value": 0.2,
                "evidence": "a",
            },
            {
                "result_id": "b",
                "claim_id": "model_pool_transfer",
                "claim": "Route labels transfer across model pools better than same-budget direct retraining.",
                "status": "not_supported",
                "primary_metric": "delta",
                "primary_value": -0.1,
                "evidence": "b",
            },
        ]
    )

    summary = aggregate_claim_tables(per_run)
    row = summary.set_index("claim_id").loc["model_pool_transfer"]

    assert row["global_status"] == "mixed_evidence"
    assert row["status_counts"] == "diagnostic_alive=1; not_supported=1"
