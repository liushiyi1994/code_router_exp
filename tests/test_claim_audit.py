from __future__ import annotations

from pathlib import Path

import pandas as pd

from routecode.eval.claim_audit import audit_claims


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_claim_audit_blocks_inferred_label_claim_when_recovery_is_below_threshold(tmp_path):
    _write_csv(
        tmp_path / "table_predictability_constrained.csv",
        [
            {"method": "best_single", "recovered_gap_vs_oracle": 0.0},
            {"method": "d2_embedding_centroid", "recovered_gap_vs_oracle": 0.34, "utility_ci_low": 0.70},
            {"method": "d2_logistic_label_predictor", "recovered_gap_vs_oracle": 0.22, "utility_ci_low": 0.62},
            {"method": "d2_joint_oracle_labels", "recovered_gap_vs_oracle": 0.97, "utility_ci_low": 0.85},
        ],
    )
    _write_csv(
        tmp_path / "table_recovered_gap.csv",
        [
            {"method": "dataset_label_lookup", "recovered_gap_vs_oracle": 0.38},
            {"method": "predicted_topic_lookup", "recovered_gap_vs_oracle": 0.12},
        ],
    )

    table = audit_claims(tmp_path)
    claim = table.set_index("claim_id").loc["small_inferred_labels"]

    assert claim["status"] == "not_supported"
    assert claim["primary_metric"] == "best_inferred_recovered_gap_vs_oracle"
    assert claim["primary_value"] == 0.34
    assert "0.85" in claim["threshold"]
    assert "d2_embedding_centroid" in claim["evidence"]


def test_claim_audit_marks_transfer_calibration_and_split_diagnostics(tmp_path):
    _write_csv(
        tmp_path / "table_model_pool_transfer.csv",
        [
            {
                "method": "source_d2_label_transfer",
                "recovered_gap_vs_oracle": 0.21,
                "same_budget_as_direct_retraining": True,
            },
            {
                "method": "target_direct_mlp",
                "recovered_gap_vs_oracle": 0.04,
                "same_budget_as_direct_retraining": True,
            },
        ],
    )
    _write_csv(
        tmp_path / "table_new_model_integration.csv",
        [
            {"method": "routecode_label_calibration", "recovered_gap_vs_oracle": 0.30},
            {"method": "direct_retraining_budgeted_mlp", "recovered_gap_vs_oracle": -0.02},
        ],
    )
    _write_csv(
        tmp_path / "table_split_rank_correlation.csv",
        [
            {"scenario": "leave_dataset_out:mbpp", "rank_correlation_vs_random": 0.12},
            {"scenario": "cluster_held_out:1", "rank_correlation_vs_random": 0.48},
        ],
    )
    _write_csv(
        tmp_path / "table_residual_risk.csv",
        [
            {"score": "low_route_label_confidence", "top_fraction": 0.10, "regret_mass_fraction": 0.11},
        ],
    )

    table = audit_claims(tmp_path).set_index("claim_id")

    assert table.loc["model_pool_transfer", "status"] == "diagnostic_alive"
    assert table.loc["new_model_calibration", "status"] == "diagnostic_alive"
    assert table.loc["benchmark_diagnosis", "status"] == "diagnostic_supported"
    assert table.loc["adaptive_refinement", "status"] == "not_supported"


def test_claim_audit_compares_transfer_within_matching_scenarios(tmp_path):
    _write_csv(
        tmp_path / "table_model_pool_transfer.csv",
        [
            {
                "method": "source_d2_label_transfer",
                "transfer_scenario": "easy",
                "recovered_gap_vs_oracle": 0.90,
            },
            {
                "method": "target_direct_mlp",
                "transfer_scenario": "easy",
                "recovered_gap_vs_oracle": 0.89,
            },
            {
                "method": "source_d2_label_transfer",
                "transfer_scenario": "hard",
                "recovered_gap_vs_oracle": 0.10,
            },
            {
                "method": "target_direct_mlp",
                "transfer_scenario": "hard",
                "recovered_gap_vs_oracle": 0.70,
            },
        ],
    )

    table = audit_claims(tmp_path).set_index("claim_id")

    assert table.loc["model_pool_transfer", "status"] == "not_supported"
    assert table.loc["model_pool_transfer", "primary_metric"] == "mean_matched_transfer_minus_direct_recovered_gap"
    assert table.loc["model_pool_transfer", "primary_value"] == -0.295
