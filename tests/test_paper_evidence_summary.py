from __future__ import annotations

from pathlib import Path

import pandas as pd

from routecode.eval.paper_evidence import build_paper_evidence_summary


def test_paper_evidence_summary_recommends_information_frontier_when_inferred_labels_fail():
    claims = pd.DataFrame(
        [
            {
                "claim_id": "low_rate_oracle_codes",
                "claim": "Useful low-rate utility route codes exist.",
                "global_status": "diagnostic_supported",
                "best_primary_value": 1.0,
                "worst_primary_value": 0.95,
                "best_result_id": "pilot",
                "evidence_summary": "pilot: diagnostic_supported",
                "interpretation": "Use diagnostic framing.",
            },
            {
                "claim_id": "small_inferred_labels",
                "claim": "Small inferred route labels recover most routing performance.",
                "global_status": "not_supported",
                "best_primary_value": 0.34,
                "worst_primary_value": 0.09,
                "best_result_id": "pilot",
                "evidence_summary": "pilot: not_supported",
                "interpretation": "Do not claim this.",
            },
            {
                "claim_id": "new_model_calibration",
                "claim": "New models can be integrated with fewer calibration examples than direct retraining.",
                "global_status": "diagnostic_alive",
                "best_primary_value": 0.81,
                "worst_primary_value": 0.23,
                "best_result_id": "broad20",
                "evidence_summary": "broad20: diagnostic_alive",
                "interpretation": "Use diagnostic framing.",
            },
        ]
    )
    readiness = pd.DataFrame(
        [
            {
                "check_id": "routecode_upstream_avengerspro_metric",
                "status": "available",
                "runnable_now": True,
                "exact_upstream_command": False,
                "routecode_metric_compatible": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/avengerspro_upstream_metric/raw_routing_details.json",
            },
            {
                "check_id": "routellm_bert_cli",
                "status": "blocked",
                "runnable_now": False,
                "exact_upstream_command": False,
                "routecode_metric_compatible": False,
                "blocking_reasons": "missing_bert_checkpoint",
                "execution_evidence": "",
            },
        ]
    )

    summary = build_paper_evidence_summary(
        claims,
        {"pilot": readiness},
        readiness_paths={"pilot": Path("results/pilot/table_external_command_readiness.csv")},
    )
    rows = summary.set_index(["section", "item"])

    direction = rows.loc[("paper_direction", "recommended_framing")]
    assert direction["status"] == "information_frontier_diagnostic"
    assert "information-frontier" in direction["interpretation"]
    assert "few inferred bits" in direction["interpretation"]

    readiness_row = rows.loc[("external_baselines", "readiness_overview")]
    assert readiness_row["status"] == "partial"
    assert "2 rows" in readiness_row["key_value"]
    assert "1 runnable" in readiness_row["key_value"]
    assert "0 exact" in readiness_row["key_value"]

    blocker = rows.loc[("external_baselines", "routellm_bert_cli")]
    assert blocker["status"] == "blocked"
    assert "missing_bert_checkpoint" in blocker["interpretation"]

    claim = rows.loc[("claim", "small_inferred_labels")]
    assert claim["status"] == "not_supported"
    assert claim["key_value"] == "best=0.3400; worst=0.0900"
