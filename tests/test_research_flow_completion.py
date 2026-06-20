from __future__ import annotations

from pathlib import Path

import pandas as pd

from routecode.eval.research_flow_completion import audit_research_flow_completion


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def test_research_flow_completion_audit_marks_complete_deferred_and_conservative_claim_phases(tmp_path):
    for path in [
        "results/demo/table_routability.csv",
        "results/demo/table_recovered_gap.csv",
        "results/demo/table_rate_distortion.csv",
        "results/demo/fig_compression_ladder.pdf",
        "results/demo/fig_rate_distortion.pdf",
        "results/llmrouterbench_pilot/table_routability.csv",
        "results/llmrouterbench_pilot/table_recovered_gap.csv",
        "results/llmrouterbench_pilot/table_rate_distortion.csv",
        "results/llmrouterbench_pilot/table_residual_concentration.csv",
        "results/llmrouterbench_pilot/table_split_sensitivity.csv",
        "results/llmrouterbench_pilot/phase_d5_adaptive_refinement_gate_memo.md",
        "results/llmrouterbench_pilot/phase_c_observation_memo.md",
        "results/llmrouterbench_pilot/table_predictability_constrained.csv",
        "results/llmrouterbench_pilot/code_cards.md",
        "results/llmrouterbench_pilot/table_new_model_integration.csv",
        "results/llmrouterbench_pilot/table_external_command_readiness.csv",
        "results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv",
        "results/llmrouterbench_pilot/table_ablation_summary.csv",
        "results/llmrouterbench_pilot/table_sensitivity_summary.csv",
        "results/table_claim_status_global.csv",
        "results/table_paper_evidence_summary.csv",
        "paper_notes.md",
    ]:
        _touch(tmp_path / path)
    pd.DataFrame(
        [
            {
                "check_id": "routellm_bert_cli",
                "status": "blocked",
                "runnable_now": False,
                "routecode_metric_compatible": False,
                "exact_upstream_command": False,
                "blocking_reasons": "missing_bert_checkpoint",
            }
        ]
    ).to_csv(tmp_path / "results/llmrouterbench_pilot/table_external_command_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "requirement_id": "random",
                "requirement_type": "required",
                "status": "present",
                "evidence": "random",
            },
            {
                "requirement_id": "optional_extra_external_blockers",
                "requirement_type": "optional",
                "status": "present",
                "evidence": "routellm_bert_cli",
            },
        ]
    ).to_csv(tmp_path / "results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "claim_id": "small_inferred_labels",
                "global_status": "not_supported",
            }
        ]
    ).to_csv(tmp_path / "results/table_claim_status_global.csv", index=False)
    pd.DataFrame(
        [
            {
                "section": "paper_direction",
                "item": "recommended_framing",
                "status": "information_frontier_diagnostic",
                "interpretation": "Do not claim that few inferred bits are enough.",
            }
        ]
    ).to_csv(tmp_path / "results/table_paper_evidence_summary.csv", index=False)

    table = audit_research_flow_completion(tmp_path)
    rows = table.set_index("phase_id")

    assert rows.loc["phase_a_synthetic_sanity", "status"] == "complete"
    assert rows.loc["phase_d5_adaptive_refinement", "status"] == "deferred"
    assert rows.loc["phase_e_external_methods", "status"] == "complete"
    assert "Required Phase E baseline coverage is complete" in rows.loc["phase_e_external_methods", "notes"]
    assert "Optional checkpoint-gated rows documented: routellm_bert_cli" in rows.loc["phase_e_external_methods", "notes"]
    assert rows.loc["phase_h_final_claims", "status"] == "complete"
    assert "Conservative final claim posture documented" in rows.loc["phase_h_final_claims", "notes"]
    assert "small_inferred_labels=not_supported" in rows.loc["phase_h_final_claims", "notes"]


def test_research_flow_completion_audit_fails_phase_e_when_required_coverage_missing(tmp_path):
    for path in [
        "results/llmrouterbench_pilot/table_external_command_readiness.csv",
        "results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv",
        "results/table_external_blocker_resolution.csv",
    ]:
        _touch(tmp_path / path)
    pd.DataFrame(
        [
            {
                "check_id": "routerdc_train_cli",
                "status": "blocked",
                "runnable_now": False,
                "routecode_metric_compatible": False,
                "blocking_reasons": "missing_routerdc_local_model_checkpoint",
            }
        ]
    ).to_csv(tmp_path / "results/llmrouterbench_pilot/table_external_command_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "requirement_id": "route_llm_if_easy",
                "requirement_type": "conditional",
                "status": "missing",
                "evidence": "",
            }
        ]
    ).to_csv(tmp_path / "results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv", index=False)

    table = audit_research_flow_completion(tmp_path)
    notes = table.set_index("phase_id").loc["phase_e_external_methods", "notes"]

    assert table.set_index("phase_id").loc["phase_e_external_methods", "status"] == "missing_evidence"
    assert "Missing required Phase E baseline coverage: route_llm_if_easy" in notes
