from __future__ import annotations

from pathlib import Path

import pandas as pd

from routecode.eval.phase2_completion import audit_phase2_completion


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_phase2_completion_audit_marks_one_model_scale_partial_and_fallback_operational(tmp_path):
    phase2 = tmp_path / "results/phase2"
    for path in [
        *[f"results/phase2/{name}" for name in [
            "table_observability_strong_encoders.csv",
            "fig_observability_gap.pdf",
            "local_model_outcomes.parquet",
            "probe_features.parquet",
            "table_probe_signal_analysis.csv",
            "table_proberoute_policy.csv",
            "fig_gap_closed_vs_probe_cost.pdf",
            "table_active_new_model_calibration.csv",
            "fig_new_model_calibration_curve.pdf",
        ]],
        "src/routecode/states/observability.py",
        "src/routecode/probes/policies.py",
        "src/routecode/probes/probe_features.py",
        "src/routecode/local_eval/generation_runner.py",
        "src/routecode/local_eval/probe_runner.py",
        "src/routecode/local_eval/serve.py",
        "src/routecode/eval/new_model_calibration.py",
        "results/phase2/m0_previous_findings_recap.md",
        "results/phase2/local_server_readiness_vllm_qwen3_4b/table_local_server_readiness.csv",
        "results/phase2/exact_manifest_probes_vllm_qwen3_4b_all200_eval/table_probe_signal_analysis.csv",
        "results/phase2/table_active_calibration_replicate_summary.csv",
        "results/phase2/table_probe_cost_sensitivity_summary.csv",
        "results/phase2/table_active_calibration_sensitivity_summary.csv",
    ]:
        _touch(tmp_path / path)
    report = phase2 / "PHASE2_EVIDENCE_REPORT.md"
    report.write_text(
        "\n".join(
            [
                "Does the observability gap persist with strong encoders? Mixed, not resolved.",
                "Do cheap probes close a meaningful fraction of the gap? Not supported yet.",
                "Does VOI probing beat threshold or always-probe baselines after cost accounting? Not supported.",
                "Does active route-state calibration reduce new-model evaluations? Not supported yet.",
                "ICML/ICLR-style paper? Not yet.",
                "no GPT/Claude/Gemini API calls were made",
                "OpenAI GPT-family Anthropic Claude-family Google Gemini-family",
            ]
        ),
        encoding="utf-8",
    )
    _write_parquet(
        pd.DataFrame(
            {
                "query_id": [f"q{i}" for i in range(200)],
                "model_id": ["qwen3-4b"] * 200,
                "quality": [1.0] * 200,
                "error_type": [None] * 200,
            }
        ),
        phase2 / "local_vllm_qwen3_4b_all200_nothink/local_model_outcomes.parquet",
    )
    _write_parquet(
        pd.DataFrame(
            {
                "query_id": [f"q{i}" for i in range(20)],
                "model_id": ["qwen3-4b"] * 20,
                "quality": [1.0] * 20,
                "error_type": [None] * 20,
            }
        ),
        phase2 / "local_vllm_qwen3_4b_exact_smoke_nothink/local_model_outcomes.parquet",
    )
    _write_parquet(
        pd.DataFrame({"query_id": ["q0"], "error_type": [None]}),
        phase2 / "exact_manifest_probes_vllm_qwen3_4b_all200/exact_manifest_probe_features.parquet",
    )
    _write_csv(
        pd.DataFrame(
            {
                "predictor": ["query_plus_probe_state_predictor"],
                "state_prediction_accuracy": [0.65],
            }
        ),
        phase2 / "exact_manifest_probes_vllm_qwen3_4b_all200_eval/table_probe_signal_analysis.csv",
    )
    _write_csv(
        pd.DataFrame(
            {
                "policy": ["never_probe"],
                "selection_basis": ["current_phase2_policy_table"],
                "relative_gap_to_oracle": [0.08],
            }
        ),
        phase2 / "oracle_gap_gate_vllm_all200/table_oracle_gap_gate.csv",
    )
    _write_csv(
        pd.DataFrame({"relative_gap_to_oracle": [0.0]}),
        phase2 / "benchmark_label_policy_exact_math_vllm_all200/table_policy_summary.csv",
    )
    _write_csv(
        pd.DataFrame({"mean_probe_cost_proxy": [0.1], "mean_net_utility": [0.9]}),
        phase2 / "true_probe_policy_vllm_qwen3_4b_all200/table_proberoute_policy.csv",
    )
    _write_csv(
        pd.DataFrame(
            {
                "k": [4, 32],
                "selected_by_val": [True, False],
                "selected_by_target_rate": [False, True],
                "policy_slice_within_threshold": [False, True],
                "policy_slice_relative_gap_to_oracle": [0.05, 0.02],
                "val_relative_gap_to_oracle": [0.04, 0.06],
                "val_selection_rank": [1, 2],
            }
        ),
        phase2 / "routecode_exact_math_selection/table_routecode_exact_math_selection.csv",
    )

    table = audit_phase2_completion(tmp_path, phase2)
    rows = table.set_index("requirement_id")

    assert rows.loc["task_6_local_scale", "status"] == "partial"
    assert rows.loc["task_6_local_scale", "metric"] == "queries=200;rows=200;local_models=1"
    assert rows.loc["oracle_gap_core_policy_3pct", "status"] == "not_supported"
    assert rows.loc["oracle_gap_core_policy_3pct", "metric"] == "best_current_policy_relative_gap=0.0800"
    assert rows.loc["oracle_gap_true_probe_policy_3pct", "status"] == "not_supported"
    assert rows.loc["oracle_gap_true_probe_policy_3pct", "metric"] == "best_true_probe_policy_relative_gap=0.0800"
    assert rows.loc["oracle_gap_operational_fallback_3pct", "status"] == "operational_fallback"
    assert rows.loc["oracle_gap_target_rate_routecode_3pct", "status"] == "complete"
    assert rows.loc["oracle_gap_target_rate_routecode_3pct", "metric"] == (
        "target_k=32;target_rate_policy_slice_gap=0.0200;target_rate_val_gap=0.0600;"
        "target_rate_val_rank=2"
    )
    assert rows.loc["constraint_no_closed_api", "status"] == "complete"


def test_phase2_completion_audit_prefers_two_model_scale_artifacts(tmp_path):
    phase2 = tmp_path / "results/phase2"
    _write_parquet(
        pd.DataFrame(
            {
                "query_id": [f"q{i}" for i in range(200) for _ in range(2)],
                "model_id": ["qwen3_4b_vllm", "qwen3_0_6b_vllm"] * 200,
                "quality": [1.0, 0.0] * 200,
                "error_type": [None] * 400,
            }
        ),
        phase2 / "local_vllm_two_model_all200_nothink/local_model_outcomes.parquet",
    )
    for path in [
        "results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_utility.csv",
        "results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_utility.csv",
        "results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_query_model_utility.csv",
        "results/phase2/true_probe_policy_phase2_local_vllm_two_model_all200_nothink/table_proberoute_policy.csv",
    ]:
        _touch(tmp_path / path)

    table = audit_phase2_completion(tmp_path, phase2)
    rows = table.set_index("requirement_id")

    assert rows.loc["task_6_local_scale", "status"] == "complete"
    assert rows.loc["task_6_local_scale", "metric"] == "queries=200;rows=400;local_models=2"
    assert "local_vllm_two_model_all200_nothink" in rows.loc["task_6_local_scale", "evidence_paths"]
    assert rows.loc["task_6b_local_policy_matrix_handoff", "status"] == "complete"
    assert "Two-model local handoff" in rows.loc["task_6b_local_policy_matrix_handoff", "notes"]
