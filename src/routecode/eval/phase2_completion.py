from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AuditRow:
    requirement_id: str
    category: str
    requirement: str
    status: str
    evidence_paths: str
    metric: str
    notes: str


MINIMUM_DELIVERABLES = (
    "table_observability_strong_encoders.csv",
    "fig_observability_gap.pdf",
    "local_model_outcomes.parquet",
    "probe_features.parquet",
    "table_probe_signal_analysis.csv",
    "table_proberoute_policy.csv",
    "fig_gap_closed_vs_probe_cost.pdf",
    "table_active_new_model_calibration.csv",
    "fig_new_model_calibration_curve.pdf",
    "PHASE2_EVIDENCE_REPORT.md",
)


def audit_phase2_completion(root: str | Path, phase2_dir: str | Path = "results/phase2") -> pd.DataFrame:
    root_path = Path(root)
    out_dir = Path(phase2_dir)
    if not out_dir.is_absolute():
        out_dir = root_path / out_dir
    rows: list[AuditRow] = []
    rows.extend(_minimum_deliverable_rows(root_path, out_dir))
    rows.extend(_implementation_task_rows(root_path, out_dir))
    rows.extend(_definition_of_done_rows(root_path, out_dir))
    rows.extend(_constraint_rows(root_path, out_dir))
    rows.extend(_oracle_target_rows(root_path, out_dir))
    return pd.DataFrame([row.__dict__ for row in rows])


def _minimum_deliverable_rows(root: Path, out_dir: Path) -> list[AuditRow]:
    rows = []
    for filename in MINIMUM_DELIVERABLES:
        path = out_dir / filename
        rows.append(
            AuditRow(
                requirement_id=f"deliverable_{filename}",
                category="minimum_deliverable",
                requirement=f"`results/phase2/{filename}` exists and is non-empty.",
                status="complete" if _exists_nonempty(path) else "missing",
                evidence_paths=_relative(root, path),
                metric="present=1" if _exists_nonempty(path) else "present=0",
                notes="Minimum deliverable from CODEX_GOAL_PHASE2.md.",
            )
        )
    return rows


def _implementation_task_rows(root: Path, out_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    module_paths = [
        root / "src/routecode/states/observability.py",
        root / "src/routecode/probes/policies.py",
        root / "src/routecode/probes/probe_features.py",
        root / "src/routecode/local_eval/generation_runner.py",
        root / "src/routecode/local_eval/probe_runner.py",
        root / "src/routecode/eval/new_model_calibration.py",
    ]
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_1_modules",
            category="implementation_task",
            requirement="Add modules for latent states, probes, belief updates, VOI policies, local evaluation, and active calibration.",
            paths=module_paths,
            notes="Core module files exist.",
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_2_phase1_recap",
            category="implementation_task",
            requirement="Reproduce Phase 1 observability-gap numbers from existing results.",
            paths=[out_dir / "m0_previous_findings_recap.md"],
            notes="Recap memo is present.",
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_3_strong_encoder_audit",
            category="implementation_task",
            requirement="Run strong-encoder observability audit.",
            paths=[out_dir / "table_observability_strong_encoders.csv", out_dir / "fig_observability_gap.pdf"],
            notes="Strong-encoder audit artifacts are present.",
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_4_vllm_runner",
            category="implementation_task",
            requirement="Implement local generation runner with vLLM-compatible OpenAI client and dry-run mode.",
            paths=[
                root / "src/routecode/local_eval/generation_runner.py",
                root / "src/routecode/local_eval/serve.py",
                out_dir / "local_server_readiness_vllm_qwen3_4b/table_local_server_readiness.csv",
            ],
            notes="Runner and vLLM readiness artifacts are present.",
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_5_vllm_20_query_smoke",
            category="implementation_task",
            requirement="Run a 20-query local smoke test with one model.",
            paths=[out_dir / "local_vllm_qwen3_4b_exact_smoke_nothink/local_model_outcomes.parquet"],
            notes=_local_outcome_notes(out_dir / "local_vllm_qwen3_4b_exact_smoke_nothink/local_model_outcomes.parquet"),
        )
    )
    rows.append(_local_scale_row(root, out_dir))
    rows.append(_local_vllm_pipeline_row(root, out_dir))
    rows.append(_local_policy_handoff_row(root, out_dir))
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_7_probe_features",
            category="implementation_task",
            requirement="Collect probe features with short cheap probes.",
            paths=[out_dir / "exact_manifest_probes_vllm_qwen3_4b_all200/exact_manifest_probe_features.parquet"],
            notes=_probe_feature_notes(out_dir / "exact_manifest_probes_vllm_qwen3_4b_all200/exact_manifest_probe_features.parquet"),
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_8_query_probe_predictor",
            category="implementation_task",
            requirement="Train query+probe state predictor.",
            paths=[out_dir / "exact_manifest_probes_vllm_qwen3_4b_all200_eval/table_probe_signal_analysis.csv"],
            notes=_probe_signal_notes(out_dir / "exact_manifest_probes_vllm_qwen3_4b_all200_eval/table_probe_signal_analysis.csv"),
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_9_voi_policy",
            category="implementation_task",
            requirement="Implement VOIProbePolicy and compare with baselines.",
            paths=[root / "src/routecode/probes/policies.py", out_dir / "true_probe_policy_vllm_qwen3_4b_all200/table_proberoute_policy.csv"],
            notes=_policy_notes(out_dir / "true_probe_policy_vllm_qwen3_4b_all200/table_proberoute_policy.csv"),
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_10_active_calibration",
            category="implementation_task",
            requirement="Implement active new-model calibration and compare with baselines.",
            paths=[
                out_dir / "table_active_new_model_calibration.csv",
                out_dir / "table_active_calibration_replicate_summary.csv",
            ],
            notes="Active calibration and replicate summary artifacts are present.",
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_11_ablations_sensitivity",
            category="implementation_task",
            requirement="Run ablations and sensitivity.",
            paths=[
                out_dir / "table_probe_cost_sensitivity_summary.csv",
                out_dir / "table_active_calibration_sensitivity_summary.csv",
            ],
            notes="Probe-cost and active-calibration sensitivity artifacts are present.",
        )
    )
    rows.append(
        _path_bundle_row(
            root=root,
            requirement_id="task_12_evidence_report",
            category="implementation_task",
            requirement="Write `results/phase2/PHASE2_EVIDENCE_REPORT.md`.",
            paths=[out_dir / "PHASE2_EVIDENCE_REPORT.md"],
            notes="Evidence report is present.",
        )
    )
    return rows


def _definition_of_done_rows(root: Path, out_dir: Path) -> list[AuditRow]:
    report = out_dir / "PHASE2_EVIDENCE_REPORT.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    checks = [
        (
            "dod_observability_gap",
            "Evidence report clearly states whether the observability gap persists with strong encoders.",
            ["Does the observability gap persist with strong encoders?", "Mixed, not resolved."],
            "answered_mixed",
        ),
        (
            "dod_cheap_probes",
            "Evidence report clearly states whether cheap probes close a meaningful fraction of the gap.",
            ["Do cheap probes close a meaningful fraction", "Not supported yet."],
            "answered_not_supported",
        ),
        (
            "dod_voi_policy",
            "Evidence report clearly states whether VOI beats threshold/always-probe after cost accounting.",
            ["Does VOI probing beat threshold", "Not supported."],
            "answered_not_supported",
        ),
        (
            "dod_active_calibration",
            "Evidence report clearly states whether active state-level calibration reduces new-model evaluations.",
            ["Does active route-state calibration reduce new-model evaluations?", "Not supported yet."],
            "answered_not_supported",
        ),
        (
            "dod_paper_readiness",
            "Evidence report clearly states whether ProbeRoute++ is ready for an ICML/ICLR-style paper.",
            ["ICML/ICLR-style paper", "Not yet."],
            "answered_not_ready",
        ),
    ]
    rows = []
    for requirement_id, requirement, phrases, metric in checks:
        present = all(phrase in text for phrase in phrases)
        rows.append(
            AuditRow(
                requirement_id=requirement_id,
                category="definition_of_done",
                requirement=requirement,
                status="complete" if present else "missing",
                evidence_paths=_relative(root, report),
                metric=metric if present else "answer_missing",
                notes="Definition-of-done question is answered in the evidence report." if present else "Expected answer text not found.",
            )
        )
    return rows


def _constraint_rows(root: Path, out_dir: Path) -> list[AuditRow]:
    report = out_dir / "PHASE2_EVIDENCE_REPORT.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    rows = [
        AuditRow(
            requirement_id="constraint_no_closed_api",
            category="hard_constraint",
            requirement="No GPT/Claude/Gemini API calls unless explicitly configured later.",
            status="complete" if "no GPT/Claude/Gemini API calls were made" in text else "missing",
            evidence_paths=_relative(root, report),
            metric="closed_source_calls=0" if "no GPT/Claude/Gemini API calls were made" in text else "unknown",
            notes="Evidence report records closed-source provider non-use.",
        ),
        AuditRow(
            requirement_id="constraint_provider_cost_scope",
            category="hard_constraint",
            requirement="Keep GPT/Claude/Gemini in provider-aware cost/model-pool plan.",
            status="complete"
            if all(phrase in text for phrase in ["OpenAI GPT-family", "Anthropic Claude-family", "Google Gemini-family"])
            else "missing",
            evidence_paths=_relative(root, report),
            metric="provider_scope_documented",
            notes="Closed-source provider families are documented for future provider-aware runs.",
        ),
        AuditRow(
            requirement_id="constraint_probe_cost_accounting",
            category="hard_constraint",
            requirement="Probe cost must be included in utility/cost accounting.",
            status="complete" if _policy_has_probe_cost(out_dir / "true_probe_policy_vllm_qwen3_4b_all200/table_proberoute_policy.csv") else "missing",
            evidence_paths=_relative(root, out_dir / "true_probe_policy_vllm_qwen3_4b_all200/table_proberoute_policy.csv"),
            metric="mean_probe_cost_proxy_present",
            notes="M5 policy table contains probe-cost accounting columns.",
        ),
    ]
    return rows


def _oracle_target_rows(root: Path, out_dir: Path) -> list[AuditRow]:
    path = out_dir / "oracle_gap_gate_vllm_all200/table_oracle_gap_gate.csv"
    if not _exists_nonempty(path):
        return [
            AuditRow(
                requirement_id="oracle_gap_core_policy_3pct",
                category="user_gate",
                requirement="Core Phase 2 policy should be within 3% of oracle.",
                status="missing",
                evidence_paths=_relative(root, path),
                metric="oracle_gap_table_missing",
                notes="Oracle-gap gate table missing.",
            )
        ]
    table = pd.read_csv(path)
    current = table[table["selection_basis"].astype(str).eq("current_phase2_policy_table")]
    best_current = float(current["relative_gap_to_oracle"].min()) if not current.empty else float("nan")
    candidate_names = (
        current["candidate"].astype(str)
        if "candidate" in current.columns
        else current.get("policy", pd.Series("", index=current.index)).astype(str)
    )
    true_probe_current = current[~candidate_names.str.startswith("target_rate_routecode:")]
    best_true_probe = (
        float(true_probe_current["relative_gap_to_oracle"].min()) if not true_probe_current.empty else float("nan")
    )
    exported = out_dir / "benchmark_label_policy_exact_math_vllm_all200/table_policy_summary.csv"
    export_gap = _exported_policy_gap(exported)
    selection = _routecode_selection_gate(out_dir / "routecode_exact_math_selection/table_routecode_exact_math_selection.csv")
    target_rate = _routecode_target_rate_gate(out_dir / "routecode_exact_math_selection/table_routecode_exact_math_selection.csv")
    specialized = _routecode_selection_gate(
        out_dir / "routecode_exact_math_specialized_selection/table_routecode_exact_math_selection.csv"
    )
    return [
        AuditRow(
            requirement_id="oracle_gap_core_policy_3pct",
            category="user_gate",
            requirement="Best deployable Phase 2 latent-state policy artifact should be within 3% of oracle.",
            status="complete" if best_current == best_current and best_current <= 0.03 else "not_supported",
            evidence_paths=_relative(root, path),
            metric=f"best_current_policy_relative_gap={best_current:.4f}",
            notes=(
                "Best deployable latent-state policy artifact passes the 3% gate; inspect the strict true-probe row "
                "before claiming cheap-probe or VOI success."
                if best_current == best_current and best_current <= 0.03
                else "No deployable Phase 2 latent-state policy artifact passes the 3% gate."
            ),
        ),
        AuditRow(
            requirement_id="oracle_gap_true_probe_policy_3pct",
            category="user_gate",
            requirement="Strict true-probe/VOI Phase 2 policy should be within 3% of oracle.",
            status="complete" if best_true_probe == best_true_probe and best_true_probe <= 0.03 else "not_supported",
            evidence_paths=_relative(root, path),
            metric=f"best_true_probe_policy_relative_gap={best_true_probe:.4f}",
            notes=(
                "Strict true-probe/VOI policy passes the 3% gate."
                if best_true_probe == best_true_probe and best_true_probe <= 0.03
                else "Strict true-probe/VOI policy still fails the 3% gate; target-rate RouteCode is the working policy artifact."
            ),
        ),
        AuditRow(
            requirement_id="oracle_gap_operational_fallback_3pct",
            category="user_gate",
            requirement="Have a runnable working system within 3% of oracle.",
            status="operational_fallback" if export_gap <= 0.03 else "missing",
            evidence_paths=f"{_relative(root, path)};{_relative(root, exported)}",
            metric=f"benchmark_label_relative_gap={export_gap:.4f}" if export_gap == export_gap else "benchmark_label_relative_gap=nan",
            notes="Benchmark-label route rule passes but is not the core latent-state ProbeRoute++ method.",
        ),
        AuditRow(
            requirement_id="oracle_gap_val_selected_routecode_3pct",
            category="user_gate",
            requirement="Validation-selected RouteCode candidate should be within 3% of oracle on the held-out policy slice.",
            status="complete" if selection["val_selected_gap"] <= 0.03 else "not_supported",
            evidence_paths=_relative(root, out_dir / "routecode_exact_math_selection/table_routecode_exact_math_selection.csv"),
            metric=(
                f"val_selected_policy_slice_gap={selection['val_selected_gap']:.4f};"
                f"best_policy_slice_candidate_gap={selection['best_candidate_gap']:.4f};"
                f"best_policy_slice_candidate_val_rank={selection['best_candidate_rank']:.0f}"
            ),
            notes=selection["notes"],
        ),
        AuditRow(
            requirement_id="oracle_gap_target_rate_routecode_3pct",
            category="user_gate",
            requirement="Predeclared target-rate RouteCode candidate should be within 3% of oracle on the held-out policy slice.",
            status="complete" if target_rate["target_rate_gap"] <= 0.03 else "not_supported",
            evidence_paths=_relative(root, out_dir / "routecode_exact_math_selection/table_routecode_exact_math_selection.csv"),
            metric=(
                f"target_k={target_rate['target_k']:.0f};"
                f"target_rate_policy_slice_gap={target_rate['target_rate_gap']:.4f};"
                f"target_rate_val_gap={target_rate['target_rate_val_gap']:.4f};"
                f"target_rate_val_rank={target_rate['target_rate_val_rank']:.0f}"
            ),
            notes=target_rate["notes"],
        ),
        AuditRow(
            requirement_id="oracle_gap_math_specialized_routecode_3pct",
            category="user_gate",
            requirement="Math-specialized validation-selected RouteCode candidate should be within 3% of oracle on the held-out policy slice.",
            status="complete" if specialized["val_selected_gap"] <= 0.03 else "not_supported",
            evidence_paths=_relative(
                root,
                out_dir / "routecode_exact_math_specialized_selection/table_routecode_exact_math_selection.csv",
            ),
            metric=(
                f"val_selected_policy_slice_gap={specialized['val_selected_gap']:.4f};"
                f"best_policy_slice_candidate_gap={specialized['best_candidate_gap']:.4f};"
                f"best_policy_slice_candidate_val_rank={specialized['best_candidate_rank']:.0f}"
            ),
            notes=(
                "Math-specialized RouteCode validation-selected candidate misses the 3% gate; non-selected "
                "specialized candidates are within 3% on the policy slice and should be treated as candidates only."
                if specialized["val_selected_gap"] > 0.03
                else "Math-specialized validation-selected RouteCode candidate passes the 3% policy-slice gate."
            ),
        ),
    ]


def _local_scale_row(root: Path, out_dir: Path) -> AuditRow:
    path = _first_existing(
        [
            out_dir / "local_vllm_two_model_all200_nothink/local_model_outcomes.parquet",
            out_dir / "local_vllm_qwen3_4b_all200_nothink/local_model_outcomes.parquet",
        ]
    )
    if not _exists_nonempty(path):
        return AuditRow(
            requirement_id="task_6_local_scale",
            category="implementation_task",
            requirement="Scale to 200--500 queries and 2--4 local models on exact-scored datasets.",
            status="missing",
            evidence_paths=_relative(root, path),
            metric="rows=0;models=0",
            notes="No vLLM all200 local outcome artifact found.",
        )
    table = pd.read_parquet(path)
    n_rows = int(len(table))
    n_queries = int(table["query_id"].nunique()) if "query_id" in table.columns else n_rows
    n_models = int(table["model_id"].nunique()) if "model_id" in table.columns else 0
    status = "complete" if n_queries >= 200 and n_models >= 2 else "partial"
    multi_endpoint_config = root / "configs/phase2_local_vllm_two_model_all200_nothink.yaml"
    support_note = (
        " Multi-endpoint vLLM config exists for the next run."
        if multi_endpoint_config.exists() and multi_endpoint_config.stat().st_size > 0
        else ""
    )
    notes = (
        "Reached the lower query-count target, but only one local model is present; the 2--4 local-model scope is not complete."
        + support_note
        if status == "partial"
        else "Reached query-count and local-model scale targets."
    )
    return AuditRow(
        requirement_id="task_6_local_scale",
        category="implementation_task",
        requirement="Scale to 200--500 queries and 2--4 local models on exact-scored datasets.",
        status=status,
        evidence_paths=_relative(root, path),
        metric=f"queries={n_queries};rows={n_rows};local_models={n_models}",
        notes=notes,
    )


def _local_policy_handoff_row(root: Path, out_dir: Path) -> AuditRow:
    two_model_paths = [
        out_dir / "local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_utility.csv",
        out_dir / "local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_utility.csv",
        out_dir / "true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_query_model_utility.csv",
        out_dir / "true_probe_policy_phase2_local_vllm_two_model_all200_nothink/table_proberoute_policy.csv",
    ]
    one_model_paths = [
        out_dir / "local_policy_matrices_vllm_qwen3_4b_all200/local_query_model_utility.csv",
        out_dir / "local_policy_matrices_vllm_qwen3_4b_all200/local_state_model_utility.csv",
        out_dir / "true_probe_policy_inputs_local_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv",
        out_dir / "true_probe_policy_local_vllm_qwen3_4b_all200/table_proberoute_policy.csv",
    ]
    if all(_exists_nonempty(path) for path in two_model_paths):
        return _path_bundle_row(
            root=root,
            requirement_id="task_6b_local_policy_matrix_handoff",
            category="implementation_task",
            requirement="Convert exact-scored local outcomes into local policy matrices and run the ProbeRoute++ policy handoff.",
            paths=two_model_paths,
            notes="Two-model local handoff reaches the policy table stage.",
        )
    return _path_bundle_row(
        root=root,
        requirement_id="task_6b_local_policy_matrix_handoff",
        category="implementation_task",
        requirement="Convert exact-scored local outcomes into local policy matrices and run the ProbeRoute++ policy handoff.",
        paths=one_model_paths,
        notes="One-model local handoff reaches the policy table stage; this is pipeline readiness, not a 2--4 model routing result.",
    )


def _local_vllm_pipeline_row(root: Path, out_dir: Path) -> AuditRow:
    readiness = out_dir / "local_server_readiness_phase2_local_vllm_two_model_all200_nothink/table_local_server_readiness.csv"
    memo = out_dir / "phase2_local_vllm_two_model_all200_nothink_local_vllm_policy_pipeline_memo.md"
    launch_memo = out_dir / "local_vllm_launch_attempt_memo.md"
    evidence_paths = [_relative(root, readiness), _relative(root, memo)]
    if _exists_nonempty(launch_memo):
        evidence_paths.append(_relative(root, launch_memo))
    evidence = ";".join(evidence_paths)
    if not _exists_nonempty(readiness):
        return AuditRow(
            requirement_id="task_6a_local_vllm_policy_pipeline",
            category="implementation_task",
            requirement="Run the chained local-vLLM readiness-to-policy pipeline for the 2-model config.",
            status="missing",
            evidence_paths=evidence,
            metric="readiness_rows=0",
            notes="Pipeline readiness table is missing.",
        )
    table = pd.read_csv(readiness)
    blocked = int(table["status"].astype(str).eq("blocked").sum()) if "status" in table.columns else len(table)
    if blocked:
        status = "blocked_readiness"
        notes = "Pipeline stopped before generation because at least one configured local vLLM endpoint is blocked."
        if _exists_nonempty(launch_memo):
            notes += " Direct vLLM launch attempt failed with `RuntimeError: UVA is not available` in this WSL/CUDA runtime."
    else:
        status = "complete"
        notes = "Pipeline readiness passed; downstream outputs should be inspected for completion."
    return AuditRow(
        requirement_id="task_6a_local_vllm_policy_pipeline",
        category="implementation_task",
        requirement="Run the chained local-vLLM readiness-to-policy pipeline for the 2-model config.",
        status=status,
        evidence_paths=evidence,
        metric=f"readiness_rows={len(table)};blocked={blocked}",
        notes=notes,
    )


def _path_bundle_row(
    *,
    root: Path,
    requirement_id: str,
    category: str,
    requirement: str,
    paths: list[Path],
    notes: str,
) -> AuditRow:
    missing = [path for path in paths if not _exists_nonempty(path)]
    return AuditRow(
        requirement_id=requirement_id,
        category=category,
        requirement=requirement,
        status="complete" if not missing else "missing",
        evidence_paths=";".join(_relative(root, path) for path in paths),
        metric=f"present={len(paths) - len(missing)}/{len(paths)}",
        notes=notes if not missing else "Missing: " + ";".join(_relative(root, path) for path in missing),
    )


def _local_outcome_notes(path: Path) -> str:
    if not _exists_nonempty(path):
        return "Local outcome file missing."
    table = pd.read_parquet(path)
    errors = _error_count(table)
    correct = int((table["quality"] > 0).sum()) if "quality" in table.columns else 0
    return f"rows={len(table)}; errors={errors}; exact_correct={correct}/{len(table)}."


def _probe_feature_notes(path: Path) -> str:
    if not _exists_nonempty(path):
        return "Probe feature file missing."
    table = pd.read_parquet(path)
    errors = _error_count(table)
    return f"rows={len(table)}; errors={errors}."


def _probe_signal_notes(path: Path) -> str:
    if not _exists_nonempty(path):
        return "Probe signal table missing."
    table = pd.read_csv(path)
    cols = [col for col in ["predictor", "state_prediction_accuracy"] if col in table.columns]
    if len(cols) == 2:
        best = table.sort_values("state_prediction_accuracy", ascending=False).iloc[0]
        return f"best_predictor={best['predictor']}; state_accuracy={float(best['state_prediction_accuracy']):.4f}."
    return f"rows={len(table)}."


def _policy_notes(path: Path) -> str:
    if not _exists_nonempty(path):
        return "Policy table missing."
    table = pd.read_csv(path)
    if "policy" in table.columns and "mean_net_utility" in table.columns:
        best = table.sort_values("mean_net_utility", ascending=False).iloc[0]
        return f"best_policy={best['policy']}; mean_net_utility={float(best['mean_net_utility']):.4f}."
    return f"rows={len(table)}."


def _policy_has_probe_cost(path: Path) -> bool:
    if not _exists_nonempty(path):
        return False
    table = pd.read_csv(path)
    return "mean_probe_cost_proxy" in table.columns and "mean_net_utility" in table.columns


def _exported_policy_gap(path: Path) -> float:
    if not _exists_nonempty(path):
        return float("nan")
    table = pd.read_csv(path)
    if table.empty or "relative_gap_to_oracle" not in table.columns:
        return float("nan")
    return float(table["relative_gap_to_oracle"].iloc[0])


def _routecode_selection_gate(path: Path) -> dict[str, float | str]:
    if not _exists_nonempty(path):
        return {
            "val_selected_gap": float("nan"),
            "best_candidate_gap": float("nan"),
            "best_candidate_rank": float("nan"),
            "notes": "RouteCode exact-math validation-selection table is missing.",
        }
    table = pd.read_csv(path)
    selected = table[table["selected_by_val"].astype(bool)]
    val_gap = float(selected["policy_slice_relative_gap_to_oracle"].iloc[0]) if not selected.empty else float("nan")
    candidates = table[table["policy_slice_within_threshold"].astype(bool)]
    if candidates.empty:
        return {
            "val_selected_gap": val_gap,
            "best_candidate_gap": float("nan"),
            "best_candidate_rank": float("nan"),
            "notes": "No RouteCode candidate reaches the 3% policy-slice gate.",
        }
    best = candidates.sort_values(["policy_slice_relative_gap_to_oracle", "val_selection_rank"]).iloc[0]
    notes = (
        "Validation-selected RouteCode candidate misses the 3% gate; at least one non-selected "
        "RouteCode candidate is within 3% on the policy slice and should be treated as a candidate only."
        if val_gap > 0.03
        else "Validation-selected RouteCode candidate passes the 3% policy-slice gate."
    )
    return {
        "val_selected_gap": val_gap,
        "best_candidate_gap": float(best["policy_slice_relative_gap_to_oracle"]),
        "best_candidate_rank": float(best["val_selection_rank"]),
        "notes": notes,
    }


def _routecode_target_rate_gate(path: Path) -> dict[str, float | str]:
    if not _exists_nonempty(path):
        return {
            "target_k": float("nan"),
            "target_rate_gap": float("nan"),
            "target_rate_val_gap": float("nan"),
            "target_rate_val_rank": float("nan"),
            "notes": "RouteCode target-rate selection table is missing.",
        }
    table = pd.read_csv(path)
    if "selected_by_target_rate" not in table.columns:
        return {
            "target_k": float("nan"),
            "target_rate_gap": float("nan"),
            "target_rate_val_gap": float("nan"),
            "target_rate_val_rank": float("nan"),
            "notes": "RouteCode target-rate selector was not run.",
        }
    selected = table[table["selected_by_target_rate"].astype(bool)]
    if selected.empty:
        return {
            "target_k": float("nan"),
            "target_rate_gap": float("nan"),
            "target_rate_val_gap": float("nan"),
            "target_rate_val_rank": float("nan"),
            "notes": "No target-rate RouteCode candidate was selected.",
        }
    row = selected.iloc[0]
    gap = float(row["policy_slice_relative_gap_to_oracle"])
    notes = (
        "Target-rate RouteCode candidate passes the 3% policy-slice gate. Treat this as a working "
        "engineering candidate; the stricter minimum-validation-gap selector is still reported separately."
        if gap <= 0.03
        else "Target-rate RouteCode candidate misses the 3% policy-slice gate."
    )
    return {
        "target_k": float(row["k"]),
        "target_rate_gap": gap,
        "target_rate_val_gap": float(row["val_relative_gap_to_oracle"]),
        "target_rate_val_rank": float(row["val_selection_rank"]),
        "notes": notes,
    }


def _error_count(table: pd.DataFrame) -> int:
    if "error_type" not in table.columns:
        return 0
    normalized = table["error_type"].fillna("").astype(str).str.strip().str.lower()
    no_error = {"", "none", "nan", "null"}
    return int((~normalized.isin(no_error)).sum())


def _exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if _exists_nonempty(path):
            return path
    return paths[0]


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
