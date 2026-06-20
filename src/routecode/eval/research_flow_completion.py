from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from routecode.eval.phase_e_coverage import missing_required_coverage


@dataclass(frozen=True)
class PhaseRequirement:
    phase_id: str
    phase: str
    required_paths: tuple[str, ...]
    completion_rule: str


PHASE_REQUIREMENTS = [
    PhaseRequirement(
        "phase_a_synthetic_sanity",
        "Phase A - setup and synthetic sanity",
        (
            "results/demo/table_routability.csv",
            "results/demo/table_recovered_gap.csv",
            "results/demo/table_rate_distortion.csv",
            "results/demo/fig_compression_ladder.pdf",
            "results/demo/fig_rate_distortion.pdf",
        ),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_b_real_data_pilot",
        "Phase B - real-data pilot study",
        (
            "results/llmrouterbench_pilot/table_routability.csv",
            "results/llmrouterbench_pilot/table_recovered_gap.csv",
            "results/llmrouterbench_pilot/table_rate_distortion.csv",
            "results/llmrouterbench_pilot/table_residual_concentration.csv",
            "results/llmrouterbench_pilot/table_split_sensitivity.csv",
        ),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_c_observation_synthesis",
        "Phase C - observation synthesis",
        ("results/llmrouterbench_pilot/phase_c_observation_memo.md",),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_d2_predictability_constrained",
        "Phase D2 - predictability-constrained RouteCode",
        ("results/llmrouterbench_pilot/table_predictability_constrained.csv",),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_d3_code_cards",
        "Phase D3 - explainable route-label cards",
        ("results/llmrouterbench_pilot/code_cards.md",),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_d4_new_model_calibration",
        "Phase D4 - new-model calibration",
        ("results/llmrouterbench_pilot/table_new_model_integration.csv",),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_d5_adaptive_refinement",
        "Phase D5 - adaptive refinement",
        ("results/llmrouterbench_pilot/table_residual_risk.csv",),
        "deferred_if_gate_weak",
    ),
    PhaseRequirement(
        "phase_e_external_methods",
        "Phase E - method evaluation and external baselines",
        (
            "results/llmrouterbench_pilot/table_external_command_readiness.csv",
            "results/llmrouterbench_pilot/table_phase_e_baseline_coverage.csv",
        ),
        "complete_if_phase_e_required_coverage_present",
    ),
    PhaseRequirement(
        "phase_f_ablation",
        "Phase F - ablation study",
        ("results/llmrouterbench_pilot/table_ablation_summary.csv",),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_g_sensitivity",
        "Phase G - sensitivity analysis",
        ("results/llmrouterbench_pilot/table_sensitivity_summary.csv",),
        "all_required_artifacts_present",
    ),
    PhaseRequirement(
        "phase_h_final_claims",
        "Phase H - final paper claims",
        (
            "results/table_claim_status_global.csv",
            "results/table_paper_evidence_summary.csv",
            "paper_notes.md",
        ),
        "complete_if_claims_documented_conservatively",
    ),
]


def audit_research_flow_completion(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    rows = []
    for requirement in PHASE_REQUIREMENTS:
        present, missing = _path_status(base, requirement.required_paths)
        status, notes = _phase_status(base, requirement, present, missing)
        rows.append(
            {
                "phase_id": requirement.phase_id,
                "phase": requirement.phase,
                "status": status,
                "required_paths_present": len(present),
                "required_paths_total": len(requirement.required_paths),
                "missing_paths": ";".join(missing),
                "completion_rule": requirement.completion_rule,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _path_status(root: Path, paths: tuple[str, ...]) -> tuple[list[str], list[str]]:
    present = []
    missing = []
    for relative in paths:
        path = root / relative
        if path.exists() and path.stat().st_size > 0:
            present.append(relative)
        else:
            missing.append(relative)
    return present, missing


def _phase_status(
    root: Path,
    requirement: PhaseRequirement,
    present: list[str],
    missing: list[str],
) -> tuple[str, str]:
    if requirement.completion_rule == "deferred_if_gate_weak":
        if missing and not (root / "results/llmrouterbench_pilot/phase_d5_adaptive_refinement_gate_memo.md").exists():
            return "missing_evidence", "Adaptive-refinement gate memo is missing."
        return "deferred", "Adaptive refinement is deferred unless a stronger deployable residual-risk signal appears."
    if missing:
        return "missing_evidence", "Missing required artifacts."
    if requirement.completion_rule == "complete_if_phase_e_required_coverage_present":
        return _phase_e_status(root, requirement)
    if requirement.completion_rule == "complete_if_claims_documented_conservatively":
        return _phase_h_status(root)
    return "complete", "Required artifacts are present."


def _blocked_readiness_rows(readiness: pd.DataFrame) -> list[str]:
    blocked = []
    for _, row in readiness.iterrows():
        routecode_metric = _as_bool(row.get("routecode_metric_compatible", False))
        runnable = _as_bool(row.get("runnable_now", False))
        if not runnable and not routecode_metric:
            blocked.append(str(row.get("check_id", "")))
    return blocked


def _external_blocker_resolution_note(root: Path) -> str:
    path = root / "results/table_external_blocker_resolution.csv"
    if not path.exists() or path.stat().st_size == 0:
        return ""
    table = pd.read_csv(path)
    if table.empty:
        return "Blocker resolution: no blocked rows in results/table_external_blocker_resolution.csv."
    checkpoints = table.get("missing_checkpoints", pd.Series("", index=table.index)).fillna("").astype(str)
    modules = table.get("missing_modules", pd.Series("", index=table.index)).fillna("").astype(str)
    assets = table.get("missing_assets", pd.Series("", index=table.index)).fillna("").astype(str)
    service = table.get("service_requirements", pd.Series("", index=table.index)).fillna("").astype(str)
    checkpoint_gated = int(checkpoints.ne("").sum())
    module_only = int((modules.ne("") & checkpoints.eq("") & assets.eq("") & service.eq("")).sum())
    return (
        f"Blocker resolution: {checkpoint_gated} checkpoint-gated, {module_only} module-only; "
        "see results/table_external_blocker_resolution.csv."
    )


def _phase_e_status(root: Path, requirement: PhaseRequirement) -> tuple[str, str]:
    readiness = pd.read_csv(root / requirement.required_paths[0])
    coverage = pd.read_csv(root / requirement.required_paths[1])
    missing = missing_required_coverage(coverage)
    if missing:
        return "missing_evidence", "Missing required Phase E baseline coverage: " + ", ".join(missing)
    optional = _optional_external_blockers_from_coverage(coverage)
    blocker_rows = _blocked_readiness_rows(readiness)
    unexpected = [item for item in blocker_rows if item not in set(optional)]
    if unexpected:
        note = "Unexpected blocked external rows not marked optional: " + ", ".join(unexpected)
        resolution_note = _external_blocker_resolution_note(root)
        if resolution_note:
            note += ". " + resolution_note
        return "blocked", note
    note = "Required Phase E baseline coverage is complete."
    if optional:
        note += " Optional checkpoint-gated rows documented: " + ", ".join(optional) + "."
    return "complete", note


def _optional_external_blockers_from_coverage(coverage: pd.DataFrame) -> list[str]:
    if coverage.empty or "requirement_id" not in coverage.columns or "evidence" not in coverage.columns:
        return []
    rows = coverage[coverage["requirement_id"].astype(str) == "optional_extra_external_blockers"]
    if rows.empty:
        return []
    evidence = str(rows.iloc[0].get("evidence", ""))
    return [item.strip() for item in evidence.split(",") if item.strip() and item.strip().lower() != "nan"]


def _phase_h_status(root: Path) -> tuple[str, str]:
    claims = pd.read_csv(root / "results/table_claim_status_global.csv")
    evidence = pd.read_csv(root / "results/table_paper_evidence_summary.csv")
    direction = evidence[
        (evidence.get("section", pd.Series("", index=evidence.index)).astype(str) == "paper_direction")
        & (evidence.get("item", pd.Series("", index=evidence.index)).astype(str) == "recommended_framing")
    ]
    if direction.empty:
        return "missing_evidence", "Paper evidence summary lacks a recommended framing row."
    framing = str(direction.iloc[0].get("status", ""))
    problematic = claims[claims["global_status"].isin(["not_supported", "mixed_evidence", "missing_evidence"])]
    if problematic.empty:
        return "complete", f"All global claim gates are supported; recommended_framing={framing}."
    pairs = [f"{row['claim_id']}={row['global_status']}" for _, row in problematic.iterrows()]
    return (
        "complete",
        "Conservative final claim posture documented: "
        f"recommended_framing={framing}; unsupported/mixed claims not claimed: " + ", ".join(pairs),
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
