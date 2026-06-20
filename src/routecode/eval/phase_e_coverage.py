from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


BASELINE_REQUIREMENTS = [
    ("random", "Random routing baseline", "required", [("recovered_gap", "random")]),
    ("cheapest", "Cheapest-model baseline", "required", [("recovered_gap", "cheapest"), ("routability", "cheapest")]),
    ("best_single", "Best single model", "required", [("recovered_gap", "best_single"), ("routability", "best_single")]),
    ("dataset_oracle", "Dataset oracle", "required", [("recovered_gap", "dataset_oracle")]),
    ("query_oracle", "Query oracle", "required", [("recovered_gap", "query_oracle"), ("routability", "query_oracle")]),
    (
        "dataset_label_lookup",
        "Dataset-label lookup",
        "required",
        [("recovered_gap", "dataset_label_lookup"), ("routability", "dataset_label_lookup")],
    ),
    ("predicted_topic_lookup", "Predicted-topic lookup", "required", [("recovered_gap", "predicted_topic_lookup")]),
    (
        "embedding_cluster_lookup",
        "Embedding-cluster lookup",
        "required",
        [("recovered_gap", "embedding_cluster_lookup")],
    ),
    ("knn", "kNN router", "required", [("recovered_gap", "kNN")]),
    ("logistic_mlp_svm", "MLP/SVM/simple learned routers", "required", [("recovered_gap", "logistic_embedding_router"), ("recovered_gap", "mlp_embedding_router"), ("recovered_gap", "svm_embedding_router")]),
    (
        "route_llm_if_easy",
        "RouteLLM baseline when locally runnable",
        "conditional",
        [("routellm_mf", "routellm_mf_split_aligned")],
    ),
    (
        "llmrouter_if_available",
        "LLMRouter baselines when locally available",
        "conditional",
        [("llmrouter_library", "llmrouter_library_knn"), ("llmrouter_library", "llmrouter_library_svm")],
    ),
    (
        "graphrouter_if_available",
        "GraphRouter baseline when locally available",
        "conditional",
        [("graphrouter", "graphrouter_split_aligned")],
    ),
    (
        "avengerspro_if_included",
        "Avengers-Pro baseline when included in LLMRouterBench",
        "conditional",
        [("avengerspro", "avengerspro_upstream_simple_cluster"), ("avengerspro", "avengerspro_cli_simple_cluster")],
    ),
    (
        "cost_quality_metrics",
        "Cost-quality frontier metrics",
        "required",
        [("cost_quality", "")],
    ),
]


def audit_phase_e_baseline_coverage(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for requirement_id, requirement, requirement_type, evidence_specs in BASELINE_REQUIREMENTS:
        evidence = _matched_evidence(tables, evidence_specs)
        rows.append(
            {
                "requirement_id": requirement_id,
                "requirement": requirement,
                "requirement_type": requirement_type,
                "status": "present" if evidence else "missing",
                "evidence": ",".join(evidence),
                "notes": _notes(requirement_type, evidence),
            }
        )
    optional_blockers = _optional_blockers(tables.get("readiness", pd.DataFrame()))
    rows.append(
        {
            "requirement_id": "optional_extra_external_blockers",
            "requirement": "Extra checkpoint-heavy external baselines beyond Research Flow required list",
            "requirement_type": "optional",
            "status": "present",
            "evidence": ",".join(optional_blockers),
            "notes": (
                "Optional external blockers documented, not required for Phase E completion."
                if optional_blockers
                else "No optional external command blockers in readiness table."
            ),
        }
    )
    return pd.DataFrame(rows)


def required_coverage_complete(coverage: pd.DataFrame) -> bool:
    if coverage.empty:
        return False
    required = coverage[coverage["requirement_type"].isin(["required", "conditional"])]
    return bool(not required.empty and required["status"].eq("present").all())


def missing_required_coverage(coverage: pd.DataFrame) -> list[str]:
    if coverage.empty:
        return []
    missing = coverage[
        coverage["requirement_type"].isin(["required", "conditional"])
        & coverage["status"].ne("present")
    ]
    return [str(item) for item in missing["requirement_id"]]


def _matched_evidence(tables: Mapping[str, pd.DataFrame], evidence_specs: list[tuple[str, str]]) -> list[str]:
    evidence = []
    for table_name, method_pattern in evidence_specs:
        table = tables.get(table_name)
        if table is None or table.empty:
            continue
        if method_pattern == "":
            evidence.append(table_name)
            continue
        if "method" not in table.columns:
            continue
        methods = table["method"].astype(str)
        if methods.eq(method_pattern).any() or methods.str.contains(method_pattern, regex=False).any():
            evidence.append(method_pattern)
    return evidence


def _optional_blockers(readiness: pd.DataFrame) -> list[str]:
    if readiness.empty or "check_id" not in readiness.columns:
        return []
    optional_ids = {
        "routellm_bert_cli",
        "best_route_train_cli",
        "routerdc_train_cli",
        "modelsat_train_cli",
    }
    rows = readiness[
        readiness["check_id"].astype(str).isin(optional_ids)
        & readiness.get("status", pd.Series("", index=readiness.index)).astype(str).eq("blocked")
    ]
    return [str(item) for item in rows["check_id"]]


def _notes(requirement_type: str, evidence: list[str]) -> str:
    if evidence:
        return "Evidence present: " + ",".join(evidence) + "."
    if requirement_type == "conditional":
        return "Conditional baseline is missing; inspect whether the local upstream source is available."
    return "Required baseline evidence is missing."
