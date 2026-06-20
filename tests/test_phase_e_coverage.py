from __future__ import annotations

import pandas as pd

from routecode.eval.phase_e_coverage import audit_phase_e_baseline_coverage


def test_phase_e_coverage_marks_required_baselines_present_and_optional_blockers_documented():
    tables = {
        "recovered_gap": pd.DataFrame(
            {
                "method": [
                    "random",
                    "cheapest",
                    "best_single",
                    "dataset_oracle",
                    "query_oracle",
                    "dataset_label_lookup",
                    "predicted_topic_lookup",
                    "embedding_cluster_lookup",
                    "kNN",
                    "logistic_embedding_router",
                    "mlp_embedding_router",
                    "svm_embedding_router",
                ]
            }
        ),
        "routellm_mf": pd.DataFrame({"method": ["routellm_mf_split_aligned_t0.5"]}),
        "llmrouter_library": pd.DataFrame({"method": ["llmrouter_library_knn", "llmrouter_library_svm"]}),
        "graphrouter": pd.DataFrame({"method": ["graphrouter_split_aligned_gnn"]}),
        "avengerspro": pd.DataFrame({"method": ["avengerspro_upstream_simple_cluster_postprocessed"]}),
        "cost_quality": pd.DataFrame({"method": ["routecode_predicted_labels"], "quality_at_fixed_cost": [0.7]}),
        "readiness": pd.DataFrame(
            [
                {
                    "check_id": "best_route_train_cli",
                    "status": "blocked",
                    "runnable_now": False,
                    "routecode_metric_compatible": False,
                    "blocking_reasons": "missing_best_route_local_model_checkpoint",
                }
            ]
        ),
    }

    coverage = audit_phase_e_baseline_coverage(tables)

    assert set(coverage["status"]) == {"present"}
    optional_row = coverage.set_index("requirement_id").loc["optional_extra_external_blockers"]
    assert optional_row["evidence"] == "best_route_train_cli"
    assert "documented, not required" in optional_row["notes"]


def test_phase_e_coverage_reports_missing_required_baseline():
    coverage = audit_phase_e_baseline_coverage({"recovered_gap": pd.DataFrame({"method": ["best_single"]})})
    rows = coverage.set_index("requirement_id")

    assert rows.loc["random", "status"] == "missing"
    assert rows.loc["best_single", "status"] == "present"
    assert rows.loc["route_llm_if_easy", "status"] == "missing"
