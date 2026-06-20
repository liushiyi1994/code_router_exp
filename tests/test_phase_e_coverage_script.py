from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "44_phase_e_baseline_coverage.py"
    spec = importlib.util.spec_from_file_location("phase_e_baseline_coverage", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase_e_baseline_coverage_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "README.md").write_text("# Run\n", encoding="utf-8")
    pd.DataFrame(
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
    ).to_csv(result_dir / "table_recovered_gap.csv", index=False)
    pd.DataFrame({"method": ["routellm_mf_split_aligned_t0.5"]}).to_csv(
        result_dir / "table_routellm_mf_split_aligned.csv", index=False
    )
    pd.DataFrame({"method": ["llmrouter_library_knn", "llmrouter_library_svm"]}).to_csv(
        result_dir / "table_llmrouter_library_adapters.csv", index=False
    )
    pd.DataFrame({"method": ["graphrouter_split_aligned_gnn"]}).to_csv(
        result_dir / "table_graphrouter_split_aligned.csv", index=False
    )
    pd.DataFrame({"method": ["avengerspro_upstream_simple_cluster_postprocessed"]}).to_csv(
        result_dir / "table_avengerspro_upstream_metric.csv", index=False
    )
    pd.DataFrame({"method": ["routecode_predicted_labels"], "quality_at_fixed_cost": [0.7]}).to_csv(
        result_dir / "table_cost_quality_frontier.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "check_id": "routellm_bert_cli",
                "status": "blocked",
                "runnable_now": False,
                "routecode_metric_compatible": False,
                "blocking_reasons": "missing_bert_checkpoint",
            }
        ]
    ).to_csv(result_dir / "table_external_command_readiness.csv", index=False)

    module.run(result_dir)

    table_path = result_dir / "table_phase_e_baseline_coverage.csv"
    memo_path = result_dir / "phase_e_baseline_coverage_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert table["status"].eq("present").all()
    assert "optional_extra_external_blockers" in set(table["requirement_id"])

    memo = memo_path.read_text(encoding="utf-8")
    assert "Phase E Baseline Coverage Memo" in memo
    assert "Required/conditional baseline coverage complete: `True`" in memo
    readme = (result_dir / "README.md").read_text(encoding="utf-8")
    assert "## Phase E Baseline Coverage" in readme
    assert "table_phase_e_baseline_coverage.csv" in readme
