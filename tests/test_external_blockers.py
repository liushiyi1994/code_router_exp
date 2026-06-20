from __future__ import annotations

import pandas as pd

from routecode.eval.external_blockers import summarize_external_blockers


def test_summarize_external_blockers_groups_repeated_blockers_across_runs():
    pilot = pd.DataFrame(
        [
            {
                "check_id": "best_route_train_cli",
                "status": "blocked",
                "runnable_now": False,
                "blocking_reasons": "missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender",
            },
            {
                "check_id": "graphrouter_cli",
                "status": "executed",
                "runnable_now": True,
                "blocking_reasons": "",
            },
        ]
    )
    broad = pd.DataFrame(
        [
            {
                "check_id": "best_route_train_cli",
                "status": "blocked",
                "runnable_now": False,
                "blocking_reasons": "missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender",
            }
        ]
    )

    summary = summarize_external_blockers({"pilot": pilot, "broad20": broad})
    row = summary.set_index("check_id").loc["best_route_train_cli"]

    assert row["blocked_runs"] == "broad20,pilot"
    assert row["blocked_run_count"] == 2
    assert row["missing_modules"] == "llm_blender"
    assert row["missing_checkpoints"] == "missing_best_route_local_model_checkpoint"
    assert row["missing_assets"] == ""
    assert not bool(row["can_progress_without_download"])
    assert "Provision local checkpoint" in row["next_action"]
    assert "Install Python module" in row["next_action"]
    assert "graphrouter_cli" not in set(summary["check_id"])


def test_summarize_external_blockers_distinguishes_assets_and_service_requirements():
    table = pd.DataFrame(
        [
            {
                "check_id": "routellm_mf_eval_cli",
                "status": "blocked",
                "runnable_now": False,
                "blocking_reasons": (
                    "missing_pairwise_test_json;"
                    "embedding_config_requires_env:EMBEDDING_API_KEY;"
                    "requires_embedding_service"
                ),
            },
            {
                "check_id": "module_only_cli",
                "status": "blocked",
                "runnable_now": False,
                "blocking_reasons": "missing_python_modules:deepspeed,nltk",
            },
        ]
    )

    summary = summarize_external_blockers({"run": table}).set_index("check_id")

    service_row = summary.loc["routellm_mf_eval_cli"]
    assert service_row["missing_assets"] == "missing_pairwise_test_json"
    assert service_row["service_requirements"] == "EMBEDDING_API_KEY,requires_embedding_service"
    assert not bool(service_row["can_progress_without_download"])
    assert "Prepare missing local asset" in service_row["next_action"]
    assert "Use cached/local embeddings" in service_row["next_action"]

    module_row = summary.loc["module_only_cli"]
    assert module_row["missing_modules"] == "deepspeed,nltk"
    assert module_row["missing_checkpoints"] == ""
    assert module_row["service_requirements"] == ""
    assert bool(module_row["can_progress_without_download"])
    assert module_row["next_action"] == "Install Python modules: deepspeed,nltk."
