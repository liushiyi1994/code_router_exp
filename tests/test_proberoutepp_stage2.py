from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from routecode.proberoutepp import (
    ProbeRoutePPConfig,
    build_proberoutepp_artifacts,
    write_proberoutepp_outputs,
)


def _toy_scored_outputs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    queries = [
        ("train_math_easy", "Compute 2 + 3.", "math", "train", {"cheap_local": 1.0, "strong_local": 0.8, "gpt-5.5": 1.0}),
        ("train_math_hard", "Solve a hard olympiad equation.", "math", "train", {"cheap_local": 0.1, "strong_local": 0.5, "gpt-5.5": 1.0}),
        ("train_code", "Write a Python function.", "code", "train", {"cheap_local": 0.2, "strong_local": 1.0, "gpt-5.5": 0.8}),
        ("train_general", "What is photosynthesis?", "knowledge", "train", {"cheap_local": 0.8, "strong_local": 0.8, "gpt-5.5": 0.9}),
        ("test_math_hard", "Solve a hard contest equation.", "math", "test", {"cheap_local": 0.0, "strong_local": 0.4, "gpt-5.5": 1.0}),
        ("test_code", "Write Python code for sorting.", "code", "test", {"cheap_local": 0.2, "strong_local": 1.0, "gpt-5.5": 0.8}),
    ]
    for query_id, query_text, domain, split, qualities in queries:
        for model_id, quality in qualities.items():
            is_frontier = model_id == "gpt-5.5"
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "benchmark": domain,
                    "domain": domain,
                    "split": split,
                    "model_id": model_id,
                    "provider": "openai" if is_frontier else "local",
                    "is_local": not is_frontier,
                    "is_frontier": is_frontier,
                    "quality_score": quality,
                    "cost_total_usd": 0.01 if is_frontier else 0.0,
                    "latency_s": 3.0 if is_frontier else 0.5,
                    "status": "success",
                }
            )
    return pd.DataFrame(rows)


def test_proberoutepp_builds_state_table_and_state_mediated_decisions() -> None:
    artifacts = build_proberoutepp_artifacts(
        _toy_scored_outputs(),
        ProbeRoutePPConfig(k=2, alpha=0.5, lambda_cost=0.2, lambda_latency=0.0, probe_knn_k=2),
    )

    state_table = artifacts.state_model_utility_table
    decisions = artifacts.routing_decisions
    main_eval = artifacts.main_eval

    assert {"state_label", "model_id", "mean_utility", "mean_quality", "n_train_examples"}.issubset(
        state_table.columns
    )
    assert set(decisions["method"]) == {
        "proberoutepp_no_probe",
        "proberoutepp_threshold_probe",
        "proberoutepp_voi_probe",
    }
    no_probe = decisions[decisions["method"] == "proberoutepp_no_probe"]
    assert no_probe["probe_used"].eq(False).all()
    assert decisions["selected_model"].notna().all()
    assert decisions["selected_state"].str.startswith("z").all()
    first_distribution = json.loads(decisions.iloc[0]["state_distribution_before_probe_json"])
    assert sum(first_distribution.values()) == pytest.approx(1.0)
    assert {"best_local", "all_gpt_frontier", "cost_aware_oracle", "proberoutepp_voi_probe"}.issubset(
        set(main_eval["method"])
    )


def test_proberoutepp_writes_required_stage2_outputs(tmp_path: Path) -> None:
    artifacts = build_proberoutepp_artifacts(
        _toy_scored_outputs(),
        ProbeRoutePPConfig(k=2, alpha=0.5, lambda_cost=0.2, lambda_latency=0.0),
    )

    paths = write_proberoutepp_outputs(artifacts, tmp_path)

    assert paths["state_model_utility_table"].exists()
    assert paths["routing_decisions"].exists()
    assert paths["table_main_eval"].exists()
    assert paths["run_report"].exists()
    loaded = pd.read_parquet(paths["routing_decisions"])
    assert "decision_reason" in loaded.columns
