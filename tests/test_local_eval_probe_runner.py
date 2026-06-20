from __future__ import annotations

from routecode.local_eval.probe_runner import (
    DryRunProbeClient,
    LocalProbeTask,
    run_aligned_probe_matrix,
)
from routecode.probes.probe_features import PROBE_FEATURE_COLUMNS


def test_aligned_probe_matrix_logs_probe_schema_confidence_and_raw_outputs():
    tasks = [
        LocalProbeTask(
            query_id="bench_q0",
            query_text="Explain why one model might be better for this query.",
            dataset="demo",
            domain="reasoning",
        )
    ]

    features, raw_logs, errors = run_aligned_probe_matrix(
        tasks=tasks,
        model_ids=["dry_probe"],
        client=DryRunProbeClient(),
        generation_params={"temperature": 0.0, "max_tokens": 32},
        model_revision="dry-run",
    )

    assert list(features.columns) == PROBE_FEATURE_COLUMNS
    assert len(features) == 1
    assert not errors
    assert len(raw_logs) == 1
    row = features.iloc[0]
    assert row["query_id"] == "bench_q0"
    assert row["probe_type"] == "aligned_local_confidence_probe"
    assert row["probe_model_id"] == "dry_probe"
    assert 0.0 <= row["self_confidence"] <= 1.0
    assert row["input_tokens"] > 0
    assert row["output_tokens"] > 0
    assert "Confidence:" in row["raw_probe_output"]
    assert raw_logs[0]["prompt"]
