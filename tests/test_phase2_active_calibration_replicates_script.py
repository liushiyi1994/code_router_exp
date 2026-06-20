from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "61_active_calibration_replicates.py"
    spec = importlib.util.spec_from_file_location("phase2_active_calibration_replicates", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_active_calibration_replicates_script_summarizes_source_table(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()
    source_table = tmp_path / "replicates.csv"
    pd.DataFrame(
        [
            {
                "replicate_seed": 0,
                "method": "active_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.72,
                "recovered_gap_vs_oracle": 0.30,
            },
            {
                "replicate_seed": 1,
                "method": "active_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.76,
                "recovered_gap_vs_oracle": 0.42,
            },
            {
                "replicate_seed": 0,
                "method": "random_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.71,
                "recovered_gap_vs_oracle": 0.27,
            },
            {
                "replicate_seed": 1,
                "method": "random_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.75,
                "recovered_gap_vs_oracle": 0.39,
            },
            {
                "replicate_seed": 0,
                "method": "uniform_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.70,
                "recovered_gap_vs_oracle": 0.24,
            },
            {
                "replicate_seed": 0,
                "method": "dataset_stratified_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.69,
                "recovered_gap_vs_oracle": 0.21,
            },
            {
                "replicate_seed": 1,
                "method": "dataset_stratified_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.73,
                "recovered_gap_vs_oracle": 0.33,
            },
            {
                "replicate_seed": 0,
                "method": "embedding_cluster_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.68,
                "recovered_gap_vs_oracle": 0.18,
            },
            {
                "replicate_seed": 1,
                "method": "embedding_cluster_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.72,
                "recovered_gap_vs_oracle": 0.30,
            },
            {
                "replicate_seed": 1,
                "method": "uniform_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 8,
                "mean_utility": 0.74,
                "recovered_gap_vs_oracle": 0.36,
            },
        ]
    ).to_csv(source_table, index=False)

    replicate_table, summary, uniform_deltas, random_deltas, dataset_deltas, embedding_deltas = module.run(
        source_table_path=str(source_table),
        output_dir=str(out_dir),
    )

    assert len(replicate_table) == 10
    assert (out_dir / "table_active_calibration_replicates.csv").exists()
    assert (out_dir / "table_active_calibration_replicate_summary.csv").exists()
    assert (out_dir / "table_active_calibration_active_vs_uniform_deltas.csv").exists()
    assert (out_dir / "table_active_calibration_active_vs_random_deltas.csv").exists()
    assert (out_dir / "table_active_calibration_active_vs_dataset_deltas.csv").exists()
    assert (out_dir / "table_active_calibration_active_vs_embedding_deltas.csv").exists()
    assert (out_dir / "m6_active_calibration_replicates_memo.md").exists()
    assert (out_dir / "README.md").exists()

    active_summary = summary.set_index("method").loc["active_route_state_calibration"]
    assert active_summary["replicates"] == 2
    assert active_summary["mean_utility_mean"] == 0.74
    assert active_summary["mean_utility_std"] == 0.028284
    assert active_summary["recovered_gap_vs_oracle_mean"] == 0.36

    assert len(uniform_deltas) == 2
    assert set(uniform_deltas["active_minus_uniform_mean_utility"].round(6)) == {0.02}
    assert len(random_deltas) == 2
    assert set(random_deltas["active_minus_random_mean_utility"].round(6)) == {0.01}
    assert len(dataset_deltas) == 2
    assert set(dataset_deltas["active_minus_dataset_mean_utility"].round(6)) == {0.03}
    assert len(embedding_deltas) == 2
    assert set(embedding_deltas["active_minus_embedding_mean_utility"].round(6)) == {0.04}
    memo = (out_dir / "m6_active_calibration_replicates_memo.md").read_text(encoding="utf-8")
    assert "active_minus_uniform_mean_utility_mean" in memo
    assert "active_minus_random_mean_utility_mean" in memo
    assert "active_minus_dataset_mean_utility_mean" in memo
    assert "active_minus_embedding_mean_utility_mean" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Phase 2 Active Calibration Replicates" in readme
