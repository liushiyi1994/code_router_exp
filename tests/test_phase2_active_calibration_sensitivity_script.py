from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "62_active_calibration_sensitivity.py"
    spec = importlib.util.spec_from_file_location("phase2_active_calibration_sensitivity", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_active_calibration_sensitivity_script_summarizes_source_table(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()
    source_table = tmp_path / "sensitivity.csv"
    pd.DataFrame(
        [
            {
                "sensitivity_name": "k_8_alpha_3",
                "sensitivity_k": 8,
                "sensitivity_alpha": 3.0,
                "replicate_seed": 0,
                "method": "active_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 16,
                "mean_utility": 0.72,
            },
            {
                "sensitivity_name": "k_8_alpha_3",
                "sensitivity_k": 8,
                "sensitivity_alpha": 3.0,
                "replicate_seed": 0,
                "method": "random_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 16,
                "mean_utility": 0.70,
            },
            {
                "sensitivity_name": "k_16_alpha_3",
                "sensitivity_k": 16,
                "sensitivity_alpha": 3.0,
                "replicate_seed": 0,
                "method": "active_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 32,
                "mean_utility": 0.71,
            },
            {
                "sensitivity_name": "k_16_alpha_3",
                "sensitivity_k": 16,
                "sensitivity_alpha": 3.0,
                "replicate_seed": 0,
                "method": "random_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 2,
                "new_model_evaluations": 32,
                "mean_utility": 0.74,
            },
        ]
    ).to_csv(source_table, index=False)

    table, summary, deltas = module.run(source_table_path=str(source_table), output_dir=str(out_dir))

    assert len(table) == 4
    assert (out_dir / "table_active_calibration_sensitivity.csv").exists()
    assert (out_dir / "table_active_calibration_sensitivity_summary.csv").exists()
    assert (out_dir / "table_active_calibration_sensitivity_deltas.csv").exists()
    assert (out_dir / "m7_active_calibration_sensitivity_memo.md").exists()
    assert (out_dir / "README.md").exists()

    active_summary = summary[
        (summary["sensitivity_name"] == "k_8_alpha_3")
        & (summary["method"] == "active_route_state_calibration")
    ].iloc[0]
    assert active_summary["mean_utility_mean"] == 0.72

    random_deltas = deltas[deltas["baseline"] == "random_route_state_calibration"].set_index("sensitivity_name")
    assert random_deltas.loc["k_8_alpha_3", "active_minus_baseline_mean"] == 0.02
    assert random_deltas.loc["k_16_alpha_3", "active_minus_baseline_mean"] == -0.03

    memo = (out_dir / "m7_active_calibration_sensitivity_memo.md").read_text(encoding="utf-8")
    assert "Phase 2 Active Calibration Sensitivity" in memo
    assert "active_minus_baseline_mean" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Phase 2 Active Calibration Sensitivity" in readme
