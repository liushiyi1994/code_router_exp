from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "55_active_new_model_calibration.py"
    spec = importlib.util.spec_from_file_location("phase2_active_new_model_calibration", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_active_new_model_calibration_script_writes_table_figure_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()
    source_table = tmp_path / "active_table.csv"
    pd.DataFrame(
        [
            {
                "method": "uniform_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 1,
                "new_model_evaluations": 4,
                "mean_utility": 0.62,
                "utility_ci_low": 0.58,
                "utility_ci_high": 0.66,
                "recovered_gap_vs_oracle": 0.20,
            },
            {
                "method": "active_route_state_calibration",
                "new_model_id": "new",
                "examples_per_label": 1,
                "new_model_evaluations": 4,
                "mean_utility": 0.68,
                "utility_ci_low": 0.64,
                "utility_ci_high": 0.72,
                "recovered_gap_vs_oracle": 0.35,
            },
        ]
    ).to_csv(source_table, index=False)

    table = module.run(source_table_path=str(source_table), output_dir=str(out_dir))

    assert len(table) == 2
    assert (out_dir / "table_active_new_model_calibration.csv").exists()
    assert (out_dir / "fig_new_model_calibration_curve.pdf").exists()
    assert (out_dir / "m6_active_new_model_calibration_memo.md").exists()
    assert (out_dir / "README.md").exists()
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Phase 2 Active New-Model Calibration" in readme
    memo = (out_dir / "m6_active_new_model_calibration_memo.md").read_text(encoding="utf-8")
    assert "active_route_state_calibration" in memo
    assert "utility_ci_low" in memo
    assert "utility_ci_high" in memo
