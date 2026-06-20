from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "07_new_model_calibration.py"
    spec = importlib.util.spec_from_file_location("new_model_calibration_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_new_model_calibration_readme_reports_compact_summary(tmp_path):
    module = _load_script()
    (tmp_path / "README.md").write_text("# Demo\n\n## Next Section\n\nkeep me\n", encoding="utf-8")
    rows = []
    for model in ["m0", "m1", "m2"]:
        for method in ["routecode_label_calibration", "direct_retraining_budgeted_logistic"]:
            for r in [1, 2]:
                rows.append(
                    {
                        "method": method,
                        "new_model_id": model,
                        "examples_per_label": r,
                        "calibration_query_count": 10 * r,
                        "mean_utility": 0.5 + 0.01 * r,
                        "recovered_gap_vs_oracle": 0.1 * r,
                    }
                )
    table = pd.DataFrame(rows)

    module.append_readme(tmp_path, "configs/example.yaml", table)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    calibration_section = readme.split("## New-Model Calibration", maxsplit=1)[1].split("## Next Section", maxsplit=1)[0]
    assert "Mean across held-out models" in calibration_section
    assert "Best budgeted rows" in calibration_section
    assert calibration_section.count("| routecode_label_calibration |") <= 3
    assert "## Next Section" in readme
