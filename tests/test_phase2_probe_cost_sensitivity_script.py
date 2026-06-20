from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "63_probe_cost_sensitivity.py"
    spec = importlib.util.spec_from_file_location("phase2_probe_cost_sensitivity", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_cost_sensitivity_script_sweeps_cost_multiplier(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()
    before = out_dir / "before.csv"
    after = out_dir / "after.csv"
    state_utility = out_dir / "state_utility.csv"
    query_utility = out_dir / "query_utility.csv"
    probe_cost = out_dir / "probe_cost.csv"
    predicted_gain = out_dir / "predicted_gain.csv"

    pd.DataFrame({"query_id": ["q0", "q1"], "z0": [0.9, 0.6], "z1": [0.1, 0.4]}).to_csv(before, index=False)
    pd.DataFrame({"query_id": ["q0", "q1"], "z0": [0.9, 0.1], "z1": [0.1, 0.9]}).to_csv(after, index=False)
    pd.DataFrame({"state_label": ["z0", "z1"], "cheap": [0.8, 0.1], "strong": [0.2, 0.9]}).to_csv(
        state_utility,
        index=False,
    )
    pd.DataFrame({"query_id": ["q0", "q1"], "cheap": [0.8, 0.1], "strong": [0.2, 0.9]}).to_csv(
        query_utility,
        index=False,
    )
    pd.DataFrame({"query_id": ["q0", "q1"], "probe_cost": [0.01, 0.01]}).to_csv(probe_cost, index=False)
    pd.DataFrame({"query_id": ["q0", "q1"], "predicted_gain": [0.0, 0.4]}).to_csv(predicted_gain, index=False)

    table, summary = module.run(
        output_dir=str(out_dir),
        before_beliefs_path=str(before),
        after_beliefs_path=str(after),
        state_model_utility_path=str(state_utility),
        query_model_utility_path=str(query_utility),
        probe_cost_path=str(probe_cost),
        predicted_gain_path=str(predicted_gain),
        probe_cost_multipliers=[1.0, 50.0],
    )

    assert (out_dir / "table_probe_cost_sensitivity.csv").exists()
    assert (out_dir / "table_probe_cost_sensitivity_summary.csv").exists()
    assert (out_dir / "fig_probe_cost_sensitivity.pdf").exists()
    assert (out_dir / "m7_probe_cost_sensitivity_memo.md").exists()
    assert (out_dir / "README.md").exists()
    assert set(table["status"]) == {"executed"}
    assert set(summary["probe_cost_multiplier"]) == {1.0, 50.0}

    voi = table[table["policy"] == "voi_probe"].set_index("probe_cost_multiplier")
    assert voi.loc[1.0, "fraction_probed"] == 0.5
    assert voi.loc[50.0, "fraction_probed"] == 0.0
    assert voi.loc[1.0, "mean_net_utility"] > voi.loc[50.0, "mean_net_utility"]

    memo = (out_dir / "m7_probe_cost_sensitivity_memo.md").read_text(encoding="utf-8")
    assert "Phase 2 Probe Cost Sensitivity" in memo
    assert "voi_probe" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Phase 2 Probe Cost Sensitivity" in readme
