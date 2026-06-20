from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "54_proberoute_policy.py"
    spec = importlib.util.spec_from_file_location("phase2_proberoute_policy", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_proberoute_policy_script_writes_blocked_table_figure_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()

    table = module.run(output_dir=str(out_dir))

    table_path = out_dir / "table_proberoute_policy.csv"
    figure_path = out_dir / "fig_gap_closed_vs_probe_cost.pdf"
    memo_path = out_dir / "m5_proberoute_policy_memo.md"
    readme_path = out_dir / "README.md"
    assert table_path.exists()
    assert figure_path.exists()
    assert memo_path.exists()
    assert readme_path.exists()
    assert set(table["status"]) == {"blocked_missing_policy_inputs"}
    assert "## Phase 2 ProbeRoute++ Policy" in readme_path.read_text(encoding="utf-8")
    assert "cannot support ProbeRoute++ policy claims" in memo_path.read_text(encoding="utf-8")


def test_proberoute_policy_script_computes_gap_closed_for_aligned_inputs(tmp_path):
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

    table = module.run(
        output_dir=str(out_dir),
        before_beliefs_path=str(before),
        after_beliefs_path=str(after),
        state_model_utility_path=str(state_utility),
        query_model_utility_path=str(query_utility),
        probe_cost_path=str(probe_cost),
        predicted_gain_path=str(predicted_gain),
    )

    assert set(table["status"]) == {"executed"}
    assert table["observability_gap_closed"].notna().all()
    assert table["mean_net_utility_ci_low"].notna().all()
    assert table["mean_net_utility_ci_high"].notna().all()
    memo = (out_dir / "m5_proberoute_policy_memo.md").read_text(encoding="utf-8")
    assert "mean_net_utility_ci_low" in memo
    assert "observability_gap_closed_ci_high" in memo
