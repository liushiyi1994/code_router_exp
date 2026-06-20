from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from routecode.probes.routecode_policy_inputs import RouteCodePolicyInputs


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "73_routecode_target_rate_policy_inputs.py"
    spec = importlib.util.spec_from_file_location("routecode_target_rate_policy_inputs_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_routecode_target_rate_policy_inputs_script_writes_expected_files(tmp_path):
    module = _load_script()
    index = pd.Index(["q0", "q1"], name="query_id")
    bundle = RouteCodePolicyInputs(
        before_beliefs=pd.DataFrame({"z0": [1.0, 0.0], "z1": [0.0, 1.0]}, index=index),
        after_beliefs=pd.DataFrame({"z0": [1.0, 0.0], "z1": [0.0, 1.0]}, index=index),
        state_model_utility=pd.DataFrame(
            {"m0": [1.0, 0.0], "m1": [0.0, 1.0]},
            index=pd.Index(["z0", "z1"], name="state_label"),
        ),
        query_model_utility=pd.DataFrame({"m0": [1.0, 0.0], "m1": [0.0, 1.0]}, index=index),
        probe_cost=pd.Series([0.0, 0.0], index=index, name="probe_cost"),
        predicted_gain=pd.Series([0.0, 0.0], index=index, name="predicted_gain"),
        metadata={
            "k": 2,
            "alpha": 0.0,
            "effective_labels": 2,
            "train_rows": 4,
            "policy_rows": 2,
            "belief_type": "one_hot_routecode_embedding_predicted",
        },
    )
    out_dir = tmp_path / "routecode_policy_inputs"

    paths = module.write_outputs(out_dir, bundle)
    module.write_memo(out_dir, "config.yaml", "query_utility.csv", paths, bundle)
    module.append_readme(out_dir, paths, bundle)

    for path in paths.values():
        assert Path(path).exists()
    assert "query_id" in pd.read_csv(paths["before_beliefs"]).columns
    assert "state_label" in pd.read_csv(paths["state_model_utility"]).columns
    assert "Target-Rate RouteCode Policy Inputs" in (out_dir / "m_routecode_target_rate_policy_inputs.md").read_text(
        encoding="utf-8"
    )
