from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "64_true_probe_policy_inputs.py"
    spec = importlib.util.spec_from_file_location("phase2_true_probe_policy_inputs", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_true_probe_policy_inputs_write_state_beliefs_for_test_queries_only(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "policy_inputs"
    probe_path = tmp_path / "probe_features.parquet"
    state_targets_path = tmp_path / "state_targets.csv"
    query_features_path = tmp_path / "query_features.csv"
    state_utility_path = tmp_path / "state_model_utility.csv"
    query_utility_path = tmp_path / "query_model_utility.csv"

    query_ids = [f"train_{idx}" for idx in range(8)] + ["test_0", "test_1", "val_0"]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1]
    splits = ["train"] * 8 + ["test", "test", "val"]
    pd.DataFrame(
        {
            "query_id": query_ids,
            "state_label": labels,
            "split": splits,
        }
    ).to_csv(state_targets_path, index=False)
    pd.DataFrame(
        {
            "query_id": query_ids,
            "hash_0": [float(label) for label in labels],
            "hash_1": [float(idx % 3) for idx in range(len(query_ids))],
        }
    ).to_csv(query_features_path, index=False)
    pd.DataFrame(
        {
            "query_id": query_ids,
            "self_confidence": [0.15 if label == 0 else 0.85 for label in labels],
            "agreement_score": 1.0,
            "knn_label_entropy": float("nan"),
            "knn_winner_entropy": float("nan"),
            "latency_sec": 0.2,
            "input_tokens": 20,
            "output_tokens": 4,
            "probe_cost_proxy": 0.01,
            "error_type": "",
        }
    ).to_parquet(probe_path, index=False)
    pd.DataFrame(
        {
            "state_label": ["z0", "z1"],
            "cheap": [0.8, 0.2],
            "strong": [0.1, 0.9],
        }
    ).to_csv(state_utility_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["test_0", "test_1"],
            "cheap": [0.8, 0.1],
            "strong": [0.2, 0.9],
        }
    ).to_csv(query_utility_path, index=False)

    paths = module.run(
        probe_features_path=str(probe_path),
        state_targets_path=str(state_targets_path),
        query_features_path=str(query_features_path),
        state_model_utility_path=str(state_utility_path),
        query_model_utility_path=str(query_utility_path),
        output_dir=str(out_dir),
    )

    before = pd.read_csv(paths["before_beliefs"])
    after = pd.read_csv(paths["after_beliefs"])
    predicted_gain = pd.read_csv(paths["predicted_gain"])
    probe_cost = pd.read_csv(paths["probe_cost"])
    query_utility = pd.read_csv(paths["query_model_utility"])
    assert before["query_id"].tolist() == ["test_0", "test_1"]
    assert after["query_id"].tolist() == ["test_0", "test_1"]
    assert query_utility["query_id"].tolist() == ["test_0", "test_1"]
    assert set(before.columns) == {"query_id", "z0", "z1"}
    assert set(after.columns) == {"query_id", "z0", "z1"}
    assert before[["z0", "z1"]].sum(axis=1).round(6).eq(1.0).all()
    assert after[["z0", "z1"]].sum(axis=1).round(6).eq(1.0).all()
    assert predicted_gain["query_id"].tolist() == ["test_0", "test_1"]
    assert probe_cost["probe_cost"].gt(0.0).all()
    memo = (out_dir / "m12_true_probe_policy_inputs_memo.md").read_text(encoding="utf-8")
    assert "latent route-state beliefs" in memo
    assert "not direct probe-to-model routing" in memo
