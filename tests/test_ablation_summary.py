from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_ablation_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "08_ablation_summary.py"
    spec = importlib.util.spec_from_file_location("ablation_summary", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _toy_outcomes() -> pd.DataFrame:
    rows = []
    for idx in range(9):
        if idx < 5:
            split = "train"
        elif idx == 5:
            split = "val"
        else:
            split = "test"
        best_model = "m0" if idx % 2 == 0 else "m1"
        for model in ["m0", "m1"]:
            rows.append(
                {
                    "query_id": f"q{idx}",
                    "query_text": f"query {idx}",
                    "dataset": "toy",
                    "domain": "toy",
                    "model_id": model,
                    "quality": 1.0 if model == best_model else 0.0,
                    "cost_input": 0.0,
                    "cost_output": 0.0,
                    "cost_total": 0.0,
                    "judge": "toy",
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _toy_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0],
            "y": [0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0],
        },
        index=[f"q{idx}" for idx in range(9)],
    )


def test_rate_penalty_rows_emit_one_d2_row_per_beta():
    module = _load_ablation_script()

    rows = module._rate_penalty_rows(
        _toy_outcomes(),
        _toy_embeddings(),
        beta_values=[0.0, 0.5],
        k=2,
        lambda_cost=0.0,
        seed=0,
        max_iter=10,
        d2_alpha=1.0,
        n_bootstrap=5,
        ci=0.95,
    )

    table = pd.DataFrame(rows)
    assert list(table["ablation"]) == ["rate_penalty", "rate_penalty"]
    assert list(table["method"]) == ["d2_embedding_centroid", "d2_embedding_centroid"]
    assert list(table["d2_beta"]) == [0.0, 0.5]
