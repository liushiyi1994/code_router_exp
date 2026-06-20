from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import numpy as np


def _load_frugalgpt_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/raw/external/LLMRouterBench/baselines/FrugalGPT/train_router_from_results.py"
    )
    spec = importlib.util.spec_from_file_location("frugalgpt_train_router_from_results", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frugalgpt_evaluate_per_dataset_handles_groupby_without_group_columns(capsys):
    module = _load_frugalgpt_module()
    test_df = pd.DataFrame(
        {
            "dataset_name": ["d0", "d0", "d1", "d1"],
            "sample_id": ["q0", "q0", "q1", "q1"],
            "label": [0, 1, 0, 1],
            "cost": [0.1, 0.2, 0.1, 0.2],
            "model_name": ["m0", "m1", "m0", "m1"],
        }
    )

    module.evaluate_per_dataset(test_df, probabilities=np.array([0.2, 0.8, 0.4, 0.3]), threshold=0.5)

    captured = capsys.readouterr()
    assert "All datasets:" in captured.out
