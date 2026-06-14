from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "11_code_card_interpretability.py"
    spec = importlib.util.spec_from_file_location("code_card_interpretability", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_interpretability_table_includes_flat_and_predictability_constrained_codebooks():
    module = _load_script()
    utility = pd.DataFrame(
        {
            "cheap": [0.9, 0.85, 0.2, 0.25],
            "strong": [0.3, 0.35, 0.95, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    query_info = pd.DataFrame(
        {
            "query_id": utility.index,
            "query_text": ["short fact", "simple lookup", "hard proof", "difficult derivation"],
            "dataset": ["easy", "easy", "hard", "hard"],
            "domain": ["general", "general", "math", "math"],
        }
    ).set_index("query_id")
    embeddings = pd.DataFrame([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]], index=utility.index)
    config = {
        "run": {"random_seed": 0},
        "routecode": {"selected_k_for_cards": 2, "max_iter": 25},
        "predictability_constrained": {
            "k": 2,
            "selected_alpha": 1.0,
            "beta": 0.0,
            "max_iter": 25,
            "refinement_iter": 3,
        },
    }

    table = module.build_interpretability_table(config, query_info, utility, embeddings)

    assert set(table["codebook"]) == {"flat_routecode", "predictability_constrained_routecode"}
    assert table.groupby("codebook")["condition"].apply(list).to_dict() == {
        "flat_routecode": ["label_only", "with_code_cards"],
        "predictability_constrained_routecode": ["label_only", "with_code_cards"],
    }
    with_cards = table[table["condition"] == "with_code_cards"]
    label_only = table[table["condition"] == "label_only"]
    assert (with_cards["human_explanation_coverage"] == 1.0).all()
    assert (label_only["human_explanation_coverage"] == 0.0).all()
