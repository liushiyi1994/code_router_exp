from __future__ import annotations

import pandas as pd

from routecode.codes.code_cards import build_code_cards
from routecode.codes.routecode import RouteCodeCodebook
from routecode.eval.code_card_interpretability import summarize_code_card_interpretability


def test_summarize_code_card_interpretability_compares_label_only_to_cards():
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
    codebook = RouteCodeCodebook(2, random_state=0).fit(query_info, utility, embeddings)
    cards = build_code_cards(codebook, query_info, utility, max_examples=2)

    summary = summarize_code_card_interpretability("flat_routecode", codebook, cards)

    assert list(summary["condition"]) == ["label_only", "with_code_cards"]
    label_only = summary.set_index("condition").loc["label_only"]
    with_cards = summary.set_index("condition").loc["with_code_cards"]
    assert label_only["human_explanation_coverage"] == 0.0
    assert with_cards["human_explanation_coverage"] == 1.0
    assert with_cards["representative_query_coverage"] == 1.0
    assert with_cards["failure_case_coverage"] == 1.0
    assert with_cards["available_explainability_fields"] > label_only["available_explainability_fields"]
