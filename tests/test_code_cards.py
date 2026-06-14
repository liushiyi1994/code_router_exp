import json

import pandas as pd

from routecode.codes.code_cards import build_code_cards, write_code_cards_json
from routecode.codes.routecode import RouteCodeCodebook


def code_card_case():
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
            "query_text": [
                "short fact",
                "simple lookup",
                "hard proof",
                "difficult derivation",
            ],
            "dataset": ["easy", "easy", "hard", "hard"],
            "domain": ["general", "general", "math", "math"],
        }
    ).set_index("query_id")
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        index=utility.index,
    )
    return query_info, utility, embeddings


def test_build_code_cards_includes_required_explainability_fields():
    query_info, utility, embeddings = code_card_case()
    codebook = RouteCodeCodebook(2, random_state=0).fit(query_info, utility, embeddings)

    cards = build_code_cards(codebook, query_info, utility, max_examples=2)

    assert len(cards) == 2
    card = cards[0]
    for field in [
        "label_id",
        "short_name",
        "best_model",
        "second_best_model",
        "model_margin",
        "top_datasets",
        "top_domains",
        "representative_queries",
        "high_regret_failure_cases",
        "model_utility_vector",
        "size",
        "human_readable_explanation",
    ]:
        assert field in card


def test_write_code_cards_json_writes_machine_readable_cards(tmp_path):
    query_info, utility, embeddings = code_card_case()
    codebook = RouteCodeCodebook(2, random_state=0).fit(query_info, utility, embeddings)
    path = tmp_path / "code_cards.json"

    write_code_cards_json(path, codebook, query_info, utility, max_examples=1)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "routecode.code_cards.v1"
    assert len(data["cards"]) == 2
    assert "representative_queries" in data["cards"][0]
