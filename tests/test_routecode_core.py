import numpy as np
import pandas as pd

from routecode.data.schema import REQUIRED_OUTCOME_COLUMNS, validate_outcomes
from routecode.data.splits import split_by_query
from routecode.matrix import build_matrices
from routecode.metrics import (
    bootstrap_mean_ci,
    model_win_entropy,
    recovered_gap,
    router_summary,
)
from routecode.routers.dataset_lookup import DatasetLabelRouter
from routecode.routers.oracle import OracleRouter
from routecode.codes.routecode import RouteCodeCodebook


def tiny_outcomes() -> pd.DataFrame:
    rows = []
    qualities = {
        "q0": {"cheap": 0.8, "strong": 0.9},
        "q1": {"cheap": 0.7, "strong": 0.6},
        "q2": {"cheap": 0.4, "strong": 0.9},
        "q3": {"cheap": 0.6, "strong": 0.8},
    }
    datasets = {"q0": "easy", "q1": "easy", "q2": "hard", "q3": "hard"}
    costs = {"cheap": 0.1, "strong": 0.5}
    for qid, by_model in qualities.items():
        for model_id, quality in by_model.items():
            rows.append(
                {
                    "query_id": qid,
                    "query_text": f"query {qid}",
                    "dataset": datasets[qid],
                    "domain": datasets[qid],
                    "model_id": model_id,
                    "quality": quality,
                    "cost_input": costs[model_id] / 2,
                    "cost_output": costs[model_id] / 2,
                    "cost_total": costs[model_id],
                    "latency": costs[model_id] * 10,
                    "tokens_input": 10,
                    "tokens_output": 20,
                    "judge": "synthetic",
                    "metadata_json": "{}",
                }
            )
    return pd.DataFrame(rows)


def test_required_schema_columns_validate():
    outcomes = tiny_outcomes()
    assert set(REQUIRED_OUTCOME_COLUMNS).issubset(outcomes.columns)
    validated = validate_outcomes(outcomes)
    assert len(validated) == len(outcomes)


def test_utility_matrix_uses_quality_minus_lambda_cost():
    matrices = build_matrices(tiny_outcomes(), lambda_cost=0.25)
    q0_cheap = matrices.utility.loc["q0", "cheap"]
    assert np.isclose(q0_cheap, 0.8 - 0.25 * 0.1)
    assert list(matrices.utility.index) == ["q0", "q1", "q2", "q3"]
    assert list(matrices.utility.columns) == ["cheap", "strong"]


def test_split_by_query_keeps_all_model_rows_together():
    split = split_by_query(
        tiny_outcomes(),
        train_frac=0.5,
        val_frac=0.25,
        test_frac=0.25,
        seed=1,
    )
    per_query = split.groupby("query_id")["split"].nunique()
    assert per_query.max() == 1
    assert set(split["split"]) == {"train", "val", "test"}


def test_oracle_router_selects_query_best_model():
    matrices = build_matrices(tiny_outcomes(), lambda_cost=0.0)
    router = OracleRouter()
    selected = router.predict(matrices.utility)
    assert selected.loc["q1"] == "cheap"
    assert selected.loc["q2"] == "strong"


def test_dataset_label_router_fits_tables_from_train_only():
    outcomes = tiny_outcomes()
    train = outcomes[outcomes["query_id"].isin(["q0", "q1", "q2"])]
    test_query_info = outcomes[outcomes["query_id"].isin(["q3"])].drop_duplicates("query_id")
    train_matrices = build_matrices(train, lambda_cost=0.0)
    router = DatasetLabelRouter(label_column="dataset").fit(train_matrices.query_info, train_matrices.utility)
    selected = router.predict(test_query_info)
    assert selected.loc["q3"] == "strong"


def test_metrics_recovered_gap_entropy_and_bootstrap_ci_are_bounded():
    selected = np.array([0.5, 0.7, 0.9, 1.1])
    summary = router_summary(selected, oracle_utility=np.array([1.0, 1.0, 1.0, 1.2]))
    assert np.isclose(summary["mean_utility"], 0.8)
    assert summary["oracle_regret"] > 0
    assert np.isclose(recovered_gap(method=0.8, baseline=0.5, reference=1.0), 0.6)
    assert recovered_gap(method=0.4, baseline=0.5, reference=0.4) == 0.0
    assert model_win_entropy(["a", "a", "b", "b"]) == 1.0
    low, high = bootstrap_mean_ci(selected, n_bootstrap=100, seed=0)
    assert low <= selected.mean() <= high


def test_routecode_codebook_assigns_predictable_labels_without_test_utilities():
    utility = pd.DataFrame(
        {
            "cheap": [0.9, 0.85, 0.2, 0.25],
            "strong": [0.3, 0.35, 0.95, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    query_info = pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2", "q3"],
            "dataset": ["easy", "easy", "hard", "hard"],
            "domain": ["easy", "easy", "hard", "hard"],
        }
    ).set_index("query_id")
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        index=utility.index,
    )
    codebook = RouteCodeCodebook(n_labels=2, random_state=0).fit(query_info, utility, embeddings)
    labels = codebook.predict_labels(embeddings)
    selected = codebook.predict(query_info, embeddings)
    assert set(labels).issubset({0, 1})
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q3"] == "strong"


def test_routecode_codebook_can_assign_oracle_labels_from_utility_vectors():
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
            "dataset": ["easy", "easy", "hard", "hard"],
            "domain": ["easy", "easy", "hard", "hard"],
        }
    ).set_index("query_id")
    embeddings = pd.DataFrame(
        # Deliberately uninformative embeddings; oracle labels should use utility, not text geometry.
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        index=utility.index,
    )
    codebook = RouteCodeCodebook(n_labels=2, random_state=0).fit(query_info, utility, embeddings)

    utility_labels = codebook.predict_utility_labels(utility)
    selected = codebook.predict_from_labels(utility_labels)

    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q3"] == "strong"
