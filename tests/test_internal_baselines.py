import pandas as pd

from routecode.predictors.classifiers import SVMModelRouter
from routecode.routers.dataset_lookup import DatasetOracleRouter
from routecode.routers.random import RandomRouter


def _query_info() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2", "q3"],
            "dataset": ["a", "a", "b", "b"],
            "domain": ["a", "a", "b", "b"],
        }
    ).set_index("query_id")


def _utility() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m0": [0.9, 0.8, 0.1, 0.2],
            "m1": [0.1, 0.2, 0.9, 0.8],
        },
        index=["q0", "q1", "q2", "q3"],
    )


def test_random_router_is_seeded_and_uses_train_model_pool():
    query_info = _query_info()
    utility = _utility()

    router = RandomRouter(random_state=11).fit(query_info.iloc[:2], utility.iloc[:2])
    first = router.predict(query_info)
    second = router.predict(query_info)

    assert first.equals(second)
    assert set(first).issubset({"m0", "m1"})
    assert list(first.index) == list(query_info.index)


def test_dataset_oracle_fits_dataset_best_models_from_supplied_matrix():
    query_info = _query_info()
    utility = _utility()

    oracle = DatasetOracleRouter(label_column="dataset").fit(query_info, utility)
    selected = oracle.predict(query_info)

    assert selected.loc["q0"] == "m0"
    assert selected.loc["q1"] == "m0"
    assert selected.loc["q2"] == "m1"
    assert selected.loc["q3"] == "m1"


def test_svm_model_router_predicts_model_ids_from_embeddings():
    query_info = _query_info()
    utility = _utility()
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]],
        index=utility.index,
    )

    router = SVMModelRouter(random_state=0).fit(query_info, utility, embeddings)
    selected = router.predict(query_info, embeddings)

    assert selected.loc["q0"] == "m0"
    assert selected.loc["q3"] == "m1"
    assert set(selected).issubset({"m0", "m1"})
