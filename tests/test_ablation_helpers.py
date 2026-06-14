import pandas as pd

from routecode.eval.ablation import configured_sweep_values, sample_train_query_ids


def test_configured_sweep_values_include_base_once_in_order():
    config = {
        "utility": {"lambda_cost": 0.35},
        "ablation": {"lambda_values": [0.0, 0.35, 0.7]},
    }

    values = configured_sweep_values(config, section="ablation", key="lambda_values", base_value=0.35)

    assert values == [0.0, 0.35, 0.7]


def test_configured_sweep_values_add_missing_base_value():
    config = {"ablation": {"seeds": [3, 5]}}

    values = configured_sweep_values(config, section="ablation", key="seeds", base_value=7, cast=int)

    assert values == [3, 5, 7]


def test_sample_train_query_ids_is_seeded_and_keeps_at_least_one_query():
    index = pd.Index([f"q{i}" for i in range(10)], name="query_id")

    first = sample_train_query_ids(index, fraction=0.3, seed=11)
    second = sample_train_query_ids(index, fraction=0.3, seed=11)
    minimum = sample_train_query_ids(index, fraction=0.0, seed=11)

    assert first.equals(second)
    assert len(first) == 3
    assert len(minimum) == 1
    assert set(first).issubset(set(index))
