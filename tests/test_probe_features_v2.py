import pandas as pd

from routecode.states.probe_features_v2 import build_local_behavior_probe_features, numeric_feature_columns


def test_build_local_behavior_probe_features_uses_observable_local_outputs_only():
    outputs = pd.DataFrame(
        [
            {
                "query_id": "q0",
                "query_text": "solve 1+1",
                "split": "train",
                "benchmark": "math",
                "domain": "math",
                "model_id": "small-local",
                "is_local": True,
                "parsed_answer": "2",
                "status": "success",
                "output_tokens": 3,
                "latency_s": 0.1,
                "quality_score": 1.0,
                "utility": 1.0,
                "gold_answer": "2",
            },
            {
                "query_id": "q0",
                "query_text": "solve 1+1",
                "split": "train",
                "benchmark": "math",
                "domain": "math",
                "model_id": "big-local",
                "is_local": True,
                "parsed_answer": "2",
                "status": "success",
                "output_tokens": 4,
                "latency_s": 0.2,
                "quality_score": 1.0,
                "utility": 1.0,
                "gold_answer": "2",
            },
            {
                "query_id": "q0",
                "query_text": "solve 1+1",
                "split": "train",
                "benchmark": "math",
                "domain": "math",
                "model_id": "gpt-frontier",
                "is_local": False,
                "parsed_answer": "frontier secret",
                "status": "success",
                "output_tokens": 99,
                "latency_s": 9.0,
                "quality_score": 1.0,
                "utility": 1.0,
                "gold_answer": "2",
            },
        ]
    )

    features = build_local_behavior_probe_features(outputs)

    assert len(features) == 1
    row = features.iloc[0]
    assert row["probe_valid_answer_count"] == 2.0
    assert row["probe_all_agree"] == 1.0
    assert row["probe_vote_frac"] == 1.0
    assert row["probe_output_tokens_max"] == 4.0
    assert "quality_score" not in features.columns
    assert "utility" not in features.columns
    assert "gold_answer" not in features.columns
    assert all(not col.startswith("probe_gpt") for col in features.columns)


def test_build_local_behavior_probe_features_detects_disagreement():
    outputs = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "choose answer",
                "model_id": "a-local",
                "is_local": True,
                "parsed_answer": "A",
                "status": "success",
            },
            {
                "query_id": "q1",
                "query_text": "choose answer",
                "model_id": "b-local",
                "is_local": True,
                "parsed_answer": "B",
                "status": "success",
            },
        ]
    )

    features = build_local_behavior_probe_features(outputs)

    assert features.loc[0, "probe_any_disagree"] == 1.0
    assert features.loc[0, "probe_unique_answer_count"] == 2.0
    assert features.loc[0, "probe_vote_entropy"] == 1.0
    assert "probe_vote_entropy" in numeric_feature_columns(features)
