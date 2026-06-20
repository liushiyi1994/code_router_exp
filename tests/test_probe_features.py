from __future__ import annotations

import math

import pandas as pd

from routecode.probes.probe_features import (
    PROBE_FEATURE_COLUMNS,
    build_probe_features_from_outcomes,
    compute_knn_uncertainty,
)


def test_build_probe_features_from_local_outcomes_uses_phase2_schema():
    outcomes = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "What is 2+3?",
                "dataset": "gsm8k_smoke",
                "domain": "math",
                "model_id": "cheap_probe_a",
                "model_revision": "dry-run",
                "prompt_template": "math_answer_v1",
                "generation_params_json": '{"max_tokens": 16, "temperature": 0.0}',
                "raw_output": "Final answer: 5\nConfidence: 0.82",
                "parsed_answer": "5",
                "gold_answer": "5",
                "quality": 1.0,
                "cost_proxy": 0.004,
                "latency_sec": 0.01,
                "tokens_input": 12,
                "tokens_output": 4,
                "error_type": "",
                "error_message": "",
                "created_at": "2026-06-17T00:00:00+00:00",
            },
            {
                "query_id": "q1",
                "query_text": "What is 2+3?",
                "dataset": "gsm8k_smoke",
                "domain": "math",
                "model_id": "cheap_probe_b",
                "model_revision": "dry-run",
                "prompt_template": "math_answer_v1",
                "generation_params_json": '{"max_tokens": 16, "temperature": 0.0}',
                "raw_output": "Final answer: 5",
                "parsed_answer": "5",
                "gold_answer": "5",
                "quality": 1.0,
                "cost_proxy": 0.003,
                "latency_sec": 0.02,
                "tokens_input": 12,
                "tokens_output": 3,
                "error_type": "",
                "error_message": "",
                "created_at": "2026-06-17T00:00:01+00:00",
            },
            {
                "query_id": "q2",
                "query_text": "Pick A or B.",
                "dataset": "mmlu_smoke",
                "domain": "broad_knowledge",
                "model_id": "cheap_probe_a",
                "model_revision": "dry-run",
                "prompt_template": "multiple_choice_letter_v1",
                "generation_params_json": '{"max_tokens": 16, "temperature": 0.0}',
                "raw_output": "A",
                "parsed_answer": "A",
                "gold_answer": "B",
                "quality": 0.0,
                "cost_proxy": 0.001,
                "latency_sec": 0.03,
                "tokens_input": 10,
                "tokens_output": 1,
                "error_type": "",
                "error_message": "",
                "created_at": "2026-06-17T00:00:02+00:00",
            },
            {
                "query_id": "q2",
                "query_text": "Pick A or B.",
                "dataset": "mmlu_smoke",
                "domain": "broad_knowledge",
                "model_id": "cheap_probe_b",
                "model_revision": "dry-run",
                "prompt_template": "multiple_choice_letter_v1",
                "generation_params_json": '{"max_tokens": 16, "temperature": 0.0}',
                "raw_output": "B",
                "parsed_answer": "B",
                "gold_answer": "B",
                "quality": 1.0,
                "cost_proxy": 0.001,
                "latency_sec": 0.04,
                "tokens_input": 10,
                "tokens_output": 1,
                "error_type": "",
                "error_message": "",
                "created_at": "2026-06-17T00:00:03+00:00",
            },
        ]
    )

    features = build_probe_features_from_outcomes(outcomes)

    assert list(features.columns) == PROBE_FEATURE_COLUMNS
    assert len(features) == 4
    assert set(features["probe_type"]) == {"local_answer_probe"}
    q1_a = features[(features["query_id"] == "q1") & (features["probe_model_id"] == "cheap_probe_a")].iloc[0]
    assert q1_a["probe_id"] == "local_answer_probe:cheap_probe_a:math_answer_v1"
    assert q1_a["parsed_probe_answer"] == "5"
    assert q1_a["self_confidence"] == 0.82
    assert q1_a["agreement_score"] == 1.0
    assert q1_a["latency_sec"] == 0.01
    assert q1_a["input_tokens"] == 12
    assert q1_a["output_tokens"] == 4
    assert q1_a["probe_cost_proxy"] == 0.004

    q2_a = features[(features["query_id"] == "q2") & (features["probe_model_id"] == "cheap_probe_a")].iloc[0]
    assert q2_a["agreement_score"] == 0.5
    assert math.isnan(q2_a["self_confidence"])
    assert math.isnan(q2_a["knn_label_entropy"])
    assert math.isnan(q2_a["knn_winner_entropy"])


def test_knn_uncertainty_uses_train_neighbors_only():
    embeddings = pd.DataFrame(
        [
            {"query_id": "train_a", "emb_0": 0.0, "emb_1": 0.0},
            {"query_id": "train_b", "emb_0": 0.2, "emb_1": 0.0},
            {"query_id": "test_q", "emb_0": 0.1, "emb_1": 0.0},
        ]
    )
    labels = pd.DataFrame(
        [
            {"query_id": "train_a", "route_label": 0, "winner_model_id": "model_a"},
            {"query_id": "train_b", "route_label": 1, "winner_model_id": "model_a"},
            {"query_id": "test_q", "route_label": 9, "winner_model_id": "model_z"},
        ]
    )

    uncertainty = compute_knn_uncertainty(
        embeddings=embeddings,
        labels=labels,
        train_query_ids=["train_a", "train_b"],
        target_query_ids=["test_q"],
        k=2,
    )

    row = uncertainty.iloc[0]
    assert row["query_id"] == "test_q"
    assert row["knn_label_entropy"] == 1.0
    assert row["knn_winner_entropy"] == 0.0
