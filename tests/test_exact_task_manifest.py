from __future__ import annotations

import json

import pandas as pd

from routecode.local_eval.task_manifest import (
    EXACT_TASK_MANIFEST_COLUMNS,
    build_exact_task_manifest,
    tasks_from_manifest,
)


def _row(query_id: str, dataset: str, split: str, ground_truth: str, model_id: str = "m1") -> dict:
    return {
        "query_id": query_id,
        "query_text": f"Question for {query_id}",
        "dataset": dataset,
        "domain": "math",
        "split": split,
        "source_split": "test",
        "record_index": int(query_id.rsplit("_", 1)[-1]),
        "model_id": model_id,
        "quality": 1.0,
        "cost_total": 0.0,
        "metadata_json": json.dumps({"ground_truth": ground_truth, "source": "LLMRouterBench"}),
    }


def test_build_exact_task_manifest_uses_split_ground_truth_and_round_robin_datasets():
    outcomes = pd.DataFrame(
        [
            _row("aime_0", "aime", "test", "10", model_id="m1"),
            _row("aime_0", "aime", "test", "10", model_id="m2"),
            _row("math500_1", "math500", "test", "x + y"),
            _row("aime_2", "aime", "train", "12"),
            _row("mbpp_3", "mbpp", "test", ""),
            _row("math500_4", "math500", "test", "42"),
        ]
    )

    manifest = build_exact_task_manifest(
        outcomes,
        datasets=["aime", "math500", "mbpp"],
        split="test",
        max_queries=3,
    )

    assert list(manifest.columns) == EXACT_TASK_MANIFEST_COLUMNS
    assert manifest["query_id"].tolist() == ["aime_0", "math500_1", "math500_4"]
    assert manifest["task_type"].tolist() == ["math", "math", "math"]
    assert manifest["gold_answer"].tolist() == ["10", "x + y", "42"]
    assert manifest["routecode_split"].eq("test").all()
    assert "mbpp_3" not in set(manifest["query_id"])


def test_build_exact_task_manifest_split_all_keeps_all_routecode_splits():
    outcomes = pd.DataFrame(
        [
            _row("aime_0", "aime", "train", "10"),
            _row("aime_1", "aime", "val", "11"),
            _row("math500_2", "math500", "test", "12"),
            _row("math500_3", "math500", "train", "13"),
        ]
    )

    manifest = build_exact_task_manifest(
        outcomes,
        datasets=["aime", "math500"],
        split="all",
        max_queries=10,
    )

    assert manifest["query_id"].tolist() == ["aime_0", "math500_2", "aime_1", "math500_3"]
    assert set(manifest["routecode_split"]) == {"train", "val", "test"}


def test_tasks_from_manifest_reconstructs_local_eval_tasks():
    manifest = pd.DataFrame(
        [
            {
                "query_id": "aime_0",
                "query_text": "Find x.",
                "dataset": "aime",
                "domain": "math",
                "source_split": "test",
                "routecode_split": "test",
                "task_type": "math",
                "gold_answer": "7",
                "choices_json": "[]",
                "metadata_json": "{}",
            }
        ],
        columns=EXACT_TASK_MANIFEST_COLUMNS,
    )

    tasks = tasks_from_manifest(manifest)

    assert len(tasks) == 1
    assert tasks[0].query_id == "aime_0"
    assert tasks[0].task_type == "math"
    assert tasks[0].gold_answer == "7"
