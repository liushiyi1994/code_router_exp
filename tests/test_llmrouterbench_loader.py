import json

import pandas as pd

from routecode.data.load_llmrouterbench import load_llmrouterbench_outcomes
from routecode.pipeline import prepare_from_config


def write_result(path, dataset, split, model, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, score in enumerate(scores):
        records.append(
            {
                "index": index,
                "origin_query": f"{dataset} query {index}",
                "prompt": f"Prompt for {dataset} {index}",
                "prediction": f"pred-{model}-{index}",
                "ground_truth": f"gold-{index}",
                "score": score,
                "prompt_tokens": 10 + index,
                "completion_tokens": 5 + index,
                "cost": 0.001 * (index + 1),
                "raw_output": {"text": "ignored"},
            }
        )
    payload = {
        "dataset_name": dataset,
        "split": split,
        "model_name": model,
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_llmrouterbench_outcomes_converts_official_result_shape(tmp_path):
    root = tmp_path / "bench"
    write_result(root / "aime" / "test" / "model_a" / "20260101_000000.json", "aime", "test", "model_a", [1, 0])
    write_result(root / "aime" / "test" / "model_b" / "20260101_000000.json", "aime", "test", "model_b", [0, 1])

    outcomes = load_llmrouterbench_outcomes(root)

    assert len(outcomes) == 4
    assert set(outcomes["model_id"]) == {"model_a", "model_b"}
    assert set(outcomes["quality"]) == {0.0, 1.0}
    assert set(outcomes["query_id"]) == {"aime:test:0", "aime:test:1"}
    assert outcomes.loc[outcomes["query_id"] == "aime:test:0", "query_text"].nunique() == 1
    assert {"cost_input", "cost_output", "cost_total", "metadata_json"}.issubset(outcomes.columns)


def test_prepare_from_config_loads_llmrouterbench_source(tmp_path):
    root = tmp_path / "bench"
    write_result(root / "aime" / "test" / "model_a" / "20260101_000000.json", "aime", "test", "model_a", [1, 0, 1])
    write_result(root / "aime" / "test" / "model_b" / "20260101_000000.json", "aime", "test", "model_b", [0, 1, 0])
    config = {
        "run": {"random_seed": 0, "output_dir": str(tmp_path / "out")},
        "data": {"source": "llmrouterbench", "results_dir": str(root)},
        "utility": {"lambda_cost": 0.0},
        "split": {"train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
    }

    prepared = prepare_from_config(config)

    assert prepared.outcomes["query_id"].nunique() == 3
    assert isinstance(prepared.embeddings, pd.DataFrame)
    assert set(prepared.matrices) == {"train", "val", "test"}


def test_prepare_from_config_applies_domain_map_before_splitting(tmp_path):
    root = tmp_path / "bench"
    write_result(root / "aime" / "test" / "model_a" / "20260101_000000.json", "aime", "test", "model_a", [1, 0, 1])
    write_result(root / "aime" / "test" / "model_b" / "20260101_000000.json", "aime", "test", "model_b", [0, 1, 0])
    write_result(root / "mbpp" / "test" / "model_a" / "20260101_000000.json", "mbpp", "test", "model_a", [1, 0, 1])
    write_result(root / "mbpp" / "test" / "model_b" / "20260101_000000.json", "mbpp", "test", "model_b", [0, 1, 0])
    config = {
        "run": {"random_seed": 0, "output_dir": str(tmp_path / "out")},
        "data": {
            "source": "llmrouterbench",
            "results_dir": str(root),
            "domain_map": {"aime": "math", "mbpp": "code"},
        },
        "utility": {"lambda_cost": 0.0},
        "split": {"train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
    }

    prepared = prepare_from_config(config)

    query_domains = prepared.outcomes.drop_duplicates("query_id").set_index("dataset")["domain"].to_dict()
    assert query_domains == {"aime": "math", "mbpp": "code"}
    assert prepared.outcomes.groupby("query_id")["domain"].nunique().max() == 1


def test_prepare_from_config_applies_task_taxonomy_before_splitting(tmp_path):
    root = tmp_path / "bench"
    write_result(root / "aime" / "test" / "model_a" / "20260101_000000.json", "aime", "test", "model_a", [1, 0, 1])
    write_result(root / "aime" / "test" / "model_b" / "20260101_000000.json", "aime", "test", "model_b", [0, 1, 0])
    write_result(root / "gpqa" / "test" / "model_a" / "20260101_000000.json", "gpqa", "test", "model_a", [1, 0, 1])
    write_result(root / "gpqa" / "test" / "model_b" / "20260101_000000.json", "gpqa", "test", "model_b", [0, 1, 0])
    config = {
        "run": {"random_seed": 0, "output_dir": str(tmp_path / "out")},
        "data": {
            "source": "llmrouterbench",
            "results_dir": str(root),
            "task_taxonomy_map": {
                "aime": {"task_family": "math_reasoning", "task_subtype": "competition_math"},
                "gpqa": {"task_family": "science_reasoning", "task_subtype": "graduate_science_qa"},
            },
        },
        "utility": {"lambda_cost": 0.0},
        "split": {"train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
    }

    prepared = prepare_from_config(config)

    query_taxonomy = (
        prepared.outcomes.drop_duplicates("query_id")
        .drop_duplicates("dataset")
        .set_index("dataset")[["task_family", "task_subtype"]]
        .to_dict("index")
    )
    assert query_taxonomy == {
        "aime": {"task_family": "math_reasoning", "task_subtype": "competition_math"},
        "gpqa": {"task_family": "science_reasoning", "task_subtype": "graduate_science_qa"},
    }
    assert {"task_family", "task_subtype"}.issubset(prepared.matrices["train"].query_info.columns)


def test_prepare_from_config_applies_domain_map_to_cached_outcomes(tmp_path):
    cache_path = tmp_path / "cache" / "outcomes.csv"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "query_id": "aime:test:0",
                "query_text": "cached aime query",
                "dataset": "aime",
                "domain": "aime",
                "source_split": "test",
                "record_index": 0,
                "model_id": "model_a",
                "quality": 1.0,
                "cost_input": 0.0,
                "cost_output": 0.0,
                "cost_total": 0.0,
                "latency": None,
                "tokens_input": 0,
                "tokens_output": 0,
                "judge": "cached",
                "metadata_json": "{}",
            },
            {
                "query_id": "aime:test:1",
                "query_text": "cached aime query 1",
                "dataset": "aime",
                "domain": "aime",
                "source_split": "test",
                "record_index": 1,
                "model_id": "model_a",
                "quality": 0.0,
                "cost_input": 0.0,
                "cost_output": 0.0,
                "cost_total": 0.0,
                "latency": None,
                "tokens_input": 0,
                "tokens_output": 0,
                "judge": "cached",
                "metadata_json": "{}",
            },
            {
                "query_id": "aime:test:2",
                "query_text": "cached aime query 2",
                "dataset": "aime",
                "domain": "aime",
                "source_split": "test",
                "record_index": 2,
                "model_id": "model_a",
                "quality": 1.0,
                "cost_input": 0.0,
                "cost_output": 0.0,
                "cost_total": 0.0,
                "latency": None,
                "tokens_input": 0,
                "tokens_output": 0,
                "judge": "cached",
                "metadata_json": "{}",
            },
        ]
    ).to_csv(cache_path, index=False)
    config = {
        "run": {"random_seed": 0, "output_dir": str(tmp_path / "out")},
        "data": {
            "source": "llmrouterbench",
            "results_dir": str(tmp_path / "missing"),
            "cache_path": str(cache_path),
            "domain_map": {"aime": "math"},
        },
        "utility": {"lambda_cost": 0.0},
        "split": {"train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
    }

    prepared = prepare_from_config(config)

    assert set(prepared.outcomes["domain"]) == {"math"}


def test_prepare_from_config_applies_task_taxonomy_to_cached_outcomes(tmp_path):
    cache_path = tmp_path / "cache" / "outcomes.csv"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "query_id": f"aime:test:{idx}",
                "query_text": f"cached aime query {idx}",
                "dataset": "aime",
                "domain": "aime",
                "source_split": "test",
                "record_index": idx,
                "model_id": "model_a",
                "quality": float(idx % 2),
                "cost_input": 0.0,
                "cost_output": 0.0,
                "cost_total": 0.0,
                "latency": None,
                "tokens_input": 0,
                "tokens_output": 0,
                "judge": "cached",
                "metadata_json": "{}",
            }
            for idx in range(3)
        ]
    ).to_csv(cache_path, index=False)
    config = {
        "run": {"random_seed": 0, "output_dir": str(tmp_path / "out")},
        "data": {
            "source": "llmrouterbench",
            "results_dir": str(tmp_path / "missing"),
            "cache_path": str(cache_path),
            "task_taxonomy_map": {
                "aime": {"task_family": "math_reasoning", "task_subtype": "competition_math"}
            },
        },
        "utility": {"lambda_cost": 0.0},
        "split": {"train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
    }

    prepared = prepare_from_config(config)

    assert set(prepared.outcomes["task_family"]) == {"math_reasoning"}
    assert set(prepared.outcomes["task_subtype"]) == {"competition_math"}


def test_prepare_from_config_writes_and_reuses_cache(tmp_path):
    root = tmp_path / "bench"
    cache_path = tmp_path / "cache" / "outcomes.csv"
    write_result(root / "aime" / "test" / "model_a" / "20260101_000000.json", "aime", "test", "model_a", [1, 0, 1])
    write_result(root / "aime" / "test" / "model_b" / "20260101_000000.json", "aime", "test", "model_b", [0, 1, 0])
    config = {
        "run": {"random_seed": 0, "output_dir": str(tmp_path / "out")},
        "data": {"source": "llmrouterbench", "results_dir": str(root), "cache_path": str(cache_path)},
        "utility": {"lambda_cost": 0.0},
        "split": {"train_frac": 0.34, "val_frac": 0.33, "test_frac": 0.33},
    }

    first = prepare_from_config(config)
    assert cache_path.exists()
    for file_path in root.rglob("*.json"):
        file_path.unlink()
    second = prepare_from_config(config)
    assert second.outcomes["query_id"].nunique() == first.outcomes["query_id"].nunique()
