from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from routecode.eval.external_baseline_assets import (
    build_external_baseline_assets,
    summarize_external_baseline_assets,
    write_external_baseline_assets,
)
from routecode.eval.external_command_readiness import inspect_external_command_readiness
from routecode.matrix import Matrices


def _matrices(query_ids: list[str], split: str) -> Matrices:
    models = ["m0", "m1"]
    quality = pd.DataFrame(
        [[0.9, 0.2], [0.3, 0.8], [0.7, 0.6]][: len(query_ids)],
        index=pd.Index(query_ids, name="query_id"),
        columns=models,
    )
    cost = pd.DataFrame(
        [[0.1, 0.3], [0.1, 0.3], [0.1, 0.3]][: len(query_ids)],
        index=pd.Index(query_ids, name="query_id"),
        columns=models,
    )
    query_info = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "query_text": f"Question {query_id}",
                "dataset": "demo",
                "domain": "reasoning",
                "task_family": "synthetic",
                "split": split,
            }
            for query_id in query_ids
        ]
    ).set_index("query_id")
    return Matrices(
        quality=quality,
        cost=cost,
        utility=quality - 0.1 * cost,
        query_info=query_info,
        model_ids=models,
    )


def test_build_and_write_external_baseline_assets_removes_missing_asset_readiness_blockers(tmp_path):
    matrices = {
        "train": _matrices(["q0", "q1"], "train"),
        "val": _matrices(["q2"], "val"),
        "test": _matrices(["q3"], "test"),
    }
    embeddings = pd.DataFrame(
        np.arange(16, dtype=float).reshape(4, 4),
        index=pd.Index(["q0", "q1", "q2", "q3"], name="query_id"),
    )

    assets = build_external_baseline_assets(matrices, embeddings)
    written = write_external_baseline_assets(assets, tmp_path / "results/run")
    summary = summarize_external_baseline_assets(assets)

    assert written.frugalgpt_train_path.exists()
    assert written.frugalgpt_test_path.exists()
    assert written.embedllm_train_path.exists()
    assert written.embedllm_test_path.exists()
    assert written.embedllm_smoke_train_path.exists()
    assert written.embedllm_smoke_test_path.exists()
    assert written.embedllm_question_embeddings_path.exists()
    assert written.embedllm_mf_question_embeddings_path.exists()
    assert written.best_route_train_path.exists()
    assert written.best_route_validation_path.exists()
    assert written.best_route_test_path.exists()
    assert written.routerdc_train_path.exists()
    assert written.routerdc_test_path.exists()
    assert written.routerdc_final_eval_path.exists()
    assert written.modelsat_train_path.exists()
    assert written.modelsat_validation_path.exists()
    assert written.modelsat_ood_path.exists()
    assert written.modelsat_model_description_path.exists()

    assert set(summary["asset_family"]) == {"frugalgpt", "embedllm", "best_route", "routerdc", "modelsat"}
    assert summary.set_index("asset_family").loc["frugalgpt", "train_records"] == 2
    assert summary.set_index("asset_family").loc["embedllm", "train_records"] == 4
    assert summary.set_index("asset_family").loc["best_route", "validation_records"] == 1

    first_frugal = json.loads(written.frugalgpt_train_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_frugal["query"] == "Question q0"
    assert first_frugal["records"] == {"m0": 0.9, "m1": 0.2}
    assert first_frugal["usages"]["m0"]["cost"] == 0.1

    embedllm_train = pd.read_csv(written.embedllm_train_path)
    assert set(["model_id", "model_name", "prompt_id", "prompt", "label", "query_id"]).issubset(
        embedllm_train.columns
    )
    assert embedllm_train["model_id"].tolist() == [0, 1, 0, 1]
    smoke_train = pd.read_csv(written.embedllm_smoke_train_path)
    smoke_test = pd.read_csv(written.embedllm_smoke_test_path)
    assert smoke_train["prompt_id"].nunique() == 2
    assert smoke_test["prompt_id"].nunique() == 1
    assert smoke_train["model_id"].tolist() == [0, 1, 0, 1]
    assert smoke_test["model_id"].tolist() == [0, 1]
    mf_question_embeddings = torch.load(written.embedllm_mf_question_embeddings_path)
    assert tuple(mf_question_embeddings.shape) == (4, 3584)
    assert torch.allclose(mf_question_embeddings[:, :4], torch.tensor(embeddings.to_numpy(dtype=np.float32)))
    assert torch.count_nonzero(mf_question_embeddings[:, 4:]) == 0

    best_route_row = json.loads(written.best_route_train_path.read_text(encoding="utf-8").splitlines()[0])
    assert best_route_row["instruction"] == "Question q0"
    assert best_route_row["candidates"][0]["scores"]["quality"] == 0.9
    assert best_route_row["candidates"][0]["token_num_prompt"] == 1

    routerdc_records = json.loads(written.routerdc_train_path.read_text(encoding="utf-8"))
    assert routerdc_records[0]["question"] == "Question q0"
    assert routerdc_records[0]["scores"] == {"m0": 0.9, "m1": 0.2}
    assert "cluster_id" in routerdc_records[0]

    modelsat_records = json.loads(written.modelsat_train_path.read_text(encoding="utf-8"))
    assert modelsat_records[0]["query"] == "Question q0"
    assert modelsat_records[0]["is_correct_sc"]["m0"] is True
    descriptions = json.loads(written.modelsat_model_description_path.read_text(encoding="utf-8"))
    assert descriptions["m0"]["model"] == "m0"

    baseline_root = tmp_path / "data/raw/external/LLMRouterBench/baselines"
    for path in [
        baseline_root / "FrugalGPT/train_router_from_results.py",
        baseline_root / "EmbedLLM/algorithm/knn.py",
        baseline_root / "EmbedLLM/algorithm/mf.py",
        baseline_root / "Best-route-llm/train_router.py",
        baseline_root / "RouterDC/train_router_mdeberta_7b.py",
        baseline_root / "MODEL-SAT/model_sat_train.py",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# entrypoint\n", encoding="utf-8")

    readiness = inspect_external_command_readiness(
        tmp_path,
        result_dir=tmp_path / "results/run",
        module_availability={
            "deepspeed": False,
            "llm_blender": False,
            "nltk": False,
            "sentence_transformers": False,
            "sklearn": True,
            "torch": True,
            "transformers": True,
            "wandb": False,
        },
        env={},
    ).set_index("check_id")

    assert "missing_frugalgpt_split_aligned_train_jsonl" not in readiness.loc[
        "frugalgpt_local_scorer_cli", "blocking_reasons"
    ]
    assert "missing_embedllm_train_csv" not in readiness.loc["embedllm_knn_cli", "blocking_reasons"]
    assert "missing_embedllm_question_embeddings" not in readiness.loc["embedllm_mf_cli", "blocking_reasons"]
    assert "missing_best_route_train_data" not in readiness.loc["best_route_train_cli", "blocking_reasons"]
    assert "missing_routerdc_train_data" not in readiness.loc["routerdc_train_cli", "blocking_reasons"]
    assert "missing_modelsat_train_data" not in readiness.loc["modelsat_train_cli", "blocking_reasons"]
    assert "missing_local_encoder_checkpoint" in readiness.loc["frugalgpt_local_scorer_cli", "blocking_reasons"]
    assert "missing_best_route_local_model_checkpoint" in readiness.loc[
        "best_route_train_cli", "blocking_reasons"
    ]


def test_external_baseline_assets_reject_missing_embeddings():
    matrices = {"train": _matrices(["q0"], "train"), "test": _matrices(["q1"], "test")}
    embeddings = pd.DataFrame([[1.0, 0.0]], index=pd.Index(["q0"], name="query_id"))

    try:
        build_external_baseline_assets(matrices, embeddings)
    except ValueError as exc:
        assert "Missing embedding rows for external baseline assets" in str(exc)
    else:
        raise AssertionError("missing embeddings should fail")
