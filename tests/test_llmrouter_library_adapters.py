from __future__ import annotations

from pathlib import Path
import json
import pickle
import sys

import pandas as pd
import torch
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from routecode.eval.llmrouter_library_adapters import (
    evaluate_llmrouter_cli_predictions,
    evaluate_llmrouter_library_adapters,
)
from routecode.matrix import Matrices


def _matrices(query_ids: list[str], utility_rows: list[list[float]]) -> Matrices:
    models = ["m0", "m1"]
    utility = pd.DataFrame(utility_rows, index=query_ids, columns=models)
    quality = utility.copy()
    cost = pd.DataFrame(0.0, index=query_ids, columns=models)
    query_info = pd.DataFrame(
        {
            "query_text": [f"prompt {query_id}" for query_id in query_ids],
            "dataset": ["demo"] * len(query_ids),
            "domain": ["demo"] * len(query_ids),
        },
        index=query_ids,
    )
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=models)


def test_llmrouter_library_adapters_train_upstream_knn_and_svm_without_api_calls(tmp_path):
    train = _matrices(
        ["q0", "q1", "q2", "q3"],
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
    )
    test = _matrices(["q4", "q5"], [[0.85, 0.15], [0.15, 0.85]])
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05], [0.05, 0.95]],
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )

    table = evaluate_llmrouter_library_adapters(
        train,
        test,
        embeddings,
        output_dir=tmp_path,
        llmrouter_root=Path("data/raw/external/LLMRouter"),
        seed=7,
        n_bootstrap=5,
        knn_k=1,
    )

    assert set(table["method"]) == {"llmrouter_library_knn", "llmrouter_library_svm"}
    assert table["routecode_metric_compatible"].all()
    assert table["split_aligned_with_routecode"].all()
    assert table["upstream_training_class_used"].all()
    assert not table["exact_upstream_command"].any()
    assert not table["external_api_calls"].any()
    assert set(table["selected_models"]) == {"m0,m1"}
    assert table.set_index("method").loc["llmrouter_library_knn", "mean_utility"] == 0.85
    assert (tmp_path / "llmrouter_library_adapters" / "knn_model.pkl").exists()
    assert (tmp_path / "llmrouter_library_adapters" / "svm_model.pkl").exists()

    asset_dir = tmp_path / "llmrouter_library_adapters"
    assert (asset_dir / "query_train.jsonl").exists()
    assert (asset_dir / "routing_train.jsonl").exists()
    assert (asset_dir / "query_embeddings.pt").exists()
    assert (asset_dir / "query_embedding_lookup.pt").exists()
    assert (asset_dir / "query_inference_smoke.jsonl").exists()
    assert (asset_dir / "query_inference_test.jsonl").exists()
    assert (asset_dir / "llm_candidates.json").exists()
    assert (asset_dir / "knnrouter_train.yaml").exists()
    assert (asset_dir / "svmrouter_train.yaml").exists()

    embeddings_pt = torch.load(asset_dir / "query_embeddings.pt", map_location="cpu")
    assert tuple(embeddings_pt.shape) == (4, 2)
    embedding_lookup = torch.load(asset_dir / "query_embedding_lookup.pt", map_location="cpu")
    assert set(embedding_lookup) == {
        "prompt q0",
        "prompt q1",
        "prompt q2",
        "prompt q3",
        "prompt q4",
        "prompt q5",
    }
    smoke_queries = [
        json.loads(line) for line in (asset_dir / "query_inference_smoke.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(smoke_queries) == 2
    assert smoke_queries[0]["query"] == "prompt q4"
    test_queries = [
        json.loads(line) for line in (asset_dir / "query_inference_test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["query_id"] for row in test_queries] == ["q4", "q5"]
    assert [row["query"] for row in test_queries] == ["prompt q4", "prompt q5"]
    routing_rows = [
        json.loads(line) for line in (asset_dir / "routing_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(routing_rows) == 8
    assert {"query", "model_name", "performance", "embedding_id"}.issubset(routing_rows[0])
    knn_config = yaml.safe_load((asset_dir / "knnrouter_train.yaml").read_text(encoding="utf-8"))
    assert knn_config["data_path"]["routing_data_train"] == str((asset_dir / "routing_train.jsonl").resolve())
    assert knn_config["data_path"]["query_embedding_lookup"] == str((asset_dir / "query_embedding_lookup.pt").resolve())
    assert knn_config["model_path"]["save_model_path"] == str((asset_dir / "knn_cli_model.pkl").resolve())
    assert knn_config["model_path"]["load_model_path"] == str((asset_dir / "knn_cli_model.pkl").resolve())
    assert knn_config["hparam"]["n_neighbors"] == 1


def test_evaluate_llmrouter_cli_predictions_scores_exact_route_only_output(tmp_path):
    train = _matrices(
        ["q0", "q1", "q2", "q3"],
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
    )
    test = _matrices(["q4", "q5"], [[0.85, 0.15], [0.15, 0.85]])
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05], [0.05, 0.95]],
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )
    prediction_path = tmp_path / "llmrouter_knn_full_predictions.json"
    prediction_path.write_text(
        json.dumps(
            [
                {"success": True, "query": "prompt q4", "model_name": "m0"},
                {"success": True, "query": "prompt q5", "model_name": "m1"},
            ]
        ),
        encoding="utf-8",
    )

    table = evaluate_llmrouter_cli_predictions(
        train,
        test,
        embeddings,
        predictions={"knn": prediction_path},
        seed=3,
        n_bootstrap=5,
    )

    assert list(table["method"]) == ["llmrouter_cli_knn"]
    row = table.iloc[0]
    assert row["mean_utility"] == 0.85
    assert bool(row["routecode_metric_compatible"])
    assert bool(row["exact_upstream_command"])
    assert row["prediction_count"] == 2
    assert row["prediction_source"] == str(prediction_path)


def test_llmrouter_library_adapters_reject_missing_embeddings(tmp_path):
    train = _matrices(["q0", "q1"], [[0.9, 0.1], [0.1, 0.9]])
    test = _matrices(["q2"], [[0.8, 0.2]])
    embeddings = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["q0", "q1"])

    try:
        evaluate_llmrouter_library_adapters(
            train,
            test,
            embeddings,
            output_dir=tmp_path,
            llmrouter_root=Path("data/raw/external/LLMRouter"),
        )
    except ValueError as exc:
        assert "Missing embedding rows for LLMRouter library adapters" in str(exc)
    else:
        raise AssertionError("missing test embeddings should fail")


def test_upstream_llmrouter_knn_and_svm_route_from_embedding_cache_without_longformer(tmp_path, monkeypatch):
    asset_dir = tmp_path / "llmrouter_assets"
    asset_dir.mkdir()

    query_rows = [
        {"id": "q0", "query": "cached prompt a", "task_name": "demo"},
        {"id": "q1", "query": "cached prompt b", "task_name": "demo"},
    ]
    routing_rows = [
        {"query": "cached prompt a", "model_name": "m0", "performance": 1.0, "embedding_id": 0},
        {"query": "cached prompt a", "model_name": "m1", "performance": 0.0, "embedding_id": 0},
        {"query": "cached prompt b", "model_name": "m0", "performance": 0.0, "embedding_id": 1},
        {"query": "cached prompt b", "model_name": "m1", "performance": 1.0, "embedding_id": 1},
    ]
    _write_jsonl(asset_dir / "query_train.jsonl", query_rows)
    _write_jsonl(asset_dir / "routing_train.jsonl", routing_rows)
    torch.save(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32), asset_dir / "query_embeddings.pt")
    torch.save(
        {
            "cached prompt a": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "cached prompt b": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
        asset_dir / "query_embedding_lookup.pt",
    )

    knn = KNeighborsClassifier(n_neighbors=1).fit([[1.0, 0.0], [0.0, 1.0]], ["m0", "m1"])
    svm = SVC(kernel="linear").fit([[1.0, 0.0], [0.0, 1.0]], ["m0", "m1"])
    with (asset_dir / "knn_cli_model.pkl").open("wb") as handle:
        pickle.dump(knn, handle)
    with (asset_dir / "svm_cli_model.pkl").open("wb") as handle:
        pickle.dump(svm, handle)

    knn_config = _write_cache_inference_config(asset_dir / "knnrouter_train.yaml", asset_dir, "knn", "minkowski")
    svm_config = _write_cache_inference_config(asset_dir / "svmrouter_train.yaml", asset_dir, "svm", "linear")

    llmrouter_root = Path("data/raw/external/LLMRouter").resolve()
    sys.path.insert(0, str(llmrouter_root))
    try:
        from llmrouter.models.knnrouter import router as knn_module
        from llmrouter.models.svmrouter import router as svm_module
        from llmrouter.utils import embeddings as embedding_module

        def fail_longformer(_text):
            raise AssertionError("Longformer fallback should not be called for cached query text")

        monkeypatch.setattr(embedding_module, "get_longformer_embedding", fail_longformer)

        knn_router = knn_module.KNNRouter(str(knn_config))
        svm_router = svm_module.SVMRouter(str(svm_config))

        assert knn_router.route_single({"query": "cached prompt a"})["model_name"] == "m0"
        assert svm_router.route_single({"query": "cached prompt b"})["model_name"] == "m1"
    finally:
        try:
            sys.path.remove(str(llmrouter_root))
        except ValueError:
            pass


def _write_cache_inference_config(path: Path, asset_dir: Path, router: str, metric_or_kernel: str) -> Path:
    model_name = "knn" if router == "knn" else "svm"
    hparam = (
        {"n_neighbors": 1, "metric": metric_or_kernel}
        if router == "knn"
        else {"kernel": metric_or_kernel, "gamma": "scale"}
    )
    config = {
        "data_path": {
            "query_data_train": str((asset_dir / "query_train.jsonl").resolve()),
            "query_embedding_data": str((asset_dir / "query_embeddings.pt").resolve()),
            "query_embedding_lookup": str((asset_dir / "query_embedding_lookup.pt").resolve()),
            "routing_data_train": str((asset_dir / "routing_train.jsonl").resolve()),
        },
        "model_path": {
            "ini_model_path": "",
            "save_model_path": str((asset_dir / f"{model_name}_cli_model.pkl").resolve()),
            "load_model_path": str((asset_dir / f"{model_name}_cli_model.pkl").resolve()),
        },
        "hparam": hparam,
        "metric": {"weights": {"performance": 1}},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
