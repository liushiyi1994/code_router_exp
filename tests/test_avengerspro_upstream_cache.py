from __future__ import annotations

import json
from pathlib import Path
import sys


def test_avengerspro_simple_cluster_router_uses_embedding_cache_without_service(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    baseline_path = tmp_path / "baseline_scores.json"
    cache_path = tmp_path / "embedding_cache.jsonl"

    _write_jsonl(
        train_path,
        [
            {"query": "alpha prompt", "dataset": "demo", "records": {"m0": 1.0, "m1": 0.0}},
            {"query": "beta prompt", "dataset": "demo", "records": {"m0": 0.0, "m1": 1.0}},
        ],
    )
    _write_jsonl(
        test_path,
        [
            {"query": "alpha heldout", "dataset": "demo", "records": {"m0": 1.0, "m1": 0.0}},
        ],
    )
    baseline_path.write_text(json.dumps({"m0": {"demo": 1.0}, "m1": {"demo": 0.0}}), encoding="utf-8")
    _write_jsonl(
        cache_path,
        [
            {"query": "alpha prompt", "embedding": [1.0, 0.0]},
            {"query": "beta prompt", "embedding": [0.0, 1.0]},
            {"query": "alpha heldout", "embedding": [0.95, 0.05]},
        ],
    )

    package_root = Path("data/raw/external/LLMRouterBench").resolve()
    sys.path.insert(0, str(package_root))
    try:
        from baselines.AvengersPro import simple_cluster_router as module
        from baselines.AvengersPro.config import SimpleClusterConfig

        def fail_generator(*_args, **_kwargs):
            raise AssertionError("cache-only mode should not create an embedding service generator")

        monkeypatch.setattr(module, "create_generator", fail_generator)

        config = SimpleClusterConfig(
            train_data_path=str(train_path),
            test_data_path=str(test_path),
            baseline_scores_path=str(baseline_path),
            embedding_api_key="",
            embedding_cache_path=str(cache_path),
            n_clusters=2,
            max_workers=1,
            cluster_batch_size=4,
        )
        router = module.SimpleClusterRouter(config)
        train_data, test_data = router.load_and_split_data()
        router.build_cluster_model(train_data)
        results = router.evaluate_routing(test_data)

        assert results["total_queries"] == 1
        assert results["routing_details"][0]["selected_models"][0] == "m0"
    finally:
        try:
            sys.path.remove(str(package_root))
        except ValueError:
            pass


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
