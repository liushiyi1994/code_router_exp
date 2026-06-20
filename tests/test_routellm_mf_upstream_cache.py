from __future__ import annotations

import json
import sys
from pathlib import Path


def test_routellm_mf_router_loads_embedding_cache_without_generator(tmp_path):
    cache_path = tmp_path / "embedding_cache.jsonl"
    cache_path.write_text(
        json.dumps({"prompt": "hello prompt", "embedding": [0.1, 0.2, 0.3]}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "embedding_config.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                "embedding_model:",
                "  api_model_name: routecode-cache",
                "  name: routecode-cache",
                f"embedding_cache_path: {cache_path}",
            ]
        ),
        encoding="utf-8",
    )

    package_root = Path("data/raw/external/LLMRouterBench").resolve()
    sys.path.insert(0, str(package_root))
    try:
        from baselines.RouteLLM.routers import routers as module
    finally:
        sys.path.remove(str(package_root))

    module.create_generator = None

    embedder, model_name = module.MatrixFactorizationRouter._load_embedding_generator(str(config_path))

    assert model_name == "routecode-cache"
    assert embedder.generate_embedding("hello prompt").embeddings == [0.1, 0.2, 0.3]


def test_routellm_controller_imports_without_litellm_for_mf_eval():
    package_root = Path("data/raw/external/LLMRouterBench").resolve()
    sys.path.insert(0, str(package_root))
    try:
        from baselines.RouteLLM import controller
    finally:
        sys.path.remove(str(package_root))

    assert controller.Controller is not None
