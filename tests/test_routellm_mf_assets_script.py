from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "15_routellm_mf_assets.py"
    spec = importlib.util.spec_from_file_location("routellm_mf_assets", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mf_assets_script_writes_official_trainer_inputs_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n\n## Next Steps\n\n- old\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 5",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 36",
                "  n_models: 2",
                "  n_domains: 2",
                "  n_route_labels: 4",
                "  embedding_dim: 5",
                "  model_ids: [Qwen3-8B, Qwen2.5-Coder-7B-Instruct]",
                "  model_costs:",
                "    Qwen3-8B: 0.20",
                "    Qwen2.5-Coder-7B-Instruct: 0.10",
                "utility:",
                "  lambda_cost: 0.35",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "external_baselines:",
                "  strong_model: Qwen3-8B",
                "  weak_model: Qwen2.5-Coder-7B-Instruct",
                "  mf_hidden_dim: 8",
                "  mf_num_epochs: 3",
                "  mf_batch_size: 4",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    assets_dir = out_dir / "routellm_mf_assets"
    train_path = assets_dir / "pairwise_train.json"
    test_path = assets_dir / "pairwise_test.json"
    embeddings_path = assets_dir / "prompt_embeddings.npy"
    prompt_index_path = assets_dir / "prompt_index.json"
    train_config_path = assets_dir / "mf_train_config.local.json"
    eval_config_path = assets_dir / "mf_eval_config.local.json"
    embedding_config_path = assets_dir / "embedding_config.local.yaml"
    embedding_cache_path = assets_dir / "embedding_cache.jsonl"
    metadata_path = assets_dir / "metadata.json"
    table_path = out_dir / "table_routellm_mf_assets.csv"
    memo_path = out_dir / "phase_e_routellm_mf_assets_memo.md"
    for path in [
        train_path,
        test_path,
        embeddings_path,
        prompt_index_path,
        train_config_path,
        eval_config_path,
        embedding_config_path,
        embedding_cache_path,
        metadata_path,
        table_path,
        memo_path,
    ]:
        assert path.exists(), path

    train_records = json.loads(train_path.read_text(encoding="utf-8"))
    test_records = json.loads(test_path.read_text(encoding="utf-8"))
    prompt_index = json.loads(prompt_index_path.read_text(encoding="utf-8"))
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    eval_config = json.loads(eval_config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    first_cache_row = json.loads(embedding_cache_path.read_text(encoding="utf-8").splitlines()[0])
    embeddings = np.load(embeddings_path)
    assert train_records
    assert test_records
    assert {row["winner"] for row in train_records} <= {"model_a", "model_b"}
    assert {row["winner"] for row in test_records} <= {"model_a", "model_b", "tie"}
    assert {"idx", "dataset_id", "score_model_a", "score_model_b", "cost_model_a", "cost_model_b"} <= set(
        train_records[0]
    )
    assert embeddings.shape[0] == len(prompt_index)
    assert train_config["json_path"] == str(train_path)
    assert train_config["npy_path"] == str(embeddings_path)
    assert train_config["dim"] == embeddings.shape[1]
    assert train_config["num_epochs"] == 3
    assert train_config["batch_size"] == 4
    assert train_config["device"] == "cpu"
    assert eval_config["mf"]["checkpoint_path"].endswith("mf_model.pt")
    assert eval_config["mf"]["embedding_config_path"].endswith("embedding_config.local.yaml")
    assert eval_config["mf"]["hidden_size"] == embeddings.shape[1]
    assert first_cache_row["prompt"]
    assert first_cache_row["embedding"]
    assert metadata["official_trainer_compatible"] is True
    assert metadata["official_routellm_result"] is False
    assert metadata["eval_config_path"].endswith("mf_eval_config.local.json")
    assert metadata["embedding_cache_path"].endswith("embedding_cache.jsonl")

    table = pd.read_csv(table_path)
    assert set(table["split"]) == {"train", "test", "overall"}
    assert table["split_aligned_with_routecode"].all()
    assert table["official_trainer_compatible"].all()
    assert not table["official_routellm_result"].any()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## RouteLLM MF Trainer Assets" in readme
    assert "not a trained RouteLLM MF result" in readme
    assert "mf_eval_config.local.json" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "ready for the local LLMRouterBench RouteLLM MF trainer" in memo
    assert "cache-backed upstream RouteLLM MF evaluation" in memo
