from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "14_routellm_pairwise_alignment.py"
    spec = importlib.util.spec_from_file_location("routellm_pairwise_alignment", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pairwise_alignment_script_writes_substrate_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n\n## Next Steps\n\n- old\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 3",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 30",
                "  n_models: 3",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  embedding_dim: 6",
                "  model_ids: [tiny_cheap, general_8b, frontier_expensive]",
                "  model_costs:",
                "    tiny_cheap: 0.04",
                "    general_8b: 0.11",
                "    frontier_expensive: 0.68",
                "utility:",
                "  lambda_cost: 0.35",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "external_baselines:",
                "  strong_model: frontier_expensive",
                "  weak_model: tiny_cheap",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    pairwise_dir = out_dir / "routellm_pairwise"
    train_path = pairwise_dir / "pairwise_train.json"
    test_path = pairwise_dir / "pairwise_test.json"
    metadata_path = pairwise_dir / "metadata.json"
    table_path = out_dir / "table_routellm_pairwise_alignment.csv"
    memo_path = out_dir / "phase_e_routellm_pairwise_alignment_memo.md"
    assert train_path.exists()
    assert test_path.exists()
    assert metadata_path.exists()
    assert table_path.exists()
    assert memo_path.exists()

    train_records = json.loads(train_path.read_text(encoding="utf-8"))
    test_records = json.loads(test_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert train_records
    assert test_records
    assert {row["query_id"] for row in train_records}.isdisjoint({row["query_id"] for row in test_records})
    assert {row["winner"] for row in train_records + test_records} <= {"model_a", "model_b", "tie"}
    assert train_records[0]["model_a"] == "frontier_expensive"
    assert train_records[0]["model_b"] == "tiny_cheap"
    assert train_records[0]["prompt"].startswith("Synthetic ")
    assert metadata["split_aligned_with_routecode"] is True
    assert metadata["official_routellm_result"] is False

    table = pd.read_csv(table_path)
    assert set(table["split"]) == {"train", "test", "overall"}
    assert (table["strong_model"] == "frontier_expensive").all()
    assert (table["weak_model"] == "tiny_cheap").all()
    assert table["split_aligned_with_routecode"].all()
    assert not table["official_routellm_result"].any()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## RouteLLM Pairwise Alignment Substrate" in readme
    assert "not an official RouteLLM MF/BERT result" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "official RouteLLM evaluation remains incomplete" in memo
