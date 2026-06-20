from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "57_aligned_local_probe_collection.py"
    spec = importlib.util.spec_from_file_location("phase2_aligned_local_probe_collection", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aligned_local_probe_script_writes_dry_run_features_logs_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: aligned_local_probe_smoke",
                "  output_dir: " + str(out_dir),
                "  random_seed: 0",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 50",
                "  n_models: 4",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  residual_scale: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "utility:",
                "  lambda_cost: 0.0",
                "predictability_constrained:",
                "  k: 4",
                "  selected_alpha: 1.0",
                "  beta: 0.0",
                "  max_iter: 10",
                "  refinement_iter: 2",
                "phase2_aligned_local_probe:",
                "  dry_run: true",
                "  model_ids: [dry_probe]",
                "  max_queries: 6",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 32",
            ]
        ),
        encoding="utf-8",
    )

    features = module.run(config_path=str(config_path), output_dir=str(out_dir))

    features_path = out_dir / "aligned_local_probe_features.parquet"
    raw_path = out_dir / "aligned_local_probe_raw_outputs.jsonl"
    metadata_path = out_dir / "aligned_local_probe_run_metadata.json"
    readme_path = out_dir / "README.md"
    assert len(features) == 6
    assert features_path.exists()
    assert raw_path.exists()
    assert metadata_path.exists()
    assert readme_path.exists()
    saved = pd.read_parquet(features_path)
    assert set(saved["probe_type"]) == {"aligned_local_confidence_probe"}
    assert saved["query_id"].is_unique
    assert "## Phase 2 Aligned Local Probes" in readme_path.read_text(encoding="utf-8")
