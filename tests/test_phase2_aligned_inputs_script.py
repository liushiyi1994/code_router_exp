from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "56_aligned_offline_probe_inputs.py"
    spec = importlib.util.spec_from_file_location("phase2_aligned_offline_probe_inputs", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aligned_offline_probe_inputs_script_writes_all_policy_inputs(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: aligned_smoke",
                "  output_dir: " + str(out_dir),
                "  random_seed: 0",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 80",
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
            ]
        ),
        encoding="utf-8",
    )

    paths = module.run(config_path=str(config_path), output_dir=str(out_dir), n_neighbors=3)

    expected = {
        "probe_features",
        "state_targets",
        "query_features",
        "before_beliefs",
        "after_beliefs",
        "state_model_utility",
        "query_model_utility",
        "probe_cost",
        "predicted_gain",
    }
    assert expected.issubset(paths)
    for path in paths.values():
        assert Path(path).exists()
    state_targets = pd.read_csv(paths["state_targets"])
    assert {"train", "test"}.issubset(set(state_targets["split"]))
    assert (out_dir / "m7_aligned_offline_probe_inputs_memo.md").exists()
    phase2_readme = Path("results/phase2/README.md")
    if phase2_readme.exists():
        assert str(out_dir) not in phase2_readme.read_text(encoding="utf-8")
