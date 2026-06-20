from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "23_stronger_direct_router_probe.py"
    spec = importlib.util.spec_from_file_location("stronger_direct_router_probe", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stronger_direct_router_probe_writes_bounded_probe_outputs(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Probe\n", encoding="utf-8")
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
                "  n_queries: 96",
                "  n_models: 5",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  embedding_dim: 8",
                "  model_ids: [m0, m1, m2, m3, m4]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.06",
                "    m2: 0.07",
                "    m3: 0.08",
                "    m4: 0.09",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "predictability_constrained:",
                "  k: 4",
                "  selected_alpha: 1.0",
                "  beta: 0.0",
                "stronger_direct_router_probe:",
                "  k: 4",
                "  alpha: 1.0",
                "  beta: 0.0",
                "  max_holdout_models: 2",
                "  r_values: [2, 4]",
                "  direct_router_methods: [logistic, svm, knn, mlp, gradient_boosting]",
                "  direct_router_max_iter: 50",
                "  direct_router_knn_k: 1",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_stronger_direct_router_probe.csv"
    memo_path = out_dir / "phase_e_stronger_direct_router_probe_memo.md"
    assert table_path.exists()
    assert memo_path.exists()

    table = pd.read_csv(table_path)
    assert {"routecode_label_calibration", "direct_retraining_budgeted_mlp"}.issubset(set(table["method"]))
    assert "direct_retraining_budgeted_gradient_boosting" in set(table["method"])
    assert set(table["probe_scope"]) == {"bounded_stronger_direct_router_probe"}

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Stronger Direct-Router Probe" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "bounded stronger direct-router probe" in memo
