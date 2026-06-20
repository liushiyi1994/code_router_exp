from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "50_observability_gap_strong_encoders.py"
    spec = importlib.util.spec_from_file_location("phase2_observability_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_phase1_tables(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"method": "best_single", "mean_utility": 0.50, "K": float("nan")},
            {"method": "query_oracle", "mean_utility": 0.90, "K": float("nan")},
        ]
    ).to_csv(path / "table_recovered_gap.csv", index=False)
    pd.DataFrame(
        [
            {"method": "flat_routecode_utility_oracle", "mean_utility": 0.86, "K": 16},
            {"method": "flat_routecode_logistic_label_predictor", "mean_utility": 0.62, "K": 16},
            {"method": "d2_joint_oracle_labels", "mean_utility": 0.74, "K": 16, "alpha": 3.0},
            {
                "method": "d2_embedding_centroid",
                "mean_utility": 0.72,
                "K": 16,
                "alpha": 3.0,
                "label_accuracy": 0.95,
            },
        ]
    ).to_csv(path / "table_predictability_constrained.csv", index=False)


def test_phase2_observability_script_writes_m0_outputs(tmp_path):
    module = _load_script()
    result_dir = tmp_path / "phase1_pilot"
    _write_phase1_tables(result_dir)
    out_dir = tmp_path / "phase2"

    module.run([result_dir], out_dir)

    table_path = out_dir / "table_observability_strong_encoders.csv"
    memo_path = out_dir / "m0_previous_findings_recap.md"
    fig_path = out_dir / "fig_observability_gap.pdf"
    readme_path = out_dir / "README.md"
    assert table_path.exists()
    assert memo_path.exists()
    assert fig_path.exists()
    assert readme_path.exists()

    table = pd.read_csv(table_path)
    assert set(table["result_id"]) == {"phase1_pilot"}
    assert "state_observability_gap" in table.columns
    assert "flat_routecode_logistic_label_predictor" in set(table["comparison"])
    assert "Phase 2 M0 Previous Findings Recap" in memo_path.read_text(encoding="utf-8")
    assert "Strong encoders have not been run by this script" in memo_path.read_text(encoding="utf-8")
    assert "## Phase 2 Observability Gap" in readme_path.read_text(encoding="utf-8")


def test_phase2_observability_script_appends_injected_strong_encoder_rows(tmp_path):
    module = _load_script()
    result_dir = tmp_path / "phase1_pilot"
    _write_phase1_tables(result_dir)
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 4",
                f"  output_dir: {tmp_path / 'ignored'}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 24",
                "  n_models: 3",
                "  n_domains: 2",
                "  n_route_labels: 2",
                "  embedding_dim: 4",
                "utility:",
                "  lambda_cost: 0.0",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "transformer_backbones:",
                "  requested_model_ids:",
                "    - local/test-encoder",
                "phase2_observability:",
                "  k: 2",
                "  alpha: 0.0",
                "  state_families: [flat_routecode]",
                "  state_predictors: [centroid]",
                "  knn_k: 1",
                "bootstrap:",
                "  n_bootstrap: 5",
            ]
        ),
        encoding="utf-8",
    )
    readiness = pd.DataFrame(
        [
            {
                "model_id": "local/test-encoder",
                "cache_status": "cached",
                "runnable_as_encoder_baseline": True,
                "reason": "cached_encoder_candidate",
                "architecture": "BertModel",
                "model_type": "bert",
                "hidden_size": 4,
                "size_gb": 0.01,
                "local_path": "/tmp/test-encoder",
            }
        ]
    )

    def provider(_row: pd.Series, query_info: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[float(idx % 2), float((idx + 1) % 2), 0.0, 0.0] for idx in range(len(query_info))],
            index=query_info.index,
        )

    module.run(
        [result_dir],
        out_dir,
        config_paths=[config_path],
        strong_readiness_table=readiness,
        strong_embedding_provider=provider,
    )

    table = pd.read_csv(out_dir / "table_observability_strong_encoders.csv")
    strong = table[table["evidence_source"] == "phase2_strong_encoder_state_predictor"]
    assert not strong.empty
    assert set(strong["strong_encoder_status"]) == {"executed"}
    assert set(strong["routing_invariant"]) == {"query_to_state_to_model"}
    assert "Strong encoder rows" in (out_dir / "m0_previous_findings_recap.md").read_text(encoding="utf-8")
