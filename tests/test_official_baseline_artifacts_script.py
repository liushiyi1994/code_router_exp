from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "12_official_baseline_artifacts.py"
    spec = importlib.util.spec_from_file_location("official_baseline_artifacts", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_official_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n\n## Next Steps\n\n- old\n", encoding="utf-8")
    results_dir = tmp_path / "official" / "RouteLLM" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "mf_results_seed42.json").write_text(
        json.dumps(
            {
                "total": 10,
                "selection_accuracy": 0.7,
                "routing_accuracy": 0.6,
                "total_cost": 1.25,
                "datasets": {
                    "aime": {
                        "total": 2,
                        "selection_accuracy": 0.5,
                        "routing_accuracy": 1.0,
                        "total_cost": 0.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "mf_selection_accuracy_by_seed.csv").write_text(
        "seed,aime,sample_avg\n42,50.0,70.0\n",
        encoding="utf-8",
    )
    (results_dir / "mf_total_cost_by_seed.csv").write_text(
        "seed,aime,total_cost\n42,0.2,1.25\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: llmrouterbench",
                "external_baselines:",
                f"  official_routellm_results_dir: {results_dir}",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_official_external_artifacts.csv"
    memo_path = out_dir / "phase_e_official_baseline_artifacts_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert set(table["scope"]) == {"overall", "dataset"}
    assert (table["baseline_family"] == "official_external_artifact").all()
    assert not table["split_aligned_with_routecode"].any()
    assert not table["routecode_metric_compatible"].any()
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Official External Baseline Artifacts" in readme
    assert "not RouteCode split-aligned" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "not evaluated on the RouteCode train/test split" in memo
