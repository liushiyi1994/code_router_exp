from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "53_probe_signal_analysis.py"
    spec = importlib.util.spec_from_file_location("phase2_probe_signal_analysis", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_signal_analysis_script_writes_blocker_table_figure_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()
    probe_features_path = out_dir / "probe_features.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "probe_q0",
                "self_confidence": float("nan"),
                "agreement_score": 1.0,
                "knn_label_entropy": float("nan"),
                "knn_winner_entropy": float("nan"),
                "latency_sec": 0.01,
                "input_tokens": 10,
                "output_tokens": 1,
                "probe_cost_proxy": 0.001,
                "error_type": "",
            },
            {
                "query_id": "probe_q1",
                "self_confidence": float("nan"),
                "agreement_score": 1.0,
                "knn_label_entropy": float("nan"),
                "knn_winner_entropy": float("nan"),
                "latency_sec": 0.01,
                "input_tokens": 10,
                "output_tokens": 1,
                "probe_cost_proxy": 0.001,
                "error_type": "",
            },
        ]
    ).to_parquet(probe_features_path, index=False)

    table = module.run(probe_features_path=str(probe_features_path), output_dir=str(out_dir))

    table_path = out_dir / "table_probe_signal_analysis.csv"
    figure_path = out_dir / "fig_probe_signal_gain.pdf"
    memo_path = out_dir / "m4_probe_signal_analysis_memo.md"
    readme_path = out_dir / "README.md"
    assert table_path.exists()
    assert figure_path.exists()
    assert memo_path.exists()
    assert readme_path.exists()
    assert set(table["status"]) == {"blocked_missing_state_targets"}
    assert "## Phase 2 Probe Signal Analysis" in readme_path.read_text(encoding="utf-8")
    assert "cannot support probe-signal claims" in memo_path.read_text(encoding="utf-8")
