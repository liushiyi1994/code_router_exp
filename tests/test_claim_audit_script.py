from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "34_claim_audit.py"
    spec = importlib.util.spec_from_file_location("claim_audit_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_claim_audit_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    (out_dir / "table_predictability_constrained.csv").write_text(
        "\n".join(
            [
                "method,recovered_gap_vs_oracle,utility_ci_low",
                "best_single,0.0,0.5",
                "d2_embedding_centroid,0.09,0.6",
                "d2_joint_oracle_labels,0.96,0.8",
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "table_split_rank_correlation.csv").write_text(
        "\n".join(
            [
                "scenario,rank_correlation_vs_random",
                "leave_dataset_out:mbpp,0.12",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  output_dir: {out_dir}",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_claim_status.csv"
    memo_path = out_dir / "phase_h_claim_status_memo.md"
    assert table_path.exists()
    assert memo_path.exists()

    table = pd.read_csv(table_path)
    assert "small_inferred_labels" in set(table["claim_id"])
    assert "not_supported" in set(table["status"])
    assert "diagnostic_supported" in set(table["status"])

    memo = memo_path.read_text(encoding="utf-8")
    assert "Phase H Claim Status Memo" in memo
    assert "Do not use unsupported claims" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Phase H Claim Audit" in readme
