from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "43_external_blocker_resolution.py"
    spec = importlib.util.spec_from_file_location("external_blocker_resolution", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_external_blocker_resolution_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    pilot_dir = tmp_path / "llmrouterbench_pilot"
    broad_dir = tmp_path / "llmrouterbench_broad20"
    pilot_dir.mkdir()
    broad_dir.mkdir()
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "README.md").write_text("# Results\n", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "check_id": "routellm_bert_cli",
                "status": "blocked",
                "runnable_now": False,
                "blocking_reasons": "missing_bert_checkpoint",
            },
            {
                "check_id": "avengerspro_cli",
                "status": "executed",
                "runnable_now": True,
                "blocking_reasons": "",
            },
        ]
    ).to_csv(pilot_dir / "table_external_command_readiness.csv", index=False)
    pd.DataFrame(
        [
            {
                "check_id": "routerdc_train_cli",
                "status": "blocked",
                "runnable_now": False,
                "blocking_reasons": "missing_routerdc_local_model_checkpoint;missing_python_modules:deepspeed",
            }
        ]
    ).to_csv(broad_dir / "table_external_command_readiness.csv", index=False)

    module.run(
        [
            str(pilot_dir / "table_external_command_readiness.csv"),
            str(broad_dir / "table_external_command_readiness.csv"),
        ],
        output_dir,
    )

    table_path = output_dir / "table_external_blocker_resolution.csv"
    memo_path = output_dir / "phase_e_external_blocker_resolution_memo.md"
    assert table_path.exists()
    assert memo_path.exists()

    table = pd.read_csv(table_path).set_index("check_id")
    assert table.loc["routellm_bert_cli", "missing_checkpoints"] == "missing_bert_checkpoint"
    assert table.loc["routerdc_train_cli", "missing_modules"] == "deepspeed"

    memo = memo_path.read_text(encoding="utf-8")
    assert "Phase E External Blocker Resolution Memo" in memo
    assert "Checkpoint-gated blocked rows: `2`" in memo
    assert "routerdc_train_cli" in memo

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "## External Blocker Resolution" in readme
    assert "table_external_blocker_resolution.csv" in readme
