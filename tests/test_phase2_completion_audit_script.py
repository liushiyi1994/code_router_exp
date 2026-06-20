from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "69_phase2_completion_audit.py"
    spec = importlib.util.spec_from_file_location("phase2_completion_audit_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase2_completion_audit_script_writes_table_memo_and_report(tmp_path):
    module = _load_script()
    phase2 = tmp_path / "results/phase2"
    phase2.mkdir(parents=True)
    (phase2 / "PHASE2_EVIDENCE_REPORT.md").write_text("# Phase 2 Evidence Report\n", encoding="utf-8")

    paths = module.run(root=tmp_path, output_dir=phase2)

    assert Path(paths["table"]).exists()
    assert Path(paths["memo"]).exists()
    assert "Phase 2 Completion Audit" in Path(paths["memo"]).read_text(encoding="utf-8")
    report = Path(paths["report"]).read_text(encoding="utf-8")
    assert "## Phase 2 Completion Audit" in report
    assert "table_phase2_completion_audit.csv" in report
