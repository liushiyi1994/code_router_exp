from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "42_research_flow_completion_audit.py"
    spec = importlib.util.spec_from_file_location("research_flow_completion_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_research_flow_completion_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "README.md").write_text("# RouteCode Results\n", encoding="utf-8")

    module.run(tmp_path, output_dir)

    table_path = output_dir / "table_research_flow_completion.csv"
    memo_path = output_dir / "phase_h_research_flow_completion_audit.md"
    assert table_path.exists()
    assert memo_path.exists()
    memo = memo_path.read_text(encoding="utf-8")
    assert "Research Flow Completion Audit" in memo
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "## Research Flow Completion Audit" in readme
    assert "table_research_flow_completion.csv" in readme
