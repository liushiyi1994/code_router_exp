from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "41_paper_evidence_summary.py"
    spec = importlib.util.spec_from_file_location("paper_evidence_summary_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paper_evidence_summary_script_writes_outputs_and_updates_indexes(tmp_path):
    module = _load_script()
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "README.md").write_text("# RouteCode Results\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "claim_id": "small_inferred_labels",
                "claim": "Small inferred route labels recover most routing performance.",
                "global_status": "not_supported",
                "best_primary_value": 0.34,
                "worst_primary_value": 0.09,
                "best_result_id": "pilot",
                "evidence_summary": "pilot: not_supported",
                "interpretation": "Do not claim this.",
            },
            {
                "claim_id": "low_rate_oracle_codes",
                "claim": "Useful low-rate utility route codes exist.",
                "global_status": "diagnostic_supported",
                "best_primary_value": 1.0,
                "worst_primary_value": 0.95,
                "best_result_id": "pilot",
                "evidence_summary": "pilot: diagnostic_supported",
                "interpretation": "Use diagnostic framing.",
            }
        ]
    ).to_csv(output_dir / "table_claim_status_global.csv", index=False)
    readiness_path = tmp_path / "pilot" / "table_external_command_readiness.csv"
    readiness_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "check_id": "routellm_bert_cli",
                "status": "blocked",
                "runnable_now": False,
                "exact_upstream_command": False,
                "routecode_metric_compatible": False,
                "blocking_reasons": "missing_bert_checkpoint",
                "execution_evidence": "",
            }
        ]
    ).to_csv(readiness_path, index=False)
    pd.DataFrame(
        [
            {
                "phase_id": "phase_a_synthetic_sanity",
                "status": "complete",
                "notes": "Required artifacts are present.",
            },
            {
                "phase_id": "phase_e_external_methods",
                "status": "blocked",
                "notes": "Blocked external rows: routellm_bert_cli.",
            },
        ]
    ).to_csv(output_dir / "table_research_flow_completion.csv", index=False)
    paper_notes_path = tmp_path / "paper_notes.md"

    module.run(output_dir, [readiness_path], paper_notes_path)

    table_path = output_dir / "table_paper_evidence_summary.csv"
    memo_path = output_dir / "phase_h_paper_evidence_summary.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert "paper_direction" in set(table["section"])
    assert "external_baselines" in set(table["section"])
    assert "information_frontier_diagnostic" in set(table["status"])
    assert "Paper Evidence Summary" in memo_path.read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "## Paper Evidence Summary" in readme
    assert "../paper_notes.md" in readme
    paper_notes = paper_notes_path.read_text(encoding="utf-8")
    assert "RouteCode Paper Notes" in paper_notes
    assert "information-frontier" in paper_notes
    assert "## Research Flow Completion" in paper_notes
    assert "Complete phases: `1`" in paper_notes
    assert "Blocked phases: `1` (`phase_e_external_methods`)" in paper_notes
