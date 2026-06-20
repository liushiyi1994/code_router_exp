from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "52_probe_collection.py"
    spec = importlib.util.spec_from_file_location("phase2_probe_collection", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_collection_script_writes_probe_features_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    out_dir.mkdir()
    outcomes_path = out_dir / "local_model_outcomes.parquet"
    pd.DataFrame(
        [
            {
                "query_id": "q1",
                "query_text": "What is 2+3?",
                "dataset": "gsm8k_smoke",
                "domain": "math",
                "model_id": "dry_run_model",
                "model_revision": "dry-run",
                "prompt_template": "math_answer_v1",
                "generation_params_json": '{"max_tokens": 16, "temperature": 0.0}',
                "raw_output": "Final answer: 5",
                "parsed_answer": "5",
                "gold_answer": "5",
                "quality": 1.0,
                "cost_proxy": 0.003,
                "latency_sec": 0.01,
                "tokens_input": 12,
                "tokens_output": 3,
                "error_type": "",
                "error_message": "",
                "created_at": "2026-06-17T00:00:00+00:00",
            },
            {
                "query_id": "q2",
                "query_text": "Choose the answer.",
                "dataset": "mmlu_smoke",
                "domain": "broad_knowledge",
                "model_id": "dry_run_model",
                "model_revision": "dry-run",
                "prompt_template": "multiple_choice_letter_v1",
                "generation_params_json": '{"max_tokens": 16, "temperature": 0.0}',
                "raw_output": "C",
                "parsed_answer": "C",
                "gold_answer": "C",
                "quality": 1.0,
                "cost_proxy": 0.001,
                "latency_sec": 0.02,
                "tokens_input": 10,
                "tokens_output": 1,
                "error_type": "",
                "error_message": "",
                "created_at": "2026-06-17T00:00:01+00:00",
            },
        ]
    ).to_parquet(outcomes_path, index=False)

    features = module.run(outcomes_path=str(outcomes_path), output_dir=str(out_dir))

    features_path = out_dir / "probe_features.parquet"
    memo_path = out_dir / "m3_probe_collection_memo.md"
    readme_path = out_dir / "README.md"
    assert features_path.exists()
    assert memo_path.exists()
    assert readme_path.exists()
    assert len(features) == 2

    saved = pd.read_parquet(features_path)
    assert len(saved) == 2
    assert set(saved["probe_type"]) == {"local_answer_probe"}
    assert "## Phase 2 Probe Features" in readme_path.read_text(encoding="utf-8")
    memo = memo_path.read_text(encoding="utf-8")
    assert "not evidence that probes close the observability gap" in memo
    assert "This M3 step collects" in memo
    assert "This M3 dry-run" not in memo
