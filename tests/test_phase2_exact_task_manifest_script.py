from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "59_exact_task_manifest.py"
    spec = importlib.util.spec_from_file_location("phase2_exact_task_manifest", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exact_task_manifest_script_writes_manifest_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    cache_path = tmp_path / "outcomes.csv"
    rows = []
    for idx, dataset in enumerate(["aime", "math500", "aime", "math500"]):
        for model_id in ["m1", "m2"]:
            rows.append(
                {
                    "query_id": f"{dataset}:test:{idx}",
                    "query_text": f"Math question {idx}",
                    "dataset": dataset,
                    "domain": "math",
                    "source_split": "test",
                    "record_index": idx,
                    "model_id": model_id,
                    "quality": 1.0,
                    "cost_input": 0.0,
                    "cost_output": 0.0,
                    "cost_total": 0.0,
                    "latency": "",
                    "tokens_input": 1,
                    "tokens_output": 1,
                    "judge": "fixture",
                    "metadata_json": json.dumps({"ground_truth": str(idx)}),
                }
            )
    pd.DataFrame(rows).to_csv(cache_path, index=False)
    config_path = tmp_path / "manifest.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: exact_manifest_smoke",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: llmrouterbench",
                f"  cache_path: {cache_path}",
                "  results_dir: unused",
                "  datasets: [aime, math500]",
                "split:",
                "  train_frac: 0.25",
                "  val_frac: 0.25",
                "  test_frac: 0.5",
                "utility:",
                "  lambda_cost: 0.0",
                "phase2_exact_task_manifest:",
                "  datasets: [aime, math500]",
                "  split: test",
                "  max_queries: 2",
            ]
        ),
        encoding="utf-8",
    )

    manifest = module.run(config_path=str(config_path), output_dir=str(out_dir))

    manifest_path = out_dir / "local_exact_task_manifest.csv"
    memo_path = out_dir / "m10_exact_task_manifest_memo.md"
    readme_path = out_dir / "README.md"
    assert len(manifest) == 2
    assert manifest_path.exists()
    assert memo_path.exists()
    assert readme_path.exists()
    saved = pd.read_csv(manifest_path)
    assert set(saved["task_type"]) == {"math"}
    assert "## Phase 2 Exact Task Manifest" in readme_path.read_text(encoding="utf-8")
