from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from routecode.config import load_config
from routecode.eval.benchmark_coverage import (
    build_broad_coverage_candidates,
    scan_llmrouterbench_coverage,
    summarize_dataset_coverage,
)


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "20_benchmark_coverage.py"
    spec = importlib.util.spec_from_file_location("benchmark_coverage", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_result(root: Path, dataset: str, split: str, model: str, stamp: str, indices: list[int]) -> None:
    model_dir = root / dataset / split / model
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset,
        "split": split,
        "model_name": model,
        "records": [
            {
                "index": index,
                "origin_query": f"{dataset} query {index}",
                "score": float(index % 2),
                "cost": 0.0,
            }
            for index in indices
        ],
    }
    (model_dir / f"{dataset}-{split}-{model}-{stamp}.json").write_text(json.dumps(payload), encoding="utf-8")


def _fixture_results(root: Path) -> None:
    _write_result(root, "mathset", "test", "m0", "20250101_000000", [0])
    _write_result(root, "mathset", "test", "m0", "20250102_000000", [0, 1])
    _write_result(root, "mathset", "test", "m1", "20250102_000000", [0, 1])
    _write_result(root, "mathset", "test", "m2", "20250102_000000", [0])
    _write_result(root, "codeset", "test", "m0", "20250102_000000", [0])
    _write_result(root, "codeset", "test", "m1", "20250102_000000", [0])
    _write_result(root, "codeset", "test", "m2", "20250102_000000", [0])


def test_scan_llmrouterbench_coverage_keeps_latest_file_per_dataset_split_model(tmp_path):
    root = tmp_path / "bench"
    _fixture_results(root)

    coverage = scan_llmrouterbench_coverage(root)

    assert len(coverage) == 6
    latest_m0 = coverage[(coverage["dataset"] == "mathset") & (coverage["model_id"] == "m0")].iloc[0]
    assert latest_m0["record_count"] == 2
    assert latest_m0["record_indices"] == "0;1"


def test_summarize_dataset_coverage_adds_taxonomy_and_model_counts(tmp_path):
    root = tmp_path / "bench"
    _fixture_results(root)
    coverage = scan_llmrouterbench_coverage(root)

    summary = summarize_dataset_coverage(
        coverage,
        domain_map={"mathset": "math"},
        taxonomy_map={"mathset": {"task_family": "math_reasoning", "task_subtype": "toy_math"}},
    )
    by_dataset = summary.set_index("dataset")

    assert by_dataset.loc["mathset", "model_count"] == 3
    assert by_dataset.loc["mathset", "domain"] == "math"
    assert by_dataset.loc["mathset", "task_family"] == "math_reasoning"
    assert bool(by_dataset.loc["mathset", "has_taxonomy"])
    assert not bool(by_dataset.loc["codeset", "has_taxonomy"])


def test_build_broad_coverage_candidates_counts_complete_query_intersections(tmp_path):
    root = tmp_path / "bench"
    _fixture_results(root)
    coverage = scan_llmrouterbench_coverage(root)

    candidates = build_broad_coverage_candidates(coverage, model_counts=[2, 3])
    by_count = candidates.set_index("model_count")

    assert by_count.loc[2, "models"] == "m0;m1"
    assert by_count.loc[2, "dataset_count"] == 2
    assert by_count.loc[2, "complete_query_count"] == 3
    assert by_count.loc[3, "models"] == "m0;m1;m2"
    assert by_count.loc[3, "complete_query_count"] == 2


def test_benchmark_coverage_script_writes_tables_memo_and_readme(tmp_path):
    module = _load_script()
    results_dir = tmp_path / "bench"
    _fixture_results(results_dir)
    out_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  output_dir: {out_dir}",
                "data:",
                f"  results_dir: {results_dir}",
                "  domain_map:",
                "    mathset: math",
                "  task_taxonomy_map:",
                "    mathset:",
                "      task_family: math_reasoning",
                "      task_subtype: toy_math",
                "benchmark_coverage:",
                "  model_counts: [2, 3]",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    coverage = pd.read_csv(out_dir / "table_benchmark_file_coverage.csv")
    datasets = pd.read_csv(out_dir / "table_benchmark_dataset_coverage.csv")
    candidates = pd.read_csv(out_dir / "table_broad_coverage_candidates.csv")
    assert len(coverage) == 6
    assert set(datasets["dataset"]) == {"mathset", "codeset"}
    assert set(candidates["model_count"]) == {2, 3}
    assert (out_dir / "phase_g_benchmark_coverage_memo.md").exists()
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Benchmark Coverage Audit" in readme


def test_broad_llmrouterbench_config_taxonomy_covers_local_coverage_datasets():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "llmrouterbench.yaml")
    datasets = set(pd.read_csv(root / "results" / "llmrouterbench_pilot" / "table_benchmark_dataset_coverage.csv")["dataset"])
    domain_map = set(config["data"]["domain_map"])
    taxonomy_map = set(config["data"]["task_taxonomy_map"])

    assert datasets <= domain_map
    assert datasets <= taxonomy_map


def test_broad20_config_matches_audited_20_model_rectangle():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "llmrouterbench_broad20.yaml")
    candidate = pd.read_csv(root / "results" / "llmrouterbench_pilot" / "table_broad_coverage_candidates.csv")
    row = candidate[candidate["model_count"].eq(20)].iloc[0]
    expected_datasets = row["datasets"].split(";")
    expected_models = row["models"].split(";")

    assert config["run"]["output_dir"] == "results/llmrouterbench_broad20"
    assert config["data"]["cache_path"] == "data/processed/llmrouterbench_broad20/outcomes.csv"
    assert config["data"]["datasets"] == expected_datasets
    assert config["data"]["models"] == expected_models
    assert set(expected_datasets) <= set(config["data"]["domain_map"])
    assert set(expected_datasets) <= set(config["data"]["task_taxonomy_map"])
    assert config["routecode"]["k_values"] == [1, 2, 4, 8, 16, 32, 64, 128]


def test_broad10_config_matches_audited_10_model_rectangle():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "llmrouterbench_broad10.yaml")
    candidate = pd.read_csv(root / "results" / "llmrouterbench_pilot" / "table_broad_coverage_candidates.csv")
    row = candidate[candidate["model_count"].eq(10)].iloc[0]
    expected_datasets = row["datasets"].split(";")
    expected_models = row["models"].split(";")

    assert config["run"]["output_dir"] == "results/llmrouterbench_broad10"
    assert config["data"]["cache_path"] == "data/processed/llmrouterbench_broad10/outcomes.csv"
    assert config["data"]["datasets"] == expected_datasets
    assert config["data"]["models"] == expected_models
    assert set(expected_datasets) <= set(config["data"]["domain_map"])
    assert set(expected_datasets) <= set(config["data"]["task_taxonomy_map"])
    assert config["model_pool_scale"]["sizes"] == [2, 4, 8, 10]
    assert config["model_pool_transfer"]["source_sizes"] == [4, 6]
    assert config["model_pool_transfer"]["target_sizes"] == [4]


def test_broad20_config_uses_stronger_direct_router_comparisons():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "llmrouterbench_broad20.yaml")
    expected = {"logistic", "svm", "knn", "mlp", "gradient_boosting"}

    assert expected <= set(config["stronger_direct_router_probe"]["direct_router_methods"])
    assert expected <= set(config["model_pool_transfer"]["direct_router_methods"])
    assert set(config["new_model_calibration"]["direct_router_methods"]) == {"logistic", "svm", "knn"}


def test_broad20_config_calibrates_every_model_as_holdout():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "llmrouterbench_broad20.yaml")

    assert config["new_model_calibration"]["holdout_models"] == config["data"]["models"]
    assert len(config["new_model_calibration"]["holdout_models"]) == 20
    assert config["new_model_calibration"]["direct_router_logistic_solver"] == "saga"
    assert config["new_model_calibration"]["direct_router_svm_backend"] == "sgd"
    assert config["new_model_calibration"]["direct_router_max_iter"] <= 100


def test_broad20_config_uses_expanded_split_sensitivity_scope():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "llmrouterbench_broad20.yaml")
    split_config = config["split_sensitivity"]

    assert split_config["max_group_scenarios"] is None
    assert split_config["cluster_count"] == 4
    assert split_config["cluster_holdout_count"] == 4
    assert split_config["max_model_pool_scenarios"] is None
