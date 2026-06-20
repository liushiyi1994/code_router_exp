from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "21_external_command_readiness.py"
    spec = importlib.util.spec_from_file_location("external_command_readiness", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_external_command_readiness_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: llmrouterbench",
                "external_command_readiness:",
                f"  project_root: {tmp_path}",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_external_command_readiness.csv"
    memo_path = out_dir / "phase_e_external_command_readiness_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert "routellm_mf_train_cli" in set(table["check_id"])
    assert "graphrouter_cli" in set(table["check_id"])
    metric_row = table.set_index("check_id").loc["routecode_local_routellm_mf_metric"]
    assert str(out_dir / "table_routellm_mf_split_aligned.csv") in metric_row["command"]
    memo = memo_path.read_text(encoding="utf-8")
    assert "exact upstream-command readiness" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## External Command Readiness" in readme


def test_external_command_readiness_memo_reports_frugalgpt_smoke_without_blocked_claim(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "frugalgpt_local_scorer_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/frugalgpt_split_aligned/output/frugalgpt_smoke_stdout.log",
            }
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "FrugalGPT local scorer has a successful bounded smoke execution" in memo
    assert "FrugalGPT, EmbedLLM" not in memo


def test_external_command_readiness_memo_reports_frugalgpt_metric_when_available(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "routecode_local_frugalgpt_metric",
                "status": "available",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": True,
                "exact_upstream_command": False,
                "blocking_reasons": "",
                "execution_evidence": "",
            },
            {
                "check_id": "frugalgpt_local_scorer_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/frugalgpt_split_aligned/output/frugalgpt_smoke_stdout.log",
            },
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "local metric-bearing FrugalGPT adapter row is available" in memo
    assert "runtime evidence for the command path" in memo


def test_external_command_readiness_memo_does_not_list_missing_local_metric_as_blocked_command(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "routecode_local_frugalgpt_metric",
                "status": "missing_metric",
                "runnable_now": False,
                "no_api_compatible": True,
                "routecode_metric_compatible": True,
                "exact_upstream_command": False,
                "blocking_reasons": "missing_frugalgpt_metric_table",
                "execution_evidence": "",
            },
            {
                "check_id": "graphrouter_cli",
                "status": "blocked",
                "runnable_now": False,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "missing_python_modules:torch_geometric",
                "execution_evidence": "",
            },
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    blocked_line = next(line for line in memo.splitlines() if line.startswith("- Still-blocked"))
    assert "graphrouter_cli" in blocked_line
    assert "routecode_local_frugalgpt_metric" not in blocked_line


def test_external_command_readiness_memo_reports_embedllm_knn_smoke(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "embedllm_knn_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/embedllm_assets/embedllm_knn_smoke_stdout.log",
            }
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "EmbedLLM KNN has a successful bounded smoke execution" in memo


def test_external_command_readiness_memo_reports_embedllm_knn_full_execution(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "embedllm_knn_cli",
                "status": "executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/embedllm_knn_cli_metrics/embedllm_knn_k131_stdout.log",
            }
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "EmbedLLM KNN has successful full-split tensor executions" in memo
    assert "correctness metrics, not RouteCode routing utility" in memo


def test_external_command_readiness_memo_reports_embedllm_mf_smoke(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "embedllm_mf_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/embedllm_assets/embedllm_mf_smoke_stdout.log",
            }
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "EmbedLLM MF has a successful bounded smoke execution" in memo


def test_external_command_readiness_memo_reports_embedllm_mf_full_execution(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "embedllm_mf_cli",
                "status": "executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/embedllm_mf_cli_metrics/embedllm_mf_stdout.log",
            }
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "EmbedLLM MF has successful full-split upstream router-mode execution" in memo
    assert "upstream router accuracy, not RouteCode routing utility" in memo


def test_external_command_readiness_memo_reports_avengerspro_full_execution(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "avengerspro_cli",
                "status": "executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/avengerspro_cli_metrics/simple_cluster_full_results.json",
            }
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "Avengers-Pro simple cluster router has successful full-split exact upstream execution" in memo
    assert "upstream accuracy and cost, not RouteCode routing utility" in memo


def test_external_command_readiness_memo_reports_avengerspro_upstream_metric_when_available(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "routecode_upstream_avengerspro_metric",
                "status": "available",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": True,
                "exact_upstream_command": False,
                "blocking_reasons": "",
                "execution_evidence": "results/run/avengerspro_upstream_metric/raw_routing_details.json",
            },
            {
                "check_id": "avengerspro_cli",
                "status": "executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/avengerspro_cli_metrics/simple_cluster_full_results.json",
            },
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "upstream-code Avengers-Pro RouteCode metric row is available" in memo
    assert "not an exact upstream command output" in memo
    assert "upstream accuracy and cost, not RouteCode routing utility" in memo


def test_external_command_readiness_memo_reports_llmrouter_train_smoke(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "llmrouter_knn_train_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/llmrouter_library_adapters/llmrouter_knn_train_stdout.log",
            },
            {
                "check_id": "llmrouter_svm_train_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/llmrouter_library_adapters/llmrouter_svm_train_stdout.log",
            },
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "LLMRouter KNN/SVM training CLIs have successful bounded smoke executions" in memo


def test_external_command_readiness_memo_reports_llmrouter_infer_execution(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "llmrouter_knn_infer_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/llmrouter_library_adapters/llmrouter_knn_infer_stdout.log",
            },
            {
                "check_id": "llmrouter_svm_infer_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/llmrouter_library_adapters/llmrouter_svm_infer_stdout.log",
            },
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "LLMRouter KNN/SVM route-only inference CLIs have successful no-API executions" in memo
    assert "Full-split outputs are used when available; otherwise bounded smoke outputs are reported" in memo
    assert "precomputed RouteCode embedding cache" in memo


def test_external_command_readiness_memo_reports_embedllm_knn_metric_when_available(tmp_path):
    module = _load_script()
    table = pd.DataFrame(
        [
            {
                "check_id": "routecode_local_embedllm_knn_metric",
                "status": "available",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": True,
                "exact_upstream_command": False,
                "blocking_reasons": "",
                "execution_evidence": "",
            },
            {
                "check_id": "embedllm_knn_cli",
                "status": "smoke_executed",
                "runnable_now": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "blocking_reasons": "",
                "execution_evidence": "results/run/embedllm_assets/embedllm_knn_smoke_stdout.log",
            },
        ]
    )
    module.write_memo(tmp_path, "config.yaml", table)

    memo = (tmp_path / "phase_e_external_command_readiness_memo.md").read_text(encoding="utf-8")
    assert "local metric-bearing EmbedLLM KNN adapter row is available" in memo
    assert "This validates the patched local command path" in memo
