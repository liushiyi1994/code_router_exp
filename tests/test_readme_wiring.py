from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_rate_distortion_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "02_rate_distortion_curve.py"
    spec = importlib.util.spec_from_file_location("rate_distortion_curve", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extra_commands_include_sensitivity_suite_for_synthetic_and_real_configs():
    module = _load_rate_distortion_script()

    synthetic_commands = module._extra_commands({"data": {"source": "synthetic"}}, "configs/synthetic.yaml")
    real_commands = module._extra_commands({"data": {"source": "llmrouterbench"}}, "configs/llmrouterbench_pilot.yaml")

    assert "python experiments/09_sensitivity_suite.py --config configs/synthetic.yaml" in synthetic_commands
    assert "python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/10_external_baseline_surrogates.py --config configs/synthetic.yaml" in synthetic_commands
    assert "python experiments/10_external_baseline_surrogates.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/11_code_card_interpretability.py --config configs/synthetic.yaml" in synthetic_commands
    assert "python experiments/11_code_card_interpretability.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/12_official_baseline_artifacts.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/13_transformer_backbone_readiness.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/28_transformer_embedding_router.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/28_transformer_embedding_router.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/29_embedllm_knn_split_aligned.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/29_embedllm_knn_split_aligned.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/30_frugalgpt_split_aligned.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/30_frugalgpt_split_aligned.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/14_routellm_pairwise_alignment.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/15_routellm_mf_assets.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/15_routellm_mf_assets.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/16_routellm_mf_split_aligned.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/16_routellm_mf_split_aligned.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/17_avengerspro_split_aligned.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/17_avengerspro_split_aligned.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/37_avengerspro_cli_metrics.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/37_avengerspro_cli_metrics.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/40_avengerspro_upstream_metric.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/40_avengerspro_upstream_metric.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/38_graphrouter_cli_metrics.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/38_graphrouter_cli_metrics.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/39_graphrouter_split_aligned.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/39_graphrouter_split_aligned.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/18_model_pool_scale.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/18_model_pool_scale.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/19_model_pool_transfer.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/19_model_pool_transfer.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/20_benchmark_coverage.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/20_benchmark_coverage.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/21_external_command_readiness.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/21_external_command_readiness.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/22_cost_quality_frontier.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/22_cost_quality_frontier.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/23_stronger_direct_router_probe.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/23_stronger_direct_router_probe.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/26_external_baseline_assets.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/26_external_baseline_assets.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/27_llmrouter_library_adapters.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/27_llmrouter_library_adapters.py --config configs/llmrouterbench_pilot.yaml" in real_commands
    assert "python experiments/25_provider_price_sensitivity.py --config configs/synthetic.yaml" not in synthetic_commands
    assert "python experiments/25_provider_price_sensitivity.py --config configs/llmrouterbench_pilot.yaml" in real_commands


def test_extra_outputs_include_sensitivity_suite_artifacts_for_synthetic_and_real_configs():
    module = _load_rate_distortion_script()

    synthetic_outputs = "\n".join(module._extra_outputs({"data": {"source": "synthetic"}}))
    real_outputs = "\n".join(module._extra_outputs({"data": {"source": "llmrouterbench"}}))

    assert "table_sensitivity_summary.csv" in synthetic_outputs
    assert "phase_g_sensitivity_memo.md" in synthetic_outputs
    assert "table_sensitivity_summary.csv" in real_outputs
    assert "phase_g_sensitivity_memo.md" in real_outputs
    assert "table_residual_risk.csv" in real_outputs
    assert "fig_risk_coverage.pdf" in real_outputs
    assert "table_external_baselines.csv" in synthetic_outputs
    assert "phase_e_external_baseline_memo.md" in real_outputs
    assert "table_code_card_interpretability.csv" in synthetic_outputs
    assert "phase_f_code_card_interpretability_memo.md" in real_outputs
    assert "table_official_external_artifacts.csv" not in synthetic_outputs
    assert "table_official_external_artifacts.csv" in real_outputs
    assert "phase_e_official_baseline_artifacts_memo.md" in real_outputs
    assert "table_transformer_backbone_readiness.csv" not in synthetic_outputs
    assert "table_transformer_backbone_readiness.csv" in real_outputs
    assert "phase_f_g_transformer_backbone_readiness_memo.md" in real_outputs
    assert "table_transformer_embedding_router.csv" not in synthetic_outputs
    assert "table_transformer_embedding_router.csv" in real_outputs
    assert "phase_f_g_transformer_embedding_router_memo.md" in real_outputs
    assert "table_embedllm_knn_split_aligned.csv" not in synthetic_outputs
    assert "table_embedllm_knn_split_aligned.csv" in real_outputs
    assert "phase_e_embedllm_knn_split_aligned_memo.md" in real_outputs
    assert "table_frugalgpt_split_aligned.csv" not in synthetic_outputs
    assert "table_frugalgpt_split_aligned.csv" in real_outputs
    assert "phase_e_frugalgpt_split_aligned_memo.md" in real_outputs
    assert "table_routellm_pairwise_alignment.csv" not in synthetic_outputs
    assert "table_routellm_pairwise_alignment.csv" in real_outputs
    assert "phase_e_routellm_pairwise_alignment_memo.md" in real_outputs
    assert "table_routellm_mf_assets.csv" not in synthetic_outputs
    assert "table_routellm_mf_assets.csv" in real_outputs
    assert "phase_e_routellm_mf_assets_memo.md" in real_outputs
    assert "table_routellm_mf_split_aligned.csv" not in synthetic_outputs
    assert "table_routellm_mf_split_aligned.csv" in real_outputs
    assert "phase_e_routellm_mf_split_aligned_memo.md" in real_outputs
    assert "table_avengerspro_split_aligned.csv" not in synthetic_outputs
    assert "table_avengerspro_split_aligned.csv" in real_outputs
    assert "phase_e_avengerspro_split_aligned_memo.md" in real_outputs
    assert "table_avengerspro_cli_metrics.csv" not in synthetic_outputs
    assert "table_avengerspro_cli_metrics.csv" in real_outputs
    assert "phase_e_avengerspro_cli_metrics_memo.md" in real_outputs
    assert "table_avengerspro_upstream_metric.csv" not in synthetic_outputs
    assert "table_avengerspro_upstream_metric.csv" in real_outputs
    assert "phase_e_avengerspro_upstream_metric_memo.md" in real_outputs
    assert "table_graphrouter_cli_metrics.csv" not in synthetic_outputs
    assert "table_graphrouter_cli_metrics.csv" in real_outputs
    assert "phase_e_graphrouter_cli_metrics_memo.md" in real_outputs
    assert "table_graphrouter_split_aligned.csv" not in synthetic_outputs
    assert "table_graphrouter_split_aligned.csv" in real_outputs
    assert "phase_e_graphrouter_split_aligned_memo.md" in real_outputs
    assert "table_model_pool_scale.csv" not in synthetic_outputs
    assert "table_model_pool_scale.csv" in real_outputs
    assert "phase_f_g_model_pool_scale_memo.md" in real_outputs
    assert "table_model_pool_transfer.csv" not in synthetic_outputs
    assert "table_model_pool_transfer.csv" in real_outputs
    assert "phase_f_g_model_pool_transfer_memo.md" in real_outputs
    assert "table_benchmark_dataset_coverage.csv" not in synthetic_outputs
    assert "table_benchmark_dataset_coverage.csv" in real_outputs
    assert "table_broad_coverage_candidates.csv" in real_outputs
    assert "phase_g_benchmark_coverage_memo.md" in real_outputs
    assert "table_external_command_readiness.csv" not in synthetic_outputs
    assert "table_external_command_readiness.csv" in real_outputs
    assert "phase_e_external_command_readiness_memo.md" in real_outputs
    assert "table_cost_quality_summary.csv" not in synthetic_outputs
    assert "table_cost_quality_summary.csv" in real_outputs
    assert "table_cost_quality_frontier.csv" in real_outputs
    assert "phase_e_cost_quality_memo.md" in real_outputs
    assert "table_stronger_direct_router_probe.csv" not in synthetic_outputs
    assert "table_stronger_direct_router_probe.csv" in real_outputs
    assert "phase_e_stronger_direct_router_probe_memo.md" in real_outputs
    assert "table_external_baseline_assets.csv" not in synthetic_outputs
    assert "table_external_baseline_assets.csv" in real_outputs
    assert "phase_e_external_baseline_assets_memo.md" in real_outputs
    assert "table_llmrouter_library_adapters.csv" not in synthetic_outputs
    assert "table_llmrouter_library_adapters.csv" in real_outputs
    assert "phase_e_llmrouter_library_adapters_memo.md" in real_outputs
    assert "table_provider_price_schedule.csv" not in synthetic_outputs
    assert "table_provider_price_schedule.csv" in real_outputs
    assert "table_provider_cost_quality_summary.csv" in real_outputs
    assert "table_provider_cost_quality_frontier.csv" in real_outputs
    assert "phase_g_provider_pricing_memo.md" in real_outputs


def test_experiment_scripts_include_regret_objective_routecode_rows():
    root = Path(__file__).resolve().parents[1]
    rate_distortion_source = (root / "experiments" / "02_rate_distortion_curve.py").read_text(encoding="utf-8")
    ablation_source = (root / "experiments" / "08_ablation_summary.py").read_text(encoding="utf-8")

    assert "regret_routecode_oracle_labels" in rate_distortion_source
    assert "regret_routecode_utility_oracle" in ablation_source
