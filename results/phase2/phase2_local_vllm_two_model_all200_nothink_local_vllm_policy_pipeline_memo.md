# Local vLLM Policy Pipeline

Command:

```bash
PYTHONPATH=src python experiments/71_local_vllm_policy_pipeline.py --config configs/phase2_local_vllm_two_model_all200_nothink.yaml
```

Status: `completed`.

Local vLLM generation, local policy matrices, true-probe policy inputs, M5 policy evaluation, and audit refresh completed.

Outputs:

| artifact | path |
| --- | --- |
| readiness_table | results/phase2/local_server_readiness_phase2_local_vllm_two_model_all200_nothink/table_local_server_readiness.csv |
| local_outcomes | results/phase2/local_vllm_two_model_all200_nothink/local_model_outcomes.parquet |
| matrices_query_model_utility | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_utility.csv |
| matrices_query_model_quality | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_quality.csv |
| matrices_query_model_cost | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_query_model_cost.csv |
| matrices_state_model_utility | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_utility.csv |
| matrices_state_model_quality | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_quality.csv |
| matrices_state_model_cost | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_state_model_cost.csv |
| matrices_metadata | results/phase2/local_policy_matrices_phase2_local_vllm_two_model_all200_nothink/local_policy_matrix_metadata.json |
| inputs_before_beliefs | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_before_beliefs.csv |
| inputs_after_beliefs | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_after_beliefs.csv |
| inputs_state_model_utility | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_state_model_utility.csv |
| inputs_query_model_utility | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_query_model_utility.csv |
| inputs_probe_cost | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_cost.csv |
| inputs_predicted_gain | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_predicted_gain.csv |
| inputs_metadata | results/phase2/true_probe_policy_inputs_phase2_local_vllm_two_model_all200_nothink/true_probe_policy_input_metadata.json |
| policy_table | results/phase2/true_probe_policy_phase2_local_vllm_two_model_all200_nothink/table_proberoute_policy.csv |
| policy_figure | results/phase2/true_probe_policy_phase2_local_vllm_two_model_all200_nothink/fig_gap_closed_vs_probe_cost.pdf |
| audit_table | /home/liush/projects/code_router_exp/results/phase2/table_phase2_completion_audit.csv |
| audit_memo | /home/liush/projects/code_router_exp/results/phase2/phase2_completion_audit.md |
| audit_report | /home/liush/projects/code_router_exp/results/phase2/PHASE2_EVIDENCE_REPORT.md |
