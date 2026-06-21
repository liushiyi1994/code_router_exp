# RouteCode Results

## Phase 3 Final Package

The current paper-facing result package lives in `phase3_final/`.

Read in this order:

1. `phase3_final/README.md`
2. `phase3_final/EXPERIMENT_PROTOCOL.md`
3. `phase3_final/FINAL_EVALUATION_REPORT.md`
4. `phase3_final/TWO_CLAIM_LIVE_COMPLETION_AUDIT.md`

Current bottom line:

- ProbeCode-StateCal quality `0.8547` vs oracle `0.8721`.
- ProbeCode-StateCal utility `0.8238` vs oracle `0.8463`.
- Oracle utility ratio `0.9735`.
- Frontier-call rate `0.1919`.

Benchmark scope:

- current final states are learned/selected using Broad100 train/validation
  splits from `aime`, `bbh`, `gpqa`, `gsm8k`, `humaneval`, `livemathbench`,
  `math500`, `mbpp`, and `mmlupro`;
- final eval is on held-out test queries from the same benchmark families;
- new-benchmark-family generalization remains a required next experiment.

## Phase 3 New-Benchmark Live Smoke

The first out-of-benchmark-family live smoke lives in
`phase3_new_benchmark_live/`.

Read:

1. `phase3_new_benchmark_live/README.md`
2. `phase3_new_benchmark_live/NEW_BENCHMARK_MANIFEST_MEMO.md`
3. `phase3_new_benchmark_live/table_new_benchmark_routing_summary.csv`
4. `phase3_new_benchmark_live/table_new_benchmark_by_dataset_model.csv`

Scope and result:

- benchmarks: `simpleqa_verified`, `livebench_math`, `livebench_reasoning`;
- models: `qwen3-0.6b-probe` through vLLM and cached `gpt-5.5`; Gemini was
  attempted but returned HTTP 429 for all 15 rows;
- all GPT quality `0.7333`, utility `0.3833`, remote cost `$0.1336`;
- cost-aware local/GPT oracle quality `0.7333`, utility `0.5803`, frontier
  rate `0.6667`, remote cost `$0.0584`.

Frozen-state follow-up:

- `phase3_new_benchmark_live/frozen_state_prediction/README.md`
- comparable action pool: `qwen3-4b-local`, `gpt-5.5`;
- common-model oracle quality `0.7333`, utility `0.5494`;
- all GPT quality `0.7333`, utility `0.3833`;
- all frozen-state variants routed all rows to Qwen3-4B and scored quality
  `0.0000`, utility `0.0000`.

Interpretation: this shows a live routing opportunity on new benchmark
families, but the current frozen Broad100 state predictor/action table fails
this tiny out-of-benchmark transfer check.

## Global Claim Audit

Command:

```bash
python experiments/35_global_claim_audit.py --result-dir results/llmrouterbench_pilot --result-dir results/llmrouterbench_broad10 --result-dir results/llmrouterbench_broad20 --result-dir results/llmrouterbench_scale20 --result-dir results/llmrouterbench_32model --output-dir results
```

Outputs:

- `table_claim_status_by_run.csv`: per-run claim gates.
- `table_claim_status_global.csv`: conservative cross-run claim status.
- `phase_h_global_claim_status_memo.md`: global claim-gate memo.
- `../paper_notes.md`: conservative paper-positioning notes based on the current claim gates and external-baseline readiness.

| claim_id | claim | global_status | run_count | status_counts | best_primary_value | worst_primary_value | best_result_id | evidence_summary | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_rate_oracle_codes | Useful low-rate utility route codes exist. | diagnostic_supported | 5 | diagnostic_supported=5 | 1.0000 | 0.9535 | llmrouterbench_pilot | llmrouterbench_pilot: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=1.0000); llmrouterbench_broad10: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=1.0000); llmrouterbench_broad20: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9681); llmrouterbench_scale20: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9799); llmrouterbench_32model: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9535) | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| small_inferred_labels | Small inferred route labels recover most routing performance. | not_supported | 5 | not_supported=5 | 0.3459 | 0.0233 | llmrouterbench_pilot | llmrouterbench_pilot: not_supported (best_inferred_recovered_gap_vs_oracle=0.3459); llmrouterbench_broad10: not_supported (best_inferred_recovered_gap_vs_oracle=0.1891); llmrouterbench_broad20: not_supported (best_inferred_recovered_gap_vs_oracle=0.0906); llmrouterbench_scale20: not_supported (best_inferred_recovered_gap_vs_oracle=0.1409); llmrouterbench_32model: not_supported (best_inferred_recovered_gap_vs_oracle=0.0233) | Do not claim that small inferred route labels recover most routing performance across current runs. |
| model_pool_transfer | Route labels transfer across model pools better than same-budget direct retraining. | mixed_evidence | 5 | diagnostic_alive=4; not_supported=1 | 0.3083 | -0.0537 | llmrouterbench_pilot | llmrouterbench_pilot: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.3083); llmrouterbench_broad10: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1523); llmrouterbench_broad20: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1472); llmrouterbench_scale20: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1616); llmrouterbench_32model: not_supported (mean_matched_transfer_minus_direct_recovered_gap=-0.0537) | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| new_model_calibration | New models can be integrated with fewer calibration examples than direct retraining. | diagnostic_alive | 5 | diagnostic_alive=5 | 0.8140 | 0.2339 | llmrouterbench_32model | llmrouterbench_pilot: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.2339); llmrouterbench_broad10: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.4106); llmrouterbench_broad20: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.7402); llmrouterbench_scale20: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.5096); llmrouterbench_32model: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.8140) | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| benchmark_diagnosis | Benchmark routing results expose compressibility or split-design artifacts. | mixed_evidence | 5 | diagnostic_supported=2; not_supported=3 | 0.7904 | 0.1198 | llmrouterbench_32model | llmrouterbench_pilot: diagnostic_supported (min_split_rank_correlation=0.1928); llmrouterbench_broad10: not_supported (min_split_rank_correlation=0.7488); llmrouterbench_broad20: diagnostic_supported (min_split_rank_correlation=0.1198); llmrouterbench_scale20: not_supported (min_split_rank_correlation=0.5367); llmrouterbench_32model: not_supported (min_split_rank_correlation=0.7904) | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| adaptive_refinement | Adaptive refinement improves cost-quality by refining uncertain queries. | not_supported | 5 | not_supported=5 | 0.2683 | 0.1521 | llmrouterbench_pilot | llmrouterbench_pilot: not_supported (top10_regret_mass_fraction=0.2683); llmrouterbench_broad10: not_supported (top10_regret_mass_fraction=0.2048); llmrouterbench_broad20: not_supported (top10_regret_mass_fraction=0.1521); llmrouterbench_scale20: not_supported (top10_regret_mass_fraction=0.1910); llmrouterbench_32model: not_supported (top10_regret_mass_fraction=0.2000) | Current cross-run evidence does not support this claim. |

## Paper Evidence Summary

Command:

```bash
python experiments/41_paper_evidence_summary.py --output-dir results --readiness-table results/llmrouterbench_pilot/table_external_command_readiness.csv --readiness-table results/llmrouterbench_broad20/table_external_command_readiness.csv --paper-notes paper_notes.md
```

Outputs:

- `table_paper_evidence_summary.csv`: paper-facing claim and baseline posture table.
- `phase_h_paper_evidence_summary.md`: conservative paper-positioning memo.
- `../paper_notes.md`: root paper notes generated from the same evidence table.

| section | item | status | key_value | evidence | interpretation |
| --- | --- | --- | --- | --- | --- |
| paper_direction | recommended_framing | information_frontier_diagnostic | small_inferred_labels=not_supported; low_rate_oracle_codes=diagnostic_supported; new_model_calibration=diagnostic_alive; model_pool_transfer=mixed_evidence | results/table_claim_status_global.csv | Recommended framing: information-frontier and benchmark-diagnostic paper. Do not claim that few inferred bits are enough; current evidence supports low-rate oracle structure, modest deployable inferred-label recovery, and diagnostic calibration/transfer threads. |
| claim | low_rate_oracle_codes | diagnostic_supported | best=1.0000; worst=0.9535 | llmrouterbench_pilot: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=1.0000); llmrouterbench_broad10: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=1.0000); llmrouterbench_broad20: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9681); llmrouterbench_scale20: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9799); llmrouterbench_32model: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9535) | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| claim | small_inferred_labels | not_supported | best=0.3459; worst=0.0233 | llmrouterbench_pilot: not_supported (best_inferred_recovered_gap_vs_oracle=0.3459); llmrouterbench_broad10: not_supported (best_inferred_recovered_gap_vs_oracle=0.1891); llmrouterbench_broad20: not_supported (best_inferred_recovered_gap_vs_oracle=0.0906); llmrouterbench_scale20: not_supported (best_inferred_recovered_gap_vs_oracle=0.1409); llmrouterbench_32model: not_supported (best_inferred_recovered_gap_vs_oracle=0.0233) | Do not claim that small inferred route labels recover most routing performance across current runs. |
| claim | model_pool_transfer | mixed_evidence | best=0.3083; worst=-0.0537 | llmrouterbench_pilot: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.3083); llmrouterbench_broad10: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1523); llmrouterbench_broad20: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1472); llmrouterbench_scale20: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1616); llmrouterbench_32model: not_supported (mean_matched_transfer_minus_direct_recovered_gap=-0.0537) | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| claim | new_model_calibration | diagnostic_alive | best=0.8140; worst=0.2339 | llmrouterbench_pilot: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.2339); llmrouterbench_broad10: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.4106); llmrouterbench_broad20: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.7402); llmrouterbench_scale20: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.5096); llmrouterbench_32model: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.8140) | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| claim | benchmark_diagnosis | mixed_evidence | best=0.7904; worst=0.1198 | llmrouterbench_pilot: diagnostic_supported (min_split_rank_correlation=0.1928); llmrouterbench_broad10: not_supported (min_split_rank_correlation=0.7488); llmrouterbench_broad20: diagnostic_supported (min_split_rank_correlation=0.1198); llmrouterbench_scale20: not_supported (min_split_rank_correlation=0.5367); llmrouterbench_32model: not_supported (min_split_rank_correlation=0.7904) | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| claim | adaptive_refinement | not_supported | best=0.2683; worst=0.1521 | llmrouterbench_pilot: not_supported (top10_regret_mass_fraction=0.2683); llmrouterbench_broad10: not_supported (top10_regret_mass_fraction=0.2048); llmrouterbench_broad20: not_supported (top10_regret_mass_fraction=0.1521); llmrouterbench_scale20: not_supported (top10_regret_mass_fraction=0.1910); llmrouterbench_32model: not_supported (top10_regret_mass_fraction=0.2000) | Current cross-run evidence does not support this claim. |
| external_baselines | readiness_overview | partial | 38 rows; 30 runnable; 22 exact | results/llmrouterbench_pilot/table_external_command_readiness.csv; results/llmrouterbench_broad20/table_external_command_readiness.csv | RouteCode-compatible metric rows available: routecode_local_embedllm_knn_metric, routecode_local_frugalgpt_metric, routecode_local_routellm_mf_metric, routecode_upstream_avengerspro_metric. Blocked rows remain. |
| external_baselines | best_route_train_cli | blocked |  | results/llmrouterbench_pilot/table_external_command_readiness.csv; results/llmrouterbench_broad20/table_external_command_readiness.csv | missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender |
| external_baselines | modelsat_train_cli | blocked |  | results/llmrouterbench_pilot/table_external_command_readiness.csv; results/llmrouterbench_broad20/table_external_command_readiness.csv | missing_modelsat_base_model_checkpoint;missing_modelsat_embedding_model_checkpoint;missing_python_modules:nltk,deepspeed |
| external_baselines | routellm_bert_cli | blocked |  | results/llmrouterbench_pilot/table_external_command_readiness.csv; results/llmrouterbench_broad20/table_external_command_readiness.csv | missing_bert_checkpoint |
| external_baselines | routerdc_train_cli | blocked |  | results/llmrouterbench_pilot/table_external_command_readiness.csv; results/llmrouterbench_broad20/table_external_command_readiness.csv | missing_routerdc_local_model_checkpoint;missing_python_modules:deepspeed |

## Research Flow Completion Audit

Command:

```bash
python experiments/42_research_flow_completion_audit.py --root /home/liush/projects/code_router_exp --output-dir /home/liush/projects/code_router_exp/results
```

Outputs:

- `table_research_flow_completion.csv`: phase-by-phase completion evidence.
- `phase_h_research_flow_completion_audit.md`: completion audit memo.

| phase_id | phase | status | required_paths_present | required_paths_total | missing_paths | completion_rule | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase_a_synthetic_sanity | Phase A - setup and synthetic sanity | complete | 5 | 5 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_b_real_data_pilot | Phase B - real-data pilot study | complete | 5 | 5 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_c_observation_synthesis | Phase C - observation synthesis | complete | 1 | 1 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_d2_predictability_constrained | Phase D2 - predictability-constrained RouteCode | complete | 1 | 1 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_d3_code_cards | Phase D3 - explainable route-label cards | complete | 1 | 1 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_d4_new_model_calibration | Phase D4 - new-model calibration | complete | 1 | 1 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_d5_adaptive_refinement | Phase D5 - adaptive refinement | deferred | 1 | 1 |  | deferred_if_gate_weak | Adaptive refinement is deferred unless a stronger deployable residual-risk signal appears. |
| phase_e_external_methods | Phase E - method evaluation and external baselines | complete | 2 | 2 |  | complete_if_phase_e_required_coverage_present | Required Phase E baseline coverage is complete. Optional checkpoint-gated rows documented: routellm_bert_cli, best_route_train_cli, routerdc_train_cli, modelsat_train_cli. |
| phase_f_ablation | Phase F - ablation study | complete | 1 | 1 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_g_sensitivity | Phase G - sensitivity analysis | complete | 1 | 1 |  | all_required_artifacts_present | Required artifacts are present. |
| phase_h_final_claims | Phase H - final paper claims | complete | 3 | 3 |  | complete_if_claims_documented_conservatively | Conservative final claim posture documented: recommended_framing=information_frontier_diagnostic; unsupported/mixed claims not claimed: small_inferred_labels=not_supported, model_pool_transfer=mixed_evidence, benchmark_diagnosis=mixed_evidence, adaptive_refinement=not_supported |

## External Blocker Resolution

Command:

```bash
python experiments/43_external_blocker_resolution.py
```

Inputs:

- `/home/liush/projects/code_router_exp/results/llmrouterbench_pilot/table_external_command_readiness.csv`
- `/home/liush/projects/code_router_exp/results/llmrouterbench_broad20/table_external_command_readiness.csv`

Outputs:

- `table_external_blocker_resolution.csv`: blocked external-command rows grouped across runs with missing modules, checkpoints, local assets, service requirements, and next actions.
- `phase_e_external_blocker_resolution_memo.md`: interpretation memo for the unresolved Phase E blockers.

Blocked rows: `4`.
Checkpoint-gated blocked rows: `4`.

| check_id | blocked_runs | blocked_run_count | blocking_reasons | missing_modules | missing_checkpoints | missing_assets | service_requirements | other_blockers | can_progress_without_download | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_route_train_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender | llm_blender | missing_best_route_local_model_checkpoint |  |  |  | False | Provision local checkpoints: missing_best_route_local_model_checkpoint. Install Python module: llm_blender. |
| modelsat_train_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_modelsat_base_model_checkpoint;missing_modelsat_embedding_model_checkpoint;missing_python_modules:nltk,deepspeed | deepspeed,nltk | missing_modelsat_base_model_checkpoint,missing_modelsat_embedding_model_checkpoint |  |  |  | False | Provision local checkpoints: missing_modelsat_base_model_checkpoint,missing_modelsat_embedding_model_checkpoint. Install Python modules: deepspeed,nltk. |
| routellm_bert_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_bert_checkpoint |  | missing_bert_checkpoint |  |  |  | False | Provision local checkpoints: missing_bert_checkpoint. |
| routerdc_train_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_python_modules:deepspeed;missing_routerdc_local_model_checkpoint | deepspeed | missing_routerdc_local_model_checkpoint |  |  |  | False | Provision local checkpoints: missing_routerdc_local_model_checkpoint. Install Python module: deepspeed. |

## Phase 3 Oracle-Level Modification Summary

Command:

```bash
PYTHONPATH=src python experiments/221_phase3_oracle_level_modification_summary.py
```

Outputs:

- `controlled/phase3_oracle_level_modification/table_oracle_level_modification.csv`
- `controlled/phase3_oracle_level_modification/ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md`

The summary shows that the current Broad100 learned-verifiability /
verifiable-action-pool method reaches the configured oracle-level gate
(`quality_gap=0.0174`, `utility_ratio=0.9735`, `frontier_call_rate=0.1919`).
The compact RouteCode state policy also passes the 3-point and 95% utility
gate (`quality_gap=0.0233`, `utility_ratio=0.9614`). The clean no-tool oracle
does not pass against the full oracle (`quality_gap=0.0465`,
`utility_ratio=0.9338`), and forcing GPT-strong on the eight residual rows
matches oracle quality but fails utility (`utility_ratio=0.9234`).
