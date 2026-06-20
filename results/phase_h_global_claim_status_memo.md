# Global Phase H Claim Audit

Command:

```bash
python experiments/35_global_claim_audit.py --result-dir results/llmrouterbench_pilot --result-dir results/llmrouterbench_broad10 --result-dir results/llmrouterbench_broad20 --result-dir results/llmrouterbench_scale20 --result-dir results/llmrouterbench_32model --output-dir results
```

This memo aggregates per-run Phase H claim gates across the supplied result directories. It is intentionally conservative: contradictory nonmissing evidence becomes `mixed_evidence`, and missing run evidence is counted but does not by itself create support.

Result directories:

- `results/llmrouterbench_pilot`
- `results/llmrouterbench_broad10`
- `results/llmrouterbench_broad20`
- `results/llmrouterbench_scale20`
- `results/llmrouterbench_32model`

## Global Claim Status

| claim_id | claim | global_status | run_count | status_counts | best_primary_value | worst_primary_value | best_result_id | evidence_summary | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low_rate_oracle_codes | Useful low-rate utility route codes exist. | diagnostic_supported | 5 | diagnostic_supported=5 | 1.0000 | 0.9535 | llmrouterbench_pilot | llmrouterbench_pilot: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=1.0000); llmrouterbench_broad10: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=1.0000); llmrouterbench_broad20: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9681); llmrouterbench_scale20: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9799); llmrouterbench_32model: diagnostic_supported (best_low_rate_oracle_recovered_gap_vs_oracle=0.9535) | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| small_inferred_labels | Small inferred route labels recover most routing performance. | not_supported | 5 | not_supported=5 | 0.3459 | 0.0233 | llmrouterbench_pilot | llmrouterbench_pilot: not_supported (best_inferred_recovered_gap_vs_oracle=0.3459); llmrouterbench_broad10: not_supported (best_inferred_recovered_gap_vs_oracle=0.1891); llmrouterbench_broad20: not_supported (best_inferred_recovered_gap_vs_oracle=0.0906); llmrouterbench_scale20: not_supported (best_inferred_recovered_gap_vs_oracle=0.1409); llmrouterbench_32model: not_supported (best_inferred_recovered_gap_vs_oracle=0.0233) | Do not claim that small inferred route labels recover most routing performance across current runs. |
| model_pool_transfer | Route labels transfer across model pools better than same-budget direct retraining. | mixed_evidence | 5 | diagnostic_alive=4; not_supported=1 | 0.3083 | -0.0537 | llmrouterbench_pilot | llmrouterbench_pilot: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.3083); llmrouterbench_broad10: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1523); llmrouterbench_broad20: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1472); llmrouterbench_scale20: diagnostic_alive (mean_matched_transfer_minus_direct_recovered_gap=0.1616); llmrouterbench_32model: not_supported (mean_matched_transfer_minus_direct_recovered_gap=-0.0537) | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| new_model_calibration | New models can be integrated with fewer calibration examples than direct retraining. | diagnostic_alive | 5 | diagnostic_alive=5 | 0.8140 | 0.2339 | llmrouterbench_32model | llmrouterbench_pilot: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.2339); llmrouterbench_broad10: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.4106); llmrouterbench_broad20: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.7402); llmrouterbench_scale20: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.5096); llmrouterbench_32model: diagnostic_alive (mean_matched_routecode_minus_direct_recovered_gap=0.8140) | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| benchmark_diagnosis | Benchmark routing results expose compressibility or split-design artifacts. | mixed_evidence | 5 | diagnostic_supported=2; not_supported=3 | 0.7904 | 0.1198 | llmrouterbench_32model | llmrouterbench_pilot: diagnostic_supported (min_split_rank_correlation=0.1928); llmrouterbench_broad10: not_supported (min_split_rank_correlation=0.7488); llmrouterbench_broad20: diagnostic_supported (min_split_rank_correlation=0.1198); llmrouterbench_scale20: not_supported (min_split_rank_correlation=0.5367); llmrouterbench_32model: not_supported (min_split_rank_correlation=0.7904) | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| adaptive_refinement | Adaptive refinement improves cost-quality by refining uncertain queries. | not_supported | 5 | not_supported=5 | 0.2683 | 0.1521 | llmrouterbench_pilot | llmrouterbench_pilot: not_supported (top10_regret_mass_fraction=0.2683); llmrouterbench_broad10: not_supported (top10_regret_mass_fraction=0.2048); llmrouterbench_broad20: not_supported (top10_regret_mass_fraction=0.1521); llmrouterbench_scale20: not_supported (top10_regret_mass_fraction=0.1910); llmrouterbench_32model: not_supported (top10_regret_mass_fraction=0.2000) | Current cross-run evidence does not support this claim. |
