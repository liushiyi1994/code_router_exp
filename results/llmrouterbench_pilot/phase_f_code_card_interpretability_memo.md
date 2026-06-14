# Phase F Code-Card Interpretability Memo

Command: `python experiments/11_code_card_interpretability.py --config configs/llmrouterbench_pilot.yaml`

This is an observability ablation, not a routing-utility result. It compares what is inspectable from a route-label lookup table alone against what is inspectable after generating code cards from train-set labels and utility profiles.

## Summary Table

| codebook | condition | n_labels | available_explainability_fields | best_model_coverage | domain_summary_coverage | dataset_summary_coverage | representative_query_coverage | failure_case_coverage | utility_vector_coverage | human_explanation_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_routecode | label_only | 16 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flat_routecode | with_code_cards | 16 | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| predictability_constrained_routecode | label_only | 16 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| predictability_constrained_routecode | with_code_cards | 16 | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Current Readout

- Label-only rows expose at most `1` explanatory field in this audit.
- Code-card rows expose at least `9` explanatory fields across the tested codebooks.
- Minimum code-card coverage: human explanations `1.0000`, representative queries `1.0000`, and high-regret examples `1.0000`.
- The result supports treating code cards as an explainability and diagnosis layer. It does not show that code cards improve routing utility, because model selection is unchanged by this audit.
