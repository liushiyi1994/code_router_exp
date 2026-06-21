# Phase 3 State Semantic / Confidence Audit

This audit checks the mechanism proposed after the frozen-state transfer failure:

```text
query semantic similarity + local confidence
  -> decide whether state assignment is safe
  -> then assign to a RouteCode state/action
```

Inputs:

- Broad100 utility/output table: `results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet`
- Broad100 probe features: `results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv`
- New benchmark smoke outputs: `results/phase3_new_benchmark_live/live_smoke_qwen4_gpt_15/model_outputs.parquet`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` loaded locally, with TF-IDF fallback.

## Test-Split Semantic Coherence

Higher cosine means queries in the same group are semantically closer.

| group method | queries | groups | pairwise cosine | query-centroid cosine | benchmark purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| benchmark_label | 170 | 9 | 0.4176 | 0.6624 | 1.0000 |
| embedding_cluster_k16 | 170 | 16 | 0.4317 | 0.6962 | 0.6529 |
| random_size_matched_k16 | 170 | 16 | 0.1840 | 0.5082 | 0.2824 |
| utility_state_k16 | 170 | 16 | 0.2801 | 0.5684 | 0.4824 |
| embedding_cluster_k24 | 170 | 23 | 0.4387 | 0.7307 | 0.7471 |
| random_size_matched_k24 | 170 | 22 | 0.1863 | 0.5254 | 0.2765 |
| utility_state_k24 | 170 | 23 | 0.2832 | 0.5972 | 0.5353 |

## Confidence Separability

`eta_squared` is the share of variance in a local-confidence feature explained by
utility state ID. Higher means that feature differs clearly by state.

| K | split | feature | mean | eta_squared |
| ---: | --- | --- | ---: | ---: |
| 16 | all | local_all_agree | 0.1413 | 0.7475 |
| 16 | all | small_medium_agree | 0.2875 | 0.6697 |
| 16 | all | small_medium_disagree | 0.7125 | 0.6697 |
| 16 | all | q4_q14_agree | 0.2875 | 0.6697 |
| 16 | all | q8_q14_agree | 0.2568 | 0.6564 |
| 16 | all | local_vote_margin | 0.3627 | 0.6522 |
| 16 | all | local_vote_entropy | 1.2733 | 0.6036 |
| 16 | all | local_vote_frac | 0.5804 | 0.6003 |
| 16 | all | local_unique_answer_count | 2.9226 | 0.5293 |
| 16 | all | q14_q32_agree | 0.5086 | 0.5134 |
| 16 | all | medium_unique_answer_count | 1.4914 | 0.5134 |
| 16 | all | small_unique_answer_count | 1.5209 | 0.4766 |

## New Benchmark Semantic Gate Summary

`train_percentile` compares each new query's nearest-state cosine to Broad100
train queries. Very low values are OOD warnings.

| K | new queries | nearest states used | mean nearest cosine | min train percentile | below train p10 rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 15 | 8 | 0.2640 | 0.0000 | 0.8000 |
| 24 | 15 | 11 | 0.2774 | 0.0000 | 0.8667 |

## Files

- `table_state_semantic_coherence.csv`
- `table_state_confidence_separability.csv`
- `table_utility_state_cards.csv`
- `table_utility_state_assignments.csv`
- `table_new_benchmark_semantic_gate.csv`
