# Frozen State Prediction On New Benchmarks

This run freezes the Broad100-trained predicted RouteCode state machinery and
applies it to the new-benchmark live smoke.

Important scope:

- state clusters are learned from Broad100 train utility vectors only;
- the state predictor is trained on Broad100 train features only;
- the state-to-action table is trained on Broad100 train rows only;
- no threshold or action rule is selected on the new benchmarks;
- the comparable action pool is restricted to `qwen3-4b-local, gpt-5.5`.

The new benchmark rows are from `simpleqa_verified`, `livebench_math`, and
`livebench_reasoning`.

## Result

| method | queries | quality | utility | frontier rate | remote cost usd | selected models |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| quality_oracle_common_model | 15 | 0.7333 | 0.5494 | 0.7333 | 0.0702 | {"gpt-5.5": 11, "qwen3-4b-local": 4} |
| cost_aware_oracle_common_model | 15 | 0.7333 | 0.5494 | 0.7333 | 0.0702 | {"gpt-5.5": 11, "qwen3-4b-local": 4} |
| all_gpt-5.5 | 15 | 0.7333 | 0.3833 | 1.0000 | 0.1336 | {"gpt-5.5": 15} |
| random_common_model | 15 | 0.3333 | 0.1931 | 0.4000 | 0.0535 | {"gpt-5.5": 6, "qwen3-4b-local": 9} |
| all_qwen3-4b-local | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | {"qwen3-4b-local": 15} |
| frozen_state_rf_probe_only_k16 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | {"qwen3-4b-local": 15} |
| frozen_state_rf_probe_plus_benchmark_k16 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | {"qwen3-4b-local": 15} |
| frozen_state_rf_probe_only_k24 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | {"qwen3-4b-local": 15} |
| frozen_state_rf_probe_plus_benchmark_k24 | 15 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | {"qwen3-4b-local": 15} |

Interpretation:

- the common-model cost-aware oracle still shows a routing opportunity;
- all GPT is strong but costly;
- all Qwen3-4B is cheap but fails this small slice;
- the frozen state policy is the deployable test. If it routes mostly to local
  and underperforms, that is evidence that the current Broad100 states/action
  table do not yet transfer to these new benchmark families.

## Per-Benchmark Model Inputs

| benchmark | model | queries | quality | utility | cost usd |
| --- | --- | ---: | ---: | ---: | ---: |
| livebench_math | gpt-5.5 | 5 | 1.0000 | 0.8433 | 0.0199 |
| livebench_math | qwen3-4b-local | 5 | 0.0000 | 0.0000 | 0.0000 |
| livebench_reasoning | gpt-5.5 | 5 | 0.4000 | -0.2278 | 0.0799 |
| livebench_reasoning | qwen3-4b-local | 5 | 0.0000 | 0.0000 | 0.0000 |
| simpleqa_verified | gpt-5.5 | 5 | 0.8000 | 0.5346 | 0.0338 |
| simpleqa_verified | qwen3-4b-local | 5 | 0.0000 | 0.0000 | 0.0000 |

## Frozen State Diagnostics

| method | Broad100 train queries | new queries | new groups used | fallback |
| --- | ---: | ---: | ---: | --- |
| frozen_state_rf_probe_only_k16 | 492 | 15 | 1 | qwen3-4b-local |
| frozen_state_rf_probe_plus_benchmark_k16 | 492 | 15 | 1 | qwen3-4b-local |
| frozen_state_rf_probe_only_k24 | 492 | 15 | 1 | qwen3-4b-local |
| frozen_state_rf_probe_plus_benchmark_k24 | 492 | 15 | 1 | qwen3-4b-local |

## Commands

```bash
bash scripts/start_vllm_qwen3_4b.sh

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \
  --config configs/proberoute_controlled_broad100.yaml \
  --output-dir results/phase3_new_benchmark_live/live_smoke_qwen4_gpt_15 \
  --run-suffix new_benchmark_gpt512_smoke \
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \
  --frontier-model-ids gpt-5.5 \
  --local-model-ids qwen3-4b-local \
  --allow-frontier-calls \
  --retry-errors \
  --max-calls-per-frontier-model 15 \
  --max-calls-per-local-model 15 \
  --frontier-concurrency 1 \
  --max-output-tokens 512 \
  --local-max-output-tokens 128 \
  --request-timeout-s 120

PYTHONPATH=src python experiments/244_phase3_frozen_state_new_benchmark.py
```

## Artifacts

- `table_frozen_state_policy.csv`
- `table_frozen_state_assignments.csv`
- `table_frozen_state_action_table.csv`
- `table_frozen_state_diagnostics.csv`
- `table_frozen_state_model_summary.csv`
