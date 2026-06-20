# Phase 3 New-Benchmark Live Smoke

This folder records the first small out-of-benchmark-family live smoke for
RouteCode/ProbeCode.

Status: `routing_opportunity_observed`

Cost-aware oracle matched GPT quality (0.7333) while reducing frontier rate from 1.0000 to 0.6667 and remote spend by 56.3%. The cheap local model alone was weak (quality 0.0667), so this is an oracle opportunity, not a deployed-state result.

This is not yet a proof that the learned states generalize. It shows that the
live harness can ingest new benchmark families and that a local-vs-frontier
oracle opportunity exists on this tiny slice.

## Benchmarks

These benchmarks were not in the Broad100 state-learning pool
(`aime`, `bbh`, `gpqa`, `gsm8k`, `humaneval`, `livemathbench`, `math500`,
`mbpp`, `mmlupro`).

| dataset | tasks |
| --- | ---: |
| livebench_math | 5 |
| livebench_reasoning | 5 |
| simpleqa_verified | 5 |

HLE was considered but not included because `cais/hle` is gated in this
environment. BigCodeBench was considered but deferred because pass@1 code
execution needs a separate harness.

## Models

| model | status | calls | mean quality | total cost usd | mean latency s | cache hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-5.5 | success | 15 | 0.7333 | 0.1336 | 6.178 | 15 |
| qwen3-0.6b-probe | success | 15 | 0.0667 | 0.0000 | 0.846 | 0 |

Gemini was attempted in `live_smoke_gpt_gemini_15`, but all 15 Gemini calls
returned HTTP 429, so Gemini has no usable quality result in this smoke.

## Routing Summary

| method | queries | quality | utility | frontier rate | remote cost usd | mean latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quality_oracle | 15 | 0.7333 | 0.5803 | 0.6667 | 0.0584 | 3.156 |
| cost_aware_oracle | 15 | 0.7333 | 0.5803 | 0.6667 | 0.0584 | 3.156 |
| all_gpt-5.5 | 15 | 0.7333 | 0.3833 | 1.0000 | 0.1336 | 6.178 |
| best_frontier | 15 | 0.7333 | 0.3833 | 1.0000 | 0.1336 | 6.178 |
| all_qwen3-0.6b-probe | 15 | 0.0667 | 0.0667 | 0.0000 | 0.0000 | 0.846 |
| best_local | 15 | 0.0667 | 0.0667 | 0.0000 | 0.0000 | 0.846 |
| domain_rule_code_to_gpt_else_local | 15 | 0.0667 | 0.0667 | 0.0000 | 0.0000 | 0.846 |
| local_consistency_rescue_gpt | 15 | 0.0667 | 0.0667 | 0.0000 | 0.0000 | 0.846 |
| selective_code_consistency_rescue_gpt | 15 | 0.0667 | 0.0667 | 0.0000 | 0.0000 | 0.846 |

Interpretation:

- all GPT is accurate on this tiny slice but expensive under the cost-aware
  utility normalization;
- the 0.6B local model alone is cheap and fast but too weak;
- the cost-aware oracle can keep the same quality as GPT while using local on
  one third of the rows, but this is an upper bound because no deployable state
  policy was selected on these new benchmarks.

## Per-Benchmark Model Results

| benchmark | model | status | rows | quality | cost usd | latency s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| livebench_math | gpt-5.5 | success | 5 | 1.0000 | 0.0199 | 3.337 |
| livebench_math | qwen3-0.6b-probe | success | 5 | 0.0000 | 0.0000 | 1.509 |
| livebench_reasoning | gpt-5.5 | success | 5 | 0.4000 | 0.0799 | 9.665 |
| livebench_reasoning | qwen3-0.6b-probe | success | 5 | 0.2000 | 0.0000 | 0.402 |
| simpleqa_verified | gpt-5.5 | success | 5 | 0.8000 | 0.0338 | 5.532 |
| simpleqa_verified | qwen3-0.6b-probe | success | 5 | 0.0000 | 0.0000 | 0.628 |

## Provider Failures

| model | provider | error | count |
| --- | --- | --- | ---: |
| gemini-3.5-flash | google | HTTPError | 15 |
| gpt-5.5 | openai | HTTPError | 1 |

## Commands Run

```bash
PYTHONPATH=src python experiments/242_phase3_new_benchmark_manifest.py \
  --output-dir results/phase3_new_benchmark_live \
  --per-dataset 5 \
  --seed 42

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \
  --config configs/proberoute_controlled_broad100.yaml \
  --output-dir results/phase3_new_benchmark_live/live_smoke_gpt_gemini_15 \
  --run-suffix new_benchmark_live_smoke \
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \
  --frontier-model-ids gpt-5.5,gemini-3.5-flash \
  --allow-frontier-calls \
  --retry-errors \
  --max-calls-per-frontier-model 15 \
  --frontier-concurrency 1 \
  --max-output-tokens 96 \
  --request-timeout-s 120

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \
  --config configs/proberoute_controlled_broad100.yaml \
  --output-dir results/phase3_new_benchmark_live/live_smoke_gpt_15_max512 \
  --run-suffix new_benchmark_gpt512_smoke \
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \
  --frontier-model-ids gpt-5.5 \
  --allow-frontier-calls \
  --retry-errors \
  --max-calls-per-frontier-model 15 \
  --frontier-concurrency 1 \
  --max-output-tokens 512 \
  --request-timeout-s 120

bash scripts/start_vllm_qwen3_0_6b.sh

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \
  --config configs/proberoute_controlled_broad100.yaml \
  --output-dir results/phase3_new_benchmark_live/live_smoke_qwen06_gpt_15 \
  --run-suffix new_benchmark_gpt512_smoke \
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \
  --frontier-model-ids gpt-5.5 \
  --local-model-ids qwen3-0.6b-probe \
  --allow-frontier-calls \
  --retry-errors \
  --max-calls-per-frontier-model 15 \
  --max-calls-per-local-model 15 \
  --frontier-concurrency 1 \
  --max-output-tokens 512 \
  --local-max-output-tokens 128 \
  --request-timeout-s 120

PYTHONPATH=src python experiments/243_phase3_new_benchmark_smoke_summary.py
```

## Artifacts

- `new_benchmark_manifest.csv`
- `NEW_BENCHMARK_MANIFEST_MEMO.md`
- `table_new_benchmark_model_summary.csv`
- `table_new_benchmark_routing_summary.csv`
- `table_new_benchmark_by_dataset_model.csv`
- `table_new_benchmark_provider_failures.csv`
- `live_smoke_qwen06_gpt_15/`
- `live_smoke_gpt_15_max512_rescored/`
- `live_smoke_gpt_gemini_15/`

## Next Required Test

To say the states generalize, run a larger benchmark-heldout protocol:

1. collect local/probe outputs for 50-100 rows per new benchmark;
2. freeze the Broad100-trained state predictor and action table;
3. select no thresholds on the new benchmarks;
4. report RouteCode/ProbeCode, all-local, all-GPT/Gemini, random routing, and
   local-vs-frontier oracle on the held-out new-benchmark rows.
