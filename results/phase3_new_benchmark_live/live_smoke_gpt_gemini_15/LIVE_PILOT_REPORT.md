# Live Controlled Pilot

Run id: `controlled_surrogate_pilot_v1_new_benchmark_live_smoke`
Task manifest: `results/phase3_new_benchmark_live/new_benchmark_manifest.csv`; datasets: `all`; task count: `15`.

Fresh frontier calls requested in this invocation: `True`.
Frontier pool for this invocation: `gemini-3.5-flash, gpt-5.5`.
Claude/Anthropic models were not used.
Ready local vLLM endpoints are called and cached with zero remote-dollar cost; cached local rows are reused when the endpoint is offline.
Local readiness is recorded in `local_readiness.csv`; use `--local-model-ids` for one-model-at-a-time vLLM collection and `--frontier-model-ids` for provider-specific paid runs.
Configured launcher commands are listed in `configs/model_servers.yaml`.

## Pre-call Cost Estimate

Caps: total `$14.90`, per model `$14.90`.

| model | estimated cost usd |
| --- | ---: |
| gemini-3.5-flash | 0.0191 |
| gpt-5.5 | 0.0638 |

## Local vLLM Readiness

| model | backend | status | fallback |
| --- | --- | --- | --- |
| qwen3-0.6b-probe | vllm | unavailable | controlled_surrogate |
| qwen3-4b-local | vllm | unavailable | controlled_surrogate |
| qwen3-8b-local | vllm | unavailable | controlled_surrogate |
| qwen3-14b-awq-local | vllm | unavailable | controlled_surrogate |
| qwen3-32b-awq-local | vllm | unavailable | controlled_surrogate |
| qwen3-32b-awq-thinking-local | vllm | unavailable | controlled_surrogate |
| qwen3.5-9b-local | vllm | unavailable | controlled_surrogate |
| qwen3-coder-30b-a3b | vllm | unavailable | controlled_surrogate |
| qwen3.6-35b-a3b | vllm | unavailable | controlled_surrogate |
| gemma-3-12b-it | vllm | unavailable | controlled_surrogate |

## Model Results

| model | status | calls | mean quality | total cost usd | mean generation latency s | mean load time s | mean warmup time s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-3.5-flash | error | 15 |  | 0.0177 | 0.254 | 0.000 | 0.000 |
| gpt-5.5 | error | 1 |  | 0.0033 | 2.779 | 0.000 | 0.000 |
| gpt-5.5 | success | 14 | 0.0 | 0.0509 | 2.658 | 0.000 | 0.000 |

## Live Routing Summary

| method | queries | mean quality | mean utility | quality gap | utility gap | frontier rate | probe rate | remote cost usd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_gpt-5.5 | 14 | 0.0000 | -0.3500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0509 |
| best_frontier | 14 | 0.0000 | -0.3500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0509 |
| quality_oracle | 14 | 0.0000 | -0.3500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0509 |
| cost_aware_oracle | 14 | 0.0000 | -0.3500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0509 |

Raw responses are cached under `results/controlled/raw_outputs`. Reruns reuse cache unless `--force-rerun` is set.
Rows in this report may be fresh responses or cached live responses; see `cache_hit` in `model_outputs.parquet`.
Generation latency excludes model loading and warmup for lazy local models. Load and warmup are reported separately in `cost_latency_summary.csv`.
Latency for fresh calls is measured end-to-end after the endpoint is ready. Cache hits use the latency saved in the raw cache when available; older cache entries created before latency persistence may show cache-read latency and should not be used for final latency claims.
