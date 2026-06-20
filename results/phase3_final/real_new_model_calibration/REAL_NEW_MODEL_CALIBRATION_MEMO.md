# Real Local/Frontier New-Model Calibration Status

This artifact covers the optional live-call Phase 3 item. It is deliberately conservative: local vLLM calls can verify serving, latency, and zero API cost, but frontier calls are not made without provider keys, budget, caching, and pricing approval.

## Commands

```bash
bash scripts/start_vllm_qwen3_0_6b.sh
python experiments/239_phase3_real_new_model_calibration.py --config configs/probecode_final_eval.yaml --run-live-smoke --smoke-limit 8
```

## Current Live Smoke

- Model: `qwen3-0.6b-probe-live-smoke` served as `Qwen/Qwen3-0.6B`
- Calls: `8`
- Mean quality on the tiny scored smoke: `0.0000`
- Mean latency: `0.7023` seconds
- API cost: `$0.00`

## Existing Live Artifact

- `gemini-3.5-flash`: status `skipped`, calls `200`, mean quality `nan` if scored, total cost `$0.3063`, mean latency `0.0000` seconds
- `gpt-5.5`: status `skipped`, calls `200`, mean quality `nan` if scored, total cost `$1.0306`, mean latency `0.0000` seconds
- `qwen3-32b-awq-thinking-local`: status `success`, calls `200`, mean quality `0.0450` if scored, total cost `$0.0000`, mean latency `3.0432` seconds

## Interpretation

- Local vLLM calls are working in the `ml-gpu` environment.
- The existing Qwen32 thinking live run is useful as a serving/cost/latency artifact, but it is not evidence that the live local model is strong; the capped thinking prompt produced many malformed answers.
- GPT/Gemini live calibration is still not run in this package because provider keys are not present in the environment.
- Therefore the real local/frontier onboarding claim remains incomplete. The cached/simulated onboarding tables are complete, but real frontier deployment evidence still needs approved calls.

## Output Tables

- `table_real_new_model_calibration.csv`
- `cost_latency_summary.csv`
- `live_readiness.csv`
- `frontier_provider_readiness.csv`
- `table_live_smoke_outputs.csv`

## Summary Table

| experiment | model_id | provider | status | n_calls | mean_quality | mean_utility | total_cost_usd | mean_latency_s | evidence_type | claim_supported | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_live_broad100_qwen32_thinking_mcq | gemini-3.5-flash | google | skipped | 200 |  |  | 0.3063 | 0.0000 | historical_live_or_frontier_cost_estimate | False | Existing local live Broad100 artifact; frontier rows were skipped/estimated, not called. |
| historical_live_broad100_qwen32_thinking_mcq | gpt-5.5 | openai | skipped | 200 |  |  | 1.0306 | 0.0000 | historical_live_or_frontier_cost_estimate | False | Existing local live Broad100 artifact; frontier rows were skipped/estimated, not called. |
| historical_live_broad100_qwen32_thinking_mcq | qwen3-32b-awq-thinking-local | local | success | 200 | 0.0450 | 0.0450 | 0.0000 | 3.0432 | historical_live_or_frontier_cost_estimate | False | Existing local live Broad100 artifact; frontier rows were skipped/estimated, not called. |
| current_live_qwen06_smoke | qwen3-0.6b-probe-live-smoke | local | success | 8 | 0.0000 | 0.0000 | 0.0000 | 0.7023 | current_local_live_smoke | False | Tiny vLLM smoke verifies serving and logging only; it is not a full calibration result. |
| frontier_live_call_readiness | gpt-5.5 | openai | blocked_no_api_key | 0 |  |  |  |  | environment_readiness | False | No closed-source calls were made without provider keys and explicit budget. |
| frontier_live_call_readiness | gemini-3.5-flash | google | blocked_no_api_key | 0 |  |  |  |  | environment_readiness | False | No closed-source calls were made without provider keys and explicit budget. |

## Cost And Latency

| source | model_id | provider | status | n_calls | total_cost_usd | cost_per_1k_calls_usd | mean_latency_s | p95_latency_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_live_broad100_qwen32_thinking_mcq | gemini-3.5-flash | google | skipped | 200 | 0.3063 | 1.5317 | 0.0000 | 0.0001 |
| historical_live_broad100_qwen32_thinking_mcq | gpt-5.5 | openai | skipped | 200 | 1.0306 | 5.1529 | 0.0000 | 0.0001 |
| historical_live_broad100_qwen32_thinking_mcq | qwen3-32b-awq-thinking-local | local | success | 200 | 0.0000 | 0.0000 | 3.0432 | 3.4039 |
| current_live_qwen06_smoke | qwen3-0.6b-probe-live-smoke | local | success | 8 | 0.0000 | 0.0000 | 0.7023 | 0.8394 |
| current_frontier_readiness | gpt-5.5 | openai | blocked_no_api_key | 0 |  |  |  |  |
| current_frontier_readiness | gemini-3.5-flash | google | blocked_no_api_key | 0 |  |  |  |  |
