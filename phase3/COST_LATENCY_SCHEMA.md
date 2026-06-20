# Cost and Latency Schema for ProbeRoute++ Controlled Experiments

This file defines exactly what Codex should log and compute.

## 1. Per-call raw output schema

Every model call must produce one row.

```text
run_id: string
query_id: string
benchmark: string
domain: string
model_id: string
provider: local|openai|anthropic|other
is_local: bool
is_frontier: bool
is_probe: bool
prompt_template_version: string
prompt_hash: string
input_tokens: int
output_tokens: int
max_output_tokens: int
start_time_unix: float
end_time_unix: float
latency_s: float
status: success|error|timeout|parse_error
error_type: string|null
raw_output_path: string
parsed_answer: string|null
quality_score: float|null
cost_input_usd: float
cost_output_usd: float
cost_total_usd: float
cache_hit: bool
server_backend: vllm|llamacpp|sglang|api|unknown
server_config_json: string
hardware_id: string
```

## 2. Utility computation

### Quality only

```text
U_quality = quality
```

### Cost-aware

```text
U_cost = quality - lambda_cost * normalized_remote_cost
```

### Cost + latency-aware

```text
U_cost_latency = quality
                 - lambda_cost * normalized_remote_cost
                 - lambda_latency * normalized_latency
```

Normalize cost and latency within the model pool or relative to all-frontier baseline. The normalization choice must be written in `RUN_REPORT.md`.

## 3. Remote cost

For local models:

```text
remote_cost_usd = 0
```

For frontier/API models:

```text
remote_cost_usd = input_tokens * input_price_per_token
                + output_tokens * output_price_per_token
```

Prices should come from `configs/model_prices.yaml` and should be timestamped.

## 4. Latency summary

Report:

```text
mean latency
p50 latency
p90 latency
p95 latency
p99 latency
router time
probe time
selected model generation time
total end-to-end time
```

## 5. Method-level summary

Each method summary row should include:

```text
method
quality_mean
utility_quality_only
utility_cost_aware
utility_cost_latency_aware
remote_cost_per_query
remote_cost_per_1k_queries
remote_cost_per_1m_queries
normalized_remote_cost_vs_all_gpt
normalized_remote_cost_vs_all_claude
frontier_call_rate
local_call_rate
probe_call_rate
latency_mean
latency_p50
latency_p95
latency_p99
```

## 6. Required plots

1. quality vs normalized remote cost;
2. quality vs p95 latency;
3. frontier-call rate vs quality;
4. probe-rate vs utility;
5. calibration budget vs utility;
6. rate-distortion curve.
