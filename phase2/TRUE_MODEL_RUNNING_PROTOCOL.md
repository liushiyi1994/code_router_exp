# TRUE_MODEL_RUNNING_PROTOCOL.md — Local Model Running and Evaluation

This document specifies how to run true local model experiments for ProbeRoute++.

The purpose is not to replace LLMRouterBench. The purpose is to collect probe signals and validate the method with actual model calls under controlled evaluation.

---

## 1. Hardware and serving stack

User has an NVIDIA RTX 5090 with 32GB GDDR7 memory.

Recommended serving stack:

1. vLLM first, because it provides an OpenAI-compatible server.
2. llama-cpp-python for GGUF quantized models.
3. SGLang if it is easier for Qwen/DeepSeek or structured generation.

### vLLM server example

```bash
pip install -U vllm
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --api-key local-routecode
```

Working RTX 5090/WSL two-endpoint launch used in the Phase 2 pilot:

```bash
env "PATH=/home/liush/miniconda3/envs/ml-gpu/bin:$PATH" \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  /home/liush/miniconda3/envs/ml-gpu/bin/vllm serve Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8001 \
  --api-key local-routecode \
  --dtype auto \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.35 \
  --served-model-name qwen3_4b_vllm

env "PATH=/home/liush/miniconda3/envs/ml-gpu/bin:$PATH" \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  /home/liush/miniconda3/envs/ml-gpu/bin/vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8002 \
  --api-key local-routecode \
  --dtype auto \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.20 \
  --served-model-name qwen3_0_6b_vllm
```

On this local stack, `VLLM_USE_V2_MODEL_RUNNER=0` avoids `RuntimeError: UVA is not available`. Keep the conda env at the front of `PATH` so `ninja` is available during JIT.

Client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local-routecode")
```

### llama-cpp-python server example

```bash
pip install 'llama-cpp-python[server]'
python -m llama_cpp.server \
  --model /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n_gpu_layers -1 \
  --n_ctx 16384
```

### SGLang server example

```bash
pip install -U sglang
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 30000
```

---

## 2. Local model pool

Start with 2 models for smoke test, then 4, then 6.

### Smoke test pool

```text
Qwen3-8B
Qwen2.5-Coder-7B-Instruct
```

### Main local pool

```text
Qwen3-8B
Qwen2.5-Coder-7B-Instruct
DeepSeek-R1-Distill-Qwen-7B
Llama-3.1-8B-Instruct
MiniCPM4.1-8B
Gemma-3-4B or another small general model
```

### Notes

- Run models sequentially if memory is limited.
- Keep generation parameters deterministic.
- Use quantized variants if full precision is too slow or too large.
- Record exact model revision/hash if downloaded from Hugging Face.

## 2.1 Later closed-source provider pool

The current working default is local vLLM. Closed-source providers stay in scope for later explicit-budget experiments because applied routing often compares local/open models with frontier provider models.

Provider families to include when API access and budget are explicitly enabled:

```text
OpenAI GPT-family
Anthropic Claude-family
Google Gemini-family
```

Rules for provider runs:

- do not call provider APIs in the default local pilot;
- start with calibration examples, not full matrices;
- cache every request and response;
- log provider, model ID, model version/date, prompt, parameters, latency, input tokens, output tokens, and request count;
- refresh pricing before the run and record source URLs plus checked dates;
- report provider token cost separately from local probe latency/GPU proxy cost.

---

## 3. Datasets for true local evaluation

Use exact-scored datasets first.

### Tier 1: easiest to evaluate

```text
GSM8K: exact numeric answer
MATH500: boxed/numeric answer extraction
MMLU-Pro or MMLU: multiple-choice letter
GPQA: multiple-choice letter
```

### Tier 2: code tasks

```text
HumanEval
MBPP
LiveCodeBench subset
```

Only run code tasks after safe sandboxed execution is implemented.

### Tier 3: avoid at first

```text
open-ended writing
long-form QA needing LLM judge
agent/tool tasks
```

These complicate evaluation and introduce judge cost/noise.

---

## 4. Dataset sizes

### Smoke test

```text
20 queries
1 model
```

Goal: verify server, client, logging, parsing, scoring.

### Pilot true local run

```text
200--500 queries
2--4 models
```

Goal: collect a real local outcome/probe matrix.

### Main true local run

```text
1,000--3,000 queries
4--8 models
```

Goal: validate claims with true model calls.

---

## 5. Generation parameters

Default answer generation:

```yaml
temperature: 0.0
max_new_tokens:
  math: 512
  multiple_choice: 128
  code: 768
stop: dataset-specific
```

Probe generation:

```yaml
temperature: 0.0
max_new_tokens: 32 or 64
```

Always log full generation parameters.

---

## 6. Prompt templates

### Math answer prompt

```text
Solve the problem. Give the final answer at the end in the format:
Final answer: <answer>

Problem:
{query}
```

### Multiple-choice prompt

```text
Answer the following multiple-choice question. Respond with only the letter A, B, C, D, or E.

Question:
{query}

Choices:
{choices}
```

### Code prompt

```text
Write a correct Python solution for the following programming task.
Return only code.

Task:
{query}
```

### Confidence probe prompt

```text
You are a small local model. Do not solve fully. Estimate whether you can answer the query correctly.
Respond in JSON:
{"confidence": <0.0 to 1.0>, "reason": "short phrase"}

Query:
{query}
```

### Short draft probe prompt

```text
Give a very short first attempt, at most 64 tokens. Then give confidence from 0 to 1.

Query:
{query}
```

---

## 7. Evaluation and parsers

### Math parser

Extract:

- `Final answer:` field;
- boxed answer `\\boxed{}`;
- last numeric expression.

Compare after normalization.

### Multiple choice parser

Extract first valid letter among A/B/C/D/E.

### Code evaluator

Use sandboxed evaluation only.

Requirements:

- run in isolated process/container;
- timeout;
- no network;
- memory limit;
- capture stdout/stderr;
- store pass/fail and error.

---

## 8. Output schemas

### Local outcomes

File:

```text
results/phase2/local_model_outcomes.parquet
```

Columns:

```text
query_id
query_text
dataset
domain
model_id
model_revision
prompt_template
generation_params_json
raw_output
parsed_answer
gold_answer
quality
cost_proxy
latency_sec
tokens_input
tokens_output
error_type
error_message
created_at
```

### Probe features

File:

```text
results/phase2/probe_features.parquet
```

Columns:

```text
query_id
probe_id
probe_type
probe_model_id
prompt_template
generation_params_json
raw_probe_output
parsed_probe_answer
self_confidence
logprob_mean
entropy_proxy
agreement_score
knn_label_entropy
knn_winner_entropy
latency_sec
input_tokens
output_tokens
probe_cost_proxy
error_type
error_message
created_at
```

---

## 9. Cost accounting

Even local probes are not free.

Track:

```text
latency seconds
input tokens
output tokens
GPU runtime proxy
probe fraction
```

For local experiments, define normalized probe cost:

```text
probe_cost = latency_weight * latency_sec + token_weight * output_tokens
```

For cost-aware analyses, sweep probe cost multipliers.

Provider-cost analyses must add input-token cost, output-token cost, and request counts from the current provider price snapshot. Keep these separate from local probe costs so the report can show both model-evaluation cost and probe-acquisition cost.

---

## 10. Safety and reproducibility

Every local run must save:

- command line;
- config file;
- git commit hash;
- model name and revision;
- prompt template ID;
- generation parameters;
- output file checksum;
- error logs.

Never overwrite raw outputs. Write new run IDs.

Suggested path:

```text
results/local_runs/{run_id}/
  config.yaml
  outcomes.parquet
  raw_outputs.jsonl
  run_metadata.json
  errors.jsonl
```

---

## 11. First local experiment recommendation

Start with:

```text
Dataset: GSM8K + MMLU/GPQA small subset
Queries: 200
Models: Qwen3-8B, Qwen2.5-Coder-7B-Instruct
Probes: Qwen3-8B confidence probe, kNN uncertainty
```

Goal:

```text
Verify end-to-end local generation, scoring, probe extraction, and logging.
```

Only after that, scale to more models/datasets.
