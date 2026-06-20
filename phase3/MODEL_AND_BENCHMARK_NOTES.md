# Model and Benchmark Notes for Controlled ProbeRoute++ Run

## 1. Model choice rationale

The actual run should use a small controlled pool, not dozens of models.

### Local models

Preferred pool:

```text
Qwen/Qwen3.5-0.8B
Qwen/Qwen3.5-9B
Qwen/Qwen3-Coder-30B-A3B-Instruct
Qwen/Qwen3.6-35B-A3B
google/gemma-3-12b-it OR mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

Why this pool:

```text
cheap probe model
modern general local model
code specialist
strong local reasoning/general model
diverse non-Qwen local model
```

### Frontier models

```text
gpt-5.5 or latest GPT API model available
claude-sonnet-4-6 or latest Claude Sonnet API model available
```

Frontier models are used because cost reduction matters only if there is an expensive high-quality option.

## 2. Serving strategy

Do not load all local models at once.

Use sequential generation:

```bash
start model server
run benchmark slice
write cached outputs
stop model server
```

Preferred backend:

```text
vLLM for HF transformer models
llama.cpp for GGUF models
SGLang as fallback/alternative
```

## 3. Benchmark choice rationale

Use exact-scored benchmarks first:

```text
GSM8K
MATH500
AIME
HumanEval
MBPP
LiveCodeBench subset
GPQA
MMLU-Pro
optional BBH/logical reasoning
```

Avoid open-ended tasks requiring LLM judges in the first controlled run.

## 4. Dataset sizes

```text
Dry run: 5 examples per benchmark
Pilot: 100 examples per benchmark
Main: 200–300 examples per benchmark if budget allows
```

## 5. Frontier cost-control rules

Before any frontier run:

1. estimate token cost;
2. print expected spend;
3. require config flag `allow_frontier_calls: true`;
4. enforce `max_frontier_spend_usd`;
5. cache all outputs.

## 6. Output token caps

Suggested caps:

```text
GSM8K: 512
MATH500: 1024
AIME: 1024
HumanEval: 1024
MBPP: 1024
LiveCodeBench: 1536
GPQA: 256
MMLU-Pro: 256
BBH/logical: 512
probe: 16–32
```
