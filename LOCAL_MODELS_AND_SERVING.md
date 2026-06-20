# Local Models and Serving Plan

This file describes how to use the user's RTX 5090 for optional local validation. The main project should start with precomputed benchmarks, not local generation.

---

## 1. Hardware assumption

User has an NVIDIA RTX 5090-class local GPU. Treat this as enough for:

- embedding extraction;
- small classifier training;
- inference with 7B--14B models depending on quantization and context length;
- small local validation matrices.

Do not require local generation for the first synthetic or LLMRouterBench pilots.

---

## 2. Recommended stack

### Primary: vLLM

Use for Hugging Face transformer models with OpenAI-compatible API.

```bash
pip install -U vllm
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --api-key local-routecode
```

Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local-routecode")
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Classify this query domain: ..."}],
    temperature=0,
)
```

For Phase 2 multi-model local runs, use one vLLM server per base model and give each server a separate port. The current Phase 2 runner supports this through `phase2_local_eval.openai_endpoints`.

Example two-model layout:

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

On this WSL/CUDA/vLLM 0.23.0 setup, `VLLM_USE_V2_MODEL_RUNNER=0` was required to avoid `RuntimeError: UVA is not available`. The quoted `PATH` prefix keeps the conda env `ninja` binary visible for JIT.

Then run the 200-query exact-scored Phase 2 matrix:

```bash
python experiments/51_true_model_generation_matrix.py \
  --config configs/phase2_local_vllm_two_model_all200_nothink.yaml
```

This writes `results/phase2/local_vllm_two_model_all200_nothink/local_model_outcomes.parquet`. The run still makes no GPT/Claude/Gemini API calls.

To run the full Phase 2 chain after both servers are live, use:

```bash
PYTHONPATH=src python experiments/71_local_vllm_policy_pipeline.py \
  --config configs/phase2_local_vllm_two_model_all200_nothink.yaml
```

This checks readiness, runs local generation, converts local outcomes into policy matrices, evaluates ProbeRoute++ policies, and refreshes the Phase 2 completion audit.

### Secondary: llama.cpp / llama-cpp-python

Use for GGUF quantized models.

```bash
pip install 'llama-cpp-python[server]'
python -m llama_cpp.server \
  --model /path/to/model.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n_gpu_layers -1 \
  --n_ctx 32768
```

Client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
```

### Alternative: SGLang

```bash
pip install -U sglang
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 30000
```

---

## 3. Candidate models

### Generation / optional validation

- Qwen3-4B / Qwen3-8B instruct models.
- Qwen Coder 7B/14B for code-heavy tasks.
- Llama/Gemma/Mistral/Phi 7B--8B class models for diversity.

### Embeddings

- Qwen3-Embedding 0.6B/4B/8B.
- BAAI/bge-large-en-v1.5 or bge-m3.
- sentence-transformers/all-MiniLM-L6-v2 as a cheap baseline.

### Classifiers

- Logistic regression / MLP on embeddings.
- ModernBERT-base/large classifier.
- DeBERTa-v3-base classifier.
- Optional QLoRA 7B classifier only after cheap baselines work.

---

## 4. API and closed-source model caution

GPT, Claude, and Gemini chat subscriptions are useful for writing, debugging, and manual checks. They should not be assumed to cover batch API experiments.

Closed-source provider families to keep in the future model-pool plan:

```text
OpenAI GPT-family
Anthropic Claude-family
Google Gemini-family
```

For any API-based experiment:

- obtain API keys/credits explicitly;
- cache every request/response;
- record model version, date, prompt, temperature, token counts, and cost;
- refresh provider pricing before the run and record source URLs plus checked dates;
- keep provider token cost separate from local probe latency/GPU proxy cost;
- start with calibration examples only, not full matrices.

---

## 5. Do not do first

Do not start the project by:

- fine-tuning a 7B router;
- using GPT/Claude/Gemini for full matrix generation;
- building an LLM-based query summarizer for every query;
- making local serving a blocker for benchmark pilots.

The first pilot is fully offline and synthetic.
