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

## 4. API caution

GPT/Claude chat subscriptions are useful for writing, debugging, and manual checks. They should not be assumed to cover batch API experiments.

For any API-based experiment:

- obtain API keys/credits explicitly;
- cache every request/response;
- record model version, date, prompt, temperature, token counts, and cost;
- start with calibration examples only, not full matrices.

---

## 5. Do not do first

Do not start the project by:

- fine-tuning a 7B router;
- using GPT/Claude for full matrix generation;
- building an LLM-based query summarizer for every query;
- making local serving a blocker for benchmark pilots.

The first pilot is fully offline and synthetic.

