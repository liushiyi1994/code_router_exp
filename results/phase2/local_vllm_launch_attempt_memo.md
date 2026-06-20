# Local vLLM Launch Attempt

Purpose: try to start the Phase 2 local OpenAI-compatible vLLM server for the cached `Qwen/Qwen3-4B` model.

Environment observed:

```text
conda env: ml-gpu
python: 3.12.11
vLLM: installed
torch: 2.11.0+cu130
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
cached model: Qwen/Qwen3-4B
```

Command attempted:

```bash
conda run -n ml-gpu vllm serve Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8001 \
  --api-key local-routecode \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45 \
  --served-model-name qwen3_4b_vllm
```

Compatibility retries attempted:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 ... --max-model-len 2048 --enforce-eager
VLLM_USE_V1=0 ... --max-model-len 2048
```

Initial result:

```text
vLLM starts the API server process, resolves Qwen3ForCausalLM, then fails during engine initialization.
The root error is: RuntimeError: UVA is not available.
VLLM_USE_V1 is not recognized by this installed vLLM 0.23.0 package.
No server remains listening on http://127.0.0.1:8001/v1/models.
```

Working launch discovered:

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
```

Second local vLLM endpoint used for the two-model run:

```bash
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

Result:

- `VLLM_USE_V2_MODEL_RUNNER=0` avoids the UVA failure in this vLLM 0.23.0 runtime.
- Quoting `PATH=/home/liush/miniconda3/envs/ml-gpu/bin:$PATH` keeps the `ninja` binary visible to FlashInfer/JIT.
- `Qwen/Qwen3-4B` served on `http://127.0.0.1:8001/v1` as `qwen3_4b_vllm`.
- `Qwen/Qwen3-0.6B` served on `http://127.0.0.1:8002/v1` as `qwen3_0_6b_vllm`.
- The two-endpoint readiness table passed with `blocked=0` in `results/phase2/local_server_readiness_phase2_local_vllm_two_model_all200_nothink/table_local_server_readiness.csv`.
- The full two-model local vLLM pipeline completed and wrote `results/phase2/local_vllm_two_model_all200_nothink/local_model_outcomes.parquet`.
