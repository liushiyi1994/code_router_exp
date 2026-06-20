#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/liush/miniconda3/envs/ml-gpu/bin:${PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B-AWQ}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"

exec /home/liush/miniconda3/envs/ml-gpu/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name Qwen/Qwen3-32B-AWQ \
  --host 127.0.0.1 \
  --port 8007 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --quantization awq_marlin \
  --enforce-eager \
  --trust-remote-code
