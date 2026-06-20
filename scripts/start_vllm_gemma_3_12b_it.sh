#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/liush/miniconda3/envs/ml-gpu/bin:${PATH}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

exec /home/liush/miniconda3/envs/ml-gpu/bin/python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-12b-it \
  --served-model-name google/gemma-3-12b-it \
  --host 127.0.0.1 \
  --port 8005 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code
