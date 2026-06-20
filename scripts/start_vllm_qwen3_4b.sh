#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/liush/miniconda3/envs/ml-gpu/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

exec /home/liush/miniconda3/envs/ml-gpu/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8002 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.45 \
  --max-num-seqs 4 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --trust-remote-code
