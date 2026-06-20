#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/liush/miniconda3/envs/ml-gpu/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-/home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-14B-AWQ/snapshots/31c69efc29464b6bb0aee1398b5a7b50a99340c3}"

exec /home/liush/miniconda3/envs/ml-gpu/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name Qwen/Qwen3-14B-AWQ \
  --host 127.0.0.1 \
  --port 8006 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.86 \
  --max-num-seqs 2 \
  --quantization awq_marlin \
  --enforce-eager \
  --disable-custom-all-reduce \
  --trust-remote-code
