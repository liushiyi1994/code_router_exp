#!/usr/bin/env bash
set -euo pipefail

port="${1:?usage: scripts/stop_vllm_port.sh PORT}"

pattern="vllm.entrypoints.openai.api_server .*--port ${port}"
for pid in $(pgrep -f "$pattern" || true); do
  if [[ "$pid" == "$$" || "$pid" == "$PPID" ]]; then
    continue
  fi
  kill "$pid" 2>/dev/null || true
done
