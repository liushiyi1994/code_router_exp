from __future__ import annotations

import argparse
from pathlib import Path

from routecode.controlled.live_stage0 import run_live_stage0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 live Stage 0 frontier smoke with cache and cost caps.")
    parser.add_argument("--config", default="configs/proberoute_controlled.yaml")
    parser.add_argument("--output-dir", default="results/controlled/live_stage0")
    parser.add_argument("--examples-per-benchmark", type=int, default=None)
    parser.add_argument("--run-suffix", default="live_stage0")
    parser.add_argument("--allow-frontier-calls", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--force-local-rerun", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-calls-per-frontier-model", type=int, default=None)
    parser.add_argument("--max-calls-per-local-model", type=int, default=None)
    parser.add_argument("--frontier-concurrency", type=int, default=1)
    parser.add_argument("--task-manifest", default="")
    parser.add_argument("--task-datasets", default="")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--local-max-output-tokens", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument(
        "--local-model-ids",
        default="",
        help="Comma-separated local model ids to collect. Use this for one-model-at-a-time lazy vLLM runs.",
    )
    parser.add_argument(
        "--frontier-model-ids",
        default="",
        help="Comma-separated frontier model ids to estimate/collect. Use this for provider-specific paid runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_live_stage0(
        Path(args.config),
        allow_frontier_calls=args.allow_frontier_calls,
        output_dir=args.output_dir,
        examples_per_benchmark=args.examples_per_benchmark,
        run_suffix=args.run_suffix,
        force_rerun=args.force_rerun,
        force_local_rerun=args.force_local_rerun,
        retry_errors=args.retry_errors,
        max_calls_per_frontier_model=args.max_calls_per_frontier_model,
        max_calls_per_local_model=args.max_calls_per_local_model,
        frontier_concurrency=args.frontier_concurrency,
        task_manifest_path=args.task_manifest or None,
        task_datasets=[item.strip() for item in args.task_datasets.split(",") if item.strip()] or None,
        max_tasks=args.max_tasks,
        max_output_tokens_override=args.max_output_tokens,
        local_max_output_tokens_override=args.local_max_output_tokens,
        local_model_ids=[item.strip() for item in args.local_model_ids.split(",") if item.strip()] or None,
        frontier_model_ids=[item.strip() for item in args.frontier_model_ids.split(",") if item.strip()] or None,
        request_timeout_s_override=args.request_timeout_s,
    )
    print(f"Wrote live Stage 0 outputs to {paths['output_dir']}")
    print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()
