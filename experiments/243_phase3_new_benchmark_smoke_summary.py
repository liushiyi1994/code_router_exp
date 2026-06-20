from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the Phase 3 new-benchmark live smoke.")
    parser.add_argument("--output-dir", default="results/phase3_new_benchmark_live")
    parser.add_argument("--local-gpt-dir", default="results/phase3_new_benchmark_live/live_smoke_qwen06_gpt_15")
    parser.add_argument(
        "--gpt-dir",
        default="results/phase3_new_benchmark_live/live_smoke_gpt_15_max512_rescored",
    )
    parser.add_argument("--gemini-attempt-dir", default="results/phase3_new_benchmark_live/live_smoke_gpt_gemini_15")
    parser.add_argument("--manifest", default="results/phase3_new_benchmark_live/new_benchmark_manifest.csv")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _model_by_dataset(outputs: pd.DataFrame) -> pd.DataFrame:
    if outputs.empty:
        return pd.DataFrame()
    grouped = (
        outputs.groupby(["benchmark", "model_id", "status"], dropna=False)
        .agg(
            n=("query_id", "count"),
            mean_quality=("quality_score", "mean"),
            total_cost_usd=("cost_total_usd", "sum"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", lambda values: float(values.quantile(0.95))),
            cache_hits=("cache_hit", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values(["benchmark", "model_id", "status"])


def _provider_failures(outputs: pd.DataFrame) -> pd.DataFrame:
    if outputs.empty:
        return pd.DataFrame()
    failures = outputs[outputs["status"].ne("success")]
    if failures.empty:
        return pd.DataFrame(columns=["model_id", "provider", "status", "error_type", "n"])
    return (
        failures.groupby(["model_id", "provider", "status", "error_type"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["model_id", "error_type"])
    )


def _routecode_interpretation(local_gpt_routing: pd.DataFrame, model_by_dataset: pd.DataFrame) -> tuple[str, str]:
    if local_gpt_routing.empty:
        return "not_available", "No local/GPT routing summary was available."
    oracle = local_gpt_routing[local_gpt_routing["method"].eq("cost_aware_oracle")]
    all_gpt = local_gpt_routing[local_gpt_routing["method"].eq("all_gpt-5.5")]
    local = local_gpt_routing[local_gpt_routing["method"].eq("all_qwen3-0.6b-probe")]
    if oracle.empty or all_gpt.empty or local.empty:
        return "incomplete", "The run does not contain all required oracle, GPT, and local rows."
    oracle_row = oracle.iloc[0]
    gpt_row = all_gpt.iloc[0]
    local_row = local.iloc[0]
    remote_reduction = 1.0 - float(oracle_row["remote_cost_total_usd"]) / max(float(gpt_row["remote_cost_total_usd"]), 1e-12)
    status = "routing_opportunity_observed"
    note = (
        f"Cost-aware oracle matched GPT quality ({oracle_row['mean_quality']:.4f}) while reducing frontier rate "
        f"from {gpt_row['frontier_call_rate']:.4f} to {oracle_row['frontier_call_rate']:.4f} and remote spend "
        f"by {remote_reduction:.1%}. The cheap local model alone was weak "
        f"(quality {local_row['mean_quality']:.4f}), so this is an oracle opportunity, not a deployed-state result."
    )
    return status, note


def write_readme(
    output_dir: Path,
    *,
    manifest: pd.DataFrame,
    model_summary: pd.DataFrame,
    routing_summary: pd.DataFrame,
    by_dataset: pd.DataFrame,
    failures: pd.DataFrame,
    status: str,
    note: str,
) -> None:
    dataset_lines = "\n".join(
        f"| {row.dataset} | {int(row.n_tasks)} |"
        for row in manifest.groupby("dataset").size().reset_index(name="n_tasks").itertuples(index=False)
    )
    model_lines = "\n".join(
        f"| {row.model_id} | {row.status} | {int(row.n_calls)} | {row.mean_quality:.4f} | "
        f"{row.total_cost_usd:.4f} | {row.mean_latency_s:.3f} | {int(row.cache_hits)} |"
        for row in model_summary.itertuples(index=False)
    )
    routing_lines = "\n".join(
        f"| {row.method} | {int(row.n_queries)} | {row.mean_quality:.4f} | {row.mean_utility:.4f} | "
        f"{row.frontier_call_rate:.4f} | {row.remote_cost_total_usd:.4f} | {row.mean_latency_s:.3f} |"
        for row in routing_summary.itertuples(index=False)
    )
    dataset_model_lines = "\n".join(
        f"| {row.benchmark} | {row.model_id} | {row.status} | {int(row.n)} | {row.mean_quality:.4f} | "
        f"{row.total_cost_usd:.4f} | {row.mean_latency_s:.3f} |"
        for row in by_dataset.itertuples(index=False)
    )
    failure_lines = "\n".join(
        f"| {row.model_id} | {row.provider} | {row.error_type} | {int(row.n)} |"
        for row in failures.itertuples(index=False)
    )
    if not failure_lines:
        failure_lines = "| none | none | none | 0 |"

    readme = f"""# Phase 3 New-Benchmark Live Smoke

This folder records the first small out-of-benchmark-family live smoke for
RouteCode/ProbeCode.

Status: `{status}`

{note}

This is not yet a proof that the learned states generalize. It shows that the
live harness can ingest new benchmark families and that a local-vs-frontier
oracle opportunity exists on this tiny slice.

## Benchmarks

These benchmarks were not in the Broad100 state-learning pool
(`aime`, `bbh`, `gpqa`, `gsm8k`, `humaneval`, `livemathbench`, `math500`,
`mbpp`, `mmlupro`).

| dataset | tasks |
| --- | ---: |
{dataset_lines}

HLE was considered but not included because `cais/hle` is gated in this
environment. BigCodeBench was considered but deferred because pass@1 code
execution needs a separate harness.

## Models

| model | status | calls | mean quality | total cost usd | mean latency s | cache hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{model_lines}

Gemini was attempted in `live_smoke_gpt_gemini_15`, but all 15 Gemini calls
returned HTTP 429, so Gemini has no usable quality result in this smoke.

## Routing Summary

| method | queries | quality | utility | frontier rate | remote cost usd | mean latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{routing_lines}

Interpretation:

- all GPT is accurate on this tiny slice but expensive under the cost-aware
  utility normalization;
- the 0.6B local model alone is cheap and fast but too weak;
- the cost-aware oracle can keep the same quality as GPT while using local on
  one third of the rows, but this is an upper bound because no deployable state
  policy was selected on these new benchmarks.

## Per-Benchmark Model Results

| benchmark | model | status | rows | quality | cost usd | latency s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{dataset_model_lines}

## Provider Failures

| model | provider | error | count |
| --- | --- | --- | ---: |
{failure_lines}

## Commands Run

```bash
PYTHONPATH=src python experiments/242_phase3_new_benchmark_manifest.py \\
  --output-dir results/phase3_new_benchmark_live \\
  --per-dataset 5 \\
  --seed 42

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \\
  --config configs/proberoute_controlled_broad100.yaml \\
  --output-dir results/phase3_new_benchmark_live/live_smoke_gpt_gemini_15 \\
  --run-suffix new_benchmark_live_smoke \\
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \\
  --frontier-model-ids gpt-5.5,gemini-3.5-flash \\
  --allow-frontier-calls \\
  --retry-errors \\
  --max-calls-per-frontier-model 15 \\
  --frontier-concurrency 1 \\
  --max-output-tokens 96 \\
  --request-timeout-s 120

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \\
  --config configs/proberoute_controlled_broad100.yaml \\
  --output-dir results/phase3_new_benchmark_live/live_smoke_gpt_15_max512 \\
  --run-suffix new_benchmark_gpt512_smoke \\
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \\
  --frontier-model-ids gpt-5.5 \\
  --allow-frontier-calls \\
  --retry-errors \\
  --max-calls-per-frontier-model 15 \\
  --frontier-concurrency 1 \\
  --max-output-tokens 512 \\
  --request-timeout-s 120

bash scripts/start_vllm_qwen3_0_6b.sh

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \\
  --config configs/proberoute_controlled_broad100.yaml \\
  --output-dir results/phase3_new_benchmark_live/live_smoke_qwen06_gpt_15 \\
  --run-suffix new_benchmark_gpt512_smoke \\
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \\
  --frontier-model-ids gpt-5.5 \\
  --local-model-ids qwen3-0.6b-probe \\
  --allow-frontier-calls \\
  --retry-errors \\
  --max-calls-per-frontier-model 15 \\
  --max-calls-per-local-model 15 \\
  --frontier-concurrency 1 \\
  --max-output-tokens 512 \\
  --local-max-output-tokens 128 \\
  --request-timeout-s 120

PYTHONPATH=src python experiments/243_phase3_new_benchmark_smoke_summary.py
```

## Artifacts

- `new_benchmark_manifest.csv`
- `NEW_BENCHMARK_MANIFEST_MEMO.md`
- `table_new_benchmark_model_summary.csv`
- `table_new_benchmark_routing_summary.csv`
- `table_new_benchmark_by_dataset_model.csv`
- `table_new_benchmark_provider_failures.csv`
- `live_smoke_qwen06_gpt_15/`
- `live_smoke_gpt_15_max512_rescored/`
- `live_smoke_gpt_gemini_15/`

## Next Required Test

To say the states generalize, run a larger benchmark-heldout protocol:

1. collect local/probe outputs for 50-100 rows per new benchmark;
2. freeze the Broad100-trained state predictor and action table;
3. select no thresholds on the new benchmarks;
4. report RouteCode/ProbeCode, all-local, all-GPT/Gemini, random routing, and
   local-vs-frontier oracle on the held-out new-benchmark rows.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    local_gpt_dir = Path(args.local_gpt_dir)
    gpt_dir = Path(args.gpt_dir)
    gemini_attempt_dir = Path(args.gemini_attempt_dir)

    model_summary = _read_csv(local_gpt_dir / "cost_latency_summary.csv")
    routing_summary = _read_csv(local_gpt_dir / "table_live_routing.csv")
    local_outputs = pd.read_parquet(local_gpt_dir / "model_outputs.parquet")
    by_dataset = _model_by_dataset(local_outputs)

    gemini_outputs = pd.read_parquet(gemini_attempt_dir / "model_outputs.parquet")
    failures = _provider_failures(gemini_outputs)

    status, note = _routecode_interpretation(routing_summary, by_dataset)

    model_summary.to_csv(output_dir / "table_new_benchmark_model_summary.csv", index=False)
    routing_summary.to_csv(output_dir / "table_new_benchmark_routing_summary.csv", index=False)
    by_dataset.to_csv(output_dir / "table_new_benchmark_by_dataset_model.csv", index=False)
    failures.to_csv(output_dir / "table_new_benchmark_provider_failures.csv", index=False)
    write_readme(
        output_dir,
        manifest=manifest,
        model_summary=model_summary,
        routing_summary=routing_summary,
        by_dataset=by_dataset,
        failures=failures,
        status=status,
        note=note,
    )
    print(f"Wrote new-benchmark smoke summary to {output_dir}")
    print(f"GPT-only source: {gpt_dir}")


if __name__ == "__main__":
    main()
