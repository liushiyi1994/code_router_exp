# Phase 3 New-Benchmark Live Manifest

Manifest: `results/phase3_new_benchmark_live/new_benchmark_manifest.csv`

This is a small live smoke for out-of-benchmark-family evaluation. It is not a
final generalization claim by itself; it is a bounded first check using
benchmarks that were not part of the Broad100 state-learning pool.

## Included Benchmarks

| dataset | tasks |
| --- | ---: |
| livebench_math | 5 |
| livebench_reasoning | 5 |
| simpleqa_verified | 5 |

## Benchmark Decisions

- `google/simpleqa-verified`: included because it is an accessible factoid QA
  benchmark with gold short answers.
- `livebench/math`: included because it is a newer live benchmark with exact
  ordered answers.
- `livebench/reasoning`: included because it is a newer live benchmark with
  exact ordered answers.
- `cais/hle`: not included: Hugging Face access failed with DatasetNotFoundError: Dataset 'cais/hle' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/cais/hle to ask for access.
- `bigcode/bigcodebench`: accessible, but deferred because pass@1 code execution
  is a separate harness from this exact-answer smoke.

## Scoring Notes

LiveBench rows use `task_type=exact_ordered`, so ordered comma-separated answers
must match in order. SimpleQA Verified uses `task_type=exact_final_answer`.

## Follow-Up Needed

To support a strong state-generalization claim, collect local vLLM probe
behavior on these same rows and evaluate a frozen Broad100-trained state
predictor without selecting thresholds on the new benchmarks.
