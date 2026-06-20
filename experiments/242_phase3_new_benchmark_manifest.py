from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small new-benchmark live manifest for Phase 3.")
    parser.add_argument("--output-dir", default="results/phase3_new_benchmark_live")
    parser.add_argument("--per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        default="simpleqa_verified,livebench_math,livebench_reasoning",
        help="Comma-separated dataset keys to include.",
    )
    return parser.parse_args()


def _require_datasets():
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("Install the optional Hugging Face datasets package to build this manifest.") from exc
    return get_dataset_split_names, load_dataset


def _turn_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, list) and parsed:
        return str(parsed[0])
    return text


def _sample_frame(dataset: Any, n: int, seed: int) -> pd.DataFrame:
    count = min(int(n), len(dataset))
    if count <= 0:
        return pd.DataFrame()
    sampled = dataset.shuffle(seed=int(seed)).select(range(count))
    return sampled.to_pandas()


def _clean_short_answer(answer: Any) -> str:
    text = str(answer).strip()
    marker = "(acceptable range:"
    if marker in text.lower():
        start = text.lower().find(marker)
        text = text[:start].strip()
    return text


def build_simpleqa(per_dataset: int, seed: int) -> pd.DataFrame:
    _, load_dataset = _require_datasets()
    data = load_dataset("google/simpleqa-verified", "simpleqa_verified", split="eval")
    frame = _sample_frame(data, per_dataset, seed)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        topic = str(getattr(row, "topic", "factoid") or "factoid").strip() or "factoid"
        query_id = f"simpleqa_verified:{getattr(row, 'original_index')}"
        rows.append(
            {
                "query_id": query_id,
                "query_text": str(getattr(row, "problem")),
                "dataset": "simpleqa_verified",
                "domain": topic.lower().replace(" ", "_"),
                "source_split": "eval",
                "routecode_split": "new_benchmark",
                "task_type": "short_answer",
                "gold_answer": _clean_short_answer(getattr(row, "answer")),
                "choices_json": "[]",
                "metadata_json": json.dumps(
                    {
                        "hf_dataset": "google/simpleqa-verified",
                        "topic": topic,
                        "answer_type": str(getattr(row, "answer_type", "")),
                        "multi_step": bool(getattr(row, "multi_step", False)),
                        "requires_reasoning": bool(getattr(row, "requires_reasoning", False)),
                    },
                    sort_keys=True,
                ),
                "max_output_tokens": 64,
            }
        )
    return pd.DataFrame(rows)


def build_livebench_math(per_dataset: int, seed: int) -> pd.DataFrame:
    _, load_dataset = _require_datasets()
    data = load_dataset("livebench/math", split="test")
    frame = _sample_frame(data, per_dataset, seed)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        task = str(getattr(row, "task", "math") or "math")
        rows.append(
            {
                "query_id": f"livebench_math:{getattr(row, 'question_id')}",
                "query_text": _turn_text(getattr(row, "turns")),
                "dataset": "livebench_math",
                "domain": "math",
                "source_split": "test",
                "routecode_split": "new_benchmark",
                "task_type": "exact_ordered",
                "gold_answer": str(getattr(row, "ground_truth")),
                "choices_json": "[]",
                "metadata_json": json.dumps(
                    {
                        "hf_dataset": "livebench/math",
                        "category": str(getattr(row, "category", "")),
                        "task": task,
                        "subtask": str(getattr(row, "subtask", "")),
                        "livebench_release_date": str(getattr(row, "livebench_release_date", "")),
                    },
                    sort_keys=True,
                ),
                "max_output_tokens": 96,
            }
        )
    return pd.DataFrame(rows)


def build_livebench_reasoning(per_dataset: int, seed: int) -> pd.DataFrame:
    _, load_dataset = _require_datasets()
    data = load_dataset("livebench/reasoning", split="test")
    frame = _sample_frame(data, per_dataset, seed)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        task = str(getattr(row, "task", "reasoning") or "reasoning")
        rows.append(
            {
                "query_id": f"livebench_reasoning:{getattr(row, 'question_id')}",
                "query_text": _turn_text(getattr(row, "turns")),
                "dataset": "livebench_reasoning",
                "domain": "reasoning",
                "source_split": "test",
                "routecode_split": "new_benchmark",
                "task_type": "exact_ordered",
                "gold_answer": str(getattr(row, "ground_truth")),
                "choices_json": "[]",
                "metadata_json": json.dumps(
                    {
                        "hf_dataset": "livebench/reasoning",
                        "category": str(getattr(row, "category", "")),
                        "task": task,
                        "level": str(getattr(row, "level", "")),
                        "livebench_release_date": str(getattr(row, "livebench_release_date", "")),
                    },
                    sort_keys=True,
                ),
                "max_output_tokens": 96,
            }
        )
    return pd.DataFrame(rows)


BUILDERS = {
    "simpleqa_verified": build_simpleqa,
    "livebench_math": build_livebench_math,
    "livebench_reasoning": build_livebench_reasoning,
}


def hle_status() -> str:
    get_dataset_split_names, _ = _require_datasets()
    try:
        splits = get_dataset_split_names("cais/hle")
    except Exception as exc:  # noqa: BLE001 - record access status from HF client.
        return f"not included: Hugging Face access failed with {type(exc).__name__}: {exc}"
    return f"available but not sampled in this smoke; splits={splits}"


def write_memo(output_dir: Path, manifest_path: Path, manifest: pd.DataFrame, *, hle_note: str) -> Path:
    counts = manifest.groupby("dataset").size().reset_index(name="n_tasks")
    count_lines = "\n".join(f"| {row.dataset} | {int(row.n_tasks)} |" for row in counts.itertuples(index=False))
    memo = f"""# Phase 3 New-Benchmark Live Manifest

Manifest: `{manifest_path}`

This is a small live smoke for out-of-benchmark-family evaluation. It is not a
final generalization claim by itself; it is a bounded first check using
benchmarks that were not part of the Broad100 state-learning pool.

## Included Benchmarks

| dataset | tasks |
| --- | ---: |
{count_lines}

## Benchmark Decisions

- `google/simpleqa-verified`: included because it is an accessible factoid QA
  benchmark with gold short answers.
- `livebench/math`: included because it is a newer live benchmark with exact
  ordered answers.
- `livebench/reasoning`: included because it is a newer live benchmark with
  exact ordered answers.
- `cais/hle`: {hle_note}
- `bigcode/bigcodebench`: accessible, but deferred because pass@1 code execution
  is a separate harness from this exact-answer smoke.

## Scoring Notes

LiveBench rows use `task_type=exact_ordered`, so ordered comma-separated answers
must match in order. SimpleQA Verified uses `task_type=exact_final_answer`.

## Follow-Up Needed

To support a strong state-generalization claim, collect local vLLM probe
behavior on these same rows and evaluate a frozen Broad100-trained state
predictor without selecting thresholds on the new benchmarks.
"""
    path = output_dir / "NEW_BENCHMARK_MANIFEST_MEMO.md"
    path.write_text(memo, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(BUILDERS))
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}")

    frames = [BUILDERS[name](args.per_dataset, args.seed + idx) for idx, name in enumerate(requested)]
    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest = manifest.drop_duplicates("query_id", keep="first")
    manifest_path = output_dir / "new_benchmark_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    memo_path = write_memo(output_dir, manifest_path, manifest, hle_note=hle_status())
    print(f"Wrote {len(manifest)} rows to {manifest_path}")
    print(f"Wrote memo to {memo_path}")


if __name__ == "__main__":
    main()
