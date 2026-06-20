from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


RUNNABLE_DATASETS = {
    "gsm8k": {
        "display_name": "GSM8K",
        "domain": "math_easy",
        "task_type": "exact_final_answer",
        "source_type": "routellm_gsm8k_jsonl",
        "source_path": "data/raw/external/routellm/routellm/evals/gsm8k/test.jsonl",
        "source_split": "test",
        "max_output_tokens": 256,
    },
    "math500": {
        "display_name": "MATH500",
        "domain": "math",
        "task_type": "math",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "math500/test/Qwen3-8B",
        "source_split": "test",
        "max_output_tokens": 512,
    },
    "aime": {
        "display_name": "AIME",
        "domain": "math",
        "task_type": "math",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "aime/hybrid/Qwen3-8B",
        "source_split": "hybrid",
        "max_output_tokens": 512,
    },
    "livemathbench": {
        "display_name": "LiveMathBench",
        "domain": "math",
        "task_type": "math",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "livemathbench/test/Qwen3-8B",
        "source_split": "test",
        "max_output_tokens": 512,
    },
    "bbh": {
        "display_name": "BBH",
        "domain": "reasoning",
        "task_type": "exact_final_answer",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "bbh/test/Qwen3-8B",
        "source_split": "test",
        "max_output_tokens": 256,
    },
    "gpqa": {
        "display_name": "GPQA",
        "domain": "science",
        "task_type": "multiple_choice",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "gpqa/test/Qwen3-8B",
        "source_split": "test",
        "max_output_tokens": 256,
    },
    "mmlupro": {
        "display_name": "MMLU-Pro",
        "domain": "broad_knowledge",
        "task_type": "multiple_choice",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "mmlupro/test_1000/Qwen3-8B",
        "source_split": "test_1000",
        "max_output_tokens": 256,
    },
    "humaneval": {
        "display_name": "HumanEval",
        "domain": "code",
        "task_type": "pass_at_1",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "humaneval/test/Qwen3-8B",
        "source_split": "test",
        "max_output_tokens": 512,
    },
    "mbpp": {
        "display_name": "MBPP",
        "domain": "code",
        "task_type": "pass_at_1",
        "source_type": "llmrouterbench_result_json",
        "result_dir": "mbpp/test/Qwen3-8B",
        "source_split": "test",
        "max_output_tokens": 512,
    },
}


EXCLUDED_DATASETS = [
    {
        "benchmark": "LiveCodeBench",
        "dataset_key": "livecodebench",
        "status": "needs_full_lfs_test_payload",
        "reason": "The local checkout has result prompts, but the raw LiveCodeBench test payload is still a Git LFS pointer; sample-only scoring is not used as paper evidence.",
    },
]


MANIFEST_COLUMNS = [
    "query_id",
    "query_text",
    "dataset",
    "domain",
    "source_split",
    "routecode_split",
    "task_type",
    "gold_answer",
    "choices_json",
    "metadata_json",
    "max_output_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a controlled broad target-pool manifest from LLMRouterBench prompts.")
    parser.add_argument("--results-root", type=Path, default=Path("data/raw/external/LLMRouterBench/results/bench"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad_target_manifest"))
    parser.add_argument("--examples-per-dataset", type=int, default=5)
    parser.add_argument("--source-model", default="Qwen3-8B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, source_coverage = build_manifest(
        args.results_root,
        examples_per_dataset=args.examples_per_dataset,
        source_model=args.source_model,
    )
    manifest_path = args.output_dir / "broad_target_task_manifest.csv"
    coverage_path = args.output_dir / "table_broad_target_manifest_coverage.csv"
    excluded_path = args.output_dir / "table_broad_target_manifest_exclusions.csv"
    memo_path = args.output_dir / "BROAD_TARGET_MANIFEST_MEMO.md"
    manifest.to_csv(manifest_path, index=False)
    source_coverage.to_csv(coverage_path, index=False)
    pd.DataFrame(EXCLUDED_DATASETS).to_csv(excluded_path, index=False)
    write_memo(memo_path, manifest_path, manifest, source_coverage)
    print(f"Wrote broad target manifest to {manifest_path}")
    print(f"runnable_datasets={manifest['dataset'].nunique()}; tasks={len(manifest)}")


def build_manifest(
    results_root: Path,
    *,
    examples_per_dataset: int,
    source_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for dataset, spec in RUNNABLE_DATASETS.items():
        if spec.get("source_type") == "routellm_gsm8k_jsonl":
            selected, coverage = _select_gsm8k_records(dataset, spec, examples_per_dataset)
            rows.extend(selected)
            coverage_rows.append(coverage)
            continue
        result_path = _latest_result_file(results_root / str(spec["result_dir"]))
        if result_path is None:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "display_name": spec["display_name"],
                    "status": "missing_result_json",
                    "source_records": 0,
                    "selected_tasks": 0,
                    "source_path": "",
                    "notes": f"No result JSON found for source model {source_model}.",
                }
            )
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        selected = []
        for record in records:
            prompt = str(record.get("prompt") or "").strip()
            gold = _gold_for_record(dataset, spec, record)
            if not gold or not prompt:
                continue
            selected.append(record)
            if len(selected) >= int(examples_per_dataset):
                break
        coverage_rows.append(
            {
                "dataset": dataset,
                "display_name": spec["display_name"],
                "status": "ready" if selected else "no_scored_gold_records",
                "source_records": len(records),
                "selected_tasks": len(selected),
                "source_path": str(result_path),
                "notes": "",
            }
        )
        for record in selected:
            query_id = f"{dataset}:{spec['source_split']}:{int(record.get('index'))}"
            gold = _gold_for_record(dataset, spec, record)
            task_type = str(spec["task_type"])
            metadata = {
                "source": "LLMRouterBench result prompt",
                "source_model": source_model,
                "source_result_path": str(result_path),
                "source_record_index": record.get("index"),
                "origin_query": record.get("origin_query", ""),
                "source_score": record.get("score"),
                "scoring": "embedded_python_asserts" if task_type == "pass_at_1" else "exact_or_multiple_choice",
            }
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": str(record["prompt"]).strip(),
                    "dataset": dataset,
                    "domain": spec["domain"],
                    "source_split": spec["source_split"],
                    "routecode_split": "stage0",
                    "task_type": task_type,
                    "gold_answer": gold,
                    "choices_json": "[]",
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                    "max_output_tokens": int(spec["max_output_tokens"]),
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS), pd.DataFrame(coverage_rows)


def _select_gsm8k_records(
    dataset: str,
    spec: dict[str, Any],
    examples_per_dataset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(str(spec["source_path"]))
    if not path.exists():
        return [], {
            "dataset": dataset,
            "display_name": spec["display_name"],
            "status": "missing_source_jsonl",
            "source_records": 0,
            "selected_tasks": 0,
            "source_path": str(path),
            "notes": "RouteLLM GSM8K test JSONL is not present.",
        }
    selected: list[dict[str, Any]] = []
    source_records = 0
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            source_records += 1
            record = json.loads(line)
            question = str(record.get("question", "")).strip()
            gold = _extract_gsm8k_final_answer(str(record.get("answer", "")))
            if not question or not gold:
                continue
            metadata = {
                "source": "RouteLLM GSM8K test JSONL",
                "source_path": str(path),
                "source_record_index": idx,
                "scoring": "exact_final_answer",
            }
            selected.append(
                {
                    "query_id": f"{dataset}:{spec['source_split']}:{idx}",
                    "query_text": question,
                    "dataset": dataset,
                    "domain": spec["domain"],
                    "source_split": spec["source_split"],
                    "routecode_split": "stage0",
                    "task_type": spec["task_type"],
                    "gold_answer": gold,
                    "choices_json": "[]",
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                    "max_output_tokens": int(spec["max_output_tokens"]),
                }
            )
            if len(selected) >= int(examples_per_dataset):
                break
    coverage = {
        "dataset": dataset,
        "display_name": spec["display_name"],
        "status": "ready" if selected else "no_exact_gold_records",
        "source_records": source_records,
        "selected_tasks": len(selected),
        "source_path": str(path),
        "notes": "Loaded from local RouteLLM checkout because GSM8K is absent from the LLMRouterBench broad20 matrix.",
    }
    return selected, coverage


def _extract_gsm8k_final_answer(answer: str) -> str:
    marker = "####"
    if marker in answer:
        return answer.rsplit(marker, 1)[1].strip().replace(",", "")
    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    return numbers[-1].replace(",", "") if numbers else ""


def _gold_for_record(dataset: str, spec: dict[str, Any], record: dict[str, Any]) -> str:
    if str(spec["task_type"]) != "pass_at_1":
        return _clean_gold(record.get("ground_truth"))
    prompt = str(record.get("prompt") or "")
    tests = _extract_embedded_code_tests(prompt)
    if not tests:
        return ""
    origin = str(record.get("origin_query") or prompt)
    entry_point = _extract_entry_point(origin)
    if dataset == "humaneval" and not entry_point:
        return ""
    payload = {
        "benchmark": dataset,
        "entry_point": entry_point,
        "tests": tests,
        "source_record_index": record.get("index"),
    }
    return json.dumps(payload, sort_keys=True)


def _extract_embedded_code_tests(prompt: str) -> str:
    marker = "Your code should pass these tests:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].strip()


def _extract_entry_point(text: str) -> str:
    match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", text)
    return match.group(1) if match else ""


def _latest_result_file(path: Path) -> Path | None:
    files = sorted(path.glob("*.json"))
    if not files:
        return None
    return max(files, key=lambda candidate: candidate.stat().st_mtime)


def _clean_gold(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def write_memo(path: Path, manifest_path: Path, manifest: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# Broad Target Manifest Memo",
        "",
        "This manifest starts the controlled broad target-pool run from real LLMRouterBench prompts.",
        "It uses released result JSON records only as prompt/gold sources; it does not use their model predictions as target-pool outputs.",
        "",
        f"Manifest: `{manifest_path}`",
        "",
        "## Runnable Scope",
        "",
        markdown_table(coverage),
        "",
        "## Exclusions",
        "",
        markdown_table(pd.DataFrame(EXCLUDED_DATASETS)),
        "",
        "## Manifest Counts",
        "",
        f"- Runnable datasets: `{manifest['dataset'].nunique() if not manifest.empty else 0}`.",
        f"- Tasks: `{len(manifest)}`.",
        "",
        "## Use",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/81_controlled_live_stage0.py \\",
        f"  --task-manifest {manifest_path} \\",
        "  --output-dir results/controlled/live_broad_stage0 \\",
        "  --run-suffix live_broad_stage0 \\",
        "  --allow-frontier-calls \\",
        "  --max-output-tokens 256 \\",
        "  --frontier-concurrency 2",
        "```",
        "",
        "HumanEval and MBPP use embedded function/assert checks for limited pass@1 scoring. LiveCodeBench remains excluded until the full local test payload is available.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
