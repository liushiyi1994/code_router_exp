from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import load_env_values, resolve_key


GPT_MODEL = "gpt-5.5"
VARIABLE_VERIFIER_ID = "gpt-5.5-variable-option-mcq-verifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variable-option GPT MCQ verifier policy for GPQA/MMLUPro.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--benchmark-composed-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_tool_aware_benchmark_composed_policy/"
            "table_tool_aware_benchmark_composed_choices.csv"
        ),
    )
    parser.add_argument("--benchmark-composed-method", default="tool_aware_benchmark_composed_eps0.01_recall_then_quality")
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_variable_option_mcq_verifier_policy"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--benchmarks", default="gpqa,mmlupro")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--max-api-spend-usd", type=float, default=1.00)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_190")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_190")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_190")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_190")
    exp179 = load_module("experiments/179_cached_adjudicator_blend_policy.py", "adjudicator_blend_179_for_190")
    exp181 = load_module("experiments/181_task_specific_verifier_action.py", "task_verifier_181_for_190")
    strict = load_module("experiments/184_strict_mcq_verifier_policy.py", "strict_mcq_184_for_190")
    patch_strict_module(strict)

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    benchmarks = {item.strip().lower() for item in args.benchmarks.split(",") if item.strip()}
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    query_frame = (
        outputs[
            outputs["split"].astype(str).isin(splits)
            & outputs["benchmark"].astype(str).str.lower().isin(benchmarks)
        ]
        .drop_duplicates("query_id")
        .copy()
    )
    env = load_env_values(args.env_file)
    api_key = resolve_key(env, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for GPT verifier.")

    verifier = strict.collect_rows(
        query_frame,
        outputs,
        args.output_dir,
        api_key=api_key,
        max_output_tokens=int(args.max_output_tokens),
        max_api_spend_usd=float(args.max_api_spend_usd),
        concurrency=int(args.concurrency),
        force_rerun=bool(args.force_rerun),
    )
    augmented = strict.append_verifier_outputs(outputs, verifier, lambda_cost=float(args.lambda_cost))
    base_choices = exp181.practical_base_choices(target, outputs, exp172, exp175, exp177, exp179)
    policy_table, query_choices = strict.evaluate_policies(
        base_choices,
        verifier,
        augmented,
        target,
        exp172,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = strict.selected_rows(policy_table, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))

    verifier.to_csv(args.output_dir / "table_variable_option_mcq_verifier_outputs.csv", index=False)
    augmented.to_parquet(args.output_dir / "model_outputs_with_variable_option_mcq_verifier.parquet", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_variable_option_mcq_verifier_policy_all.csv",
        index=False,
    )
    selected.to_csv(args.output_dir / "table_variable_option_mcq_verifier_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_variable_option_mcq_verifier_query_choices.csv", index=False)
    write_memo(args.output_dir / "VARIABLE_OPTION_MCQ_VERIFIER_POLICY_MEMO.md", args, verifier, policy_table, selected)
    print(f"Wrote variable-option MCQ verifier results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_strict_module(strict) -> None:
    strict.STRICT_VERIFIER_ID = VARIABLE_VERIFIER_ID
    strict.prompt_for = variable_prompt_for
    strict.parse_json = variable_parse_json
    strict.cache_name = variable_cache_name


def variable_prompt_for(row: pd.Series, candidates: pd.DataFrame) -> str:
    benchmark = str(row.get("benchmark", ""))
    option_letters = infer_option_letters(str(row.get("query_text", "")), candidates)
    keep_models = [
        "qwen3-4b-local",
        "qwen3-8b-local",
        "qwen3-14b-awq-local",
        "qwen3-32b-awq-local",
        "qwen3-32b-awq-selfconsistency-n3-local",
        "gemini-3.5-flash",
        "gpt-5.5",
        "gemini-3.5-flash-strong-solve",
    ]
    lines: list[str] = []
    for model_id in keep_models:
        group = candidates[candidates["model_id"].astype(str).eq(model_id)]
        if group.empty:
            continue
        answer = compact(str(group.iloc[0].get("parsed_answer", "")), 80).upper()
        if not answer or answer.lower() in {"nan", "none"}:
            answer = "[empty]"
        lines.append(f"{model_id}: {answer}")
    allowed = "|".join(option_letters)
    return (
        "Answer this multiple-choice question. Return compact JSON only. Do not explain. Do not show reasoning.\n"
        f"The answer must be exactly one option letter from: {allowed}.\n"
        f"The JSON schema is exactly: {{\"answer\":\"{allowed}\",\"confidence\":0.0,\"supported_model\":\"MODEL_OR_NONE\"}}.\n"
        "Pick supported_model from the candidate list only if that candidate has the same final option you trust.\n"
        "If no candidate matches your trusted option, use supported_model NONE.\n"
        f"Benchmark: {benchmark}\n"
        f"Question:\n{compact(str(row['query_text']), 2300)}\n\n"
        "Candidate final options:\n"
        + "\n".join(lines)
        + "\nReturn only the JSON object."
    )


def variable_parse_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    answer = str(parsed.get("answer", "")).strip().upper()
    if not re.fullmatch(r"[A-J]", answer):
        option = re.search(r'"answer"\s*:\s*"([A-J])"', clean.upper())
        if option is None:
            option = re.search(r"\b([A-J])\b", clean.upper())
        answer = option.group(1) if option else clean[:16]
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer": answer,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "supported_model": str(parsed.get("supported_model", "")),
    }


def infer_option_letters(query_text: str, candidates: pd.DataFrame) -> list[str]:
    letters = sorted(set(re.findall(r"(?m)(?:^|\n)\s*([A-J])(?:[\).:]|\s*[-])\s+", query_text)))
    if not letters:
        candidate_letters = set()
        for value in candidates.get("parsed_answer", pd.Series(dtype=str)).astype(str):
            upper = value.strip().upper()
            if re.fullmatch(r"[A-J]", upper):
                candidate_letters.add(upper)
        letters = sorted(candidate_letters)
    if not letters:
        letters = list("ABCD")
    return letters


def variable_cache_name(query_id: str, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{GPT_MODEL}:{max_output_tokens}:variable_mcq_v1".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Variable-Option MCQ Verifier Policy",
        "",
        "This reruns the strict GPT-5.5 MCQ verifier with an A-J option parser.",
        "It fixes the Experiment 184 issue where MMLU-Pro was constrained to A-D.",
        "Claude is not used.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/190_variable_option_mcq_verifier_policy.py",
        "```",
        "",
        "## Verifier Diagnostics",
        "",
        f"- Rows: `{len(verifier)}`",
        f"- Cache-hit rate: `{float(verifier['cache_hit'].mean()) if not verifier.empty else 0.0:.4f}`",
        f"- GPT verifier cost total: `${float(verifier['cost_total_usd'].sum()) if not verifier.empty else 0.0:.4f}`",
        "",
    ]
    if not verifier.empty:
        diag = verifier.groupby(["benchmark", "split"], as_index=False).agg(
            n=("query_id", "count"),
            quality=("quality_score", "mean"),
            mean_confidence=("verifier_confidence", "mean"),
        )
        lines.extend([markdown_table(diag), ""])
    lines.extend(["## Selected Rows", ""])
    cols = [
        "selection_rule",
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_probe_cost",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "frontier_call_rate",
        "probe_call_rate",
        "override_rate",
    ]
    if selected.empty:
        lines.append("No selected rows were produced.")
    else:
        lines.append(markdown_table(selected[cols]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare to Experiment 184 strict MCQ verifier and Experiment 189 targeted residual repair.",
            "- Raw selected-action utility does not include verifier probe cost unless the verifier answer is selected as the final action.",
            "- Probe-cost utility charges the GPT verifier when used only as a routing probe.",
            "",
            "## Artifacts",
            "",
            f"- Verifier rows: `{path.parent / 'table_variable_option_mcq_verifier_outputs.csv'}`",
            f"- All policies: `{path.parent / 'table_variable_option_mcq_verifier_policy_all.csv'}`",
            f"- Selected policies: `{path.parent / 'table_variable_option_mcq_verifier_policy_selected.csv'}`",
            f"- Query choices: `{path.parent / 'table_variable_option_mcq_verifier_query_choices.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
