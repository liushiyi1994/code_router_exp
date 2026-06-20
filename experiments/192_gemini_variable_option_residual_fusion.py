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


GEMINI_MODEL = "gemini-3.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini variable-option MCQ support fused with residual repair.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--benchmarks", default="mmlupro")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--max-api-spend-usd", type=float, default=0.50)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_gemini_variable_option_residual_fusion"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = load_module("experiments/181_task_specific_verifier_action.py", "task_verifier_181_for_192")
    fusion = load_module("experiments/191_variable_verifier_residual_fusion.py", "fusion_191_for_192")
    patch_task_module(task)

    outputs = pd.read_parquet(args.outputs)
    benchmarks = {item.strip().lower() for item in str(args.benchmarks).split(",") if item.strip()}
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    query_frame = (
        outputs[
            outputs["split"].astype(str).isin(splits)
            & outputs["benchmark"].astype(str).str.lower().isin(benchmarks)
        ]
        .drop_duplicates("query_id")
        .copy()
    )
    env = load_env_values(args.env_file)
    api_key = resolve_key(env, ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise RuntimeError("Missing Gemini API key.")
    verifier = task.collect_verifier_rows(
        query_frame,
        outputs,
        args.output_dir,
        api_key=api_key,
        provider_model=GEMINI_MODEL,
        max_output_tokens=int(args.max_output_tokens),
        max_api_spend_usd=float(args.max_api_spend_usd),
        concurrency=int(args.concurrency),
        force_rerun=bool(args.force_rerun),
    )
    verifier.to_csv(args.output_dir / "table_gemini_variable_option_verifier_outputs.csv", index=False)

    fusion_args = argparse.Namespace(
        base_query_choices=Path(
            "results/controlled/broad100_targeted_residual_repair_policy/"
            "table_targeted_residual_repair_query_choices.csv"
        ),
        base_policy=fusion.BASE_POLICY,
        verifier=args.output_dir / "table_gemini_variable_option_verifier_outputs.csv",
        outputs=args.outputs,
        output_dir=args.output_dir,
        lambda_cost=float(args.lambda_cost),
        reliable_quality_threshold=0.85,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    outputs_eval = pd.read_parquet(fusion_args.outputs).copy()
    outputs_eval["utility"] = (
        outputs_eval["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs_eval["normalized_remote_cost"].astype(float)
    )
    base = pd.read_csv(fusion_args.base_query_choices)
    base = base[base["policy"].astype(str).eq(str(fusion_args.base_policy))].copy()
    action_map = {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs_eval.groupby("query_id")}
    oracle = outputs_eval.loc[outputs_eval.groupby("query_id")["utility"].idxmax()][["query_id", "utility", "quality_score"]].rename(
        columns={"utility": "oracle_utility", "quality_score": "oracle_quality"}
    )
    base = fusion.drop_prefixed(base, ["oracle_utility", "oracle_quality"]).merge(oracle, on="query_id", how="left")
    verifier_diag = fusion.verifier_diagnostics(verifier)
    rules = fusion.enumerate_rules()
    policy_table, query_choices = fusion.evaluate_rules(base, verifier, outputs_eval, action_map, rules, fusion_args)
    reliable_raw = fusion.build_reliable_policy(
        policy_table,
        verifier_diag,
        base,
        metric="mean_utility",
        quality_threshold=0.85,
    )
    reliable_costed = fusion.build_reliable_policy(
        policy_table,
        verifier_diag,
        base,
        metric="mean_utility_with_probe_cost",
        quality_threshold=0.85,
    )
    if reliable_raw:
        rules.append(reliable_raw)
    if reliable_costed:
        rules.append(reliable_costed)
    if reliable_raw or reliable_costed:
        policy_table, query_choices = fusion.evaluate_rules(base, verifier, outputs_eval, action_map, rules, fusion_args)
    selected = fusion.selected_rows(policy_table, fusion_args)
    selected_policies = set(selected["policy"].astype(str).tolist()) if not selected.empty else set()
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    verifier_diag.to_csv(args.output_dir / "table_gemini_variable_fusion_verifier_diagnostics.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output_dir / "table_gemini_variable_fusion_rule_library.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_gemini_variable_fusion_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_gemini_variable_fusion_policy_selected.csv", index=False)
    query_choices_to_write.to_csv(args.output_dir / "table_gemini_variable_fusion_query_choices.csv", index=False)
    write_memo(args.output_dir / "GEMINI_VARIABLE_OPTION_RESIDUAL_FUSION_MEMO.md", args, verifier, verifier_diag, selected)
    print(f"Wrote Gemini variable-option residual fusion results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_task_module(task) -> None:
    task.prompt_for = variable_prompt_for
    task.parse_verifier_json = variable_parse_json
    task.cache_name = variable_cache_name


def variable_prompt_for(row: pd.Series, candidates: pd.DataFrame) -> str:
    letters = infer_option_letters(str(row.get("query_text", "")), candidates)
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
    lines = []
    for model_id in keep_models:
        group = candidates[candidates["model_id"].astype(str).eq(model_id)]
        if group.empty:
            continue
        answer = compact(str(group.iloc[0].get("parsed_answer", "")), 80).upper()
        if not answer or answer.lower() in {"nan", "none"}:
            answer = "[empty]"
        lines.append(f"{model_id}: {answer}")
    allowed = "|".join(letters)
    return (
        "Answer this multiple-choice question. Return compact JSON only. Do not explain.\n"
        f"The answer must be exactly one option letter from: {allowed}.\n"
        f"Use schema: {{\"answer\":\"{allowed}\",\"confidence\":0.0,\"supported_model\":\"MODEL_OR_NONE\",\"reason\":\"short\"}}.\n"
        "Pick supported_model from the candidate list only if that candidate has the same final option you trust.\n"
        f"Benchmark: {row.get('benchmark')}\n"
        f"Question:\n{compact(str(row['query_text']), 2300)}\n\n"
        "Candidate final options:\n"
        + "\n".join(lines)
    )


def variable_parse_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I).strip()
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    parsed: dict[str, Any] = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    answer = str(parsed.get("answer", "")).strip().upper()
    if not re.fullmatch(r"[A-J]", answer):
        option = re.search(r'"answer"\s*:\s*"([A-J])"', clean.upper()) or re.search(r"\b([A-J])\b", clean.upper())
        answer = option.group(1) if option else clean[:16]
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer": answer,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "supported_model": str(parsed.get("supported_model", "")),
        "reason": compact(str(parsed.get("reason", "")), 220),
    }


def infer_option_letters(query_text: str, candidates: pd.DataFrame) -> list[str]:
    letters = sorted(set(re.findall(r"(?m)(?:^|\n)\s*([A-J])(?:[\).:]|\s*[-])\s+", query_text)))
    if not letters:
        found = set()
        for value in candidates.get("parsed_answer", pd.Series(dtype=str)).astype(str):
            upper = value.strip().upper()
            if re.fullmatch(r"[A-J]", upper):
                found.add(upper)
        letters = sorted(found)
    return letters or list("ABCD")


def variable_cache_name(query_id: str, provider_model: str, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{provider_model}:{max_output_tokens}:gemini_variable_mcq_v1".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, verifier_diag: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Gemini Variable-Option Residual Fusion",
        "",
        "This tests Gemini 3.5 Flash as a cheaper variable-option MCQ support verifier, fused with Experiment 189 residual repair.",
        "Claude is not used.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/192_gemini_variable_option_residual_fusion.py",
        "```",
        "",
        "## Verifier Rows",
        "",
        f"- Rows: `{len(verifier)}`",
        f"- Success rows: `{int(verifier['status'].astype(str).eq('success').sum()) if not verifier.empty else 0}`",
        f"- Cache-hit rate: `{float(verifier['cache_hit'].mean()) if not verifier.empty else 0.0:.4f}`",
        f"- Gemini verifier cost total: `${float(verifier['cost_total_usd'].sum()) if not verifier.empty else 0.0:.4f}`",
        "",
        markdown_table(verifier_diag) if not verifier_diag.empty else "No verifier diagnostics.",
        "",
        "## Selected Rows",
        "",
        markdown_table(
            selected[
                [
                    "selection_rule",
                    "policy",
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
            ]
        )
        if not selected.empty
        else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- Compare against Experiment 189 (`0.7736` held-out utility) and Experiment 191 GPT verifier fusion.",
        "- Probe-cost utility is the route-time metric because Gemini verifier calls are paid probes unless selected as the final answer.",
        "",
        "## Artifacts",
        "",
        f"- Verifier outputs: `{path.parent / 'table_gemini_variable_option_verifier_outputs.csv'}`",
        f"- Policy table: `{path.parent / 'table_gemini_variable_fusion_policy_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_gemini_variable_fusion_policy_selected.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
