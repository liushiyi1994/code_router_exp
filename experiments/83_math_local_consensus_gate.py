from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import normalize_answer, score_output


FRONTIER_MODELS = ("gemini-3.5-flash", "gpt-5.5")
LOCAL_MODELS = ("qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MATH500 local-consensus routing with cached live outputs.")
    parser.add_argument("--run-dir", default="results/controlled/math500_qwen8_live_pilot_1024")
    parser.add_argument("--manifest", default="results/phase2/all200_exact_task_manifest/local_exact_task_manifest.csv")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def load_outputs(run_dir: Path, manifest_path: Path, dataset: str, lambda_cost: float) -> tuple[pd.DataFrame, float]:
    outputs = pd.read_parquet(run_dir / "model_outputs.parquet")
    outputs = outputs[outputs["status"].eq("success")].copy()
    rescored = [
        score_output(str(parsed), str(gold), str(metric))
        for parsed, gold, metric in zip(outputs["parsed_answer"], outputs["gold_answer"], outputs["metric"])
    ]
    outputs["parsed_answer"] = [parsed for parsed, _ in rescored]
    outputs["quality_score"] = [quality for _, quality in rescored]
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["dataset"].eq(dataset)][["query_id", "routecode_split"]].copy()
    outputs = outputs.merge(manifest, on="query_id", how="left")
    gpt_cost = outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()), 1e-12)
    outputs["utility"] = outputs["quality_score"] - lambda_cost * outputs["cost_total_usd"] / cost_norm
    return outputs, cost_norm


def iter_queries(outputs: pd.DataFrame, split: str):
    required = set(FRONTIER_MODELS) | set(LOCAL_MODELS)
    for query_id, group in outputs[outputs["routecode_split"].eq(split)].groupby("query_id", sort=True):
        by_model = group.drop_duplicates("model_id").set_index("model_id")
        if required.issubset(set(by_model.index)):
            yield query_id, by_model


def all_model(model_id: str) -> Callable[[pd.DataFrame], tuple[str, float]]:
    return lambda _by_model: (model_id, 0.0)


def local_4b_8b_agreement_else_gemini(by_model: pd.DataFrame) -> tuple[str, float]:
    answer_4b = normalize_answer(by_model.loc["qwen3-4b-local", "parsed_answer"])
    answer_8b = normalize_answer(by_model.loc["qwen3-8b-local", "parsed_answer"])
    if answer_4b and answer_4b == answer_8b:
        return "qwen3-4b-local", 2.0 / 3.0
    return "gemini-3.5-flash", 2.0 / 3.0


def local_majority_else_gemini(by_model: pd.DataFrame) -> tuple[str, float]:
    answers = {model_id: normalize_answer(by_model.loc[model_id, "parsed_answer"]) for model_id in LOCAL_MODELS}
    counts = collections.Counter(answer for answer in answers.values() if answer)
    if counts and counts.most_common(1)[0][1] >= 2:
        target = counts.most_common(1)[0][0]
        for model_id in ("qwen3-4b-local", "qwen3-8b-local", "qwen3-0.6b-probe"):
            if answers[model_id] == target:
                return model_id, 1.0
    return "gemini-3.5-flash", 1.0


def probe_agrees_with_local_else_gemini(by_model: pd.DataFrame) -> tuple[str, float]:
    probe_answer = normalize_answer(by_model.loc["qwen3-0.6b-probe", "parsed_answer"])
    for model_id in ("qwen3-4b-local", "qwen3-8b-local"):
        if probe_answer and normalize_answer(by_model.loc[model_id, "parsed_answer"]) == probe_answer:
            return model_id, 1.0
    return "gemini-3.5-flash", 1.0


def evaluate_policy(
    outputs: pd.DataFrame,
    *,
    split: str,
    method: str,
    selector: Callable[[pd.DataFrame], tuple[str, float]],
) -> dict[str, object]:
    quality: list[float] = []
    utility: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    frontier: list[bool] = []
    probe_rates: list[float] = []
    oracle_quality: list[float] = []
    oracle_utility: list[float] = []
    choices: list[str] = []
    for _, by_model in iter_queries(outputs, split):
        selected_model, probe_rate = selector(by_model)
        quality.append(float(by_model.loc[selected_model, "quality_score"]))
        utility.append(float(by_model.loc[selected_model, "utility"]))
        costs.append(float(by_model.loc[selected_model, "cost_total_usd"]))
        latencies.append(float(by_model.loc[selected_model, "latency_s"]))
        frontier.append(selected_model in FRONTIER_MODELS)
        probe_rates.append(float(probe_rate))
        choices.append(selected_model)
        candidate_models = list(FRONTIER_MODELS) + list(LOCAL_MODELS)
        oracle_quality.append(float(by_model.loc[candidate_models, "quality_score"].max()))
        oracle_utility.append(float(by_model.loc[candidate_models, "utility"].max()))
    if not quality:
        raise ValueError(f"No complete rows for split={split}")
    mean_quality = float(np.mean(quality))
    mean_utility = float(np.mean(utility))
    oracle_mean_quality = float(np.mean(oracle_quality))
    oracle_mean_utility = float(np.mean(oracle_utility))
    choice_counts = collections.Counter(choices)
    return {
        "method": method,
        "split": split,
        "n_queries": len(quality),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "oracle_mean_quality": oracle_mean_quality,
        "oracle_mean_utility": oracle_mean_utility,
        "quality_gap_to_oracle": oracle_mean_quality - mean_quality,
        "utility_gap_to_oracle": oracle_mean_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / oracle_mean_utility if abs(oracle_mean_utility) > 1e-12 else np.nan,
        "remote_cost_total_usd": float(np.sum(costs)),
        "frontier_call_rate": float(np.mean(frontier)),
        "probe_call_rate": float(np.mean(probe_rates)),
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
        "choice_counts": dict(choice_counts),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    outputs, _ = load_outputs(run_dir, Path(args.manifest), args.dataset, args.lambda_cost)
    policies: dict[str, Callable[[pd.DataFrame], tuple[str, float]]] = {
        "all_gemini-3.5-flash": all_model("gemini-3.5-flash"),
        "all_gpt-5.5": all_model("gpt-5.5"),
        "all_qwen3-4b-local": all_model("qwen3-4b-local"),
        "all_qwen3-8b-local": all_model("qwen3-8b-local"),
        "local_4b_8b_agreement_else_gemini": local_4b_8b_agreement_else_gemini,
        "local_majority_else_gemini": local_majority_else_gemini,
        "probe_agrees_with_local_else_gemini": probe_agrees_with_local_else_gemini,
    }
    rows = []
    for split in ("train", "val", "test"):
        for method, selector in policies.items():
            rows.append(evaluate_policy(outputs, split=split, method=method, selector=selector))
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    table_path = run_dir / "table_math_local_consensus_gate.csv"
    table.to_csv(table_path, index=False)

    test_rows = table[table["split"].eq("test")].copy()
    deployable = test_rows[test_rows["method"].str.contains("agreement|majority|probe_agrees", regex=True)]
    best_deployable = deployable.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]
    memo_path = run_dir / "MATH500_QWEN8_LOCAL_CONSENSUS_MEMO.md"
    memo = [
        "# MATH500 Qwen3-8B Local Consensus Memo",
        "",
        f"Run directory: `{run_dir}`.",
        f"Manifest: `{args.manifest}`.",
        "",
        "This analysis uses only cached live outputs. GPT-5.5 and Gemini rows are reused from cache; no new frontier calls are made.",
        "",
        "The consensus policies are deployable in the sense that they use query text and model outputs, not gold answers, to choose the routed model. They still require local generation before routing to the frontier.",
        "",
        "## Held-Out Test Results",
        "",
        markdown_table(
            test_rows[
                [
                    "method",
                    "n_queries",
                    "mean_quality",
                    "mean_utility",
                    "quality_gap_to_oracle",
                    "oracle_utility_ratio",
                    "frontier_call_rate",
                    "probe_call_rate",
                    "remote_cost_total_usd",
                    "p95_latency_s",
                ]
            ]
        ),
        "",
        "## Best Deployable Consensus Policy",
        "",
        (
            f"`{best_deployable.method}` reaches quality `{best_deployable.mean_quality:.4f}` "
            f"versus oracle quality `{best_deployable.oracle_mean_quality:.4f}`, leaving quality gap "
            f"`{best_deployable.quality_gap_to_oracle:.4f}`. Utility ratio is "
            f"`{best_deployable.oracle_utility_ratio:.4f}`, frontier-call rate is "
            f"`{best_deployable.frontier_call_rate:.4f}`, and remote cost is "
            f"`${best_deployable.remote_cost_total_usd:.4f}`."
        ),
        "",
        "Interpretation: Qwen3-8B adds complementary local wins but does not make live MATH500 meet the <=3 quality-point target with these deployable gates.",
        "",
        "## Files",
        "",
        f"- `{table_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
