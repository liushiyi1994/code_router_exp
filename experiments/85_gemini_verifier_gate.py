from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from routecode.controlled.live_stage0 import extract_gemini_text, load_env_values, resolve_key


GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
LOCAL_MODEL = "qwen3-8b-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect/evaluate Gemini verifier gates for mixed exact math.")
    parser.add_argument("--query-table", default="results/controlled/mixed_exact_math_gate/query_table.csv")
    parser.add_argument("--output-dir", default="results/controlled/gemini_verifier_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def verifier_prompt(row: pd.Series) -> str:
    return (
        "Return exactly YES or NO. Do not think step by step.\n\n"
        f"Problem:\n{row['query_text']}\n\n"
        f"Candidate final answer:\n{row[f'{GEMINI_MODEL}_answer']}\n\n"
        "Is the candidate correct?"
    )


def call_gemini_verifier(prompt: str, api_key: str, timeout_s: float = 60.0) -> dict:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 16,
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def verdict_from_text(text: object) -> str:
    cleaned = str(text or "").strip().upper()
    if cleaned.startswith("YES"):
        return "YES"
    if cleaned.startswith("NO"):
        return "NO"
    return "OTHER"


def collect_verifier_rows(query_table: pd.DataFrame, output_dir: Path, api_key: str, concurrency: int) -> pd.DataFrame:
    cache_dir = output_dir / "raw_verifier" / GEMINI_MODEL
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(row: pd.Series) -> dict:
        query_id = str(row["query_id"])
        cache_path = cache_dir / f"{query_id.replace(':', '_')}.json"
        prompt = verifier_prompt(row)
        cache_hit = cache_path.exists()
        start = time.time()
        status = "success"
        error_type = ""
        payload: dict = {}
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_gemini_verifier(prompt, api_key)
            except Exception as exc:  # Cache failures as rows; rerun overwrites only if file is removed.
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        text = extract_gemini_text(payload) if status == "success" else ""
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        return {
            "query_id": query_id,
            "verifier_status": str(payload.get("_status", status)),
            "verifier_error_type": str(payload.get("_error_type", error_type)),
            "verifier_text": text,
            "verifier_verdict": verdict_from_text(text),
            "verifier_input_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "verifier_output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "verifier_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "verifier_cache_hit": cache_hit,
            "verifier_raw_path": str(cache_path),
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in query_table.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def add_verifier_cost(table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    table = table.copy()
    input_price = 1.50 / 1_000_000
    output_price = 9.00 / 1_000_000
    table["verifier_cost"] = table["verifier_input_tokens"].astype(float) * input_price + table[
        "verifier_output_tokens"
    ].astype(float) * output_price
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    return table, cost_norm


def policy_actions(table: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "verifier_yes_else_gpt":
        actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
        actions.loc[table["verifier_verdict"].eq("YES")] = "gemini"
        return actions
    if policy == "qwen_agree_or_verifier_yes_else_gpt":
        actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
        accept = table["qwen8_gemini_agree"].astype(bool) | table["verifier_verdict"].eq("YES")
        actions.loc[accept] = "gemini"
        return actions
    raise ValueError(policy)


def evaluate_policy(table: pd.DataFrame, actions: pd.Series, method: str, split: str, lambda_cost: float, cost_norm: float) -> dict:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        verifier_cost = float(row["verifier_cost"])
        if action == "local":
            quality = float(row[f"{LOCAL_MODEL}_quality"])
            cost = verifier_cost
            gemini = False
            gpt = False
        elif action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"]) + verifier_cost
            gemini = True
            gpt = False
        elif action == "gemini_then_gpt_guarded":
            gpt_available = str(row.get("gpt_answer_available", "True")).lower() != "false"
            quality = float(row[f"{GPT_MODEL}_quality"]) if gpt_available else float(row[f"{GEMINI_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"]) + float(row[f"{GPT_MODEL}_cost"]) + verifier_cost
            gemini = True
            gpt = True
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gemini_calls.append(gemini)
        gpt_calls.append(gpt)
    oracle_quality = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    oracle_utility = table[[f"{LOCAL_MODEL}_utility_selected_cost", f"{GEMINI_MODEL}_utility_selected_cost", "gemini_then_gpt_guarded_utility"]].max(axis=1)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(np.mean(qualities) - lambda_cost * (np.mean(costs) / cost_norm))
    oracle_mean_utility = float(oracle_utility.mean())
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_oracle": float(oracle_quality.mean() - mean_quality),
        "utility_ratio_to_cost_oracle": float(mean_utility / oracle_mean_utility) if abs(oracle_mean_utility) > 1e-12 else np.nan,
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum()),
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gemini_calls, gpt_calls)])),
        "gemini_call_rate": float(np.mean(gemini_calls)),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "remote_cost_total_usd": float(np.sum(costs)),
        "action_counts": json.dumps(actions.value_counts().to_dict(), sort_keys=True),
    }


def train_classifier_policy(train: pd.DataFrame, predict_frame: pd.DataFrame, model_name: str) -> pd.Series:
    frame = train.copy()
    target = frame["gemini_then_gpt_guarded_utility"].astype(float) > frame[f"{GEMINI_MODEL}_utility_selected_cost"].astype(float)
    target = target.astype(int)
    features = [
        "dataset",
        "verifier_verdict",
        "qwen8_gemini_agree",
        "query_len",
        "number_count",
        "latex_count",
        f"{LOCAL_MODEL}_answer_len",
        f"{GEMINI_MODEL}_answer_len",
    ]
    if target.nunique() < 2:
        actions = pd.Series("gemini", index=predict_frame.index)
        if int(target.iloc[0]) == 1:
            actions.loc[:] = "gemini_then_gpt_guarded"
        return actions
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["dataset", "verifier_verdict", "qwen8_gemini_agree"]),
            ("num", StandardScaler(), ["query_len", "number_count", "latex_count", f"{LOCAL_MODEL}_answer_len", f"{GEMINI_MODEL}_answer_len"]),
        ]
    )
    if model_name == "logistic":
        clf = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))
    elif model_name == "rf":
        clf = make_pipeline(pre, RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=42))
    elif model_name == "extra_trees":
        clf = make_pipeline(pre, ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=42))
    else:
        raise ValueError(model_name)
    clf.fit(frame[features], target)
    gpt = clf.predict(predict_frame[features]).astype(bool)
    actions = pd.Series("gemini", index=predict_frame.index)
    actions.loc[gpt] = "gemini_then_gpt_guarded"
    return actions


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_table = pd.read_csv(args.query_table)
    if args.max_rows:
        query_table = query_table.head(int(args.max_rows)).copy()
    api_key = resolve_key(load_env_values(args.env_file), ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise ValueError("Gemini API key not found.")
    verifier = collect_verifier_rows(query_table, output_dir, api_key, args.concurrency)
    verifier.to_csv(output_dir / "table_gemini_verifier_outputs.csv", index=False)
    table = query_table.merge(verifier, on="query_id", how="left")
    table, cost_norm = add_verifier_cost(table)
    table.to_csv(output_dir / "query_table_with_verifier.csv", index=False)

    rows: list[dict] = []
    train = table[table["split"].eq("train")].copy()
    for split, frame in table.groupby("split", sort=False):
        for policy in ["verifier_yes_else_gpt", "qwen_agree_or_verifier_yes_else_gpt"]:
            rows.append(evaluate_policy(frame, policy_actions(frame, policy), policy, str(split), args.lambda_cost, cost_norm))
        for model_name in ["logistic", "rf", "extra_trees"]:
            actions = train_classifier_policy(train, frame, model_name)
            rows.append(
                evaluate_policy(
                    frame,
                    actions,
                    f"verifier_{model_name}_gpt_rescue",
                    str(split),
                    args.lambda_cost,
                    cost_norm,
                )
            )
    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    table_path = output_dir / "table_gemini_verifier_gate.csv"
    results.to_csv(table_path, index=False)

    test_rows = results[results["split"].eq("test")].copy()
    memo = [
        "# Gemini Verifier Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Rows with verifier outputs: `{len(table)}`.",
        f"Verifier cost total: `${table['verifier_cost'].sum():.4f}`.",
        "",
        "The verifier is Gemini 3.5 Flash with thinking budget disabled. Policy utility includes the original Gemini answer cost plus verifier cost, and GPT cost when a rescue is selected.",
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
                    "utility_ratio_to_cost_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "frontier_call_rate",
                    "gpt_call_rate",
                    "remote_cost_total_usd",
                    "action_counts",
                ]
            ]
        ),
        "",
        "## Verifier Signal",
        "",
        markdown_table(pd.crosstab(table["verifier_verdict"], table[f"{GEMINI_MODEL}_quality"]).reset_index()),
        "",
        "## Files",
        "",
        f"- `{table_path}`",
        f"- `{output_dir / 'table_gemini_verifier_outputs.csv'}`",
        f"- `{output_dir / 'query_table_with_verifier.csv'}`",
    ]
    memo_path = output_dir / "GEMINI_VERIFIER_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
