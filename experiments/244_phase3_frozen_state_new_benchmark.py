from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


PREDICTED_STATE_HELPERS = Path("experiments/240_phase3_predicted_utility_state_calibration.py")
ID_COLS = {"query_id", "query_text", "split", "benchmark", "domain", "metric"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen Broad100-trained predicted RouteCode state policy to new benchmark rows."
    )
    parser.add_argument(
        "--broad-outputs",
        type=Path,
        default=Path("results/phase3_final/live_predicted_utility_states/live_outputs_with_splits_and_utility.parquet"),
    )
    parser.add_argument(
        "--broad-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--new-outputs",
        type=Path,
        default=Path("results/phase3_new_benchmark_live/live_smoke_qwen4_gpt_15/model_outputs.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3_new_benchmark_live/frozen_state_prediction"))
    parser.add_argument("--action-models", nargs="*", default=["qwen3-4b-local", "gpt-5.5"])
    parser.add_argument("--k-values", type=int, nargs="*", default=[16, 24])
    parser.add_argument("--feature-views", nargs="*", default=["probe_only", "probe_plus_benchmark"])
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    helper = load_module("routecode_predicted_state_helpers_frozen", PREDICTED_STATE_HELPERS)

    broad_outputs = pd.read_parquet(args.broad_outputs).copy()
    broad_outputs = broad_outputs[broad_outputs["status"].astype(str).eq("success")].copy()
    broad_outputs["query_id"] = broad_outputs["query_id"].astype(str)
    broad_outputs["model_id"] = broad_outputs["model_id"].astype(str)
    broad_query_table = query_metadata(broad_outputs)
    broad_features = helper.load_feature_table(args.broad_features, broad_query_table)

    new_outputs = prepare_new_outputs(args.new_outputs, lambda_cost=float(args.lambda_cost))
    new_features = build_new_feature_table(new_outputs, broad_features)

    model_pool = sorted(broad_outputs["model_id"].astype(str).unique())
    action_models = [model for model in args.action_models if model in set(model_pool)]
    new_action_models = [model for model in action_models if model in set(new_outputs["model_id"].astype(str))]
    if len(new_action_models) < 2:
        raise ValueError(f"Need at least two common action models; got {new_action_models}")
    action_models = new_action_models

    policy_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for k in args.k_values:
        for feature_view in args.feature_views:
            result = frozen_state_policy(
                broad_outputs,
                broad_features,
                broad_query_table,
                new_outputs,
                new_features,
                model_pool=model_pool,
                action_models=action_models,
                k=int(k),
                feature_view=str(feature_view),
                seed=int(args.seed),
            )
            policy_rows.append(result["policy_row"])
            assignment_frames.append(result["assignments"])
            action_frames.append(result["action_table"])
            diagnostics.append(result["diagnostics"])

    baseline = baseline_rows(new_outputs, action_models=action_models, seed=int(args.seed))
    policy_table = pd.DataFrame([*baseline, *policy_rows]).sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    )
    assignments = pd.concat(assignment_frames, ignore_index=True)
    action_table = pd.concat(action_frames, ignore_index=True)
    diagnostics_table = pd.DataFrame(diagnostics)
    model_summary = model_summary_table(new_outputs, action_models)

    policy_table.to_csv(args.output_dir / "table_frozen_state_policy.csv", index=False)
    assignments.to_csv(args.output_dir / "table_frozen_state_assignments.csv", index=False)
    action_table.to_csv(args.output_dir / "table_frozen_state_action_table.csv", index=False)
    diagnostics_table.to_csv(args.output_dir / "table_frozen_state_diagnostics.csv", index=False)
    model_summary.to_csv(args.output_dir / "table_frozen_state_model_summary.csv", index=False)
    write_readme(args.output_dir, args, policy_table, model_summary, diagnostics_table)

    print(f"Wrote frozen-state new-benchmark prediction to {args.output_dir}")
    print(policy_table[["method", "mean_quality", "mean_utility", "frontier_call_rate", "remote_cost_total_usd"]].to_string(index=False))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    cols = ["query_id", "query_text", "split", "benchmark", "domain", "metric"]
    present = [col for col in cols if col in outputs.columns]
    return outputs[present].drop_duplicates("query_id").copy()


def prepare_new_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    out = pd.read_parquet(path).copy()
    out = out[out["status"].astype(str).eq("success")].copy()
    out["query_id"] = out["query_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["quality_score"] = pd.to_numeric(out["quality_score"], errors="coerce")
    out["cost_total_usd"] = pd.to_numeric(out["cost_total_usd"], errors="coerce").fillna(0.0)
    out["latency_s"] = pd.to_numeric(out["latency_s"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["quality_score"])
    gpt_cost = out[out["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()) if not gpt_cost.empty else float(out["cost_total_usd"].max()), 1e-12)
    out["normalized_remote_cost"] = out["cost_total_usd"] / cost_norm
    out["utility"] = out["quality_score"] - float(lambda_cost) * out["normalized_remote_cost"]
    out["split"] = "new_benchmark"
    return out


def build_new_feature_table(new_outputs: pd.DataFrame, broad_features: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in broad_features.columns if col not in ID_COLS and pd.api.types.is_numeric_dtype(broad_features[col])]
    rows: list[dict[str, Any]] = []
    qwen = new_outputs[new_outputs["model_id"].eq("qwen3-4b-local")].set_index("query_id")
    for query_id, frame in new_outputs.groupby("query_id", sort=False):
        first = frame.iloc[0]
        query_text = str(first.get("query_text", ""))
        metric = str(first.get("metric", ""))
        row: dict[str, Any] = {
            "query_id": str(query_id),
            "query_text": query_text,
            "split": "new_benchmark",
            "benchmark": str(first.get("benchmark", "")),
            "domain": str(first.get("domain", "")),
            "metric": metric,
        }
        for col in numeric_cols:
            row[col] = 0.0
        row["query_chars"] = float(len(query_text))
        row["query_words"] = float(len(re.findall(r"\w+", query_text)))
        row["is_multiple_choice_prompt"] = float(metric == "multiple_choice")
        row["is_exact_answer_prompt"] = float(metric in {"exact_final_answer", "exact_ordered", "short_answer"})

        if str(query_id) in qwen.index:
            qrow = qwen.loc[str(query_id)]
            if isinstance(qrow, pd.DataFrame):
                qrow = qrow.iloc[0]
            parsed = str(qrow.get("parsed_answer", ""))
            valid = float(str(qrow.get("status", "success")) == "success" and parsed != "")
            answer_chars = float(len(parsed))
            output_tokens = float(qrow.get("output_tokens", 0.0) or 0.0)
            latency = float(qrow.get("latency_s", 0.0) or 0.0)
            updates = {
                "local_valid_count": valid,
                "local_missing_count": 1.0 - valid,
                "local_unique_answer_count": valid,
                "local_top_vote_count": valid,
                "local_vote_frac": valid,
                "local_vote_margin": valid,
                "local_vote_entropy": 0.0,
                "local_all_agree": valid,
                "small_valid_count": valid,
                "small_unique_answer_count": valid,
                "answer_chars_mean": answer_chars,
                "answer_chars_std": 0.0,
                "output_tokens_mean": output_tokens,
                "output_tokens_std": 0.0,
                "qwen4b_valid": valid,
                "qwen4b_answer_chars": answer_chars,
                "qwen4b_status_success": valid,
                "qwen4b_output_tokens": output_tokens,
                "qwen4b_latency_s": latency,
            }
            for key, value in updates.items():
                if key in row:
                    row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def frozen_state_policy(
    broad_outputs: pd.DataFrame,
    broad_features: pd.DataFrame,
    broad_query_table: pd.DataFrame,
    new_outputs: pd.DataFrame,
    new_features: pd.DataFrame,
    *,
    model_pool: list[str],
    action_models: list[str],
    k: int,
    feature_view: str,
    seed: int,
) -> dict[str, Any]:
    matrix = (
        broad_outputs[broad_outputs["model_id"].isin(model_pool)]
        .pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="mean")
        .dropna(axis=0)
    )
    train_ids = broad_query_table[broad_query_table["split"].astype(str).eq("train")]["query_id"].astype(str)
    train_matrix = matrix.reindex(train_ids).dropna(axis=0)
    if len(train_matrix) < k:
        raise ValueError(f"Not enough train rows for k={k}")

    scaler = StandardScaler()
    train_utility = scaler.fit_transform(train_matrix.to_numpy(dtype=float))
    clusterer = KMeans(n_clusters=int(k), random_state=int(seed), n_init=30)
    oracle_labels = clusterer.fit_predict(train_utility)

    x_train_all = design_matrix(broad_features, feature_view)
    x_new = design_matrix(new_features, feature_view, train_columns=list(x_train_all.columns))
    x_train = x_train_all.reindex(train_matrix.index.astype(str)).fillna(0.0)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
    )
    clf.fit(x_train.to_numpy(dtype=float), oracle_labels.astype(int))
    train_pred = pd.Series(clf.predict(x_train.to_numpy(dtype=float)).astype(int), index=x_train.index.astype(str))
    new_pred = pd.Series(clf.predict(x_new.to_numpy(dtype=float)).astype(int), index=x_new.index.astype(str))
    method = f"frozen_state_rf_{feature_view}_k{k}"

    train_groups = pd.DataFrame(
        {
            "query_id": train_pred.index.astype(str).to_numpy(),
            "group_id": ("p" + train_pred.astype(str).str.zfill(2)).to_numpy(),
        }
    ).reset_index(drop=True)
    new_assignments = new_features[["query_id", "benchmark", "domain"]].copy()
    new_assignments["method"] = method
    new_assignments["group_id"] = "p" + new_assignments["query_id"].map(new_pred).astype(int).astype(str).str.zfill(2)

    action_table, fallback_model = train_action_table(
        broad_outputs,
        train_groups,
        action_models=action_models,
        method=method,
    )
    selected = new_assignments[["query_id", "group_id"]].merge(
        action_table[["group_id", "selected_model_id"]], on="group_id", how="left"
    )
    selected["selected_model_id"] = selected["selected_model_id"].fillna(fallback_model)
    selected_rows = select_rows(new_outputs, selected.rename(columns={"selected_model_id": "model_id"}))
    policy_row = routing_row(method, selected_rows, new_outputs, action_models)
    policy_row["k"] = int(k)
    policy_row["feature_view"] = feature_view
    policy_row["fallback_model_id"] = fallback_model

    diagnostics = {
        "method": method,
        "k": int(k),
        "feature_view": feature_view,
        "train_queries": int(len(train_matrix)),
        "feature_count": int(len(x_train_all.columns)),
        "new_queries": int(new_features["query_id"].nunique()),
        "new_group_count": int(new_assignments["group_id"].nunique()),
        "train_group_count": int(train_groups["group_id"].nunique()),
        "fallback_model_id": fallback_model,
    }
    return {
        "policy_row": policy_row,
        "assignments": new_assignments,
        "action_table": action_table,
        "diagnostics": diagnostics,
    }


def design_matrix(feature_table: pd.DataFrame, feature_view: str, train_columns: list[str] | None = None) -> pd.DataFrame:
    work = feature_table.copy()
    numeric_cols = [col for col in work.columns if col not in ID_COLS and pd.api.types.is_numeric_dtype(work[col])]
    x = work[["query_id", *numeric_cols]].copy()
    if feature_view in {"probe_plus_benchmark", "probe_plus_metadata"}:
        one_hot = pd.get_dummies(work[["benchmark", "domain"]].fillna("unknown").astype(str), prefix=["bench", "domain"])
        x = pd.concat([x, one_hot], axis=1)
    elif feature_view != "probe_only":
        raise ValueError(f"Unknown feature view: {feature_view}")
    x = x.set_index("query_id").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if train_columns is not None:
        x = x.reindex(columns=train_columns, fill_value=0.0)
    return x


def train_action_table(
    broad_outputs: pd.DataFrame,
    train_groups: pd.DataFrame,
    *,
    action_models: list[str],
    method: str,
) -> tuple[pd.DataFrame, str]:
    train = broad_outputs[
        broad_outputs["split"].astype(str).eq("train") & broad_outputs["model_id"].astype(str).isin(action_models)
    ].copy()
    train = train.merge(train_groups, on="query_id", how="inner")
    fallback_model = (
        train.groupby("model_id")["utility"].mean().sort_values(ascending=False).index.astype(str).tolist()[0]
    )
    means = (
        train.groupby(["group_id", "model_id"], as_index=False)
        .agg(mean_train_utility=("utility", "mean"), mean_train_quality=("quality_score", "mean"), n_train=("query_id", "nunique"))
        .sort_values(["group_id", "mean_train_utility"], ascending=[True, False])
    )
    best = means.groupby("group_id", as_index=False).head(1).copy()
    best = best.rename(columns={"model_id": "selected_model_id", "mean_train_utility": "selected_train_utility"})
    best["method"] = method
    best["fallback_model_id"] = fallback_model
    return best, str(fallback_model)


def select_rows(outputs: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    return selected.merge(outputs, on=["query_id", "model_id"], how="left").dropna(subset=["quality_score"]).copy()


def baseline_rows(outputs: pd.DataFrame, *, action_models: list[str], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in action_models:
        selected = pd.DataFrame({"query_id": sorted(outputs["query_id"].unique()), "model_id": model})
        rows.append(routing_row(f"all_{model}", select_rows(outputs, selected), outputs, action_models))

    rng = np.random.default_rng(seed)
    query_ids = sorted(outputs["query_id"].unique())
    random_models = rng.choice(action_models, size=len(query_ids), replace=True)
    random_selected = pd.DataFrame({"query_id": query_ids, "model_id": random_models})
    rows.append(routing_row("random_common_model", select_rows(outputs, random_selected), outputs, action_models))

    quality_oracle = outputs.loc[outputs.groupby("query_id")["quality_score"].idxmax()].copy()
    cost_oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()].copy()
    rows.append(routing_row("quality_oracle_common_model", quality_oracle, outputs, action_models))
    rows.append(routing_row("cost_aware_oracle_common_model", cost_oracle, outputs, action_models))
    return rows


def routing_row(method: str, selected_rows: pd.DataFrame, all_outputs: pd.DataFrame, action_models: list[str]) -> dict[str, Any]:
    cost_oracle = all_outputs.loc[all_outputs.groupby("query_id")["utility"].idxmax()].copy()
    quality_oracle = all_outputs.loc[all_outputs.groupby("query_id")["quality_score"].idxmax()].copy()
    mean_utility = float(selected_rows["utility"].mean()) if not selected_rows.empty else float("nan")
    oracle_utility = float(cost_oracle["utility"].mean())
    model_counts = selected_rows["model_id"].astype(str).value_counts().to_dict()
    return {
        "method": method,
        "n_queries": int(selected_rows["query_id"].nunique()),
        "action_models": ",".join(action_models),
        "mean_quality": float(selected_rows["quality_score"].mean()) if not selected_rows.empty else float("nan"),
        "mean_utility": mean_utility,
        "quality_oracle_mean_quality": float(quality_oracle["quality_score"].mean()),
        "cost_oracle_mean_utility": oracle_utility,
        "quality_gap_to_oracle": float(quality_oracle["quality_score"].mean() - selected_rows["quality_score"].mean()),
        "utility_gap_to_oracle": float(oracle_utility - mean_utility),
        "oracle_utility_ratio": mean_utility / oracle_utility if abs(oracle_utility) > 1e-12 else float("nan"),
        "remote_cost_total_usd": float(selected_rows["cost_total_usd"].sum()),
        "frontier_call_rate": float(selected_rows["is_frontier"].astype(bool).mean()) if not selected_rows.empty else float("nan"),
        "mean_latency_s": float(selected_rows["latency_s"].mean()) if not selected_rows.empty else float("nan"),
        "p95_latency_s": float(selected_rows["latency_s"].quantile(0.95)) if not selected_rows.empty else float("nan"),
        "selected_model_counts_json": json.dumps(model_counts, sort_keys=True),
    }


def model_summary_table(outputs: pd.DataFrame, action_models: list[str]) -> pd.DataFrame:
    return (
        outputs[outputs["model_id"].isin(action_models)]
        .groupby(["benchmark", "model_id", "provider"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            mean_quality=("quality_score", "mean"),
            mean_utility=("utility", "mean"),
            total_cost_usd=("cost_total_usd", "sum"),
            mean_latency_s=("latency_s", "mean"),
        )
        .sort_values(["benchmark", "model_id"])
    )


def write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    policy_table: pd.DataFrame,
    model_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    top_lines = "\n".join(
        f"| {row.method} | {int(row.n_queries)} | {row.mean_quality:.4f} | {row.mean_utility:.4f} | "
        f"{row.frontier_call_rate:.4f} | {row.remote_cost_total_usd:.4f} | {row.selected_model_counts_json} |"
        for row in policy_table.itertuples(index=False)
    )
    model_lines = "\n".join(
        f"| {row.benchmark} | {row.model_id} | {int(row.n_queries)} | {row.mean_quality:.4f} | "
        f"{row.mean_utility:.4f} | {row.total_cost_usd:.4f} |"
        for row in model_summary.itertuples(index=False)
    )
    diag_lines = "\n".join(
        f"| {row.method} | {int(row.train_queries)} | {int(row.new_queries)} | {int(row.new_group_count)} | "
        f"{row.fallback_model_id} |"
        for row in diagnostics.itertuples(index=False)
    )
    readme = f"""# Frozen State Prediction On New Benchmarks

This run freezes the Broad100-trained predicted RouteCode state machinery and
applies it to the new-benchmark live smoke.

Important scope:

- state clusters are learned from Broad100 train utility vectors only;
- the state predictor is trained on Broad100 train features only;
- the state-to-action table is trained on Broad100 train rows only;
- no threshold or action rule is selected on the new benchmarks;
- the comparable action pool is restricted to `{', '.join(args.action_models)}`.

The new benchmark rows are from `simpleqa_verified`, `livebench_math`, and
`livebench_reasoning`.

## Result

| method | queries | quality | utility | frontier rate | remote cost usd | selected models |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{top_lines}

Interpretation:

- the common-model cost-aware oracle still shows a routing opportunity;
- all GPT is strong but costly;
- all Qwen3-4B is cheap but fails this small slice;
- the frozen state policy is the deployable test. If it routes mostly to local
  and underperforms, that is evidence that the current Broad100 states/action
  table do not yet transfer to these new benchmark families.

## Per-Benchmark Model Inputs

| benchmark | model | queries | quality | utility | cost usd |
| --- | --- | ---: | ---: | ---: | ---: |
{model_lines}

## Frozen State Diagnostics

| method | Broad100 train queries | new queries | new groups used | fallback |
| --- | ---: | ---: | ---: | --- |
{diag_lines}

## Commands

```bash
bash scripts/start_vllm_qwen3_4b.sh

PYTHONPATH=src python experiments/81_controlled_live_stage0.py \\
  --config configs/proberoute_controlled_broad100.yaml \\
  --output-dir results/phase3_new_benchmark_live/live_smoke_qwen4_gpt_15 \\
  --run-suffix new_benchmark_gpt512_smoke \\
  --task-manifest results/phase3_new_benchmark_live/new_benchmark_manifest.csv \\
  --frontier-model-ids gpt-5.5 \\
  --local-model-ids qwen3-4b-local \\
  --allow-frontier-calls \\
  --retry-errors \\
  --max-calls-per-frontier-model 15 \\
  --max-calls-per-local-model 15 \\
  --frontier-concurrency 1 \\
  --max-output-tokens 512 \\
  --local-max-output-tokens 128 \\
  --request-timeout-s 120

PYTHONPATH=src python experiments/244_phase3_frozen_state_new_benchmark.py
```

## Artifacts

- `table_frozen_state_policy.csv`
- `table_frozen_state_assignments.csv`
- `table_frozen_state_action_table.csv`
- `table_frozen_state_diagnostics.csv`
- `table_frozen_state_model_summary.csv`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
