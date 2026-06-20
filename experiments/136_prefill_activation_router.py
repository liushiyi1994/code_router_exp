from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DEFAULT_MODEL_PATH = (
    "/home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen prefill activation probes for broad100 routing.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_qwen4_prefill_activation_router"),
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--datasets", default="gpqa,mmlupro,math500")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per benchmark/split group.")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    outputs = package.load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    query_info = filtered_query_info(outputs, args)
    outputs = outputs[outputs["query_id"].astype(str).isin(set(query_info.index.astype(str)))].copy()
    activation_cache = args.output_dir / "qwen3_4b_prefill_activations.parquet"
    activations = load_or_collect_activations(
        query_info,
        model_path=str(args.model_path),
        cache_path=activation_cache,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        force=bool(args.force_recompute),
    )

    reference = run_reference_policies(package, outputs, lambda_cost=float(args.lambda_cost))
    utility = run_activation_utility_regression(package, outputs, activations, lambda_cost=float(args.lambda_cost))
    combined = pd.concat([reference, utility], ignore_index=True)
    selected = validation_selected_rows(combined)

    reference.to_csv(args.output_dir / "table_prefill_activation_reference_policies.csv", index=False)
    utility.to_csv(args.output_dir / "table_prefill_activation_utility_regression.csv", index=False)
    combined.to_csv(args.output_dir / "table_prefill_activation_router_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_prefill_activation_router_selected.csv", index=False)
    write_figure(args.output_dir, combined)
    write_memo(args.output_dir / "PREFILL_ACTIVATION_ROUTER_MEMO.md", args, query_info, activations, combined, selected)
    print(f"Wrote prefill activation router experiment to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def filtered_query_info(outputs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    datasets = {item.strip() for item in str(args.datasets).split(",") if item.strip()}
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    query_info = outputs.drop_duplicates("query_id").set_index("query_id").copy()
    if datasets:
        query_info = query_info[query_info["benchmark"].astype(str).isin(datasets)]
    if splits:
        query_info = query_info[query_info["split"].astype(str).isin(splits)]
    split_order = {"train": 0, "val": 1, "test": 2}
    query_info["_split_order"] = query_info["split"].map(split_order).fillna(99)
    query_info = query_info.sort_values(["benchmark", "_split_order", "query_id"]).drop(columns=["_split_order"])
    if args.limit is not None:
        query_info = query_info.groupby(["benchmark", "split"], group_keys=False).head(int(args.limit))
    return query_info


def load_or_collect_activations(
    query_info: pd.DataFrame,
    *,
    model_path: str,
    cache_path: Path,
    max_length: int,
    batch_size: int,
    force: bool,
) -> pd.DataFrame:
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        cached["query_id"] = cached["query_id"].astype(str)
        cached = cached.set_index("query_id")
        missing = set(query_info.index.astype(str)) - set(cached.index.astype(str))
        if not missing:
            return cached.loc[query_info.index.astype(str)]

    activations = collect_qwen_activations(
        query_info,
        model_path=model_path,
        max_length=max_length,
        batch_size=batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    activations.reset_index(names="query_id").to_parquet(cache_path, index=False)
    return activations


def collect_qwen_activations(
    query_info: pd.DataFrame,
    *,
    model_path: str,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    texts = query_info["query_text"].fillna("").astype(str).tolist()
    query_ids = query_info.index.astype(str).tolist()
    last_chunks: list[np.ndarray] = []
    mean_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = texts[start : start + int(batch_size)]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[-1].detach()
            mask = encoded["attention_mask"].to(hidden.device)
            lengths = mask.sum(dim=1).clamp(min=1) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            last = hidden[rows, lengths]
            mean = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            last_chunks.append(last.float().cpu().numpy())
            mean_chunks.append(mean.float().cpu().numpy())
    last_values = np.vstack(last_chunks) if last_chunks else np.empty((0, 0), dtype=np.float32)
    mean_values = np.vstack(mean_chunks) if mean_chunks else np.empty((0, 0), dtype=np.float32)
    columns = [f"last_{idx}" for idx in range(last_values.shape[1])] + [
        f"mean_{idx}" for idx in range(mean_values.shape[1])
    ]
    values = np.hstack([last_values, mean_values]).astype(np.float32)
    frame = pd.DataFrame(values, index=query_ids, columns=columns)
    frame.attrs["collection_seconds"] = time.time() - started
    return frame


def run_reference_policies(package, outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        target_ids = split_query_ids(outputs, split)
        for method, selected in [
            ("tool_probe_profile_v4_activation_subset", package.profile_v4_selection_for_split(outputs, split=split).loc[target_ids]),
            (
                "observable_local_state_v5_activation_subset",
                package.observable_local_state_selection(outputs, split=split).loc[target_ids],
            ),
        ]:
            row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
            row.update(
                {
                    "method": method,
                    "family": "reference_policy",
                    "activation_view": "cached_policy",
                    "alpha": np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_activation_utility_regression(
    package,
    outputs: pd.DataFrame,
    activations: pd.DataFrame,
    *,
    lambda_cost: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for activation_view in ["last", "mean", "last_plus_mean"]:
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            bundle = fit_activation_regressor(package, outputs, activations, activation_view=activation_view, alpha=alpha)
            for split in ["val", "test"]:
                selected = predict_activation_regression(package, outputs, activations, bundle=bundle, split=split)
                row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
                row.update(
                    {
                        "method": f"ridge_activation_{activation_view}_alpha{alpha:g}",
                        "family": "prefill_activation_utility_regression",
                        "activation_view": activation_view,
                        "alpha": float(alpha),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def fit_activation_regressor(
    package,
    outputs: pd.DataFrame,
    activations: pd.DataFrame,
    *,
    activation_view: str,
    alpha: float,
) -> dict[str, Any]:
    train_ids = split_query_ids(outputs, "train")
    train_x, mean, scale = activation_matrix(activations, train_ids, activation_view=activation_view)
    candidate_models = candidate_model_ids(package, outputs)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
    train_y = utility.loc[train_ids, candidate_models].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha)).fit(train_x, train_y)
    return {
        "model": model,
        "candidate_models": candidate_models,
        "activation_view": activation_view,
        "mean": mean,
        "scale": scale,
    }


def predict_activation_regression(
    package,
    outputs: pd.DataFrame,
    activations: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    split: str,
) -> pd.Series:
    target_ids = split_query_ids(outputs, split)
    target_x, _, _ = activation_matrix(
        activations,
        target_ids,
        activation_view=str(bundle["activation_view"]),
        mean=bundle["mean"],
        scale=bundle["scale"],
    )
    pred = np.asarray(bundle["model"].predict(target_x), dtype=float)
    candidate_models = list(bundle["candidate_models"])
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for row_index, query_id in enumerate(target_ids):
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        selected[query_id] = candidate_models[int(np.argmax(pred[row_index]))]
    return pd.Series(selected)


def activation_matrix(
    activations: pd.DataFrame,
    query_ids: list[str],
    *,
    activation_view: str,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if activation_view == "last":
        columns = [column for column in activations.columns if column.startswith("last_")]
    elif activation_view == "mean":
        columns = [column for column in activations.columns if column.startswith("mean_")]
    elif activation_view == "last_plus_mean":
        columns = list(activations.columns)
    else:
        raise ValueError(f"Unknown activation view: {activation_view}")
    values = activations.loc[query_ids, columns].to_numpy(dtype=np.float32)
    if mean is None:
        mean = values.mean(axis=0)
    if scale is None:
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
    values = (values - mean) / scale
    return values, mean, scale


def split_query_ids(outputs: pd.DataFrame, split: str) -> list[str]:
    return (
        outputs[outputs["split"].eq(split)]
        .drop_duplicates("query_id")
        .sort_values(["benchmark", "query_id"])["query_id"]
        .astype(str)
        .tolist()
    )


def candidate_model_ids(package, outputs: pd.DataFrame) -> list[str]:
    return [
        model_id
        for model_id in sorted(outputs["model_id"].astype(str).unique())
        if model_id != package.TOOL_MODEL
    ]


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    lambda_cost: float,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row("candidate", selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["benchmarks_json"] = selected_rows["benchmark"].value_counts().sort_index().to_json()
    return row


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[group["split"].eq("test") & group["method"].eq(best["method"])]
        if not test.empty:
            rows.append(test.iloc[0])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["selection_rule"] = "validation_best_mean_utility_by_family"
    return out


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["activation_view"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], plot["mean_utility"].iloc[::-1], color="#4c78a8")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Qwen3-4B Prefill Activation Routing")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_prefill_activation_router_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    query_info: pd.DataFrame,
    activations: pd.DataFrame,
    combined: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    split_counts = query_info.groupby(["split", "benchmark"]).size().rename("n_queries").reset_index()
    best_test = combined[combined["split"].eq("test")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    ).head(8)
    lines = [
        "# Qwen3-4B Prefill Activation Router",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Model path: `{args.model_path}`.",
        f"Activation shape: `{activations.shape}`.",
        "",
        "This run makes no external model or provider API calls. It uses local Transformers hidden states because the vLLM OpenAI API does not expose activations.",
        "",
        "## Scope",
        "",
        markdown_table(split_counts),
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Interpretation",
        "",
        "- This is the first prefill/activation probe in the Phase 3 queue.",
        "- It should be treated as an observability experiment, not a final system claim.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
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
