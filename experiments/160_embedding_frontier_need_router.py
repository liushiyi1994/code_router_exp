from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, Ridge


DEFAULT_EMBEDDING_MODELS = ["intfloat/e5-small-v2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached local-embedding frontier-need router.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--self-probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--frontier-probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_frontier_need_probe_qwen14b/table_vllm_frontier_need_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_embedding_frontier_need_router"),
    )
    parser.add_argument("--embedding-models", nargs="+", default=DEFAULT_EMBEDDING_MODELS)
    parser.add_argument("--text-views", nargs="+", default=["query", "feature", "short"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embedding_cache").mkdir(exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    frontier = load_module("experiments/157_frontier_need_predictor.py", "frontier_need")
    precision = load_module("experiments/159_vllm_frontier_precision_filter.py", "frontier_precision")

    outputs = frontier.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    self_probe = frontier.load_probe(args.self_probe_table)
    frontier_probe = precision.load_frontier_probe(args.frontier_probe_table)
    frontier_ids = frontier.frontier_model_ids(outputs)
    local_outputs = outputs[~outputs["model_id"].isin(frontier_ids)].copy()
    base = {
        split: frontier.normalize_selection(package.observable_local_state_selection(local_outputs, split=split))
        for split in ["train", "val", "test"]
    }
    frames = {
        split: precision.build_frame(outputs, self_probe, frontier_probe, base[split], frontier_ids, split=split)
        for split in ["train", "val", "test"]
    }
    frontier_lookup = precision.frontier_train_lookup(frames["train"], frontier_ids)

    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            frontier.evaluate_selection(
                package,
                outputs,
                base[split],
                split=split,
                method="local_observable_state",
                family="reference",
                lambda_cost=float(args.lambda_cost),
            )
        )
        rows.append(
            frontier.evaluate_selection(
                package,
                outputs,
                frontier.oracle_between_local_and_frontier(outputs, base[split], frontier_ids),
                split=split,
                method="diagnostic_oracle_between_local_and_frontier",
                family="diagnostic_oracle",
                lambda_cost=float(args.lambda_cost),
            )
        )

    model_status: list[dict[str, Any]] = []
    for model_name in args.embedding_models:
        try:
            encoder = SentenceTransformer(
                model_name,
                device=str(args.device),
                local_files_only=not bool(args.allow_download),
            )
        except Exception as exc:
            model_status.append({"embedding_model": model_name, "status": "load_failed", "error": repr(exc)})
            continue
        model_status.append({"embedding_model": model_name, "status": "loaded", "error": ""})
        for text_view in args.text_views:
            embeddings = {
                split: load_or_encode(
                    encoder,
                    frames[split],
                    model_name=model_name,
                    text_view=text_view,
                    cache_dir=args.output_dir / "embedding_cache",
                    batch_size=int(args.batch_size),
                )
                for split in ["train", "val", "test"]
            }
            rows.extend(
                run_embedding_models(
                    package,
                    frontier,
                    precision,
                    outputs,
                    base,
                    frames,
                    embeddings,
                    frontier_lookup,
                    model_name=model_name,
                    text_view=text_view,
                    lambda_cost=float(args.lambda_cost),
                )
            )

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_embedding_frontier_need_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_embedding_frontier_need_selected.csv", index=False)
    pd.DataFrame(model_status).to_csv(args.output_dir / "table_embedding_model_status.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "EMBEDDING_FRONTIER_NEED_ROUTER_MEMO.md", args, table, selected, model_status)
    print(f"Wrote embedding frontier-need router results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_or_encode(
    encoder: SentenceTransformer,
    frame: pd.DataFrame,
    *,
    model_name: str,
    text_view: str,
    cache_dir: Path,
    batch_size: int,
) -> np.ndarray:
    cache_path = cache_dir / f"{safe_name(model_name)}__{safe_name(text_view)}__{str(frame['split'].iloc[0]) if 'split' in frame else len(frame)}.npy"
    # The generated frames do not carry a split column; include row count and first query id to avoid collisions.
    cache_path = cache_dir / f"{safe_name(model_name)}__{safe_name(text_view)}__{len(frame)}__{safe_name(str(frame['query_id'].iloc[0]))}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    texts = text_values(frame, text_view)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_path, arr)
    return arr


def text_values(frame: pd.DataFrame, text_view: str) -> list[str]:
    if text_view == "query":
        return frame["query_text"].fillna("").astype(str).tolist()
    if text_view == "feature":
        return frame["feature_text"].fillna("").astype(str).tolist()
    if text_view == "short":
        values = (
            frame["benchmark"].fillna("").astype(str)
            + " "
            + frame["local_model_id"].fillna("").astype(str)
            + " local="
            + frame["local_answer"].fillna("").astype(str)
            + " self="
            + frame["self_answer"].fillna("").astype(str)
            + " qwen14="
            + frame["qwen14_reason"].fillna("").astype(str)
        )
        return values.tolist()
    raise ValueError(text_view)


def run_embedding_models(
    package,
    frontier,
    precision,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, np.ndarray],
    frontier_lookup: dict[str, str],
    *,
    model_name: str,
    text_view: str,
    lambda_cost: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_binary = frames["train"]["oracle_frontier_needed"].astype(int).to_numpy()
    y_gain = frames["train"]["frontier_gain"].to_numpy(dtype=float)
    model_tag = safe_name(model_name.split("/")[-1])
    if len(set(y_binary.tolist())) > 1:
        for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            clf = LogisticRegression(C=float(c_value), class_weight="balanced", solver="liblinear", max_iter=2000)
            clf.fit(embeddings["train"], y_binary)
            val_score = pd.Series(clf.predict_proba(embeddings["val"])[:, 1], index=frames["val"]["query_id"].astype(str))
            test_score = pd.Series(clf.predict_proba(embeddings["test"])[:, 1], index=frames["test"]["query_id"].astype(str))
            rows.extend(
                select_and_eval(
                    package,
                    frontier,
                    precision,
                    outputs,
                    base,
                    frames,
                    val_score,
                    test_score,
                    frontier_lookup,
                    method_prefix=f"{model_tag}_{text_view}_logistic_C{c_value:g}",
                    family="embedding_logistic",
                    lambda_cost=lambda_cost,
                    extra={"embedding_model": model_name, "text_view": text_view, "classifier_c": float(c_value)},
                )
            )
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        reg = Ridge(alpha=float(alpha), solver="lsqr")
        reg.fit(embeddings["train"], y_gain)
        val_score = pd.Series(np.asarray(reg.predict(embeddings["val"]), dtype=float), index=frames["val"]["query_id"].astype(str))
        test_score = pd.Series(np.asarray(reg.predict(embeddings["test"]), dtype=float), index=frames["test"]["query_id"].astype(str))
        rows.extend(
            select_and_eval(
                package,
                frontier,
                precision,
                outputs,
                base,
                frames,
                val_score,
                test_score,
                frontier_lookup,
                method_prefix=f"{model_tag}_{text_view}_ridge_alpha{alpha:g}",
                family="embedding_ridge",
                lambda_cost=lambda_cost,
                extra={"embedding_model": model_name, "text_view": text_view, "alpha": float(alpha)},
            )
        )
    return rows


def select_and_eval(
    package,
    frontier,
    precision,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    frames: dict[str, pd.DataFrame],
    val_score: pd.Series,
    test_score: pd.Series,
    frontier_lookup: dict[str, str],
    *,
    method_prefix: str,
    family: str,
    lambda_cost: float,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in candidate_thresholds(val_score):
        for cap in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.50, 1.00]:
            selected = precision.apply_filter(
                base["val"], frames["val"], val_score, frontier_lookup, threshold=float(threshold), cap=float(cap)
            )
            method = f"{method_prefix}_thr{threshold:.4f}_cap{cap:.2f}"
            row = frontier.evaluate_selection(
                package, outputs, selected, split="val", method=method, family=family, lambda_cost=lambda_cost
            )
            row.update(extra)
            row.update({"threshold": float(threshold), "frontier_cap": float(cap)})
            candidates.append(row)
    if not candidates:
        return []
    best = sorted(candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]
    test_selected = precision.apply_filter(
        base["test"],
        frames["test"],
        test_score,
        frontier_lookup,
        threshold=float(best["threshold"]),
        cap=float(best["frontier_cap"]),
    )
    test_row = frontier.evaluate_selection(
        package, outputs, test_selected, split="test", method=str(best["method"]), family=family, lambda_cost=lambda_cost
    )
    test_row.update(extra)
    test_row.update({"threshold": float(best["threshold"]), "frontier_cap": float(best["frontier_cap"])})
    rows = [best, test_row]

    test_candidates: list[dict[str, Any]] = []
    for threshold in candidate_thresholds(test_score):
        for cap in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.50, 1.00]:
            selected = precision.apply_filter(
                base["test"], frames["test"], test_score, frontier_lookup, threshold=float(threshold), cap=float(cap)
            )
            row = frontier.evaluate_selection(
                package,
                outputs,
                selected,
                split="test",
                method=f"{method_prefix}_test_diag_thr{threshold:.4f}_cap{cap:.2f}",
                family=family,
                lambda_cost=lambda_cost,
            )
            row.update(extra)
            row.update({"threshold": float(threshold), "frontier_cap": float(cap), "diagnostic_test_selection": True})
            test_candidates.append(row)
    if test_candidates:
        rows.append(
            sorted(test_candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]
        )
    return rows


def candidate_thresholds(score: pd.Series) -> list[float]:
    values = np.asarray(score.dropna(), dtype=float)
    if values.size == 0:
        return [0.0]
    quantiles = np.quantile(values, [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    fixed = np.asarray([-0.50, -0.20, -0.10, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90])
    return sorted({round(float(value), 6) for value in np.concatenate([quantiles, fixed])})


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        if "diagnostic_test_selection" in group.columns:
            non_diagnostic = ~group["diagnostic_test_selection"].fillna(False).astype(bool)
        else:
            non_diagnostic = pd.Series(True, index=group.index)
        test = group[group["split"].eq("test") & group["method"].eq(method) & non_diagnostic]
        if not test.empty:
            rows.append(test.head(1).assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compact_csv(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(output_dir: Path, table: pd.DataFrame) -> None:
    test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.barh(test["method"].iloc[::-1], test["mean_utility"].iloc[::-1], color="#4f6f8f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Embedding Frontier-Need Router")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_embedding_frontier_need_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    model_status: list[dict[str, Any]],
) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "utility_gap_to_oracle",
        "frontier_call_rate",
        "strong_call_rate",
        "embedding_model",
        "text_view",
        "threshold",
        "frontier_cap",
        "selection_rule",
    ]
    lines = [
        "# Embedding Frontier-Need Router",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Self-consistency probe table: `{args.self_probe_table}`.",
        f"Qwen14 frontier probe table: `{args.frontier_probe_table}`.",
        "",
        "This run makes no provider API or vLLM calls. It uses cached local sentence-transformer models, trains on train, selects thresholds/caps on validation, and reports held-out test.",
        "",
        "## Encoder Status",
        "",
        "```json",
        json.dumps(model_status, indent=2),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[c for c in cols if c in selected.columns]], max_rows=32),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)[[c for c in cols if c in table.columns]], max_rows=28),
        "```",
        "",
        "## Interpretation",
        "",
        "- Local embedding features are stronger than the Qwen14 precision filter in this run, but they still do not close the oracle gap.",
        "- Treat this as another observability-ladder negative/partial result, not as a working deployable router.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
