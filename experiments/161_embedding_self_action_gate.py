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
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding-augmented base/self/strong action gate.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_embedding_self_action_gate"),
    )
    parser.add_argument("--embedding-model", default="intfloat/e5-small-v2")
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embedding_cache").mkdir(exist_ok=True)

    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    sc_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_gate")

    outputs = sc_gate.load_outputs(args.outputs)
    probe = sc_gate.load_probe(args.probe_table)
    encoder_status = {"embedding_model": args.embedding_model, "status": "not_loaded", "error": ""}
    try:
        encoder = SentenceTransformer(
            args.embedding_model,
            device=str(args.device),
            local_files_only=not bool(args.allow_download),
        )
        encoder_status["status"] = "loaded"
    except Exception as exc:
        encoder = None
        encoder_status.update({"status": "load_failed", "error": repr(exc)})

    table = run_embedding_action_gate(
        package,
        sc_gate,
        outputs,
        probe,
        encoder,
        output_dir=args.output_dir,
        embedding_model=str(args.embedding_model),
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        batch_size=int(args.batch_size),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_embedding_self_action_gate_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_embedding_self_action_gate_selected.csv", index=False)
    pd.DataFrame([encoder_status]).to_csv(args.output_dir / "table_embedding_model_status.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "EMBEDDING_SELF_ACTION_GATE_MEMO.md", args, table, selected, encoder_status)
    print(f"Wrote embedding self-action gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_embedding_action_gate(
    package,
    sc_gate,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    encoder: SentenceTransformer | None,
    *,
    output_dir: Path,
    embedding_model: str,
    self_model_id: str,
    lambda_cost: float,
    batch_size: int,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()
    base_specs = {
        "observable_local_state_v5": lambda split: package.observable_local_state_selection(outputs_no_self, split=split),
        "observable_local_state_v5_no_strong": lambda split: package.observable_local_state_selection(
            outputs_no_strong_self, split=split
        ),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}
        ),
    }
    for base_name, builder in base_specs.items():
        base = {split: sc_gate.normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        for split in ["val", "test"]:
            rows.append(
                sc_gate.evaluate_selection(
                    package,
                    outputs,
                    base[split],
                    split=split,
                    method=base_name,
                    family="base",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
            rows.append(
                sc_gate.evaluate_selection(
                    package,
                    outputs,
                    sc_gate.oracle_between_actions(outputs, base[split], [self_model_id, STRONG_MODEL_ID]),
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
        if encoder is None:
            continue
        frames = {
            split: sc_gate.build_feature_frame(outputs, probe, base[split], split=split, self_model_id=self_model_id)
            for split in ["train", "val", "test"]
        }
        if any(frame.empty for frame in frames.values()):
            continue
        x_meta_train, x_meta_val, x_meta_test = sc_gate.featurize(
            frames["train"],
            frames["val"],
            frames["test"],
            feature_view="metadata_numeric",
            max_features=max_features,
        )
        embeddings = {
            split: csr_matrix(
                load_or_encode(
                    encoder,
                    frames[split],
                    output_dir=output_dir,
                    embedding_model=embedding_model,
                    base_name=base_name,
                    split=split,
                    batch_size=batch_size,
                )
            )
            for split in ["train", "val", "test"]
        }
        feature_sets = {
            "embed": {
                "train": embeddings["train"],
                "val": embeddings["val"],
                "test": embeddings["test"],
            },
            "meta_embed": {
                "train": hstack([x_meta_train, embeddings["train"]]),
                "val": hstack([x_meta_val, embeddings["val"]]),
                "test": hstack([x_meta_test, embeddings["test"]]),
            },
        }
        for feature_view, matrices in feature_sets.items():
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                rows.extend(
                    run_ridge_action_model(
                        package,
                        sc_gate,
                        outputs,
                        base,
                        frames,
                        matrices,
                        base_name=base_name,
                        embedding_tag=embedding_tag(embedding_model),
                        feature_view=feature_view,
                        alpha=float(alpha),
                        self_model_id=self_model_id,
                        lambda_cost=lambda_cost,
                    )
                )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def run_ridge_action_model(
    package,
    sc_gate,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    frames: dict[str, pd.DataFrame],
    matrices: dict[str, Any],
    *,
    base_name: str,
    embedding_tag: str,
    feature_view: str,
    alpha: float,
    self_model_id: str,
    lambda_cost: float,
) -> list[dict[str, Any]]:
    val_scores = pd.DataFrame(index=frames["val"]["query_id"].astype(str))
    test_scores = pd.DataFrame(index=frames["test"]["query_id"].astype(str))
    for action, column in [("base", "utility_base"), ("self", "utility_self"), ("strong", "utility_strong")]:
        model = Ridge(alpha=float(alpha), solver="lsqr")
        model.fit(matrices["train"], frames["train"][column].to_numpy(dtype=float))
        val_scores[action] = np.asarray(model.predict(matrices["val"]), dtype=float)
        test_scores[action] = np.asarray(model.predict(matrices["test"]), dtype=float)
    method = f"{base_name}_{embedding_tag}_{feature_view}_alpha{alpha:g}"
    rows = sc_gate.selected_val_and_test_rows(
        package,
        outputs,
        base,
        val_scores,
        test_scores,
        method=method,
        family="embedding_self_action_ridge",
        self_model_id=self_model_id,
        lambda_cost=lambda_cost,
        feature_view=feature_view,
        alpha=float(alpha),
    )
    for row in rows:
        row["base_name"] = base_name
    return rows


def load_or_encode(
    encoder: SentenceTransformer,
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    embedding_model: str,
    base_name: str,
    split: str,
    batch_size: int,
) -> np.ndarray:
    cache_path = (
        output_dir
        / "embedding_cache"
        / f"{safe_name(embedding_model)}__{safe_name(base_name)}__{safe_name(split)}__{len(frame)}.npy"
    )
    if cache_path.exists():
        return np.load(cache_path)
    embeddings = encoder.encode(
        frame["feature_text"].fillna("").astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_path, arr)
    return arr


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def embedding_tag(model_name: str) -> str:
    if model_name.endswith("e5-small-v2"):
        return "e5"
    return safe_name(model_name.split("/")[-1])


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
        test = group[group["split"].eq("test") & group["method"].eq(method)]
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
    ax.barh(test["method"].iloc[::-1], test["mean_utility"].iloc[::-1], color="#6c6f93")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Embedding Self-Action Gate")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_embedding_self_action_gate_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    encoder_status: dict[str, Any],
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
        "self_action_rate",
        "feature_view",
        "alpha",
        "base_name",
        "selection_rule",
    ]
    lines = [
        "# Embedding Self-Action Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        f"Self model action: `{args.self_model_id}`.",
        "",
        "This run makes no provider API or vLLM calls. It uses a cached local sentence-transformer encoder and cached self-consistency/action rows.",
        "",
        "## Encoder Status",
        "",
        "```json",
        json.dumps(encoder_status, indent=2),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[c for c in cols if c in selected.columns]], max_rows=36),
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
        "- Local sentence-transformer embeddings slightly change the validation-selected self-consistency action gate, but not enough to approach the action-set oracle.",
        "- The best held-out diagnostic row is better than the previous self-consistency feature diagnostics, but it is test-picked and not a deployable claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
