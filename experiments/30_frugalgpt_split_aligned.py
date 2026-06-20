from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.external_baseline_assets import build_external_baseline_assets, write_external_baseline_assets
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


RUN_DIRNAME = "frugalgpt_split_aligned_metric"
FRUGAL_SOURCE = ROOT / "data/raw/external/LLMRouterBench/baselines/FrugalGPT/train_router_from_results.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    baseline_config = config.get("external_baselines", {})
    seed = int(config.get("run", {}).get("random_seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    assets = build_external_baseline_assets({"train": train, "test": test}, prepared.embeddings)
    written = write_external_baseline_assets(assets, out_dir)
    run_dir = out_dir / RUN_DIRNAME
    scorer_dir = run_dir / "scorers"
    run_dir.mkdir(parents=True, exist_ok=True)

    frugal = _load_frugalgpt_module()
    score_threshold = float(baseline_config.get("frugalgpt_score_threshold", 0.5))
    train_df = frugal.load_jsonl_split(written.frugalgpt_train_path, split="train")
    test_df = frugal.load_jsonl_split(written.frugalgpt_test_path, split="test")
    train_df["label"] = (train_df["score"] >= score_threshold).astype(int)
    test_df["label"] = (test_df["score"] >= score_threshold).astype(int)
    train_df["dataset_name"] = train_df["dataset_name"].fillna("unknown")
    test_df["dataset_name"] = test_df["dataset_name"].fillna("unknown")

    model_names = sorted(train_df["model_name"].dropna().astype(str).unique().tolist())
    probabilities = _collect_model_probabilities(
        frugal=frugal,
        train_df=train_df,
        test_df=test_df,
        model_names=model_names,
        out_dir=out_dir,
        run_scorer_dir=scorer_dir,
        baseline_config=baseline_config,
        seed=seed,
    )

    thresholds = [float(value) for value in baseline_config.get("frugalgpt_prob_thresholds", [0.5])]
    table, raw_predictions = _evaluate_thresholds(
        test_df=test_df,
        probabilities=probabilities,
        thresholds=thresholds,
        train_matrices=train,
        test_matrices=test,
        embeddings=prepared.embeddings,
        config=config,
        seed=seed,
        model_names=model_names,
        score_threshold=score_threshold,
    )
    table["split_aligned_with_routecode"] = True
    table["routecode_metric_compatible"] = True
    table["official_training_code_used"] = True
    table["official_upstream_checkpoint"] = False
    table["exact_upstream_command"] = False
    table["baseline_family"] = "frugalgpt_local_scorer_metric_adapter"
    table["implementation_note"] = (
        "Local RouteCode metric adapter using the LLMRouterBench FrugalGPT local scorer source. "
        "Selection follows the FrugalGPT report rule: cheapest predicted-positive model, otherwise highest probability with cost tie-break."
    )

    table.to_csv(out_dir / "table_frugalgpt_split_aligned.csv", index=False)
    _write_json(run_dir / "raw_predictions.json", raw_predictions)
    _write_json(run_dir / "run_config.json", _run_config_payload(baseline_config, score_threshold, scorer_dir))
    write_memo(out_dir, config_path, run_dir, table)
    append_readme(out_dir, config_path, run_dir, table)
    print(f"Wrote split-aligned FrugalGPT outputs to {run_dir}")


def _load_frugalgpt_module():
    spec = importlib.util.spec_from_file_location("routecode_external_frugalgpt_local_scorer", FRUGAL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FrugalGPT source: {FRUGAL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["routecode_external_frugalgpt_local_scorer"] = module
    spec.loader.exec_module(module)
    return module


def _collect_model_probabilities(
    *,
    frugal,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_names: list[str],
    out_dir: Path,
    run_scorer_dir: Path,
    baseline_config: dict[str, Any],
    seed: int,
) -> pd.Series:
    reuse_saved = bool(baseline_config.get("frugalgpt_reuse_saved_scorers", False))
    saved_root = Path(
        baseline_config.get("frugalgpt_saved_scorer_dir", out_dir / "frugalgpt_split_aligned/output")
    ).expanduser()
    args = _training_args(baseline_config, out_dir, run_scorer_dir, seed)
    all_probabilities = pd.Series(np.nan, index=test_df.index, dtype=float)

    for model_name in model_names:
        sub_train = train_df[train_df["model_name"].astype(str) == model_name].reset_index(drop=True)
        sub_test = test_df[test_df["model_name"].astype(str) == model_name].reset_index(drop=True)
        if sub_train.empty or sub_test.empty:
            continue
        saved_model_dir = saved_root / model_name
        if reuse_saved and (saved_model_dir / "config.json").exists():
            test_probabilities = _predict_saved_scorer(frugal, saved_model_dir, sub_test, args)
        else:
            output_dir_model = run_scorer_dir / model_name
            _, test_probabilities = frugal.train_single_model(args, sub_train, sub_test, model_name, output_dir_model)
        if test_probabilities is None or len(test_probabilities) != len(sub_test):
            raise RuntimeError(f"FrugalGPT probabilities missing or misaligned for model {model_name}")
        mask = test_df["model_name"].astype(str) == model_name
        all_probabilities.loc[mask] = np.asarray(test_probabilities, dtype=float)

    if all_probabilities.isna().any():
        missing = test_df.loc[all_probabilities.isna(), "model_name"].value_counts().to_dict()
        raise RuntimeError(f"Missing FrugalGPT probabilities for test rows: {missing}")
    return all_probabilities


def _training_args(
    baseline_config: dict[str, Any],
    out_dir: Path,
    scorer_dir: Path,
    seed: int,
) -> argparse.Namespace:
    local_base = Path(
        baseline_config.get("frugalgpt_local_base", out_dir / "external_checkpoints/local_encoder")
    ).expanduser()
    if not local_base.exists():
        raise FileNotFoundError(f"FrugalGPT local encoder checkpoint not found: {local_base}")
    local_tokenizer_value = baseline_config.get("frugalgpt_local_tokenizer")
    local_tokenizer = Path(local_tokenizer_value).expanduser() if local_tokenizer_value else None
    return argparse.Namespace(
        local_base=local_base,
        local_tokenizer=local_tokenizer,
        backbone_type=str(baseline_config.get("frugalgpt_backbone_type", "sequence-classification")),
        pooling=str(baseline_config.get("frugalgpt_pooling", "cls")),
        trust_remote_code=bool(baseline_config.get("frugalgpt_trust_remote_code", False)),
        truncation_side=str(baseline_config.get("frugalgpt_truncation_side", "right")),
        max_length=int(baseline_config.get("frugalgpt_max_length", 128)),
        deepspeed=None,
        local_rank=-1,
        batch_size=int(baseline_config.get("frugalgpt_batch_size", 16)),
        eval_batch_size=int(baseline_config.get("frugalgpt_eval_batch_size", 64)),
        grad_accum=int(baseline_config.get("frugalgpt_grad_accum", 1)),
        lr=float(baseline_config.get("frugalgpt_lr", 2e-5)),
        weight_decay=float(baseline_config.get("frugalgpt_weight_decay", 0.01)),
        max_steps=_optional_int(baseline_config.get("frugalgpt_max_steps")),
        epochs=int(baseline_config.get("frugalgpt_epochs", 1)),
        warmup_ratio=float(baseline_config.get("frugalgpt_warmup_ratio", 0.06)),
        bf16=bool(baseline_config.get("frugalgpt_bf16", False)),
        seed=int(seed),
        log_interval=int(baseline_config.get("frugalgpt_log_interval", 50)),
        output_dir=scorer_dir,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return int(value)


def _predict_saved_scorer(frugal, model_dir: Path, test_df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    model, tokenizer = frugal.load_base_model_and_tokenizer(
        model_dir,
        None,
        num_labels=2,
        backbone_type=args.backbone_type,
        pooling=args.pooling,
        trust_remote_code=args.trust_remote_code,
        truncation_side=args.truncation_side,
    )
    dataset = frugal.TextClsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len=args.max_length)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            output = model(**inputs)
            batch_logits = getattr(output, "logits", None)
            if batch_logits is None and isinstance(output, (tuple, list)) and output:
                batch_logits = output[0]
            if batch_logits is None:
                raise ValueError("Saved FrugalGPT scorer did not return logits")
            logits.append(batch_logits.detach().cpu().float())
    return frugal.logits_to_probabilities(torch.cat(logits, dim=0).numpy())


def _evaluate_thresholds(
    *,
    test_df: pd.DataFrame,
    probabilities: pd.Series,
    thresholds: list[float],
    train_matrices,
    test_matrices,
    embeddings: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    model_names: list[str],
    score_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_single = BestSingleRouter().fit(train_matrices.query_info, train_matrices.utility).predict(test_matrices.query_info)
    baseline_mean = float(selected_values(test_matrices.utility, best_single).mean())
    oracle_mean = float(test_matrices.utility.max(axis=1).mean())
    knn = KNNRouter(int(config.get("routers", {}).get("knn_k", 15))).fit(
        train_matrices.query_info,
        train_matrices.utility,
        embeddings,
    ).predict(test_matrices.query_info, embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test_matrices.utility, knn).mean()))
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))

    work = test_df.copy().reset_index(drop=True)
    work["probability"] = probabilities.reset_index(drop=True).to_numpy(dtype=float)
    rows = []
    raw: dict[str, Any] = {
        "model_names": model_names,
        "score_threshold": float(score_threshold),
        "thresholds": thresholds,
        "predictions": {},
    }
    for idx, threshold in enumerate(thresholds):
        selected = _select_by_frugalgpt_rule(work, threshold)
        eval_row = evaluate_selection(
            method=f"frugalgpt_local_scorer_t{threshold:g}",
            selected_models=selected["selected_model"],
            matrices=test_matrices,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + 300 + idx,
            k=len(model_names),
            labels=selected["selected_model"],
        )
        classification = _classification_metrics(work, threshold)
        eval_row.update(
            {
                "prob_threshold": float(threshold),
                "score_threshold": float(score_threshold),
                **classification,
                "selected_label_rate": float(selected["selected_label"].mean()),
                "mean_selected_probability": float(selected["selected_probability"].mean()),
                "mean_selected_cost": float(selected["selected_cost"].mean()),
            }
        )
        rows.append(eval_row)
        raw["predictions"][str(threshold)] = {
            "selected_models": selected["selected_model"].astype(str).to_dict(),
            "selected_probability": selected["selected_probability"].astype(float).to_dict(),
            "selected_label": selected["selected_label"].astype(int).to_dict(),
        }
    return pd.DataFrame(rows), raw


def _select_by_frugalgpt_rule(work: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for query_id, group in work.groupby("record_index", dropna=False):
        sorted_group = group.sort_values(["cost", "probability"], ascending=[True, False])
        positives = sorted_group[sorted_group["probability"] >= threshold]
        if positives.empty:
            chosen = sorted_group.sort_values(["probability", "cost"], ascending=[False, True]).iloc[0]
        else:
            chosen = positives.iloc[0]
        rows.append(
            {
                "query_id": str(query_id),
                "selected_model": str(chosen["model_name"]),
                "selected_label": int(chosen["label"]),
                "selected_probability": float(chosen["probability"]),
                "selected_cost": float(chosen["cost"]),
            }
        )
    selected = pd.DataFrame(rows).set_index("query_id").sort_index()
    return selected


def _classification_metrics(work: pd.DataFrame, threshold: float) -> dict[str, float]:
    probabilities = work["probability"].to_numpy(dtype=float)
    labels = work["label"].to_numpy(dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    prompt_labels = work.groupby("record_index")["label"].max()
    prompt_predictions = work.assign(prediction=predictions).groupby("record_index")["prediction"].max()
    metrics = {
        "record_accuracy": float(accuracy_score(labels, predictions)),
        "prompt_accuracy": float(accuracy_score(prompt_labels, prompt_predictions)),
    }
    try:
        metrics["record_roc_auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["record_roc_auc"] = float("nan")
    return metrics


def _run_config_payload(baseline_config: dict[str, Any], score_threshold: float, scorer_dir: Path) -> dict[str, Any]:
    return {
        "score_threshold": float(score_threshold),
        "prob_thresholds": [float(v) for v in baseline_config.get("frugalgpt_prob_thresholds", [0.5])],
        "reuse_saved_scorers": bool(baseline_config.get("frugalgpt_reuse_saved_scorers", False)),
        "scorer_dir": str(scorer_dir),
        "epochs": int(baseline_config.get("frugalgpt_epochs", 1)),
        "max_steps": _optional_int(baseline_config.get("frugalgpt_max_steps")),
        "max_length": int(baseline_config.get("frugalgpt_max_length", 128)),
    }


def append_readme(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## FrugalGPT Split-Aligned Evaluation"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/30_frugalgpt_split_aligned.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        f"- `{RUN_DIRNAME}/scorers/`: per-model local scorer checkpoints when training was run.",
        f"- `{RUN_DIRNAME}/raw_predictions.json`: selected-model and probability evidence.",
        f"- `{RUN_DIRNAME}/run_config.json`: local training/evaluation settings.",
        "- `table_frugalgpt_split_aligned.csv`: RouteCode utility metrics for the local metric adapter.",
        "- `phase_e_frugalgpt_split_aligned_memo.md`: Phase E memo and caveats.",
        "",
        f"Artifact directory: `{run_dir}`.",
        "",
        "This is a local metric adapter around the LLMRouterBench FrugalGPT local scorer source; it is not an upstream published checkpoint.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Phase E FrugalGPT Split-Aligned Memo",
        "",
        f"Command: `python experiments/30_frugalgpt_split_aligned.py --config {config_path}`",
        "",
        f"Artifact directory: `{run_dir}`.",
        "",
        "This run evaluates a RouteCode-compatible local metric adapter for FrugalGPT local scorers.",
        "",
        "It trains or reuses per-model local scorers from the LLMRouterBench FrugalGPT source, then selects the cheapest predicted-positive model for each query. If no model is predicted positive, it falls back to the highest probability with cost as a deterministic tie-break.",
        "",
        "It is split-aligned with RouteCode and does not make external API calls. It is not an upstream published FrugalGPT checkpoint.",
        "",
        "## Evaluation Summary",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "phase_e_frugalgpt_split_aligned_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
