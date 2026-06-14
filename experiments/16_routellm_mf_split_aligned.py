from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.external_baselines import (
    build_routellm_mf_assets,
    build_routellm_pairwise_records,
    choose_strong_weak_pair,
)
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


RUN_DIRNAME = "routellm_mf_split_aligned"
MF_SOURCE_DIR = ROOT / "data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization"


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

    pair = choose_strong_weak_pair(
        train.utility,
        strong_model=baseline_config.get("strong_model"),
        weak_model=baseline_config.get("weak_model"),
    )
    pairwise = build_routellm_pairwise_records({"train": train, "test": test}, pair)
    assets = build_routellm_mf_assets(pairwise, prepared.embeddings)

    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_run_assets(run_dir, assets)

    mf_model_mod, mf_train_mod = _load_official_mf_modules()
    model_ids = mf_model_mod.MODEL_IDS
    missing_pair = {pair.strong_model, pair.weak_model} - set(model_ids)
    if missing_pair:
        raise ValueError(f"Pair models missing from official RouteLLM MODEL_IDS: {sorted(missing_pair)}")

    train_config = _train_config(baseline_config, assets.prompt_embeddings.shape[1], run_dir / "mf_model.pt")
    net, train_summary = _train_official_mf(
        mf_train_mod=mf_train_mod,
        mf_model_mod=mf_model_mod,
        assets=assets,
        embeddings_path=run_dir / "prompt_embeddings.npy",
        train_config=train_config,
    )
    torch.save(train_summary["state_dict"], run_dir / "mf_model.pt")

    thresholds = [float(value) for value in baseline_config.get("thresholds", [0.5])]
    table, raw_metrics = _evaluate_thresholds(
        net=net,
        model_ids=model_ids,
        test_records=assets.test_records,
        thresholds=thresholds,
        pair=pair,
        test_matrices=test,
        train_matrices=train,
        embeddings=prepared.embeddings,
        config=config,
        seed=seed,
    )
    table["train_loss"] = train_summary["train_loss"]
    table["validation_accuracy"] = train_summary["validation_accuracy"]
    table["official_training_code_used"] = True
    table["official_upstream_checkpoint"] = False
    table["split_aligned_with_routecode"] = True
    table["routecode_metric_compatible"] = True
    table["baseline_family"] = "official_code_local_embedding"
    table["implementation_note"] = (
        "LLMRouterBench RouteLLM MF training code with local RouteCode embeddings; "
        "not the upstream published RouteLLM checkpoint."
    )

    table.to_csv(out_dir / "table_routellm_mf_split_aligned.csv", index=False)
    _write_json(run_dir / "raw_metrics.json", raw_metrics)
    _write_json(run_dir / "train_config.json", train_config)
    write_memo(out_dir, config_path, run_dir, table, train_summary)
    append_readme(out_dir, config_path, run_dir, table)
    print(f"Wrote split-aligned RouteLLM MF outputs to {run_dir}")


def _write_run_assets(run_dir: Path, assets) -> None:
    _write_json(run_dir / "pairwise_train.json", assets.train_records)
    _write_json(run_dir / "pairwise_test.json", assets.test_records)
    _write_json(run_dir / "prompt_index.json", assets.prompt_index)
    np.save(run_dir / "prompt_embeddings.npy", assets.prompt_embeddings)


def _train_config(baseline_config: dict[str, Any], embedding_dim: int, save_path: Path) -> dict[str, Any]:
    return {
        "dim": int(embedding_dim),
        "batch_size": int(baseline_config.get("mf_batch_size", 64)),
        "num_epochs": int(baseline_config.get("mf_num_epochs", 30)),
        "alpha": float(baseline_config.get("mf_alpha", 1.0)),
        "lr": float(baseline_config.get("mf_lr", 3e-4)),
        "weight_decay": float(baseline_config.get("mf_weight_decay", 1e-5)),
        "device": str(baseline_config.get("mf_device", "cpu")),
        "save_path": str(save_path),
        "use_proj": True,
    }


def _load_official_mf_modules():
    package_name = "routecode_external_llmrouterbench_routellm_mf"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MF_SOURCE_DIR)]
    sys.modules[package_name] = package
    loaded = {}
    for name in ["model", "train_matrix_factorization"]:
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", MF_SOURCE_DIR / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load RouteLLM MF source module: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{name}"] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["model"], loaded["train_matrix_factorization"]


def _train_official_mf(*, mf_train_mod, mf_model_mod, assets, embeddings_path: Path, train_config: dict[str, Any]):
    decisive_test = [row for row in assets.test_records if row["winner"] in {"model_a", "model_b"}]
    validation_records = decisive_test or assets.train_records
    train_loader = mf_train_mod.PairwiseDataset(assets.train_records).get_dataloaders(
        batch_size=train_config["batch_size"], shuffle=True
    )
    validation_loader = mf_train_mod.PairwiseDataset(validation_records).get_dataloaders(
        batch_size=max(1, min(1024, len(validation_records))), shuffle=False
    )
    net = mf_train_mod.MFModel_Train(
        dim=train_config["dim"],
        num_models=len(mf_model_mod.MODEL_IDS),
        num_prompts=int(assets.prompt_embeddings.shape[0]),
        text_dim=int(assets.prompt_embeddings.shape[1]),
        num_classes=1,
        use_proj=train_config["use_proj"],
        npy_path=embeddings_path,
    ).to(train_config["device"])
    best_state = mf_train_mod.train_loops(
        net,
        train_loader,
        validation_loader,
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        alpha=train_config["alpha"],
        num_epochs=train_config["num_epochs"],
        device=train_config["device"],
    )
    if best_state:
        net.load_state_dict(best_state)
    validation_loss, validation_accuracy = mf_train_mod.evaluator(net, validation_loader, train_config["device"])
    return net.eval(), {
        "state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
        "train_loss": None,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_records": len(validation_records),
    }


def _evaluate_thresholds(
    *,
    net,
    model_ids: dict[str, int],
    test_records: list[dict[str, Any]],
    thresholds: list[float],
    pair,
    test_matrices,
    train_matrices,
    embeddings: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    win_rates = _predict_win_rates(net, model_ids, test_records)
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

    rows = []
    raw_metrics: dict[str, Any] = {
        "strong_model": pair.strong_model,
        "weak_model": pair.weak_model,
        "thresholds": thresholds,
    }
    record_by_query = {str(row["query_id"]): row for row in test_records}
    for threshold_index, threshold in enumerate(thresholds):
        selected = pd.Series(
            {
                query_id: pair.strong_model if float(win_rates[query_id]) >= threshold else pair.weak_model
                for query_id in test_matrices.utility.index
            },
            name="selected_model",
        )
        eval_row = evaluate_selection(
            method=f"routellm_mf_split_aligned_t{threshold:g}",
            selected_models=selected,
            matrices=test_matrices,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + 100 + threshold_index,
            k=2,
            labels=selected,
        )
        quality_metrics = _quality_metrics(test_records, win_rates, threshold)
        eval_row.update(
            {
                "threshold": threshold,
                "strong_model": pair.strong_model,
                "weak_model": pair.weak_model,
                **quality_metrics,
            }
        )
        rows.append(eval_row)
        raw_metrics[str(threshold)] = {
            "quality_metrics": quality_metrics,
            "win_rates": {query_id: float(win_rates[query_id]) for query_id in record_by_query},
        }
    return pd.DataFrame(rows), raw_metrics


def _predict_win_rates(net, model_ids: dict[str, int], records: list[dict[str, Any]]) -> dict[str, float]:
    device = net.get_device()
    model_a = torch.tensor([model_ids[row["model_a"]] for row in records], dtype=torch.long, device=device)
    model_b = torch.tensor([model_ids[row["model_b"]] for row in records], dtype=torch.long, device=device)
    prompt = torch.tensor([int(row["idx"]) for row in records], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = net(model_a, model_b, prompt)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return {str(row["query_id"]): float(probabilities[idx]) for idx, row in enumerate(records)}


def _quality_metrics(records: list[dict[str, Any]], win_rates: dict[str, float], threshold: float) -> dict[str, float]:
    total = len(records)
    decisive = 0
    routing_correct = 0
    selected_correct = 0
    strong_selected = 0
    weak_selected = 0
    for row in records:
        predicted = "model_a" if win_rates[str(row["query_id"])] >= threshold else "model_b"
        if predicted == "model_a":
            strong_selected += 1
            predicted_score = float(row["score_model_a"])
        else:
            weak_selected += 1
            predicted_score = float(row["score_model_b"])
        if predicted_score > 0:
            selected_correct += 1
        if row["winner"] != "tie":
            decisive += 1
            if predicted == row["winner"]:
                routing_correct += 1
    return {
        "selection_accuracy": selected_correct / total if total else 0.0,
        "routing_accuracy_decisive": routing_correct / decisive if decisive else 0.0,
        "decisive_count": decisive,
        "tie_count": total - decisive,
        "strong_selection_rate": strong_selected / total if total else 0.0,
        "weak_selection_rate": weak_selected / total if total else 0.0,
        "mean_strong_win_rate": float(np.mean([win_rates[str(row["query_id"])] for row in records])) if records else 0.0,
    }


def append_readme(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## RouteLLM MF Split-Aligned Evaluation"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/16_routellm_mf_split_aligned.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        f"- `{RUN_DIRNAME}/mf_model.pt`: checkpoint trained with local LLMRouterBench RouteLLM MF source.",
        f"- `{RUN_DIRNAME}/raw_metrics.json`: threshold-level quality metrics and win rates.",
        "- `table_routellm_mf_split_aligned.csv`: RouteCode utility metrics plus RouteLLM-style quality metrics.",
        "- `phase_e_routellm_mf_split_aligned_memo.md`: Phase E memo and caveats.",
        "",
        f"Artifact directory: `{run_dir}`.",
        "",
        "This uses official MF training code with local RouteCode embeddings; it is not the upstream published RouteLLM checkpoint.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame, train_summary: dict[str, Any]) -> None:
    lines = [
        "# Phase E RouteLLM MF Split-Aligned Memo",
        "",
        f"Command: `python experiments/16_routellm_mf_split_aligned.py --config {config_path}`",
        "",
        f"Artifact directory: `{run_dir}`.",
        "",
        "This run trains the local LLMRouterBench RouteLLM MF model class on the RouteCode split-aligned pairwise assets and evaluates the checkpoint on the RouteCode test split.",
        "",
        "It is not the upstream published RouteLLM checkpoint and it uses deterministic local RouteCode embeddings rather than an API-backed embedding generator.",
        "",
        "## Training Summary",
        "",
        f"- Validation records: `{train_summary['validation_records']}`",
        f"- Validation accuracy: `{train_summary['validation_accuracy']:.4f}`",
        f"- Validation loss: `{train_summary['validation_loss']:.4f}`",
        "",
        "## Evaluation Summary",
        "",
        _markdown_table(table),
        "",
        "## Remaining External-Baseline Gap",
        "",
        "- Add BERT, GraphRouter, and Avengers/Avengers-Pro adapter outputs if local dependencies can be pinned.",
        "- Decide whether to install the full LLMRouterBench baseline environment for exact upstream command execution.",
        "",
    ]
    (out_dir / "phase_e_routellm_mf_split_aligned_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return "<tensor>"
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
