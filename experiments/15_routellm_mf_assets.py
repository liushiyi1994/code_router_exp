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

from routecode.config import load_config, output_dir
from routecode.eval.external_baselines import (
    build_routellm_mf_assets,
    build_routellm_pairwise_records,
    choose_strong_weak_pair,
)
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


ASSET_DIRNAME = "routellm_mf_assets"
OFFICIAL_MODEL_FILE = (
    ROOT / "data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization/model.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    baseline_config = config.get("external_baselines", {})
    pair = choose_strong_weak_pair(
        prepared.matrices["train"].utility,
        strong_model=baseline_config.get("strong_model"),
        weak_model=baseline_config.get("weak_model"),
    )
    pairwise = build_routellm_pairwise_records(
        {
            "train": prepared.matrices["train"],
            "test": prepared.matrices["test"],
        },
        pair,
    )
    assets = build_routellm_mf_assets(pairwise, prepared.embeddings)

    assets_dir = out_dir / ASSET_DIRNAME
    assets_dir.mkdir(parents=True, exist_ok=True)
    train_path = assets_dir / "pairwise_train.json"
    test_path = assets_dir / "pairwise_test.json"
    prompt_index_path = assets_dir / "prompt_index.json"
    embeddings_path = assets_dir / "prompt_embeddings.npy"
    train_config_path = assets_dir / "mf_train_config.local.json"
    eval_config_path = assets_dir / "mf_eval_config.local.json"
    embedding_config_path = assets_dir / "embedding_config.local.yaml"
    embedding_cache_path = assets_dir / "embedding_cache.jsonl"
    metadata_path = assets_dir / "metadata.json"

    _write_json(train_path, assets.train_records)
    _write_json(test_path, assets.test_records)
    _write_json(prompt_index_path, assets.prompt_index)
    np.save(embeddings_path, assets.prompt_embeddings)
    _write_jsonl(embedding_cache_path, _embedding_cache_rows(assets))
    embedding_config_path.write_text(
        "\n".join(
            [
                "embedding_model:",
                "  api_model_name: routecode-cache",
                "  name: routecode-cache",
                f"embedding_cache_path: {embedding_cache_path.resolve()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    official_model_ids = _load_official_model_ids()
    pair_in_official_ids = pair.strong_model in official_model_ids and pair.weak_model in official_model_ids
    trainer_config = _trainer_config(
        baseline_config,
        train_path=train_path,
        embeddings_path=embeddings_path,
        save_path=assets_dir / "mf_model.pt",
        embedding_dim=int(assets.prompt_embeddings.shape[1]),
    )
    _write_json(train_config_path, trainer_config)
    eval_config = _eval_config(
        checkpoint_path=assets_dir / "mf_model.pt",
        embedding_config_path=embedding_config_path,
        embedding_dim=int(assets.prompt_embeddings.shape[1]),
    )
    _write_json(eval_config_path, eval_config)

    metadata = _metadata(
        config_path=config_path,
        config=config,
        pair=pair,
        assets=assets,
        pair_in_official_ids=pair_in_official_ids,
        train_config_path=train_config_path,
        eval_config_path=eval_config_path,
        embedding_cache_path=embedding_cache_path,
        official_model_ids_source=OFFICIAL_MODEL_FILE,
    )
    _write_json(metadata_path, metadata)
    table = _summary_table(assets, pair, metadata)
    table.to_csv(out_dir / "table_routellm_mf_assets.csv", index=False)
    write_memo(out_dir, config_path, assets_dir, table, metadata)
    append_readme(out_dir, config_path, assets_dir, table)
    print(f"Wrote RouteLLM MF trainer assets to {assets_dir}")


def _trainer_config(
    baseline_config: dict[str, Any],
    *,
    train_path: Path,
    embeddings_path: Path,
    save_path: Path,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "json_path": str(train_path),
        "npy_path": str(embeddings_path),
        "dim": int(embedding_dim),
        "requested_dim": int(baseline_config.get("mf_hidden_dim", embedding_dim)),
        "dim_note": (
            "LLMRouterBench MFModel_Train copies prompt_embeddings.npy into Q, "
            "so dim must equal the prompt embedding width."
        ),
        "use_proj": True,
        "batch_size": int(baseline_config.get("mf_batch_size", 64)),
        "num_epochs": int(baseline_config.get("mf_num_epochs", 30)),
        "alpha": float(baseline_config.get("mf_alpha", 0.1)),
        "lr": float(baseline_config.get("mf_lr", 3e-4)),
        "weight_decay": float(baseline_config.get("mf_weight_decay", 1e-5)),
        "device": str(baseline_config.get("mf_device", "cpu")),
        "save_path": str(save_path),
    }


def _eval_config(*, checkpoint_path: Path, embedding_config_path: Path, embedding_dim: int) -> dict[str, Any]:
    return {
        "mf": {
            "checkpoint_path": str(checkpoint_path),
            "hidden_size": int(embedding_dim),
            "num_models": 33,
            "text_dim": int(embedding_dim),
            "num_classes": 1,
            "use_proj": True,
            "embedding_config_path": str(embedding_config_path),
        }
    }


def _embedding_cache_rows(assets) -> list[dict[str, Any]]:
    rows_by_prompt: dict[str, dict[str, Any]] = {}
    for record in assets.train_records + assets.test_records:
        prompt = str(record["prompt"])
        if prompt in rows_by_prompt:
            continue
        query_id = str(record["query_id"])
        idx = int(assets.prompt_index[query_id])
        rows_by_prompt[prompt] = {
            "query_id": query_id,
            "prompt": prompt,
            "embedding": assets.prompt_embeddings[idx].astype(float).tolist(),
        }
    return list(rows_by_prompt.values())


def _metadata(
    *,
    config_path: str,
    config: dict[str, Any],
    pair,
    assets,
    pair_in_official_ids: bool,
    train_config_path: Path,
    eval_config_path: Path,
    embedding_cache_path: Path,
    official_model_ids_source: Path,
) -> dict[str, Any]:
    train_ids = {row["query_id"] for row in assets.train_records}
    test_ids = {row["query_id"] for row in assets.test_records}
    official_trainer_compatible = bool(
        pair_in_official_ids
        and len(assets.train_records) > 0
        and assets.prompt_embeddings.ndim == 2
        and all(row["winner"] in {"model_a", "model_b"} for row in assets.train_records)
    )
    return {
        "config_path": config_path,
        "data_source": config.get("data", {}).get("source", "synthetic"),
        "strong_model": pair.strong_model,
        "weak_model": pair.weak_model,
        "model_a": pair.strong_model,
        "model_b": pair.weak_model,
        "split_aligned_with_routecode": True,
        "official_trainer_compatible": official_trainer_compatible,
        "official_routellm_result": False,
        "routecode_metric_compatible": False,
        "winner_objective": "quality",
        "utility_fields_retained": True,
        "query_id_overlap_train_test": len(train_ids & test_ids),
        "record_counts": {
            "train": len(assets.train_records),
            "test": len(assets.test_records),
        },
        "prompt_count": int(assets.prompt_embeddings.shape[0]),
        "embedding_dim": int(assets.prompt_embeddings.shape[1]),
        "pair_in_official_model_ids": pair_in_official_ids,
        "official_model_ids_source": str(official_model_ids_source),
        "train_config_path": str(train_config_path),
        "eval_config_path": str(eval_config_path),
        "embedding_cache_path": str(embedding_cache_path),
        "compatibility_note": (
            "Assets are ready for the local LLMRouterBench RouteLLM MF trainer and "
            "cache-backed upstream RouteLLM MF evaluation. This script does not train "
            "or evaluate an official RouteLLM MF model."
        ),
    }


def _summary_table(assets, pair, metadata: dict[str, Any]) -> pd.DataFrame:
    rows = [
        _summary_row("train", assets.train_records, pair, metadata),
        _summary_row("test", assets.test_records, pair, metadata),
        _summary_row("overall", assets.train_records + assets.test_records, pair, metadata),
    ]
    return pd.DataFrame(rows)


def _summary_row(split: str, records: list[dict[str, Any]], pair, metadata: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    count = int(len(frame))
    model_a_wins = int((frame["winner"] == "model_a").sum()) if count else 0
    model_b_wins = int((frame["winner"] == "model_b").sum()) if count else 0
    ties = int((frame["winner"] == "tie").sum()) if count else 0
    utility_model_a_wins = int((frame["utility_winner"] == "model_a").sum()) if count else 0
    utility_model_b_wins = int((frame["utility_winner"] == "model_b").sum()) if count else 0
    utility_ties = int((frame["utility_winner"] == "tie").sum()) if count else 0
    return {
        "split": split,
        "record_count": count,
        "decisive_count": model_a_wins + model_b_wins,
        "tie_count": ties,
        "model_a_quality_win_count": model_a_wins,
        "model_b_quality_win_count": model_b_wins,
        "model_a_quality_win_rate": model_a_wins / count if count else 0.0,
        "model_b_quality_win_rate": model_b_wins / count if count else 0.0,
        "quality_tie_rate": ties / count if count else 0.0,
        "model_a_utility_win_count": utility_model_a_wins,
        "model_b_utility_win_count": utility_model_b_wins,
        "utility_tie_count": utility_ties,
        "mean_utility_margin_model_a_minus_b": float(frame["utility_margin_model_a_minus_b"].mean()) if count else 0.0,
        "strong_model": pair.strong_model,
        "weak_model": pair.weak_model,
        "split_aligned_with_routecode": metadata["split_aligned_with_routecode"],
        "official_trainer_compatible": metadata["official_trainer_compatible"],
        "official_routellm_result": metadata["official_routellm_result"],
        "routecode_metric_compatible": metadata["routecode_metric_compatible"],
        "implementation_note": (
            "Official-trainer-compatible RouteLLM MF assets with local RouteCode "
            "embeddings; not a trained RouteLLM MF result."
        ),
    }


def _load_official_model_ids() -> set[str]:
    if not OFFICIAL_MODEL_FILE.exists():
        return set()
    spec = importlib.util.spec_from_file_location("llmrouterbench_routellm_mf_model", OFFICIAL_MODEL_FILE)
    if spec is None or spec.loader is None:
        return set()
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return set()
    return set(getattr(module, "MODEL_IDS", {}).keys())


def append_readme(out_dir: Path, config_path: str, assets_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## RouteLLM MF Trainer Assets"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/15_routellm_mf_assets.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        f"- `{ASSET_DIRNAME}/pairwise_train.json`: quality-winner RouteLLM-MF train records, with utility fields retained.",
        f"- `{ASSET_DIRNAME}/pairwise_test.json`: quality-winner RouteLLM-MF test records, with ties retained.",
        f"- `{ASSET_DIRNAME}/prompt_embeddings.npy`: RouteCode deterministic query embeddings aligned to `idx`.",
        f"- `{ASSET_DIRNAME}/prompt_index.json`: query-id to MF prompt-index mapping.",
        f"- `{ASSET_DIRNAME}/mf_train_config.local.json`: local CPU config for the LLMRouterBench RouteLLM MF trainer.",
        f"- `{ASSET_DIRNAME}/mf_eval_config.local.json`, `{ASSET_DIRNAME}/embedding_config.local.yaml`, and `{ASSET_DIRNAME}/embedding_cache.jsonl`: no-API config/cache for bounded upstream RouteLLM MF evaluation.",
        "- `table_routellm_mf_assets.csv`: asset compatibility and winner-distribution summary.",
        "- `phase_e_routellm_mf_assets_memo.md`: Phase E memo explaining the remaining train/eval step.",
        "",
        f"Artifact directory: `{assets_dir}`.",
        "",
        "These files are trainer/eval inputs, not a trained RouteLLM MF result.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(
    out_dir: Path,
    config_path: str,
    assets_dir: Path,
    table: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Phase E RouteLLM MF Asset Memo",
        "",
        f"Command: `python experiments/15_routellm_mf_assets.py --config {config_path}`",
        "",
        f"Artifact directory: `{assets_dir}`.",
        "",
        f"Binary pair: strong/model_a `{metadata['strong_model']}`, weak/model_b `{metadata['weak_model']}`.",
        "",
        "These assets are ready for the local LLMRouterBench RouteLLM MF trainer: they include `idx`, score/cost fields, `prompt_embeddings.npy`, and a local CPU training config. They also include a cache-backed upstream RouteLLM MF evaluation config that can evaluate `mf_model.pt` without embedding API calls.",
        "",
        "This is not a trained RouteLLM MF result. The next step is to run the local MF trainer and evaluate the checkpoint on the RouteCode test split.",
        "",
        "## Compatibility",
        "",
        f"- `split_aligned_with_routecode`: `{metadata['split_aligned_with_routecode']}`",
        f"- Train/test query overlap: `{metadata['query_id_overlap_train_test']}`",
        f"- `official_trainer_compatible`: `{metadata['official_trainer_compatible']}`",
        f"- `official_routellm_result`: `{metadata['official_routellm_result']}`",
        f"- Pair present in official `MODEL_IDS`: `{metadata['pair_in_official_model_ids']}`",
        f"- Local training config: `{metadata['train_config_path']}`",
        f"- Local eval config: `{metadata['eval_config_path']}`",
        f"- Local eval embedding cache: `{metadata['embedding_cache_path']}`",
        "",
        "## Asset Summary",
        "",
        _markdown_table(table),
        "",
        "## Remaining External-Baseline Gap",
        "",
        "- Run the LLMRouterBench RouteLLM MF trainer on `mf_train_config.local.json`.",
        "- Run the cache-backed upstream RouteLLM MF evaluator on `mf_eval_config.local.json` and `pairwise_test.json`.",
        "- Convert selections back to RouteCode utility metrics separately; the upstream eval output is accuracy/cost readiness evidence.",
        "- Keep BERT, GraphRouter, and Avengers/Avengers-Pro as separate adapter tasks.",
        "",
    ]
    (out_dir / "phase_e_routellm_mf_assets_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
