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
from routecode.eval.external_baseline_assets import build_external_baseline_assets, write_external_baseline_assets
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


ADAPTER_SCRIPT = ROOT / "experiments/30_frugalgpt_split_aligned.py"
RUN_DIRNAME = "frugalgpt_cli_metrics"


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
    np.random.seed(seed)

    assets = build_external_baseline_assets({"train": train, "test": test}, prepared.embeddings)
    written = write_external_baseline_assets(assets, out_dir)
    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = _load_adapter_script()
    frugal = adapter._load_frugalgpt_module()
    score_threshold = float(baseline_config.get("frugalgpt_score_threshold", 0.5))
    train_df = frugal.load_jsonl_split(written.frugalgpt_train_path, split="train")
    test_df = frugal.load_jsonl_split(written.frugalgpt_test_path, split="test")
    train_df["label"] = (train_df["score"] >= score_threshold).astype(int)
    test_df["label"] = (test_df["score"] >= score_threshold).astype(int)
    train_df["dataset_name"] = train_df["dataset_name"].fillna("unknown")
    test_df["dataset_name"] = test_df["dataset_name"].fillna("unknown")

    scorer_dir = _scorer_dir(out_dir, baseline_config)
    model_names = sorted(test_df["model_name"].dropna().astype(str).unique().tolist())
    probabilities = _score_saved_scorers(
        adapter=adapter,
        frugal=frugal,
        baseline_config=baseline_config,
        out_dir=out_dir,
        scorer_dir=scorer_dir,
        test_df=test_df,
        model_names=model_names,
        seed=seed,
    )
    thresholds = [float(value) for value in baseline_config.get("frugalgpt_prob_thresholds", [0.5])]
    table, raw_predictions = adapter._evaluate_thresholds(
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
    table["method"] = table["prob_threshold"].map(lambda value: f"frugalgpt_cli_saved_scorer_t{float(value):g}")
    table["split_aligned_with_routecode"] = True
    table["routecode_metric_compatible"] = True
    table["official_training_code_used"] = True
    table["official_upstream_checkpoint"] = False
    table["exact_upstream_command"] = True
    table["exact_command_output"] = True
    table["baseline_family"] = "frugalgpt_exact_cli_saved_scorer_postprocessed"
    table["prediction_source"] = str(scorer_dir)
    table["prediction_count"] = int(test.utility.shape[0])
    table["implementation_note"] = (
        "RouteCode post-processing over saved scorer directories emitted by the FrugalGPT command. "
        "The upstream command emits scorer checkpoints and aggregate logs, not RouteCode utility metrics."
    )

    table.to_csv(out_dir / "table_frugalgpt_cli_metrics.csv", index=False)
    raw_predictions["prediction_source"] = str(scorer_dir)
    raw_predictions["exact_command_output"] = True
    _write_json(run_dir / "raw_predictions.json", raw_predictions)
    _write_json(
        run_dir / "run_config.json",
        {
            "config_path": config_path,
            "scorer_dir": str(scorer_dir),
            "score_threshold": score_threshold,
            "prob_thresholds": thresholds,
            "prediction_count": int(test.utility.shape[0]),
        },
    )
    write_memo(out_dir, config_path, scorer_dir, table)
    append_readme(out_dir, config_path, scorer_dir, table)
    print(f"Wrote FrugalGPT CLI metric outputs to {out_dir}")


def _load_adapter_script():
    spec = importlib.util.spec_from_file_location("routecode_frugalgpt_split_aligned_adapter", ADAPTER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FrugalGPT adapter script: {ADAPTER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["routecode_frugalgpt_split_aligned_adapter"] = module
    spec.loader.exec_module(module)
    return module


def _scorer_dir(out_dir: Path, baseline_config: dict[str, Any]) -> Path:
    configured = baseline_config.get("frugalgpt_cli_scorer_dir")
    if configured:
        return Path(configured).expanduser()
    return out_dir / "frugalgpt_split_aligned/output"


def _score_saved_scorers(
    *,
    adapter,
    frugal,
    baseline_config: dict[str, Any],
    out_dir: Path,
    scorer_dir: Path,
    test_df: pd.DataFrame,
    model_names: list[str],
    seed: int,
) -> pd.Series:
    args = adapter._training_args(baseline_config, out_dir, scorer_dir, seed)
    all_probabilities = pd.Series(np.nan, index=test_df.index, dtype=float)
    missing_dirs: list[str] = []
    for model_name in model_names:
        model_dir = scorer_dir / model_name
        if not (model_dir / "scorer_meta.json").exists():
            missing_dirs.append(model_name)
            continue
        sub_test = test_df[test_df["model_name"].astype(str) == model_name].reset_index(drop=True)
        test_probabilities = adapter._predict_saved_scorer(frugal, model_dir, sub_test, args)
        if test_probabilities is None or len(test_probabilities) != len(sub_test):
            raise RuntimeError(f"FrugalGPT CLI probabilities missing or misaligned for model {model_name}")
        mask = test_df["model_name"].astype(str) == model_name
        all_probabilities.loc[mask] = np.asarray(test_probabilities, dtype=float)
    if missing_dirs:
        raise FileNotFoundError(
            "Missing FrugalGPT saved scorer directories for models: " + ", ".join(sorted(missing_dirs))
        )
    if all_probabilities.isna().any():
        missing = test_df.loc[all_probabilities.isna(), "model_name"].value_counts().to_dict()
        raise RuntimeError(f"Missing FrugalGPT CLI probabilities for test rows: {missing}")
    return all_probabilities


def write_memo(out_dir: Path, config_path: str, scorer_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Phase E FrugalGPT CLI Metrics Memo",
        "",
        f"Command: `python experiments/32_frugalgpt_cli_metrics.py --config {config_path}`",
        "",
        f"Saved scorer source: `{scorer_dir}`.",
        "",
        "This table scores saved scorer directories emitted by the FrugalGPT command with RouteCode test-split utility. The upstream command writes scorer checkpoints and aggregate logs, not RouteCode utility metrics, so the metrics here are RouteCode post-processing over exact command outputs.",
        "",
        "Outputs:",
        "",
        "- `table_frugalgpt_cli_metrics.csv`",
        f"- `{RUN_DIRNAME}/raw_predictions.json`",
        f"- `{RUN_DIRNAME}/run_config.json`",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "phase_e_frugalgpt_cli_metrics_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, scorer_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## FrugalGPT CLI Metrics"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/32_frugalgpt_cli_metrics.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_frugalgpt_cli_metrics.csv`: RouteCode utility metrics over FrugalGPT exact-command saved scorer outputs.",
        "- `phase_e_frugalgpt_cli_metrics_memo.md`: compatibility notes for these post-processed exact-command rows.",
        f"- `{RUN_DIRNAME}/raw_predictions.json`: selected-model and probability evidence.",
        "",
        f"Saved scorer source: `{scorer_dir}`.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
                    "exact_upstream_command",
                ]
            ]
            if not table.empty
            else table
        ),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
