from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "experiments" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tiny_bert(path: Path) -> None:
    path.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "def", "return", "input", "output", "0", "1"]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    tokenizer.save_pretrained(path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(path)


def test_frugalgpt_cli_metrics_scores_saved_exact_cli_scorers(tmp_path):
    train_script = _load_script("30_frugalgpt_split_aligned.py")
    cli_metric_script = _load_script("32_frugalgpt_cli_metrics.py")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n", encoding="utf-8")
    local_base = tmp_path / "tiny_bert"
    _write_tiny_bert(local_base)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 13",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 24",
                "  n_models: 2",
                "  n_domains: 2",
                "  n_route_labels: 4",
                "  embedding_dim: 6",
                "  model_ids: [m0, m1]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.10",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "external_baselines:",
                f"  frugalgpt_local_base: {local_base}",
                "  frugalgpt_epochs: 1",
                "  frugalgpt_max_steps: 1",
                "  frugalgpt_max_length: 16",
                "  frugalgpt_batch_size: 4",
                "  frugalgpt_eval_batch_size: 8",
                "  frugalgpt_prob_thresholds: [0.5]",
                "  frugalgpt_reuse_saved_scorers: false",
                f"  frugalgpt_cli_scorer_dir: {out_dir / 'frugalgpt_split_aligned_metric' / 'scorers'}",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    train_script.run(str(config_path))
    cli_metric_script.run(str(config_path))

    table_path = out_dir / "table_frugalgpt_cli_metrics.csv"
    memo_path = out_dir / "phase_e_frugalgpt_cli_metrics_memo.md"
    raw_path = out_dir / "frugalgpt_cli_metrics" / "raw_predictions.json"
    assert table_path.exists()
    assert memo_path.exists()
    assert raw_path.exists()

    table = pd.read_csv(table_path)
    assert table["method"].tolist() == ["frugalgpt_cli_saved_scorer_t0.5"]
    row = table.iloc[0]
    assert row["baseline_family"] == "frugalgpt_exact_cli_saved_scorer_postprocessed"
    assert bool(row["split_aligned_with_routecode"])
    assert bool(row["routecode_metric_compatible"])
    assert bool(row["exact_upstream_command"])
    assert bool(row["exact_command_output"])
    assert not bool(row["official_upstream_checkpoint"])
    assert row["prob_threshold"] == 0.5
    assert row["prediction_count"] > 0

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## FrugalGPT CLI Metrics" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "saved scorer directories emitted by the FrugalGPT command" in memo
