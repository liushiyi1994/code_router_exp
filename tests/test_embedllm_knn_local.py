from __future__ import annotations

from pathlib import Path


def test_embedllm_knn_cli_uses_defined_argparse_destinations():
    source = (
        Path(__file__).resolve().parents[1]
        / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/knn.py"
    ).read_text(encoding="utf-8")

    assert "args.train_csv_path" in source
    assert "args.test_csv_path" in source
    assert "args.save_train_x_path" in source
    assert "args.save_train_y_path" in source
    assert "args.save_test_x_path" in source
    assert "args.save_test_y_path" in source
    assert "args.train_csv," not in source
    assert "args.test_csv)" not in source
    assert "args.save_train_x," not in source
    assert "args.save_train_y)" not in source
    assert "args.save_test_x," not in source
    assert "args.save_test_y)" not in source
