from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_embedllm_mf_help_runs_without_installed_wandb():
    script = (
        Path(__file__).resolve().parents[1]
        / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/mf.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--eval-mode" in completed.stdout
