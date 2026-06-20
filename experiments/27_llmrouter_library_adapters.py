from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.llmrouter_library_adapters import evaluate_llmrouter_library_adapters
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


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
    adapter_config = config.get("llmrouter_library_adapters", {})
    bootstrap = config.get("bootstrap", {})
    seed = int(config.get("run", {}).get("random_seed", 0))
    llmrouter_root = Path(adapter_config.get("llmrouter_root", ROOT / "data/raw/external/LLMRouter"))
    if not llmrouter_root.is_absolute():
        llmrouter_root = ROOT / llmrouter_root

    table = evaluate_llmrouter_library_adapters(
        train,
        test,
        prepared.embeddings,
        output_dir=out_dir,
        llmrouter_root=llmrouter_root,
        seed=seed,
        n_bootstrap=int(bootstrap.get("n_bootstrap", 300)),
        ci=float(bootstrap.get("ci", 0.95)),
        knn_k=int(adapter_config.get("knn_k", config.get("routers", {}).get("knn_k", 15))),
        svm_kernel=str(adapter_config.get("svm_kernel", "rbf")),
    )
    table.to_csv(out_dir / "table_llmrouter_library_adapters.csv", index=False)
    write_memo(out_dir, config_path, table, llmrouter_root)
    append_readme(out_dir, config_path, table)
    print(f"Wrote LLMRouter library adapter outputs to {out_dir}")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, llmrouter_root: Path) -> None:
    best = table.sort_values("mean_utility", ascending=False).iloc[0] if not table.empty else None
    lines = [
        "# Phase E LLMRouter Library Adapter Memo",
        "",
        f"Command: `python experiments/27_llmrouter_library_adapters.py --config {config_path}`",
        "",
        "This run trains local LLMRouter trainer classes on RouteCode precomputed embeddings and evaluates the saved sklearn artifacts on the RouteCode test split.",
        "",
        "These are not exact upstream command-path results: the script does not call LLMRouter route methods, does not compute Longformer embeddings, and makes no external API calls.",
        "",
        f"LLMRouter checkout: `{llmrouter_root}`",
        "",
        "Outputs:",
        "",
        "- `table_llmrouter_library_adapters.csv`",
        "- `llmrouter_library_adapters/knn_model.pkl`",
        "- `llmrouter_library_adapters/svm_model.pkl` when the train split has at least two oracle-winner classes",
        "- `llmrouter_library_adapters/knnrouter_train.yaml` and `svmrouter_train.yaml` for exact upstream LLMRouter training and route-only inference CLI smoke checks",
        "- `llmrouter_library_adapters/query_train.jsonl`, `routing_train.jsonl`, `query_embeddings.pt`, `query_embedding_lookup.pt`, `query_inference_smoke.jsonl`, and `llm_candidates.json`",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "Best adapter row:",
                "",
                f"- `{best['method']}` mean utility `{float(best['mean_utility']):.4f}` and recovered gap vs oracle `{float(best['recovered_gap_vs_oracle']):.4f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "Compatibility:",
            "",
            _markdown_table(
                table[
                    [
                        "method",
                        "mean_utility",
                        "recovered_gap_vs_oracle",
                        "split_aligned_with_routecode",
                        "routecode_metric_compatible",
                        "exact_upstream_command",
                    ]
                ]
                if not table.empty
                else table
            ),
            "",
        ]
    )
    (out_dir / "phase_e_llmrouter_library_adapters_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## LLMRouter Library Adapters"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/27_llmrouter_library_adapters.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_llmrouter_library_adapters.csv`: split-aligned RouteCode utility metrics for local LLMRouter trainer-class adapters.",
        "- `phase_e_llmrouter_library_adapters_memo.md`: memo explaining compatibility and remaining upstream-command gap.",
        "- `llmrouter_library_adapters/`: saved local sklearn artifacts trained by LLMRouter trainer classes.",
        "- `llmrouter_library_adapters/*router_train.yaml`: split-aligned assets for exact upstream LLMRouter training and route-only inference CLI smoke checks.",
        "- `llmrouter_library_adapters/query_embedding_lookup.pt` and `query_inference_smoke.jsonl`: precomputed query embedding cache and bounded route-only smoke input for upstream inference CLI checks.",
        "",
        "These metric rows use local LLMRouter trainer classes but are not exact upstream command-path metric results. Exact upstream training and route-only inference CLI smoke evidence is tracked separately in the external-command readiness table.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "routecode_metric_compatible",
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


if __name__ == "__main__":
    main()
