from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.external_command_readiness import inspect_external_command_readiness
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    readiness_config = config.get("external_command_readiness", {})
    project_root = Path(readiness_config.get("project_root", ROOT)).expanduser()
    table = inspect_external_command_readiness(project_root, result_dir=out_dir)
    table.to_csv(out_dir / "table_external_command_readiness.csv", index=False)
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote external command readiness outputs to {out_dir}")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    runnable = int(table["runnable_now"].sum()) if not table.empty else 0
    exact_ready = int((table["runnable_now"] & table["exact_upstream_command"]).sum()) if not table.empty else 0
    lines = [
        "# Phase E External Command Readiness Memo",
        "",
        f"Command: `python experiments/21_external_command_readiness.py --config {config_path}`",
        "",
        "This memo records exact upstream-command readiness for the remaining external baselines. It checks local files, Python modules, and environment-sensitive embedding configs only. It performs no downloads and makes no external API calls.",
        "",
        f"Runnable rows now: `{runnable}`.",
        f"Runnable exact upstream-command rows now: `{exact_ready}`.",
        "",
        "## Summary",
        "",
        _markdown_table(_summary_table(table)),
        "",
        "## Interpretation",
        "",
        *_interpretation_lines(table),
        "",
    ]
    (out_dir / "phase_e_external_command_readiness_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## External Command Readiness"
    exact_ready = int((table["runnable_now"] & table["exact_upstream_command"]).sum()) if not table.empty else 0
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/21_external_command_readiness.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_external_command_readiness.csv`: reproducible readiness table for exact upstream external-baseline commands.",
        "- `phase_e_external_command_readiness_memo.md`: memo explaining runnable rows and blockers.",
        "",
        f"Runnable exact upstream-command rows now: `{exact_ready}`.",
        "",
        _markdown_table(_summary_table(table)),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "check_id",
        "status",
        "runnable_now",
        "no_api_compatible",
        "routecode_metric_compatible",
        "exact_upstream_command",
        "blocking_reasons",
        "execution_evidence",
    ]
    return table.loc[:, [column for column in columns if column in table.columns]]


def _interpretation_lines(table: pd.DataFrame) -> list[str]:
    lines = [
        "- `routecode_local_routellm_mf_metric` is the current metric-bearing RouteLLM-MF local-code row.",
        "- The exact RouteLLM-MF training CLI can be runnable/executed without API calls when its local pairwise assets and dependencies are present.",
    ]
    rows = table.set_index("check_id") if "check_id" in table.columns else pd.DataFrame()
    if "routellm_mf_eval_cli" in rows.index:
        row = rows.loc["routellm_mf_eval_cli"]
        status = str(row.get("status", ""))
        if status == "smoke_executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- RouteLLM-MF upstream evaluation CLI has a successful cache-backed smoke execution "
                f"recorded at `{evidence}`. This validates the exact evaluator command path without embedding API calls; RouteCode utility rows remain tracked separately."
            )
        elif bool(row.get("runnable_now", False)):
            lines.append(
                "- RouteLLM-MF upstream evaluation CLI is runnable with local cache-backed assets, but no smoke execution has been recorded."
            )
        else:
            blockers = str(row.get("blocking_reasons", ""))
            lines.append(f"- RouteLLM-MF upstream evaluation CLI remains blocked: `{blockers}`.")
    if "routecode_local_embedllm_knn_metric" in rows.index:
        row = rows.loc["routecode_local_embedllm_knn_metric"]
        if str(row.get("status", "")) == "available":
            lines.append(
                "- A local metric-bearing EmbedLLM KNN adapter row is available. It is RouteCode-metric-compatible, but it is not an exact upstream command or published checkpoint."
            )
    if "routecode_local_frugalgpt_metric" in rows.index:
        row = rows.loc["routecode_local_frugalgpt_metric"]
        if str(row.get("status", "")) == "available":
            lines.append(
                "- A local metric-bearing FrugalGPT adapter row is available. It is RouteCode-metric-compatible, but it is not an exact upstream command or published checkpoint."
            )
    if "routecode_upstream_avengerspro_metric" in rows.index:
        row = rows.loc["routecode_upstream_avengerspro_metric"]
        if str(row.get("status", "")) == "available":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- An upstream-code Avengers-Pro RouteCode metric row is available "
                f"with captured routing details at `{evidence}`. It uses upstream `SimpleClusterRouter` model code, "
                "but it is not an exact upstream command output because the exact CLI JSON omits per-query routing details."
            )
    llmrouter_smoke_rows = [
        rows.loc[check_id]
        for check_id in ["llmrouter_knn_train_cli", "llmrouter_svm_train_cli"]
        if check_id in rows.index and str(rows.loc[check_id].get("status", "")) == "smoke_executed"
    ]
    if llmrouter_smoke_rows:
        evidence = ", ".join(f"`{str(row.get('execution_evidence', ''))}`" for row in llmrouter_smoke_rows)
        lines.append(
            "- LLMRouter KNN/SVM training CLIs have successful bounded smoke executions "
            f"recorded at {evidence}. These validate exact upstream training command paths on split-aligned RouteCode assets; metric-bearing RouteCode adapter rows remain tracked separately."
        )
    llmrouter_infer_rows = [
        rows.loc[check_id]
        for check_id in ["llmrouter_knn_infer_cli", "llmrouter_svm_infer_cli"]
        if check_id in rows.index and str(rows.loc[check_id].get("status", "")) in {"executed", "smoke_executed"}
    ]
    if llmrouter_infer_rows:
        evidence = ", ".join(f"`{str(row.get('execution_evidence', ''))}`" for row in llmrouter_infer_rows)
        lines.append(
            "- LLMRouter KNN/SVM route-only inference CLIs have successful no-API executions "
            f"recorded at {evidence}. Full-split outputs are used when available; otherwise bounded smoke outputs are reported. These use the precomputed RouteCode embedding cache to avoid Longformer downloads and external API calls; metric-bearing rows are tracked separately."
        )
    if "frugalgpt_local_scorer_cli" in rows.index:
        row = rows.loc["frugalgpt_local_scorer_cli"]
        status = str(row.get("status", ""))
        if status == "smoke_executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- FrugalGPT local scorer has a successful bounded smoke execution "
                f"recorded at `{evidence}`. This is runtime evidence for the command path; metric-bearing RouteCode adapter rows are tracked separately when `routecode_local_frugalgpt_metric` is available."
            )
        elif bool(row.get("runnable_now", False)):
            if "routecode_local_frugalgpt_metric" in rows.index and str(
                rows.loc["routecode_local_frugalgpt_metric"].get("status", "")
            ) == "available":
                lines.append(
                    "- FrugalGPT exact CLI is runnable with local split-aligned assets and a local encoder checkpoint; metric-bearing RouteCode adapter rows are tracked separately."
                )
            else:
                lines.append(
                    "- FrugalGPT local scorer is now runnable with local split-aligned assets and a local encoder checkpoint, but no full trained baseline metric row has been recorded."
                )
        else:
            blockers = str(row.get("blocking_reasons", ""))
            lines.append(f"- FrugalGPT local scorer remains blocked: `{blockers}`.")

    if "embedllm_knn_cli" in rows.index:
        row = rows.loc["embedllm_knn_cli"]
        status = str(row.get("status", ""))
        if status == "executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- EmbedLLM KNN has successful full-split tensor executions "
                f"recorded at `{evidence}`. These validate the exact upstream command path on split-aligned tensor assets and report upstream correctness metrics, not RouteCode routing utility; RouteCode utility rows remain tracked separately when `routecode_local_embedllm_knn_metric` is available."
            )
        elif status == "smoke_executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- EmbedLLM KNN has a successful bounded smoke execution "
                f"recorded at `{evidence}`. This validates the patched local command path; metric-bearing RouteCode adapter rows are tracked separately when `routecode_local_embedllm_knn_metric` is available."
            )
        elif bool(row.get("runnable_now", False)):
            if "routecode_local_embedllm_knn_metric" in rows.index and str(
                rows.loc["routecode_local_embedllm_knn_metric"].get("status", "")
            ) == "available":
                lines.append(
                    "- EmbedLLM KNN exact CLI is runnable with split-aligned CSV assets and local `sentence_transformers`; metric-bearing RouteCode adapter rows are tracked separately."
                )
            else:
                lines.append(
                    "- EmbedLLM KNN is now runnable with split-aligned CSV assets and local `sentence_transformers`, but no full split metric row has been recorded."
                )
        else:
            blockers = str(row.get("blocking_reasons", ""))
            lines.append(f"- EmbedLLM KNN remains blocked: `{blockers}`.")

    if "embedllm_mf_cli" in rows.index:
        row = rows.loc["embedllm_mf_cli"]
        status = str(row.get("status", ""))
        if status == "executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- EmbedLLM MF has successful full-split upstream router-mode execution "
                f"recorded at `{evidence}`. This validates the exact upstream command path on split-aligned CSV assets and reports upstream router accuracy, not RouteCode routing utility."
            )
        elif status == "smoke_executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- EmbedLLM MF has a successful bounded smoke execution "
                f"recorded at `{evidence}`. This validates the patched local command path with split-aligned CSV assets and upstream-compatible question embeddings; it is not a RouteCode metric row."
            )
        elif bool(row.get("runnable_now", False)):
            lines.append(
                "- EmbedLLM MF is now runnable with split-aligned CSV assets and upstream-compatible question embeddings, but no smoke execution or metric-bearing row has been recorded."
            )
        else:
            blockers = str(row.get("blocking_reasons", ""))
            lines.append(f"- EmbedLLM MF remains blocked: `{blockers}`.")

    if "avengerspro_cli" in rows.index:
        row = rows.loc["avengerspro_cli"]
        status = str(row.get("status", ""))
        if status == "executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- Avengers-Pro simple cluster router has successful full-split exact upstream execution "
                f"recorded at `{evidence}`. This validates the cache-backed command path without embedding API calls and reports upstream accuracy and cost, not RouteCode routing utility; split-aligned RouteCode utility rows remain tracked separately in the local compatibility table."
            )
        elif status == "smoke_executed":
            evidence = str(row.get("execution_evidence", ""))
            lines.append(
                "- Avengers-Pro simple cluster router has a successful bounded cache-backed upstream smoke execution "
                f"recorded at `{evidence}`. This validates the exact command path without embedding API calls; split-aligned RouteCode metric rows are tracked separately in the local compatibility table."
            )
        elif bool(row.get("runnable_now", False)):
            lines.append(
                "- Avengers-Pro simple cluster router is runnable with cache-backed split-aligned smoke assets, but no smoke execution has been recorded."
            )
        else:
            blockers = str(row.get("blocking_reasons", ""))
            lines.append(f"- Avengers-Pro remains blocked: `{blockers}`.")

    runnable_mask = table.get("runnable_now", pd.Series(False, index=table.index)).map(_as_bool)
    local_metric_mask = (
        table.get("routecode_metric_compatible", pd.Series(False, index=table.index)).map(_as_bool)
        & ~table.get("exact_upstream_command", pd.Series(False, index=table.index)).map(_as_bool)
        if "check_id" in table
        else pd.Series(False, index=table.index)
    )
    blocked = table[(~runnable_mask) & (~local_metric_mask)].copy()
    if not blocked.empty and "check_id" in blocked:
        blocked_ids = ", ".join(f"`{item}`" for item in blocked["check_id"].astype(str))
        lines.append(f"- Still-blocked command-path rows: {blocked_ids}.")
    lines.append("- This artifact is readiness evidence, not a new routing metric table.")
    return lines


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


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
