from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir as config_output_dir
from routecode.local_eval.server_readiness import inspect_local_server_readiness
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    run(config_path=args.config, output_dir=args.output_dir or None)


def run(
    *,
    config_path: str,
    output_dir: str | None = None,
    client: Any | None = None,
    clients_by_base_url: dict[str, Any] | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    out_dir = Path(output_dir) if output_dir else config_output_dir(config)
    readiness_config = _readiness_config(config)
    endpoint_specs = _readiness_endpoint_specs(readiness_config)
    if endpoint_specs:
        frames = []
        for spec in endpoint_specs:
            base_url = str(spec.get("base_url", "")).rstrip("/")
            if not base_url:
                raise ValueError("Each openai_endpoints readiness spec requires base_url")
            endpoint_client = (clients_by_base_url or {}).get(base_url)
            frames.append(_inspect_spec(spec, client=endpoint_client))
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        model_ids = [str(model_id) for model_id in readiness_config.get("model_ids", ["Qwen3-8B"])]
        if client is not None:
            model_ids = _resolve_readiness_model_ids(model_ids, client)
        table = inspect_local_server_readiness(
            base_url=str(readiness_config.get("base_url", "http://localhost:8000/v1")),
            api_key=str(readiness_config.get("api_key", "local-routecode")),
            model_ids=model_ids,
            generation_params=dict(readiness_config.get("generation_params", {"temperature": 0.0, "max_tokens": 8})),
            timeout_sec=float(readiness_config.get("timeout_sec", 5.0)),
            client=client,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "table_local_server_readiness.csv"
    table.to_csv(table_path, index=False)
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote Phase 2 local server readiness table to {table_path}")
    return table


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    lines = [
        "# Phase 2 Local Server Readiness",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/58_local_server_readiness.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        _status_sentence(table),
        "",
        "Outputs:",
        "",
        "- `table_local_server_readiness.csv`: per-model local OpenAI-compatible endpoint readiness.",
        "- `m9_local_server_readiness_memo.md`: this memo.",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "m9_local_server_readiness_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Local Server Readiness"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/58_local_server_readiness.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        _status_sentence(table),
        "",
        "Outputs:",
        "",
        "- `table_local_server_readiness.csv`",
        "- `m9_local_server_readiness_memo.md`",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _readiness_config(config: dict[str, Any]) -> dict[str, Any]:
    readiness = dict(config.get("phase2_local_server_readiness", {}))
    local_eval = config.get("phase2_local_eval", {})
    if "openai_endpoints" not in readiness and "openai_endpoints" in local_eval:
        readiness["openai_endpoints"] = local_eval["openai_endpoints"]
    if "model_ids" not in readiness and "model_ids" in local_eval:
        readiness["model_ids"] = local_eval["model_ids"]
    for key in ["base_url", "api_key", "timeout_sec", "generation_params"]:
        if key not in readiness and key in local_eval:
            readiness[key] = local_eval[key]
    return readiness


def _readiness_endpoint_specs(readiness_config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = readiness_config.get("openai_endpoints", [])
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("openai_endpoints must be a list")
    defaults = {
        "api_key": readiness_config.get("api_key", "local-routecode"),
        "timeout_sec": readiness_config.get("timeout_sec", 5.0),
        "generation_params": readiness_config.get("generation_params", {"temperature": 0.0, "max_tokens": 8}),
    }
    merged = []
    for spec in specs:
        item = dict(defaults)
        item.update(dict(spec))
        item["generation_params"] = {
            **dict(defaults["generation_params"]),
            **dict(spec.get("generation_params", {})),
        }
        if "model_ids" not in item:
            item["model_ids"] = ["__first_listed__"]
        merged.append(item)
    return merged


def _inspect_spec(spec: dict[str, Any], *, client: Any | None = None) -> pd.DataFrame:
    return inspect_local_server_readiness(
        base_url=str(spec.get("base_url", "http://localhost:8000/v1")),
        api_key=str(spec.get("api_key", "local-routecode")),
        model_ids=[str(model_id) for model_id in spec.get("model_ids", ["__first_listed__"])],
        generation_params=dict(spec.get("generation_params", {"temperature": 0.0, "max_tokens": 8})),
        timeout_sec=float(spec.get("timeout_sec", 5.0)),
        client=client,
    )


def _resolve_readiness_model_ids(model_ids: list[str], client: Any) -> list[str]:
    if "__first_listed__" not in model_ids:
        return model_ids
    listed = list(client.list_models())
    if not listed:
        raise ValueError("model_ids includes __first_listed__, but the local server returned no models")
    return [listed[0] if model_id == "__first_listed__" else model_id for model_id in model_ids]


def _status_sentence(table: pd.DataFrame) -> str:
    if table.empty:
        return "No local model IDs were configured for readiness checking."
    counts = table["status"].astype(str).value_counts().to_dict()
    if counts.get("blocked", 0):
        return (
            "At least one configured local model is blocked. This means true local Phase 2 runs should not be "
            "started until the local OpenAI-compatible endpoint is available."
        )
    if counts.get("warning", 0):
        return "The endpoint can generate, but at least one configured model had a readiness warning."
    return "All configured local models passed the OpenAI-compatible server readiness check."


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
