from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the frozen Phase 3 ProbeCode-StateCal method.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "final_method"
    out_dir.mkdir(parents=True, exist_ok=True)

    method_cfg = config["method"]
    current_eval = pd.read_csv(config["inputs"]["broad100_current_best_eval"])
    action_mix = pd.read_csv(config["inputs"]["broad100_current_best_action_mix"])
    final_claims = pd.read_csv(config["inputs"]["final_claims"])
    state_cards = build_state_cards(config, method_cfg)

    state_cards.to_csv(out_dir / "table_final_state_cards.csv", index=False)
    write_code_cards(out_dir / "code_cards.md", state_cards)
    write_method_card(out_dir / "METHOD_CARD.md", config, current_eval, action_mix, final_claims, state_cards)
    write_manifest(out_dir / "manifest.json", config, current_eval, state_cards)
    print(f"Wrote final method package to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def build_state_cards(config: dict[str, Any], method_cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(config["inputs"]["learned_verifiability_code_cards"])
    cards = pd.read_csv(path)
    compact_method = str(method_cfg["compact_state_method"])
    selected = cards[cards["method"].astype(str).eq(compact_method)].copy()
    if selected.empty:
        selected = cards[cards["method"].astype(str).str.contains("state_k8", regex=False)].copy()
    if selected.empty:
        selected = cards.head(16).copy()
    keep_cols = [
        col
        for col in [
            "method",
            "family",
            "classifier",
            "threshold",
            "k",
            "probe_state",
            "n_all",
            "n_train",
            "chosen_policy",
            "train_need_large_rate",
            "train_true_tool_available_rate",
            "train_pred_tool_available_rate",
            "train_mean_local_utility",
            "train_mean_large_utility",
            "top_feature_diffs_json",
            "benchmark_mix_json",
        ]
        if col in selected.columns
    ]
    out = selected[keep_cols].copy()
    out.insert(0, "final_method_name", str(method_cfg["name"]))
    out["source_table"] = str(path)
    return out.sort_values(["k", "probe_state"], kind="stable").reset_index(drop=True)


def write_code_cards(path: Path, cards: pd.DataFrame) -> None:
    lines = ["# ProbeCode-StateCal Code Cards", ""]
    for row in cards.to_dict("records"):
        state = row.get("probe_state", "")
        lines.append(f"## State {state}")
        lines.append("")
        for key in [
            "chosen_policy",
            "n_train",
            "train_need_large_rate",
            "train_true_tool_available_rate",
            "train_pred_tool_available_rate",
            "train_mean_local_utility",
            "train_mean_large_utility",
        ]:
            if key in row:
                lines.append(f"- `{key}`: `{row[key]}`")
        if "benchmark_mix_json" in row:
            lines.append(f"- `benchmark_mix_json`: `{row['benchmark_mix_json']}`")
        if "top_feature_diffs_json" in row:
            lines.append(f"- `top_feature_diffs_json`: `{row['top_feature_diffs_json']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_method_card(
    path: Path,
    config: dict[str, Any],
    current_eval: pd.DataFrame,
    action_mix: pd.DataFrame,
    final_claims: pd.DataFrame,
    state_cards: pd.DataFrame,
) -> None:
    method_cfg = config["method"]
    current = current_eval[
        current_eval["method"].astype(str).eq(str(method_cfg["current_best_method"]))
        & current_eval["split"].astype(str).eq("test")
    ].copy()
    if current.empty:
        current = current_eval[current_eval["package_role"].astype(str).eq("current_best_validation_selected")].copy()
    row = current.iloc[0] if not current.empty else pd.Series(dtype=object)
    mix = action_mix[action_mix["method"].astype(str).eq(str(method_cfg["current_best_method"]))].copy()
    claim_rows = final_claims.to_dict("records")

    lines = [
        "# ProbeCode-StateCal Method Card",
        "",
        "## Method",
        "",
        f"- Name: `{method_cfg['name']}`",
        f"- Current best policy: `{method_cfg['current_best_method']}`",
        f"- Compact state policy: `{method_cfg['compact_state_method']}`",
        f"- Base policy: `{method_cfg['base_method']}`",
        f"- Lambda cost: `{method_cfg['lambda_cost']}`",
        f"- Verifiable actions allowed: `{method_cfg['allow_verifiable_actions']}`",
        f"- Caveat: {method_cfg['caveat']}",
        "",
        "## Data Flow",
        "",
        "```text",
        "query -> cheap local/verifiable behavior -> probe/route state -> state-to-action utility table -> action",
        "```",
        "",
        "## Current Broad100 Test Evidence",
        "",
    ]
    if not row.empty:
        lines.extend(
            [
                f"- Mean quality: `{float(row['mean_quality']):.6f}`",
                f"- Mean utility: `{float(row['mean_utility']):.6f}`",
                f"- Quality gap to oracle: `{float(row['quality_gap_to_full_oracle']):.6f}`",
                f"- Oracle utility ratio: `{float(row['oracle_utility_ratio']):.6f}`",
                f"- Frontier-call rate: `{float(row['frontier_call_rate']):.6f}`",
                f"- N test queries: `{int(row['n_queries'])}`",
            ]
        )
    lines.extend(["", "## Action Mix", ""])
    if not mix.empty:
        for item in mix.sort_values("n_queries", ascending=False).to_dict("records"):
            lines.append(f"- `{item['selected_action']}`: `{int(item['n_queries'])}` queries")
    lines.extend(["", "## State Cards", "", f"- State-card rows: `{len(state_cards)}`"])
    lines.extend(["", "## Current Claim Ledger", ""])
    for claim in claim_rows:
        lines.append(f"- `{claim.get('claim_id')}`: `{claim.get('status')}`; {claim.get('evidence')}")
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
            f"- Current best eval: `{config['inputs']['broad100_current_best_eval']}`",
            f"- Current best choices: `{config['inputs']['broad100_current_best_choices']}`",
            f"- State cards source: `{config['inputs']['learned_verifiability_code_cards']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, config: dict[str, Any], current_eval: pd.DataFrame, state_cards: pd.DataFrame) -> None:
    manifest = {
        "method_name": config["method"]["name"],
        "current_best_method": config["method"]["current_best_method"],
        "compact_state_method": config["method"]["compact_state_method"],
        "lambda_cost": config["method"]["lambda_cost"],
        "n_current_eval_rows": int(len(current_eval)),
        "n_state_card_rows": int(len(state_cards)),
        "inputs": config["inputs"],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

