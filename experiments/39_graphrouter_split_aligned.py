from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import yaml

from routecode.config import load_config, output_dir
from routecode.eval.graphrouter_assets import build_graphrouter_assets, write_graphrouter_assets
from routecode.eval.graphrouter_split_aligned import (
    build_routecode_split_masks,
    evaluate_graphrouter_selected_models,
)
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


GRAPHROUTER_SOURCE = ROOT / "data/raw/external/LLMRouterBench/baselines/GraphRouter"
RUN_DIRNAME = "graphrouter_split_aligned"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    run(args.config, epochs=args.epochs)


def run(config_path: str, *, epochs: int = 1) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    seed = int(config.get("run", {}).get("random_seed", 0))
    bootstrap = config.get("bootstrap", {})
    baseline_config = config.get("external_baselines", {})
    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)

    assets = build_graphrouter_assets(prepared.matrices, prepared.embeddings, seed=seed)
    written_assets = write_graphrouter_assets(assets, out_dir / "graphrouter_assets")
    graph_config = _split_aligned_config(
        yaml.safe_load(written_assets.config_path.read_text(encoding="utf-8")),
        asset_dir=written_assets.asset_dir,
        run_dir=run_dir,
        epochs=epochs,
        baseline_config=baseline_config,
    )
    config_path_out = run_dir / "config.split_aligned.yaml"
    config_path_out.write_text(yaml.safe_dump(graph_config, sort_keys=False), encoding="utf-8")

    raw_predictions = _run_graphrouter_split_aligned(graph_config)
    raw_path = run_dir / "raw_predictions.json"
    raw_path.write_text(
        json.dumps(raw_predictions.to_dict(orient="records"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected = pd.Series(
        raw_predictions["selected_model"].astype(str).tolist(),
        index=pd.Index(raw_predictions["query_id"].astype(str).tolist(), name=prepared.matrices["test"].utility.index.name),
        name="selected_model",
    )
    row = evaluate_graphrouter_selected_models(
        prepared.matrices["train"],
        prepared.matrices["test"],
        prepared.embeddings,
        selected,
        prediction_source=str(raw_path),
        seed=seed,
        n_bootstrap=int(bootstrap.get("n_bootstrap", 300)),
        ci=float(bootstrap.get("ci", 0.95)),
        knn_k=int(config.get("routers", {}).get("knn_k", 15)),
    )
    row.update(
        {
            "epochs": int(epochs),
            "checkpoint_selection_split": "val" if bool((raw_predictions["checkpoint_selection_split"] == "val").all()) else "train",
            "config_path": str(config_path_out),
            "model_path": str(graph_config["model_path"]),
        }
    )
    table = pd.DataFrame([row])
    table.to_csv(out_dir / "table_graphrouter_split_aligned.csv", index=False)
    _write_run_config(run_dir, config_path, graph_config, raw_predictions)
    write_memo(out_dir, config_path, table, raw_path)
    append_readme(out_dir, config_path, table, raw_path)
    print(f"Wrote split-aligned GraphRouter outputs to {out_dir}")


def _split_aligned_config(
    base_config: dict[str, Any],
    *,
    asset_dir: Path,
    run_dir: Path,
    epochs: int,
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    config = dict(base_config)
    config.update(
        {
            "saved_router_data_path": str((asset_dir / "router_data.csv").resolve()),
            "llm_description_path": str((asset_dir / "LLM_Descriptions.json").resolve()),
            "llm_embedding_path": str((asset_dir / "llm_description_embedding.pkl").resolve()),
            "model_path": str((run_dir / "model_path/best_model.pth").resolve()),
            "train_epoch": int(epochs),
            "wandb_key": "",
            "output_dir": str(run_dir.resolve()),
        }
    )
    config["embedding_dim"] = int(baseline_config.get("graphrouter_embedding_dim", config.get("embedding_dim", 8)))
    config["batch_size"] = int(baseline_config.get("graphrouter_batch_size", config.get("batch_size", 32)))
    Path(config["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    return config


def _run_graphrouter_split_aligned(config: dict[str, Any]) -> pd.DataFrame:
    sys.path.insert(0, str(GRAPHROUTER_SOURCE / "model"))
    sys.path.insert(0, str(GRAPHROUTER_SOURCE))
    from multi_task_graph_router import graph_router_prediction
    from graph_nn import form_data

    class SplitAlignedGraphRouter(graph_router_prediction):
        def split_data(self):  # type: ignore[override]
            masks = build_routecode_split_masks(self.data_df, num_llms=self.num_llms)
            self.routecode_masks = masks
            self.combined_edge = np.concatenate(
                (self.cost_list.reshape(-1, 1), self.effect_list.reshape(-1, 1)),
                axis=1,
            )
            self.scenario = self.config["scenario"]
            if self.scenario == "Performance First":
                self.effect_list = 1.0 * self.effect_list - 0.0 * self.cost_list
            elif self.scenario == "Balance":
                self.effect_list = 0.5 * self.effect_list - 0.5 * self.cost_list
            else:
                self.effect_list = 0.2 * self.effect_list - 0.8 * self.cost_list

            utility_matrix = self.effect_list.reshape(-1, self.num_llms)
            tie_atol = self.config.get("tie_atol", 1e-8)
            row_max = np.max(utility_matrix, axis=1, keepdims=True)
            tie_mask = np.isclose(utility_matrix, row_max, atol=tie_atol)
            self.label = tie_mask.astype(np.float32).reshape(-1, 1)
            self.edge_org_id = [num for num in range(self.num_query) for _ in range(self.num_llms)]
            self.edge_des_id = list(range(self.edge_org_id[0], self.edge_org_id[0] + self.num_llms)) * self.num_query

            edge_count = len(self.edge_org_id)
            self.mask_train = torch.zeros(edge_count)
            self.mask_validate = torch.zeros(edge_count)
            self.mask_test = torch.zeros(edge_count)
            self.mask_train[masks.train_row_indices] = 1
            self.mask_validate[masks.val_row_indices] = 1
            self.mask_test[masks.test_row_indices] = 1

        def train_GNN(self):  # type: ignore[override]
            self.data_for_GNN_train = self.form_data.formulation(
                task_id=self.task_embedding_list,
                query_feature=self.query_embedding_list,
                llm_feature=self.llm_description_embedding,
                org_node=self.edge_org_id,
                des_node=self.edge_des_id,
                edge_feature=self.effect_list,
                edge_mask=self.mask_train,
                label=self.label,
                combined_edge=self.combined_edge,
                train_mask=self.mask_train,
                valide_mask=self.mask_validate,
                test_mask=self.mask_test,
                cost_usd=self.cost_usd_list,
            )
            self.data_for_GNN_validate = self.form_data.formulation(
                task_id=self.task_embedding_list,
                query_feature=self.query_embedding_list,
                llm_feature=self.llm_description_embedding,
                org_node=self.edge_org_id,
                des_node=self.edge_des_id,
                edge_feature=self.effect_list,
                edge_mask=self.mask_validate,
                label=self.label,
                combined_edge=self.combined_edge,
                train_mask=self.mask_train,
                valide_mask=self.mask_validate,
                test_mask=self.mask_test,
                cost_usd=self.cost_usd_list,
            )
            self.data_for_test = self.form_data.formulation(
                task_id=self.task_embedding_list,
                query_feature=self.query_embedding_list,
                llm_feature=self.llm_description_embedding,
                org_node=self.edge_org_id,
                des_node=self.edge_des_id,
                edge_feature=self.effect_list,
                edge_mask=self.mask_test,
                label=self.label,
                combined_edge=self.combined_edge,
                train_mask=self.mask_train,
                valide_mask=self.mask_validate,
                test_mask=self.mask_test,
                cost_usd=self.cost_usd_list,
            )
            selection_data = (
                self.data_for_GNN_validate if int(self.mask_validate.sum().item()) > 0 else self.data_for_GNN_train
            )
            self.checkpoint_selection_split = "val" if selection_data is self.data_for_GNN_validate else "train"
            self.GNN_predict.train_validate(
                data=self.data_for_GNN_train,
                data_validate=self.data_for_GNN_validate,
                data_for_test=selection_data,
                query_task_ids=self.query_task_ids,
            )

    previous_wandb_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"
    try:
        import wandb

        run = wandb.init(project="graph_router", mode="offline", dir=config["output_dir"], reinit=True)
        router = SplitAlignedGraphRouter(
            router_data_path=config["saved_router_data_path"],
            llm_path=config["llm_description_path"],
            llm_embedding_path=config["llm_embedding_path"],
            config=config,
            wandb=wandb,
        )
        if run is not None:
            run.finish()
    finally:
        if previous_wandb_mode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = previous_wandb_mode

    model_path = Path(config["model_path"])
    if model_path.exists():
        device = next(router.GNN_predict.model.parameters()).device
        router.GNN_predict.model.load_state_dict(torch.load(model_path, map_location=device))
    return _extract_test_predictions(router)


def _extract_test_predictions(router: Any) -> pd.DataFrame:
    data = router.data_for_test
    predictor = router.GNN_predict
    predictor.model.eval()
    mask = torch.tensor(data.edge_mask, dtype=torch.bool)
    edge_can_see = predictor.train_mask
    with torch.no_grad():
        edge_scores = predictor.model(
            task_id=data.task_id,
            query_features=data.query_features,
            llm_features=data.llm_features,
            edge_index=data.edge_index,
            edge_mask=mask,
            edge_can_see=edge_can_see,
            edge_weight=data.combined_edge,
        )
    score_matrix = edge_scores.reshape(-1, router.config["llm_num"]).detach().cpu().numpy()
    selected_idx = score_matrix.argmax(axis=1)
    max_score = score_matrix.max(axis=1)
    masks = router.routecode_masks
    return pd.DataFrame(
        {
            "query_id": masks.test_query_ids,
            "task_id": [router.query_task_ids[position] for position in masks.test_query_positions],
            "selected_model": [router.llm_names[int(index)] for index in selected_idx],
            "selected_model_index": selected_idx.astype(int),
            "selected_score": max_score.astype(float),
            "checkpoint_selection_split": router.checkpoint_selection_split,
        }
    )


def _write_run_config(
    run_dir: Path,
    config_path: str,
    graph_config: dict[str, Any],
    raw_predictions: pd.DataFrame,
) -> None:
    payload = {
        "config_path": config_path,
        "graphrouter_config": graph_config,
        "prediction_count": int(len(raw_predictions)),
        "checkpoint_selection_split": (
            str(raw_predictions["checkpoint_selection_split"].iloc[0]) if len(raw_predictions) else ""
        ),
        "source_repo": "https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/GraphRouter",
        "split_aligned_with_routecode": True,
        "routecode_metric_compatible": True,
        "exact_upstream_command": False,
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, raw_path: Path) -> None:
    row = table.iloc[0]
    lines = [
        "# Phase E GraphRouter Split-Aligned Memo",
        "",
        f"Command: `python experiments/39_graphrouter_split_aligned.py --config {config_path}`",
        "",
        "This run uses the upstream GraphRouter GNN/model code with RouteCode train/validation/test masks. "
        "The checkpoint is selected on the RouteCode validation split when available, then predictions are scored "
        "with RouteCode test-split utility. It is not an exact upstream command because the unmodified command "
        "does not consume arbitrary RouteCode split masks or emit RouteCode utility metrics.",
        "",
        "Outputs:",
        "",
        "- `table_graphrouter_split_aligned.csv`",
        f"- `{raw_path}`",
        f"- `{RUN_DIRNAME}/config.split_aligned.yaml`",
        f"- `{RUN_DIRNAME}/model_path/best_model.pth`",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
                    "checkpoint_selection_split",
                    "routecode_metric_compatible",
                ]
            ]
        ),
        "",
        f"Selected models: `{row['selected_models']}`.",
    ]
    (out_dir / "phase_e_graphrouter_split_aligned_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame, raw_path: Path) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## GraphRouter Split-Aligned Metrics"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/39_graphrouter_split_aligned.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_graphrouter_split_aligned.csv`: RouteCode utility metrics over split-aligned GraphRouter GNN selections.",
        "- `phase_e_graphrouter_split_aligned_memo.md`: compatibility and leakage notes.",
        f"- `{raw_path}`: selected-model predictions for RouteCode test queries.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
                    "checkpoint_selection_split",
                ]
            ]
        ),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    columns = list(table.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
