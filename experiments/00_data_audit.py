from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.metrics import dominance_ratio, model_win_entropy, selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_model_win_distribution, save_oracle_gap_by_dataset
from routecode.routers.dataset_lookup import DatasetLabelRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter, CheapestRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]

    source = config.get("data", {}).get("source", "synthetic")
    prepared.outcomes.to_csv(out_dir / "outcomes.csv", index=False)
    prepared.embeddings.to_csv(out_dir / "query_embeddings.csv")
    if source == "synthetic":
        prepared.outcomes.to_csv(out_dir / "synthetic_outcomes.csv", index=False)
        prepared.embeddings.to_csv(out_dir / "synthetic_query_embeddings.csv")

    oracle_selected = OracleRouter().predict(test.utility)
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    cheapest = CheapestRouter().fit(train.query_info, train.cost).predict(test.query_info)
    dataset_selected = DatasetLabelRouter("dataset").fit(train.query_info, train.utility).predict(test.query_info)

    baseline_mean = selected_values(test.utility, best_single).mean()
    oracle_mean = test.utility.max(axis=1).mean()
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    seed = int(config.get("run", {}).get("random_seed", 0))

    rows = []
    for method, selected, labels in [
        ("cheapest", cheapest, None),
        ("best_single", best_single, None),
        ("dataset_label_lookup", dataset_selected, test.query_info["dataset"]),
        ("query_oracle", oracle_selected, oracle_selected),
    ]:
        rows.append(
            evaluate_selection(
                method=method,
                selected_models=selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=oracle_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
                k=int(labels.nunique()) if isinstance(labels, pd.Series) else None,
                labels=labels if isinstance(labels, pd.Series) else None,
            )
        )

    table = pd.DataFrame(rows)
    table["oracle_model_win_entropy"] = model_win_entropy(oracle_selected.astype(str).tolist())
    table["oracle_model_dominance_ratio"] = dominance_ratio(oracle_selected.astype(str).tolist())
    table.to_csv(out_dir / "table_routability.csv", index=False)

    gap_rows = []
    best_util = selected_values(test.utility, best_single)
    oracle_util = test.utility.max(axis=1)
    for dataset, query_ids in test.query_info.groupby("dataset").groups.items():
        query_ids = list(query_ids)
        gap_rows.append(
            {
                "dataset": dataset,
                "n_queries": len(query_ids),
                "best_single_utility": float(best_util.loc[query_ids].mean()),
                "oracle_utility": float(oracle_util.loc[query_ids].mean()),
                "oracle_gap": float((oracle_util.loc[query_ids] - best_util.loc[query_ids]).mean()),
            }
        )
    gap_table = pd.DataFrame(gap_rows).sort_values("oracle_gap", ascending=False)
    gap_table.to_csv(out_dir / "table_oracle_gap_by_dataset.csv", index=False)

    save_model_win_distribution(oracle_selected, out_dir / "fig_model_win_distribution.pdf")
    save_oracle_gap_by_dataset(gap_table, out_dir / "fig_oracle_gap_by_dataset.pdf")
    print(f"Wrote data audit outputs to {out_dir}")


if __name__ == "__main__":
    main()
