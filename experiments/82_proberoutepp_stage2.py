from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from routecode.proberoutepp import (
    ProbeRoutePPConfig,
    build_proberoutepp_artifacts,
    prepare_scored_outputs,
    write_proberoutepp_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProbeRoute++ Stage 2 state-mediated routing evaluation.")
    parser.add_argument("--scored-outputs", default="results/controlled/scored_outputs.parquet")
    parser.add_argument("--model-outputs", default="results/controlled/model_outputs.parquet")
    parser.add_argument("--config", default="configs/proberoute_controlled.yaml")
    parser.add_argument("--output-dir", default="results/controlled/proberoutepp_stage2")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lambda-cost", type=float, default=None)
    parser.add_argument("--lambda-latency", type=float, default=None)
    parser.add_argument("--probe-knn-k", type=int, default=5)
    parser.add_argument("--probe-blend-weight", type=float, default=0.65)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--voi-min-gain", type=float, default=0.0)
    parser.add_argument("--probe-cost-utility", type=float, default=0.0)
    parser.add_argument("--probe-latency-s", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = _load_yaml(Path(args.config))
    routing = base_config.get("routing", {}) if isinstance(base_config, dict) else {}
    config = ProbeRoutePPConfig(
        k=int(args.k if args.k is not None else routing.get("stage2_k", routing.get("default_k", 16))),
        alpha=float(args.alpha if args.alpha is not None else routing.get("stage2_alpha", 0.5)),
        beta=float(args.beta),
        lambda_cost=float(args.lambda_cost if args.lambda_cost is not None else routing.get("lambda_cost", 0.35)),
        lambda_latency=float(args.lambda_latency if args.lambda_latency is not None else routing.get("lambda_latency", 0.05)),
        probe_knn_k=int(args.probe_knn_k),
        probe_blend_weight=float(args.probe_blend_weight),
        entropy_threshold=args.entropy_threshold,
        margin_threshold=float(args.margin_threshold),
        voi_min_gain=float(args.voi_min_gain),
        probe_cost_utility=float(args.probe_cost_utility),
        probe_latency_s=float(args.probe_latency_s),
    )
    scored_path = Path(args.scored_outputs)
    model_path = Path(args.model_outputs)
    if scored_path.exists():
        scored = pd.read_parquet(scored_path)
    elif model_path.exists():
        scored = prepare_scored_outputs(pd.read_parquet(model_path), config)
    else:
        raise FileNotFoundError(f"Missing scored outputs {scored_path} and model outputs {model_path}")
    artifacts = build_proberoutepp_artifacts(scored, config)
    paths = write_proberoutepp_outputs(artifacts, Path(args.output_dir))
    print(f"Wrote ProbeRoute++ Stage 2 outputs to {Path(args.output_dir)}")
    print(f"Main eval: {paths['table_main_eval']}")
    print(f"Routing decisions: {paths['routing_decisions']}")
    print(f"Report: {paths['run_report']}")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    main()
