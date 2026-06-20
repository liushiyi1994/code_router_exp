from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from routecode.eval.routecode_selection_gate import RouteCodeSelectionConfig, evaluate_routecode_selection_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llmrouterbench_pilot.yaml")
    parser.add_argument(
        "--query-model-utility",
        default="results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv",
    )
    parser.add_argument("--output-dir", default="results/phase2/routecode_exact_math_selection")
    parser.add_argument("--k-values", default="4,8,16,32,64,128")
    parser.add_argument("--alpha-values", default="0.0,0.05,0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--training-datasets", default="")
    parser.add_argument("--validation-datasets", default="aime,math500")
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--target-k", type=int, default=32)
    args = parser.parse_args()
    paths = run(
        config_path=args.config,
        query_model_utility_path=args.query_model_utility,
        output_dir=args.output_dir,
        k_values=_parse_ints(args.k_values),
        alpha_values=_parse_floats(args.alpha_values),
        training_datasets=_parse_strings(args.training_datasets) or None,
        validation_datasets=tuple(item.strip() for item in args.validation_datasets.split(",") if item.strip()),
        threshold=args.threshold,
        target_k=args.target_k,
    )
    print(f"Wrote RouteCode exact-math selection table to {paths['table']}")


def run(
    *,
    config_path: str,
    query_model_utility_path: str,
    output_dir: str,
    k_values: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    alpha_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0),
    training_datasets: tuple[str, ...] | None = None,
    validation_datasets: tuple[str, ...] = ("aime", "math500"),
    threshold: float = 0.03,
    target_k: int | None = 32,
) -> dict[str, str]:
    return evaluate_routecode_selection_gate(
        config_path=config_path,
        query_model_utility_path=query_model_utility_path,
        output_dir=output_dir,
        selection=RouteCodeSelectionConfig(
            k_values=k_values,
            alpha_values=alpha_values,
            training_datasets=training_datasets,
            validation_datasets=validation_datasets,
            threshold=threshold,
            target_k=target_k,
        ),
    )


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
