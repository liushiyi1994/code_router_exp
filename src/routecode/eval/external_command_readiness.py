from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
from typing import Mapping

import pandas as pd
import yaml


def inspect_external_command_readiness(
    project_root: str | Path,
    *,
    result_dir: str | Path | None = None,
    module_availability: Mapping[str, bool] | None = None,
    env: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Inspect exact upstream-command readiness without importing external repos.

    The goal is to make official-baseline blockers reproducible. This function
    checks files, Python modules, and environment-sensitive embedding configs
    only; it does not run upstream code, download checkpoints, or call APIs.
    """

    root = Path(project_root)
    results = Path(result_dir) if result_dir is not None else root / "results/llmrouterbench_pilot"
    modules = dict(module_availability or {})
    environment = dict(os.environ if env is None else env)
    rows = [
        _routecode_local_mf_metric(root, results),
        _routecode_local_embedllm_knn_metric(root, results),
        _routecode_local_frugalgpt_metric(root, results),
        _routecode_upstream_avengerspro_metric(root, results),
        _llmrouter_train_cli(root, results, modules, "knn"),
        _llmrouter_train_cli(root, results, modules, "svm"),
        _llmrouter_infer_cli(root, results, modules, "knn"),
        _llmrouter_infer_cli(root, results, modules, "svm"),
        _routellm_mf_train_cli(root, results, modules),
        _routellm_mf_eval_cli(root, results, modules, environment),
        _routellm_bert_cli(root, modules),
        _avengerspro_cli(root, results, modules, environment),
        _graphrouter_cli(root, results, modules),
        _frugalgpt_local_scorer_cli(root, results, modules),
        _embedllm_knn_cli(root, results, modules),
        _embedllm_mf_cli(root, results, modules),
        _best_route_train_cli(root, results, modules),
        _routerdc_train_cli(root, results, modules),
        _modelsat_train_cli(root, results, modules),
    ]
    return pd.DataFrame(rows)


def _routecode_local_mf_metric(root: Path, result_dir: Path) -> dict[str, object]:
    metric = result_dir / "table_routellm_mf_split_aligned.csv"
    available = metric.exists() and metric.stat().st_size > 0
    return _row(
        check_id="routecode_local_routellm_mf_metric",
        baseline="RouteLLM-MF local-code metric",
        upstream_source="LLMRouterBench RouteLLM MF source loaded by RouteCode",
        command=f"RouteCode metric table: {metric}",
        required_paths=[metric],
        required_modules=[],
        blocking_reasons=[] if available else ["missing_routecode_metric_table"],
        no_api_compatible=True,
        routecode_metric_compatible=True,
        exact_upstream_command=False,
        status="available" if available else "missing_metric",
        runnable_override=available,
    )


def _routecode_local_embedllm_knn_metric(root: Path, result_dir: Path) -> dict[str, object]:
    metric = result_dir / "table_embedllm_knn_split_aligned.csv"
    available = metric.exists() and metric.stat().st_size > 0
    return _row(
        check_id="routecode_local_embedllm_knn_metric",
        baseline="EmbedLLM KNN local metric adapter",
        upstream_source="RouteCode adapter around LLMRouterBench EmbedLLM KNN correctness routing",
        command=f"RouteCode metric table: {metric}",
        required_paths=[metric],
        required_modules=[],
        blocking_reasons=[] if available else ["missing_embedllm_knn_metric_table"],
        no_api_compatible=True,
        routecode_metric_compatible=True,
        exact_upstream_command=False,
        status="available" if available else "missing_metric",
        runnable_override=available,
    )


def _routecode_local_frugalgpt_metric(root: Path, result_dir: Path) -> dict[str, object]:
    metric = result_dir / "table_frugalgpt_split_aligned.csv"
    available = metric.exists() and metric.stat().st_size > 0
    return _row(
        check_id="routecode_local_frugalgpt_metric",
        baseline="FrugalGPT local scorer metric adapter",
        upstream_source="RouteCode adapter around LLMRouterBench FrugalGPT local scorer source",
        command=f"RouteCode metric table: {metric}",
        required_paths=[metric],
        required_modules=[],
        blocking_reasons=[] if available else ["missing_frugalgpt_metric_table"],
        no_api_compatible=True,
        routecode_metric_compatible=True,
        exact_upstream_command=False,
        status="available" if available else "missing_metric",
        runnable_override=available,
    )


def _routecode_upstream_avengerspro_metric(root: Path, result_dir: Path) -> dict[str, object]:
    metric = result_dir / "table_avengerspro_upstream_metric.csv"
    routing_details = result_dir / "avengerspro_upstream_metric/raw_routing_details.json"
    missing = []
    if not metric.exists() or metric.stat().st_size == 0:
        missing.append("missing_avengerspro_upstream_metric_table")
    if not routing_details.exists() or routing_details.stat().st_size == 0:
        missing.append("missing_avengerspro_upstream_routing_details")
    available = not missing
    return _row(
        check_id="routecode_upstream_avengerspro_metric",
        baseline="Avengers-Pro upstream model-code RouteCode metric",
        upstream_source="LLMRouterBench Avengers-Pro SimpleClusterRouter loaded by RouteCode",
        command=(
            "RouteCode utility table over upstream SimpleClusterRouter routing details: "
            f"{metric}"
        ),
        required_paths=[metric, routing_details],
        required_modules=[],
        blocking_reasons=missing,
        no_api_compatible=True,
        routecode_metric_compatible=True,
        exact_upstream_command=False,
        status="available" if available else "missing_metric",
        runnable_override=available,
        execution_evidence=str(routing_details) if routing_details.exists() else "",
    )


def _llmrouter_train_cli(root: Path, result_dir: Path, modules: Mapping[str, bool], router_short: str) -> dict[str, object]:
    router_name = f"{router_short}router"
    script = root / "data/raw/external/LLMRouter/llmrouter/cli/router_train.py"
    llmrouter_root = root / "data/raw/external/LLMRouter"
    asset_dir = result_dir / "llmrouter_library_adapters"
    config = asset_dir / f"{router_name}_train.yaml"
    checkpoint = asset_dir / f"{router_short}_cli_model.pkl"
    smoke_log = asset_dir / f"llmrouter_{router_short}_train_stdout.log"
    reasons = _missing_paths([script, config])
    reasons.extend(_missing_modules(["torch", "numpy", "pandas", "sklearn", "yaml"], modules))
    smoke_executed = not reasons and checkpoint.exists() and _llmrouter_train_log_succeeded(smoke_log, router_name)
    return _row(
        check_id=f"llmrouter_{router_short}_train_cli",
        baseline=f"LLMRouter {router_name} upstream training CLI",
        upstream_source="data/raw/external/LLMRouter",
        command=(
            f"PYTHONPATH={_command_path(llmrouter_root)} "
            f"python -m llmrouter.cli.router_train --router {router_name} "
            f"--config {_command_path(config)} --device cpu --quiet"
        ),
        required_paths=[script, config, checkpoint],
        required_modules=["torch", "numpy", "pandas", "sklearn", "yaml"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked"),
        execution_evidence=str(smoke_log) if smoke_executed else "",
    )


def _llmrouter_train_log_succeeded(smoke_log: Path, router_name: str) -> bool:
    if not smoke_log.exists():
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    success_markers = [f"Training completed for {router_name}!", "Successfully saved pickle model:"]
    return any(marker in text for marker in success_markers) and "Traceback" not in text and "Error:" not in text


def _llmrouter_infer_cli(root: Path, result_dir: Path, modules: Mapping[str, bool], router_short: str) -> dict[str, object]:
    router_name = f"{router_short}router"
    script = root / "data/raw/external/LLMRouter/llmrouter/cli/router_inference.py"
    llmrouter_root = root / "data/raw/external/LLMRouter"
    asset_dir = result_dir / "llmrouter_library_adapters"
    config = asset_dir / f"{router_name}_train.yaml"
    checkpoint = asset_dir / f"{router_short}_cli_model.pkl"
    smoke_input = asset_dir / "query_inference_smoke.jsonl"
    full_input = asset_dir / "query_inference_test.jsonl"
    full_output = asset_dir / f"llmrouter_{router_short}_full_predictions.json"
    full_log = asset_dir / f"llmrouter_{router_short}_full_infer_stdout.log"
    embedding_lookup = asset_dir / "query_embedding_lookup.pt"
    smoke_log = asset_dir / f"llmrouter_{router_short}_infer_stdout.log"
    reasons = _missing_paths([script, config, checkpoint, smoke_input, embedding_lookup])
    reasons.extend(_missing_modules(["torch", "numpy", "pandas", "sklearn", "yaml"], modules))
    full_executed = not reasons and full_input.exists() and _llmrouter_full_predictions_succeeded(full_output)
    smoke_executed = not reasons and _llmrouter_infer_log_succeeded(smoke_log)
    command_input = full_input if full_executed or full_input.exists() else smoke_input
    command_output = f" --output {_command_path(full_output)}" if full_input.exists() else ""
    return _row(
        check_id=f"llmrouter_{router_short}_infer_cli",
        baseline=f"LLMRouter {router_name} upstream route-only inference CLI",
        upstream_source="data/raw/external/LLMRouter",
        command=(
            f"PYTHONPATH={_command_path(llmrouter_root)} "
            f"python -m llmrouter.cli.router_inference --router {router_name} "
            f"--config {_command_path(config)} "
            f"--input {_command_path(command_input)} --route-only{command_output}"
        ),
        required_paths=[script, config, checkpoint, smoke_input, embedding_lookup],
        required_modules=["torch", "numpy", "pandas", "sklearn", "yaml"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="executed" if full_executed else ("smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked")),
        execution_evidence=str(full_output) if full_executed else (str(smoke_log) if smoke_executed else ""),
    )


def _llmrouter_full_predictions_succeeded(output: Path) -> bool:
    if not output.exists() or output.stat().st_size == 0:
        return False
    try:
        import json

        payload = json.loads(output.read_text(encoding="utf-8"))
    except Exception:
        return False
    rows = payload if isinstance(payload, list) else [payload]
    return bool(rows) and all(isinstance(row, dict) and row.get("success") and row.get("model_name") for row in rows)


def _llmrouter_infer_log_succeeded(smoke_log: Path) -> bool:
    if not smoke_log.exists():
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    return (
        '"success": true' in lowered
        and '"model_name"' in lowered
        and '"success": false' not in lowered
        and "traceback" not in lowered
        and "error loading router" not in lowered
    )


def _routellm_mf_train_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    train_script = (
        root
        / "data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization/train_matrix_factorization.py"
    )
    config = result_dir / "routellm_mf_assets/mf_train_config.local.json"
    checkpoint = result_dir / "routellm_mf_assets/mf_model.pt"
    reasons = _missing_paths([train_script, config])
    reasons.extend(_missing_modules(["loguru", "torch", "numpy", "tqdm"], modules))
    executed = not reasons and checkpoint.exists() and checkpoint.stat().st_size > 0
    return _row(
        check_id="routellm_mf_train_cli",
        baseline="RouteLLM-MF upstream training CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/RouteLLM",
        command=(
            "PYTHONPATH=data/raw/external/LLMRouterBench "
            "python -m baselines.RouteLLM.routers.matrix_factorization.train_matrix_factorization "
            f"--config {config}"
        ),
        required_paths=[train_script, config],
        required_modules=["loguru", "torch", "numpy", "tqdm"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="executed" if executed else ("ready" if not reasons else "blocked"),
        execution_evidence=str(checkpoint) if executed else "",
    )


def _routellm_mf_eval_cli(
    root: Path,
    result_dir: Path,
    modules: Mapping[str, bool],
    env: Mapping[str, str],
) -> dict[str, object]:
    eval_script = root / "data/raw/external/LLMRouterBench/baselines/RouteLLM/evaluate_mf.py"
    llmrouterbench_root = root / "data/raw/external/LLMRouterBench"
    asset_dir = result_dir / "routellm_mf_assets"
    config = asset_dir / "mf_eval_config.local.json"
    pairwise = asset_dir / "pairwise_test.json"
    checkpoint = asset_dir / "mf_model.pt"
    embedding_config = asset_dir / "embedding_config.local.yaml"
    embedding_cache = asset_dir / "embedding_cache.jsonl"
    metadata = asset_dir / "metadata.json"
    output = asset_dir / "mf_eval_smoke_results.json"
    smoke_log = asset_dir / "routellm_mf_eval_stdout.log"
    fallback_embedding_config = root / "data/raw/external/LLMRouterBench/config/embedding_config.yaml"
    reasons = _missing_paths([eval_script, config, pairwise, checkpoint, embedding_config, embedding_cache])
    reasons.extend(_missing_modules(["loguru", "torch", "numpy"], modules))
    cache_ready = config.exists() and embedding_config.exists() and embedding_cache.exists()
    if cache_ready:
        no_api_compatible = True
    else:
        env_requirements = _embedding_env_requirements(fallback_embedding_config)
        missing_env = [name for name in env_requirements if not env.get(name)]
        if missing_env:
            reasons.append("embedding_config_requires_env:" + ",".join(missing_env))
        no_api_compatible = not env_requirements
        if not no_api_compatible:
            reasons.append("requires_embedding_service")
    metadata_payload = _read_json_object(metadata)
    strong_model = str(metadata_payload.get("strong_model", "Qwen3-8B"))
    weak_model = str(metadata_payload.get("weak_model", "Qwen2.5-Coder-7B-Instruct"))
    smoke_executed = not reasons and _routellm_mf_eval_log_succeeded(smoke_log, output)
    return _row(
        check_id="routellm_mf_eval_cli",
        baseline="RouteLLM-MF upstream evaluation CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/RouteLLM",
        command=(
            f"PYTHONPATH={_command_path(llmrouterbench_root)} "
            "python -m baselines.RouteLLM.evaluate_mf "
            f"--config {_command_path(config)} "
            f"--data-dir {_command_path(asset_dir)} "
            f"--strong-model {strong_model} --weak-model {weak_model} --threshold 0.5 "
            f"--output {_command_path(output)}"
        ),
        required_paths=[eval_script, config, pairwise, checkpoint, embedding_config, embedding_cache],
        required_modules=["loguru", "torch", "numpy"],
        blocking_reasons=reasons,
        no_api_compatible=no_api_compatible,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked"),
        execution_evidence=str(smoke_log) if smoke_executed else "",
    )


def _routellm_mf_eval_log_succeeded(smoke_log: Path, output: Path) -> bool:
    if not smoke_log.exists() or not output.exists() or output.stat().st_size == 0:
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    return (
        "evaluation complete" in lowered
        and "saved metrics to" in lowered
        and "traceback" not in lowered
        and "error" not in lowered
    )


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _routellm_bert_cli(root: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    router_source = root / "data/raw/external/routellm/routellm/routers/routers.py"
    checkpoint = root / "data/raw/external/routellm/checkpoints/bert"
    reasons = _missing_paths([router_source])
    if not checkpoint.exists():
        reasons.append("missing_bert_checkpoint")
    reasons.extend(_missing_modules(["transformers", "torch"], modules))
    return _row(
        check_id="routellm_bert_cli",
        baseline="RouteLLM-BERT upstream router",
        upstream_source="data/raw/external/routellm",
        command="RouteLLM BERTRouter checkpoint evaluation; no split-aligned command is pinned yet",
        required_paths=[router_source, checkpoint],
        required_modules=["transformers", "torch"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=False,
        status="blocked",
    )


def _avengerspro_cli(
    root: Path,
    result_dir: Path,
    modules: Mapping[str, bool],
    env: Mapping[str, str],
) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/AvengersPro/simple_cluster_router.py"
    llmrouterbench_root = root / "data/raw/external/LLMRouterBench"
    asset_dir = result_dir / "avengerspro_split_aligned"
    train = asset_dir / "train.jsonl"
    test = asset_dir / "test.jsonl"
    smoke_train = asset_dir / "smoke_train.jsonl"
    smoke_test = asset_dir / "smoke_test.jsonl"
    baseline_scores = asset_dir / "baseline_scores.json"
    embedding_cache = asset_dir / "embedding_cache.jsonl"
    config = asset_dir / "simple_cluster_config.local.json"
    output = asset_dir / "simple_cluster_smoke_results.json"
    smoke_log = asset_dir / "avengerspro_simple_cluster_smoke_stdout.log"
    metric_dir = result_dir / "avengerspro_cli_metrics"
    full_config = metric_dir / "simple_cluster_config.full.json"
    full_embedding_cache = metric_dir / "full_embedding_cache.jsonl"
    full_output = metric_dir / "simple_cluster_full_results.json"
    full_log = metric_dir / "avengerspro_simple_cluster_stdout.log"
    use_full_command = any(path.exists() for path in [full_config, full_embedding_cache, full_output, full_log])
    command_config = full_config if use_full_command else config
    command_output = full_output if use_full_command else output
    cache_ready = (
        full_config.exists() and full_embedding_cache.exists()
        if use_full_command
        else config.exists() and embedding_cache.exists()
    )
    required_runtime_paths = [full_config, full_embedding_cache] if use_full_command else [
        smoke_train,
        smoke_test,
        config,
        embedding_cache,
    ]
    reasons = _missing_paths([script, train, test, baseline_scores, *required_runtime_paths])
    reasons.extend(_missing_modules(["datasets", "joblib", "tiktoken", "yaml", "sklearn", "numpy", "tqdm"], modules))
    if not cache_ready:
        missing_env = [name for name in ["EMBEDDING_API_KEY", "EMBEDDING_BASE_URL"] if not env.get(name)]
        if missing_env:
            reasons.append("embedding_service_env_missing:" + ",".join(missing_env))
        reasons.append("requires_embedding_service")
    full_executed = not reasons and _avengerspro_full_output_succeeded(full_log, full_output)
    smoke_executed = not reasons and _avengerspro_smoke_log_succeeded(smoke_log, output)
    return _row(
        check_id="avengerspro_cli",
        baseline="Avengers-Pro upstream command path",
        upstream_source="data/raw/external/LLMRouterBench/baselines/AvengersPro",
        command=(
            f"PYTHONPATH={_command_path(llmrouterbench_root)} "
            "python -m baselines.AvengersPro.simple_cluster_router "
            f"--config {_command_path(command_config)} --output {_command_path(command_output)}"
        ),
        required_paths=[script, train, test, baseline_scores, *required_runtime_paths],
        required_modules=["datasets", "joblib", "tiktoken", "yaml", "sklearn", "numpy", "tqdm"],
        blocking_reasons=reasons,
        no_api_compatible=cache_ready,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="executed" if full_executed else ("smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked")),
        execution_evidence=str(full_output) if full_executed else (str(smoke_log) if smoke_executed else ""),
    )


def _avengerspro_full_output_succeeded(log: Path, output: Path) -> bool:
    if not _avengerspro_log_succeeded(log) or not output.exists() or output.stat().st_size == 0:
        return False
    try:
        import json

        payload = json.loads(output.read_text(encoding="utf-8"))
    except Exception:
        return False
    results = payload.get("results") if isinstance(payload, dict) else None
    return isinstance(results, dict) and results.get("total_queries", 0) > 0 and "accuracy" in results


def _avengerspro_smoke_log_succeeded(smoke_log: Path, output: Path) -> bool:
    if not _avengerspro_log_succeeded(smoke_log) or not output.exists() or output.stat().st_size == 0:
        return False
    return True


def _avengerspro_log_succeeded(log: Path) -> bool:
    if not log.exists():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    return (
        "routing evaluation completed successfully" in lowered
        and "results saved to" in lowered
        and "traceback" not in lowered
        and "error during execution" not in lowered
    )


def _graphrouter_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    graphrouter_root = root / "data/raw/external/LLMRouterBench/baselines/GraphRouter"
    script = graphrouter_root / "run_exp.py"
    local_asset_dir = result_dir / "graphrouter_assets"
    metric_dir = result_dir / "graphrouter_cli_metrics"
    local_router_data = local_asset_dir / "router_data.csv"
    local_llm_embeddings = local_asset_dir / "llm_description_embedding.pkl"
    local_config = local_asset_dir / "config.local.yaml"
    smoke_config = metric_dir / "config.smoke.yaml"
    smoke_log = metric_dir / "graphrouter_stdout.log"
    smoke_model = metric_dir / "model_path/best_model.pth"
    upstream_router_data = root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/data/router_data.csv"
    upstream_llm_embeddings = (
        root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/configs/llm_description_embedding.pkl"
    )
    upstream_config = root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/configs/config.yaml"
    router_data = local_router_data if local_router_data.exists() else upstream_router_data
    llm_embeddings = local_llm_embeddings if local_llm_embeddings.exists() else upstream_llm_embeddings
    config = smoke_config if smoke_config.exists() else (local_config if local_config.exists() else upstream_config)
    reasons = _missing_paths([script])
    if not router_data.exists():
        reasons.append("missing_graphrouter_router_data")
    if not llm_embeddings.exists():
        reasons.append("missing_graphrouter_llm_description_embeddings")
    required_modules = ["bert_score", "litellm", "torch_geometric", "wandb"]
    reasons.extend(_missing_modules(required_modules, modules))
    executed = not reasons and smoke_config.exists() and smoke_model.exists() and _graphrouter_log_succeeded(smoke_log)
    return _row(
        check_id="graphrouter_cli",
        baseline="GraphRouter upstream command path",
        upstream_source="data/raw/external/LLMRouterBench/baselines/GraphRouter",
        command=(
            f"cd {_command_path(graphrouter_root)} && "
            f"WANDB_MODE=offline python run_exp.py --config_file {_command_path(config)}"
        ),
        required_paths=[script, router_data, llm_embeddings, config],
        required_modules=required_modules,
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="executed" if executed else ("ready" if not reasons else "blocked"),
        execution_evidence=str(smoke_log) if executed else "",
    )


def _graphrouter_log_succeeded(smoke_log: Path) -> bool:
    if not smoke_log.exists() or smoke_log.stat().st_size == 0:
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    return (
        "best test checkpoint metrics" in lowered
        and "dataset-level average accuracy" in lowered
        and "sample-level average accuracy" in lowered
        and "traceback" not in lowered
        and "error" not in lowered
    )


def _frugalgpt_local_scorer_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/FrugalGPT/train_router_from_results.py"
    train = result_dir / "frugalgpt_split_aligned/train.jsonl"
    test = result_dir / "frugalgpt_split_aligned/test.jsonl"
    local_base = result_dir / "external_checkpoints/local_encoder"
    smoke_log = result_dir / "frugalgpt_split_aligned/output/frugalgpt_smoke_stdout.log"
    reasons = _missing_paths([script])
    if not train.exists():
        reasons.append("missing_frugalgpt_split_aligned_train_jsonl")
    if not test.exists():
        reasons.append("missing_frugalgpt_split_aligned_test_jsonl")
    if not local_base.exists():
        reasons.append("missing_local_encoder_checkpoint")
    reasons.extend(_missing_modules(["torch", "numpy", "pandas", "sklearn", "transformers", "tqdm", "scipy"], modules))
    smoke_executed = not reasons and _frugalgpt_smoke_log_succeeded(smoke_log)
    return _row(
        check_id="frugalgpt_local_scorer_cli",
        baseline="FrugalGPT local scorer CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/FrugalGPT",
        command=(
            "python data/raw/external/LLMRouterBench/baselines/FrugalGPT/train_router_from_results.py "
            f"--train-jsonl {train} --test-jsonl {test} --local-base {local_base} "
            f"--output-dir {result_dir / 'frugalgpt_split_aligned/output'} "
            "--epochs 1 --max-steps 1 --max-samples 1000 --max-length 128 "
            "--batch-size 16 --eval-batch-size 64 --prob-threshold 0.5"
        ),
        required_paths=[script, train, test, local_base],
        required_modules=["torch", "numpy", "pandas", "sklearn", "transformers", "tqdm", "scipy"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked"),
        execution_evidence=str(smoke_log) if smoke_executed else "",
    )


def _frugalgpt_smoke_log_succeeded(smoke_log: Path) -> bool:
    if not smoke_log.exists():
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    return "Done. Per-model scorers saved inside output dir if provided." in text and "Traceback" not in text


def _embedllm_knn_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/knn.py"
    train = result_dir / "embedllm_assets/train.csv"
    test = result_dir / "embedllm_assets/test.csv"
    smoke_train = result_dir / "embedllm_assets/smoke_train.csv"
    smoke_test = result_dir / "embedllm_assets/smoke_test.csv"
    smoke_log = result_dir / "embedllm_assets/embedllm_knn_smoke_stdout.log"
    tensor_paths = {
        "train_x": result_dir / "embedllm_assets/knn_train_x.pth",
        "train_y": result_dir / "embedllm_assets/knn_train_y.pth",
        "test_x": result_dir / "embedllm_assets/knn_test_x.pth",
        "test_y": result_dir / "embedllm_assets/knn_test_y.pth",
    }
    full_log = _embedllm_knn_full_log(result_dir)
    reasons = _missing_paths([script])
    if not train.exists():
        reasons.append("missing_embedllm_train_csv")
    if not test.exists():
        reasons.append("missing_embedllm_test_csv")
    reasons.extend(_embedllm_knn_argparse_reasons(script))
    reasons.extend(_missing_modules(["sklearn", "torch", "numpy", "pandas", "sentence_transformers", "tqdm"], modules))
    tensor_ready = all(path.exists() for path in tensor_paths.values())
    full_executed = not reasons and tensor_ready and full_log is not None
    smoke_executed = not reasons and _embedllm_knn_smoke_log_succeeded(smoke_log)
    command_train = smoke_train if smoke_train.exists() and smoke_test.exists() else train
    command_test = smoke_test if smoke_train.exists() and smoke_test.exists() else test
    if tensor_ready:
        command = (
            "cd data/raw/external/LLMRouterBench/baselines/EmbedLLM && "
            "python algorithm/knn.py --input-format tensor "
            f"--train-x-path {_command_path(tensor_paths['train_x'])} "
            f"--train-y-path {_command_path(tensor_paths['train_y'])} "
            f"--test-x-path {_command_path(tensor_paths['test_x'])} "
            f"--test-y-path {_command_path(tensor_paths['test_y'])}"
        )
    else:
        command = (
            "cd data/raw/external/LLMRouterBench/baselines/EmbedLLM && "
            f"python algorithm/knn.py --input-format csv --train-csv-path {command_train} --test-csv-path {command_test}"
        )
    return _row(
        check_id="embedllm_knn_cli",
        baseline="EmbedLLM KNN upstream CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/EmbedLLM",
        command=command,
        required_paths=[script, train, test],
        required_modules=["sklearn", "torch", "numpy", "pandas", "sentence_transformers", "tqdm"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="executed" if full_executed else ("smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked")),
        execution_evidence=str(full_log) if full_executed else (str(smoke_log) if smoke_executed else ""),
    )


def _embedllm_knn_argparse_reasons(script: Path) -> list[str]:
    if not script.exists():
        return []
    source = script.read_text(encoding="utf-8")
    reasons = []
    if "--train-csv-path" in source and re.search(r"\bargs\.train_csv\b", source):
        reasons.append("upstream_argparse_mismatch:train_csv_path")
    if "--test-csv-path" in source and re.search(r"\bargs\.test_csv\b", source):
        reasons.append("upstream_argparse_mismatch:test_csv_path")
    return reasons


def _embedllm_knn_smoke_log_succeeded(smoke_log: Path) -> bool:
    if not smoke_log.exists():
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    return "Mean Test Accuracy" in text and "Traceback" not in text


def _embedllm_knn_full_log(result_dir: Path) -> Path | None:
    metric_dir = result_dir / "embedllm_knn_cli_metrics"
    for name in ["embedllm_knn_k131_stdout.log", "embedllm_knn_k15_stdout.log", "embedllm_knn_k3_stdout.log"]:
        log = metric_dir / name
        if _embedllm_knn_smoke_log_succeeded(log):
            return log
    for log in sorted(metric_dir.glob("embedllm_knn_k*_stdout.log")):
        if _embedllm_knn_smoke_log_succeeded(log):
            return log
    return None


def _embedllm_mf_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/mf.py"
    script_dir = root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM"
    train = result_dir / "embedllm_assets/train.csv"
    test = result_dir / "embedllm_assets/test.csv"
    smoke_train = result_dir / "embedllm_assets/smoke_train.csv"
    smoke_test = result_dir / "embedllm_assets/smoke_test.csv"
    full_log = _embedllm_mf_full_log(result_dir)
    use_smoke_inputs = full_log is None and smoke_train.exists() and smoke_test.exists()
    command_train = smoke_train if use_smoke_inputs else train
    command_test = smoke_test if use_smoke_inputs else test
    routecode_question_embeddings = result_dir / "embedllm_assets/question_embeddings.pth"
    mf_question_embeddings = result_dir / "embedllm_assets/question_embeddings_3584.pth"
    question_embeddings = mf_question_embeddings if mf_question_embeddings.exists() else routecode_question_embeddings
    full_metric_dir = result_dir / "embedllm_mf_cli_metrics"
    model_embeddings = (
        result_dir / "embedllm_assets/model_embeddings_smoke.pth"
        if use_smoke_inputs
        else full_metric_dir / "model_embeddings.pth"
    )
    model_path = (
        result_dir / "embedllm_assets/saved_model_smoke.pth"
        if use_smoke_inputs
        else full_metric_dir / "saved_model.pth"
    )
    smoke_log = result_dir / "embedllm_assets/embedllm_mf_smoke_stdout.log"
    reasons = _missing_paths([script])
    if not train.exists():
        reasons.append("missing_embedllm_train_csv")
    if not test.exists():
        reasons.append("missing_embedllm_test_csv")
    if not question_embeddings.exists():
        reasons.append("missing_embedllm_question_embeddings")
    else:
        reasons.extend(_embedllm_question_embedding_reasons(question_embeddings, modules))
    required_modules = ["torch", "numpy", "pandas", "tqdm"]
    if not _embedllm_mf_has_optional_wandb(script):
        required_modules.append("wandb")
    reasons.extend(_missing_modules(required_modules, modules))
    full_executed = not reasons and full_log is not None
    smoke_executed = not reasons and _embedllm_mf_smoke_log_succeeded(smoke_log)
    model_embedding_dim = 16
    bounded_flags = "--num-epochs 1 --batch-size 32768 "
    return _row(
        check_id="embedllm_mf_cli",
        baseline="EmbedLLM MF upstream CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/EmbedLLM",
        command=(
            f"cd {_command_path(script_dir)} && "
            f"python algorithm/mf.py --train-data-path {_command_path(command_train)} "
            f"--test-data-path {_command_path(command_test)} "
            f"--question-embedding-path {_command_path(question_embeddings)} "
            f"--embedding-save-path {_command_path(model_embeddings)} "
            f"--model-save-path {_command_path(model_path)} {bounded_flags}"
            f"--embedding-dim {model_embedding_dim} --eval-mode router --wandb-run-name routecode-mf-smoke"
        ),
        required_paths=[script, train, test, question_embeddings],
        required_modules=required_modules,
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="executed" if full_executed else ("smoke_executed" if smoke_executed else ("ready" if not reasons else "blocked")),
        execution_evidence=str(full_log) if full_executed else (str(smoke_log) if smoke_executed else ""),
    )


def _embedllm_mf_has_optional_wandb(script: Path) -> bool:
    if not script.exists():
        return False
    source = script.read_text(encoding="utf-8")
    return "_NoOpWandbRun" in source and "except ModuleNotFoundError" in source


def _embedllm_mf_smoke_log_succeeded(smoke_log: Path) -> bool:
    if not smoke_log.exists():
        return False
    text = smoke_log.read_text(encoding="utf-8", errors="replace")
    return "Best Dataset-Level Accuracy" in text and "Traceback" not in text and "ModuleNotFoundError" not in text


def _embedllm_mf_full_log(result_dir: Path) -> Path | None:
    log = result_dir / "embedllm_mf_cli_metrics/embedllm_mf_stdout.log"
    if _embedllm_mf_smoke_log_succeeded(log):
        return log
    return None


def _embedllm_question_embedding_reasons(
    question_embeddings: Path,
    modules: Mapping[str, bool],
    *,
    expected_dim: int = 3584,
) -> list[str]:
    """Check the upstream EmbedLLM MF script's fixed text embedding contract."""

    if not _module_available("torch", modules):
        return []
    try:
        import torch

        tensor = torch.load(question_embeddings, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive audit path
        return [f"embedllm_question_embedding_unreadable:{type(exc).__name__}"]
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 2:
        return ["embedllm_question_embedding_shape_invalid"]
    actual_dim = int(shape[-1])
    if actual_dim != expected_dim:
        return [f"embedllm_question_embedding_dim_mismatch:expected_{expected_dim},got_{actual_dim}"]
    return []


def _command_path(path: Path) -> str:
    return str(path.resolve() if not path.is_absolute() else path)


def _best_route_train_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/Best-route-llm/train_router.py"
    train = result_dir / "best_route_assets/train.jsonl"
    validation = result_dir / "best_route_assets/validation.jsonl"
    test = result_dir / "best_route_assets/test.jsonl"
    local_model = result_dir / "external_checkpoints/deberta-v3-small"
    reasons = _missing_paths([script])
    if not train.exists():
        reasons.append("missing_best_route_train_data")
    if not validation.exists():
        reasons.append("missing_best_route_validation_data")
    if not test.exists():
        reasons.append("missing_best_route_test_data")
    if not local_model.exists():
        reasons.append("missing_best_route_local_model_checkpoint")
    reasons.extend(_missing_modules(["torch", "transformers", "llm_blender"], modules))
    return _row(
        check_id="best_route_train_cli",
        baseline="BEST-Route / HybridLLM upstream training CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/Best-route-llm",
        command=(
            "cd data/raw/external/LLMRouterBench/baselines/Best-route-llm && "
            f"python train_router.py --model_name {local_model} --train_data_path {train} "
            f"--eval_data_path {validation} --test_data_path {test} "
            f"--output_dir {result_dir / 'best_route_assets/output'} --do_train True --do_eval True --do_predict True"
        ),
        required_paths=[script, train, validation, test, local_model],
        required_modules=["torch", "transformers", "llm_blender"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="ready" if not reasons else "blocked",
    )


def _routerdc_train_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/RouterDC/train_router_mdeberta_7b.py"
    train = result_dir / "routerdc_assets/train.json"
    test = result_dir / "routerdc_assets/test.json"
    final_eval = result_dir / "routerdc_assets/final_eval.json"
    local_model = result_dir / "external_checkpoints/mdeberta-v3-base"
    reasons = _missing_paths([script])
    if not train.exists():
        reasons.append("missing_routerdc_train_data")
    if not test.exists():
        reasons.append("missing_routerdc_test_data")
    if not final_eval.exists():
        reasons.append("missing_routerdc_final_eval_data")
    if not local_model.exists():
        reasons.append("missing_routerdc_local_model_checkpoint")
    reasons.extend(_missing_modules(["torch", "transformers", "deepspeed", "wandb"], modules))
    return _row(
        check_id="routerdc_train_cli",
        baseline="RouterDC upstream training CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/RouterDC",
        command=(
            "cd data/raw/external/LLMRouterBench/baselines/RouterDC && "
            f"python train_router_mdeberta_7b.py --data_paths {train} --test_data_paths {test} "
            f"--final_eval_data_paths {final_eval} --save_path {result_dir / 'routerdc_assets/output'}"
        ),
        required_paths=[script, train, test, final_eval, local_model],
        required_modules=["torch", "transformers", "deepspeed", "wandb"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="ready" if not reasons else "blocked",
    )


def _modelsat_train_cli(root: Path, result_dir: Path, modules: Mapping[str, bool]) -> dict[str, object]:
    script = root / "data/raw/external/LLMRouterBench/baselines/MODEL-SAT/model_sat_train.py"
    train = result_dir / "modelsat_assets/seed42/train.json"
    validation = result_dir / "modelsat_assets/seed42/test.json"
    ood = result_dir / "modelsat_assets/seed42/ood.json"
    model_description = result_dir / "modelsat_assets/seed42/model_description.json"
    base_model = result_dir / "external_checkpoints/qwen2.5-7b-instruct"
    embed_model = result_dir / "external_checkpoints/gte-qwen2-7b-instruct"
    reasons = _missing_paths([script])
    if not train.exists():
        reasons.append("missing_modelsat_train_data")
    if not validation.exists():
        reasons.append("missing_modelsat_validation_data")
    if not ood.exists():
        reasons.append("missing_modelsat_ood_data")
    if not model_description.exists():
        reasons.append("missing_modelsat_model_description")
    if not base_model.exists():
        reasons.append("missing_modelsat_base_model_checkpoint")
    if not embed_model.exists():
        reasons.append("missing_modelsat_embedding_model_checkpoint")
    reasons.extend(_missing_modules(["datasets", "nltk", "sentence_transformers", "deepspeed"], modules))
    return _row(
        check_id="modelsat_train_cli",
        baseline="MODEL-SAT upstream training CLI",
        upstream_source="data/raw/external/LLMRouterBench/baselines/MODEL-SAT",
        command=(
            "cd data/raw/external/LLMRouterBench/baselines/MODEL-SAT && "
            f"python model_sat_train.py --base_model_name_or_path {base_model} "
            f"--embed_model_name_or_path {embed_model} --experts_information_file {model_description} "
            f"--dataset_name '{{\"train\":\"{train}\",\"validation\":\"{validation}\",\"ood\":\"{ood}\"}}' "
            f"--output_dir {result_dir / 'modelsat_assets/output'}"
        ),
        required_paths=[script, train, validation, ood, model_description, base_model, embed_model],
        required_modules=["datasets", "nltk", "sentence_transformers", "deepspeed"],
        blocking_reasons=reasons,
        no_api_compatible=True,
        routecode_metric_compatible=False,
        exact_upstream_command=True,
        status="ready" if not reasons else "blocked",
    )


def _row(
    *,
    check_id: str,
    baseline: str,
    upstream_source: str,
    command: str,
    required_paths: list[Path],
    required_modules: list[str],
    blocking_reasons: list[str],
    no_api_compatible: bool,
    routecode_metric_compatible: bool,
    exact_upstream_command: bool,
    status: str,
    runnable_override: bool | None = None,
    execution_evidence: str = "",
) -> dict[str, object]:
    runnable = bool(runnable_override) if runnable_override is not None else not blocking_reasons
    return {
        "check_id": check_id,
        "baseline": baseline,
        "status": status,
        "runnable_now": runnable,
        "no_api_compatible": no_api_compatible,
        "routecode_metric_compatible": routecode_metric_compatible,
        "exact_upstream_command": exact_upstream_command,
        "blocking_reasons": ";".join(dict.fromkeys(reason for reason in blocking_reasons if reason)),
        "required_modules": ",".join(required_modules),
        "required_paths_present": int(sum(path.exists() for path in required_paths)),
        "required_paths_total": len(required_paths),
        "execution_evidence": execution_evidence,
        "upstream_source": upstream_source,
        "command": command,
    }


def _missing_paths(paths: list[Path]) -> list[str]:
    return [f"missing_path:{path}" for path in paths if not path.exists()]


def _module_available(module: str, overrides: Mapping[str, bool]) -> bool:
    if module in overrides:
        return bool(overrides[module])
    return importlib.util.find_spec(module) is not None


def _missing_modules(modules: list[str], overrides: Mapping[str, bool]) -> list[str]:
    missing = [module for module in modules if not _module_available(module, overrides)]
    return ["missing_python_modules:" + ",".join(missing)] if missing else []


def _embedding_env_requirements(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    model_cfg = dict(config.get("embedding_model") or {})
    required = []
    api_key = model_cfg.get("api_key")
    if isinstance(api_key, str) and api_key.isupper() and "_" in api_key:
        required.append(api_key)
    base_url = model_cfg.get("base_url")
    if isinstance(base_url, str) and base_url.isupper() and "_" in base_url:
        required.append(base_url)
    return required
