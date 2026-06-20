from __future__ import annotations

from pathlib import Path

import torch

from routecode.eval.external_command_readiness import inspect_external_command_readiness


def test_external_command_readiness_marks_missing_cli_dependency_and_graphrouter_stack(tmp_path):
    project_root = tmp_path
    (project_root / "data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization").mkdir(
        parents=True
    )
    (project_root / "data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization/train_matrix_factorization.py").write_text(
        "# train\n",
        encoding="utf-8",
    )
    (project_root / "data/raw/external/LLMRouterBench/baselines/RouteLLM/evaluate_mf.py").write_text(
        "# eval\n",
        encoding="utf-8",
    )
    (project_root / "data/raw/external/LLMRouterBench/config").mkdir(parents=True)
    (project_root / "data/raw/external/LLMRouterBench/config/embedding_config.yaml").write_text(
        "embedding_model:\n  api_key: EMBEDDING_API_KEY\n",
        encoding="utf-8",
    )
    (project_root / "data/raw/external/LLMRouterBench/baselines/GraphRouter").mkdir(parents=True)
    (project_root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/run_exp.py").write_text(
        "# graph\n",
        encoding="utf-8",
    )

    table = inspect_external_command_readiness(
        project_root,
        module_availability={
            "loguru": False,
            "torch": True,
            "bert_score": False,
            "litellm": False,
            "torch_geometric": False,
            "transformers": True,
            "wandb": False,
        },
        env={},
    )

    rows = table.set_index("check_id")
    assert not bool(rows.loc["routellm_mf_train_cli", "runnable_now"])
    assert "missing_python_modules:loguru" in rows.loc["routellm_mf_train_cli", "blocking_reasons"]
    assert not bool(rows.loc["routellm_mf_eval_cli", "no_api_compatible"])
    assert "embedding_config_requires_env:EMBEDDING_API_KEY" in rows.loc["routellm_mf_eval_cli", "blocking_reasons"]
    assert not bool(rows.loc["graphrouter_cli", "runnable_now"])
    assert "missing_python_modules:bert_score,litellm,torch_geometric,wandb" in rows.loc[
        "graphrouter_cli", "blocking_reasons"
    ]


def test_external_command_readiness_can_mark_routecode_local_mf_metric_present(tmp_path):
    project_root = tmp_path
    result_dir = project_root / "results/custom_run"
    metric = result_dir / "table_routellm_mf_split_aligned.csv"
    metric.parent.mkdir(parents=True)
    metric.write_text("method,mean_utility\nroutellm_mf_split_aligned_t0.5,0.7\n", encoding="utf-8")

    table = inspect_external_command_readiness(project_root, result_dir=result_dir, module_availability={}, env={})
    row = table.set_index("check_id").loc["routecode_local_routellm_mf_metric"]

    assert bool(row["runnable_now"])
    assert bool(row["routecode_metric_compatible"])
    assert row["status"] == "available"
    assert str(metric) in row["command"]


def test_external_command_readiness_can_mark_routecode_local_embedllm_knn_metric_present(tmp_path):
    project_root = tmp_path
    result_dir = project_root / "results/custom_run"
    metric = result_dir / "table_embedllm_knn_split_aligned.csv"
    metric.parent.mkdir(parents=True)
    metric.write_text("method,mean_utility\nembedllm_knn_split_aligned_k131,0.75\n", encoding="utf-8")

    table = inspect_external_command_readiness(project_root, result_dir=result_dir, module_availability={}, env={})
    row = table.set_index("check_id").loc["routecode_local_embedllm_knn_metric"]

    assert bool(row["runnable_now"])
    assert bool(row["routecode_metric_compatible"])
    assert row["status"] == "available"
    assert str(metric) in row["command"]


def test_external_command_readiness_can_mark_routecode_local_frugalgpt_metric_present(tmp_path):
    project_root = tmp_path
    result_dir = project_root / "results/custom_run"
    metric = result_dir / "table_frugalgpt_split_aligned.csv"
    metric.parent.mkdir(parents=True)
    metric.write_text("method,mean_utility\nfrugalgpt_local_scorer_t0.5,0.75\n", encoding="utf-8")

    table = inspect_external_command_readiness(project_root, result_dir=result_dir, module_availability={}, env={})
    row = table.set_index("check_id").loc["routecode_local_frugalgpt_metric"]

    assert bool(row["runnable_now"])
    assert bool(row["routecode_metric_compatible"])
    assert row["status"] == "available"
    assert str(metric) in row["command"]


def test_external_command_readiness_can_mark_avengerspro_upstream_metric_present(tmp_path):
    project_root = tmp_path
    result_dir = project_root / "results/custom_run"
    metric_dir = result_dir / "avengerspro_upstream_metric"
    metric_dir.mkdir(parents=True)
    metric = result_dir / "table_avengerspro_upstream_metric.csv"
    routing_details = metric_dir / "raw_routing_details.json"
    metric.write_text(
        "method,mean_utility\navengerspro_upstream_simple_cluster_postprocessed,0.74\n",
        encoding="utf-8",
    )
    routing_details.write_text('[{"query_id": "q0", "selected_model": "m0"}]\n', encoding="utf-8")

    table = inspect_external_command_readiness(project_root, result_dir=result_dir, module_availability={}, env={})
    row = table.set_index("check_id").loc["routecode_upstream_avengerspro_metric"]

    assert bool(row["runnable_now"])
    assert bool(row["routecode_metric_compatible"])
    assert not bool(row["exact_upstream_command"])
    assert row["status"] == "available"
    assert str(metric) in row["command"]
    assert row["execution_evidence"].endswith("avengerspro_upstream_metric/raw_routing_details.json")


def test_external_command_readiness_marks_upstream_mf_training_cli_executed_when_checkpoint_exists(tmp_path):
    project_root = tmp_path
    train_script = (
        project_root
        / "data/raw/external/LLMRouterBench/baselines/RouteLLM/routers/matrix_factorization/train_matrix_factorization.py"
    )
    train_script.parent.mkdir(parents=True)
    train_script.write_text("# train\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "routellm_mf_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "mf_train_config.local.json").write_text("{}", encoding="utf-8")
    (asset_dir / "mf_model.pt").write_bytes(b"checkpoint")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"loguru": True, "torch": True, "numpy": True, "tqdm": True},
        env={},
    )

    row = table.set_index("check_id").loc["routellm_mf_train_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "executed"
    assert row["execution_evidence"].endswith("routellm_mf_assets/mf_model.pt")
    assert str(asset_dir / "mf_train_config.local.json") in row["command"]


def test_external_command_readiness_marks_upstream_mf_eval_cli_cache_smoke_executed(tmp_path):
    project_root = tmp_path
    eval_script = project_root / "data/raw/external/LLMRouterBench/baselines/RouteLLM/evaluate_mf.py"
    eval_script.parent.mkdir(parents=True)
    eval_script.write_text("# eval\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "routellm_mf_assets"
    asset_dir.mkdir(parents=True)
    for name in [
        "pairwise_test.json",
        "mf_model.pt",
        "mf_eval_config.local.json",
        "embedding_config.local.yaml",
        "embedding_cache.jsonl",
        "metadata.json",
    ]:
        (asset_dir / name).write_text("{}\n", encoding="utf-8")
    output = asset_dir / "mf_eval_smoke_results.json"
    output.write_text('{"total": 1, "selection_accuracy": 1.0}\n', encoding="utf-8")
    log = asset_dir / "routellm_mf_eval_stdout.log"
    log.write_text("Evaluation complete:\nSaved metrics to output.json\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"loguru": True, "torch": True, "numpy": True},
        env={},
    )

    row = table.set_index("check_id").loc["routellm_mf_eval_cli"]
    assert bool(row["runnable_now"])
    assert bool(row["no_api_compatible"])
    assert row["status"] == "smoke_executed"
    assert row["execution_evidence"].endswith("routellm_mf_assets/routellm_mf_eval_stdout.log")
    assert "--config" in row["command"]
    assert "mf_eval_config.local.json" in row["command"]
    assert "mf_eval_smoke_results.json" in row["command"]


def test_external_command_readiness_marks_llmrouter_inference_cli_smoke_executed(tmp_path):
    project_root = tmp_path
    cli_dir = project_root / "data/raw/external/LLMRouter/llmrouter/cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "router_inference.py").write_text("# infer\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "llmrouter_library_adapters"
    asset_dir.mkdir(parents=True)
    (asset_dir / "knnrouter_train.yaml").write_text("model_path: {}\n", encoding="utf-8")
    (asset_dir / "knn_cli_model.pkl").write_bytes(b"pickle")
    (asset_dir / "query_inference_smoke.jsonl").write_text('{"query": "hello"}\n', encoding="utf-8")
    (asset_dir / "query_embedding_lookup.pt").write_bytes(b"cache")
    (asset_dir / "llmrouter_knn_infer_stdout.log").write_text(
        '[{"success": true, "query": "hello", "model_name": "m0"}]\n',
        encoding="utf-8",
    )

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"torch": True, "numpy": True, "pandas": True, "sklearn": True, "yaml": True},
        env={},
    )

    row = table.set_index("check_id").loc["llmrouter_knn_infer_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "smoke_executed"
    assert row["execution_evidence"].endswith("llmrouter_library_adapters/llmrouter_knn_infer_stdout.log")
    assert "--route-only" in row["command"]
    assert "query_inference_smoke.jsonl" in row["command"]


def test_external_command_readiness_prefers_llmrouter_full_inference_output(tmp_path):
    project_root = tmp_path
    cli_dir = project_root / "data/raw/external/LLMRouter/llmrouter/cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "router_inference.py").write_text("# infer\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "llmrouter_library_adapters"
    asset_dir.mkdir(parents=True)
    (asset_dir / "knnrouter_train.yaml").write_text("model_path: {}\n", encoding="utf-8")
    (asset_dir / "knn_cli_model.pkl").write_bytes(b"pickle")
    (asset_dir / "query_inference_smoke.jsonl").write_text('{"query": "hello"}\n', encoding="utf-8")
    (asset_dir / "query_inference_test.jsonl").write_text('{"query": "hello"}\n{"query": "world"}\n', encoding="utf-8")
    (asset_dir / "query_embedding_lookup.pt").write_bytes(b"cache")
    (asset_dir / "llmrouter_knn_full_predictions.json").write_text(
        '[{"success": true, "query": "hello", "model_name": "m0"}, {"success": true, "query": "world", "model_name": "m1"}]\n',
        encoding="utf-8",
    )
    (asset_dir / "llmrouter_knn_full_infer_stdout.log").write_text("", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"torch": True, "numpy": True, "pandas": True, "sklearn": True, "yaml": True},
        env={},
    )

    row = table.set_index("check_id").loc["llmrouter_knn_infer_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "executed"
    assert row["execution_evidence"].endswith("llmrouter_library_adapters/llmrouter_knn_full_predictions.json")
    assert "query_inference_test.jsonl" in row["command"]
    assert "--output" in row["command"]


def test_external_command_readiness_marks_avengerspro_cache_smoke_executed(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/AvengersPro/simple_cluster_router.py"
    script.parent.mkdir(parents=True)
    script.write_text("# avengers\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "avengerspro_split_aligned"
    asset_dir.mkdir(parents=True)
    for name in [
        "train.jsonl",
        "test.jsonl",
        "smoke_train.jsonl",
        "smoke_test.jsonl",
        "baseline_scores.json",
        "embedding_cache.jsonl",
        "simple_cluster_config.local.json",
    ]:
        (asset_dir / name).write_text("{}\n", encoding="utf-8")
    output = asset_dir / "simple_cluster_smoke_results.json"
    output.write_text('{"results": {"accuracy": 100.0, "total_queries": 1}}\n', encoding="utf-8")
    log = asset_dir / "avengerspro_simple_cluster_smoke_stdout.log"
    log.write_text("Routing evaluation completed successfully\nResults saved to: output.json\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "datasets": True,
            "joblib": True,
            "tiktoken": True,
            "yaml": True,
            "sklearn": True,
            "numpy": True,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["avengerspro_cli"]
    assert bool(row["runnable_now"])
    assert bool(row["no_api_compatible"])
    assert row["status"] == "smoke_executed"
    assert row["execution_evidence"].endswith("avengerspro_split_aligned/avengerspro_simple_cluster_smoke_stdout.log")
    assert "--config" in row["command"]
    assert "simple_cluster_config.local.json" in row["command"]


def test_external_command_readiness_prefers_avengerspro_full_cli_metric_output(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/AvengersPro/simple_cluster_router.py"
    script.parent.mkdir(parents=True)
    script.write_text("# avengers\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "avengerspro_split_aligned"
    asset_dir.mkdir(parents=True)
    for name in [
        "train.jsonl",
        "test.jsonl",
        "smoke_train.jsonl",
        "smoke_test.jsonl",
        "baseline_scores.json",
        "embedding_cache.jsonl",
        "simple_cluster_config.local.json",
    ]:
        (asset_dir / name).write_text("{}\n", encoding="utf-8")
    metric_dir = result_dir / "avengerspro_cli_metrics"
    metric_dir.mkdir(parents=True)
    (metric_dir / "simple_cluster_config.full.json").write_text("{}\n", encoding="utf-8")
    (metric_dir / "full_embedding_cache.jsonl").write_text("{}\n", encoding="utf-8")
    (metric_dir / "simple_cluster_full_results.json").write_text(
        '{"results": {"accuracy": 62.5, "total_queries": 8}}\n',
        encoding="utf-8",
    )
    (metric_dir / "avengerspro_simple_cluster_stdout.log").write_text(
        "Routing evaluation completed successfully\nResults saved to: full_output.json\n",
        encoding="utf-8",
    )

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "datasets": True,
            "joblib": True,
            "tiktoken": True,
            "yaml": True,
            "sklearn": True,
            "numpy": True,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["avengerspro_cli"]
    assert bool(row["runnable_now"])
    assert bool(row["no_api_compatible"])
    assert row["status"] == "executed"
    assert row["execution_evidence"].endswith("avengerspro_cli_metrics/simple_cluster_full_results.json")
    assert "simple_cluster_config.full.json" in row["command"]
    assert "simple_cluster_full_results.json" in row["command"]


def test_external_command_readiness_uses_result_dir_graphrouter_assets(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/run_exp.py"
    script.parent.mkdir(parents=True)
    script.write_text("# graph\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "graphrouter_assets"
    asset_dir.mkdir(parents=True)
    router_data = asset_dir / "router_data.csv"
    llm_embeddings = asset_dir / "llm_description_embedding.pkl"
    config = asset_dir / "config.local.yaml"
    router_data.write_text("query_id,llm,effect,cost\nq0,m0,1.0,0.0\n", encoding="utf-8")
    llm_embeddings.write_bytes(b"pickle")
    config.write_text("saved_router_data_path: router_data.csv\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "bert_score": False,
            "litellm": False,
            "torch_geometric": False,
            "wandb": False,
        },
        env={},
    )

    row = table.set_index("check_id").loc["graphrouter_cli"]
    assert not bool(row["runnable_now"])
    assert "missing_graphrouter_router_data" not in row["blocking_reasons"]
    assert "missing_graphrouter_llm_description_embeddings" not in row["blocking_reasons"]
    assert "missing_python_modules:bert_score,litellm,torch_geometric,wandb" in row["blocking_reasons"]
    assert str(config) in row["command"]


def test_external_command_readiness_marks_graphrouter_cli_executed_from_smoke_log(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/run_exp.py"
    script.parent.mkdir(parents=True)
    script.write_text("# graph\n", encoding="utf-8")
    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "graphrouter_assets"
    metric_dir = result_dir / "graphrouter_cli_metrics"
    asset_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    (asset_dir / "router_data.csv").write_text("query_id,llm,effect,cost\nq0,m0,1.0,0.0\n", encoding="utf-8")
    (asset_dir / "llm_description_embedding.pkl").write_bytes(b"pickle")
    (asset_dir / "config.local.yaml").write_text("saved_router_data_path: router_data.csv\n", encoding="utf-8")
    (metric_dir / "config.smoke.yaml").write_text("saved_router_data_path: router_data.csv\n", encoding="utf-8")
    (metric_dir / "graphrouter_stdout.log").write_text(
        "\n".join(
            [
                "BEST TEST CHECKPOINT METRICS (used for model selection)",
                "Dataset-Level Average Accuracy: 0.5000",
                "Sample-Level Average Accuracy:  0.6000",
            ]
        ),
        encoding="utf-8",
    )
    (metric_dir / "model_path").mkdir()
    (metric_dir / "model_path/best_model.pth").write_bytes(b"checkpoint")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"bert_score": True, "litellm": True, "torch_geometric": True, "wandb": True},
        env={},
    )

    row = table.set_index("check_id").loc["graphrouter_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "executed"
    assert row["blocking_reasons"] == ""
    assert row["execution_evidence"].endswith("graphrouter_cli_metrics/graphrouter_stdout.log")
    assert "graphrouter_cli_metrics/config.smoke.yaml" in row["command"]


def test_external_command_readiness_records_absolute_graphrouter_smoke_config_for_relative_result_dir(
    tmp_path, monkeypatch
):
    project_root = tmp_path
    monkeypatch.chdir(project_root)
    script = project_root / "data/raw/external/LLMRouterBench/baselines/GraphRouter/run_exp.py"
    script.parent.mkdir(parents=True)
    script.write_text("# graph\n", encoding="utf-8")
    result_dir = Path("results/custom_run")
    asset_dir = result_dir / "graphrouter_assets"
    metric_dir = result_dir / "graphrouter_cli_metrics"
    asset_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    (asset_dir / "router_data.csv").write_text("query_id,llm,effect,cost\nq0,m0,1.0,0.0\n", encoding="utf-8")
    (asset_dir / "llm_description_embedding.pkl").write_bytes(b"pickle")
    (asset_dir / "config.local.yaml").write_text("saved_router_data_path: router_data.csv\n", encoding="utf-8")
    smoke_config = metric_dir / "config.smoke.yaml"
    smoke_config.write_text("saved_router_data_path: router_data.csv\n", encoding="utf-8")
    (metric_dir / "graphrouter_stdout.log").write_text(
        "\n".join(
            [
                "BEST TEST CHECKPOINT METRICS (used for model selection)",
                "Dataset-Level Average Accuracy: 0.5000",
                "Sample-Level Average Accuracy:  0.6000",
            ]
        ),
        encoding="utf-8",
    )
    (metric_dir / "model_path").mkdir()
    (metric_dir / "model_path/best_model.pth").write_bytes(b"checkpoint")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"bert_score": True, "litellm": True, "torch_geometric": True, "wandb": True},
        env={},
    )

    row = table.set_index("check_id").loc["graphrouter_cli"]
    assert row["status"] == "executed"
    assert f"--config_file {smoke_config.resolve()}" in row["command"]


def test_external_command_readiness_covers_additional_llmrouterbench_baseline_families(tmp_path):
    project_root = tmp_path
    baseline_root = project_root / "data/raw/external/LLMRouterBench/baselines"
    paths = [
        baseline_root / "FrugalGPT/train_router_from_results.py",
        baseline_root / "EmbedLLM/algorithm/knn.py",
        baseline_root / "EmbedLLM/algorithm/mf.py",
        baseline_root / "Best-route-llm/train_router.py",
        baseline_root / "RouterDC/train_router_mdeberta_7b.py",
        baseline_root / "MODEL-SAT/model_sat_train.py",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# baseline entrypoint\n", encoding="utf-8")

    result_dir = project_root / "results/custom_run"
    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "datasets": False,
            "deepspeed": False,
            "llm_blender": False,
            "nltk": False,
            "sentence_transformers": False,
            "sklearn": True,
            "torch": True,
            "transformers": True,
            "wandb": False,
        },
        env={},
    )

    rows = table.set_index("check_id")
    expected = {
        "frugalgpt_local_scorer_cli",
        "embedllm_knn_cli",
        "embedllm_mf_cli",
        "best_route_train_cli",
        "routerdc_train_cli",
        "modelsat_train_cli",
    }
    assert expected.issubset(set(rows.index))

    assert "missing_frugalgpt_split_aligned_train_jsonl" in rows.loc[
        "frugalgpt_local_scorer_cli", "blocking_reasons"
    ]
    assert "missing_local_encoder_checkpoint" in rows.loc["frugalgpt_local_scorer_cli", "blocking_reasons"]
    assert "missing_embedllm_train_csv" in rows.loc["embedllm_knn_cli", "blocking_reasons"]
    assert "missing_embedllm_question_embeddings" in rows.loc["embedllm_mf_cli", "blocking_reasons"]
    assert "missing_best_route_train_data" in rows.loc["best_route_train_cli", "blocking_reasons"]
    assert "missing_routerdc_train_data" in rows.loc["routerdc_train_cli", "blocking_reasons"]
    assert "missing_modelsat_train_data" in rows.loc["modelsat_train_cli", "blocking_reasons"]
    assert "missing_python_modules:llm_blender" in rows.loc["best_route_train_cli", "blocking_reasons"]
    assert "missing_python_modules:deepspeed,wandb" in rows.loc["routerdc_train_cli", "blocking_reasons"]
    assert "missing_python_modules:datasets,nltk,sentence_transformers,deepspeed" in rows.loc[
        "modelsat_train_cli", "blocking_reasons"
    ]


def test_external_command_readiness_reports_embedllm_mf_embedding_dimension_mismatch(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/mf.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# embedllm mf\n", encoding="utf-8")

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,prompt_id,label\n0,0,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,prompt_id,label\n0,0,1\n", encoding="utf-8")
    torch.save(torch.zeros((1, 256), dtype=torch.float32), asset_dir / "question_embeddings.pth")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={"torch": True, "numpy": True, "pandas": True, "wandb": True, "tqdm": True},
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_mf_cli"]
    assert not bool(row["runnable_now"])
    assert "embedllm_question_embedding_dim_mismatch:expected_3584,got_256" in row["blocking_reasons"]


def test_external_command_readiness_reports_embedllm_knn_argparse_mismatch(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/knn.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "parser.add_argument('--train-csv-path')\n"
        "parser.add_argument('--test-csv-path')\n"
        "load_csv_data(args.train_csv, args.test_csv)\n",
        encoding="utf-8",
    )

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "sklearn": True,
            "torch": True,
            "numpy": True,
            "pandas": True,
            "sentence_transformers": True,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_knn_cli"]
    assert not bool(row["runnable_now"])
    assert "upstream_argparse_mismatch:train_csv_path" in row["blocking_reasons"]


def test_external_command_readiness_checks_frugalgpt_scipy_dependency(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/FrugalGPT/train_router_from_results.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# frugalgpt\n", encoding="utf-8")

    result_dir = project_root / "results/custom_run"
    split_dir = result_dir / "frugalgpt_split_aligned"
    split_dir.mkdir(parents=True)
    (split_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (split_dir / "test.jsonl").write_text("{}\n", encoding="utf-8")
    (result_dir / "external_checkpoints/local_encoder").mkdir(parents=True)

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "torch": True,
            "numpy": True,
            "pandas": True,
            "sklearn": True,
            "transformers": True,
            "tqdm": True,
            "scipy": False,
        },
        env={},
    )

    row = table.set_index("check_id").loc["frugalgpt_local_scorer_cli"]
    assert not bool(row["runnable_now"])
    assert "missing_python_modules:scipy" in row["blocking_reasons"]


def test_external_command_readiness_records_frugalgpt_smoke_execution(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/FrugalGPT/train_router_from_results.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# frugalgpt\n", encoding="utf-8")

    result_dir = project_root / "results/custom_run"
    split_dir = result_dir / "frugalgpt_split_aligned"
    split_dir.mkdir(parents=True)
    (split_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (split_dir / "test.jsonl").write_text("{}\n", encoding="utf-8")
    (result_dir / "external_checkpoints/local_encoder").mkdir(parents=True)
    output_dir = split_dir / "output"
    output_dir.mkdir()
    smoke_log = output_dir / "frugalgpt_smoke_stdout.log"
    smoke_log.write_text("[record_accuracy]=0.5000\nDone. Per-model scorers saved inside output dir if provided.\n")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "torch": True,
            "numpy": True,
            "pandas": True,
            "sklearn": True,
            "transformers": True,
            "tqdm": True,
            "scipy": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["frugalgpt_local_scorer_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "smoke_executed"
    assert row["execution_evidence"].endswith("frugalgpt_smoke_stdout.log")
    assert "--max-steps 1" in row["command"]
    assert "--max-samples 1000" in row["command"]


def test_external_command_readiness_records_embedllm_knn_smoke_execution(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/knn.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "parser.add_argument('--train-csv-path')\n"
        "parser.add_argument('--test-csv-path')\n"
        "load_csv_data(args.train_csv_path, args.test_csv_path)\n",
        encoding="utf-8",
    )

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    (asset_dir / "smoke_train.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    (asset_dir / "smoke_test.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    smoke_log = asset_dir / "embedllm_knn_smoke_stdout.log"
    smoke_log.write_text("Mean Test Accuracy for 3 neighbors: 0.5972\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "sklearn": True,
            "torch": True,
            "numpy": True,
            "pandas": True,
            "sentence_transformers": True,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_knn_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "smoke_executed"
    assert row["execution_evidence"].endswith("embedllm_knn_smoke_stdout.log")
    assert "smoke_train.csv" in row["command"]
    assert "smoke_test.csv" in row["command"]


def test_external_command_readiness_prefers_embedllm_knn_full_tensor_execution(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/knn.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "parser.add_argument('--train-csv-path')\n"
        "parser.add_argument('--test-csv-path')\n"
        "load_csv_data(args.train_csv_path, args.test_csv_path)\n",
        encoding="utf-8",
    )

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    metric_dir = result_dir / "embedllm_knn_cli_metrics"
    asset_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    for name in ["knn_train_x.pth", "knn_train_y.pth", "knn_test_x.pth", "knn_test_y.pth"]:
        (asset_dir / name).write_bytes(b"tensor")
    full_log = metric_dir / "embedllm_knn_k131_stdout.log"
    full_log.write_text("Mean Test Accuracy for 131 neighbors: 0.6860\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "sklearn": True,
            "torch": True,
            "numpy": True,
            "pandas": True,
            "sentence_transformers": True,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_knn_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "executed"
    assert row["execution_evidence"].endswith("embedllm_knn_k131_stdout.log")
    assert "--input-format tensor" in row["command"]
    assert "knn_train_x.pth" in row["command"]


def test_external_command_readiness_uses_embedllm_mf_compatibility_embeddings(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/mf.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("# mf\n", encoding="utf-8")

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,prompt_id,prompt,label\n0,0,q,1\n", encoding="utf-8")
    torch.save(torch.ones((1, 256)), asset_dir / "question_embeddings.pth")
    torch.save(torch.ones((1, 3584)), asset_dir / "question_embeddings_3584.pth")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "torch": True,
            "numpy": True,
            "pandas": True,
            "wandb": False,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_mf_cli"]
    assert "question_embeddings_3584.pth" in row["command"]
    assert "embedllm_question_embedding_dim_mismatch" not in row["blocking_reasons"]
    assert row["blocking_reasons"] == "missing_python_modules:wandb"


def test_external_command_readiness_records_embedllm_mf_smoke_execution(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/mf.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "try:\n    import wandb\nexcept ModuleNotFoundError:\n    _NoOpWandbRun = object\n",
        encoding="utf-8",
    )

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,model_name,prompt_id,prompt,label\n0,m0,0,q,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,model_name,prompt_id,prompt,label\n0,m0,0,q,1\n", encoding="utf-8")
    (asset_dir / "smoke_train.csv").write_text(
        "model_id,model_name,prompt_id,prompt,label\n0,m0,0,q,1\n",
        encoding="utf-8",
    )
    (asset_dir / "smoke_test.csv").write_text(
        "model_id,model_name,prompt_id,prompt,label\n0,m0,0,q,1\n",
        encoding="utf-8",
    )
    torch.save(torch.ones((1, 3584)), asset_dir / "question_embeddings_3584.pth")
    smoke_log = asset_dir / "embedllm_mf_smoke_stdout.log"
    smoke_log.write_text("Best Dataset-Level Accuracy: 1.0000\nModel saved to saved_model_smoke.pth\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "torch": True,
            "numpy": True,
            "pandas": True,
            "wandb": False,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_mf_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "smoke_executed"
    assert row["execution_evidence"].endswith("embedllm_mf_smoke_stdout.log")
    assert "smoke_train.csv" in row["command"]
    assert "smoke_test.csv" in row["command"]
    assert "--num-epochs 1" in row["command"]
    assert "--embedding-dim 16" in row["command"]
    assert "wandb" not in row["required_modules"]
    assert row["blocking_reasons"] == ""


def test_external_command_readiness_records_embedllm_mf_full_execution(tmp_path):
    project_root = tmp_path
    script = project_root / "data/raw/external/LLMRouterBench/baselines/EmbedLLM/algorithm/mf.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "try:\n    import wandb\nexcept ModuleNotFoundError:\n    _NoOpWandbRun = object\n",
        encoding="utf-8",
    )

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "embedllm_assets"
    metric_dir = result_dir / "embedllm_mf_cli_metrics"
    asset_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)
    (asset_dir / "train.csv").write_text("model_id,model_name,prompt_id,prompt,label\n0,m0,0,q,1\n", encoding="utf-8")
    (asset_dir / "test.csv").write_text("model_id,model_name,prompt_id,prompt,label\n0,m0,0,q,1\n", encoding="utf-8")
    torch.save(torch.ones((1, 3584)), asset_dir / "question_embeddings_3584.pth")
    full_log = metric_dir / "embedllm_mf_stdout.log"
    full_log.write_text("Best Dataset-Level Accuracy: 1.0000 at Epoch 1\nModel saved to saved_model.pth\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "torch": True,
            "numpy": True,
            "pandas": True,
            "wandb": False,
            "tqdm": True,
        },
        env={},
    )

    row = table.set_index("check_id").loc["embedllm_mf_cli"]
    assert bool(row["runnable_now"])
    assert row["status"] == "executed"
    assert row["execution_evidence"].endswith("embedllm_mf_cli_metrics/embedllm_mf_stdout.log")
    assert "embedllm_assets/train.csv" in row["command"]
    assert "embedllm_assets/test.csv" in row["command"]
    assert "smoke_train.csv" not in row["command"]
    assert "--embedding-dim 16" in row["command"]
    assert "wandb" not in row["required_modules"]
    assert row["blocking_reasons"] == ""


def test_external_command_readiness_records_llmrouter_exact_train_smoke_execution(tmp_path):
    project_root = tmp_path
    cli = project_root / "data/raw/external/LLMRouter/llmrouter/cli/router_train.py"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("# train cli\n", encoding="utf-8")

    result_dir = project_root / "results/custom_run"
    asset_dir = result_dir / "llmrouter_library_adapters"
    asset_dir.mkdir(parents=True)
    for router in ["knn", "svm"]:
        (asset_dir / f"{router}router_train.yaml").write_text("data_path: {}\n", encoding="utf-8")
        (asset_dir / f"{router}_cli_model.pkl").write_bytes(b"model")
        log = asset_dir / f"llmrouter_{router}_train_stdout.log"
        log.write_text(f"Successfully saved pickle model: {asset_dir / f'{router}_cli_model.pkl'}\n", encoding="utf-8")

    table = inspect_external_command_readiness(
        project_root,
        result_dir=result_dir,
        module_availability={
            "torch": True,
            "numpy": True,
            "pandas": True,
            "sklearn": True,
            "yaml": True,
        },
        env={},
    )

    rows = table.set_index("check_id")
    for check_id, router in [
        ("llmrouter_knn_train_cli", "knnrouter"),
        ("llmrouter_svm_train_cli", "svmrouter"),
    ]:
        row = rows.loc[check_id]
        assert bool(row["runnable_now"])
        assert row["status"] == "smoke_executed"
        assert row["execution_evidence"].endswith(f"llmrouter_{router.replace('router', '')}_train_stdout.log")
        assert f"--router {router}" in row["command"]
        assert "--device cpu --quiet" in row["command"]
        assert row["blocking_reasons"] == ""
