from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from routecode.controlled.config import load_controlled_inputs, load_env_keys
from routecode.controlled.costing import enforce_frontier_budget
from routecode.controlled.exact_math_tools import deterministic_exact_math_answer
from routecode.controlled.live_stage0 import (
    cache_path,
    extract_gemini_text,
    extract_openai_text,
    filter_frontier_models,
    generate_stage0_tasks_from_manifest,
    lazy_load_metadata,
    live_routing_summary,
    local_servers_for_collection,
    normalize_answer,
    score_output,
)
from routecode.controlled.surrogate import ControlledModel, run_controlled_surrogate


def test_lowercase_env_keys_are_detected_without_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("openai_api_key=sk-test\ngemini_api_key=gm-test\n", encoding="utf-8")

    present = load_env_keys(env_path)

    assert present == {"openai_api_key": True, "gemini_api_key": True}


def test_frontier_budget_guard_rejects_over_cap() -> None:
    with pytest.raises(ValueError, match="exceeds total cap"):
        enforce_frontier_budget(
            {"gpt-5.5": 6.0, "gemini-3.5-flash": 2.5},
            max_total_frontier_spend_usd=8.0,
            max_spend_per_frontier_model_usd=6.0,
        )
    with pytest.raises(ValueError, match="above per-model cap"):
        enforce_frontier_budget(
            {"gpt-5.5": 6.5},
            max_total_frontier_spend_usd=8.0,
            max_spend_per_frontier_model_usd=6.0,
        )


def test_controlled_config_has_no_claude_or_anthropic() -> None:
    loaded = load_controlled_inputs("configs/proberoute_controlled.yaml")
    text = yaml.safe_dump(loaded["config"]) + yaml.safe_dump(loaded["servers"]) + yaml.safe_dump(loaded["prices"])

    assert "claude" not in text.lower()
    assert "anthropic" not in text.lower()


def test_local_model_configs_expose_lazy_load_commands() -> None:
    loaded = load_controlled_inputs("configs/proberoute_controlled.yaml")
    local_models = loaded["servers"]["local_models"]

    enabled = [model for model in local_models if model.get("enabled", True)]
    assert enabled
    for model in enabled:
        assert model.get("load_mode") == "lazy"
        assert model.get("start_command")
        assert model.get("served_model_name")


def test_lazy_load_metadata_excludes_load_and_warmup_from_generation_latency() -> None:
    metadata = lazy_load_metadata(
        {"id": "demo-local", "load_mode": "lazy", "model_load_time_s": 12.5, "warmup_time_s": 1.25},
        generation_latency_s=0.75,
    )

    assert metadata["load_mode"] == "lazy"
    assert metadata["model_load_time_s"] == 12.5
    assert metadata["warmup_time_s"] == 1.25
    assert metadata["latency_s"] == 0.75
    assert metadata["latency_excludes_load_warmup"] is True


def test_local_collection_uses_cache_for_stopped_lazy_model(tmp_path: Path) -> None:
    tasks = pd.DataFrame([{"query_id": "math500:live:000"}])
    servers = {
        "local_models": [
            {
                "id": "demo-local",
                "enabled": True,
                "role": "general_local",
                "backend": "vllm",
                "load_mode": "lazy",
                "base_url": "http://localhost:9999/v1",
            }
        ]
    }
    readiness = pd.DataFrame([{"model_id": "demo-local", "status": "unavailable"}])
    local_model = ControlledModel(
        id="demo-local",
        provider="local",
        role="general_local",
        is_local=True,
        is_frontier=False,
        server_backend="vllm",
    )
    raw_path = cache_path(tmp_path, "lazy_run", local_model, "math500:live:000")
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text('{"_status": "success", "_parsed_text": "7"}', encoding="utf-8")

    selected = local_servers_for_collection(
        servers,
        readiness,
        cache_dir=tmp_path,
        run_id="lazy_run",
        tasks=tasks,
        force_rerun=False,
        force_local_rerun=False,
        local_model_ids=["demo-local"],
    )
    forced = local_servers_for_collection(
        servers,
        readiness,
        cache_dir=tmp_path,
        run_id="lazy_run",
        tasks=tasks,
        force_rerun=False,
        force_local_rerun=True,
        local_model_ids=["demo-local"],
    )
    filtered = local_servers_for_collection(
        servers,
        readiness,
        cache_dir=tmp_path,
        run_id="lazy_run",
        tasks=tasks,
        force_rerun=False,
        force_local_rerun=False,
        local_model_ids=["other-local"],
    )

    assert [server["id"] for server in selected] == ["demo-local"]
    assert selected[0]["_ready"] is False
    assert forced == []
    assert filtered == []


def test_frontier_model_filter_limits_budgeted_provider_pool() -> None:
    models = [
        ControlledModel(
            id="qwen3-8b-local",
            provider="local",
            role="local",
            is_local=True,
            is_frontier=False,
            server_backend="vllm",
        ),
        ControlledModel(
            id="gpt-5.5",
            provider="openai",
            role="frontier",
            is_local=False,
            is_frontier=True,
            server_backend="api",
        ),
        ControlledModel(
            id="gemini-3.5-flash",
            provider="google",
            role="frontier",
            is_local=False,
            is_frontier=True,
            server_backend="api",
        ),
    ]

    selected = filter_frontier_models(models, ["gemini-3.5-flash"])

    assert [model.id for model in selected] == ["gemini-3.5-flash"]
    with pytest.raises(ValueError, match="Unknown or disabled frontier model"):
        filter_frontier_models(models, ["claude"])


def test_controlled_dry_run_writes_required_artifacts(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/proberoute_controlled.yaml").read_text(encoding="utf-8"))
    config["run_id"] = "pytest_controlled_dry_run"
    config["budget"]["cache_dir"] = str(tmp_path / "raw_outputs")
    config["budget"]["env_file"] = str(tmp_path / ".env")
    config["outputs"]["output_dir"] = str(tmp_path / "controlled")
    config_path = tmp_path / "proberoute_controlled.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / ".env").write_text("openai_api_key=x\ngemini_api_key=y\n", encoding="utf-8")

    paths = run_controlled_surrogate(config_path, stage="dry_run")
    output_dir = paths["output_dir"]

    required = [
        "model_outputs.parquet",
        "scored_outputs.parquet",
        "cost_latency_summary.csv",
        "table_routability.csv",
        "table_rate_distortion.csv",
        "table_observability_gap.csv",
        "table_main_eval.csv",
        "table_calibration.csv",
        "table_ablation.csv",
        "table_sensitivity.csv",
        "fig_quality_cost_frontier.pdf",
        "fig_latency_breakdown.pdf",
        "fig_rate_distortion.pdf",
        "fig_observability_gap.pdf",
        "fig_calibration_curve.pdf",
        "RUN_REPORT.md",
        "EXPECTED_RESULTS_STATUS.md",
        "PILOT_OBSERVATION_MEMO.md",
    ]
    for name in required:
        assert (output_dir / name).exists(), name

    main = pd.read_csv(output_dir / "table_main_eval.csv")
    assert {"gpt-5.5", "gemini-3.5-flash"}.issubset(
        set(pd.read_parquet(output_dir / "model_outputs.parquet")["model_id"])
    )
    assert "proberoute_threshold_probe" in set(main["method"])
    assert main.loc[main["method"] == "proberoute_threshold_probe", "probe_call_rate"].iloc[0] > 0


def test_live_stage0_response_text_extractors_and_scoring() -> None:
    openai_payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "  Answer: 7. "}],
            }
        ]
    }
    gemini_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "B"}],
                    "role": "model",
                }
            }
        ]
    }

    assert extract_openai_text(openai_payload).strip() == "Answer: 7."
    assert extract_gemini_text(gemini_payload).strip() == "B"
    assert score_output("Answer: 7.", "7", "exact_final_answer") == ("7", 1.0)
    assert score_output("B. because", "B", "multiple_choice") == ("B", 1.0)
    assert score_output("1, 2, 3", "1,2,3", "exact_ordered") == ("1,2,3", 1.0)
    assert score_output("3, 2, 1", "1,2,3", "exact_ordered") == ("3,2,1", 0.0)
    assert score_output("Pope Innocent X", "Innocent X", "short_answer") == ("popeinnocentx", 1.0)
    assert score_output("US$1.65 million", "1.65 million USD", "short_answer")[1] == 1.0
    assert score_output("6,229 °F", "6229 °F", "short_answer")[1] == 1.0
    assert score_output("***trance***", "trance", "exact_ordered")[1] == 1.0
    assert score_output("Reasoning. **no, yes, yes**", "no, yes, yes", "exact_ordered")[1] == 1.0


def test_live_stage0_pass_at_1_scoring_runs_embedded_asserts() -> None:
    gold = json.dumps(
        {
            "benchmark": "mbpp",
            "entry_point": "",
            "tests": "\n".join(
                [
                    "assert add_one(1) == 2",
                    "assert add_one(-1) == 0",
                ]
            ),
        }
    )

    parsed, quality = score_output("```python\ndef add_one(x):\n    return x + 1\n```", gold, "pass_at_1")
    _, bad_quality = score_output("```python\ndef add_one(x):\n    return x\n```", gold, "pass_at_1")

    assert parsed == "passed"
    assert quality == 1.0
    assert bad_quality == 0.0


def test_broad_target_manifest_includes_gsm8k_and_code_rows() -> None:
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "124_phase3_broad_target_manifest.py"
    spec = importlib.util.spec_from_file_location("broad_target_manifest", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    manifest, coverage = module.build_manifest(
        Path("data/raw/external/LLMRouterBench/results/bench"),
        examples_per_dataset=1,
        source_model="Qwen3-8B",
    )

    assert {"gsm8k", "humaneval", "mbpp"}.issubset(set(manifest["dataset"]))
    assert set(manifest.loc[manifest["dataset"].isin(["humaneval", "mbpp"]), "task_type"]) == {"pass_at_1"}
    assert coverage.set_index("dataset").loc["gsm8k", "status"] == "ready"


def test_broad_target_method_package_builds_profile_candidate(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "125_phase3_broad_target_method_package.py"
    spec = importlib.util.spec_from_file_location("broad_target_method", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    model_specs = {
        "gpt-5.5": {"is_local": False, "is_frontier": True, "cost": 0.010, "latency": 1.5},
        "gemini-3.5-flash": {"is_local": False, "is_frontier": True, "cost": 0.001, "latency": 1.0},
        "qwen3-14b-awq-local": {"is_local": True, "is_frontier": False, "cost": 0.0, "latency": 0.2},
        "qwen3-4b-local": {"is_local": True, "is_frontier": False, "cost": 0.0, "latency": 0.3},
        "qwen3-8b-local": {"is_local": True, "is_frontier": False, "cost": 0.0, "latency": 0.4},
    }
    rows = []
    for benchmark, metric, domain in [
        ("bbh", "multiple_choice", "reasoning"),
        ("math500", "exact_final_answer", "math"),
        ("humaneval", "pass_at_1", "code"),
    ]:
        for query_i in range(5):
            query_id = f"{benchmark}:pytest:{query_i:03d}"
            is_test = query_i == 4
            for model_id, spec_row in model_specs.items():
                quality = 0.0
                parsed = "wrong"
                if is_test and benchmark == "bbh" and model_id == "qwen3-4b-local":
                    parsed = "other"
                if is_test and benchmark == "math500" and model_id == "qwen3-4b-local":
                    parsed = "other"
                if is_test and benchmark == "math500" and model_id == "qwen3-8b-local":
                    parsed = "third"
                if is_test and benchmark == "bbh" and model_id == "gemini-3.5-flash":
                    quality = 1.0
                if is_test and benchmark == "math500" and model_id == "gpt-5.5":
                    quality = 1.0
                if is_test and benchmark == "humaneval" and model_id == "qwen3-14b-awq-local":
                    quality = 1.0
                    parsed = "passed"
                rows.append(
                    {
                        "status": "success",
                        "query_id": query_id,
                        "benchmark": benchmark,
                        "domain": domain,
                        "metric": metric,
                        "model_id": model_id,
                        "quality_score": quality,
                        "parsed_answer": parsed,
                        "cost_total_usd": spec_row["cost"],
                        "is_local": spec_row["is_local"],
                        "is_frontier": spec_row["is_frontier"],
                        "latency_s": spec_row["latency"],
                    }
                )

    outputs_path = tmp_path / "model_outputs.parquet"
    pd.DataFrame(rows).to_parquet(outputs_path, index=False)
    outputs = module.load_outputs(outputs_path, lambda_cost=0.35)
    main_eval, selections = module.build_main_eval(outputs, lambda_cost=0.35)
    calibration = module.build_calibration_table(outputs)
    ablation = module.build_ablation_table(outputs, selections)
    sensitivity = module.build_sensitivity_table(outputs)

    out_dir = tmp_path / "method"
    out_dir.mkdir()
    main_eval.to_csv(out_dir / "table_broad_target_main_eval.csv", index=False)
    calibration.to_csv(out_dir / "table_broad_target_calibration.csv", index=False)
    ablation.to_csv(out_dir / "table_broad_target_ablation.csv", index=False)
    sensitivity.to_csv(out_dir / "table_broad_target_sensitivity.csv", index=False)
    module.write_figures(out_dir, main_eval, sensitivity)
    module.write_memo(out_dir / "BROAD_TARGET_METHOD_MEMO.md", outputs_path, main_eval, calibration, ablation, sensitivity)

    selected = main_eval.set_index("method").loc[module.DEFAULT_METHOD]
    oracle = main_eval.set_index("method").loc["cost_aware_oracle"]
    assert selected["n_queries"] == 3
    assert selected["mean_quality"] == oracle["mean_quality"]
    assert selected["oracle_utility_ratio"] == pytest.approx(1.0)
    assert (out_dir / "fig_broad_target_main_eval.pdf").exists()
    assert (out_dir / "BROAD_TARGET_METHOD_MEMO.md").exists()


def test_live_stage0_exact_scoring_normalizes_latex_answer_forms() -> None:
    assert normalize_answer(r"$(3, \frac{\pi}{2})$") == normalize_answer(
        r"\left( 3, \frac{\pi}{2} \right)"
    )
    assert normalize_answer(r"$(3, \frac{\pi{2)$") == normalize_answer(
        r"\left( 3, \frac{\pi}{2} \right)"
    )
    assert normalize_answer(r"$ \sqrt{51} $") == normalize_answer(r"\sqrt{51}")
    assert normalize_answer(r"11√2") == normalize_answer(r"11\sqrt2")
    assert normalize_answer(r"evelyn") == normalize_answer(r"\text{Evelyn}")
    assert normalize_answer(r"(-2, 1)") == normalize_answer(r"(-2,1)")
    assert normalize_answer(r"$4210_5$") == normalize_answer(r"4210_{5}")
    assert score_output(r"$6-5i$", "6 - 5i", "exact_final_answer")[1] == 1.0
    assert score_output("12.00", "12", "exact_final_answer") == ("12", 1.0)
    assert score_output(r"1-\sqrt{19}, 1+\sqrt{19}", r"1 \pm \sqrt{19}", "exact_final_answer")[1] == 1.0
    assert score_output("x,y", "y,x", "exact_final_answer")[1] == 1.0


def test_live_routing_summary_counts_selective_consistency_probe_rate() -> None:
    rows = [
        {
            "query_id": "code:0",
            "query_text": "A Python function returns (3 * 2) + 4. What integer does it return?",
            "domain": "code",
            "model_id": "local",
            "quality_score": 0.0,
            "parsed_answer": "9",
            "cost_total_usd": 0.0,
            "is_local": True,
            "is_frontier": False,
            "status": "success",
            "latency_s": 0.1,
        },
        {
            "query_id": "code:0",
            "query_text": "A Python function returns (3 * 2) + 4. What integer does it return?",
            "domain": "code",
            "model_id": "gpt-5.5",
            "quality_score": 1.0,
            "parsed_answer": "10",
            "cost_total_usd": 0.01,
            "is_local": False,
            "is_frontier": True,
            "status": "success",
            "latency_s": 1.0,
        },
        {
            "query_id": "math:0",
            "query_text": "Compute 3 + 4. Return only the integer.",
            "domain": "math_easy",
            "model_id": "local",
            "quality_score": 1.0,
            "parsed_answer": "7",
            "cost_total_usd": 0.0,
            "is_local": True,
            "is_frontier": False,
            "status": "success",
            "latency_s": 0.1,
        },
        {
            "query_id": "math:0",
            "query_text": "Compute 3 + 4. Return only the integer.",
            "domain": "math_easy",
            "model_id": "gpt-5.5",
            "quality_score": 1.0,
            "parsed_answer": "7",
            "cost_total_usd": 0.01,
            "is_local": False,
            "is_frontier": True,
            "status": "success",
            "latency_s": 1.0,
        },
    ]

    summary = live_routing_summary(pd.DataFrame(rows), lambda_cost=0.0)
    policy = summary.set_index("method").loc["selective_code_consistency_rescue_gpt"]

    assert policy["probe_call_rate"] == 0.5
    assert policy["frontier_call_rate"] == 0.5
    assert policy["mean_quality"] == 1.0


def test_generate_stage0_tasks_from_manifest_filters_and_prompts(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "query_id": "aime:1",
                "query_text": "Find x.",
                "dataset": "aime",
                "domain": "math",
                "task_type": "math",
                "gold_answer": "42",
            },
            {
                "query_id": "math500:1",
                "query_text": "Compute y.",
                "dataset": "math500",
                "domain": "math",
                "task_type": "math",
                "gold_answer": "7",
            },
        ]
    ).to_csv(manifest, index=False)

    tasks = generate_stage0_tasks_from_manifest(manifest, datasets=["aime"], max_tasks=1)

    assert tasks["query_id"].tolist() == ["aime:1"]
    assert tasks["benchmark"].tolist() == ["aime"]
    assert tasks["metric"].tolist() == ["exact_final_answer"]
    assert "Return only the final answer" in tasks.loc[0, "query_text"]


def test_tool_augmented_policy_reports_quality_conservative_validation_selection() -> None:
    spec = importlib.util.spec_from_file_location(
        "tool_augmented_aime_policy",
        Path("experiments/112_tool_augmented_aime_policy.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows = [
        {
            "method": "min_cost",
            "split": "val",
            "mean_quality": 0.79,
            "quality_gap_to_strong_inclusive_oracle": 0.02,
            "normalized_cost_vs_all_strong": 0.10,
            "utility_ratio_to_strong_inclusive_oracle": 0.99,
            "strong_cost_cap": 0.025,
        },
        {
            "method": "quality_conservative",
            "split": "val",
            "mean_quality": 0.79,
            "quality_gap_to_strong_inclusive_oracle": 0.02,
            "normalized_cost_vs_all_strong": 0.20,
            "utility_ratio_to_strong_inclusive_oracle": 0.98,
            "strong_cost_cap": 0.05,
        },
        {
            "method": "min_cost",
            "split": "test",
            "mean_quality": 0.81,
            "quality_gap_to_strong_inclusive_oracle": 0.03,
            "normalized_cost_vs_all_strong": 0.11,
            "utility_ratio_to_strong_inclusive_oracle": 1.01,
            "strong_cost_cap": 0.025,
        },
        {
            "method": "quality_conservative",
            "split": "test",
            "mean_quality": 0.85,
            "quality_gap_to_strong_inclusive_oracle": 0.00,
            "normalized_cost_vs_all_strong": 0.21,
            "utility_ratio_to_strong_inclusive_oracle": 1.02,
            "strong_cost_cap": 0.05,
        },
    ]
    selected = module.select_rows(
        pd.DataFrame(rows),
        SimpleNamespace(quality_gap_target=0.03, cost_target=0.35, utility_ratio_target=0.95),
    )

    by_rule = selected.set_index("selection_rule")
    assert by_rule.loc["validation_feasible_min_cost", "method"] == "min_cost"
    assert by_rule.loc["validation_feasible_quality_conservative", "method"] == "quality_conservative"
    assert by_rule.loc["validation_feasible_quality_conservative_test", "mean_quality"] == 0.85


def test_budgeted_rescue_boolean_parser_handles_string_false() -> None:
    spec = importlib.util.spec_from_file_location(
        "budgeted_strong_rescue_policy",
        Path("experiments/116_budgeted_strong_rescue_policy.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.as_bool("False") is False
    assert module.as_bool("true") is True
    assert module.as_bool(False) is False


def test_extended_exact_math_tools_score_known_templates() -> None:
    cases = [
        (
            "Four unit squares form a $2 \\times 2$ grid. Each of the 12 unit line segments forming the sides of the squares is colored either red or blue in such a way that each unit square has 2 red sides and 2 blue sides. Find the number of such colorings.",
            "82",
        ),
        (
            "There are $8!=40320$ eight-digit positive integers that use each of the digits $1,2,3,4,5,6,7,8$ exactly once. Let $N$ be the number of these integers that are divisible by 22. Find the difference between $N$ and 2025.",
            "279",
        ),
        (
            "Let $b \\geq 2$ be an integer. Call a positive integer $n$ $b$\\textit{-eautiful} if it has exactly two digits when expressed in base $b$, and these two digits sum to $\\sqrt{n}$. Find the least integer $b \\geq 2$ for which there are more than ten $b$-eautiful integers.",
            "211",
        ),
        (
            "Let $ S $ be the set of vertices of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex in $ S $ is an endpoint of exactly one of the 12 segments.",
            "113",
        ),
        (
            "Let $p(x)$ be a polynomial of degree 5 such that \\[p(n) = \\frac{n}{n^2 - 1}\\]for $n = 2,$ 3, 4, $\\dots,$ 7. Find $p(8).$",
            r"\frac{3}{56}",
        ),
        (
            "Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$, inclusive, such that $12x^{2}-xy-6y^{2}=0$.",
            "117",
        ),
        (
            "The sequence $\\{a_n\\}$ satisfies $a_1=1$, and for any positive integer $n$, we have $a_{n+1}=10^{n}{a_n^2}$. What is the general term formula for $\\{a_n\\}$?",
            r"10^{2^n-n-1}",
        ),
        (
            "The binary number $10101001110_{2}$ is equal to what number in base eight?",
            "2516_8",
        ),
        (
            "Express $555_{10}$ in base $5$.",
            "4210_{5}",
        ),
        (
            "Let $z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i$, and let $c = 2 - 3i$. Let $w$ be the result when $z$ is rotated around $c$ by $\\frac{\\pi}{4}$ counter-clockwise. Find $w.$",
            "6 - 5i",
        ),
    ]
    for query, gold in cases:
        answer = deterministic_exact_math_answer(query)
        assert answer is not None
        assert score_output(answer, gold, "exact_final_answer")[1] == 1.0
