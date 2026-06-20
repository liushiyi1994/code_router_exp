from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "71_local_vllm_policy_pipeline.py"
    spec = importlib.util.spec_from_file_location("local_vllm_policy_pipeline", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_vllm_policy_pipeline_orchestrates_generation_to_policy(tmp_path, monkeypatch):
    module = _load_script()
    phase2 = tmp_path / "results/phase2"
    generation_dir = phase2 / "local_vllm_two_model_all200_nothink"
    config_path = tmp_path / "phase2_local_vllm_two_model.yaml"
    state_targets = tmp_path / "state_targets.csv"
    query_features = tmp_path / "query_features.csv"
    probe_features = tmp_path / "probe_features.parquet"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: local_vllm_two_model_all200_nothink",
                f"  output_dir: {generation_dir}",
                "phase2_local_eval:",
                "  openai_endpoints:",
                "    - base_url: http://localhost:8001/v1",
            ]
        ),
        encoding="utf-8",
    )
    state_targets.write_text("query_id,state_label,split\nq0,0,train\n", encoding="utf-8")
    query_features.write_text("query_id,x\nq0,1\n", encoding="utf-8")
    pd.DataFrame({"query_id": ["q0"]}).to_parquet(probe_features)
    calls = []

    def fake_script(name: str):
        calls.append(("load", name))
        if name == "58_local_server_readiness":
            return SimpleNamespace(
                run=lambda **_kwargs: pd.DataFrame(
                    {"status": ["ready"], "model_id": ["m0"], "base_url": ["http://localhost:8001/v1"]}
                )
            )
        if name == "51_true_model_generation_matrix":
            def run(config_path):
                calls.append(("generation", config_path))
                generation_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"query_id": ["q0"], "model_id": ["m0"], "quality": [1.0], "cost_proxy": [0.0]}).to_parquet(
                    generation_dir / "local_model_outcomes.parquet"
                )
                return pd.DataFrame()

            return SimpleNamespace(run=run)
        if name == "70_local_outcomes_policy_matrices":
            def run(**kwargs):
                calls.append(("matrices", kwargs["local_outcomes_path"]))
                out = Path(kwargs["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                for file_name in ["local_query_model_utility.csv", "local_state_model_utility.csv"]:
                    (out / file_name).write_text("query_id,m0\nq0,1\n", encoding="utf-8")
                return {
                    "query_model_utility": str(out / "local_query_model_utility.csv"),
                    "state_model_utility": str(out / "local_state_model_utility.csv"),
                }

            return SimpleNamespace(run=run)
        if name == "64_true_probe_policy_inputs":
            def run(**kwargs):
                calls.append(("inputs", kwargs["query_model_utility_path"]))
                out = Path(kwargs["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key in [
                    "before_beliefs",
                    "after_beliefs",
                    "state_model_utility",
                    "query_model_utility",
                    "probe_cost",
                    "predicted_gain",
                ]:
                    path = out / f"{key}.csv"
                    path.write_text("query_id,m0\nq0,1\n", encoding="utf-8")
                    paths[key] = str(path)
                return paths

            return SimpleNamespace(run=run)
        if name == "54_proberoute_policy":
            def run(**kwargs):
                calls.append(("policy", kwargs["query_model_utility_path"]))
                out = Path(kwargs["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                (out / "table_proberoute_policy.csv").write_text("policy,status\nnever_probe,executed\n", encoding="utf-8")
                (out / "fig_gap_closed_vs_probe_cost.pdf").write_bytes(b"%PDF-1.4\n")
                return pd.DataFrame()

            return SimpleNamespace(run=run)
        if name == "69_phase2_completion_audit":
            def run(**kwargs):
                calls.append(("audit", str(kwargs["output_dir"])))
                out = Path(kwargs["output_dir"])
                out.mkdir(parents=True, exist_ok=True)
                table = out / "table_phase2_completion_audit.csv"
                memo = out / "phase2_completion_audit.md"
                report = out / "PHASE2_EVIDENCE_REPORT.md"
                table.write_text("requirement_id,status\nx,complete\n", encoding="utf-8")
                memo.write_text("memo\n", encoding="utf-8")
                report.write_text("# report\n", encoding="utf-8")
                return {"table": str(table), "memo": str(memo), "report": str(report)}

            return SimpleNamespace(run=run)
        raise AssertionError(name)

    monkeypatch.setattr(module, "_script", fake_script)

    paths = module.run(
        config_path=str(config_path),
        phase2_dir=str(phase2),
        state_targets_path=str(state_targets),
        query_features_path=str(query_features),
        probe_features_path=str(probe_features),
        skip_readiness=False,
    )

    assert paths["local_outcomes"].endswith("local_model_outcomes.parquet")
    assert paths["policy_table"].endswith("table_proberoute_policy.csv")
    assert paths["audit_table"].endswith("table_phase2_completion_audit.csv")
    memo = phase2 / "local_vllm_two_model_all200_nothink_local_vllm_policy_pipeline_memo.md"
    assert memo.exists()
    assert "Status: `completed`" in memo.read_text(encoding="utf-8")
    assert ("generation", str(config_path)) in calls
    assert any(call[0] == "policy" for call in calls)


def test_local_vllm_policy_pipeline_stops_when_readiness_blocked(tmp_path, monkeypatch):
    module = _load_script()
    phase2 = tmp_path / "results/phase2"
    config_path = tmp_path / "phase2_local_vllm_two_model.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: blocked_run",
                f"  output_dir: {phase2 / 'blocked_run'}",
            ]
        ),
        encoding="utf-8",
    )

    def fake_script(name: str):
        if name == "58_local_server_readiness":
            return SimpleNamespace(run=lambda **_kwargs: pd.DataFrame({"status": ["blocked"]}))
        raise AssertionError(f"Should not load {name} after blocked readiness")

    monkeypatch.setattr(module, "_script", fake_script)

    try:
        module.run(
            config_path=str(config_path),
            phase2_dir=str(phase2),
            state_targets_path=str(tmp_path / "state.csv"),
            query_features_path=str(tmp_path / "query.csv"),
            probe_features_path=str(tmp_path / "probe.parquet"),
        )
    except SystemExit as exc:
        assert "Blocked" in str(exc)
    else:
        raise AssertionError("Expected blocked readiness SystemExit")
    memo = phase2 / "blocked_run_local_vllm_policy_pipeline_memo.md"
    assert memo.exists()
    assert "blocked_readiness" in memo.read_text(encoding="utf-8")
