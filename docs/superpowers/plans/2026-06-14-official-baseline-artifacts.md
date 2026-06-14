# Official Baseline Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-API Phase E artifact inspection path for official upstream RouteLLM MF result files from the local LLMRouterBench checkout.

**Architecture:** Keep official upstream artifacts separate from RouteCode split-aligned utility metrics. Parse the upstream JSON and CSV files into a compatibility-tagged table, then write a memo and README section that explain why these rows are evidence of baseline inspection rather than direct method-ranking results.

**Tech Stack:** Python standard library, pandas, pytest, existing `routecode.config` and `routecode.reporting` helpers.

---

### Task 1: Official RouteLLM Artifact Parser

**Files:**
- Modify: `src/routecode/eval/external_baselines.py`
- Test: `tests/test_external_baseline_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_official_routellm_artifacts_parses_json_and_seed_csvs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "mf_results_seed42.json").write_text(
        json.dumps(
            {
                "total": 10,
                "selection_accuracy": 0.7,
                "routing_accuracy": 0.6,
                "total_cost": 1.25,
                "datasets": {
                    "aime": {
                        "total": 2,
                        "selection_accuracy": 0.5,
                        "routing_accuracy": 1.0,
                        "total_cost": 0.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "mf_selection_accuracy_by_seed.csv").write_text(
        "seed,aime,sample_avg\n42,50.0,70.0\n",
        encoding="utf-8",
    )
    (results / "mf_total_cost_by_seed.csv").write_text(
        "seed,aime,total_cost\n42,0.2,1.25\n",
        encoding="utf-8",
    )

    table = load_official_routellm_artifacts(results)

    assert set(table["scope"]) == {"overall", "dataset"}
    assert table.loc[table["scope"].eq("overall"), "csv_selection_accuracy"].iloc[0] == 0.7
    assert table.loc[table["dataset"].eq("aime"), "csv_total_cost"].iloc[0] == 0.2
    assert not table["split_aligned_with_routecode"].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_external_baseline_helpers.py::test_load_official_routellm_artifacts_parses_json_and_seed_csvs -q`

Expected: FAIL because `load_official_routellm_artifacts` is not implemented.

- [ ] **Step 3: Write minimal implementation**

Add a parser that reads `mf_results_seed*.json`, adds one overall row and one row per dataset, enriches rows from optional seed CSVs, and marks rows as not RouteCode split-aligned.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_external_baseline_helpers.py::test_load_official_routellm_artifacts_parses_json_and_seed_csvs -q`

Expected: PASS.

### Task 2: Official Artifact Experiment Script

**Files:**
- Create: `experiments/12_official_baseline_artifacts.py`
- Modify: `experiments/02_rate_distortion_curve.py`
- Modify: `tests/test_readme_wiring.py`
- Test: `tests/test_official_baseline_artifacts_script.py`

- [ ] **Step 1: Write failing script and README wiring tests**

```python
def test_official_script_writes_table_memo_and_readme(tmp_path):
    ...
```

```python
assert "python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_pilot.yaml" in real_commands
assert "table_official_external_artifacts.csv" in real_outputs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_official_baseline_artifacts_script.py tests/test_readme_wiring.py -q`

Expected: FAIL because the script and README wiring are not implemented.

- [ ] **Step 3: Write minimal implementation**

Create the script, write `table_official_external_artifacts.csv`, write `phase_e_official_baseline_artifacts_memo.md`, and upsert a README section.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_official_baseline_artifacts_script.py tests/test_readme_wiring.py -q`

Expected: PASS.

### Task 3: Generate Pilot Artifacts

**Files:**
- Generate: `results/llmrouterbench_pilot/table_official_external_artifacts.csv`
- Generate: `results/llmrouterbench_pilot/phase_e_official_baseline_artifacts_memo.md`
- Modify: `results/llmrouterbench_pilot/README.md`

- [ ] **Step 1: Run the experiment**

Run: `python experiments/12_official_baseline_artifacts.py --config configs/llmrouterbench_pilot.yaml`

Expected: command exits 0 and writes the table and memo.

- [ ] **Step 2: Verify repo tests**

Run: `pytest -q`

Expected: all tests pass.
