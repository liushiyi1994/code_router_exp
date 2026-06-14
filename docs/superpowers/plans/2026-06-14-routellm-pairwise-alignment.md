# RouteLLM Pairwise Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-API RouteLLM pairwise-substrate export for the RouteCode train/test split so official RouteLLM-MF/BERT can later be run on a split-aligned strong/weak comparison.

**Architecture:** Keep this separate from official baseline claims. Add a helper that converts prepared RouteCode matrices into strong/weak pairwise records, then add an experiment script that writes train/test JSON, metadata, a readiness table, a memo, and a README section. The table must clearly say this is split-aligned data substrate, not an official RouteLLM result.

**Tech Stack:** Python standard library, pandas, pytest, existing RouteCode config, pipeline, reporting, and external-baseline helpers.

---

### Task 1: Pairwise Substrate Helper

**Files:**
- Modify: `src/routecode/eval/external_baselines.py`
- Test: `tests/test_external_baseline_helpers.py`

- [ ] **Step 1: Write failing tests**

Add tests that build tiny train/test matrices with strong/weak models, call `build_routellm_pairwise_records`, and assert:

- records include `query_id`, `split`, `model_a`, `model_b`, utility/quality/cost fields, and `winner`;
- winners are `model_a`, `model_b`, or `tie`;
- train/test query IDs remain disjoint;
- missing strong/weak model columns raise a `ValueError`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_external_baseline_helpers.py::test_build_routellm_pairwise_records_exports_split_aligned_winners tests/test_external_baseline_helpers.py::test_build_routellm_pairwise_records_rejects_missing_pair_model -q`

Expected: FAIL because `build_routellm_pairwise_records` is not implemented.

- [ ] **Step 3: Implement helper**

Add `build_routellm_pairwise_records(matrices_by_split, pair, *, epsilon=1e-12)` returning `dict[str, list[dict]]`. Use `Matrices.utility`, `Matrices.quality`, `Matrices.cost`, and `Matrices.query_info`; do not inspect any held-out split while constructing another split's records.

- [ ] **Step 4: Run helper tests**

Run: `pytest tests/test_external_baseline_helpers.py -q`

Expected: PASS.

### Task 2: Pairwise Alignment Experiment

**Files:**
- Create: `experiments/14_routellm_pairwise_alignment.py`
- Modify: `experiments/02_rate_distortion_curve.py`
- Modify: `tests/test_readme_wiring.py`
- Test: `tests/test_routellm_pairwise_alignment_script.py`

- [ ] **Step 1: Write failing script/README tests**

Add a script test with a tiny synthetic config that asserts the script writes:

- `routellm_pairwise/pairwise_train.json`;
- `routellm_pairwise/pairwise_test.json`;
- `routellm_pairwise/metadata.json`;
- `table_routellm_pairwise_alignment.csv`;
- `phase_e_routellm_pairwise_alignment_memo.md`;
- a README section.

Update README wiring tests so real configs include script 14 and its outputs, while synthetic configs do not.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_routellm_pairwise_alignment_script.py tests/test_readme_wiring.py -q`

Expected: FAIL because the script and README wiring are not implemented.

- [ ] **Step 3: Implement script and wiring**

The script should prepare data from config, choose the configured strong/weak pair from train utility, export pairwise JSON for train/test, write metadata, summarize split alignment and winner distributions, upsert README, and write a memo with explicit non-claim language.

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/test_routellm_pairwise_alignment_script.py tests/test_readme_wiring.py -q`

Expected: PASS.

### Task 3: Generate Pilot Artifacts And Verify

**Files:**
- Generate: `results/llmrouterbench_pilot/routellm_pairwise/pairwise_train.json`
- Generate: `results/llmrouterbench_pilot/routellm_pairwise/pairwise_test.json`
- Generate: `results/llmrouterbench_pilot/routellm_pairwise/metadata.json`
- Generate: `results/llmrouterbench_pilot/table_routellm_pairwise_alignment.csv`
- Generate: `results/llmrouterbench_pilot/phase_e_routellm_pairwise_alignment_memo.md`
- Modify: `results/llmrouterbench_pilot/README.md`
- Modify: `results/llmrouterbench_pilot/research_flow_audit.md`
- Modify: `results/llmrouterbench_pilot/baseline_readiness_audit.md`

- [ ] **Step 1: Run the experiment**

Run: `python experiments/14_routellm_pairwise_alignment.py --config configs/llmrouterbench_pilot.yaml`

Expected: command exits 0 and writes the pairwise substrate artifacts.

- [ ] **Step 2: Update audits**

Update the research-flow and baseline-readiness audits to state that split-aligned RouteLLM pairwise substrate exists, but official MF/BERT evaluation remains incomplete.

- [ ] **Step 3: Verify**

Run:

```bash
pytest -q
python -m compileall -q src experiments tests
```

Expected: all commands exit 0.
