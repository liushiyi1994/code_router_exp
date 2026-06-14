# Curated Task Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add curated task-family and task-subtype metadata to the LLMRouterBench pilot and include it in Phase G domain-granularity sensitivity.

**Architecture:** Apply configured per-dataset taxonomy metadata during `prepare_llmrouterbench`, before query-id splitting and matrix construction. Reuse the existing domain-granularity sensitivity machinery by adding `task_family` and `task_subtype` to `domain_granularity_columns`.

**Tech Stack:** Python, pandas, PyYAML configs, pytest.

---

### Task 1: Pipeline Taxonomy Mapping

**Files:**
- Modify: `src/routecode/pipeline.py`
- Test: `tests/test_llmrouterbench_loader.py`

- [x] **Step 1: Write failing tests**

Added tests for fresh and cached LLMRouterBench preparation that require `task_family` and `task_subtype` columns to exist before splitting.

- [x] **Step 2: Verify RED**

Ran the focused tests and observed missing-column failures.

- [x] **Step 3: Implement mapper**

Added `_apply_task_taxonomy_map`, keyed by dataset name, preserving cached and freshly loaded outcomes.

- [x] **Step 4: Verify GREEN**

Ran the focused loader tests and confirmed the taxonomy columns reach train/val/test query info.

### Task 2: Sensitivity Config And Memo

**Files:**
- Modify: `configs/llmrouterbench_pilot.yaml`
- Modify: `configs/llmrouterbench.yaml`
- Modify: `experiments/09_sensitivity_suite.py`
- Test: `tests/test_sensitivity_suite_script.py`

- [x] **Step 1: Write failing memo test**

Added a test that requires the sensitivity memo to mention curated task-family/task-subtype taxonomy coverage.

- [x] **Step 2: Verify RED**

Ran the focused memo test and observed the old wording failure.

- [x] **Step 3: Update configs and memo wording**

Added configured taxonomy maps and included `task_family`/`task_subtype` in domain-granularity columns.

- [x] **Step 4: Regenerate outputs**

Ran `python experiments/09_sensitivity_suite.py --config configs/llmrouterbench_pilot.yaml`.

### Task 3: Verification

**Files:**
- Generated: `results/llmrouterbench_pilot/table_sensitivity_summary.csv`
- Generated: `results/llmrouterbench_pilot/phase_g_sensitivity_memo.md`
- Modify: `results/llmrouterbench_pilot/research_flow_audit.md`

- [ ] **Step 1: Run full tests**

Run: `pytest -q`

- [ ] **Step 2: Compile check**

Run: `python -m compileall -q src experiments tests`

- [ ] **Step 3: Artifact check**

Confirm `table_sensitivity_summary.csv` contains `task_family:*` and `task_subtype:*` domain-granularity rows.
