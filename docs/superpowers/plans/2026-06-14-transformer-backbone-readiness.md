# Transformer Backbone Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a no-download readiness artifact for transformer text-backbone baselines requested by Research Flow Phase F/G.

**Architecture:** Keep this as an audit/readiness artifact rather than a routing metric table. The scanner reads local Hugging Face cache metadata only, reports requested ModernBERT/DeBERTa models as missing when absent, classifies cached models by config architecture, and explains why no transformer embedding/direct-router baseline is executed unless a suitable local encoder checkpoint exists.

**Tech Stack:** Python standard library, pandas, pytest, existing `routecode.config` and `routecode.reporting` helpers.

---

### Task 1: Cache Scanner

**Files:**
- Create: `src/routecode/eval/transformer_backbones.py`
- Test: `tests/test_transformer_backbone_readiness.py`

- [ ] **Step 1: Write the failing test**

```python
def test_inspect_transformer_backbone_cache_marks_missing_requested_and_cached_causal_lm(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transformer_backbone_readiness.py -q`

Expected: FAIL because the scanner module does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement `inspect_transformer_backbone_cache(cache_dir, requested_model_ids, max_runnable_gb)` and return a table with `model_id`, `cache_status`, `runnable_as_encoder_baseline`, `reason`, `architecture`, `model_type`, `size_gb`, and `local_path`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transformer_backbone_readiness.py -q`

Expected: PASS.

### Task 2: Readiness Experiment

**Files:**
- Create: `experiments/13_transformer_backbone_readiness.py`
- Modify: `experiments/02_rate_distortion_curve.py`
- Modify: `tests/test_readme_wiring.py`
- Test: `tests/test_transformer_backbone_readiness_script.py`

- [ ] **Step 1: Write failing script and README wiring tests**

The script test should create a fake HF cache with one cached CausalLM config and one missing requested encoder, then assert the table, memo, and README section are written.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transformer_backbone_readiness_script.py tests/test_readme_wiring.py -q`

Expected: FAIL because the script and README wiring are missing.

- [ ] **Step 3: Write minimal implementation**

Create the experiment script and add it to the real-data README command/output wiring only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transformer_backbone_readiness_script.py tests/test_readme_wiring.py -q`

Expected: PASS.

### Task 3: Generate Pilot Artifacts

**Files:**
- Generate: `results/llmrouterbench_pilot/table_transformer_backbone_readiness.csv`
- Generate: `results/llmrouterbench_pilot/phase_f_g_transformer_backbone_readiness_memo.md`
- Modify: `results/llmrouterbench_pilot/README.md`
- Modify: `results/llmrouterbench_pilot/research_flow_audit.md`

- [ ] **Step 1: Run the experiment**

Run: `python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_pilot.yaml`

Expected: command exits 0 and writes the readiness table and memo without downloading models.

- [ ] **Step 2: Verify repo tests**

Run: `pytest -q`

Expected: all tests pass.
