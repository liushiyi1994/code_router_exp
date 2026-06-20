# Phase 3 Agent Handoff

Last updated: 2026-06-20

This is the current handoff for the RouteCode / ProbeCode Phase 3 work. It is
written for a new agent entering the repo after the synthetic and Phase 2 work.

## Current Bottom Line

The currently supported method is:

```text
query + cheap local/verifiable behavior
  -> learned verifiability / probe state
  -> cost-aware action
```

The strongest supported claim is not that clean query-only routing is solved.
The supported claim is narrower:

```text
ProbeCode / ProbeRoute++ reaches the configured oracle-level target on cached
Broad100 and controlled exact-math when the action pool includes learned
verifiability states plus verifiable local/tool actions.
```

Do not claim a clean no-tool router reaches the full oracle target. The no-tool
action-pool oracle itself misses the full-action-pool oracle target.

## Read This First

For current Phase 3 state, read in this order:

1. `AGENTS.md`
2. `phase3/README.md`
3. `phase3/BENCHMARK_AGNOSTIC_PROBE_STATE_PLAN.md`
4. `results/controlled/phase3_final_claim_package/PHASE3_FINAL_CLAIM_PACKAGE.md`
5. `results/controlled/phase3_oracle_level_modification/ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md`
6. `results/controlled/PHASE3_GOAL_COMPLETION_AUDIT.md`
7. `results/controlled/RUN_REPORT.md`
8. `paper_notes.md`

For the next final-evaluation run, use:

- `phase3/PHASE3_FINAL_EVALUATION_GOAL.md`

Older starter files such as `START_HERE_FOR_CODEX.md` still describe the
original synthetic pilot, but this handoff is the better entry point for the
current Phase 3 state.

## Data Flow

The core flow is:

```text
configs
  -> task/query manifest
  -> model/action outputs
  -> scoring
  -> cost and latency accounting
  -> utility matrix
  -> oracle and baselines
  -> learned/probe states
  -> validation-selected policy
  -> held-out test report
  -> final claim package
```

Concrete files:

- Configs:
  - `configs/proberoute_controlled.yaml`
  - `configs/proberoute_controlled_broad100.yaml`
  - `configs/model_prices.yaml`
  - `configs/model_servers.yaml`
  - `configs/benchmark_sampling.yaml`
- Main controlled code:
  - `src/routecode/controlled/surrogate.py`
  - `src/routecode/controlled/live_stage0.py`
  - `src/routecode/controlled/costing.py`
  - `src/routecode/controlled/exact_math_tools.py`
  - `src/routecode/controlled/code_scoring.py`
- Main experiment/package scripts:
  - `experiments/80_controlled_surrogate_pilot.py`
  - `experiments/81_controlled_live_stage0.py`
  - `experiments/119_phase3_exact_math_summary.py`
  - `experiments/120_phase3_exact_math_calibration.py`
  - `experiments/121_phase3_exact_math_ablation_sensitivity.py`
  - `experiments/122_phase3_goal_completion_audit.py`
  - `experiments/216_broad100_current_best_method_package.py`
  - `experiments/217_broad100_no_tool_feasibility_bound.py`
  - `experiments/218_phase3_final_claim_package.py`
  - `experiments/221_phase3_oracle_level_modification_summary.py`

## What A Row Means

The canonical unit is usually one query-action row:

```text
query_id
query_text
benchmark / dataset
split
model_id or action_id
raw answer
parsed answer
quality
cost
latency
cache_hit
```

Rows are cached under `results/controlled/` as Parquet, CSV, and raw JSON.
Policies aggregate these rows into one selected action per query.

## Utility Objective

The main objective is cost-aware utility:

```text
U(q, a) = quality(q, a) - lambda_cost * normalized_cost(q, a)
```

Some controlled reports also track latency:

```text
utility_cost_latency_aware =
  quality - lambda_cost * normalized_cost - lambda_latency * normalized_latency
```

The current controlled config uses:

```text
lambda_cost = 0.35
lambda_latency = 0.05
```

Cost is part of the method, not an after-the-fact report.

## Main Metrics

The important metrics are:

- `mean_quality`: average correctness/score of selected actions.
- `mean_utility`: average cost-aware utility.
- `quality_gap_to_oracle`: oracle quality minus policy quality.
- `oracle_utility_ratio`: policy utility divided by oracle utility.
- `oracle_regret`: oracle utility minus policy utility.
- `frontier_call_rate`: fraction of queries routed to GPT/Gemini/frontier action.
- `large_call_rate`: fraction routed to the large-action side of a local-vs-large abstraction.
- `probe_rate`: fraction requiring a probe action before final routing.
- `normalized_remote_cost`: remote cost normalized to the configured reference.
- `p95_latency_ratio_vs_all_gpt`: latency compared with all-GPT reference.
- `rate_log2K` and `empirical_H_Z`: code rate / route-state size.
- Calibration metrics: target-model evaluations and quality/utility under a fixed evaluation budget.

The configured oracle-level gate is:

```text
quality gap <= 0.03
oracle utility ratio >= 0.95
frontier call rate <= 0.40
```

Some final summaries also report the stricter `>= 0.97` utility-ratio check.

## What Was Tested

### 1. Controlled surrogate pilot

Purpose: validate schema, routing math, cost accounting, latency accounting,
figures, and claim gates before live generation.

Command:

```bash
PYTHONPATH=src python experiments/80_controlled_surrogate_pilot.py --config configs/proberoute_controlled.yaml
```

Main outputs:

- `results/controlled/model_outputs.parquet`
- `results/controlled/scored_outputs.parquet`
- `results/controlled/table_main_eval.csv`
- `results/controlled/table_rate_distortion.csv`
- `results/controlled/table_observability_gap.csv`
- `results/controlled/RUN_REPORT.md`

### 2. Live Stage 0 / cached controlled model calls

Purpose: collect or reuse local vLLM/frontier cached outputs on controlled
tasks, with no Claude/Anthropic in runnable configs.

Main script:

```bash
PYTHONPATH=src python experiments/81_controlled_live_stage0.py
```

Important outputs:

- `results/controlled/live_stage0/`
- `results/controlled/live_broad_stage0/`
- `results/controlled/raw_outputs/`
- `results/controlled/vllm_server_logs/`

### 3. Exact-math method package

Purpose: test whether verifiable local/tool actions can route exact-answer
math cheaply while staying near oracle.

Benchmarks/slice:

```text
AIME
LiveMathBench
MATH500
66 held-out mixed exact-math test queries
```

Key result:

```text
method: exact_math_tool_augmented_min_cost
quality gap: 0.0152
oracle utility ratio: 0.9739
frontier call rate: 0.1061
normalized remote cost: 0.0463
p95 latency ratio vs all GPT: 0.4799
```

Main outputs:

- `results/controlled/table_phase3_exact_math_main_eval.csv`
- `results/controlled/table_phase3_exact_math_calibration.csv`
- `results/controlled/table_phase3_exact_math_ablation.csv`
- `results/controlled/table_phase3_exact_math_sensitivity.csv`
- `results/controlled/PHASE3_EXACT_MATH_SUMMARY.md`

### 4. Broad100 current-best package

Purpose: test a broader cached benchmark slice with learned verifiability and
verifiable local/tool actions.

Held-out test size:

```text
172 Broad100 test queries
```

Current best method:

```text
et_flip_leaf4_thr0.8502_capNone
```

Key result:

```text
oracle quality: 0.8721
method quality: 0.8547
quality gap: 0.0174
oracle utility ratio: 0.9735
frontier call rate: 0.1919
```

More compact RouteCode-style state policy:

```text
method: gb_depth2_thr0.9844_state_k8
quality gap: 0.0233
oracle utility ratio: 0.9614
frontier call rate: 0.2384
```

Main outputs:

- `results/controlled/broad100_current_best_method_package/BROAD100_CURRENT_BEST_METHOD_PACKAGE.md`
- `results/controlled/broad100_current_best_method_package/table_broad100_current_best_main_eval.csv`
- `results/controlled/broad100_current_best_method_package/table_broad100_current_best_summary.csv`
- `results/controlled/broad100_current_best_method_package/table_broad100_current_best_action_mix.csv`

### 5. No-tool feasibility bound

Purpose: test whether the same routing problem can be solved without
deterministic/verifiable local/tool actions.

Key negative result:

```text
no-tool oracle quality gap to full oracle: 0.0465
no-tool oracle utility ratio to full oracle: 0.9338
```

Interpretation: even a perfect router over the restricted no-tool action pool
misses the full-action-pool oracle target.

Main outputs:

- `results/controlled/broad100_no_tool_feasibility_bound/NO_TOOL_FEASIBILITY_BOUND_MEMO.md`
- `results/controlled/broad100_no_tool_feasibility_bound/table_no_tool_feasibility_bound.csv`

### 6. GPT-strong residual repair

Purpose: test whether simply calling GPT-5.5 on residual failures fixes the
problem.

Key result:

```text
force GPT-5.5 strong-solve on 8 residual rows:
quality gap: 0.0000
oracle utility ratio: 0.9234
```

Interpretation: quality can be fixed by expensive GPT calls, but the
cost-aware utility target fails.

Main outputs:

- `results/controlled/broad100_gpt_strong_residual2048/`
- `results/controlled/phase3_oracle_level_modification/ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md`

### 7. Final claim package

Purpose: combine the supported, scoped, and negative results into one current
claim ledger.

Command:

```bash
PYTHONPATH=src python experiments/218_phase3_final_claim_package.py
```

Main outputs:

- `results/controlled/phase3_final_claim_package/PHASE3_FINAL_CLAIM_PACKAGE.md`
- `results/controlled/phase3_final_claim_package/table_phase3_final_claims.csv`
- `results/controlled/phase3_final_claim_package/table_phase3_final_method_evidence.csv`
- `results/controlled/phase3_final_claim_package/table_phase3_final_requirement_snapshot.csv`

## Models Used In Current Controlled Artifacts

Frontier / closed-source:

- `gpt-5.5`
- `gemini-3.5-flash`

Local / open or local rows:

- Qwen-family vLLM/cache rows, including Qwen3 4B/8B/14B/32B variants in
  controlled broad artifacts.
- Other cached local rows in earlier controlled artifacts include Gemma and
  Qwen coder/probe variants.

Claude/Anthropic:

- Not used in current runnable controlled configs or final Phase 3 artifacts.

## Where Progress Is Stored

The progress record is spread across these places:

- Current single-source claim summary:
  - `results/controlled/phase3_final_claim_package/PHASE3_FINAL_CLAIM_PACKAGE.md`
- Current oracle-level modification summary:
  - `results/controlled/phase3_oracle_level_modification/ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md`
- Completion audit:
  - `results/controlled/PHASE3_GOAL_COMPLETION_AUDIT.md`
  - `results/controlled/table_phase3_goal_completion_audit.csv`
- Run report and command/output history:
  - `results/controlled/RUN_REPORT.md`
  - `results/README.md`
- Paper-facing conservative interpretation:
  - `paper_notes.md`
- Phase 3 plan and success targets:
  - `phase3/`
- Raw/cached outputs:
  - `results/controlled/raw_outputs/`
  - `results/controlled/*/model_outputs.parquet`
  - `results/controlled/*/scored_outputs.parquet`
- vLLM logs:
  - `results/controlled/vllm_server_logs/`

## Verification Commands

Common no-call verification:

```bash
PYTHONPATH=src python -m py_compile \
  experiments/122_phase3_goal_completion_audit.py \
  experiments/218_phase3_final_claim_package.py \
  experiments/221_phase3_oracle_level_modification_summary.py

PYTHONPATH=src pytest -q \
  tests/test_controlled_phase3.py \
  tests/test_phase2_completion_audit_script.py
```

Recently verified result:

```text
19 passed, 2 warnings
```

Regenerate current final package:

```bash
PYTHONPATH=src python experiments/216_broad100_current_best_method_package.py
PYTHONPATH=src python experiments/217_broad100_no_tool_feasibility_bound.py
PYTHONPATH=src python experiments/122_phase3_goal_completion_audit.py
PYTHONPATH=src python experiments/218_phase3_final_claim_package.py
PYTHONPATH=src python experiments/221_phase3_oracle_level_modification_summary.py
```

These package scripts are cache-backed/no-call unless their dependencies have
been changed.

## Main Caveat

The working method depends on verifiability/action-pool design. The clean
no-tool version is a negative diagnostic, not a solved result. Future paper
language should keep that distinction explicit.

## Next Research Move

Broaden the verifiability/action-pool method into the final evaluation package
specified in `phase3/PHASE3_FINAL_EVALUATION_GOAL.md`. Keep the no-tool line as
a feasibility bound unless a new action pool or stronger observable probe
changes the bound.
