# Starting Prompt for Codex: Controlled ProbeRoute++ Experiments

Paste this into Codex at the root of the repository.

```text
We are starting Phase 2 controlled experiments for ProbeRoute++, an LLM routing research project.

Read these files first:
1. CONTROLLED_EXPERIMENT_PLAN.md
2. CODEX_GOAL_CONTROLLED_EXPERIMENTS.md
3. COST_LATENCY_SCHEMA.md
4. MODEL_AND_BENCHMARK_NOTES.md
5. CONFIG_TEMPLATE_CONTROLLED.yaml

Goal:
Build the experiment pipeline and run a controlled AIMS-style evaluation of ProbeRoute++ using 4–5 modern local models and 2 frontier models across exact-scored benchmarks. The system must treat quality, remote API cost, frontier-call rate, and latency as first-class metrics.

Do not run a huge uncontrolled model sweep. Do not start with all 20/32 LLMRouterBench models. The actual run uses a controlled deployment model pool.

Core system:
query -> belief over latent route states -> optional cheap probe -> updated belief -> selected model by cost-aware state-to-model utility.

First tasks:
1. Create configs from CONFIG_TEMPLATE_CONTROLLED.yaml.
2. Implement model output cache and per-call cost/latency logging.
3. Implement benchmark sampling and scoring stubs for GSM8K, MATH500, AIME, HumanEval, MBPP, LiveCodeBench, GPQA, MMLU-Pro, optional BBH.
4. Implement dry-run with 5 examples per benchmark.
5. Implement model runners for local OpenAI-compatible servers and frontier API wrappers, but do not enable frontier calls until cost estimate is printed and allow_frontier_calls=true.
6. Implement cost-aware utility and latency-aware utility.
7. Implement A1 routability audit, A2 latent state frontier, A3 observability gap, and A4 probe feasibility.
8. After pilot, write results/controlled/PILOT_OBSERVATION_MEMO.md before running full method evaluation.

Hard constraints:
- Cache all model calls.
- Enforce max_frontier_spend_usd.
- Exact-score first; avoid LLM judges.
- Run local models sequentially if needed.
- Log input tokens, output tokens, cost, latency, and status for every call.
- Report p50/p95 latency and normalized remote cost.
- Do not make final claims until RUN_REPORT.md maps evidence to claims.

Deliverables:
results/controlled/model_outputs.parquet
results/controlled/scored_outputs.parquet
results/controlled/cost_latency_summary.csv
results/controlled/table_routability.csv
results/controlled/table_rate_distortion.csv
results/controlled/table_observability_gap.csv
results/controlled/table_main_eval.csv
results/controlled/table_calibration.csv
results/controlled/table_ablation.csv
results/controlled/table_sensitivity.csv
results/controlled/fig_quality_cost_frontier.pdf
results/controlled/fig_latency_breakdown.pdf
results/controlled/fig_rate_distortion.pdf
results/controlled/fig_observability_gap.pdf
results/controlled/fig_calibration_curve.pdf
results/controlled/PILOT_OBSERVATION_MEMO.md
results/controlled/RUN_REPORT.md
```


Expected-results rule: Treat `EXPECTED_RESULTS_AND_SUCCESS_CRITERIA.md` as claim gates. The target is within 3 absolute quality points of the cost-aware oracle, 0.15x--0.35x all-frontier remote cost, competitive p95 latency, and 3x--5x fewer new-model calibration evaluations. If actual results miss these targets, weaken the claim; do not overclaim SOTA.
