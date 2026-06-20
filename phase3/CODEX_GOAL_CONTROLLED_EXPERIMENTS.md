# Codex Goal: Controlled ProbeRoute++ Phase 2 Experiments

Use this file with Codex `/goal` for a long-running experiment task.

```text
/goal Run the controlled ProbeRoute++ Phase 2 experiment plan.

Context:
We are working on a research project about LLM routing. The upgraded system is ProbeRoute++. It learns latent route states from model-utility patterns, predicts a belief over states from the query, runs cheap probes only when the route state is uncertain, and selects a model using a cost-aware state-to-model utility table. The system is designed to reduce frontier API usage, remote cost, latency exposure, and new-model calibration burden.

Important prior finding:
RouteCode Phase 1 found strong low-rate oracle routing structure, but deployable query-to-label prediction was weak. Oracle route labels reached near query-oracle performance, but predicted labels did not. Therefore Phase 2 must test a new claim: routing states are low-dimensional but partially observable, and cheap probes plus state-level calibration can make them useful.

Read first:
1. CONTROLLED_EXPERIMENT_PLAN.md
2. EXPECTED_RESULTS_AND_SUCCESS_CRITERIA.md
3. COST_LATENCY_SCHEMA.md
4. CONFIG_TEMPLATE_CONTROLLED.yaml
5. MODEL_AND_BENCHMARK_NOTES.md

Primary success targets:
- ProbeRoute++ within 3 absolute quality points of the cost-aware oracle.
- ProbeRoute++ reaches >=95--97% of cost-aware oracle utility.
- Normalized remote API cost <=0.15x--0.35x of all-frontier.
- p95 latency <= all-frontier p95 or <=1.2x all-frontier p95.
- Frontier-call rate <=25%--40%; probe rate <=20%--40%.
- Active state-level calibration uses 3x--5x fewer new-model evaluations than direct router retraining.
- No extensive router training: no LoRA/large-router fine-tuning in the main method; report training time and calibration calls.

Primary experiment philosophy:
Do not run a huge uncontrolled model sweep. Run a clean AIMS-style controlled deployment experiment:
- 4–5 latest local models;
- 2 frontier models;
- 8–9 exact-scored benchmarks;
- quality, remote API cost, frontier-call rate, local/probe latency, and p95 latency as first-class metrics;
- focused ablations and sensitivity.

Controlled model pool:
Local preferred:
- Qwen/Qwen3.5-0.8B as cheap probe/tiny local baseline
- Qwen/Qwen3.5-9B as general local model
- Qwen/Qwen3-Coder-30B-A3B-Instruct, quantized if needed, as code specialist
- Qwen/Qwen3.6-35B-A3B, quantized/FP8 if needed, as strong local general/reasoning model
- google/gemma-3-12b-it or mistralai/Mistral-Small-3.2-24B-Instruct-2506 as diverse non-Qwen model
Frontier:
- GPT-5.5 or latest GPT model available through API
- Claude Sonnet 4.6 or latest Claude Sonnet API model available

Benchmarks:
- GSM8K
- MATH500
- AIME
- HumanEval
- MBPP
- LiveCodeBench subset
- GPQA
- MMLU-Pro
- optional BBH/logical reasoning subset

Stage 0: dry run
Run 5 examples per benchmark across all selected models. Verify:
- local serving works or fallback is documented;
- frontier APIs work or are disabled safely;
- raw output cache works;
- parsing/scoring works;
- input/output tokens logged;
- cost estimated;
- latency p50/p95 computed.

Stage 1: pilot observations
Run about 100 examples per benchmark.
Implement and report:
A1 cost-aware routability audit;
A2 low-dimensional latent state frontier with K=4,8,16,32;
A3 observability gap using 2–3 strong query feature/predictor choices;
A4 cheap probe feasibility using Qwen3.5-0.8B and non-generative uncertainty features.

Stop after Stage 1 and write:
results/controlled/PILOT_OBSERVATION_MEMO.md
The memo must answer:
- Does cost-aware oracle gap exist?
- Do K=8/16 latent states recover most oracle gap?
- Is there an observability gap from query-only predictors?
- Do cheap probes improve state inference or utility?
- Should the paper continue as ProbeRoute++ or pivot to calibration-only?

Stage 2: main method evaluation
Only proceed if Stage 1 supports it.
Implement/evaluate:
- Best local
- All GPT frontier
- All Claude frontier
- Best single overall
- Query oracle
- Cost-aware oracle
- Dataset/domain lookup
- Embedding-cluster lookup
- kNN router
- Direct MLP/BERT router
- Confidence cascade
- ProbeRoute++ no-probe
- ProbeRoute++ threshold probe
- ProbeRoute++ VOI probe

Metrics:
- quality/exact match/pass@1
- cost-aware utility
- remote API cost per 1K queries
- normalized remote cost vs all-frontier
- frontier-call rate
- probe rate
- mean/p50/p95 latency
- cost at fixed quality
- quality at fixed cost

Stage 3: active new-model calibration
Hold out one model at a time:
- Qwen3-Coder or local code model
- Qwen3.6 or strong local model
- GPT frontier
- Claude frontier
Use K=8 and K=16 latent states.
Sweep examples per state r=4,8,16,32.
Compare:
- random calibration
- dataset-stratified calibration
- embedding-cluster calibration
- uniform latent-state calibration
- active latent-state calibration
- direct router retraining under same budget
Report performance vs number of new-model evaluations and calibration dollars.

Stage 4: ablations
Run focused ablations:
1. component ablation: full, w/o latent states, w/o probe, w/o VOI, w/o active calibration;
2. K sweep: 4,8,16,32;
3. probe policy: never, always, entropy threshold, margin threshold, VOI;
4. calibration budget: r=4,8,16,32.

Stage 5: sensitivity
Run focused sensitivity:
- lambda_cost = 0, low, medium, high;
- latency budget = 2s, 5s, 10s, 20s;
- frontier price multiplier = 0.5, 1, 2, 5;
- local speed multiplier = 1, 2, 4;
- held-out benchmark/domain generalization.

Required outputs:
configs/proberoute_controlled.yaml
configs/model_prices.yaml
configs/model_servers.yaml
configs/benchmark_sampling.yaml
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
results/controlled/RUN_REPORT.md
results/controlled/EXPECTED_RESULTS_STATUS.md

Hard constraints:
- Cache every model output.
- Never rerun a model call if cached output exists unless explicitly forced.
- Keep a running cost estimate before calling frontier APIs.
- Enforce a configurable max API spend.
- Use exact scoring where possible; avoid LLM judges in the first controlled run.
- Treat local remote-dollar cost as zero, but log local latency and optional GPU-time separately.
- Do not load all large local models simultaneously on one RTX 5090; run sequentially and cache.
- If a model cannot run, document the error and use the fallback pool.
- Do not make final paper claims until RUN_REPORT.md maps each claim to evidence.

Success criteria:
- Dry run completes without uncaught errors.
- Pilot observation memo is written.
- At least one pilot result table and three figures are produced.
- Cost and latency are logged for every model call.
- Main evaluation includes cost, latency, frontier-call rate, and quality.
- New-model calibration curves are produced if Stage 1 gates are passed.
- RUN_REPORT.md and EXPECTED_RESULTS_STATUS.md clearly state which numerical targets are met, missed, or partially supported. Do not claim SOTA unless the expected-results thresholds are met.
```
