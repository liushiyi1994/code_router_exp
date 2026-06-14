# Codex Goal: RouteCode Minimum Viable Research Repo

Use this file as the durable `/goal` specification for Codex.

## Goal

Build the minimum viable RouteCode research repository and run the first pilot experiment without external APIs.

## Research context

RouteCode studies LLM routing as a routing-information problem. A standard router maps:

```text
query -> selected model
```

RouteCode maps:

```text
query -> learned route label -> selected model
```

A route label is discrete, utility-aware, and explainable. It groups queries that have similar model-selection consequences under a quality-cost utility. It is not merely a semantic topic label, and it is not an uninterpretable dense vector.

## Core question

How many bits of query information does an LLM router need to choose a good model?

## Main deliverable for this goal

A working Python repo that runs a synthetic end-to-end pilot and produces:

- a data audit / oracle gap table;
- a compression ladder table;
- a rate--distortion curve for K route labels;
- bootstrap confidence intervals;
- basic plots;
- unit tests.

## Read before coding

1. `PROJECT.md` for the full research design.
2. `AGENTS.md` for operating rules and leakage controls.
3. `EXPERIMENTS.md` for concrete experiment definitions.
4. `METHOD_SPEC.md` for exact router input/output and objective.
5. `EXPERIMENTS.md` for concrete experiment definitions.
6. `CLAIMS_AND_EVALUATION.md` for claim thresholds and evidence requirements.
7. `DATA_AND_COST_PLAN.md` for data sizes and costs.
8. `REFERENCES.md` and `PAPERS_AND_BASELINES.md` for papers, benchmarks, and repos.

## Relevant papers/repos to keep in mind

Do not claim novelty until checking these links in `REFERENCES.md`:

- LLMRouterBench
- RouteLLM
- LLMRouter
- RouterBench
- WebRouter
- FineRouter
- Select-then-Route
- IRT-Router
- GraphRouter
- BEST-Route
- Universal Model Routing
- kNN routing paper
- Causal LLM Routing
- Router-R1

## Implementation tasks

1. Create package skeleton with `pyproject.toml`.
2. Implement a synthetic query--model outcome generator.
3. Define canonical data schema.
4. Compute utility matrix `U = quality - lambda * cost`.
5. Implement metrics:
   - mean utility;
   - oracle regret;
   - recovered gap vs learned router;
   - recovered gap vs oracle;
   - bootstrap CIs.
6. Implement routers:
   - best single;
   - oracle;
   - dataset-label lookup;
   - embedding-cluster lookup;
   - kNN;
   - flat RouteCode for `K = 1,2,4,8,16,32`.
7. Generate code cards for learned route labels.
8. Add experiment scripts:
   - `experiments/00_data_audit.py`;
   - `experiments/01_compression_ladder.py`;
   - `experiments/02_rate_distortion_curve.py`.
9. Add plots:
   - `fig_compression_ladder.pdf`;
   - `fig_rate_distortion.pdf`.
10. Add tests for utility, metrics, splits, oracle routing, dataset-label routing, and RouteCode assignment.

## Success criteria

The goal is complete when:

- `pytest` passes;
- the synthetic demo runs without API keys;
- `results/demo/table_routability.csv` exists;
- `results/demo/table_recovered_gap.csv` exists;
- `results/demo/table_rate_distortion.csv` exists;
- `results/demo/code_cards.md` exists;
- `results/demo/fig_compression_ladder.pdf` exists;
- `results/demo/fig_rate_distortion.pdf` exists;
- `results/demo/README.md` explains commands, outputs, and next steps.

## Hard constraints

- No external API calls.
- No GPT/Claude required.
- No LoRA/fine-tuning yet.
- No adaptive refinement yet.
- Do not touch paid APIs.
- Do not download huge models during the synthetic pilot.
- No train/test leakage: fit codebooks, clusters, lookup tables, scalers, and thresholds on train/validation only.
- Do not overclaim from synthetic data.

## Definition of done

After implementation, run the commands, fix failures, and summarize:

1. what was completed;
2. exact commands run;
3. where outputs were written;
4. first observed synthetic results;
5. next blocked/uncertain items before real benchmark work.
