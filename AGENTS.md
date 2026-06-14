# AGENTS.md — Codex Instructions for RouteCode

This repository is for the RouteCode research project.

Codex should treat this file as the durable working agreement for all coding and research tasks.

---

## Mission

Build and evaluate **RouteCode**, a framework for learning explainable, utility-aware routing labels for LLM model routing.

The project asks:

> How many bits of query information does an LLM router need to choose a good model?

The core artifact is a routing rate--distortion curve and a RouteCode method that learns compact route labels optimized for model-selection utility.

---

## Core idea in plain language

A normal LLM router does:

```text
query -> selected model
```

RouteCode does:

```text
query -> route label -> selected model
```

The route label is a learned, explainable label for model-selection behavior. It is not just a topic label. It is learned from the query--model utility matrix.

Example:

```text
"Write binary search in Python"
  -> route_label = routine_code_generation
  -> selected_model = code_model_7B
```

---

## Non-negotiable framing

Do not frame the project as saving router tokens. That is too small.

Frame it as:

- measuring routing information requirements;
- learning minimal sufficient routing labels;
- improving model-pool transfer;
- reducing calibration examples for new models;
- diagnosing benchmark compressibility;
- optionally doing adaptive refinement for uncertain queries.

---

## Main claims to test

Main:

1. Small route labels recover most routing performance.
2. New models can be integrated with far fewer calibration examples.

Secondary:

3. Route labels transfer across model pools better than direct learned routers.

Diagnostic:

4. Routing benchmarks are compressible to different degrees; high compressibility may reveal coarse-domain artifacts.

Optional:

5. Adaptive refinement improves cost--quality by refining only ambiguous queries.

Do not claim all five unless results support them.

---

## Required benchmarks and baselines

Primary benchmark:

- LLMRouterBench if accessible.

Secondary:

- RouterBench.
- RouteLLM data/eval.
- LLMRouter library baselines.

Required baselines:

- random;
- cheapest;
- best single model;
- dataset oracle;
- query oracle;
- dataset-label lookup;
- predicted-topic lookup;
- embedding-cluster lookup;
- kNN;
- MLP/logistic classifier;
- RouteLLM MF/BERT if easy;
- GraphRouter/LLMRouter baselines if easy.

Do not block first pilot on complex baselines.

---

## Implementation phases

### Phase 0: repo skeleton

Create files and folders according to `PROJECT.md`. Use clear config-driven scripts.

### Phase 1: data loading

Goal: produce canonical `outcomes.parquet`.

Required columns:

```text
query_id
query_text
dataset
domain optional
model_id
quality
cost_input optional
cost_output optional
cost_total
latency optional
tokens_input optional
tokens_output optional
judge
metadata_json optional
```

### Phase 2: utility and oracle metrics

Implement:

- utility matrix `U = quality - lambda * normalized_cost`;
- best single model;
- oracle router;
- oracle gap;
- model-win entropy;
- per-domain routability.

### Phase 3: compression ladder

Implement:

- dataset-label router;
- predicted-topic router;
- embedding-cluster router;
- kNN router;
- simple learned router.

Report:

- recovered gap vs learned;
- recovered gap vs oracle;
- leakage gap;
- bootstrap confidence intervals.

### Phase 4: RouteCode

Implement:

- utility-vector clustering;
- regret-optimized codebook;
- predictability-constrained RouteCode;
- text-to-label predictor;
- code cards;
- rate--distortion curves.

### Phase 5: transfer

Implement:

- model holdout;
- new-model calibration with r examples per label;
- direct router retraining comparison under same budget.

### Phase 6: adaptive refinement

Only after prior phases work. Implement entropy/margin/VOI refinement.

### Phase 7: robustness

Run K, lambda, embeddings, predictors, splits, label noise, model-pool composition, and seed sweeps.

---

## Data leakage rules

Before reporting any result, verify:

- clusters fit on train only;
- codebooks learned on train only;
- dataset/domain best-model tables computed on train only;
- topic classifier trained on train only;
- K and thresholds selected on validation, not test;
- test results run once after method selection;
- dataset-label router described as diagnostic upper bound, not deployable baseline.

---

## Required metrics in every main result

- mean utility;
- accuracy/quality;
- normalized cost;
- oracle regret;
- recovered gap vs learned router;
- recovered gap vs oracle;
- code rate: `log2(K)` and empirical `H(Z)`;
- label extraction latency/cost;
- bootstrap CI over queries.

Transfer metrics:

- calibration examples per label;
- total model evaluations;
- performance vs direct router retraining under same calibration budget.

Interpretability metrics:

- top representative queries per label;
- label best model and second-best model;
- utility margin;
- dominant domains/datasets;
- high-regret failure examples.

---

## Offensive claim threshold

Only use a title/claim like “routers barely use fine-grained query information” if:

```text
predicted-topic or predicted-code router recovers >=85% of best learned-router gain
AND lower bootstrap CI >=80%
AND it recovers at least 50--60% of oracle gain
```

If recovery is 50--80%, write an information-frontier/decomposition paper, not the offensive paper.

---

## Local compute plan

Hardware: RTX 5090 with 32GB VRAM.

Use local inference only after matrix experiments work.

Preferred serving:

1. vLLM
2. llama.cpp / llama-cpp-python
3. SGLang

Start with embeddings + small classifiers. Do not begin with LoRA or API generation.

Recommended model classes:

- embeddings: MiniLM, BGE, Qwen embedding, ModernBERT embeddings;
- classifiers: logistic regression, MLP, ModernBERT/DeBERTa;
- optional generation/probe models: Qwen3-8B, Qwen Coder, Llama/Mistral/Gemma/Phi 7B--14B class.

---

## Code quality expectations

- Use Python 3.11+.
- Prefer simple, testable modules.
- Use config files for benchmark/model/predictor settings.
- Cache embeddings and processed matrices.
- Save every experiment config and seed.
- Use deterministic seeds where possible.
- Every script should write outputs under `results/` with timestamped subfolders.
- Add unit tests for utility, splits, oracle, codebook assignment, and metrics.

---

## First tasks for Codex

1. Create repo skeleton.
2. Create `pyproject.toml` with dependencies.
3. Implement canonical data schema dataclasses.
4. Implement `utility.py` and `metrics.py`.
5. Implement synthetic matrix generator for early testing.
6. Implement best single and oracle routers.
7. Implement dataset-label and embedding-cluster routers.
8. Implement first `00_data_audit.py` and `01_compression_ladder.py` scripts.
9. Add tests.
10. Produce a small synthetic demo result before downloading big data.

---

## Do not do yet

- Do not run API generation.
- Do not fine-tune LoRA.
- Do not implement adaptive refinement first.
- Do not over-optimize architecture.
- Do not make claims before bootstrap CIs and leakage checks.
- Do not present dataset-label routing as deployable.

---

## Update protocol

After each milestone, update:

- `PROJECT.md` if research framing changes;
- `CODEX_GOAL.md` if success criteria change;
- `EXPERIMENTS.md` if experiment plan changes;
- `results/README.md` with commands, configs, and outputs;
- `paper_notes.md` with supported and falsified claims.



## Reference links for agents

For papers, benchmark URLs, and open-source repos, always check `REFERENCES.md`. At minimum, keep these in mind while implementing baselines and benchmark loaders:

- LLMRouterBench: primary target benchmark.
- RouteLLM: canonical open-source router framework and baseline.
- LLMRouter: library of router implementations.
- RouterBench: secondary public routing benchmark.
- WebRouter and FineRouter: close novelty-boundary papers; do not claim first information bottleneck or first latent task discovery.
- GraphRouter, BEST-Route, Universal Model Routing: top-conference comparison/positioning references.
