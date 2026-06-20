# Start Here for Codex

This is the first file Codex should read after `AGENTS.md`.

## Project in one sentence

Learn explainable, utility-aware route labels for LLM routing and measure how many bits of query information are needed to select a good model.

## First goal

Build a synthetic end-to-end demo before touching real LLM benchmarks.

## Current Phase 3 handoff

If this checkout already contains Phase 3 controlled results, read
`PHASE3_AGENT_HANDOFF.md` after `AGENTS.md`. It summarizes the current
ProbeCode / ProbeRoute++ method, data flow, metrics, result artifacts, and
claim caveats.

## Files to read in order

1. `AGENTS.md`
2. `CODEX_GOAL.md`
3. `METHOD_SPEC.md`
4. `EXPERIMENTS.md`
5. `REFERENCES.md`
6. `PAPERS_AND_BASELINES.md`
7. `DATA_AND_COST_PLAN.md`
8. `CLAIMS_AND_EVALUATION.md`
9. `LOCAL_MODELS_AND_SERVING.md`
10. `PROJECT.md`

## First coding tasks

Create:

```text
pyproject.toml
src/routecode/
experiments/
tests/
results/demo/
```

Implement:

```text
synthetic data generator
utility matrix builder
train/val/test split by query
best single model router
oracle router
dataset-label lookup router
embedding-cluster lookup router
kNN router
flat RouteCode codebook
metrics and bootstrap CIs
rate--distortion curve plot
```

## Do not do yet

```text
no API calls
no LLMRouterBench download if synthetic demo not working
no LoRA
no GPT/Claude batches
no adaptive refinement
```

## Completion condition

The first goal is complete only when:

```text
pytest passes
python experiments/00_data_audit.py --config configs/synthetic.yaml works
python experiments/01_compression_ladder.py --config configs/synthetic.yaml works
python experiments/02_rate_distortion_curve.py --config configs/synthetic.yaml works
results/demo/*.csv exist
results/demo/*.pdf exist
results/demo/README.md explains what happened
```
