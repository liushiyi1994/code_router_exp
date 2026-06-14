# Starting Prompt for Codex

Paste this into Codex at the root of the repository. Use `/goal` if available.

```text
/goal Build the minimum viable RouteCode research repository and run the first synthetic pilot experiment.

Context:
We are starting a PhD research project on LLM routing called RouteCode. A normal LLM router maps query -> selected model. RouteCode maps query -> route label -> selected model. A route label is a learned, discrete, explainable, utility-aware label for model-selection behavior. It is like a label for the query, but not a normal topic label. Labels are learned from query--model utility patterns and later explained with code cards.

Research question:
How many bits of query information does an LLM router need to choose a good model?

Core contribution:
Define and evaluate a routing rate--distortion framework. Learn minimal sufficient route labels that preserve model-selection utility under a cost-quality objective. Test whether small route labels recover most learned-router performance and enable cheap new-model calibration.

Important framing:
Do not frame this as saving router tokens. The main value is information structure, sample efficiency, model-pool transfer, benchmark diagnosis, and explainability.

Read these files first, in order:
1. AGENTS.md
2. START_HERE_FOR_CODEX.md
3. CODEX_GOAL.md
4. METHOD_SPEC.md
5. EXPERIMENTS.md
6. CLAIMS_AND_EVALUATION.md
7. DATA_AND_COST_PLAN.md
8. REFERENCES.md
9. PAPERS_AND_BASELINES.md
10. LOCAL_MODELS_AND_SERVING.md
11. PROJECT.md

Reference policy:
- Use REFERENCES.md and PAPERS_AND_BASELINES.md as the live bibliography and baseline/source list.
- Before making novelty claims, check WebRouter, FineRouter, RouteLLM, LLMRouterBench, GraphRouter, BEST-Route, Universal Model Routing, Causal LLM Routing, and the kNN routing paper.
- Before implementing baselines, inspect LLMRouterBench, LLMRouter, RouteLLM, RouterBench, and official repos linked in REFERENCES.md.
- If a paper/repo link is stale or missing, update REFERENCES.md before proceeding.
- Record which external repos/papers are used in results/demo/README.md.

Important paper/repo links to inspect before implementation or novelty claims:
- LLMRouterBench: https://arxiv.org/abs/2601.07206 and https://github.com/ynulihao/LLMRouterBench
- RouteLLM: https://github.com/lm-sys/routellm and https://arxiv.org/abs/2406.18665
- LLMRouter: https://github.com/ulab-uiuc/LLMRouter
- RouterBench: https://github.com/withmartian/routerbench and https://arxiv.org/abs/2403.12031
- WebRouter: https://arxiv.org/abs/2510.11221
- FineRouter: https://arxiv.org/abs/2603.19415
- BEST-Route: https://openreview.net/forum?id=tFBIbCVXkG and https://github.com/microsoft/best-route-llm
- GraphRouter: https://openreview.net/forum?id=eU39PDsZtT
- Universal Model Routing: https://openreview.net/pdf?id=ka82fvJ5f1
- kNN routing paper: https://arxiv.org/abs/2505.12601

First implementation goal:
Create a small, testable Python repo that can run a synthetic pilot end-to-end and produce the first compression ladder and rate--distortion curve.

Implement the following, in order:
1. repo/package skeleton with pyproject.toml;
2. configs/synthetic.yaml;
3. synthetic query--model outcome generator with latent domains, latent route labels, model skills, costs, and controllable residual interaction;
4. canonical data schema;
5. split by query_id, not by query-model row;
6. utility matrix U = quality - lambda * cost;
7. metrics: mean utility, oracle regret, recovered gap vs learned, recovered gap vs oracle, model-win entropy, bootstrap CI;
8. routers: best single, oracle, dataset-label lookup, embedding-cluster lookup, kNN;
9. flat RouteCode codebook for K = 1,2,4,8,16,32;
10. code cards for learned route labels;
11. evaluation scripts:
   - experiments/00_data_audit.py
   - experiments/01_compression_ladder.py
   - experiments/02_rate_distortion_curve.py
12. plots:
   - fig_compression_ladder.pdf
   - fig_rate_distortion.pdf
13. tests for utility, metrics, splits, oracle router, dataset-label router, and RouteCode assignment.

Success criteria:
- pytest passes;
- synthetic demo runs without API keys;
- results/demo/table_routability.csv exists;
- results/demo/table_recovered_gap.csv exists;
- results/demo/table_rate_distortion.csv exists;
- results/demo/code_cards.md exists;
- results/demo/fig_compression_ladder.pdf exists;
- results/demo/fig_rate_distortion.pdf exists;
- results/demo/README.md explains commands, outputs, references used, and next steps.

Hard constraints:
- no external API calls;
- no GPT/Claude required;
- no LoRA/fine-tuning yet;
- no adaptive refinement yet;
- no real benchmark download until synthetic demo passes;
- no data leakage: codebooks/clusters/tables fit on train only;
- do not overclaim from synthetic data.

After implementing, run the commands, fix failures, and summarize what was completed and what remains blocked.
```
