# RouteCode References, Links, Repos, and Novelty Boundaries

This file is a first-class input for Codex/agents. Before making novelty claims, choosing baselines, or implementing external comparisons, inspect and update this file.

Last updated: 2026-06-19

---

## 1. Primary benchmark/data sources

### LLMRouterBench — primary benchmark
- Paper: https://arxiv.org/abs/2601.07206
- Code/data: https://github.com/ynulihao/LLMRouterBench
- Why we care: recent large unified LLM routing benchmark; >400K instances, 21 datasets, 33 models, 10 representative baselines; supports performance-only and performance-cost routing.
- First Codex action: write a loader that converts its data to `outcomes.parquet`.
- Use for: rate--distortion curves, baseline comparison, model-pool transfer, benchmark compressibility.

### RouterBench — secondary benchmark
- Paper/OpenReview: https://openreview.net/forum?id=IVXmV8Uxwh
- Paper/arXiv: https://arxiv.org/abs/2403.12031
- Code/data: https://github.com/withmartian/routerbench
- Why we care: established routing benchmark with >405K inference outcomes.
- Use for: replication after LLMRouterBench.

### RouteLLM — canonical routing framework and data
- Paper/arXiv: https://arxiv.org/abs/2406.18665
- OpenReview: https://openreview.net/forum?id=8sSqNntaMr
- Code: https://github.com/lm-sys/routellm
- Project page: https://sky.cs.berkeley.edu/project/routellm/
- Why we care: canonical preference-data routing framework; open source; provides trained routers and evaluation tools.
- Use for: RouteLLM-MF/BERT baselines and binary strong/weak routing experiments.

### LLMRouter — open-source baseline library
- Code: https://github.com/ulab-uiuc/LLMRouter
- Docs: https://ulab-uiuc.github.io/LLMRouter/
- Why we care: open-source implementation hub for many routers.
- Candidate methods: kNN, SVM, MLP, Matrix Factorization, Elo, RouterDC, AutoMix, HybridLLM, GraphRouter, causal-LM router, etc.
- First Codex action: inspect supported routers and build adapters to our canonical schema.

---

## 2. Open-source baselines to implement or call

### Required simple baselines — implement ourselves
- Random router
- Cheapest model
- Best single model
- Dataset-label lookup
- Predicted-topic lookup
- Embedding-cluster lookup
- kNN router
- Oracle router

### RouteLLM baselines
- RouteLLM matrix factorization router
- RouteLLM BERT router
- RouteLLM causal LLM router if practical
- RouteLLM Elo / weighted Elo if exposed
- Repo: https://github.com/lm-sys/routellm

### LLMRouter baselines
- kNN router: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/knnrouter/README.md
- SVM router: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/svmrouter/README.md
- MLP router: inspect LLMRouter repo
- GraphRouter: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/graphrouter/README.md
- Library: https://github.com/ulab-uiuc/LLMRouter

### GraphRouter — ICLR 2025
- OpenReview: https://openreview.net/forum?id=eU39PDsZtT
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/41b6674c28a9b93ec8d22a53ca25bc3b-Abstract-Conference.html
- arXiv: https://arxiv.org/abs/2410.03834
- Code path: likely via LLMRouter first.
- Role: structured graph-based router baseline; useful for model-pool generalization comparison.

### Avengers-Pro — cluster-based performance-efficiency routing
- arXiv: https://arxiv.org/abs/2508.12631
- Code: https://github.com/ZhangYiqun018/AvengersPro
- LLMRouterBench source path inspected locally: `data/raw/external/LLMRouterBench/baselines/AvengersPro`
- Role: cluster-routing performance/cost baseline; useful as a split-aligned local compatibility baseline before exact upstream command-path reproduction.

### BEST-Route — ICML 2025
- arXiv: https://arxiv.org/abs/2506.22716
- OpenReview: https://openreview.net/forum?id=tFBIbCVXkG
- Code: https://github.com/microsoft/best-route-llm
- Role: adaptive routing with model + best-of-n/sample-count action. Optional unless our experiment includes test-time compute actions.

### IRT-Router — ACL 2025
- Paper: https://aclanthology.org/2025.acl-long.761/
- Code: https://github.com/Mercidaiha/IRT-Router
- Role: optional difficulty/model-ability interpretability baseline.

---

## 3. Close novelty-boundary papers

These are the papers most likely to make reviewers ask, “Isn’t this already done?” Always distinguish RouteCode from these.

### WebRouter — cost-aware VIB router
- arXiv: https://arxiv.org/abs/2510.11221
- HTML: https://arxiv.org/html/2510.11221v1
- Why close: uses a cost-aware Variational Information Bottleneck for LLM routing in web agents.
- Novelty warning: do **not** claim first information-bottleneck router.
- Our distinction: RouteCode studies routing rate--distortion curves and discrete explainable route labels; WebRouter is a compressed neural router for web-agent prompts.

### FineRouter — latent task discovery
- arXiv: https://arxiv.org/abs/2603.19415
- HTML: https://arxiv.org/html/2603.19415v1
- Why close: discovers latent task types and uses task-aware quality estimation.
- Novelty warning: do **not** claim first latent-task discovery for routing.
- Our distinction: RouteCode asks for the minimal query information/labels needed for routing and evaluates transfer/calibration/compressibility.

### Select-then-Route — taxonomy-guided routing
- Paper: https://aclanthology.org/2025.emnlp-industry.28/
- Role: taxonomy-guided routing reference and possible hierarchical baseline.
- Our distinction: learned utility-aware route labels, not fixed taxonomy selection.

### Universal Model Routing — ICLR 2026
- OpenReview/PDF: https://openreview.net/pdf?id=ka82fvJ5f1
- arXiv: https://arxiv.org/abs/2502.08773
- Role: model-pool transfer and unseen model routing.
- Our distinction: they focus on model-side representations; RouteCode focuses on query-side route labels and code-to-model recalibration.

### Rethinking Predictive LLM Routing / kNN paper
- arXiv: https://arxiv.org/abs/2505.12601
- OpenReview: https://openreview.net/forum?id=Chn50flK4X
- Role: motivation for why simple/local routers may be strong.
- Our distinction: explain and measure why simple routing works via rate--distortion and route labels.

### LLMRouterBench — benchmark diagnosis prior
- arXiv: https://arxiv.org/abs/2601.07206
- Code/data: https://github.com/ynulihao/LLMRouterBench
- Role: both primary benchmark and motivation; reports similar performance across many routers and persistent model-recall failures.

## 3A. Probe-signal and SLM/LLM gap papers

These are for Phase 3 probe research: finding cheap early signals that predict when local/small models differ from stronger local or frontier models in a utility-improving way.

### LLMs Encode Their Failures - pre-generation activation success prediction
- arXiv: https://arxiv.org/abs/2602.09924
- Code: https://github.com/KabakaWilliam/llms_know_difficulty
- Role: strongest direct motivation for frozen pre-generation activation probes that predict model success before generation.

### LLM Router: Rethinking Routing with Prefill Activations
- arXiv: https://arxiv.org/abs/2603.20895
- Role: uses open-weight encoder/prefill activations to estimate target-model correctness, including closed-source targets.

### No Answer Needed - question-only correctness probes
- arXiv: https://arxiv.org/abs/2509.10625
- Role: tests whether hidden states after reading the question, before answer generation, predict model correctness.

### Query-Level Uncertainty in LLMs
- arXiv: https://arxiv.org/abs/2506.09669
- Code: https://github.com/tigerchen52/query_level_uncertainty
- Role: non-training internal-confidence signal for deciding whether a model can address a query before generating tokens.

### R2R - small-large token divergence routing
- OpenReview: https://openreview.net/forum?id=DpeJYRFRQY
- arXiv: https://arxiv.org/abs/2505.21600
- Code: https://github.com/thu-nics/R2R
- Role: studies where SLM and LLM reasoning paths diverge; useful for SLM-vs-medium divergence signals and path-divergence route labels.

### Fast and Slow Generating - SLM/LLM collaborative decoding
- arXiv: https://arxiv.org/abs/2406.12295
- Code: https://github.com/TsinghuaC3I/FS-GEN
- Role: analyzes differential knowledge between SLMs and LLMs and motivates uncertainty-based collaboration points.

### Semantic Entropy and Semantic Entropy Probes
- Semantic Entropy: https://www.nature.com/articles/s41586-024-07421-0
- Semantic Entropy Probes: https://arxiv.org/abs/2406.15927
- SEP code: https://github.com/OATML/semantic-entropy-probes
- Role: use sampled meaning-level uncertainty as a non-training signal, then optionally distill it into cheap hidden-state probes.

### PredictaBoard - predictability diagnosis
- Paper: https://aclanthology.org/2025.findings-acl.790/
- Code: https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard
- Role: evaluate whether failures are predictable, not only whether average model accuracy is high.

### Speculative Cascades / Faster Cascades via Speculative Decoding
- arXiv: https://arxiv.org/abs/2405.19261
- Google Research blog: https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/
- Role: combines cascade deferral and speculative verification; useful for SLM/LLM disagreement and early deferral rules.

### Learning to Decode Collaboratively with Multiple Language Models
- ACL 2024: https://aclanthology.org/2024.acl-long.701/
- Role: token-level latent model collaboration; useful conceptual precedent for sub-query and reasoning-step routing.

---

## 4. Top-conference papers to cite for paper writing

- RouteLLM, ICLR 2025: https://openreview.net/forum?id=8sSqNntaMr
- BEST-Route, ICML 2025: https://openreview.net/forum?id=tFBIbCVXkG
- A Unified Approach to Routing and Cascading for LLMs, ICML 2025: https://openreview.net/forum?id=AAl89VNNy1
- GraphRouter, ICLR 2025: https://openreview.net/forum?id=eU39PDsZtT
- Avengers-Pro, arXiv 2025: https://arxiv.org/abs/2508.12631
- Capability Instruction Tuning, AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/34790
- Causal LLM Routing, NeurIPS 2025: https://openreview.net/forum?id=iZC5xoQQkX
- Router-R1, NeurIPS 2025: https://openreview.net/forum?id=DWf4vroKWJ
- Universal Model Routing, ICLR 2026: https://openreview.net/pdf?id=ka82fvJ5f1
- LLMRouterBench, arXiv 2026: https://arxiv.org/abs/2601.07206
- WebRouter, arXiv 2025: https://arxiv.org/abs/2510.11221
- FineRouter, arXiv 2026: https://arxiv.org/abs/2603.19415
- IRT-Router, ACL 2025: https://aclanthology.org/2025.acl-long.761/

---

## 5. Classical/CS inspirations outside LLM routing

These are for related work and framing, not direct baselines.

### Information Bottleneck / rate--distortion
- Information Bottleneck Method: https://arxiv.org/abs/physics/0004057
- Deep Variational Information Bottleneck: https://arxiv.org/abs/1612.00410
- How to use: RouteCode asks for the compressed query code that preserves information relevant to model selection.

### Mixture-of-Experts gating / conditional computation
- Sparsely-Gated Mixture-of-Experts: https://arxiv.org/abs/1701.06538
- How to use: LLM routing is externalized MoE; RouteCode studies what information the gate needs.

### Early exit / adaptive computation
- BranchyNet: https://arxiv.org/abs/1709.01686
- How to use: easy routing decisions can exit after coarse labels; ambiguous ones refine.

### Selective prediction / learning to defer
- Selective classification / defer-to-expert literature should be added later if used for adaptive refinement.
- How to use: route with coarse label when safe; defer/refine/escalate when risk is high.

---

## 6. Local serving and model infrastructure links

Use only after benchmark-only pilots work.

- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- llama-cpp-python server: https://llama-cpp-python.readthedocs.io/en/latest/server/
- SGLang OpenAI-compatible API docs: https://docs.sglang.ai/basic_usage/openai_api_completions.html
- NVIDIA RTX 5090 product page: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/

Candidate local models:
- Qwen3-8B / Qwen3-4B / Qwen3-Embedding models via Hugging Face.
- BGE / E5 / sentence-transformers embeddings.
- ModernBERT/DeBERTa classifiers.

---

## 7. Codex and agent workflow links

- Codex AGENTS.md guide: https://developers.openai.com/codex/guides/agents-md
- Codex Goals cookbook: https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex
- General AGENTS.md format: https://agents.md/

---

## 8. What to record for every external method

For each external baseline or repo, record in `results/<run>/README.md`:

- paper link;
- repo link;
- commit hash or release version;
- exact command;
- dataset split;
- hyperparameters;
- hardware/runtime;
- whether data leakage checks passed;
- whether any code changes/adapters were needed.
