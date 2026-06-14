# Papers, Repos, and Baselines for RouteCode

This file is for Codex/agents. It tells the agent which papers/repos are relevant, which methods are mandatory baselines, and which methods are related work only.

---

## 0. Baseline philosophy

Do not compare against every routing paper. Compare against open-source, reproducible baselines that answer one of these questions:

1. How strong are trivial non-routing methods?
2. How much routing gain comes from coarse labels/clusters?
3. How strong are standard learned routers?
4. How strong are recent open-source paper baselines?
5. Does RouteCode improve transfer/calibration, not just one-shot performance?

Main benchmark target: LLMRouterBench.

---

## 1. Primary benchmark and data

### LLMRouterBench
- Paper: https://arxiv.org/abs/2601.07206
- Code/data: https://github.com/ynulihao/LLMRouterBench
- Role: primary benchmark if accessible.
- Why important: recent large unified benchmark for LLM routing; useful for rate--distortion curves, model-pool transfer, and benchmark compressibility.
- First action: inspect repo data format and write a loader to convert it to canonical `outcomes.parquet`.
- Baselines in/around the benchmark may include: RouterDC, EmbedLLM, MODEL-SAT, GraphRouter, Avengers, HybridLLM, FrugalGPT, RouteLLM, Avengers-Pro, OpenRouter-style methods. Verify exact names from repo/paper before reporting.

### RouterBench
- Paper/OpenReview: https://openreview.net/forum?id=IVXmV8Uxwh
- Paper/arXiv: https://arxiv.org/abs/2403.12031
- Code/data: https://github.com/withmartian/routerbench
- Role: secondary benchmark / replication setting.

### RouteLLM
- Paper: https://arxiv.org/abs/2406.18665
- OpenReview: https://openreview.net/forum?id=8sSqNntaMr
- Code: https://github.com/lm-sys/routellm
- Role: canonical learned-router baseline.
- Methods to compare: matrix factorization router, BERT router, causal LLM router if practical, random, Elo/weighted Elo if available.

---

## 2. Required baselines for our paper

### A. Reference bounds — implement ourselves

1. Random router.
2. Cheapest model.
3. Best single model.
4. Dataset oracle: best model per dataset/domain, fit on train only.
5. Query oracle: best model per query, true upper bound.

### B. Compression baselines — implement ourselves

1. Dataset-label lookup router.
2. Predicted-topic lookup router.
3. Embedding-cluster lookup router.
4. Utility-cluster oracle codebook.
5. RouteCode flat codebook.
6. RouteCode predicted-code router.

These are central because the paper asks whether small labels/codes preserve routing utility.

### C. Simple learned routers — implement ourselves and/or use LLMRouter

1. kNN router.
2. Logistic regression on embeddings.
3. MLP on embeddings.
4. SVM router.
5. Gradient-boosted tree if cheap.
6. ModernBERT/DeBERTa query-to-model classifier.

### D. Paper baselines — use open source if possible

1. RouteLLM-MF / RouteLLM-BERT from https://github.com/lm-sys/routellm
2. LLMRouter kNN/SVM/MLP/GraphRouter from https://github.com/ulab-uiuc/LLMRouter
3. GraphRouter via LLMRouter if available.
4. Avengers-Pro cluster routing from https://github.com/ZhangYiqun018/AvengersPro or the LLMRouterBench copy under `data/raw/external/LLMRouterBench/baselines/AvengersPro`.
5. BEST-Route only for adaptive-compute action spaces: https://github.com/microsoft/best-route-llm
6. IRT-Router optional: https://github.com/Mercidaiha/IRT-Router

---

## 3. Open-source routing libraries/baselines

### LLMRouter library
- Code: https://github.com/ulab-uiuc/LLMRouter
- Docs: https://ulab-uiuc.github.io/LLMRouter/
- Role: open-source implementation hub for many baselines.
- Candidate baselines: kNN, SVM, MLP, MatrixFactorization, Elo, RouterDC, AutoMix, HybridLLM, GraphRouter, causal LLM router, others if available.
- First action: inspect supported routers and identify which can run on LLMRouterBench/RouterBench with minimal adapter code.

### kNN router
- LLMRouter implementation: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/knnrouter/README.md
- Related paper: https://arxiv.org/abs/2505.12601
- Role: very important simple baseline.

### SVM router
- LLMRouter implementation: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/svmrouter/README.md
- Role: simple supervised learned-router baseline.

### GraphRouter
- Paper/OpenReview: https://openreview.net/forum?id=eU39PDsZtT
- ICLR proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/41b6674c28a9b93ec8d22a53ca25bc3b-Abstract-Conference.html
- arXiv: https://arxiv.org/abs/2410.03834
- Code: use LLMRouter implementation first: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/graphrouter/README.md
- Role: strong structured-router baseline; ICLR 2025.

### Avengers-Pro
- Paper/arXiv: https://arxiv.org/abs/2508.12631
- Code: https://github.com/ZhangYiqun018/AvengersPro
- Local LLMRouterBench source path: `data/raw/external/LLMRouterBench/baselines/AvengersPro`
- Role: cluster-based performance-efficiency routing baseline.
- Use first as a split-aligned local compatibility baseline with deterministic RouteCode embeddings; only call it an official baseline after the upstream embedding/cache command path is pinned.

### BEST-Route
- Paper/arXiv: https://arxiv.org/abs/2506.22716
- OpenReview: https://openreview.net/forum?id=tFBIbCVXkG
- Code: https://github.com/microsoft/best-route-llm
- Role: recent ICML 2025 adaptive-compute routing baseline.
- Use only when evaluating performance-cost or model-plus-sampling action spaces. Not mandatory for first synthetic pilot.

### IRT-Router
- Paper: https://aclanthology.org/2025.acl-long.761/
- Code: https://github.com/Mercidaiha/IRT-Router
- Role: optional interpretability/difficulty baseline. Useful if comparing route labels to explicit difficulty/model-ability decomposition.

---

## 4. Close novelty-boundary papers

These are not necessarily baselines in the first implementation, but they must be cited and checked to avoid false novelty claims.

### WebRouter
- Paper: https://arxiv.org/abs/2510.11221
- HTML: https://arxiv.org/html/2510.11221v1
- Why close: uses a cost-aware variational information bottleneck for LLM routing.
- Novelty warning: do not claim first information-bottleneck router.
- Difference we should maintain: RouteCode studies routing rate--distortion frontier and discrete explainable route labels, not just a compressed neural representation.
- Baseline status: related-work only unless official code appears or a simple VIB baseline is easy.

### FineRouter
- Paper: https://arxiv.org/abs/2603.19415
- HTML: https://arxiv.org/html/2603.19415v1
- Why close: latent task discovery for prompt routing.
- Novelty warning: do not claim first latent task discovery for routing.
- Difference we should maintain: RouteCode asks how many bits/query labels are minimally sufficient for routing and evaluates model-pool transfer and benchmark compressibility.
- Baseline status: related-work only unless official code appears or simplified latent-task baseline is needed.

### Select-then-Route
- Paper: https://aclanthology.org/2025.emnlp-industry.28/
- Role: taxonomy-guided routing reference and possible simple hierarchical baseline.

### Universal Model Routing
- Paper/OpenReview PDF: https://openreview.net/pdf?id=ka82fvJ5f1
- arXiv: https://arxiv.org/abs/2502.08773
- Role: related work for unseen/new model transfer. It focuses on model-side representations; RouteCode focuses on query-side route labels.

---

## 5. Top-conference routing papers to cite

- RouteLLM, ICLR 2025: https://openreview.net/forum?id=8sSqNntaMr
- BEST-Route, ICML 2025: https://openreview.net/forum?id=tFBIbCVXkG
- A Unified Approach to Routing and Cascading for LLMs, ICML 2025: https://openreview.net/forum?id=AAl89VNNy1
- GraphRouter, ICLR 2025: https://openreview.net/forum?id=eU39PDsZtT
- Capability Instruction Tuning, AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/34790
- Causal LLM Routing, NeurIPS 2025: https://openreview.net/forum?id=iZC5xoQQkX
- Router-R1, NeurIPS 2025: https://openreview.net/forum?id=DWf4vroKWJ
- Universal Model Routing, ICLR 2026: https://openreview.net/pdf?id=ka82fvJ5f1
- LLMRouterBench, arXiv 2026: https://arxiv.org/abs/2601.07206
- WebRouter, arXiv 2025: https://arxiv.org/abs/2510.11221
- FineRouter, arXiv 2026: https://arxiv.org/abs/2603.19415
- IRT-Router, ACL 2025: https://aclanthology.org/2025.acl-long.761/

---

## 6. Baseline implementation priority

### First synthetic pilot

Implement ourselves:

1. best single model;
2. oracle router;
3. dataset-label lookup;
4. predicted-topic lookup if synthetic topics exist;
5. embedding-cluster lookup;
6. kNN router;
7. flat RouteCode.

### First real benchmark run

Add:

1. RouteLLM matrix factorization or BERT router;
2. LLMRouter kNN/SVM/MLP;
3. GraphRouter if library adapter is straightforward;
4. LLMRouterBench built-in baselines if reproducible.

### Later/optional

1. BEST-Route for adaptive compute;
2. IRT-Router for interpretability/difficulty;
3. WebRouter/FineRouter reimplementation only if code is released or easy to reproduce;
4. local RTX validation models.

---

## 7. What to record in every experiment README

For each external baseline, record:

- paper link;
- code repo link;
- commit hash or release version;
- exact command run;
- dataset split;
- whether train/val/test leakage was avoided;
- hyperparameters;
- hardware/runtime;
- any implementation mismatch from the paper.
