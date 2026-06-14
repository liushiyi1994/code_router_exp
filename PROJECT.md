# RouteCode Project Blueprint

**Working title:** *How Many Bits Does an LLM Router Need? Learning Explainable Route Labels for Model Routing*

**Short name:** RouteCode / Routing Bottleneck / Explainable Route Labels

**Project status:** research design + pilot implementation starter pack

---

## 0. One-paragraph summary

RouteCode studies LLM model routing. A normal router reads a full query and directly selects a model. RouteCode instead learns a small, discrete, explainable **routing label** for the query, then maps that label to a model. The routing label is not merely a human topic such as `math` or `code`; it is a **utility-aware label** learned from query--model outcome data. Queries share a label if they have similar model-selection consequences under a cost--quality objective. The central research question is: **how much query information does an LLM router actually need to select a good model?** The project measures this through a routing rate--distortion curve and tests whether small route labels can recover most learned-router performance, transfer across model pools, and reduce new-model calibration cost.

---

## 1. The exact problem we solve

### 1.1 Standard LLM routing problem

Input:

```text
query q
candidate model pool M = {m1, m2, ..., mM}
quality-cost preference lambda
```

Output:

```text
selected model m
```

A standard learned router does:

```text
q -> selected model
```

RouteCode does:

```text
q -> routing label z -> selected model
```

### 1.2 Our router input and output at inference time

**Router input:**

```json
{
  "query_text": "Write a Python function to find the longest palindromic substring.",
  "candidate_models": ["cheap_general_8B", "code_model_7B", "math_model_7B", "frontier_model"],
  "lambda_cost": 0.3
}
```

**Router output:**

```json
{
  "route_label": "routine_code_generation",
  "route_label_id": 7,
  "selected_model": "code_model_7B",
  "confidence": 0.91,
  "fallback_action": "none"
}
```

The selected model then generates the final answer. RouteCode itself does not answer the query.

### 1.3 What is a route label?

A route label is a learned, discrete, explainable label for a query. It means:

> queries with this label should be routed similarly because they have similar model utility profiles.

Examples:

| Query | Normal topic | Route label | Selected model |
|---|---|---|---|
| "What is the capital of Canada?" | knowledge | easy_general | cheap_general_8B |
| "Write binary search in Python." | code | routine_code_generation | code_model_7B |
| "Debug this CUDA race condition." | code | hard_systems_code | frontier_model |
| "Prove every finite integral domain is a field." | math | hard_symbolic_reasoning | frontier_model |
| "Convert this JSON object to a TypeScript interface." | code | simple_code_transform | code_model_7B |

A normal topic label asks: *what is this query about?*  
A route label asks: *which model-selection behavior does this query need?*

### 1.4 Explainable but not hand-written

Do not manually define route labels as the main method. The correct design is:

1. Learn route labels from query--model utility patterns.
2. Train a cheap query-to-label predictor.
3. Explain each label afterward with a **code card**.

Example code card:

```text
Route label 7: routine_code_generation
Best model: code_model_7B
Second best: frontier_model
Why this label exists: code_model_7B matches frontier quality on these queries at lower cost.
Common query patterns: Python function generation, SQL transformations, JavaScript snippets.
Failure cases: concurrency bugs, CUDA, complex algorithm proof.
Representative queries: [...]
Model utility table: {...}
```

This is important for novelty: previous studies may learn hidden latent vectors; RouteCode learns **inspectable routing labels**.

---

## 2. Research thesis

### 2.1 Main thesis

> LLM routing often has a low-dimensional decision structure. A small number of utility-aware route labels can preserve much of the model-selection utility of full-text learned routers.

### 2.2 What this is not

This project is **not** about saving a few router tokens. Router tokens are too cheap for that to be the main contribution. If we need to call another LLM to summarize/classify every query, the runtime argument becomes weak.

The project is about:

1. measuring the information requirement of routing;
2. learning explainable, utility-aware routing labels;
3. reducing calibration data needed for new model pools;
4. diagnosing benchmark compressibility and leakage;
5. optionally performing adaptive refinement only for ambiguous queries.

### 2.3 Core claim hierarchy

Do not claim all benefits equally. The target paper should have a clear center.

**Main claims to aim for:**

1. Small route labels recover most routing performance.
2. New models can be integrated with far fewer calibration examples.

**Secondary claim:**

3. Route labels transfer across model pools better than direct learned routers.

**Diagnostic claim:**

4. Some routing benchmarks are highly compressible, revealing evaluation artifacts or coarse-domain shortcuts.

**Optional extension:**

5. Adaptive refinement improves cost--quality by spending extra routing computation only on ambiguous cases.

---

## 3. Formal setup

### 3.1 Data

Assume a query--model outcome matrix:

- queries: \(q_i, i=1,\ldots,N\)
- models: \(m_j, j=1,\ldots,M\)
- quality/correctness: \(y_{ij}\)
- cost: \(c_{ij}\)
- optional latency: \(\ell_{ij}\)
- metadata: dataset, domain, task, query length, judge, timestamp, model family

### 3.2 Utility

Default utility:

\[
U(q_i,m_j) = y_{ij} - \lambda c_{ij}
\]

Optional latency-aware utility:

\[
U(q_i,m_j) = y_{ij} - \lambda c_{ij} - \mu \ell_{ij}
\]

Oracle route:

\[
m^*(q_i) = \arg\max_{m_j} U(q_i,m_j)
\]

Oracle utility:

\[
U^*(q_i)=\max_j U(q_i,m_j)
\]

### 3.3 Route label

Route label:

\[
z = g(q), \quad z \in \{1,\ldots,K\}
\]

Code-to-model table:

\[
\pi(z)=\arg\max_m \mathbb{E}[U(q,m)\mid g(q)=z]
\]

Routing regret:

\[
D(g,\pi)=\mathbb{E}_q[U^*(q)-U(q,\pi(g(q)))]
\]

Rate:

\[
R(g)=\log_2 K
\]

or empirical entropy:

\[
H(Z)=-\sum_z p(z)\log_2 p(z)
\]

Acquisition cost:

\[
C_g(q)=\text{latency/cost of computing route label}
\]

Full objective:

\[
\min_{g,\pi}\; \mathbb{E}_q[U^*(q)-U(q,\pi(g(q)))] + \beta R(g) + \gamma \mathbb{E}_q[C_g(q)]
\]

Plain English:

- choose labels that minimize lost utility versus oracle;
- use as few labels/bits as possible;
- do not use a route-label extractor that costs more than it saves.

---

## 4. Method: RouteCode

### 4.1 Overview

RouteCode has three layers:

1. **Offline route-label discovery**: learn labels from a query--model utility matrix.
2. **Online query-to-label prediction**: train a cheap predictor from query text/embedding to label.
3. **Code-to-model routing table**: map each label to the best model under the cost--quality objective.

Optional fourth layer:

4. **Adaptive refinement**: if label confidence is low or expected regret is high, use a finer label, kNN, a stronger router, or fallback model.

### 4.2 Offline route-label discovery

Input:

```text
U matrix: N queries x M models
number of labels K
```

Output:

```text
label assignment z_i for each training query
code-to-model table pi(z)
```

A simple alternating algorithm:

1. Initialize labels using embedding k-means, dataset labels, or random labels.
2. For each label z, choose best model:

\[
\pi(z)=\arg\max_m \sum_{i:g(i)=z} U_{im}
\]

3. Reassign each query to the label that minimizes routing regret:

\[
g(i)=\arg\min_z [U_i^* - U_{i,\pi(z)}]
\]

4. Repeat until convergence.

Important caveat: this can collapse multiple labels to the same model. Use entropy/balance regularization, initialization diversity, or split constraints if needed.

### 4.3 Utility-profile clustering baseline

Cluster rows of the utility matrix:

\[
\mathbf{u}_i=[U(q_i,m_1),\ldots,U(q_i,m_M)]
\]

This gives an oracle label structure. It is not deployable by itself because test-time utilities are unknown. It answers:

> does a compact routing-label structure exist at all?

### 4.4 Predictability-constrained RouteCode

Oracle labels may not be predictable from query text. So add a predictor:

\[
h_\theta(q) \rightarrow p_\theta(z\mid q)
\]

Train with utility-aware loss:

\[
\mathcal{L}(\theta)=\mathbb{E}_q[U^*(q)-\sum_z p_\theta(z\mid q)U(q,\pi(z))] + \alpha CE(z,h_\theta(q))
\]

This is better than normal label cross-entropy because confusing two labels should only be strongly punished if it causes a bad routing decision.

### 4.5 Label explanation: code cards

For every learned route label, generate a code card:

- label id;
- short human name;
- best model and second-best model;
- average utility per model;
- dominant datasets/domains;
- representative queries;
- high-regret failure cases;
- label confidence and entropy;
- calibration examples needed for new models.

This converts hidden latent routing into explainable routing labels.

### 4.6 Adaptive refinement

Only implement after flat RouteCode works.

At inference:

1. predict coarse label distribution `p(z|q)`;
2. compute confidence/margin/entropy;
3. estimate expected regret;
4. refine only if value of information is positive:

\[
VOI(q)=\mathbb{E}[U_{after}(q)]-U_{current}(q)-C_{refine}(q)>0
\]

Refinement options:

- finer route label in hierarchy;
- kNN local router;
- full learned router;
- strongest model fallback;
- top-2 verification;
- optional small local model probe.

Default should be non-generative and cheap.

---

## 5. Data and training plan

### 5.1 Training data needed

The ideal training data is:

```text
query_text
model_id
quality/correctness
cost/tokens
latency optional
dataset/domain metadata
```

The expensive object is the query--model outcome matrix, not the small router training.

### 5.2 Do not generate expensive data first

Start with public routing matrices. Initial API cost should be zero.

Primary benchmark:

- **LLMRouterBench**: recent 2026 benchmark, >400K instances, 21 datasets, 33 models, 10 baselines.

Secondary benchmarks/frameworks:

- **RouterBench**
- **RouteLLM**
- **LLMRouter** library baselines

### 5.3 Amount of data

Pilot:

```text
2K--5K queries
K = 4, 8, 16, 32 labels
embedding + MLP predictor
```

Serious experiments:

```text
20K--100K queries if available
K = 1, 2, 4, 8, 16, 32, 64, 128
multiple splits and model pools
```

New-model calibration simulation:

```text
K labels
r examples per label
new model calibration calls = K * r
```

Example:

```text
K=32, r=8 -> 256 calibration examples per new model
K=32, r=16 -> 512 calibration examples per new model
```

This is the key practical benefit: evaluate a new model per route label rather than across the whole query set.

---

## 6. Baselines and comparison methods

### 6.1 Required reference baselines

- Random model
- Cheapest model
- Best single model
- Dataset oracle: best model per dataset label, train only
- Query oracle: best model per query, upper bound

### 6.2 Required compressed baselines

- Dataset-label lookup router
- Predicted-topic lookup router
- Embedding-cluster lookup router
- Utility-cluster oracle codebook
- kNN router

### 6.3 Required learned baselines

Use open-source where possible:

- RouteLLM matrix-factorization router
- RouteLLM BERT router
- LLMRouter kNN/SVM/MLP/Elo baselines
- LLMRouter GraphRouter if accessible
- LLMRouterBench included baselines if easy to run
- Avengers / Avengers-Pro if exposed by LLMRouterBench

### 6.4 Optional baselines

- BEST-Route if evaluating model plus number-of-samples action space
- IRT-Router for difficulty/ability interpretability comparison
- simplified FineRouter-style latent-task clustering
- simplified WebRouter-style bottleneck router

Do not block the first pilot on optional baselines.

---

## 7. Experiments

### E0. Data audit and oracle gap

Purpose: determine whether routing is meaningful.

Metrics:

- best single utility;
- oracle utility;
- oracle gap;
- model-win distribution;
- winner entropy;
- per-dataset/domain oracle gap;
- model dominance ratio.

Go/no-go:

- If oracle gap is tiny, routing is not meaningful for that subset.
- If one model dominates everywhere, analyze separately.

### E1. Three-tier compression ladder

Purpose: test how much routing signal lives in coarse labels.

Methods:

1. dataset-label router;
2. predicted-topic router;
3. embedding-cluster router;
4. learned RouteCode;
5. full learned router;
6. oracle.

Metrics:

- recovered gap vs learned router;
- recovered gap vs oracle;
- dataset-label leakage gap;
- confidence intervals;
- performance by domain.

Important threshold:

- Only claim “routers barely use fine-grained query information” if predicted-topic or predicted-code recovers >=85% of the best learned router gain, with lower bootstrap CI >=80%.

### E2. Rate--distortion curve

Purpose: measure routing performance vs route-label bits.

Sweep:

```text
K = 1, 2, 4, 8, 16, 32, 64, 128
```

Plot:

```text
routing regret / recovered gap vs log2(K)
```

Expected strong result:

- small K, e.g. 8--32, recovers most learned-router gain.

### E3. RouteCode vs semantic labels

Purpose: show labels are not merely topics.

Compare at same K:

- semantic embedding k-means;
- predicted topic labels;
- utility-vector clustering;
- regret-optimized RouteCode;
- predictability-constrained RouteCode.

Strong result:

- RouteCode beats semantic clustering at fixed K.

### E4. Text-to-label predictor

Train:

- logistic regression on embeddings;
- MLP on embeddings;
- ModernBERT/DeBERTa classifier;
- optional LoRA 7B baseline.

Metrics:

- label accuracy;
- utility-weighted label confusion;
- routing regret after predicted labels;
- calibration/ECE;
- OOD/domain-held-out performance.

### E5. Model-pool transfer and new-model calibration

Protocol:

1. train query-to-label on model pool A;
2. hold out models or add simulated new models;
3. evaluate new model on `r` examples per route label;
4. update label-to-model table;
5. compare to direct router retraining under the same calibration budget.

Sweep:

```text
r = 1, 2, 4, 8, 16, 32, 64 examples per label
```

Strong result:

- RouteCode reaches competitive utility with far fewer model evaluations than direct routers.

### E6. Benchmark compressibility and leaderboard stability

Splits:

- random mixed split;
- leave-one-dataset-out;
- leave-one-domain-out;
- domain-homogeneous split;
- cluster-held-out split;
- model-pool holdout.

Metrics:

- method ranking under each split;
- Spearman/Kendall correlation between rankings;
- utility degradation;
- rate--distortion curve shape by split.

Strong result:

- mixed benchmarks are more compressible than domain-homogeneous ones;
- rankings may reorder under controlled splits.

### E7. Adaptive refinement

Only after E1--E5 work.

Compare:

- always coarse label;
- always fine label;
- adaptive confidence threshold;
- adaptive VOI refinement;
- full learned router.

Metrics:

- utility;
- regret;
- refinement rate;
- acquisition cost;
- risk-coverage curve.

---

## 8. Metrics

Every main table should include:

- accuracy/quality;
- normalized cost;
- mean utility;
- oracle regret;
- recovered gap vs best learned router;
- recovered gap vs oracle;
- code rate: `log2(K)` and empirical `H(Z)`;
- acquisition cost/latency of label extraction;
- bootstrap confidence intervals.

Transfer experiments additionally include:

- examples per label;
- total model evaluations;
- performance under same calibration budget;
- direct-router retraining comparison.

Code interpretability metrics:

- label purity by selected model;
- model-winner entropy per label;
- representative query examples;
- high-regret examples;
- label stability across seeds.

---

## 9. Ablations and sensitivity

Required:

1. `K` sweep: 1, 2, 4, 8, 16, 32, 64, 128.
2. Cost weight `lambda` sweep.
3. Code discovery objective: semantic k-means vs utility clustering vs regret RouteCode vs predictability-constrained RouteCode.
4. Predictor family: linear, MLP, ModernBERT/DeBERTa, optional LoRA 7B.
5. Embedding backbone: MiniLM, BGE, Qwen embedding, ModernBERT embeddings.
6. Model pool size: 2, 4, 8, 16+ if available.
7. Model pool composition: dominated vs complementary pool.
8. Train data fraction: 1%, 5%, 10%, 25%, 50%, 100%.
9. Split strategy: random, leave-dataset-out, leave-domain-out, cluster-held-out.
10. Label noise: random flips, judge-biased noise, domain-specific noise, cost misestimation.
11. Code imbalance: balanced vs unbalanced labels, entropy regularization.
12. Code stability: adjusted Rand index across seeds.
13. Acquisition cost: no classifier, cheap features, sentence embedding, ModernBERT classifier, optional LLM classifier.

---

## 10. Local compute plan

Hardware available:

```text
NVIDIA RTX 5090, 32GB VRAM
```

Use local inference only after benchmark-matrix experiments work. Initial project should require no API generation.

Preferred local serving:

1. vLLM for Hugging Face transformer models.
2. llama.cpp / llama-cpp-python for GGUF quantized models.
3. SGLang as alternative.

Recommended local models:

Generation/probe:

- Qwen3-8B or instruct variant;
- Qwen Coder 7B/14B for code;
- Llama/Mistral/Gemma/Phi 7B--14B-class alternatives.

Embeddings:

- Qwen3-Embedding family;
- BGE large/M3;
- all-MiniLM for cheap baseline;
- ModernBERT embeddings.

Classifier training:

- start with frozen embeddings + logistic regression/MLP;
- then ModernBERT/DeBERTa;
- LoRA 7B only as optional stronger baseline.

Do not use GPT/Claude subscriptions for batch experiments unless API credits are available. Chat subscriptions are useful for writing/debugging/manual inspection, not automated experiment budgets.

---

## 11. Expected outcomes and claim discipline

### Strong outcome

Observed:

- predicted RouteCode recovers >=85% of best learned-router gain;
- small K, e.g. 8--32 labels, recovers 80--90% of learned-router gain;
- RouteCode beats semantic clustering at fixed K;
- RouteCode integrates new models with far fewer calibration examples;
- benchmark compressibility varies by split.

Claim:

> Many LLM routing workloads require only a few bits of utility-aware query information. Explainable route labels can preserve most learned-router utility and enable cheap model-pool adaptation.

### Solid outcome

Observed:

- compressed labels recover 50--80%;
- RouteCode beats semantic clusters;
- transfer/sample efficiency works;
- adaptive refinement helps.

Claim:

> LLM routing has a low-rate coarse component and a high-rate residual component. RouteCode identifies and exploits this structure.

### Negative but useful outcome

Observed:

- compressed labels perform poorly;
- rate--distortion curve is slow;
- full-text routers are necessary.

Claim:

> Contrary to shortcut hypotheses, robust LLM routing requires high-rate query information. The routing rate--distortion curve distinguishes compressible from non-compressible workloads.

---

## 12. First two-week plan

### Day 1--2

- Create repo skeleton.
- Install Python dependencies.
- Download/load LLMRouterBench if accessible.
- Convert raw benchmark to canonical `outcomes.parquet`.

### Day 3--4

- Implement utility matrix.
- Implement best single and oracle.
- Compute oracle gap and routability stats.

### Day 5--7

- Implement dataset-label lookup.
- Compute leakage-aware train/val/test split.
- Cache embeddings.
- Implement embedding-cluster lookup.
- Implement kNN router.

### Week 2

- Run first compression ladder.
- Bootstrap CIs.
- Implement first utility-cluster codebook.
- Plot first rate--distortion curve.
- Decide if RouteCode is alive.

First decision question:

> Does `K=16` or `K=32` RouteCode recover a meaningful fraction of learned-router or oracle gap, and does it beat semantic clustering at the same K?

---

## 13. Repo skeleton

```text
routecode/
  AGENTS.md
  PROJECT.md
  CODEX_GOAL.md
  STARTING_PROMPT.md
  README.md
  pyproject.toml
  configs/
    default.yaml
    llmrouterbench.yaml
    routerbench.yaml
  data/
    raw/
    processed/
    cache/
  src/routecode/
    data/
      load_llmrouterbench.py
      load_routerbench.py
      normalize.py
      splits.py
    utility/
      costs.py
      utility.py
      metrics.py
    codes/
      semantic_clusters.py
      utility_clusters.py
      regret_codebook.py
      predictability.py
      hierarchical.py
      code_cards.py
    predictors/
      embedding_features.py
      classifiers.py
      train_predictor.py
    routers/
      single_best.py
      oracle.py
      dataset_lookup.py
      topic_lookup.py
      knn.py
      learned.py
      routecode.py
      adaptive.py
    eval/
      evaluate.py
      rate_distortion.py
      transfer.py
      bootstrap.py
      plots.py
  experiments/
    00_data_audit.py
    01_compression_ladder.py
    02_rate_distortion_curve.py
    03_routecode_train.py
    04_predictability.py
    05_transfer_new_models.py
    06_split_robustness.py
    07_adaptive_refinement.py
    08_ablations.py
  notebooks/
    pilot_results.ipynb
    code_interpretability.ipynb
  results/
    tables/
    figures/
    logs/
  scripts/
    serve_vllm.sh
    serve_llamacpp.sh
    serve_sglang.sh
```

---

## 14. Kill criteria

Pause or pivot if:

- benchmark cannot provide reliable query--model quality matrices;
- oracle gap is tiny across most model pools;
- utility-code oracle is not better than semantic clusters at any K;
- predicted codes cannot beat kNN/embedding clusters despite multiple predictors;
- model-pool transfer is absent;
- only result is "we save router tokens".

