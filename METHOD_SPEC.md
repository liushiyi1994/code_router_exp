# RouteCode Method Specification

This file defines exactly what the router takes as input, what it outputs, what is trained, and what problem the project solves.

---

## 1. Plain-language definition

A normal LLM router does:

```text
query -> selected model
```

RouteCode does:

```text
query -> route label -> selected model
```

The route label is a learned, discrete, explainable, utility-aware label for model-selection behavior. It is like a label for the query, but it is **not** merely a topic label. It is learned from query--model utility patterns.

A normal topic label asks:

```text
What is this query about?
```

A route label asks:

```text
Which model-selection behavior does this query need?
```

---

## 2. Inference-time input and output

### Router input

```json
{
  "query_text": "Write a Python function to find the longest palindromic substring.",
  "candidate_models": ["cheap_general_8B", "code_model_7B", "math_model_7B", "frontier_model"],
  "lambda_cost": 0.3,
  "optional_features": {
    "embedding": "cached or computed embedding",
    "query_length": 9,
    "predicted_topic": "code"
  }
}
```

### Router output

```json
{
  "route_label_id": 7,
  "route_label_name": "routine_code_generation",
  "selected_model": "code_model_7B",
  "confidence": 0.91,
  "fallback_action": "none"
}
```

The selected model then generates the final answer. RouteCode does not answer the query itself.

---

## 3. Simple examples

### Easy general knowledge

```text
Query: What is the capital of Canada?
Route label: easy_general
Selected model: cheap_general_8B
Reason: all models likely answer correctly; choose cheapest.
```

### Routine coding

```text
Query: Write binary search in Python.
Route label: routine_code_generation
Selected model: code_model_7B
Reason: code-specialized model gives high utility at lower cost than frontier model.
```

### Hard math/reasoning

```text
Query: Prove that every finite integral domain is a field.
Route label: hard_symbolic_reasoning
Selected model: frontier_model
Reason: cheap models are likely to fail.
```

### Ambiguous code + reasoning

```text
Query: Explain why this algorithm correctness proof is wrong.
Route label: ambiguous_code_reasoning
Selected model: frontier_model or refine
Reason: low-confidence boundary case; optional adaptive refinement.
```

---

## 4. What is trained?

There are three artifacts.

### A. Route-label codebook

Input:

```text
query--model utility matrix U[N, M]
number of labels K
```

Output:

```text
route label z_i for each training query
```

This is learned offline from model outcomes.

### B. Code-to-model table

For each route label, select the best model:

\[
\pi(z)=\arg\max_m \mathbb{E}[U(q,m)\mid z(q)=z]
\]

This table is cheap to update when a new model enters.

### C. Query-to-label predictor

Train a small predictor:

```text
h_theta(query) -> route_label
```

Candidate predictors:

- logistic regression on embeddings;
- MLP on embeddings;
- kNN over embeddings;
- ModernBERT/DeBERTa classifier;
- optional LoRA 7B only later as a stronger baseline.

---

## 5. What exact problem do we solve?

We solve:

> Learn the smallest query representation that preserves good model-routing decisions under a cost--quality objective.

Formal objective:

\[
U(q,m)=quality(q,m)-\lambda cost(q,m)
\]

Oracle:

\[
m^*(q)=\arg\max_m U(q,m)
\]

Route label:

\[
z=g(q), z\in\{1,\ldots,K\}
\]

Route table:

\[
\pi(z)=\arg\max_m \mathbb{E}[U(q,m)\mid g(q)=z]
\]

Optimization:

\[
\min_{g,\pi}\;\mathbb{E}_q[U(q,m^*(q))-U(q,\pi(g(q)))] + \beta R(g) + \gamma C_g(q)
\]

Plain English:

1. lose as little utility as possible versus the oracle;
2. use as few route labels/bits as possible;
3. do not spend too much compute to obtain the label.

---

## 6. What makes the label explainable?

Each learned label should have a **code card**.

Example:

```text
Route label 7: routine_code_generation
Best model: code_model_7B
Second best: frontier_model
Average utility margin: +0.18 over cheap_general_8B
Common query patterns: Python functions, SQL transformations, JS snippets
Representative queries: [...]
Failure cases: CUDA bugs, concurrency, algorithm proofs
Calibration examples needed for new model: 8--16 per label
```

This is the difference from hidden latent vectors. The label is learned, but inspectable.

---

## 7. Claims hierarchy

Do not claim all benefits equally.

### Main claims

1. Small route labels recover much of routing performance.
2. New models can be integrated with far fewer calibration examples.

### Secondary claim

3. Route labels transfer across model pools better than direct learned routers.

### Diagnostic claim

4. Routing benchmarks are compressible to different degrees; high compressibility may reveal coarse-domain artifacts.

### Optional claim

5. Adaptive refinement improves cost--quality by spending extra routing computation only on ambiguous cases.

---

## 8. Offensive-claim threshold

Only use strong wording like “routers barely use fine-grained query information” if:

```text
predicted route label router recovers >=85% of the best learned router's improvement over best single model,
with lower bootstrap CI >=80%,
and also recovers at least 50--60% of the oracle improvement.
```

If recovery is 50--80%, write the safer information-frontier/decomposition paper.

---

## 9. Minimal implementation sequence

1. Synthetic utility matrix generator.
2. Best single + oracle.
3. Dataset-label + embedding-cluster + kNN baselines.
4. Flat RouteCode for K = 1,2,4,8,16,32.
5. Rate--distortion curve.
6. Query-to-label MLP predictor.
7. Code cards.
8. New-model calibration simulation.
9. Real LLMRouterBench loader.

