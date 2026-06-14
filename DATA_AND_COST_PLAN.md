# RouteCode Data and Cost Plan

This file answers: what training data do we need, how much, and what will it cost?

---

## 1. Data needed

For each query and model, we need outcomes:

```text
query_text
model_id
quality/correctness
cost/tokens
latency optional
dataset/domain metadata
```

The core object is a matrix:

```text
U[N queries, M models]
```

where:

\[
U(q,m)=quality(q,m)-\lambda cost(q,m)
\]

The expensive part is acquiring query--model outcomes. Training the small route-label predictor is cheap.

---

## 2. Recommended data acquisition phases

### Phase 1: benchmark-only, zero API cost

Use existing precomputed routing matrices:

- LLMRouterBench
- RouterBench
- RouteLLM data/evaluation

Data size:

```text
pilot: 2K--5K queries
paper-scale: 20K--100K+ queries if available
K labels: 4, 8, 16, 32, 64, 128
```

Cost:

```text
$0 API cost
local compute only
```

### Phase 2: simulated new-model integration

Use existing benchmark models. Treat some models as held-out/new.

Protocol:

```text
freeze query-to-label predictor
sample r examples per label for held-out model
estimate new model utility per label
update label-to-model table
```

Sweep:

```text
r = 1,2,4,8,16,32,64 examples per label
```

Cost:

```text
$0 API cost
```

### Phase 3: local RTX validation

Run a small local open-model matrix only after benchmark pilot works.

Possible local model pool:

```text
Qwen3-4B / Qwen3-8B
Qwen Coder 7B/14B if code-heavy
Llama/Gemma/Mistral/Phi 7B--8B class models
```

Query count:

```text
1K--3K queries
4--8 local models
```

Cost:

```text
$0 API cost
hours to days of local GPU time depending on model/context/output
```

### Phase 4: small API calibration validation

Only after the paper looks promising.

Example budgets:

```text
K=16, r=8 -> 128 calls per new model
K=32, r=16 -> 512 calls per new model
```

This validates the practical new-model integration claim.

Avoid full API matrices unless absolutely necessary.

---

## 3. How much training data is needed?

### Synthetic pilot

```text
N = 2K--5K queries
M = 4--8 models
K = 4,8,16,32 route labels
```

Enough to test code, metrics, and curve shape.

### First real benchmark pilot

```text
N = 5K--20K queries
M = available model pool
K = 4,8,16,32,64
```

Enough to see if the idea is alive.

### Full paper experiment

```text
N = 20K--100K+ queries if available
M = all reliable benchmark models
multiple splits
multiple lambda values
multiple seeds
```

### New-model calibration

Use:

```text
K * r examples per new model
```

Example:

```text
K=32, r=8 -> 256 model calls/evaluations
K=32, r=16 -> 512 model calls/evaluations
```

---

## 4. Cost logic

### Benchmark-only work

Expected API cost:

```text
$0
```

### Local model validation

Expected API cost:

```text
$0
```

Cost is local GPU time only.

### API calibration validation

Expected cost depends on provider. Formula:

```text
cost = calls * (input_tokens * input_price_per_token + output_tokens * output_price_per_token)
```

Examples per new model:

| K | r | calls |
|---:|---:|---:|
| 16 | 4 | 64 |
| 16 | 8 | 128 |
| 16 | 16 | 256 |
| 32 | 8 | 256 |
| 32 | 16 | 512 |
| 64 | 16 | 1024 |

Do not use API until benchmark results support the project.

---

## 5. Data quality rules

- Prefer deterministic scoring datasets first: math, code, multiple-choice, exact QA.
- Avoid LLM judges in the pilot if possible.
- If LLM judges are used, cache responses and track judge model/version.
- Always keep all outcomes for a query in the same split.
- Never cluster train+test together.
- Never compute label best models using test outcomes.

---

## 6. Why RouteCode reduces calibration cost

A direct router learns:

```text
query -> model
```

When a new model arrives, it may need many query-level labels.

RouteCode learns:

```text
query -> label
label -> model
```

When a new model arrives, reuse `query -> label` and evaluate the new model only on examples from each label.

This is the practical core claim.

