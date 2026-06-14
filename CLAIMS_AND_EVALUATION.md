# RouteCode Claims and Evaluation Logic

This file prevents overclaiming and defines which experiments support which claims.

---

## 1. Main claims

### Claim 1: Small route labels recover much of routing performance

Supported if:

- RouteCode with small K, e.g. 8--32 labels, recovers a large fraction of learned-router gain.
- It beats semantic/topic/embedding clusters at the same K.
- It is close to kNN/MLP/RouteLLM under the same split.

Required metrics:

- recovered gap vs best learned router;
- recovered gap vs oracle;
- oracle regret;
- bootstrap CI;
- rate: log2(K) and H(Z).

Strong wording threshold:

```text
>=85% of best learned-router gain, lower CI >=80%, and >=50--60% of oracle gain.
```

### Claim 2: New models can be integrated with fewer calibration examples

Supported if:

- Freeze query-to-label predictor.
- Add a held-out/new model using r examples per label.
- RouteCode reaches competitive utility with fewer model evaluations than direct router retraining.

Required plot:

```text
utility vs calibration examples per new model
```

### Claim 3: Route labels transfer across model pools better than direct routers

Supported if:

- q->label predictor trained on pool A works for pool B after label->model recalibration.
- Direct q->model routers degrade more or require more new labels.

### Claim 4: Benchmark compressibility reveals evaluation artifacts

Supported if:

- dataset-label or predicted-topic routers recover large routing gain;
- mixed-domain splits are much more compressible than domain-homogeneous or leave-domain-out splits;
- router rankings change across splits.

Be careful: do not say “benchmark is broken” unless evidence is very strong.

### Claim 5: Adaptive refinement improves cost--quality

Optional. Supported if:

- coarse route labels are enough for many queries;
- uncertain/high-regret queries are identifiable;
- entropy/margin/VOI refinement improves utility per refinement cost.

Do not make this the first paper's core unless results are strong.

---

## 2. Which claims require new data?

| Claim | Existing benchmark enough? | API needed? |
|---|---:|---:|
| Small labels recover performance | yes | no |
| Transfer across model pools | yes, via held-out models | optional |
| New-model integration | yes, simulated with held-out models | optional validation |
| Benchmark compressibility | yes | no |
| Adaptive refinement | mostly yes | optional |

---

## 3. Negative result handling

If small route labels fail:

- do not hide it;
- report slow rate--distortion curve;
- conclude that fine-grained query information is necessary for that workload;
- analyze which domains/model pools are non-compressible.

If utility-only labels are strong but text-to-label predictor fails:

- report that routing structure exists but is not inferable from current query encoders;
- strengthen predictability-constrained objective;
- analyze examples where utility structure is hidden.

If benchmarks are highly compressible only under dataset labels:

- frame as benchmark partition leakage, not deployable routing performance.

---

## 4. Required result tables

1. Main cost-quality table.
2. Compression ladder table.
3. Rate--distortion table.
4. Code predictor table.
5. New-model calibration table.
6. Split robustness table.
7. Ablation table.

---

## 5. Required figures

1. Compression ladder.
2. Rate--distortion curve.
3. Recovered gap vs K.
4. Calibration examples vs utility for new model.
5. Split rank-correlation heatmap.
6. Code card/example heatmap.

