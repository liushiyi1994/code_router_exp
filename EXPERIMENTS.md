# RouteCode Experiments

This file is the concrete experiment plan for the RouteCode project.

---

## Common setup

### Input data

Canonical outcome table:

```text
query_id, query_text, dataset, model_id, quality, cost_total, latency, tokens_input, tokens_output, judge
```

From this table build:

```text
Y[N, M] = quality matrix
C[N, M] = cost matrix
U[N, M] = Y - lambda * C
```

### Default split

Use train/val/test split by query id. All model outcomes for the same query must stay in the same split.

Do not split individual query-model rows independently.

---

## E0: data audit and routability

### Goal

Check whether routing is meaningful.

### Methods

- best single model;
- oracle model per query;
- per-dataset oracle;
- per-domain oracle if available.

### Outputs

- `table_routability.csv`
- `fig_model_win_distribution.pdf`
- `fig_oracle_gap_by_dataset.pdf`

### Metrics

- best single utility;
- oracle utility;
- oracle gap;
- model-win entropy;
- per-domain oracle gap;
- dominance ratio.

### Expected observations

Some pools will be dominated by one model. Those are low-routability. Focus main experiments on pools with non-trivial oracle gap.

---

## E1: compression ladder pilot

### Goal

Measure how much routing performance is recoverable from compressed labels.

### Methods

1. Best single
2. Dataset-label lookup
3. Predicted-topic lookup
4. Embedding-cluster lookup
5. kNN router
6. Learned MLP router
7. RouteCode oracle labels
8. Query oracle

### Outputs

- `fig_compression_ladder.pdf`
- `table_recovered_gap.csv`
- `table_leakage_gap.csv`

### Metrics

- mean utility;
- accuracy;
- normalized cost;
- recovered gap vs learned router;
- recovered gap vs oracle;
- bootstrap CI;
- dataset-label leakage gap.

### Interpretation

- Dataset label strong, predicted topic weak: benchmark partition leakage.
- Predicted topic strong: coarse query labels explain routing.
- Embedding cluster strong: local embedding geometry explains routing.
- RouteCode strong: utility-aware labels add value beyond semantic labels.

---

## E2: rate--distortion curve

### Goal

Measure routing regret vs number of labels/bits.

### K values

```text
1, 2, 4, 8, 16, 32, 64, 128
```

### Methods

- random labels;
- semantic embedding k-means;
- dataset/domain labels;
- utility-vector clustering;
- flat regret-optimized RouteCode;
- predictability-constrained RouteCode;
- full learned router;
- oracle.

### Outputs

- `fig_rate_distortion_regret.pdf`
- `fig_rate_distortion_recovered_gap.pdf`
- `table_rate_distortion.csv`

### Strong result

RouteCode dominates semantic clusters at the same K, and the curve saturates early.

---

## E3: query-to-label prediction

### Goal

Turn oracle labels into a deployable router.

### Predictors

- logistic regression on embeddings;
- MLP on embeddings;
- kNN label predictor;
- ModernBERT/DeBERTa classifier;
- optional QLoRA 7B baseline.

### Outputs

- `table_predictor_comparison.csv`
- `fig_utility_weighted_confusion.pdf`
- `fig_calibration_curve.pdf`

### Metrics

- label accuracy;
- utility-weighted label confusion;
- routing utility after prediction;
- oracle-code vs predicted-code gap;
- ECE;
- OOD performance.

---

## E4: explainability / code cards

### Goal

Make learned route labels inspectable.

### For each label, report

- human name;
- best model;
- second-best model;
- utility margin;
- top datasets/domains;
- representative queries;
- high-regret failures;
- model utility vector;
- label size and entropy.

### Outputs

- `code_cards.json`
- `code_cards.md`
- `fig_code_label_heatmap.pdf`

---

## E5: new-model integration

### Goal

Show that a new model can be integrated with fewer calibration examples.

Detailed plan:

- `docs/PHASE4_PHASE5_CALIBRATION_TRANSFER_PLAN.md`

### Protocol

1. Freeze query-to-label predictor.
2. Treat one or more models as new/held-out.
3. For each label, sample r calibration queries.
4. Estimate new model utility per label.
5. Update label-to-model table.
6. Compare to direct router trained under same number of calibration labels.

### r values

```text
1, 2, 4, 8, 16, 32, 64 examples per label
```

### Outputs

- `fig_transfer_calibration_curve.pdf`
- `table_new_model_integration.csv`

### Strong result

RouteCode reaches high utility with far fewer query-model evaluations than direct router retraining.

---

## E5b: model-pool transfer

### Goal

Show that a RouteCode label space learned on one candidate model pool can be
reused on another candidate model pool by recalibrating the label-to-model
table, instead of retraining a full query-to-model router.

Detailed plan:

- `docs/PHASE4_PHASE5_CALIBRATION_TRANSFER_PLAN.md`

### Protocol

1. Fit RouteCode labels and the query-to-label predictor on a source model pool.
2. Freeze the query-to-label predictor.
3. Build a target model pool with added, removed, or replaced models.
4. Estimate target-pool utility per route label on a target calibration split.
5. Recompute the label-to-model table for the target pool.
6. Evaluate on held-out target test queries.
7. Compare against target direct routers under the same calibration budget.

### Outputs

- `table_model_pool_transfer.csv`
- `phase_f_g_model_pool_transfer_memo.md`

### Metrics

- target mean utility;
- target oracle regret;
- transfer recovered gap vs target oracle;
- transfer utility retention vs native target RouteCode;
- calibration examples/evaluations;
- source-target model overlap;
- negative transfer rate;
- bootstrap CI.

### Strong result

Transferred RouteCode labels match direct target-router utility with 3x--5x fewer
target-pool calibration examples and remain competitive under low-overlap
source/target model pools.

---

## E6: benchmark compressibility and leakage

### Goal

Diagnose whether benchmarks reward coarse dataset/domain recognition.

### Splits

- random mixed;
- leave-one-dataset-out;
- leave-one-domain-out;
- domain-homogeneous;
- cluster-held-out;
- model-pool holdout.

### Outputs

- `fig_split_rate_distortion.pdf`
- `fig_rank_correlation_heatmap.pdf`
- `table_split_robustness.csv`

### Strong result

Mixed-domain splits are highly compressible; domain-homogeneous/OOD splits require more bits and may reorder router rankings.

---

## E7: adaptive refinement

### Goal

Spend extra routing computation only on ambiguous/high-regret queries.

### Methods

- coarse RouteCode only;
- fine RouteCode only;
- confidence threshold refinement;
- entropy/margin refinement;
- VOI refinement;
- full learned router.

### Outputs

- `fig_refinement_utility_vs_cost.pdf`
- `fig_risk_coverage.pdf`
- `table_adaptive_refinement.csv`

### Strong result

Adaptive refinement matches most full-router utility while refining a minority of queries.

---

## E8: ablations and sensitivity

Required sweeps:

- K;
- lambda;
- embedding backbone;
- predictor type;
- model pool size;
- model pool composition;
- train data fraction;
- random seeds;
- label noise;
- cost noise;
- label imbalance regularization.

Outputs:

- `table_ablation_summary.csv`
- `fig_sensitivity_k_lambda.pdf`
- `fig_seed_stability.pdf`

---

## Synthetic validation

Before overinterpreting real data, create synthetic utility matrices:

```text
U(q, m) = model_skill[m]
        - query_difficulty[q]
        + domain_affinity[domain(q), m]
        + residual_interaction[q, m]
        + noise
```

Vary:

- domain affinity strength;
- residual interaction strength;
- number of models;
- number of latent labels;
- label predictability;
- noise.

Use synthetic validation to verify that RouteCode recovers known low-rate or high-rate structure.
