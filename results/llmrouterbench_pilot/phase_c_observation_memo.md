# Phase C Observation Memo: LLMRouterBench Pilot

Config: `configs/llmrouterbench_pilot.yaml`

Scope: 2,897 queries, 17,382 query-model outcome rows, 6 datasets, and 6 local outcome model IDs from released LLMRouterBench records. Utility is quality-only in this pilot (`lambda_cost = 0.0`). The scripts do not call external model APIs.

This memo is a pilot checkpoint, not a paper claim.

## 1. Is the benchmark/model pool routable?

Yes, on the random query split. Best-single utility is `0.6672`, query-oracle utility is `0.8966`, and oracle regret against best-single is `0.2293` (`table_routability.csv`). The oracle winner distribution is not uniform: oracle model-win entropy is `1.2265` bits and the dominant oracle winner takes `77.1%` of test queries.

Interpretation: routing matters, but the pool is partly dominated. Future pilots should keep reporting dominance ratio and per-dataset oracle gap before treating a model pool as a strong routing benchmark.

## 2. How compressible is routing?

There are two different answers:

- Utility-space route labels are highly compressive. RouteCode oracle labels at K=16 reach mean utility `0.8897`, oracle regret `0.0069`, and recover `97.0%` of the query-oracle gap. K=32 reaches the query oracle on this split.
- Deployable predicted labels are not yet compressive. RouteCode predicted labels at K=16 reach mean utility `0.6138`, below best-single `0.6672`.

The correct conclusion is not "few inferred bits are enough" yet. The pilot shows that low-rate utility structure exists, but the current query-to-label predictors do not recover it from query features.

## 3. Which compressed representation works best?

For deployable/query-feature methods on the random split:

| Representation | Mean utility | Recovered gap vs query oracle |
|---|---:|---:|
| Dataset-label lookup | 0.7534 | 0.3759 |
| Predicted-topic lookup | 0.7448 | 0.3383 |
| Semantic embedding k-means, K=8 | 0.7500 | 0.3609 |
| kNN | 0.7362 | 0.3008 |
| RouteCode predicted labels, K=16 | 0.6138 | -0.2331 |

For oracle-code diagnostics, utility-aware RouteCode is strongest: K=16 recovers `97.0%` of the query-oracle gap, far above semantic embedding k-means at K=16 (`30.1%`).

## 4. Is utility-aware RouteCode better than topic/embedding codes?

As an oracle code, yes. As a deployable predicted code, no.

This is the key result of the corrected pilot. RouteCode oracle labels strongly dominate semantic clusters at K=8, K=16, and K=32, but label prediction fails. E3 predictor diagnostics show low utility-oracle label accuracy:

| Predictor | Label accuracy | Mean utility | Recovered gap vs query oracle |
|---|---:|---:|---:|
| utility_oracle_labels | 1.0000 | 0.8897 | 0.9699 |
| embedding_centroid_assignment | 0.1362 | 0.6586 | -0.0376 |
| mlp_label_predictor | 0.1466 | 0.6345 | -0.1429 |
| knn_label_predictor | 0.2052 | 0.6207 | -0.2030 |
| logistic_label_predictor | 0.1517 | 0.6138 | -0.2331 |

Decision: Phase D should prioritize a predictability-constrained RouteCode objective, not a larger version of utility-only clustering.

## 5. Are failures concentrated?

Somewhat. For predicted RouteCode K=16 residual regret, the top `5%` of test queries account for `17.7%` of regret, the top `10%` account for `35.4%`, and the top `20%` account for `70.7%` (`table_residual_concentration.csv`).

Interpretation: the top 20% result is enough to keep residual diagnosis alive, but top 5% and top 10% are not concentrated enough to justify adaptive refinement yet. The next check should test whether confidence, centroid distance, margin, or kNN disagreement predicts these failures.

## 6. Are results split-sensitive?

Yes. The bounded B4 pilot covers random, leave-dataset-out, leave-domain-out, domain-homogeneous, cluster-held-out, and model-pool holdout scenarios. The LLMRouterBench pilot now uses a coarse dataset-to-domain map: `aime`/`math500` -> `math`, `humaneval`/`mbpp` -> `code`, `gpqa` -> `science`, and `mmlupro` -> `broad_knowledge`. Method rankings are most unstable under the grouped code-domain holdout:

| Scenario | Rank correlation vs random |
|---|---:|
| leave_domain_out:code | 0.1928 |
| leave_dataset_out:aime | 0.5394 |
| leave_dataset_out:gpqa | 0.5645 |
| leave_domain_out:broad_knowledge | 0.6265 |
| domain_homogeneous:broad_knowledge | 0.6723 |
| cluster_held_out:0 | 0.6826 |
| domain_homogeneous:code | 0.7958 |
| model_pool_holdout:Intern-S1-mini | 0.9940 |

Interpretation: split design still changes rankings, and grouped domain holdout is no longer identical to leave-dataset-out. The domain map is coarse and manually configured, so it supports pilot split-sensitivity diagnosis but not a broad domain-generalization claim.

## 7. Which paper claim is alive?

Alive now:

- Predictability-constrained code learning claim: useful low-rate utility codes exist, and D2 can trade some utility-oracle strength for much more predictable labels.
- Benchmark diagnosis / evaluation artifact claim: dataset/topic labels are strong, and rankings reorder under controlled splits.
- Rate-distortion diagnostic claim: the repo can produce compression curves and expose the difference between oracle-code structure and deployable-code prediction.
- Residual diagnosis claim: failures are concentrated enough at the top 20% to justify predictor analysis, but not adaptive refinement yet.

Not alive yet:

- "Few inferred bits are enough" as an offensive claim. D2 deployable labels improve over flat predicted RouteCode, but the best pilot row recovers only `34.6%` of the query-oracle gap and remains below dataset-label lookup.
- New-model calibration / model-pool transfer as a paper-level claim. A simulated calibration sweep exists and is positive diagnostically, but it still needs broader held-out models, stronger baselines, and sensitivity checks.
- Adaptive refinement as a core claim. Residual concentration needs confidence/disagreement predictability evidence first.

## Recommended Next Step

Continue the remaining external-baseline and sensitivity layer. Broaden domain metadata beyond the current coarse map before making domain-generalization claims.

## D2 Update After Implementation

`experiments/06_predictability_constrained.py` now implements the fixed-K D2 sweep and writes `table_predictability_constrained.csv`, `fig_predictability_constrained_tradeoff.pdf`, D2 code cards, and `phase_d_method_memo.md`.

On the LLMRouterBench pilot, the best deployable D2 row is `d2_embedding_centroid` at alpha `3`: mean utility `0.7466`, recovered gap vs query oracle `0.3459`, and label accuracy `0.9810` against D2 joint labels. This improves substantially over flat RouteCode logistic label prediction (`0.6138`) and is above kNN/semantic KMeans in this run (`0.7362`), but it remains below dataset-label lookup (`0.7534`).

Updated next step: run the D4/E5 new-model calibration experiment. The D2 result supports using predictable labels for the calibration test, but it does not yet support the claim that small inferred route labels recover most query-oracle routing performance.

## E5 Update After New-Model Calibration Pilot

`experiments/07_new_model_calibration.py` now implements the simulated held-out-model calibration sweep and writes `table_new_model_integration.csv`, `fig_transfer_calibration_curve.pdf`, and `phase_e5_new_model_calibration_memo.md`.

On the LLMRouterBench pilot, using all six configured models as held-out/new models, RouteCode label calibration reaches mean utility `0.7374` at r=32 averaged across held-out models, with about `411.8` new-model evaluations. The strongest matched-budget direct retraining row among logistic/SVM/kNN/MLP/gradient-boosting is MLP at r=2, mean utility `0.6672`.

Updated next step: robustness. Before making a new-model calibration claim, add transformer direct-router baselines, seeds, official external baselines, and broader model-pool/price sensitivity.

## Phase E Internal Baseline Update

The compression and rate-distortion scripts now include named `random`, `dataset_oracle`, and `svm_embedding_router` rows. On the LLMRouterBench pilot compression ladder, `dataset_oracle` reaches mean utility `0.7638`, `svm_embedding_router` reaches `0.6724`, and `random` reaches `0.5724`.

Updated next step: external/stronger baselines and robustness remain. Local internal baselines are now enough for pilot diagnosis, but RouteLLM, GraphRouter/LLMRouter, and broader ablations are still not integrated.

## Phase F/G Robustness Update

`experiments/08_ablation_summary.py` now writes `table_ablation_summary.csv`, `fig_sensitivity_k_lambda.pdf`, `fig_seed_stability.pdf`, and `phase_f_g_ablation_memo.md`.

On the LLMRouterBench pilot, D2 embedding-centroid routing is the strongest deployable method in the seed-stability slice: mean recovered gap `0.3320`, std `0.0177` over seeds `[3, 7, 11]`. kNN reaches mean recovered gap `0.2723`, std `0.0356`. The K/lambda sweep now covers K `[4, 8, 16, 32, 64, 128]`; flat utility-oracle RouteCode reaches the query oracle at K=32 and remains saturated at K=64/128 across tested lambdas, while deployable D2 remains substantially below that oracle-code ceiling.

Updated next step: finish stronger/external baselines and deeper robustness around external embedding backbones, finer domain metadata, and broader model-pool composition.

## Phase G Sensitivity Suite Update

`experiments/09_sensitivity_suite.py` now writes `table_sensitivity_summary.csv`, `fig_sensitivity_summary.pdf`, and `phase_g_sensitivity_memo.md`.

The bounded suite covers local embedding-feature variants, KMeans vs agglomerative clustering, label noise, cost mis-estimation under a sensitivity-local cost objective, query-length buckets, top-4/drop-dominant model-pool subsets, and bootstrap counts `[50, 100, 300]`. On the LLMRouterBench pilot, D2 embedding-centroid routing remains competitive with kNN across the bounded sensitivity rows, but query-length bucket results vary substantially: D2 recovered gap ranges from `-0.0476` to `0.6140`.

Updated next step: external/stronger baselines remain the largest gap. Robustness also still needs true external embedding backbones, finer domain metadata/granularity, and broader model-pool construction.

## Phase E External-Style Baseline Surrogate Update

`experiments/10_external_baseline_surrogates.py` now writes `table_external_baselines.csv` and `phase_e_external_baseline_memo.md`.

This is a local no-API surrogate layer, not an official RouteLLM/LLMRouter reproduction. It adds a low-rank utility matrix-factorization router inspired by RouteLLM/EmbedLLM MF and a RouteLLM-style binary strong/weak threshold router. The pilot binary pair is `Qwen3-8B` vs `Qwen2.5-Coder-7B-Instruct`.

On the LLMRouterBench pilot, `routellm_style_mf_utility_router` reaches mean utility `0.7052` and recovered gap `0.1654`, above best-single but below kNN (`0.7362`, recovered gap `0.3008`) and D2 (`0.7466`, recovered gap `0.3459`). The best binary threshold surrogate is threshold `0.25`, mean utility `0.6931`, recovered gap `0.1128`.

Updated next step: official external baselines are still required. Prioritize RouteLLM-MF/BERT or LLMRouterBench RouteLLM adapter output if its embedding/checkpoint pipeline can be pinned locally, then GraphRouter/Avengers-Pro if their commands and leakage controls are clear.
