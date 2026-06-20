# Probe-Signal Research Notes For Broad100

Status: side-chat handoff for the main RouteCode/ProbeRoute++ thread.

Purpose: collect 2025--2026 paper-inspired ideas for proactive information gathering in routing. This is not a result claim. It is an experiment queue for finding probe signals that make broad100 routing closer to the cost-aware oracle.

Focused follow-on plan:

- `phase3/SLM_LLM_EARLY_SIGNAL_PROBE_PLAN.md`
- Scope: non-training and low-training SLM/LLM early-signal probes, including query-level internal confidence, early-rollout instability, SLM-vs-medium divergence, semantic uncertainty, and pre-generation activation diagnostics.
- Use this as the next-step plan when the goal is easy-to-implement transferable probes rather than another trained router.

Current method-direction handoff:

- `phase3/BENCHMARK_AGNOSTIC_PROBE_STATE_PLAN.md`
- Scope: avoid benchmark-specific checkers as the main method; learn a broad probe-state abstraction from cheap local model behavior and combine it with RouteCode labels/action policies.
- Use this as the next main-thread implementation target when the goal is a broadly transferable method rather than per-benchmark verifier engineering.

## Current Problem

Broad100 has a strong oracle, but deployable routing is still weak.

- The oracle can choose the right model/action per query.
- Query-only or simple label prediction does not recover enough of that oracle.
- Existing cached probes show that the bottleneck is observability: the router does not know which model will win without extra information.

The next research question is:

```text
What cheap probe signals should the router gather before choosing a model?
```

## Papers And Ideas To Use

### LLMRouterBench

Reference:

- https://arxiv.org/abs/2601.07206
- https://github.com/ynulihao/LLMRouterBench

Idea for us:

Use broad benchmark failures as a model-recall problem, not just a classification problem. Report per-model recall: when a model is oracle-best, how often does our router select it?

Experiment:

- Add model-win recall by benchmark and model.
- Add confusion table: oracle model vs selected model.
- Identify whether misses are local-vs-frontier, model-family, or task-type misses.

### kNN Routing

Reference:

- https://arxiv.org/abs/2505.12601

Idea for us:

kNN over historical utility patterns may be stronger than learned classifiers. Use nearest train queries to estimate model utilities, not topic labels.

Experiment:

- `kNN(query embedding -> average utility vector)`.
- `kNN(query + local probe answers -> average utility vector)`.
- Compare K = 1, 3, 5, 10, 20.
- Use train only for neighbor utility vectors.

### Prefill / Hidden-State Routing

References:

- https://arxiv.org/abs/2603.20895
- https://arxiv.org/abs/2602.09924

Idea for us:

Local LLM prefill activations may encode failure likelihood before generation. This is a stronger probe than surface text embeddings.

Experiment:

- Run a local model prefill-only pass.
- Extract last-token or pooled hidden states.
- Train linear probes for:
  - each model correct/not-correct;
  - oracle model class;
  - frontier-needed;
  - high-regret risk.
- Compare against text embeddings and local-answer consensus.

Implementation note:

This needs vLLM/transformers instrumentation. It is not available from the existing cached broad100 parquet.

### Confidence And Logprob Signals

References:

- https://openreview.net/forum?id=U08mUogGDM
- https://arxiv.org/html/2605.02241v3

Idea for us:

The local probe should expose uncertainty, not just its final answer.

Experiment:

- Request vLLM logprobs for local probes.
- Compute:
  - mean final-answer token logprob;
  - min token logprob;
  - answer entropy;
  - top-2 margin for multiple-choice;
  - refusal/empty/malformed-output flags.
- Train router with these confidence features.

Expected value:

This may distinguish "local answer is right" from "local answer is plausible but unreliable."

### BEST-Route / Test-Time Compute Routing

Reference:

- https://arxiv.org/abs/2506.22716

Idea for us:

The route action should not be only `model_id`. It should be:

```text
(model_id, output_budget, sample_count, verifier/tool?)
```

Experiment:

- Add actions:
  - local one-shot;
  - local self-consistency n=3;
  - Gemini short;
  - Gemini strong/thinking;
  - GPT only for predicted high-gain cases;
  - code execution verifier for code tasks;
  - deterministic math tool for exact math.

Metrics:

- quality;
- utility;
- frontier-call rate;
- probe-call rate;
- cost normalized to all-GPT;
- p95 latency.

### R2-Router

Reference:

- https://arxiv.org/abs/2602.02823

Idea for us:

Route over output length and reasoning budget. A weaker/cheaper model with the right output budget may beat a stronger default call.

Experiment:

- Sweep max output tokens for local/Gemini/GPT:
  - 96, 192, 384, 768, 1024.
- Add route labels for budget:
  - short exact answer;
  - medium reasoning;
  - code generation;
  - strong solver.

### FineRouter / WebRouter

References:

- https://arxiv.org/html/2603.19415v1
- https://arxiv.org/html/2510.11221v1

Idea for us:

Keep RouteCode's latent-state framing, but make states observable through probes. The labels should be learned from utility patterns, then predicted using cheap evidence.

Experiment:

- Learn utility-vector states on train.
- Predict state from:
  - text only;
  - text + local answers;
  - text + local confidence/logprobs;
  - text + prefill features.
- Report the observability gap at each feature tier.

### Causal / Decision-Aware Routing

References:

- https://arxiv.org/abs/2505.16037
- https://arxiv.org/abs/2602.03478

Idea for us:

Do not train only scalar quality predictors. Train for routing regret/ranking directly.

Experiment:

- Pairwise ranking: for each query, prefer model A over B if utility(A) > utility(B).
- Regret model: predict `oracle_utility - model_utility`.
- Budgeted frontier-gain model: predict value of escalating to Gemini/GPT.
- Evaluate by route utility, not AUC.

## Concrete Main-Thread Experiment Queue

1. Cached-only sandbox:
   - kNN query utility routing.
   - kNN query+local-answer utility routing.
   - local answer vote/margin/entropy features.
   - train-only regret/ranking predictors.
   - budgeted frontier ranking.

2. vLLM logprob probes:
   - rerun small local probe subset with `logprobs`.
   - add confidence/margin features.
   - test whether confidence improves broad100 routing.

3. Prefill/activation probes:
   - use one local model first.
   - train linear probes for correctness/frontier-needed/oracle-state.
   - compare against text embeddings.

4. Test-time compute actions:
   - add local self-consistency;
   - add Gemini short vs Gemini strong;
   - add GPT only under high predicted gain.

5. RouteCode state observability ladder:
   - text only;
   - text + local answers;
   - text + confidence/logprobs;
   - text + prefill activations.

## Next Main-Thread Probe Search: Methods 2, 3, 4, 6

The next direction is still to find a general cheap probe, not a
benchmark-specific trick. Use GPQA, MMLUPro, LiveMathBench, and MATH500 only as
stress-test slices for debugging because the audit shows they expose the
largest current failures.

### 2. Pairwise Preference Probe

Template:

- RouteLLM-style pairwise preference routing.
- Decision-aware routing / pairwise ranking.

Probe idea:

- Use a cheap local model or small classifier to compare candidate answers or
  candidate actions pairwise.
- The probe should answer:

```text
Given query q and cheap evidence e, is action A better than action B?
```

Target labels:

- `utility(A) > utility(B)` from train outcomes.
- Pairwise regret margin: `U(q,A) - U(q,B)`.
- Special pairs:
  - best local vs Gemini strong;
  - best local vs GPT;
  - qwen3-14b vs qwen3-32b;
  - deterministic tool vs local model;
  - local one-shot vs local self-consistency.

Implementation sketch:

- Build train pair rows from cached outcome matrix.
- Features:
  - query text embedding;
  - local parsed answers;
  - answer agreement/vote features;
  - local model IDs and costs;
  - optional short local probe rationale.
- Models to try:
  - logistic/Ridge pairwise ranker first;
  - local vLLM JSON pairwise judge only if the classifier is insufficient.
- Select actions by tournament or Bradley-Terry-style score aggregation.

Success evidence:

- Must improve held-out broad100 cost-aware utility over current policy, not
  only pairwise accuracy.
- Report pairwise accuracy, but treat it as secondary.
- Include route/probe cost if the pairwise judge is an LLM call.

### 3. kNN Over Richer Probe State

Template:

- kNN routing paper.
- Existing cached kNN was too weak because the state was only query text plus
  shallow local answer text.

Probe idea:

- Build a richer cheap probe state and use nearest train queries to average
  utility vectors.
- The key change is that kNN should run over `query + probe state`, not just
  query or parsed final answers.

Candidate probe states:

- local self-consistency transcript summary;
- pairwise preference scores;
- local confidence/logprob features;
- answer vote distribution;
- JSON diagnostic tags such as:
  - `answer_type`;
  - `tool_applicable`;
  - `option_conflict`;
  - `local_answers_plausible`;
  - `needs_more_compute`;
  - `format_risk`.

Implementation sketch:

- Start with cached rows plus any newly collected cheap probe outputs.
- Encode probe state with TF-IDF/embedding.
- For K in `1, 3, 5, 10, 20, 40`, average train utility vectors among nearest
  neighbors.
- Select the model/action with highest estimated utility.
- Fit all vectorizers/neighbors on train only; choose K on validation; report
  test once.

Success evidence:

- Compare against previous cached kNN and `observable_local_state_v5`.
- Report whether richer probe-state kNN improves model-win recall for the
  current hardest oracle winners.

### 4. BEST-Route / Value-Of-Compute Probe

Template:

- BEST-Route / test-time compute routing.
- R2-style budget routing.

Probe idea:

- Route over actions, not only model IDs:

```text
(model_id, output_budget, sample_count, verifier/tool)
```

Candidate actions:

- local one-shot;
- local self-consistency `n=3`;
- deterministic math tool when applicable;
- Gemini short;
- Gemini strong/thinking;
- GPT only when predicted gain is high.

Probe target:

- Predict value of compute:

```text
gain(action_extra) = U(q, action_extra) - U(q, current_base_action)
```

Implementation sketch:

- Use cached Gemini strong rows already collected.
- Add local self-consistency rows first because they are cheap and local.
- Train value-of-compute predictors on train, select thresholds on validation.
- Compare:
  - base policy;
  - always strong;
  - validation-gated strong;
  - local self-consistency gate;
  - combined local-self-consistency then Gemini/GPT gate.

Success evidence:

- Main metric is cost-aware utility ratio to the augmented oracle.
- Report frontier-call rate, strong-call rate, probe rate, normalized remote
  cost, and p95 latency.
- Do not count raw quality gains as success if utility drops.

### 6. Local Self-Consistency Probe

Template:

- FrugalGPT-style cascades.
- Self-consistency as a cheap uncertainty probe.

Probe idea:

- Run a cheap local model multiple times with short outputs.
- Use vote/margin/entropy/format validity as cheap routing evidence.
- This is general across benchmarks and should be evaluated on all broad100,
  with hard slices used only for failure analysis.

Collection plan:

- Use vLLM, not provider APIs.
- Start with one local model that serves reliably:
  - Qwen3-4B or Qwen3-8B for cheapest probe;
  - Qwen3-14B/Qwen3-32B only if cheaper models do not produce useful signal.
- Generate `n=3` short final-answer samples per query.
- Keep max tokens small and parse final answers exactly.

Features:

- majority answer;
- vote count;
- vote margin;
- entropy over normalized answers;
- malformed/refusal/empty rate;
- agreement with existing local candidates;
- whether self-consistency answer equals the current base selected answer.

Policies:

- Use self-consistency answer directly when vote margin is high.
- Use self-consistency features as inputs to:
  - pairwise preference probe;
  - value-of-compute gate;
  - richer-state kNN.

Success evidence:

- Must improve held-out broad100 utility after accounting for local probe
  latency/cost.
- Report how much it reduces wrong-local-winner, missed-strong, and
  unneeded-strong error modes.

### Recommended Order

1. Implement local self-consistency collection because it creates reusable cheap
   evidence for methods 2, 3, and 4.
2. Train a value-of-compute gate from self-consistency features.
3. Add pairwise preference ranking using the same features.
4. Re-run kNN with the richer probe state.
5. Compare all four methods on the same train/validation/test split and same
   augmented model-action pool.

Guardrails:

- No Claude calls.
- GPT fixed to `gpt-5.5`; Gemini fixed to `gemini-3.5-flash`.
- Keep provider spend below `$15` per model.
- Prefer local vLLM probes first.
- Use train only for fitting probe models, kNN indices, codebooks, and
  thresholds; use validation for method/threshold choice; report held-out test
  once.
- Do not weaken the claim into a benchmark-specific rule. The claim is about
  cheap probe information for routing.

## Progress Log

Updated: 2026-06-19.

Detailed memo:

- `results/controlled/PROBE_SIGNAL_RESEARCH_PROGRESS.md`

Completed so far:

- LLMRouterBench-style model-win recall/confusion: done in
  `experiments/130_probe_signal_cached_sandbox.py`.
- kNN query utility routing and kNN query+local-answer utility routing: done in
  `experiments/130_probe_signal_cached_sandbox.py`; validation-selected kNN did
  not beat the current broad100 policy.
- Cached supervised utility regression and budgeted frontier-gain models: done
  in `experiments/131_probe_signal_supervised_cached.py`; Ridge utility
  regression improved full broad100 utility slightly (`0.6869` vs `0.6756`) but
  remains far from oracle utility (`0.7927`).
- Free-generation vLLM logprob probes: collected in
  `experiments/132_vllm_logprob_probe.py` and evaluated in
  `experiments/133_logprob_feature_router.py`; negative result because the
  probe mostly produced truncated reasoning starts.
- Choice-token vLLM logprob probes: collected in
  `experiments/134_vllm_choice_logprob_probe.py` and evaluated in
  `experiments/135_choice_logprob_feature_router.py`; mechanically valid but
  did not improve validation-selected held-out routing.
- Qwen3-4B prefill/activation probe: collected and evaluated in
  `experiments/136_prefill_activation_router.py`; negative result on the hard
  60-row GPQA/MATH500/MMLUPro held-out slice (`0.4774` validation-selected
  activation utility vs `0.5168` cached observable-state reference).
- Observable decision-aware feature router: evaluated in
  `experiments/137_observable_feature_router.py`; negative result on full
  broad100 (`0.6738` validation-selected test utility, best diagnostic
  `0.6827`, oracle `0.7927`).
- Expanded observable-feature router over the self-consistency matrix:
  re-evaluated in `experiments/137_observable_feature_router.py` with
  `--include-text-view --include-tree-learners --include-classifiers` after
  adding a local deduplication safeguard for repeated deterministic tool rows.
  This was also negative: validation selected `utility_ridge_a10_dict`
  (`0.7968` validation utility), but held-out test utility dropped to `0.6820`.
  The best held-out diagnostic row reached only `0.7082`, essentially tied with
  benchmark lookup (`0.7074`) and below the earlier self-consistency feature
  gate (`0.7155`).
- Targeted frontier-need predictor: evaluated in
  `experiments/157_frontier_need_predictor.py` using the cached
  self-consistency matrix and cached GPT/Gemini/Gemini-strong rows. This
  sharpened the target but did not solve deployable routing. The diagnostic
  local-vs-frontier oracle reached `0.8163` held-out utility and `0.8547`
  quality with only `0.2267` frontier calls, close to the full cost-aware
  oracle (`0.8463`). But validation-selected Ridge frontier-gain prediction
  reached only `0.7076` held-out utility, validation-selected logistic
  frontier-need classification reached only `0.6933`, and the best held-out
  non-oracle diagnostic row reached only `0.7175` with `0.5000` frontier calls.
- Local vLLM binary frontier-need probe: evaluated in
  `experiments/158_vllm_frontier_need_probe.py` with `Qwen/Qwen3-14B-AWQ`
  served by vLLM. It collected `264` local probe rows with `100%` success and
  no GPT/Gemini/Claude API calls. The signal had high recall but low precision
  against the local-vs-frontier oracle: held-out test TP `34`, FP `56`, FN `5`,
  TN `37`, precision `0.3778`, recall `0.8718`. Validation selected the
  no-frontier local reference, and the best held-out probe row reached only
  `0.6490` utility while using `0.4826` frontier calls. Benchmark-specific
  validation thresholding improved this to `0.6807` utility with `0.2442`
  frontier calls, still below the earlier self-consistency feature gate.
- Cached test-time-compute action routing with Gemini strong-solve: evaluated
  in `experiments/138_cached_test_time_compute_router.py`; best non-diagnostic
  held-out gate improved quality (`0.7733` vs base `0.7442`) and slightly
  improved utility (`0.7028` vs base `0.6962`), but validation-selected strong
  gate did not generalize (`0.6905`) and all remain far from augmented oracle
  utility (`0.8404`). Diagnostic base-vs-strong oracle reached `0.7707`.
- Stronger local Qwen3-32B-AWQ option-token probe: collected with vLLM in
  `results/controlled/broad100_qwen32_choice_logprob_probe` and evaluated in
  `results/controlled/broad100_qwen32_choice_logprob_feature_router`; negative
  result because held-out probe accuracy was only `0.3000` on both GPQA and
  MMLUPro and choice features did not beat no-choice baselines.
- Qwen3-32B-AWQ chat/no-thinking option-token variants: collected in
  `results/controlled/broad100_qwen32_chat_choice_logprob_probe` and
  `results/controlled/broad100_qwen32_chat_nothink_choice_logprob_probe`;
  negative result because chat thinking starts with `<think>` and no-thinking
  chat still often starts with non-option tokens, so router rows again did not
  beat no-choice baselines.
- Qwen3.6-35B-A3B vLLM attempt: `scripts/start_vllm_qwen3_6_35b_a3b.sh` did
  not keep a live `/v1/models` endpoint under the current single-GPU setup, so
  no probe rows were collected.
- Cached strong-gain regressor gate: evaluated in
  `experiments/140_cached_strong_gain_regressor_gate.py`; best held-out
  non-oracle utility was only `0.6966` while the diagnostic base-vs-Gemini
  strong oracle reached `0.7775`, so the target is valuable but shallow
  observable features still cannot predict it.
- Train-supervised Gemini strong-gain gate: collected train-split Gemini
  strong-solve rows in
  `results/controlled/broad100_gemini_strong_solver_train` and evaluated in
  `experiments/141_train_supervised_strong_gain_gate.py`. This fixed the prior
  protocol limitation by training gain predictors on train and selecting
  thresholds on validation. Best held-out non-oracle utility was `0.7062`
  (`0.7849` quality) against augmented oracle utility `0.8404` (`0.8663`
  quality), so train strong labels helped only modestly and still did not meet
  the target.
- Qwen3-32B-AWQ answer-verifier probe: collected in
  `experiments/142_vllm_answer_verifier_strong_gate.py`; the direct
  `escalate` verdict gate was negative (`0.5546` held-out utility) because it
  over-escalated to Gemini strong on cases where strong was not cost-effective.
- Verifier-feature strong-gain gate: evaluated in
  `experiments/143_verifier_feature_strong_gain_gate.py`; reusing the Qwen32
  verifier outputs as supervised train features improved over the direct
  verdict gate, but best held-out verifier-feature utility was only `0.6919`,
  still below the train-prior strong-gain result and far from the oracle.
- Probe error-mode audit: evaluated in
  `experiments/144_probe_error_mode_audit.py`; current augmented broad100 loss
  is concentrated in GPQA, MMLUPro, LiveMathBench, and MATH500, with a mix of
  wrong local winners, unneeded Gemini strong/API calls, and missed Gemini
  strong calls.
- GPT-5.5 local-only answer adjudicator: evaluated in
  `experiments/128_broad_answer_adjudicator.py` with `--candidate-mode
  local_only`; negative result because the best held-out selected-solver
  utility was only `0.6528` and route-cost-inclusive utility was only `0.4448`.
  A Gemini local-only adjudicator attempt hit HTTP `429` before writing cached
  rows.
- GPQA/MMLUPro option-sanity local-winner patch: evaluated in
  `experiments/145_option_sanity_local_winner.py`; negative result because
  validation still selected the unpatched base policy and the learned
  cached-answer patch degraded held-out utility. Diagnostic oracle patch rows
  show real headroom (`0.7828` augmented utility for local+strong oracle), so a
  better probe signal is still useful.
- Qwen3-32B-AWQ local self-consistency probe: collected and evaluated in
  `experiments/147_vllm_self_consistency_probe.py`; naive vote-threshold
  policies were negative, but the base/self/strong diagnostic oracle reached
  `0.8076` held-out utility and `0.8430` quality after adding the
  self-consistency action.
- Train-supervised self-consistency feature gate: evaluated in
  `experiments/148_self_consistency_feature_gate.py`; this is a modest positive
  result. Validation-selected held-out utility improved from `0.6962` to
  `0.7155`, and quality improved from `0.7442` to `0.8023`, but the method is
  still far from the expanded oracle utility `0.8463`.
- Pairwise self-consistency router: evaluated in
  `experiments/149_pairwise_self_consistency_router.py`; validation-selected
  held-out utility was only `0.6982`, and the best held-out diagnostic was
  `0.7278`, so pairwise margin modeling did not beat the direct
  self-consistency feature gate.
- Richer-state kNN over self-consistency features: evaluated in
  `experiments/150_richer_state_knn_self_consistency.py`; best held-out
  diagnostic action-kNN utility reached `0.7239`, but validation-selected kNN
  fell to `0.6862` on test because it over-escalated to frontier/strong
  actions. Direct model kNN was worse (`0.5287` validation-selected held-out
  utility).
- LiveMathBench/MATH500 math-verifiability patch: evaluated in
  `experiments/151_math_verifiability_patch.py`; validation-selected held-out
  utility fell from `0.6962` to `0.6834`, and the math-only slice fell from
  `0.6293` to `0.5742`. The patch reduced frontier/strong calls but removed
  too many useful Gemini/GPT math calls.
- Calibrated self-consistency action gate: evaluated in
  `experiments/152_calibrated_self_consistency_action_gate.py`; the best
  validation-selected capped gate reached only `0.7045` held-out utility
  (`0.7733` quality), and the best held-out diagnostic calibration row reached
  `0.7178`. This does not beat the earlier self-consistency feature gate, while
  the base/self/strong diagnostic oracle remains `0.8076`.
- Benchmark-stratified policy selector: evaluated in
  `experiments/153_benchmark_stratified_policy_selector.py`; validation
  per-benchmark composition overfits. `benchmark_val_best` reaches `0.8455`
  validation utility but only `0.7061` held-out utility, and shrink-to-global
  variants do not recover the global gate. The diagnostic test-best
  per-benchmark composition reaches only `0.7481`, so the existing policy
  library is still not sufficient.
- Strong/self need classifier gate: evaluated in
  `experiments/154_strong_need_classifier_gate.py`; binary high-precision
  strong/self classifiers over cached self-consistency features are negative.
  The validation-selected classifier falls to `0.6727` held-out utility; the
  best held-out diagnostic classifier reaches `0.7217`, below the best scalar
  self-consistency feature diagnostic.
- Local vLLM action-compare probe: evaluated in
  `experiments/155_vllm_action_compare_probe.py`; Qwen3-4B via vLLM produced
  valid JSON actions for all 160 GPQA/MMLUPro/LiveMathBench/MATH500 val/test
  stress rows. The original prompt's best held-out non-oracle policy reached
  only `0.6593` utility versus `0.6585` for the base policy. A conservative
  prompt fixed the "never choose base" pathology but still reached only
  `0.6597` utility and never identified useful `strong` calls. The diagnostic
  base/self/strong action-set oracle remained much higher at `0.8016` utility,
  so local answer comparison does not solve action prediction. A first
  Qwen3-32B-AWQ vLLM action-compare attempt timed out on the first request and
  is treated as a serving/prompt-size feasibility failure, not as a result.
- Action-compare feature gate: evaluated in
  `experiments/156_action_compare_feature_gate.py` after collecting train rows
  for the conservative Qwen3-4B action probe. The action-probe output has weak
  diagnostic signal: the best held-out diagnostic row reaches `0.7058` utility
  and `0.7907` quality, but it uses `0.4709` frontier calls and remains below
  the earlier self-consistency feature gate. Validation selection does not
  generalize: the validation-selected row drops to `0.6513` held-out utility,
  below the `0.6585` base policy.
- Qwen3-14B-AWQ local action-compare probe: evaluated in
  `experiments/155_vllm_action_compare_probe.py` with the conservative prompt.
  It emitted more `strong` actions than Qwen3-4B, but those calls were not
  cost-effective on held-out test. Validation selected
  `vllm_action_compare_direct_t0.85`, which reached only `0.6527` held-out
  utility; the best non-oracle held-out row was still the base policy
  (`0.6585`).
- Expanded observable-feature supervised routing over the self-consistency
  matrix: evaluated in `experiments/137_observable_feature_router.py`; negative
  result. More learner capacity over the same observable local signals did not
  beat the earlier train-supervised self-consistency feature gate.
- Targeted frontier-need prediction: evaluated in
  `experiments/157_frontier_need_predictor.py`; negative as a deployable method
  but important diagnostically. The local-vs-frontier oracle is strong and
  sparse, but the current observable/query/self-consistency features do not
  predict frontier value reliably.
- Qwen3-14B-AWQ local binary frontier-needed judgment: evaluated in
  `experiments/158_vllm_frontier_need_probe.py`; negative. The local model
  catches many true frontier-needed cases but over-escalates too many local
  cases, so cost-aware utility is worse than the target. Benchmark-specific
  calibration helps but does not close the gap.
- Qwen3-14B-AWQ frontier-needed precision filter: evaluated in
  `experiments/159_vllm_frontier_precision_filter.py`; negative. The Qwen14
  probe table now has train/validation/test rows, and train-fitted logistic and
  Ridge filters were selected on validation. The validation-selected gain
  filter fell to `0.6590` held-out utility, the validation-selected logistic
  filter fell to `0.6346`, and the best held-out diagnostic filter reached only
  `0.6758`, still below the earlier self-consistency feature gate and far below
  the `0.8163` local-vs-frontier diagnostic oracle.
- Local E5 embedding frontier-need router: evaluated in
  `experiments/160_embedding_frontier_need_router.py`; negative/partial. It
  uses cached `intfloat/e5-small-v2` sentence embeddings with no provider or
  vLLM calls. Validation-selected Ridge reaches only `0.6603` held-out utility
  and `0.7035` quality; the best test-picked diagnostic row reaches `0.7011`
  utility and `0.7616` quality, still below the earlier self-consistency
  feature gate and far below the `0.8163` local-vs-frontier oracle.
- Embedding-augmented self-action gate: evaluated in
  `experiments/161_embedding_self_action_gate.py`; tiny positive but not a
  solution. Adding cached E5 embeddings to the base/self/strong action gate
  raises the validation-selected held-out utility from the previous `0.7155` to
  `0.7161` with the same `0.8023` quality. The best test-picked diagnostic row
  reaches `0.7341` utility, but the base/self/strong action-set oracle remains
  `0.8076`, so the method is still far outside the 3% oracle target. A cached
  `all-mpnet-base-v2` repeat generalizes worse (`0.7102` selected held-out
  utility, `0.7324` best diagnostic), so E5 remains the better embedding
  variant.
- Pairwise action ranker over cached self-consistency evidence: evaluated in
  `experiments/162_pairwise_action_ranker.py`; modest positive but not a
  solution. The validation-selected pairwise logistic row reaches `0.7235`
  held-out utility, improving over the E5 self-action gate (`0.7161`) while
  using fewer frontier/strong calls (`0.2616` frontier, `0.2267` strong).
  However, the base/self/strong action-set oracle remains `0.8076`, leaving a
  `0.0841` utility gap. Pairwise Ridge overfits validation (`0.8176` validation
  utility, `0.7110` held-out utility), so pairwise ranking alone does not close
  the observability gap.
- Residual confidence rules over the pairwise action ranker: evaluated in
  `experiments/163_residual_confidence_rule_policy.py`; incremental positive
  but still not a solution. The strict validation-best rule reaches `0.7259`
  held-out utility and `0.7907` quality. A validation near-best rule with a
  cost/frontier tiebreak reaches `0.7328` held-out utility, `0.7849` quality,
  `0.8660` oracle-utility ratio, `0.2500` frontier-call rate, `0.2151`
  strong-call rate, and `0.5000` self-action rate. This is the current best
  validation-driven cached policy, but it remains far below the `0.8016`
  base/self/strong action-set oracle.
- Qwen3-14B-AWQ residual-risk vLLM probe: evaluated in
  `experiments/164_vllm_residual_risk_probe.py`; negative. It collected
  `160/160` local vLLM judgments over GPQA, MMLUPro, MATH500, and
  LiveMathBench validation/test rows. Validation still selects the residual-rule
  baseline (`0.8015` validation utility, `0.7328` held-out utility), and active
  thresholds either tie the baseline by making no effective changes or reduce
  held-out utility, with the worst swept row at `0.6696`. The same local model
  family does not provide a useful higher-level residual-risk signal.
- Frontier agreement probe policy: evaluated in
  `experiments/165_frontier_agreement_probe_policy.py`; negative after probe
  cost accounting. Cached Gemini/GPT answer agreement with local models is a
  real correctness signal, but the selected probe policies do not beat the
  residual-rule baseline. Validation selects a Gemini agreement row that reaches
  `0.7309` held-out utility and a GPT agreement row that reaches `0.7311`, both
  below the `0.7328` residual-rule baseline. The best held-out diagnostic
  Gemini agreement row reaches `0.7359`, but it is test-picked and still far
  below the `0.8016` base/self/strong action-set oracle.
- SLM/LLM early-signal probe pilot: evaluated in
  `experiments/166_slm_llm_early_signal_probe_pilot.py`; partial diagnostic
  positive, not a deployable solution. The output folder
  `results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/`
  contains the requested oracle target table, threshold policy tables,
  precision-at-cap table, figure, vLLM answerability cache, and memo. The
  binary target compares the best cached local action against the best cached
  large action using `U = quality - 0.35 * normalized_cost`. On held-out test,
  always-local utility is `0.6919`, always-large utility is `0.7689`, and the
  diagnostic local-vs-large oracle is `0.8463`. Validation selects a
  threshold-only `signal_combined_mean_risk` rule; on held-out test it reaches
  `0.7381` utility, `0.7616` quality, oracle-utility ratio `0.8721`,
  recovered gap vs local `0.2992`, large-call rate `0.7733`, and frontier-call
  rate `0.1860`. The best test-picked diagnostic threshold reaches `0.7825`
  utility, but it is not deployable evidence. Early-rollout instability and
  semantic uncertainty have useful sparse precision at the 10% cap (`0.5294`
  precision, `0.2903` recall). SLM-vs-medium divergence is weak, and the
  Qwen3-14B-AWQ vLLM one-token answerability probe is noisy and weak (`344/344`
  successful val/test calls, but held-out 10% cap precision `0.0588`, AUROC
  `0.5569`). This keeps the main bottleneck as observable evidence for when
  upward routing is cost-effective.
- Constrained YES/NO local logit probe: evaluated in
  `experiments/167_constrained_yesno_probe_policy.py`; partial positive. The
  Qwen3-14B-AWQ vLLM probe uses `logit_bias` to constrain the next token to
  single-token YES/NO variants, avoiding the noisy free-token answerability
  outputs from experiment 166. It collected `688/688` successful val/test
  prompts over two modes: query-only and local-evidence. The single-signal
  validation-selected query-only threshold reaches held-out utility `0.7573`,
  quality `0.7791`, oracle-utility ratio `0.8949`, large-call rate `0.5698`,
  and frontier-call rate `0.1221`. This is better than the previous selected
  threshold-only pilot (`0.7381`) but still below always-large (`0.7689`).
  Local-evidence risk has a better held-out diagnostic row (`0.7861` utility,
  `0.8256` quality, `0.9289` oracle-utility ratio), but it is test-picked and
  cannot be claimed as the deployed method.
- Constrained YES/NO combo policy: evaluated in
  `experiments/168_constrained_yesno_combo_policy.py`; modest deployable
  positive, still below target. It makes no model calls and searches AND/OR,
  weighted, and cap rules over the cached constrained scores, selecting on
  validation. The validation-selected two-signal AND rule reaches held-out
  utility `0.7756`, quality `0.7965`, oracle-utility ratio `0.9165`, recovered
  gap vs local `0.5425`, large-call rate `0.4128`, frontier-call rate
  `0.1163`, precision `0.2676`, and recall `0.6129`. This beats always-large
  utility (`0.7689`) while using far fewer frontier calls, but it is still well
  below the local-vs-large oracle (`0.8463` utility, `0.8721` quality). The
  best test-picked combo reaches `0.7828` utility and remains diagnostic only.
- Benchmark-aware YES/NO threshold policy: evaluated in
  `experiments/169_benchmark_aware_yesno_policy.py`; negative. Before the
  policy run, the constrained Qwen3-14B-AWQ YES/NO cache was extended to train
  rows, producing `1720/1720` successful cached prompt rows across train,
  validation, and held-out test. The benchmark-aware policy itself makes no
  model calls and fits per-benchmark threshold/cap rules from train or
  train+validation scores. Validation selects `benchmark_cap_c_0.25`, which
  reaches held-out utility `0.7576`, quality `0.7733`, oracle-utility ratio
  `0.8952`, large-call rate `0.2500`, and frontier-call rate `0.0988`. The
  best held-out diagnostic row reaches `0.7723` utility and `0.8081` quality,
  but it is test-picked and does not beat the previous validation-selected
  combo. Benchmark-specific raw thresholds overfit and do not close the
  observability gap.
- Benchmark-composed fixed YES/NO policy: evaluated in
  `experiments/170_benchmark_composed_yesno_policy.py`; current best
  validation-selected result for the local-vs-large diagnostic branch, but
  still below target. It makes no model calls and selects a fixed policy per
  benchmark on validation: always local, always large, the global constrained
  AND rule, or capped variants. Validation selects
  `benchmark_composed_eps0_utility`; on held-out test it reaches utility
  `0.7892`, quality `0.8140`, oracle-utility ratio `0.9325`, recovered gap vs
  local `0.6303`, large-call rate `0.4419`, frontier-call rate `0.1744`,
  precision `0.3158`, and recall `0.7742`. This improves over experiment 168
  (`0.7756` utility, `0.9165` oracle ratio), but still misses the Phase 3
  thresholds: 95% of the `0.8463` oracle utility is about `0.8040`, and within
  three quality points of the `0.8721` oracle quality requires at least
  `0.8421`. The best test-picked diagnostic composed row reaches `0.7978`
  utility and `0.8256` quality (`0.9427` oracle ratio), but it is not
  validation-selected and still below target.
- Tool-aware benchmark-composed fixed YES/NO policy: evaluated in
  `experiments/171_tool_aware_benchmark_composed_policy.py`; first
  validation-selected result that clears the local-vs-large diagnostic
  thresholds. It adds one route-time signal to experiment 170:
  deterministic exact-math `tool_available`. Tool-aware candidate policies
  force the local side when this tool produces a non-empty answer. The
  target-aware validation selector requires validation oracle-utility ratio
  `>=0.95`, validation quality gap `<=0.03`, and frontier-call rate `<=0.40`,
  then tie-breaks by validation quality, need-large recall, utility, and cost.
  It selects `tool_aware_benchmark_composed_eps0.01_recall_then_quality`.
  On held-out test this reaches utility `0.8163` with bootstrap CI
  `[0.7601, 0.8674]`, quality `0.8488`, oracle-utility ratio `0.9646`,
  recovered gap vs local `0.8061`, large-call rate `0.6395`, frontier-call
  rate `0.1860`, precision `0.2455`, and recall `0.8710`. Against the
  local-vs-large oracle (`0.8463` utility, `0.8721` quality), this clears both
  the 95%-utility threshold (`0.8039`) and the within-3-quality-points
  threshold (`0.8421`). Caveat: this is still a local-vs-large diagnostic
  target where choosing local means the target table's best local side. The
  next step is to replace that diagnostic local side with an actual local
  action selector.
- Tool-aware deployed-action bridge: evaluated in
  `experiments/172_tool_aware_deployed_action_policy.py`; negative for the
  actual multi-action router. The run writes
  `results/controlled/broad100_tool_aware_deployed_action_policy/TOOL_AWARE_DEPLOYED_ACTION_POLICY_MEMO.md`
  plus selected/all policy tables and query-level choices. It makes no GPT,
  Gemini, Claude, or vLLM calls and uses cached outputs only. The bridge
  replaces the 171 diagnostic best-local/best-large sides with concrete action
  selection from train-only action priors, deterministic tool availability,
  local answer agreement, and validation-selected thresholds. The best
  validation-selected deployable policy,
  `tool_then_171_gate_local_consensus_large_prior`, reaches held-out utility
  `0.7058`, quality `0.7849`, oracle-utility ratio `0.8340`, frontier-call
  rate `0.4709`, and strong-or-frontier call rate `0.8081`. The best held-out
  diagnostic threshold row reaches `0.7261` utility and `0.8140` quality, but
  it is test-picked. The full cost-aware oracle remains `0.8463` utility and
  `0.8721` quality, so no deployable 172 row is within 3% of oracle utility or
  within 3 quality points of oracle. A key diagnostic row,
  `oracle_local_vs_large_gate_train_prior`, uses the true local-vs-large label
  but train-only concrete action priors and still reaches only `0.7254`
  utility. This means the remaining bottleneck is selecting the concrete
  action/answer, not merely detecting when larger models are useful.
- Benchmark-composed deployed-action policy: evaluated in
  `experiments/173_benchmark_composed_deployed_action_policy.py`; negative.
  The run writes
  `results/controlled/broad100_benchmark_composed_deployed_action_policy/BENCHMARK_COMPOSED_DEPLOYED_ACTION_POLICY_MEMO.md`
  plus selected/all policy tables, per-benchmark choices, query choices, and a
  utility figure. It makes no GPT, Gemini, Claude, vLLM, or other model calls
  and uses cached outputs only. The experiment chooses a concrete action policy
  per benchmark on validation from train-only priors, fixed concrete models,
  tool-first variants, local consensus, 171 gates, and selected 172 threshold
  rules. The validation utility selector
  `benchmark_composed_deployed_eps0_utility` reaches held-out utility `0.6916`,
  quality `0.7558`, oracle-utility ratio `0.8172`, frontier-call rate
  `0.4651`, and strong-or-frontier call rate `0.6744`. The target-quality
  validation selector reaches held-out utility `0.7048`, quality `0.7907`, and
  oracle-utility ratio `0.8328`. The best held-out composed diagnostic row
  reaches `0.7097` utility and `0.7849` quality, but it is test-picked and
  still below the simpler 172 deployed policy comparison row (`0.7116`
  utility, `0.7907` quality). This confirms that benchmark-level composition
  over existing concrete action heuristics is not enough. The next probe should
  gather stronger evidence about answer correctness or action identity, not
  another table-level policy composition.
- Train-calibrated answer-support action policy: evaluated in
  `experiments/174_answer_support_action_policy.py`; negative, with a useful
  diagnostic. The run writes
  `results/controlled/broad100_answer_support_action_policy_benchmark_threshold/ANSWER_SUPPORT_ACTION_POLICY_MEMO.md`
  plus all/selected policy tables, query choices, train support features, and a
  utility figure. It makes no GPT, Gemini, Claude, vLLM, or other model calls.
  The method fits train-only local answer-group reliability tables and selects
  global or per-benchmark support thresholds on validation. The best
  validation-selected benchmark-threshold row reaches held-out utility
  `0.7035`, quality `0.7849`, and oracle-utility ratio `0.8313`; the best
  validation-selected global support-threshold row reaches `0.7054` utility
  and `0.7907` quality. The best held-out support diagnostic row reaches
  `0.7130` utility and `0.7965` quality, but it is not validation-selected and
  still far below the oracle (`0.8463` utility, `0.8721` quality). The
  diagnostic value is that local answer support is reliable for some math/code
  slices but noisy or anti-informative for GPQA, MMLUPro, and AIME. This closes
  the simple answer-support branch as insufficient.
- Public-test verifier policy for code tasks: evaluated in
  `experiments/175_public_test_verifier_policy.py`; partial positive but still
  below target. The run writes
  `results/controlled/broad100_public_test_verifier_policy/PUBLIC_TEST_VERIFIER_POLICY_MEMO.md`
  plus all/selected policy tables, query choices, code verifier coverage, and a
  utility figure. It makes no GPT, Gemini, Claude, vLLM, or other model calls.
  HumanEval and MBPP prompts include public tests, so the route-time signal is
  whether each cached local code action passes those tests. Held-out coverage
  is strong inside code: any passing local action exists for `90%` of HumanEval
  and `80%` of MBPP, with mean passing-local counts `3.65` and `2.45`. The best
  validation-selected deployable row, `code_public_test_else_train_benchmark_prior`,
  reaches held-out utility `0.7132`, quality `0.8081`, and oracle-utility ratio
  `0.8428`. The best held-out diagnostic threshold row reaches `0.7261`
  utility and `0.8140` quality, but it is test-picked. This is better than the
  support-only branch but still far below oracle (`0.8463` utility, `0.8721`
  quality). Interpretation: task-specific verification is useful, but code is
  not the dominant remaining gap. The next verifier should target GPQA,
  MMLUPro, AIME, and exact math.
- Activation-anomaly threshold policy: evaluated in
  `experiments/176_activation_anomaly_threshold_policy.py`; negative for the
  deployable system and weak as a diagnostic. The run writes
  `results/controlled/broad100_activation_anomaly_threshold_policy/ACTIVATION_ANOMALY_POLICY_MEMO.md`
  plus activation feature, all/selected policy, query-choice, and utility-figure
  artifacts. It makes no GPT, Gemini, Claude, vLLM, or local model calls and
  reuses the cached Qwen3-4B prefill activation table. Scope is GPQA, MATH500,
  and MMLUPro only (`60` held-out test queries). The activation-subset oracle is
  high: held-out local utility `0.6333`, always-large utility `0.8244`, and
  local-vs-large/full cost-aware oracle utility `0.9145` with quality `0.9667`.
  The validation-selected local-vs-large threshold reaches only held-out utility
  `0.7770`, quality `0.8333`, and oracle-utility ratio `0.8497`. The
  validation-selected deployed threshold is worse (`0.5816` utility, `0.7333`
  quality, `0.6359` oracle ratio). The best held-out activation diagnostic row
  approximately matches always-large (`0.8248` utility vs `0.8244`) but is
  test-picked. Interpretation: cheap activation anomaly and activation-neighbor
  summaries do not provide enough observable state to choose concrete actions;
  stronger task-specific verifier evidence remains more promising than another
  shallow activation threshold.
- Candidate-correctness ranker policy: evaluated in
  `experiments/177_candidate_correctness_ranker_policy.py`; partial positive
  for concrete action identity, still below target. The run writes
  `results/controlled/broad100_candidate_correctness_ranker_policy/CANDIDATE_CORRECTNESS_RANKER_MEMO.md`
  plus train-CV, all-policy, selected-policy, query-choice, and utility-figure
  artifacts. It makes no GPT, Gemini, Claude, vLLM, or local model calls. The
  method trains small candidate-level correctness regressors on train rows only
  and uses train group-CV for the main configuration. The train-CV selected
  `hgb_l2_all_rank_pen0.25` reaches held-out selected-solver utility `0.7523`,
  quality `0.8198`, and oracle-utility ratio `0.8890`; however, all-rank mode
  requires observing frontier candidate answers, so charging candidate
  generation gives utility only `0.1347`. The best held-out practical/gated
  diagnostic row, `hgb_l1_gate_rank_localplus_pen0.25`, reaches utility
  `0.7529`, quality `0.8140`, oracle-utility ratio `0.8896`, frontier-call
  rate `0.4012`, and strong-or-frontier rate `0.7151`, but it is test-picked
  and still below the broad100 target thresholds (`0.8209` utility, `0.8421`
  quality). Interpretation: supervised candidate correctness is better than
  the prior deployed-action bridge, but the remaining gap needs stronger
  verifier/checker evidence for GPQA, MMLUPro, AIME, and exact math rather than
  another shallow ranking signal.
- Answer-group verifier policy: evaluated in
  `experiments/178_answer_group_verifier_policy.py`; negative relative to the
  candidate-ranker branch. The run writes
  `results/controlled/broad100_answer_group_verifier_policy/ANSWER_GROUP_VERIFIER_POLICY_MEMO.md`
  plus all/selected policy, query-choice, reliability, and utility-figure
  artifacts. It makes no GPT, Gemini, Claude, vLLM, or local model calls. The
  method calibrates train-only local answer-group reliability by benchmark,
  support count, and strong-local signature, then selects thresholds on
  validation. The validation-selected frontier-<=0.40 row reaches held-out
  utility `0.7123`, quality `0.7907`, oracle-utility ratio `0.8417`, and
  frontier-call rate `0.3837`. The validation-selected frontier-<=0.45 row
  reaches held-out utility `0.7168`, quality `0.7965`, oracle-utility ratio
  `0.8471`, and frontier-call rate `0.4012`. The best held-out diagnostic row
  reaches utility `0.7198` and quality `0.8081`, but it is test-picked. This
  does not beat the practical candidate-ranker diagnostic (`0.7529` utility)
  and remains far below target. Interpretation: local answer support/agreement
  is exhausted as a standalone signal; the next useful branch needs stronger
  task-specific checking, not another answer-support threshold.
- Cached adjudicator blend policy: evaluated in
  `experiments/179_cached_adjudicator_blend_policy.py`; partial diagnostic
  only, still below target. The run writes
  `results/controlled/broad100_cached_adjudicator_blend_policy/CACHED_ADJUDICATOR_BLEND_POLICY_MEMO.md`
  plus all/selected policy, query-choice, and utility-figure artifacts. It
  makes no GPT, Gemini, Claude, vLLM, or local model calls. The method overlays
  existing cached answer-adjudicator tables on top of practical
  candidate-ranker policies and reports route-cost-charged utility by
  normalizing adjudicator cost against the mean GPT solver cost. The
  validation-selected frontier-<=0.40 solver-utility row reaches held-out
  utility `0.7190`, quality `0.7558`, oracle-utility ratio `0.8496`,
  frontier-call rate `0.2326`, and route-cost-charged utility `0.6597`.
  Selecting by route-cost-charged validation utility instead reaches held-out
  utility `0.7414`, route-cost-charged utility `0.7386`, quality `0.8023`,
  oracle-utility ratio `0.8761`, and frontier-call rate `0.3953`. The best
  held-out diagnostic row reaches utility `0.7598`, route-cost-charged utility
  `0.7522`, quality `0.8140`, oracle-utility ratio `0.8978`, and frontier-call
  rate `0.3430`, but it is test-picked. Interpretation: generic cached answer
  adjudication gives a small diagnostic gain over the practical candidate
  ranker, but validation selection and route-cost accounting keep it far below
  the broad100 targets. The next useful branch should be task-specific
  checking/verifying, not another generic adjudicator threshold.
- Benchmark policy portfolio: evaluated in
  `experiments/180_benchmark_policy_portfolio.py`; negative for deployable
  composition and useful as a ceiling audit. The run writes
  `results/controlled/broad100_benchmark_policy_portfolio/BENCHMARK_POLICY_PORTFOLIO_MEMO.md`
  plus library-eval, all/selected portfolio, portfolio-map, query-choice, and
  utility-figure artifacts. It makes no GPT, Gemini, Claude, vLLM, or local
  model calls. The library contains `821` existing candidate policy methods and
  `282424` query-choice rows. Deployable validation selection excludes
  diagnostic policy families and chooses one policy per benchmark. The best
  validation-selected route-cost objective reaches held-out utility `0.7320`,
  route-cost-charged utility `0.7292`, quality `0.7849`, oracle-utility ratio
  `0.8650`, and frontier-call rate `0.3663`. The validation-selected raw-
  utility portfolio reaches only held-out utility `0.7094` and quality
  `0.7326`. The test-picked full-library diagnostic ceiling reaches utility
  `0.7823`, route-cost-charged utility `0.7676`, quality `0.8314`, oracle-
  utility ratio `0.9244`, and frontier-call rate `0.2616`, but it is still
  below target. Interpretation: benchmark-specific composition of current
  methods is exhausted. The target cannot be reached by recombining current
  thresholds/rankers/adjudicator overrides; the method needs genuinely new
  task-specific verifier evidence.
- Task-specific verifier-answer action: evaluated in
  `experiments/181_task_specific_verifier_action.py`; new provider-probe
  branch, but not successful. The Gemini run in
  `results/controlled/broad100_task_specific_verifier_action/` attempted hard
  GPQA/MMLUPro/AIME/LiveMathBench/MATH500 slices but every Gemini request
  returned HTTP `429`, so no valid Gemini verifier rows were produced. The
  GPT-5.5 MCQ run in
  `results/controlled/broad100_task_specific_verifier_action_gpt_mcq_512/`
  used `512` max output tokens on GPQA and MMLUPro validation/test. It made
  `80` valid calls with final-run cost `$1.0940`; including GPT smoke runs,
  recorded valid GPT verifier spend for this branch is `$1.7712`, below the
  `$15` cap. Verifier quality is split by benchmark: GPQA validation `0.45`,
  GPQA test `0.20`, MMLUPro validation `0.75`, and MMLUPro test `0.80`.
  Validation-selected routing still fails: `task_verifier_conf_ge_0.95` reaches
  held-out utility `0.7323`, probe-cost-charged utility `0.6088`, quality
  `0.8198`, and frontier-call rate `0.4302`; the probe-cost-selected
  `task_verifier_conf_ge_0.85` reaches held-out utility `0.7201`, probe-cost
  utility `0.6183`, quality `0.8256`, and frontier-call rate `0.4360`.
  Interpretation: task-specific verifier evidence is the right direction, but
  this GPT verifier is too expensive and too weak on GPQA. MMLUPro may support
  a narrower verifier policy, but GPQA needs a different signal.
- Cached verifier-supported-action policy: evaluated in
  `experiments/182_cached_verifier_support_policy.py`; cached follow-up to the
  GPT verifier branch. It reuses the cached GPT-5.5 verifier rows and makes no
  new GPT, Gemini, Claude, vLLM, or local model calls. Instead of selecting the
  verifier's own answer, it uses the verifier's `supported_model` field as
  evidence for selecting an existing cached candidate action. The signal is
  asymmetric: held-out valid support covers `0.20` of GPQA and `0.75` of
  MMLUPro. Validation selects
  `verifier_supported_support_local_only_thr0_benchmmlupro`, which reaches
  held-out utility `0.7569`, quality `0.8140`, oracle-utility ratio `0.8944`,
  frontier-call rate `0.3663`, and probe-cost-charged utility `0.6960`. The
  candidate-ranker reference is `0.7529` utility with no probe cost. The best
  held-out non-oracle raw row reaches utility `0.7635`, but is test-picked and
  falls to `0.6076` after charging GPT verifier cost. Interpretation:
  supported-action evidence is better than using the verifier as the final
  answer, but the raw gain is too small to justify a GPT verifier probe. The
  next verifier/checker branch must be cheaper or more targeted.
- Local-safe gain gate: evaluated in
  `experiments/183_local_safe_gain_gate.py`; cached local-consensus branch. It
  makes no new GPT, Gemini, Claude, vLLM, or local model calls. The method
  trains a train-only gain predictor to decide when the practical
  candidate-ranker action should be replaced by a local consensus action.
  Validation selects `pred_rf_thr-0.0288`; on held-out test it reaches utility
  `0.7625`, quality `0.8140`, oracle-utility ratio `0.9011`,
  frontier-call rate `0.3198`, and strong-or-frontier rate `0.6221`, improving
  over the candidate-ranker reference utility `0.7529` while preserving
  quality and reducing frontier calls. The best held-out diagnostic policy
  reaches utility `0.7724` and quality `0.8198`. The diagnostic oracle between
  the base action and local consensus reaches utility `0.7936`, quality
  `0.8314`, oracle-utility ratio `0.9378`, and frontier-call rate `0.1744`.
  Interpretation: cheap local consensus is useful for suppressing
  over-escalation, but even its base-vs-consensus oracle remains below the
  full broad100 target. The remaining gap needs stronger correctness evidence,
  not only local agreement.
- Strict MCQ verifier policy: evaluated in
  `experiments/184_strict_mcq_verifier_policy.py`; reruns a narrower GPT-5.5
  answer-only multiple-choice verifier on GPQA and MMLUPro with
  `reasoning.effort=none` where supported and `128` max output tokens. It uses
  no Claude, no Gemini, and no vLLM/local model calls. The run evaluates `80`
  verifier rows with recorded GPT spend `$0.2351`. The stricter prompt fixes
  the no-visible-answer problem: incomplete rate is `0.0000` for GPQA and
  MMLUPro on both validation and test. Verifier quality is GPQA validation
  `0.70`, GPQA test `0.50`, MMLUPro validation `0.50`, and MMLUPro test
  `0.35`. Validation selects `strict_verifier_support_thr0.85_benchmmlupro`,
  which reaches held-out utility `0.7553`, quality `0.8140`,
  probe-cost-charged utility `0.7395`, oracle-utility ratio `0.8868`, and
  frontier-call rate `0.3663`; the candidate-ranker reference is `0.7529`
  utility and `0.8140` quality. The best held-out diagnostic strict-support
  row reaches utility `0.7607`, quality `0.8140`, and probe-cost utility
  `0.7449`, but it is test-picked. Interpretation: strict answer-only
  prompting fixes verifier output mechanics and improves GPQA relative to the
  earlier GPT verifier, but it does not beat the local-safe gain gate and does
  not close the broad100 target.
- Probe fusion policy: evaluated in
  `experiments/185_probe_fusion_policy.py`; no-new-call complementarity test
  over cached local-safe gain gates and cached strict MCQ verifier support. It
  keeps the original broad100 action matrix as the oracle/action set, so the
  strict verifier answer is used only as a probe, not as a new oracle action.
  Validation selects
  `fusion_strict_repair_base_pred_rf_thr-0.0288_strict0.95_benchmmlupro`.
  Held-out test utility is `0.7627`, quality `0.8140`, oracle-utility ratio
  `0.9013`, frontier-call rate `0.3140`, probe-call rate `0.1163`, and
  probe-cost utility `0.7469`. The same local threshold without strict probing
  reaches held-out utility `0.7625`, so the fusion's validation-selected raw
  gain is only `0.0002` and disappears after probe cost. The best test-picked
  fusion rows reach utility `0.7687` and quality `0.8198`, but they are not
  validation-selected and still have poor probe-cost utility. Interpretation:
  strict GPT support and local-safe consensus are not complementary enough to
  justify broad fusion. The next branch needs a different correctness signal,
  not another recombination of these cached probes.
- Qwen32 verifier-risk veto: evaluated in
  `experiments/186_qwen32_verifier_risk_veto.py`; no-new-call reuse of cached
  Qwen3-32B-AWQ vLLM answer-verifier outputs from Experiment 142. The cached
  verifier was originally asked about an older base answer, so this experiment
  treats it only as a query-level/local-risk signal. The signal is meaningful
  for the old answer: held-out `accept` rows have old-base quality `0.8351`
  and `escalate` rows have old-base quality `0.5405`. But it fails as a veto
  for the current policy. Validation selects
  `qwen32_veto_escalate_pred_ridge_thr-0.2129`; held-out utility is `0.7466`,
  quality `0.7965`, oracle-utility ratio `0.8823`, frontier-call rate
  `0.3023`, and verifier-call rate `0.9942`. This is below both the
  candidate-ranker reference (`0.7529` utility) and the best no-probe local-safe
  reference in the same sweep (`0.7653` utility, `0.8140` quality, `0.2965`
  frontier-call rate). Interpretation: stale answer verification is not
  transferrable enough. The next branch should judge the current candidate
  actions directly, ideally with a fresh local vLLM cache, rather than reusing
  old-base verifier verdicts as generic risk labels.
- Current-action verifier with local Qwen3-14B-AWQ: evaluated in
  `experiments/187_current_action_verifier_vllm.py`; fresh vLLM cache for the
  current local-safe action set. The run in
  `results/controlled/broad100_current_action_verifier_qwen14b/` collected
  `344/344` validation/test judgments from `Qwen/Qwen3-14B-AWQ` through local
  vLLM, with no GPT, Gemini, or Claude calls. It also writes the compact
  oracle target table requested by the SLM/LLM probe plan. Validation selects
  `current_verifier_switch_conf0.85_pred_rf_thr-0.0288`; on held-out test this
  reaches utility `0.7678`, quality `0.8198`, oracle-utility ratio `0.9073`,
  frontier-call rate `0.3256`, strong-or-frontier rate `0.6279`, verifier-call
  rate `1.0000`, switch rate `0.0058`, and override rate `0.1337`. This is a
  small raw improvement over the local-safe reference (`0.7625` utility,
  `0.8140` quality) and candidate-ranker reference (`0.7529` utility), but it
  remains below the broad100 target (`0.8463` oracle utility, `0.8721` oracle
  quality). The selected utility CI is `[0.7052, 0.8230]`. Interpretation:
  current-action verification is less stale and slightly useful, but it is not
  the missing signal. Because the verifier prompt sees cached candidate answer
  text, including frontier candidates, treat it as post-candidate verification
  rather than cheap pre-routing.
- Benchmark-composed current-verifier thresholds: evaluated in
  `experiments/188_current_verifier_benchmark_policy.py`; no-new-call
  validation composition over cached Experiment 187 choices. The run in
  `results/controlled/broad100_current_verifier_benchmark_policy/` selects one
  cached current-verifier policy per benchmark using validation. Validation
  selects `benchmark_best_eps0_fallback_global`; held-out test utility is only
  `0.7495`, quality `0.8023`, oracle-utility ratio `0.8856`, and
  frontier-call rate `0.3488`. This is worse than the global Experiment 187
  verifier (`0.7678` utility) and the local-safe reference (`0.7625` utility).
  Interpretation: per-benchmark recomposition over the same current-verifier
  outputs overfits. Do not spend more time on benchmark policy maps unless the
  underlying checker signal changes.
- Targeted residual action repair: evaluated in
  `experiments/189_targeted_residual_repair_policy.py`; no-new-call residual
  sweep over cached Experiment 187 choices. Validation residuals identify GPQA,
  BBH, GSM8K, and MMLUPro as the largest remaining gaps. The pure
  validation-best rule improves validation but has no held-out effect
  (`0.7678` test utility). A validation residual-coverage selector chooses
  `scopegpqa+bbh+gsm8k+mmlupro_selected_qwen32_qwen3-14b-awq-local_none`,
  which reaches held-out utility `0.7736`, quality `0.8256`,
  oracle-utility ratio `0.9141`, and frontier-call rate `0.3256`. This is a
  small improvement over Experiment 187 but still far below target. The best
  held-out diagnostic row reaches `0.7794` utility and `0.8314` quality, but
  it is test-picked. Interpretation: deterministic residual repairs can
  harvest a little action-identity signal, mainly by replacing some Qwen32
  choices with Qwen14 on high-residual slices, but the remaining gap still
  needs stronger current-candidate correctness evidence.
- Variable-option GPT MCQ verifier: evaluated in
  `experiments/190_variable_option_mcq_verifier_policy.py`; provider rerun of
  the strict MCQ verifier with an `A-J` option parser. This fixes the
  Experiment 184 mechanical issue where MMLU-Pro was constrained to `A-D`.
  The run in
  `results/controlled/broad100_variable_option_mcq_verifier_policy/` evaluates
  `80` GPQA/MMLU-Pro validation/test rows with GPT-5.5, costs `$0.2516`, and
  uses no Claude or Gemini. MMLU-Pro verifier quality improves to `0.9000` on
  validation and `0.9000` on test, but GPQA remains unstable (`0.8000`
  validation, `0.4500` test). Validation-selected routing is negative:
  `strict_verifier_support_thr0.5_benchgpqa-mmlupro` reaches held-out utility
  `0.7476`, probe-cost utility `0.7121`, and quality `0.7907`. The best
  held-out diagnostic support row reaches `0.7730` utility, below Experiment
  189's selected `0.7736`. Interpretation: fixing option support makes the
  verifier technically valid and useful for MMLU-Pro diagnostics, but GPT
  verifier support over GPQA/MMLU-Pro does not solve routing under validation
  selection and cost accounting.
- Variable-verifier residual fusion: evaluated in
  `experiments/191_variable_verifier_residual_fusion.py`; no-new-call fusion
  of Experiment 190 variable-option verifier support with Experiment 189
  residual repair. The raw validation-best fusion overfits because it includes
  GPQA (`0.8184` validation utility, `0.7570` held-out utility). A reliability
  selector that only activates benchmarks where validation verifier quality is
  at least `0.85` selects MMLU-Pro support and reaches held-out raw utility
  `0.7765`, but probe-cost utility is only `0.7592`, below Experiment 189.
  The best held-out diagnostic fusion row reaches `0.7833` raw utility and
  `0.8314` quality but only `0.7660` after probe cost. Interpretation:
  MMLU-Pro GPT support is a real signal, but it is too expensive as a probe
  unless reused as the final answer or replaced by a cheaper checker.
- Gemini variable-option residual fusion: evaluated in
  `experiments/192_gemini_variable_option_residual_fusion.py`; attempted the
  same MMLU-Pro variable-option support idea with Gemini 3.5 Flash as the
  cheaper verifier. The run in
  `results/controlled/broad100_gemini_variable_option_residual_fusion/`
  attempted `40` MMLU-Pro validation/test rows with estimated uncached spend
  `$0.0565`, but all rows returned HTTP 429. The verifier table has `0/40`
  success rows, `nan` validation/test verifier quality, and `$0.0000`
  recorded Gemini spend. The selected policy falls back to Experiment 189 with
  held-out utility `0.7736`, quality `0.8256`, oracle-utility ratio `0.9141`,
  frontier-call rate `0.3256`, and no probe/override use. Interpretation:
  Gemini Flash is not evaluated as a signal here because quota/rate limiting
  blocked the calls. Do not cite this as a model-quality failure; cite it as a
  provider-availability blocker and prefer local vLLM or confirmed-quota cheap
  verifiers next.
- Local vLLM solve-support residual fusion: evaluated in
  `experiments/193_local_vllm_solve_support_residual_fusion.py`; tested
  Qwen3-14B-AWQ and Qwen3-32B-AWQ as local solve-and-support verifiers over
  the high-residual val/test slices. The verifier solves independently and
  supports a candidate action only if the candidate answer matches its trusted
  answer, then support is fused with Experiment 189 residual repair. Qwen14
  run:
  `results/controlled/broad100_local_vllm_solve_support_residual_fusion/`;
  `264/264` successful rows; mean local latency `0.8930s`; validation-best
  `scopebbh_thr0_always` overfits from validation utility `0.8078` to held-out
  utility `0.7378`, quality `0.7907`, oracle ratio `0.8718`, frontier rate
  `0.3140`, probe rate `0.1163`. Reliability-gated selection falls back to
  Experiment 189. Qwen32 run:
  `results/controlled/broad100_local_vllm_solve_support_residual_fusion_qwen32/`;
  `263/264` successful rows; mean local latency `1.4168s`; validation and
  reliability-gated selection both fall back to Experiment 189 with held-out
  utility `0.7736`, quality `0.8256`, oracle ratio `0.9141`, frontier rate
  `0.3256`. Interpretation: local solve-support verification is operational
  and cheap in remote-dollar terms, but still not accurate enough on GPQA and
  MMLUPro. It should not be repeated as a plain answer-support probe without a
  new source of evidence, such as execution/checker feedback or calibrated
  abstention.
- Conservative support-abstention policy: evaluated in
  `experiments/194_conservative_support_abstention_policy.py`; this is a
  no-new-call threshold-only pilot over cached Experiment 189 choices, cached
  Qwen14/Qwen32 solve-support verifier outputs, and the cached
  self-consistency action matrix. It writes the local-vs-large oracle target
  table and cached probe-signal table under
  `results/controlled/broad100_conservative_support_abstention_policy/`.
  Validation selects `qwen14_bbh_support2_conf0_nonfrontier`: on BBH only,
  switch only when Qwen14 supports a non-frontier candidate whose answer is
  shared by at least two cached actions. Held-out utility improves from
  `0.7736` to `0.7799`, quality from `0.8256` to `0.8314`, oracle-utility ratio
  from `0.9141` to `0.9216`, and frontier-call rate drops from `0.3256` to
  `0.3140`, with `0.1163` probe-call rate and only `2/172` held-out switches.
  This is a real but small no-training positive, still below the held-out
  cost-aware oracle utility `0.8463`.
- Local consensus cost-suppression audit: evaluated in
  `experiments/195_local_consensus_cost_suppression_audit.py`; this reuses
  cached local answers and Experiment 194 choices and makes no GPT, Gemini,
  Claude, local generation, or vLLM serving calls. The validation-selected
  deployable local-majority rule is
  `local_majority_scopegsm8k_votes2_if_base_frontier_cheapest`. On held-out
  test it moves utility only from `0.7799` to `0.7806`, keeps quality at
  `0.8314`, improves oracle-utility ratio from `0.9216` to `0.9225`, and lowers
  frontier-call rate from `0.3140` to `0.2849`. A post-hoc diagnostic
  same-answer rule over GSM8K+MATH500+LiveMathBench reaches held-out utility
  `0.7898`, oracle-utility ratio `0.9333`, and frontier-call rate `0.2267`, but
  it is not deployable because it uses the selected frontier answer as an
  anchor. Interpretation: local majority is not the missing signal; the useful
  next target is a cheap pre-call equivalence checker that predicts whether a
  local answer would match the remote/frontier answer.
- Local exact-answer verifier cost-suppression: evaluated in
  `experiments/196_local_exact_answer_verifier_cost_suppression.py`; this tests
  the deployable pre-call version of the Experiment 195 same-answer diagnostic
  using local vLLM only. Qwen3-14B-AWQ run:
  `results/controlled/broad100_local_exact_answer_verifier_cost_suppression/`;
  `308/308` successful rows; validation-selected
  `exact_verifier_scopemath500_majority3_cheapest_thr0` makes no held-out
  switches and stays at held-out utility `0.7799`, quality `0.8314`,
  oracle-utility ratio `0.9216`, and frontier-call rate `0.3140`. The best
  held-out diagnostic Qwen14 row reaches only `0.7812` utility. Qwen3-32B-AWQ
  run:
  `results/controlled/broad100_local_exact_answer_verifier_cost_suppression_qwen32/`;
  `308/308` successful rows; validation-selected
  `exact_verifier_scopemath500_qwen32_thr0` overfits from validation utility
  `0.8095` to held-out utility `0.7561`, quality `0.8023`, oracle-utility ratio
  `0.8935`, and frontier-call rate `0.2674`. The best held-out diagnostic
  Qwen32 majority row reaches only `0.7810` utility. Interpretation: plain
  local LLM answer verification is not the missing equivalence signal. Future
  pre-call equivalence checks should use external evidence, symbolic/tool
  checks, execution, or stronger calibrated abstention.
- Target-gate concrete bridge sweep: evaluated in
  `experiments/197_target_gate_concrete_bridge_sweep.py`; this is a no-call
  bridge test for the strongest existing tool-aware local-vs-large gate
  (`tool_aware_benchmark_composed_eps0.01_recall_then_quality`). The script
  keeps that gate fixed and sweeps simple concrete local-action and
  large-action mappings. Output:
  `results/controlled/broad100_target_gate_concrete_bridge_sweep/`. It makes
  no GPT, Gemini, Claude, local generation, or vLLM calls. Validation selects
  `target_gate_local_prior_utility_large_large_consensus_cheapest`; on held-out
  test it reaches utility `0.6464`, quality `0.6802`, oracle-utility ratio
  `0.7639`, frontier-call rate `0.2151`, and large-gate rate `0.6395`. This
  fails the target thresholds (`0.8039` utility and `0.8421` quality), and the
  best held-out diagnostic rows are the same family. Interpretation: the
  local-vs-large abstraction can pass numerically, but simple concrete action
  mappings destroy the gain. The remaining bottleneck is concrete action
  identity after deciding that a larger action is useful.
- Residual action-identity audit: evaluated in
  `experiments/198_residual_action_identity_audit.py`; this no-call audit
  uses the current validation-selected concrete policy
  (`local_majority_scopegsm8k_votes2_if_base_frontier_cheapest`) and the cached
  broad100 action matrix to explain the remaining residual. Output:
  `results/controlled/broad100_residual_action_identity_audit/`. Held-out
  current utility is `0.7806`, quality is `0.8314`, oracle-utility ratio is
  `0.9225`, and frontier-call rate is `0.2849`; the held-out query/action
  oracle is still utility `0.8463`, quality `0.8721`, and frontier-call rate
  `0.1395`. Residual mass is concentrated on GPQA (`5.0348`), MMLUPro
  (`2.4272`), MATH500 (`1.6404`), BBH (`1.0915`), and LiveMathBench
  (`1.0089`). A perfect local-answer/oracle-answer equivalence diagnostic
  would reach `0.8225` utility, and a selected-frontier-answer local
  equivalence diagnostic would reach `0.7982`. Interpretation: equivalence
  checking is useful but not sufficient; the next probe needs concrete
  evidence for action identity, especially GPQA/MMLUPro answer adjudication
  and exact-math checker/calculator evidence.

Next:

- Follow the "Next Main-Thread Probe Search: Methods 2, 3, 4, 6" plan above.
- Pairwise preference routing, richer-state kNN, and simple math-verifiability
  suppression patches are now tested, and another calibration layer over the
  same self-consistency table plus benchmark-level policy composition and
  binary strong/self classifiers, local action comparison, calibrated
  action-compare features, expanded observable-feature learners, direct
  frontier-need predictors, and Qwen14 binary frontier-need judgment were
  negative. Qwen3-14B-AWQ as a stronger local action selector, residual-risk
  judge, and frontier-need precision-filter input was also negative. Local E5
  embeddings over query/evidence text improve diagnostics only slightly, and
  adding E5 to the self-consistency action gate gives only a tiny selected
  improvement. Cached pairwise action ranking, residual confidence rules,
  constrained local confidence, and benchmark-composed fixed policies give
  selected improvements. Tool-aware benchmark composition closes the
  local-vs-large diagnostic target, but the deployed-action bridge and
  benchmark-composed deployed-action repair both fail. Train-calibrated local
  answer support also fails. Public code-test verification helps but does not
  solve broad100 because the remaining gap is outside code. Candidate
  correctness ranking is a partial positive for action identity, but all-rank
  candidate probing is cost-infeasible and the gated rows remain below target.
  Answer-group reliability, cached generic adjudicator overrides,
  benchmark-level policy recombination, and the first GPT verifier-action
  probe plus its cached supported-action variant confirm that shallow
  agreement/adjudication/composition is too noisy or too expensive as a
  standalone verifier. Local-safe consensus gating is a partial positive for
  over-escalation suppression, but its own diagnostic oracle is still below the
  target. Strict answer-only GPT MCQ verification fixes the visible-output
  mechanics but still does not produce enough routing value, and no-call fusion
  with local-safe consensus overfits validation without improving held-out
  probe-cost utility. A cached Qwen32 verifier signal separates correctness of
  the old base answer, but it does not transfer as a veto for the current
  action policy. Gemini Flash variable-option verification is currently blocked
  by HTTP 429 and should not be treated as a measured probe. Local
  solve-support verification with Qwen14/Qwen32 is operational; conservative
  BBH-only support-abstention gives a small selected held-out improvement, but
  it remains far below the target. Local-majority cost suppression gives only a
  tiny deployable gain, while the post-hoc same-answer audit shows that
  pre-call local-vs-frontier equivalence prediction would be valuable on exact
  math slices. The first local exact-answer verifier tested this deployable
  direction with Qwen14/Qwen32 and was negative under validation selection. The
  target-gate concrete bridge sweep also shows that passing the local-vs-large
  diagnostic abstraction is not enough: simple fixed or consensus concrete
  action mappings collapse held-out utility. The residual action-identity
  audit localizes the remaining held-out residual to GPQA, MMLUPro, MATH500,
  BBH, and LiveMathBench, and shows that even a perfect local-equivalence
  diagnostic would remain below the broad100 target. A corrected validation
  policy-library portfolio with oracle/target/post-hoc rows excluded also
  fails: validation-only composition overfits, train-prior-stabilized
  composition remains below the current selected policy, and the cleaned
  test-picked diagnostic ceiling is close but still below the utility target.
  A current-policy variable-verifier fusion over cached GPT-5.5 MCQ support
  confirms the same bottleneck at the action-evidence level: validation-wide
  GPQA+MMLUPro support overfits, while reliability-constrained MMLUPro support
  gives only a small held-out raw gain (`0.7806` -> `0.7836` utility) and falls
  to `0.7663` after route-time GPT probe cost. The best held-out MMLUPro-only
  diagnostic reaches `0.7904` utility and `0.8372` quality, still below the
  target. This is a partial positive for MMLUPro action support and a negative
  result for broad GPT verifier-as-router use.
  The next implementation should focus on evidence beyond plain answer support
  or policy recombination for the current GPQA, MMLUPro, AIME, and exact-math
  candidate actions:
  task-specific answer adjudication with abstention, verifier-style probes
  backed by concrete evidence, tool/execution equivalence checking, and
  calculator/checker signals. Shallow
  prefill/activation anomaly thresholds are now tested and should not be
  repeated unless the probe exposes richer internal evidence than global
  distance or train-neighbor summaries.
- Use GPQA, MMLUPro, LiveMathBench, and MATH500 as stress tests for failure
  analysis, not as the scope of the method.
- Avoid broad paid adjudication because GPT local-only routing was not
  cost-effective.

## Success Criteria For These Experiments

The broad100 method should meet:

- within 3 absolute quality points of the cost-aware oracle;
- at least 95% cost-aware oracle utility;
- frontier-call rate <= 25%--40%;
- remote API cost <= 0.15x--0.35x all-frontier where applicable;
- no Claude calls;
- GPT model fixed to `gpt-5.5`;
- Gemini model fixed to `gemini-3.5-flash`;
- per-provider spend below $15.

## Current Caveat

The existing broad100 oracle improves when stronger candidates are added, so raw quality gains are not enough. Every new probe/action must be judged against the new cost-aware oracle for that action set.

## Benchmark-Agnostic Probe-State Result

Evidence:

- `experiments/201_benchmark_agnostic_probe_state_routecode.py`
- `results/controlled/BENCHMARK_AGNOSTIC_PROBE_STATE_DIRECTION_MEMO.md`
- `results/controlled/broad100_probe_state_routecode/PROBE_STATE_ROUTECODE_MEMO.md`
- `results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv`
- `results/controlled/broad100_probe_state_routecode/table_probe_state_policy_selected.csv`
- `results/controlled/broad100_probe_state_routecode/table_probe_state_benchmark_heldout.csv`
- `results/controlled/broad100_probe_state_routecode/probe_state_code_cards.md`

This is the first cached Broad100 implementation of the new direction:

```text
query + cheap local model behavior -> probe_state -> cost-aware action
```

It uses broad observable features only: local answer agreement, local-vs-medium
disagreement, self-consistency entropy/margins, validity/malformed proxies,
output length/latency, and cached logprob margins where present. The main
probe-state rows exclude benchmark ID; benchmark-ID rows are diagnostic.

Standard held-out split:

- benchmark lookup utility `0.6652`, quality `0.7791`, oracle-utility ratio
  `0.7860`;
- text-only utility router utility `0.6189`, quality `0.7093`,
  oracle-utility ratio `0.7314`;
- probe-state KMeans `K=16` utility `0.6876`, quality `0.7674`,
  oracle-utility ratio `0.8125`, frontier-call rate `0.5174`;
- direct probe utility regressor utility `0.6733`, quality `0.7500`,
  oracle-utility ratio `0.7956`;
- oracle fixed local-vs-large gate upper bound utility `0.7207`;
- oracle RouteCode-label upper bound utility `0.7887`.

Benchmark-heldout transfer mean test utilities:

- benchmark lookup/global best `0.4363`;
- text-only utility router `0.4823`;
- text-to-RouteCode label `0.4963`;
- text RouteCode plus probe state `0.5535`;
- probe-to-RouteCode label `0.6183`;
- probe-state KMeans `0.6217`;
- direct probe utility regressor `0.6507`;
- oracle fixed local-vs-large gate upper bound `0.6789`;
- oracle RouteCode-label upper bound `0.7792`.

Interpretation:

- Probe-state features transfer better than benchmark lookup and text-only
  routing, which supports ProbeCode as the right Phase 3 direction.
- The result is not close enough to the oracle. The cheap observable layer is
  still the bottleneck.
- Combining a predicted text RouteCode label with probe state is negative in
  this first version, so the next step should improve label observability or
  make the probe-state learner decision-aware.
- Benchmark-specific checkers should remain diagnostics/plugins, not the main
  method. The main method should stay benchmark-agnostic:
  `query + cheap local model behavior -> probe_state -> cost-aware action`.

## Benchmark-Agnostic Family Probe Gate

Evidence:

- `experiments/202_benchmark_agnostic_family_probe_gate.py`
- `results/controlled/broad100_benchmark_agnostic_family_probe_gate/BENCHMARK_AGNOSTIC_FAMILY_PROBE_GATE_MEMO.md`
- `results/controlled/broad100_benchmark_agnostic_family_probe_gate/table_family_probe_gate_selected.csv`
- `results/controlled/broad100_benchmark_agnostic_family_probe_gate/table_family_probe_gate_all.csv`
- `results/controlled/broad100_benchmark_agnostic_family_probe_gate/table_family_probe_gate_query_choices.csv`
- `results/controlled/broad100_benchmark_agnostic_family_probe_gate/fig_family_probe_gate_utility.pdf`

This no-new-call cached experiment tests whether benchmark-agnostic probe
features can observe a coarser local-vs-large action family before selecting
the exact concrete model. It excludes benchmark ID and does not use provider
calls, vLLM calls, or task-specific checkers.

Held-out test results:

- full cost-aware oracle: quality `0.8721`, utility `0.8463`, frontier-call
  rate `0.1395`;
- current concrete base policy: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`;
- validation-selected ridge family gate: quality `0.8488`, utility `0.8140`,
  oracle-utility ratio `0.9618`, frontier-call rate `0.2267`;
- validation-selected probe-state family gate: quality `0.8488`, utility
  `0.8151`, oracle-utility ratio `0.9632`, frontier-call rate `0.2384`;
- validation-selected concrete bridge: quality `0.8023`, utility `0.6902`,
  oracle-utility ratio `0.8156`, frontier-call rate `0.6105`.

Interpretation:

- This is the first benchmark-agnostic result that passes the Broad100 numeric
  utility and quality targets against the full cost-aware oracle, but only in a
  family-oracle abstraction.
- The result means cheap local behavior can observe the coarse local-vs-large
  state. It does not yet mean the deployed router is solved, because the
  family-gate rows still use the cached best concrete action inside the
  predicted family.
- The concrete bridge failure isolates the next bottleneck: exact action
  identity inside local and large families, not the broad local-vs-large
  decision.

## Current-Base Cached Adjudicator Bridge Result

Evidence:

- `experiments/203_current_base_cached_adjudicator_bridge.py`
- `results/controlled/broad100_current_base_cached_adjudicator_bridge/CACHED_ADJUDICATOR_BRIDGE_MEMO.md`
- `results/controlled/broad100_current_base_cached_adjudicator_bridge/table_cached_adjudicator_bridge_selected.csv`
- `results/controlled/broad100_current_base_cached_adjudicator_bridge/table_cached_adjudicator_bridge_all.csv`
- `results/controlled/broad100_current_base_cached_adjudicator_bridge/table_cached_adjudicator_bridge_query_choices.csv`

This no-new-call replay tests whether old cached broad GPT/Gemini adjudicator
decisions can bridge the current best concrete policy to the oracle. It is a
diagnostic, not the main benchmark-agnostic local-probe method, because some
cached adjudicator prompts included benchmark metadata and the adjudicator is
an expensive route-time probe.

Results:

- Current base test: quality `0.8314`, utility `0.7806`, oracle ratio
  `0.9225`, frontier-call rate `0.2849`.
- Validation-best raw adjudicator override:
  `gpt_with_frontier_thr0.75_if_adjudicator_local`. Validation utility is
  `0.8320`, but held-out test utility drops to `0.7587`; after charging
  adjudicator probe cost, held-out utility is only `0.5383`.
- Probe-cost-aware validation selection chooses the no-adjudicator base.
- Best raw held-out diagnostic adjudicator row reaches only `0.7834` utility
  and collapses to `0.4201` after probe cost.

Interpretation:

- There is a tiny raw residual signal in cached broad adjudication, but it is
  too costly and unstable to justify fresh paid route-time judging as the next
  main method.
- The next bridge should stay with cheap local/probe-state evidence for
  concrete action identity rather than paying GPT/Gemini to adjudicate broad
  candidate answers.

## Benchmark-Agnostic Local-Candidate Selector Result

Evidence:

- `experiments/204_benchmark_agnostic_local_candidate_selector.py`
- `results/controlled/broad100_benchmark_agnostic_local_candidate_selector/LOCAL_CANDIDATE_SELECTOR_MEMO.md`
- `results/controlled/broad100_benchmark_agnostic_local_candidate_selector/table_local_candidate_selector_selected.csv`
- `results/controlled/broad100_benchmark_agnostic_local_candidate_selector/table_local_candidate_selector_benchmark_heldout.csv`
- `results/controlled/broad100_benchmark_agnostic_local_candidate_selector/table_local_candidate_selector_query_choices.csv`

This no-new-call cached experiment directly attacks the concrete local action
identity bottleneck:

```text
query + cheap local behavior -> best local candidate -> optional base override
```

It uses broad local/probe features only, including answer support/agreement,
validity, output length/latency, local model identity, train-only global model
priors, and the cached probe-state features. It excludes benchmark ID and
benchmark-specific checkers.

Held-out standard results:

- full cost-aware oracle: quality `0.8721`, utility `0.8463`,
  frontier-call rate `0.1395`;
- current base: quality `0.8314`, utility `0.7806`, oracle-utility ratio
  `0.9225`, frontier-call rate `0.2849`;
- diagnostic current base plus all cached local candidates oracle: quality
  `0.8605`, utility `0.8321`, oracle-utility ratio `0.9833`,
  frontier-call rate `0.1279`;
- validation-selected local-selector override
  `candidate_extra_trees_leaf4_override_if_base_frontier_score_thr0.798397`:
  validation utility `0.8093`, but held-out utility `0.7711`, quality
  `0.8198`, oracle-utility ratio `0.9112`, frontier-call rate `0.2616`;
- best validation-selected always-local learned ranker: utility `0.6628`;
- probe-state local KMeans: utility `0.6453`;
- text-only local utility router: utility `0.5465`.

Benchmark-heldout transfer mean test utilities:

- current base plus all cached local candidates oracle `0.8125`;
- current base `0.7632`;
- current base plus learned local selector `0.7536`;
- local action oracle `0.7148`;
- local candidate ranker `0.5759`;
- probe-only local utility router `0.5648`;
- text-only local router `0.4852`;
- probe-state local KMeans `0.4648`;
- benchmark lookup/global-best local `0.4611`.

Interpretation:

- The local candidate action set has enough headroom: post-hoc base+local
  oracle gets within `0.0141` utility of the full oracle on held-out test.
- The learned benchmark-agnostic local selector cannot yet observe which local
  candidate is correct. Validation-selected overrides overfit and reduce
  held-out utility below the current base.
- This narrows the Phase 3 problem: keep the broad ProbeCode framing, but the
  next mechanism needs a stronger cheap evidence signal for local answer
  reliability/action identity, not benchmark-specific checkers or paid broad
  adjudication.

## Current-Base Local Support Verifier Result

Evidence:

- `experiments/205_current_base_local_support_verifier.py`
- `results/controlled/broad100_current_base_local_support_verifier/LOCAL_SUPPORT_VERIFIER_MEMO.md`
- `results/controlled/broad100_current_base_local_support_verifier/table_local_support_verifier_probe.csv`
- `results/controlled/broad100_current_base_local_support_verifier/table_local_support_verifier_policy_selected.csv`
- `results/controlled/broad100_current_base_local_support_verifier/table_local_support_verifier_query_choices.csv`

This experiment tests a generic local verifier state:

```text
query + cheap local candidate answers -> local support state -> cost-aware action
```

The verifier is Qwen3-14B-AWQ served through local vLLM. The prompt omits
benchmark ID and uses no math-only, GPQA-only, MMLUPro-only, or code-only
checker. It makes no GPT, Gemini, or Claude calls.

Verifier diagnostics:

- `344` val/test rows collected;
- validation: `93` local_supported, `79` no_reliable_local;
- test: `83` local_supported, `89` no_reliable_local;
- mean verifier latency was roughly `0.50s` to `0.66s` per row.

Selected held-out results:

- current base: quality `0.8314`, utility `0.7806`, oracle-utility ratio
  `0.9225`, frontier-call rate `0.2849`;
- validation-selected local support downshift
  `downshift_frontier_supported_conf0`: validation utility `0.8085`, but test
  quality `0.8023`, utility `0.7571`, oracle-utility ratio `0.8947`,
  frontier-call rate `0.2558`;
- best diagnostic test row `downshift_frontier_supported_conf0.95`: quality
  `0.8256`, utility `0.7754`, oracle-utility ratio `0.9163`,
  frontier-call rate `0.2791`.

Interpretation:

- The generic local verifier can reduce frontier calls slightly, but its
  support state is not reliable enough; it downshifts some frontier calls that
  were actually needed.
- The current base remains stronger than the verifier override.
- This is a useful negative result for the benchmark-agnostic ProbeCode story:
  cheap local candidate-answer support alone does not solve concrete action
  identity. The next mechanism needs better calibration of local reliability,
  richer uncertainty evidence, or a decision-aware state learner rather than a
  plain local support verifier.

## Probe-State Composed YES/NO Policy Result

Evidence:

- `experiments/206_probe_state_composed_yesno_policy.py`
- `results/controlled/broad100_probe_state_composed_yesno_policy/PROBE_STATE_COMPOSED_POLICY_MEMO.md`
- `results/controlled/broad100_probe_state_composed_yesno_policy/table_probe_state_composed_policy_selected.csv`
- `results/controlled/broad100_probe_state_composed_yesno_policy/table_probe_state_composed_policy_all.csv`
- `results/controlled/broad100_probe_state_composed_yesno_policy/probe_state_composed_code_cards.md`

This cached experiment tests a broad probe-state policy for the coarse
local-vs-large decision:

```text
query + cheap broad local signals -> probe_state -> local-vs-large policy
```

Main rows do not use benchmark ID, benchmark train priors, deterministic tool
suppression, task-specific verifiers, GPT/Gemini/Claude calls, local
generation, or vLLM. The experiment uses cached Broad100 target/action tables
and cached local behavior. Important caveat: it evaluates the local-vs-large
abstraction using cached best-local and best-large actions, not full concrete
multi-action routing.

Held-out local-vs-large reference:

- oracle local-vs-large gate: quality `0.8721`, utility `0.8463`,
  oracle-utility ratio `1.0000`, frontier-call rate `0.1395`;
- target gate used in the memo: utility at least `0.8039`, quality at least
  `0.8421`.

Validation-selected held-out rows:

- main benchmark-agnostic/no-tool probe state,
  `main_no_benchmark_no_tool_k2`: quality `0.8198`, utility `0.7747`,
  oracle-utility ratio `0.9154`, frontier-call rate `0.2907`, large-call rate
  `0.9186`;
- diagnostic train benchmark-prior row,
  `main_plus_train_benchmark_prior_k16`: quality `0.8140`, utility `0.7763`,
  oracle-utility ratio `0.9174`, frontier-call rate `0.2267`, large-call rate
  `0.5581`;
- diagnostic tool-aware row,
  `main_plus_tool_available_k8`: quality `0.8605`, utility `0.8248`,
  oracle-utility ratio `0.9747`, frontier-call rate `0.2500`, large-call rate
  `0.8023`.

Interpretation:

- Broad, cached probe states without benchmark or tool information still miss
  the Phase 3 target.
- Adding a train-only benchmark prior does not solve the gap and should remain
  diagnostic.
- Adding deterministic tool availability clears the local-vs-large abstraction
  target, so tool/verifiability evidence is a strong positive control.
- This does not license benchmark-specific checkers as the main method. The
  next ProbeCode step should learn or approximate a benchmark-agnostic
  verifiability/answer-validity state that captures the useful part of the tool
  diagnostic without hard-coding math-only, GPQA-only, MMLUPro-only, or
  code-only rules.

## Learned Verifiability Probe-State Result

Evidence:

- `experiments/207_learned_verifiability_probe_state.py`
- `results/controlled/broad100_learned_verifiability_probe_state/LEARNED_VERIFIABILITY_PROBE_STATE_MEMO.md`
- `results/controlled/broad100_learned_verifiability_probe_state/table_learned_verifiability_policy_selected.csv`
- `results/controlled/broad100_learned_verifiability_probe_state/table_learned_verifiability_classifier_summary.csv`
- `experiments/208_learned_verifiability_benchmark_heldout.py`
- `results/controlled/broad100_learned_verifiability_benchmark_heldout/LEARNED_VERIFIABILITY_BENCHMARK_HELDOUT_MEMO.md`
- `results/controlled/broad100_learned_verifiability_benchmark_heldout/table_learned_verifiability_benchmark_heldout_summary.csv`

This pair of cached experiments tests the next ProbeCode hypothesis:

```text
query + cheap local behavior -> learned verifiability state -> local-vs-large policy
```

Main learned rows use train-only verifiability labels and do not expose
benchmark ID, domain, metric, benchmark train priors, outcome utility/quality
columns, direct tool flags, or direct tool output features at validation/test
time. Direct tool-flag rows are positive-control diagnostics. No GPT, Gemini,
Claude, local generation, or vLLM calls are made.

Standard held-out split:

- local-vs-large oracle: quality `0.8721`, utility `0.8463`,
  frontier-call rate `0.1395`;
- learned global verifiability selected by validation,
  `extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75`: quality `0.8547`,
  utility `0.8232`, oracle-utility ratio `0.9727`, frontier-call rate
  `0.2209`;
- learned discrete verifiability state selected by validation,
  `gb_depth2_thr0.9844_state_k8`: quality `0.8488`, utility `0.8136`,
  oracle-utility ratio `0.9614`, frontier-call rate `0.2384`;
- best held-out diagnostic learned rows reached quality `0.8605`, utility
  `0.8249`, oracle-utility ratio `0.9748`, frontier-call rate `0.2442`.

Benchmark-heldout transfer over nine held-out benchmarks:

- direct tool-flag positive control, validation-best: quality `0.8426`,
  utility `0.8104`, oracle-utility ratio `0.9725`, frontier-call rate
  `0.2278`;
- learned global verifiability, validation-best: quality `0.8315`, utility
  `0.7981`, oracle-utility ratio `0.9598`, frontier-call rate `0.2333`;
- learned discrete verifiability state, validation-best: quality `0.8185`,
  utility `0.7808`, oracle-utility ratio `0.9488`, frontier-call rate
  `0.2593`;
- reference policy, validation-best: quality `0.7593`, utility `0.7298`,
  oracle-utility ratio `0.8308`, frontier-call rate `0.2370`.

Interpretation:

- This is the first benchmark-agnostic no-direct-tool result that clears the
  local-vs-large abstraction target on the standard held-out split.
- Benchmark-heldout transfer remains meaningfully better than reference
  policies and close to the direct tool-flag positive control for the global
  learned verifiability policy.
- The discrete state version is weaker under benchmark-heldout transfer, so the
  RouteCode-style state/action layer still needs stabilization.
- The result should be framed as a learned verifiability abstraction over
  cached best-local/best-large actions, not a full concrete deployed router.

## Decision-Aware Probe-State RouteCode Result

Evidence:

- `experiments/209_decision_aware_probe_state_routecode.py`
- `results/controlled/broad100_decision_aware_probe_state_routecode/DECISION_AWARE_PROBE_STATE_MEMO.md`
- `results/controlled/broad100_decision_aware_probe_state_routecode/table_decision_aware_probe_state_selected.csv`
- `results/controlled/broad100_decision_aware_probe_state_routecode/table_decision_aware_probe_state_benchmark_heldout_selected.csv`
- `results/controlled/broad100_decision_aware_probe_state_routecode/decision_aware_probe_state_code_cards.md`

This cached experiment tests whether the broad probe-state layer improves when
the states are learned with train-only utility/action supervision instead of
plain KMeans over probe features:

```text
query + cheap local behavior -> decision-aware probe_state -> concrete cost-aware action
```

It uses cached Broad100 outputs and the benchmark-agnostic probe-state feature
table. Main features exclude benchmark ID and task-specific checkers. No GPT,
Gemini, Claude, local generation, or vLLM calls are made.

Standard held-out split:

- previous KMeans probe state from experiment 201: quality `0.7674`, utility
  `0.6876`, oracle-utility ratio `0.8125`, frontier-call rate `0.5174`;
- best decision-aware action-probability state selected by validation,
  `et_actionprob_state_depthnone_leaf8_k32`: quality `0.7733`, utility
  `0.7064`, oracle-utility ratio `0.8348`, frontier-call rate `0.5174`;
- best decision-aware utility-direct row selected by validation,
  `et_utility_direct_est100_depthnone_leaf8`: quality `0.7965`, utility
  `0.7049`, oracle-utility ratio `0.8330`, frontier-call rate `0.5698`;
- oracle RouteCode-label upper bound from experiment 201: quality `0.8140`,
  utility `0.7887`, oracle-utility ratio `0.9320`;
- full query oracle for the same action matrix: quality `0.8721`, utility
  `0.8463`.

Benchmark-heldout transfer mean results:

- decision-aware utility-direct: utility `0.6403`, oracle-utility ratio
  `0.7388`, frontier-call rate `0.5574`;
- decision-aware utility-state: utility `0.6189`, oracle-utility ratio
  `0.7211`, frontier-call rate `0.4704`;
- decision-aware action-probability state: utility `0.5913` under the
  validation frontier-cap selector and `0.5717` under raw validation-utility
  selection;
- experiment-201 probe-state KMeans heldout mean was `0.6217`;
- experiment-201 direct probe utility regressor heldout mean was `0.6507`;
- oracle RouteCode-label upper bound heldout mean was `0.7792`.

Interpretation:

- Decision-aware supervision improves the standard split relative to plain
  KMeans, but not enough to approach oracle RouteCode labels or the full
  cost-aware oracle.
- Benchmark-heldout transfer gets worse for the discrete decision-aware states,
  which means the supervised state learner is fitting benchmark/action
  regularities that do not transfer reliably.
- Frontier-cap selection on validation does not guarantee the held-out test
  frontier-call rate stays under the cap.
- This is a negative result for the idea that a stronger generic state learner
  alone closes Phase 3. The next useful direction is to keep the learned
  verifiability signal from experiments 207/208, but solve concrete action
  identity and transfer with a more stable policy layer.

## Concrete Probe-Verifiability Policy Result

Evidence:

- `experiments/210_concrete_probe_verifiability_policy.py`
- `results/controlled/broad100_concrete_probe_verifiability_policy/CONCRETE_PROBE_VERIFIABILITY_POLICY_MEMO.md`
- `results/controlled/broad100_concrete_probe_verifiability_policy/table_concrete_probe_verifiability_selected.csv`
- `results/controlled/broad100_concrete_probe_verifiability_policy/table_concrete_probe_verifiability_benchmark_heldout_selected.csv`
- `results/controlled/broad100_concrete_probe_verifiability_policy/table_concrete_probe_verifiability_query_choices.csv`

This cached experiment tries to convert the learned-verifiability result into a
concrete action policy:

```text
query + cheap local behavior -> learned verifiability/local-candidate scores -> concrete action
```

It makes no provider calls, no vLLM calls, no local generation calls, and no
benchmark-specific verifier calls. The standard split fits broad verifiability
and local-candidate scoring on train. Benchmark-heldout rows refit both the
verifiability classifier and local ranker without the held-out benchmark before
selecting thresholds on the remaining validation rows.

Standard held-out split:

- current concrete base: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`;
- learned verifiability-to-tool gate selected by validation:
  quality `0.8314`, utility `0.7806`, oracle-utility ratio `0.9225`,
  frontier-call rate `0.2849`, override rate `0.0000`;
- local-ranker override selected by validation:
  quality `0.8198`, utility `0.7711`, oracle-utility ratio `0.9112`,
  frontier-call rate `0.2616`, override rate `0.0233`;
- combined verifiability plus local-ranker override selected by validation:
  quality `0.8198`, utility `0.7711`, oracle-utility ratio `0.9112`,
  frontier-call rate `0.2616`;
- diagnostic `current_base_plus_all_locals_oracle`: quality `0.8605`,
  utility `0.8321`, oracle-utility ratio `0.9833`, frontier-call rate
  `0.1279`;
- full query oracle: quality `0.8721`, utility `0.8463`.

Benchmark-heldout mean selected test utilities:

- current base: utility `0.7632`, oracle-utility ratio `0.9276`,
  frontier-call rate `0.2796`;
- learned verifiability-to-tool: utility `0.7632`, oracle-utility ratio
  `0.9276`, frontier-call rate `0.2796`;
- local-ranker override: utility `0.7591`, oracle-utility ratio `0.9234`,
  frontier-call rate `0.2685`;
- combined verifiability plus local-ranker override: utility `0.7591`,
  oracle-utility ratio `0.9234`, frontier-call rate `0.2685`;
- diagnostic `current_base_plus_all_locals_oracle`: utility `0.8125`,
  oracle-utility ratio `0.9852`, frontier-call rate `0.1259`.

Interpretation:

- The learned broad verifiability signal does not by itself repair concrete
  action identity; validation either selects no effective change or selects
  local-ranker overrides that hurt held-out test utility.
- The diagnostic base-plus-all-locals oracle is still strong enough to meet the
  Phase 3 utility target, so the gap is not candidate availability. It is
  transfer-stable observability of which concrete local candidate is safe.
- This strengthens the case that the next method should learn a reliability
  state for candidate answers, not just a query-level verifiability or
  candidate-utility score.
