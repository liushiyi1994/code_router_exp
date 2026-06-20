# SLM/LLM Early-Signal Probe Plan

Status: planning note for the next RouteCode/ProbeRoute probe search. This is not a result claim.

Purpose: define low-training and non-training probe strategies for deciding when a query can be handled by a local/small model and when a stronger local model or frontier LLM is worth its cost.

## Core Question

The current broad100 evidence says the oracle action set is useful, but practical predictors still miss when strong/frontier actions are cost-effective. The next question is more specific:

```text
Given query q, what early signal shows that a small/local model will differ from a larger model in a utility-improving way?
```

This should not be framed as saving router tokens. The value is to expose the information structure behind model selection: which query properties, internal states, or early generations make the SLM/LLM utility gap predictable.

## RouteCode Setting

For each query `q`, define a cost-aware utility:

```text
U(q, a) = quality(q, a) - lambda * normalized_cost(q, a)
```

Here `a` is a route action, not only a model id. Examples:

- local one-shot answer;
- local self-consistency;
- deterministic tool;
- stronger local model;
- frontier/API model from GPT-family, Claude-family, or Gemini-family when cached and explicitly enabled;
- verifier or test-time compute action.

Define two action pools:

```text
A_local = cheap local actions
A_large = stronger local or frontier actions
```

The target gap is:

```text
delta_large(q) =
  max_{a in A_large} U(q, a) - max_{a in A_local} U(q, a)
```

A strong action is useful only if `delta_large(q) > 0` after cost. A conservative deployment rule can use a larger margin:

```text
need_large(q) = 1[delta_large(q) >= tau_gain]
```

The probe should produce cheap evidence `s(q)` before final routing. We evaluate whether thresholding or hand-combining these signals recovers utility under a fixed strong/frontier call budget.

## Design Constraint

Prefer methods that do not train a router.

Allowed first:

- direct confidence statistics;
- internal-confidence scores;
- early-output instability;
- SLM-vs-medium disagreement;
- semantic uncertainty from sampled local outputs;
- train/validation threshold selection only.

Allowed second:

- light calibration such as isotonic regression if the raw signal works;
- frozen-encoder linear probes over pre-generation activations using a small calibration set.

Avoid first:

- fine-tuning;
- LoRA;
- LLM-as-router API calls;
- full neural routers;
- complex learned feature ensembles.

## Approach 1: Query-Level Internal Confidence

Type: non-training probe.

Inspired by:

- Query-Level Uncertainty in Large Language Models: https://arxiv.org/abs/2506.09669
- Language Models (Mostly) Know What They Know: https://arxiv.org/abs/2207.05221
- Learning to Route LLMs with Confidence Tokens: https://openreview.net/forum?id=U08mUogGDM

Question:

```text
Can the local model tell, before answering, whether this query is inside its capability boundary?
```

Probe options:

- Ask a one-token answerability question such as `Can you answer this correctly? yes/no`.
- If intermediate logits are available, compute `P(yes)` across layers and token positions.
- If only final logits are available, use final one-token `P(yes)` as a weaker variant.
- For multiple-choice tasks, ask whether the model can identify the correct option and compute yes/no confidence.

Candidate features:

- mean, max, and min `P(yes)` across layers;
- layer disagreement: standard deviation of `P(yes)`;
- confidence slope from early to late layers;
- entropy or top-2 margin for yes/no;
- contradiction between answerability confidence and answer-token confidence.

Hand-coded routing rule:

```text
if internal_confidence_high and layer_disagreement_low:
    use local action
elif internal_confidence_low:
    use stronger local / self-consistency / frontier action
else:
    use cheap refinement probe
```

Why this fits RouteCode:

It is a direct observability test for `need_large(q)` without learning a full router. Thresholds can be selected on validation only and transferred by recalibration.

## Approach 2: Early-Rollout Instability

Type: non-training or threshold-only probe.

Question:

```text
Does the local model show early signs that its reasoning path is unstable?
```

Run the local model for a short budget only:

- first `N` tokens;
- first reasoning sentence;
- first parsed answer attempt;
- first tool/code sketch;
- two or three cheap samples for consistency if budget allows.

Candidate features:

- average token logprob over the prefix;
- minimum token logprob over the prefix;
- top-2 margin at answer-like tokens;
- entropy spikes;
- malformed or unparsable partial answer;
- answer changes across two or three samples;
- "wait", "check", "maybe", "not sure", or self-correction markers;
- failure to produce a valid option for GPQA/MMLUPro;
- disagreement between short answer and final parsed answer.

Hand-coded routing rule:

```text
if prefix_entropy_low and parsed_answer_valid and no_self_correction:
    local safe
elif early_answer_changes or invalid_format:
    refine with self-consistency or stronger local model
elif high_entropy and benchmark_is_high_risk:
    consider frontier action
```

Important caveat:

Prior free-generation logprob and option-logprob probes were negative. This approach should not repeat those tests directly. The difference is that the signal should be evaluated as a short-path instability and SLM/LLM gap proxy, not as raw confidence alone.

## Approach 3: SLM-vs-Medium Local Divergence

Type: non-training probe.

Inspired by:

- R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing: https://openreview.net/forum?id=DpeJYRFRQY
- Fast and Slow Generating: https://arxiv.org/abs/2406.12295
- Speculative Cascades / Faster Cascades via Speculative Decoding: https://arxiv.org/abs/2405.19261
- Learning to Decode Collaboratively with Multiple Language Models: https://aclanthology.org/2024.acl-long.701/

Question:

```text
Can disagreement between a cheap local SLM and a stronger local model predict when a frontier/large action is valuable?
```

Use local model pairs such as:

- Qwen3-4B vs Qwen3-14B-AWQ;
- Qwen3-4B vs Qwen3-32B-AWQ;
- Qwen3-8B vs Qwen3-32B-AWQ.

Candidate signals:

- parsed-answer disagreement;
- multiple-choice option disagreement;
- first-step rationale disagreement;
- token mismatch rate on the first short reasoning segment;
- top-k JS/KL divergence at constrained option tokens when tokenizers match;
- medium-model confidence minus small-model confidence;
- cross-logprob of the SLM answer under the medium model;
- embedding similarity between short rationales.

Simple scale-gap score:

```text
scale_gap_score(q) =
  answer_disagreement
  + medium_confidence
  - small_confidence
  + small_entropy
```

No learned weights are required at first; use ranks or validation thresholds.

Hand-coded routing rule:

```text
if small_agrees_with_medium and small_confidence_high:
    local safe
elif medium_confidently_disagrees_with_small:
    use medium or frontier depending on cost
elif both_small_and_medium_uncertain:
    use tool, self-consistency, or frontier
else:
    use local/medium based on validation threshold
```

Why this is promising:

It asks for the missing information directly: not "is the query hard?", but "does scale change the predicted answer or reasoning path?" This is closer to the local-vs-frontier oracle gap.

## Approach 4: Semantic Uncertainty

Type: non-training first; optional probe distillation later.

Inspired by:

- Semantic Entropy: https://www.nature.com/articles/s41586-024-07421-0
- Semantic Entropy Probes: https://arxiv.org/abs/2406.15927
- SelfCheckGPT: https://aclanthology.org/2023.emnlp-main.557/

Question:

```text
Does the local model produce one stable meaning, or many incompatible meanings?
```

Probe options:

- sample `n=3` short local answers;
- normalize exact/math answers;
- normalize multiple-choice options;
- for code, run cheap syntax/unit-test checks when available;
- cluster free-form answers by embedding or simple textual equivalence;
- compute answer-cluster entropy.

Candidate features:

- semantic entropy over answer clusters;
- vote margin between top two answer clusters;
- number of unique parsed answers;
- invalid answer rate;
- contradiction rate between sampled rationales.

Hand-coded routing rule:

```text
if semantic_entropy_low and answer_cluster_valid:
    local safe
elif entropy_high and local_cost_budget_allows:
    self-consistency or stronger local model
elif entropy_high and benchmark has high frontier gain:
    frontier action
```

Optional later step:

If semantic entropy is predictive but too expensive, train a cheap pre-generation activation probe to approximate it. This follows the Semantic Entropy Probe idea, but it should be a second-stage compression of a working signal, not the first bet.

## Approach 5: Pre-Generation Activations

Type: label-light option, not the first non-training baseline.

Inspired by:

- LLMs Encode Their Failures: https://arxiv.org/abs/2602.09924
- LLM Router: Rethinking Routing with Prefill Activations: https://arxiv.org/abs/2603.20895
- No Answer Needed: https://arxiv.org/abs/2509.10625

Question:

```text
Before generation, do local hidden states encode whether local action will fail or whether scale will help?
```

Prior local evidence:

- `results/controlled/broad100_qwen4_prefill_activation_router/PREFILL_ACTIVATION_ROUTER_MEMO.md`
- That run used Qwen3-4B activations and ridge utility regression on a 60-query hard slice.
- Validation-selected test utility was `0.4774`, below the cached observable-state reference `0.5168`.
- This is a negative result for that exact setup, not a final rejection of activation probes.

Why revisit:

- The old run predicted full utility vectors directly from one 4B activation table.
- The new target should be narrower: `delta_large(q)`, `need_large(q)`, or local-success risk.
- The new experiments should sweep layers, use SLM-vs-medium divergence labels, and evaluate risk coverage, not only mean utility.

Non-training activation diagnostics:

- hidden-state norm and anisotropy;
- layerwise activation drift;
- distance to train activation centroids;
- Mahalanobis distance to the train distribution;
- nearest-neighbor local-success rate in activation space;
- disagreement between activation-space neighbors and text-embedding neighbors.

Label-light activation probes:

- frozen hidden states only;
- linear/logistic probe only;
- labels from train outcomes only;
- targets:
  - `local_correct`;
  - `need_large`;
  - `delta_large >= tau_gain`;
  - `high_regret_if_local`;
  - `semantic_entropy_high`.

Transfer rule:

The activation encoder remains local. New model or frontier calibration should only require a small validation set to recalibrate thresholds or map labels to actions.

Implementation caveat:

The vLLM OpenAI-compatible API is fine for short rollouts and logprobs, but it does not expose hidden states. Activation extraction may require a local Transformers pass unless the serving stack is extended.

## Route Labels To Try

These are observable route labels, not topic labels:

```text
local_safe
local_uncertain_but_no_large_gain
scale_divergent
semantic_uncertain
needs_medium_local
needs_frontier
needs_tool_not_large_model
```

For RouteCode, evaluate both:

- predicted label -> action table;
- signal threshold -> action directly.

The label version is useful for explainability and transfer; the direct-threshold version is the minimal working baseline.

## Evaluation Protocol

Build one probe table with one row per query:

```text
query_id
split
benchmark
query_text
best_local_action
best_large_action
best_local_utility
best_large_utility
delta_large
need_large
probe_cost
signal_internal_confidence
signal_early_instability
signal_slm_medium_divergence
signal_semantic_entropy
signal_activation_anomaly optional
```

Fit only thresholds and calibration on train/validation. Run test once after method selection.

Required metrics:

- mean utility;
- quality;
- normalized cost;
- strong/frontier call rate;
- oracle regret;
- recovered gap vs local baseline;
- recovered gap vs local-vs-large oracle;
- precision/recall for `need_large`;
- precision at frontier-call caps: 0.10, 0.20, 0.30, 0.40;
- risk-coverage curve for `local_safe`;
- Brier/ECE if a calibrated probability is reported;
- bootstrap CI over queries.

Predictability diagnosis:

If a signal improves AUROC but not utility, report why:

- too many false positives, causing over-escalation;
- false negatives concentrated in GPQA/MMLUPro/MATH500/LiveMathBench;
- signal predicts local failure but not large-model gain;
- signal predicts difficulty but not cost-effective action.

## First Experiment Queue

1. Build cached oracle targets:
   - `best_local_action`;
   - `best_large_action`;
   - `delta_large`;
   - `need_large`.

2. Run non-training signal collection:
   - query-level internal confidence;
   - short early-rollout instability;
   - SLM-vs-medium local divergence;
   - semantic uncertainty from short local samples.

3. Evaluate single-signal threshold policies:
   - choose thresholds on validation;
   - report test utility and frontier-call rate;
   - compare to current cached residual-rule policy and local-vs-frontier oracle.

4. Evaluate hand-coded combinations:
   - local safe only when internal confidence is high, divergence is low, and semantic entropy is low;
   - escalate only when medium confidently disagrees with small or semantic uncertainty is high in known high-gain benchmarks.

5. Only after the above:
   - add pre-generation activation anomaly scores;
   - add frozen linear probes for `need_large` if non-training signals show headroom.

## What Would Count As Progress

Minimum useful result:

```text
validation-selected test utility improves over the current cached residual-rule policy
AND frontier/strong call rate does not increase materially
```

Strong result:

```text
utility within 3% of the local-vs-large oracle
AND precision at the chosen frontier cap is high enough to avoid over-escalation
```

Research result even if utility does not close:

```text
signals separate local-safe from local-risky examples,
but local-risky does not imply large-model gain
```

That would support a predictability-diagnosis story: the oracle exists, but the remaining gap depends on information not available from cheap local probes.

## Do Not Overclaim

- Do not claim RouteCode solves broad routing unless test utility supports it.
- Do not treat dataset-label or oracle labels as deployable.
- Do not present synthetic or hard-slice results as full benchmark evidence.
- Do not call closed-source APIs for this probe phase unless a later task explicitly enables budget, caching, pricing, and token logging.
- Do not claim activation probes work based only on prior literature; the current local Qwen4 activation result is negative.
