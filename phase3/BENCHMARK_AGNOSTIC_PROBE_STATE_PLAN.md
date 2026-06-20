# Benchmark-Agnostic Probe-State Plan

Status: side-chat handoff for the main RouteCode controlled-experiment thread.
This is a method plan, not a result claim.

## Motivation

The current Broad100 probe work shows that the cost-aware oracle is strong, but
generic confidence probes and shallow routers do not recover enough oracle
utility. A tempting next step is to build task-specific checkers for math,
multiple-choice science, or code. That may improve a benchmark slice, but it
risks making the method benchmark dependent.

The next main method should instead learn and evaluate broad probe-state
signals:

```text
query + cheap local behavior -> probe state -> cost-aware route action
```

The probe state should measure the state of the local model and model-pool
disagreement, not hand-coded benchmark correctness rules.

## Core Principle

Do not ask:

```text
Can a custom checker solve this benchmark?
```

Ask:

```text
Does cheap local evidence show that a lower-cost action is trustworthy,
or that routing upward has positive expected utility?
```

The goal is a benchmark-agnostic evidence layer that can transfer across math,
science, code, knowledge, and reasoning tasks.

## How This Fits RouteCode

RouteCode has three conceptual layers.

1. Utility/oracle layer:

```text
cached query-action outcomes
  -> U(q, a) = quality(q, a) - lambda * normalized_cost(q, a)
  -> query oracle / route-action oracle
```

This tells us which action would have been best after observing all outcomes.

2. RouteCode compression layer:

```text
utility vectors over actions
  -> learned discrete route labels
  -> label-to-action policy
```

This tells us what compact routing states exist in the data.

3. Probe-state observability layer:

```text
query + cheap local probes
  -> observable probe state
  -> predicted route label or cost-aware action gate
```

This is the deployability layer. It tests whether the routing states can be
observed before paying for expensive actions.

The current bottleneck is layer 3. The oracle and compressed structure are
useful, but the cheap observable evidence does not yet identify the right
state/action well enough.

## Proposed Method

Build a universal probe-state table for every query. Each row should contain
only features that are available before the final expensive route action.

Candidate benchmark-agnostic probe features:

- small-model answer validity and parseability;
- local answer agreement across small/medium local models;
- self-consistency entropy across cheap local samples;
- answer stability under prompt variants or paraphrases;
- early-rollout instability from short prefixes;
- answer length and reasoning volatility;
- local-vs-medium parsed-answer disagreement;
- semantic distance between local answers;
- refusal, empty answer, malformed output, or format failure;
- available logprob or margin features from vLLM;
- cost-normalized train prior for expected upward gain.

Then learn or select a small discrete probe state:

```text
probe_state z_probe in {1, ..., K}
```

The policy can be:

```text
query -> probe features -> probe_state -> action
```

or combined with RouteCode:

```text
query -> predicted RouteCode label
query + local probes -> probe_state
(RouteCode label, probe_state) -> action
```

The second version is the cleaner research story: RouteCode learns the utility
states, and the probe layer measures whether those states are observable from
cheap local behavior.

## First Experiment To Run

Use existing cached Broad100 outputs first. Do not add task-specific checkers.

1. Build `table_probe_state_features.csv` with one row per query and split.
2. Fit probe-state quantizers on train only:
   - KMeans over standardized probe features for K = 2, 4, 8, 16;
   - optional decision-aware clustering using train utility regret;
   - no benchmark ID in the first main run.
3. For each probe state, choose the best action on train by mean utility.
4. Select K and thresholds on validation.
5. Report held-out test utility, quality, normalized cost, frontier-call rate,
   oracle utility ratio, and gap to oracle.
6. Add a benchmark-heldout transfer test:
   - fit probe states on some benchmark families;
   - hold out entire benchmark families;
   - compare to benchmark lookup and text-only routing.

## Required Ablations

Run these as separate rows so the story is clear:

- text-only RouteCode predictor;
- probe-only state, no benchmark ID;
- probe-only state with benchmark ID as a diagnostic upper bound;
- RouteCode label plus probe state;
- direct shallow router over the same features;
- oracle RouteCode label as an upper bound;
- oracle local-vs-large gate as an upper bound.

The important comparison is not just whether the probe state improves Broad100.
It is whether the probe state transfers better than benchmark lookup and
text-only learned routing.

## What Not To Do As Main Method

Do not make benchmark-specific checkers the main Phase 3 method:

- no symbolic math verifier as the main result;
- no GPQA-only option eliminator as the main result;
- no code-execution-only verifier as the main result;
- no benchmark-name lookup as the headline method.

Those can remain diagnostics or later domain plugins, but the main method needs
to work as a general probe-state abstraction.

## Success Criteria For This Direction

A useful result would show:

- probe-state policy improves over text-only and benchmark lookup on held-out
  test utility;
- probe-state policy transfers to held-out benchmark families better than
  benchmark-specific policies;
- the learned states have interpretable code cards describing model-pool
  behavior, such as stable local consensus, local disagreement, medium override,
  high volatility, or expensive-action-not-worth-it;
- frontier-call rate remains within the controlled-experiment target;
- all thresholds and state-action tables are fit on train/validation only.

Do not claim the method works broadly unless the benchmark-heldout transfer
test supports it.

## Main-Thread Handoff

The main thread should implement this as the next broad method experiment:

```text
Benchmark-agnostic ProbeCode / probe-state RouteCode
```

Recommended first artifact names:

- `experiments/199_probe_state_routecode.py`
- `results/controlled/broad100_probe_state_routecode/`
- `table_probe_state_features.csv`
- `table_probe_state_policy_all.csv`
- `table_probe_state_policy_selected.csv`
- `PROBE_STATE_ROUTECODE_MEMO.md`

