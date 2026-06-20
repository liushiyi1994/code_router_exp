# RouteCode Paper Notes

Last updated: 2026-06-18

These notes are generated from current Phase H claim gates and external-baseline readiness artifacts. They are not a paper draft and should not be read as final claims.

## Recommended Framing

- Status: `information_frontier_diagnostic`.
- Recommended framing: information-frontier and benchmark-diagnostic paper. Do not claim that few inferred bits are enough; current evidence supports low-rate oracle structure, modest deployable inferred-label recovery, and diagnostic calibration/transfer threads.

## Claim Posture

| item | status | key_value | interpretation |
| --- | --- | --- | --- |
| low_rate_oracle_codes | diagnostic_supported | best=1.0000; worst=0.9535 | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| small_inferred_labels | not_supported | best=0.3459; worst=0.0233 | Do not claim that small inferred route labels recover most routing performance across current runs. |
| model_pool_transfer | mixed_evidence | best=0.3083; worst=-0.0537 | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| new_model_calibration | diagnostic_alive | best=0.8140; worst=0.2339 | Use diagnostic framing; broader coverage is still required for a paper-level claim. |
| benchmark_diagnosis | mixed_evidence | best=0.7904; worst=0.1198 | Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it. |
| adaptive_refinement | not_supported | best=0.2683; worst=0.1521 | Current cross-run evidence does not support this claim. |

## Controlled Exact-Math Phase 3 Update

Evidence: `src/routecode/controlled/exact_math_tools.py`, `results/controlled/tool_augmented_aime_policy/table_tool_augmented_aime_policy_selected.csv`, `results/controlled/tool_augmented_aime_policy/TOOL_AUGMENTED_AIME_POLICY_MEMO.md`, `results/controlled/tool_augmented_fresh_split_confirmation/table_locked_fresh_split_summary.csv`, `results/controlled/exact_math_gpt_route_judge/table_exact_math_gpt_route_judge_selected.csv`, `results/controlled/observable_state_policy/table_observable_state_policy_selected.csv`, `results/controlled/budgeted_strong_rescue_policy/table_budgeted_strong_rescue_policy_selected.csv`, `results/controlled/budgeted_strong_rescue_policy/table_budgeted_strong_rescue_policy_fresh_summary.csv`, `results/controlled/frontier_rate_feasibility/table_frontier_rate_feasibility_summary.csv`, `results/controlled/frontier_rate_feasibility/FRONTIER_RATE_FEASIBILITY_MEMO.md`, `results/controlled/budgeted_frontier_gain_policy/table_budgeted_frontier_gain_policy_selected.csv`, `results/controlled/RUN_REPORT.md`, and `results/controlled/EXPECTED_RESULTS_STATUS.md`.

On the current 66-query held-out exact-math slice, expanded deterministic exact-math tools fire on `110/321` cached rows and score `110/110` correct, including `30` held-out test rows. With these tools, the validation-selected min-cost tool-augmented AIME policy reaches quality `0.8333`, quality gap `0.0152`, normalized cost `0.0463`, utility ratio `1.0743`, and frontier-call rate `0.1061`. The validation-selected quality-conservative row reaches quality `0.8485`, gap `0.0000`, normalized cost `0.1744`, utility ratio `1.0353`, and frontier-call rate `0.1667`. These current-split rows meet the Phase 3 quality, utility, cost, and frontier-rate targets. Claude was not used.

Follow-up checks: the all-AIME route cache covers 60/60 AIME rows at total GPT-5.5 route-label cost `$0.2008`. With frontier-aware validation selection, locked 10-seed stratified fresh-split confirmation now passes `10/10` fresh splits for both selected tool-augmented rows. The min-cost selected rows have mean quality `0.8258`, worst quality `0.7727`, max gap `0.0152`, mean normalized cost `0.1244`, mean utility ratio `1.0920`, and max frontier-call rate `0.3333`. A broader all-query GPT-5.5 route judge labeled 321 rows for `$1.2905`; after tool expansion, its quality-conservative selected held-out row reaches quality `0.8333`, gap `0.0152`, normalized cost `0.2630`, solver frontier rate `0.1970`, and utility ratio `0.9746`, but it still uses a remote route judge on `0.5455` of held-out rows.

New method probe: a train-learned budgeted strong-rescue layer over agreement baselines now has a current-split min-cost row that reaches quality `0.8485`, gap `0.0000`, normalized cost `0.0666`, utility ratio `1.0849`, and frontier-call rate `0.3485`. However, the strong-rescue fresh rows still overuse frontier solvers in most resplits when the frontier gate is enforced, so the split-stable deployable result should be attributed to the tool-augmented policy, not this rescue layer.

Frontier-rate feasibility: after expanded deterministic tools, the cached oracle-bound diagnostic supports even the stricter `0.25` frontier-call target on the current split and across fresh stratified splits. At a `0.25` frontier cap, the current utility objective reaches quality `0.8485`, gap `0.0000`, normalized cost `0.0269`, utility ratio `1.0000`, and actual frontier-call rate `0.0455`. Fresh-split pass_all is `10/10` at the 0.25 and 0.40 caps for both utility and quality objectives.

Failed deployable cap attempt: a local-first learned frontier-gain policy can stay under 0.40 frontier-call rate and reaches strong current-split quality, but it still misses the utility gate. After tool expansion, the selected held-out row has quality `0.8636`, gap `-0.0152`, normalized cost `0.2781`, utility ratio `0.9133`, and frontier-call rate `0.3939`.

Calibration refresh: a cached exact-math new-model calibration curve now holds out `gemini-3.5-flash-strong-solve` from a base pool of Qwen probe/4B/8B/14B, Gemini 3.5 Flash, and GPT-5.5. A K=4 RouteCode state table fit on train-only tool-abstain rows reaches held-out quality `0.8485` with `4` cached target-model calibration evaluations through active state calibration; the direct logistic router reaches best quality `0.7273` under the swept budgets.

Ablation refresh: exact-math component ablations show that deterministic tools are the main component behind the current min-cost result. Disabling tools drops held-out quality from `0.8333` to `0.7879` and raises frontier-call rate from `0.1061` to `0.5152`. The min-cost policy remains useful under provider-price sensitivity, with utility `0.7524` at a 5x frontier price multiplier.

Broad benchmark bridge: `results/controlled/PHASE3_BROAD_LLMROUTERBENCH_EVIDENCE.md` summarizes the existing `results/llmrouterbench_broad20` real outcome matrix into the controlled evidence ledger. It covers 8/9 requested Phase 3 benchmarks with 14,041 queries, 20 models, and 280,820 query-model rows; GSM8K is missing. Broad20 supports low-rate oracle codes as a diagnostic (`regret_routecode_oracle_labels`, K=16, recovered gap 0.9681), but does not support deployable inferred labels (best D2 inferred row recovered gap 0.0906). This is broad real evidence with a released LLMRouterBench model pool, not the controlled GPT-5.5/Gemini-3.5/local-vLLM pool.

Controlled broad Stage 0 update: `experiments/124_phase3_broad_target_manifest.py` now builds a 45-task live manifest from real local benchmark sources: 5 each from GSM8K, MATH500, AIME, HumanEval, MBPP, GPQA, MMLU-Pro, LiveMathBench, and BBH. HumanEval and MBPP use embedded function/assert pass@1 scoring; LiveCodeBench remains excluded because the full local test payload is still a Git LFS pointer. The live runner completed 45/45 calls for GPT-5.5, Gemini 3.5 Flash, Qwen3-4B, Qwen3-8B, and Qwen3-14B-AWQ with no Claude. Final cached output cost summary is `$0.2512` for GPT-5.5 and `$0.0159` for Gemini. Mean qualities were: GPT-5.5 `0.6222`, Gemini `0.2889`, Qwen3-14B-AWQ `0.5556`, Qwen3-4B `0.2889`, and Qwen3-8B `0.2667`. The broad target-pool cost-aware oracle reaches quality `0.7778`, utility `0.7444`, and frontier-call rate `0.1111`, showing useful local/frontier complementarity on this 9-dataset Stage 0 sample.

Broad Stage 0 method candidate: `experiments/125_phase3_broad_target_method_package.py` now builds a cached tool/probe/profile package from `results/controlled/live_broad_stage0/model_outputs.parquet` and `results/controlled/live_broad20_stage0/model_outputs.parquet`. The selected `tool_probe_profile_v4` policy uses deterministic exact-math tools, local answer agreement, Qwen3-32B-AWQ for code and valid GPQA answers, Qwen14 fallback for invalid GPQA choices, and selective GPT/Gemini fallback. On the tiny 9-query Stage 0 test split, it reaches quality `0.6667` versus tool-inclusive cost-aware oracle `0.6667`, utility ratio `0.9965`, and frontier-call rate `0.2222`. On the scaled 180-query broad20 package with GPT-5.5, Gemini 3.5 Flash, Qwen3-4B, Qwen3-8B, Qwen3-14B-AWQ, and Qwen3-32B-AWQ, it reaches held-out quality `0.8333` versus oracle `0.8611` (gap `0.0278`), utility ratio `0.9691`, and frontier-call rate `0.2500`. The broad20 package now also includes `40` calibration rows, `8` ablation rows, `12` sensitivity rows, and `fig_broad_target_calibration.pdf`. This supports the broad scaled Stage 0 quality, utility, and frontier-rate gates on the current split, but it is still not the full-size broad run because the package has 180 prompts rather than the Phase 3 plan's roughly 100 examples per benchmark.

Completion audit refresh: `results/controlled/table_phase3_goal_completion_audit.csv` now verifies that all 21 named Phase 3 output files/configs exist, the broad controlled scaffold has 8 benchmarks and 7 models as a surrogate, the controlled real exact-scored evidence is the 3-dataset exact-math package with 321 rows and 66 held-out test rows, the external broad20 matrix supplies supporting broad real evidence over 8/9 requested benchmarks, and the controlled target-pool Stage 0 covers 45 prompts across 9 runnable datasets with 225/225 successful model rows. The audit also tracks `target:broad_stage0_profile_method` as `supported_on_stage0_split`, `target:broad20_scaled_profile_method` as `supported_on_scaled_stage0_split`, `stage2_5:broad20_scaled_method_package` as `complete_on_scaled_stage0`, and `stage2_5:broad_real_benchmark_package` as `partial_scaled_stage0_method_supported`. The remaining not-complete scope is full-size broad coverage using the target model pool.

Claim posture: this now supports "the target is achieved on the current held-out exact-math slice" under the configured model/action pool and cost-aware objective, supports split-stable validation-selected tool policy results over the 10 locked fresh stratified resplits, supports the state-level new-model calibration mechanism on cached exact-math rows, and has exact-math ablation/sensitivity evidence. Broad20 independently supports the low-rate oracle-code diagnostic but not deployable inferred labels. The controlled broad target-pool Stage 0 is now live and cached over 9 runnable datasets; the current `tool_probe_profile_v4` policy meets quality, utility, and frontier-rate gates on the scaled broad20 split, and the scaled broad20 package now has calibration, ablation, and sensitivity artifacts. Do not generalize to the full controlled Phase 3 model-pool claim until the broad run is expanded beyond the 180-prompt scaled package.

## External Baseline Posture

| item | status | key_value | interpretation |
| --- | --- | --- | --- |
| readiness_overview | partial | 38 rows; 30 runnable; 22 exact | RouteCode-compatible metric rows available: routecode_local_embedllm_knn_metric, routecode_local_frugalgpt_metric, routecode_local_routellm_mf_metric, routecode_upstream_avengerspro_metric. Blocked rows remain. |
| best_route_train_cli | blocked |  | missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender |
| modelsat_train_cli | blocked |  | missing_modelsat_base_model_checkpoint;missing_modelsat_embedding_model_checkpoint;missing_python_modules:nltk,deepspeed |
| routellm_bert_cli | blocked |  | missing_bert_checkpoint |
| routerdc_train_cli | blocked |  | missing_routerdc_local_model_checkpoint;missing_python_modules:deepspeed |

## Research Flow Completion

Evidence: `results/table_research_flow_completion.csv` and `results/phase_h_research_flow_completion_audit.md`.

- Complete phases: `10`.
- Deferred phases: `1` (`phase_d5_adaptive_refinement`).
- Blocked phases: `0`.
- Incomplete phases: `0`.
- Current non-complete reasons: none.

## Remaining Blockers

- `best_route_train_cli`: missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender
- `modelsat_train_cli`: missing_modelsat_base_model_checkpoint;missing_modelsat_embedding_model_checkpoint;missing_python_modules:nltk,deepspeed
- `routellm_bert_cli`: missing_bert_checkpoint
- `routerdc_train_cli`: missing_routerdc_local_model_checkpoint;missing_python_modules:deepspeed

## Guardrails

- Do not frame the project as saving router tokens.
- Do not claim that small inferred route labels recover most routing performance unless the pre-committed threshold is met with confidence intervals.
- Keep calibration, transfer, and benchmark-diagnosis claims diagnostic until broader external-baseline and robustness coverage is available.
- Keep adaptive refinement deferred unless a stronger deployable residual-risk gate appears.

## Broad100 Probe-Signal Update

Evidence: `experiments/162_pairwise_action_ranker.py`,
`results/controlled/broad100_pairwise_action_ranker/table_pairwise_action_ranker_selected.csv`,
`results/controlled/broad100_pairwise_action_ranker/PAIRWISE_ACTION_RANKER_MEMO.md`, and
`results/controlled/broad100_pairwise_action_ranker/table_pairwise_logistic_residuals.csv`.

Pairwise action ranking over cached self-consistency evidence is a modest
positive but remains below the Phase 3 target. The validation-selected pairwise
logistic row reaches held-out utility `0.7235`, quality `0.7791`,
oracle-utility ratio `0.8549`, frontier-call rate `0.2616`, strong-call rate
`0.2267`, and self-action rate `0.5116`. This improves over the previous
validation-selected E5 self-action gate (`0.7161` utility), but remains far from
the base/self/strong action-set oracle (`0.8076` utility, `0.8430` quality).

Residuals remain concentrated in GPQA, MMLUPro, MATH500, and LiveMathBench.
The error pattern is mixed: the ranker overuses self on some cases where base
or strong is better, overuses strong on some cases where local/self is enough,
and still misses strong on some hard math and MMLUPro cases. A local-answer
signature prototype was negative as a validation-selected method, so the next
useful direction is stronger confidence/activation probes or verifier evidence,
not more capacity over the same cached self-consistency features.

## Broad100 Residual Rule Update

Evidence: `experiments/163_residual_confidence_rule_policy.py`,
`results/controlled/broad100_residual_confidence_rule_policy/table_residual_confidence_rule_policy_selected.csv`,
`results/controlled/broad100_residual_confidence_rule_policy/RESIDUAL_CONFIDENCE_RULE_POLICY_MEMO.md`, and
`results/controlled/broad100_residual_confidence_rule_policy/table_residual_confidence_rule_policy_residuals.csv`.

Residual confidence rules over the pairwise logistic ranker are the current
best validation-driven cached policy, but still not a solution. The strict
validation-best rule reaches held-out utility `0.7259` and quality `0.7907`.
The validation near-best rule with a cost/frontier tiebreak reaches held-out
utility `0.7328`, quality `0.7849`, oracle-utility ratio `0.8660`,
frontier-call rate `0.2500`, strong-call rate `0.2151`, and self-action rate
`0.5000`. This improves over the pairwise logistic row (`0.7235`) but remains
`0.0688` utility below the base/self/strong action-set oracle (`0.8016`).
Residual regret remains concentrated in GPQA, MMLUPro, MATH500, and
LiveMathBench, so the claim target is still blocked by observability rather
than by a missing cheap calibration layer over the same cached features.

## Broad100 Local vLLM Risk-Probe Update

Evidence: `experiments/164_vllm_residual_risk_probe.py`,
`results/controlled/broad100_vllm_residual_risk_probe_qwen14b/table_vllm_residual_risk_selected.csv`,
`results/controlled/broad100_vllm_residual_risk_probe_qwen14b/VLLM_RESIDUAL_RISK_MEMO.md`, and
`results/controlled/broad100_vllm_residual_risk_probe_qwen14b/table_vllm_residual_risk_probe.csv`.

The Qwen3-14B-AWQ residual-risk probe is a negative result. It collected
`160/160` local vLLM judgments on GPQA, MMLUPro, MATH500, and LiveMathBench
validation/test rows with no provider API calls. Validation still selects the
residual-rule baseline (`0.8015` validation utility and `0.7328` held-out
utility). Thresholded vLLM override policies either tie the baseline by making
no effective changes or degrade held-out utility, with the worst swept row at
`0.6696`. This does not close the gap to the `0.8016` base/self/strong
action-set oracle.

## Broad100 Frontier Agreement-Probe Update

Evidence: `experiments/165_frontier_agreement_probe_policy.py`,
`results/controlled/broad100_frontier_agreement_probe_policy/table_frontier_agreement_probe_policy_selected.csv`,
`results/controlled/broad100_frontier_agreement_probe_policy/FRONTIER_AGREEMENT_PROBE_POLICY_MEMO.md`, and
`results/controlled/broad100_frontier_agreement_probe_policy/table_frontier_agreement_probe_policy_all.csv`.

Frontier/local answer agreement is diagnostically strong but not a deployable
improvement after cost accounting. The evaluator charges the Gemini or GPT probe
call whenever it is used and not reused as the final selected answer. Validation
selects a Gemini agreement policy with held-out utility `0.7309` and quality
`0.7849`, below the residual-rule baseline (`0.7328` utility). Validation
selects a GPT agreement policy with held-out utility `0.7311` and quality
`0.7907`, also below the baseline because GPT probe cost offsets the agreement
signal. The best held-out diagnostic Gemini agreement row reaches `0.7359`
utility, but it is test-picked and still far from the `0.8016` base/self/strong
action-set oracle. This is another negative result for adding a shallow
agreement layer over the current cached evidence.

## Broad100 SLM/LLM Early-Signal Probe Pilot

Evidence: `experiments/166_slm_llm_early_signal_probe_pilot.py`,
`results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/table_slm_llm_oracle_targets.csv`,
`results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/table_slm_llm_threshold_policies_selected.csv`,
`results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/table_slm_llm_precision_at_caps.csv`,
`results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/table_vllm_answerability_probe.csv`, and
`results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/SLM_LLM_EARLY_SIGNAL_PROBE_MEMO.md`.

The first mostly non-training SLM/LLM early-signal pilot is implemented and
negative as a deployable method, despite useful diagnostic signal. The target
table compares per-query best local action against per-query best large action
with `U = quality - 0.35 * normalized_cost`; local actions are deterministic
math tools plus cached Qwen3-4B, Qwen3-8B, and Qwen3-14B-AWQ actions, while
large actions include Qwen3-32B-AWQ, Qwen3-32B-AWQ self-consistency, Gemini
3.5 Flash, GPT-5.5, and Gemini strong solve. On held-out test, always-local
utility is `0.6919`, always-large utility is `0.7689`, and the diagnostic
local-vs-large oracle is `0.8463`. Validation selects a threshold-only combined
risk rule that reaches held-out utility `0.7381`, quality `0.7616`,
oracle-utility ratio `0.8721`, recovered gap vs local `0.2992`, large-call
rate `0.7733`, and frontier-call rate `0.1860`. This improves over
always-local but underperforms always-large and is far from the oracle.

The useful observation is that early-rollout instability and semantic
uncertainty identify some upward-routing cases at low coverage: at a 10%
held-out cap they reach precision `0.5294` and recall `0.2903`. Query-only
surface risk is weaker but nontrivial (`0.2941` precision at the same cap),
SLM-vs-medium divergence is weak (`0.0588` precision), and the local Qwen3-14B
vLLM one-token answerability probe is too noisy to use (`344/344` successful
val/test rows but many non-YES/NO first tokens; held-out 10% cap precision
`0.0588`, AUROC `0.5569`). Do not claim this solves upward routing. It supports
the narrower claim that cached local instability has some early warning signal,
but the next useful probe needs constrained YES/NO logit scoring or local
final-answer confidence/activation evidence.

## Broad100 Constrained YES/NO Probe Update

Evidence: `experiments/167_constrained_yesno_probe_policy.py`,
`results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_probe.csv`,
`results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_policy_selected.csv`,
`results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_precision_at_caps.csv`,
and `results/controlled/broad100_constrained_yesno_probe_qwen14b/CONSTRAINED_YESNO_PROBE_MEMO.md`.

The constrained Qwen3-14B-AWQ YES/NO probe is a partial positive but not a
solution. Unlike the earlier free-token answerability call, this run uses local
vLLM `logit_bias` over single-token YES/NO variants and converts top-logprobs
into a binary local-trust score. It collected `688/688` successful val/test
prompts across query-only and local-evidence prompt modes, with no provider API
calls. The single-signal validation-selected query-only policy reaches held-out
utility `0.7573`, quality `0.7791`, oracle-utility ratio `0.8949`, large-call
rate `0.5698`, and frontier-call rate `0.1221`. This improves on the previous
selected early-signal threshold (`0.7381`) but remains below always-large
utility (`0.7689`). The local-evidence constrained score has a stronger
held-out diagnostic row (`0.7861` utility, `0.8256` quality, `0.9289`
oracle-utility ratio), but that row is test-picked and not deployable.

## Broad100 Constrained YES/NO Combo Policy

Evidence: `experiments/168_constrained_yesno_combo_policy.py`,
`results/controlled/broad100_constrained_yesno_combo_policy/table_constrained_yesno_combo_policy_selected.csv`,
`results/controlled/broad100_constrained_yesno_combo_policy/table_constrained_yesno_combo_policy_all.csv`, and
`results/controlled/broad100_constrained_yesno_combo_policy/CONSTRAINED_YESNO_COMBO_POLICY_MEMO.md`.

A non-training combo over the constrained YES/NO scores gives the first
validation-selected policy in this branch that beats always-large utility while
using much less frontier. The validation-selected two-signal AND rule reaches
held-out utility `0.7756`, quality `0.7965`, oracle-utility ratio `0.9165`,
recovered gap vs local `0.5425`, large-call rate `0.4128`, frontier-call rate
`0.1163`, precision `0.2676`, and recall `0.6129`. This is useful progress:
it beats always-large utility (`0.7689`) and uses far fewer frontier calls.
However, it still misses the Phase 3 target because the local-vs-large oracle
is `0.8463` utility and `0.8721` quality. The best held-out diagnostic combo
row reaches `0.7828` utility, but it is test-picked and still only `0.9251` of
oracle utility. The current supported claim is therefore not "solved routing";
it is that constrained local confidence adds deployable signal, but the
remaining observability gap is still large.

## Broad100 Benchmark-Aware And Benchmark-Composed YES/NO Update

Evidence: `experiments/169_benchmark_aware_yesno_policy.py`,
`experiments/170_benchmark_composed_yesno_policy.py`,
`results/controlled/broad100_benchmark_aware_yesno_policy/BENCHMARK_AWARE_YESNO_POLICY_MEMO.md`,
`results/controlled/broad100_benchmark_composed_yesno_policy/BENCHMARK_COMPOSED_YESNO_POLICY_MEMO.md`,
and the selected/all policy tables in those folders.

The constrained Qwen3-14B-AWQ YES/NO cache was extended to train rows before
these policy runs. The cache now has `1720/1720` successful local vLLM prompt
rows across train, validation, and held-out test, with no provider API calls.

Benchmark-aware raw thresholding is negative. The validation-selected
`benchmark_cap_c_0.25` row reaches held-out utility `0.7576`, quality
`0.7733`, oracle-utility ratio `0.8952`, large-call rate `0.2500`, and
frontier-call rate `0.0988`. Its best held-out diagnostic row reaches `0.7723`
utility, but that is test-picked and still below the previous selected combo
policy.

Benchmark-composed fixed policies are the current best validation-selected
local-vs-large diagnostic result, but they still do not meet the claim target.
The selected `benchmark_composed_eps0_utility` row reaches held-out utility
`0.7892`, quality `0.8140`, oracle-utility ratio `0.9325`, recovered gap vs
local `0.6303`, large-call rate `0.4419`, and frontier-call rate `0.1744`.
This improves over the previous constrained YES/NO combo (`0.7756` utility and
`0.9165` oracle ratio) while staying under the frontier-rate target. It is
still below 95% of the local-vs-large oracle utility (`0.8040`) and below the
within-three-quality-points target (`0.8421` quality). The best test-picked
composed diagnostic row reaches `0.7978` utility and `0.8256` quality, but it
is not validation-selected and still below target.

Paper interpretation: the empirical story is now clearer. The oracle is strong
and locally compressible by benchmark/action structure, and constrained local
confidence provides real deployable signal. The applied method still cannot
achieve oracle status because it does not observe enough query-specific
evidence to know when local actions fail. Do not claim solved upward routing or
within-3%-of-oracle performance from this branch.

## Broad100 Tool-Aware Benchmark-Composed Update

Evidence: `experiments/171_tool_aware_benchmark_composed_policy.py`,
`results/controlled/broad100_tool_aware_benchmark_composed_policy/TOOL_AWARE_BENCHMARK_COMPOSED_POLICY_MEMO.md`,
`results/controlled/broad100_tool_aware_benchmark_composed_policy/table_tool_aware_benchmark_composed_policy_selected.csv`,
`results/controlled/broad100_tool_aware_benchmark_composed_policy/table_tool_aware_benchmark_composed_policy_all.csv`,
and `results/controlled/broad100_tool_aware_benchmark_composed_policy/table_tool_availability_summary.csv`.

Tool-aware benchmark composition is the first validation-selected result that
clears the local-vs-large diagnostic thresholds. The new route-time signal is
deterministic exact-math tool availability. When the tool produces a non-empty
answer, tool-aware candidate policies force the local side; otherwise they use
the same fixed benchmark-composed constrained-confidence menu from experiment
170. The target-aware validation selector requires validation oracle-utility
ratio `>=0.95`, validation quality gap `<=0.03`, and frontier-call rate
`<=0.40`, then tie-breaks by validation quality, need-large recall, utility,
and cost. It selects `tool_aware_benchmark_composed_eps0.01_recall_then_quality`.

Held-out test result for that selected row:

- utility `0.8163` with bootstrap CI `[0.7601, 0.8674]`;
- quality `0.8488`;
- oracle-utility ratio `0.9646`;
- recovered gap vs local `0.8061`;
- large-call rate `0.6395`;
- frontier-call rate `0.1860`;
- need-large precision `0.2455`;
- need-large recall `0.8710`.

Against the held-out local-vs-large oracle (`0.8463` utility, `0.8721`
quality), this clears the 95%-utility threshold (`0.8039`) and the
within-3-quality-points threshold (`0.8421`). It also keeps frontier-call rate
below the `25%--40%` target.

Paper interpretation: this supports a stronger and more concrete story than
the previous constrained-confidence result. The missing observable information
was not only local confidence; for exact math slices, a verifiable deterministic
tool signal can safely suppress expensive upward routing. However, this is
still a local-vs-large diagnostic abstraction. Choosing local currently means
using the target table's best local side. Do not present this as a fully
deployed multi-action router until the local branch is replaced by an actual
local action selector and re-evaluated.

## Broad100 Tool-Aware Deployed-Action Bridge

Evidence: `experiments/172_tool_aware_deployed_action_policy.py`,
`results/controlled/broad100_tool_aware_deployed_action_policy/TOOL_AWARE_DEPLOYED_ACTION_POLICY_MEMO.md`,
`results/controlled/broad100_tool_aware_deployed_action_policy/table_tool_aware_deployed_action_policy_selected.csv`,
`results/controlled/broad100_tool_aware_deployed_action_policy/table_tool_aware_deployed_action_policy_all.csv`,
and `results/controlled/broad100_tool_aware_deployed_action_policy/table_tool_aware_deployed_action_choices.csv`.

The deployed-action bridge is a negative but important result. It makes no
new model calls and uses cached outputs only. It replaces the experiment 171
diagnostic target-table sides with concrete action selection using train-only
action priors, deterministic exact-math tool availability, local answer
agreement, and validation-selected thresholds.

The best validation-selected deployable policy is
`tool_then_171_gate_local_consensus_large_prior`. On held-out test it reaches:

- utility `0.7058`;
- quality `0.7849`;
- oracle-utility ratio `0.8340`;
- frontier-call rate `0.4709`;
- strong-or-frontier call rate `0.8081`;
- need-large recall `0.9355`, but precision only `0.2086`.

The best validation-selected threshold-action family reaches held-out utility
`0.6999` and quality `0.7733`. The best held-out diagnostic threshold row
reaches utility `0.7261` and quality `0.8140`, but it is test-picked and not a
deployable result.

Against the full held-out cost-aware oracle (`0.8463` utility, `0.8721`
quality), the 97%-oracle utility target is `0.8209` and the within-three-
quality-points target is `0.8421`. No deployable 172 row clears either target.
Even the diagnostic `oracle_local_vs_large_gate_train_prior` row, which uses
the true local-vs-large need label but train-only concrete action priors,
reaches only `0.7254` utility and `0.7558` quality.

Paper interpretation: the 171 result is a useful diagnostic success, but not a
deployed router. The 172 bridge shows the remaining hard problem is concrete
action identity: which local/tool/strong/frontier answer should be trusted for
this query. The next paper-safe claim is therefore an observability
decomposition: current cheap probes can make the local-vs-large abstraction
look close to oracle, but actual multi-action deployment still needs an
answer adjudicator, verifier, or stronger pre-generation/activation evidence.

## Broad100 Benchmark-Composed Deployed-Action Policy

Evidence: `experiments/173_benchmark_composed_deployed_action_policy.py`,
`results/controlled/broad100_benchmark_composed_deployed_action_policy/BENCHMARK_COMPOSED_DEPLOYED_ACTION_POLICY_MEMO.md`,
`results/controlled/broad100_benchmark_composed_deployed_action_policy/table_benchmark_composed_deployed_action_selected.csv`,
`results/controlled/broad100_benchmark_composed_deployed_action_policy/table_benchmark_composed_deployed_action_all.csv`,
`results/controlled/broad100_benchmark_composed_deployed_action_policy/table_benchmark_composed_deployed_action_choices.csv`,
and `results/controlled/broad100_benchmark_composed_deployed_action_policy/table_benchmark_composed_deployed_action_query_choices.csv`.

The benchmark-composed deployed-action policy is a second negative result for
actual multi-action deployment. It makes no new model calls and uses cached
outputs only. The experiment chooses a concrete policy per benchmark on
validation from train-only action priors, fixed concrete models, deterministic
tool-first variants, local consensus, the 171 local-vs-large gate, and selected
threshold rules from the 172 evidence table.

The validation utility selector picks
`benchmark_composed_deployed_eps0_utility`. On held-out test it reaches:

- utility `0.6916`;
- quality `0.7558`;
- oracle-utility ratio `0.8172`;
- frontier-call rate `0.4651`;
- strong-or-frontier call rate `0.6744`.

The target-quality validation selector picks
`benchmark_composed_deployed_eps0.04_quality`; on held-out test it reaches
utility `0.7048`, quality `0.7907`, oracle-utility ratio `0.8328`,
frontier-call rate `0.5000`, and strong-or-frontier call rate `0.7093`. The
best held-out benchmark-composed diagnostic row reaches utility `0.7097` and
quality `0.7849`, but it is test-picked and still below the simpler 172
deployed-policy comparison row (`0.7116` utility, `0.7907` quality).

Against the full held-out cost-aware oracle (`0.8463` utility, `0.8721`
quality), this does not approach the 97%-oracle utility target (`0.8209`) or
the within-three-quality-points target (`0.8421`). Paper interpretation:
benchmark-level composition over existing concrete action heuristics overfits
validation and does not repair the action-identity bottleneck. The next
paper-useful probe should provide stronger observable evidence about which
candidate answer/action is correct, such as task-specific verifiers, execution
checks, or pre-generation/activation signals.

## Broad100 Answer-Support Action Policy

Evidence: `experiments/174_answer_support_action_policy.py`,
`results/controlled/broad100_answer_support_action_policy_benchmark_threshold/ANSWER_SUPPORT_ACTION_POLICY_MEMO.md`,
`results/controlled/broad100_answer_support_action_policy_benchmark_threshold/table_answer_support_action_policy_selected.csv`,
`results/controlled/broad100_answer_support_action_policy_benchmark_threshold/table_answer_support_action_policy_all.csv`,
`results/controlled/broad100_answer_support_action_policy_benchmark_threshold/table_answer_support_action_policy_query_choices.csv`,
and `results/controlled/broad100_answer_support_action_policy_benchmark_threshold/table_answer_support_train_features.csv`.

The answer-support action policy is a cached-only action-identity probe. It
makes no new model calls. It fits train-only reliability tables for local
answer groups, then selects global or per-benchmark thresholds on validation to
decide when to trust the supported local answer group versus escalating to the
train-best large action.

The best validation-selected benchmark-threshold row,
`benchmark_support_else_large_eps0_utility`, reaches validation utility
`0.7842` and quality `0.8605`, but on held-out test it drops to:

- utility `0.7035`;
- quality `0.7849`;
- oracle-utility ratio `0.8313`;
- frontier-call rate `0.4826`;
- strong-or-frontier call rate `0.7035`.

The best validation-selected global support-threshold row reaches held-out
utility `0.7054` and quality `0.7907`. The best held-out support diagnostic
row reaches utility `0.7130` and quality `0.7965`, but it is not
validation-selected and remains far below the full cost-aware oracle (`0.8463`
utility, `0.8721` quality).

Paper interpretation: local answer support is a meaningful observable signal
for some slices, especially GSM8K, MATH500, HumanEval, and MBPP when many local
answers agree. It is not sufficient for broad deployed routing because GPQA,
MMLUPro, and AIME can have high local agreement on wrong answers. This
strengthens the action-identity bottleneck claim: simple local consensus and
train-calibrated support do not replace a real verifier or stronger internal
state signal.

## Broad100 Public-Test Verifier Policy

Evidence: `experiments/175_public_test_verifier_policy.py`,
`results/controlled/broad100_public_test_verifier_policy/PUBLIC_TEST_VERIFIER_POLICY_MEMO.md`,
`results/controlled/broad100_public_test_verifier_policy/table_public_test_verifier_policy_selected.csv`,
`results/controlled/broad100_public_test_verifier_policy/table_public_test_verifier_policy_all.csv`,
`results/controlled/broad100_public_test_verifier_policy/table_public_test_verifier_policy_query_choices.csv`,
and `results/controlled/broad100_public_test_verifier_policy/table_public_test_verifier_code_summary.csv`.

The public-test verifier policy is the first task-specific verifier branch in
the deployed-action search. It makes no new model calls. For HumanEval and
MBPP, it uses public prompt tests/pass status as route-time evidence for local
code actions, then falls back to the existing tool-aware 171 gate or constrained
risk thresholds on non-code tasks.

Held-out code verifier coverage:

- HumanEval: any local public-test pass rate `0.90`, mean passing-local count
  `3.65`;
- MBPP: any local public-test pass rate `0.80`, mean passing-local count
  `2.45`.

The best validation-selected deployable row,
`code_public_test_else_train_benchmark_prior`, reaches held-out utility
`0.7132`, quality `0.8081`, oracle-utility ratio `0.8428`, frontier-call rate
`0.6744`, and strong-or-frontier rate `0.7965`. The best held-out diagnostic
threshold row reaches utility `0.7261` and quality `0.8140`, but it is
test-picked.

Against the full held-out cost-aware oracle (`0.8463` utility, `0.8721`
quality), this still fails the target thresholds (`0.8209` utility and `0.8421`
quality). Paper interpretation: task-specific verification helps concrete
action identity and is the right kind of evidence, but code is too small a
slice of broad100. The dominant remaining gap is in GPQA, MMLUPro, AIME, and
exact math, where analogous verifier/checker evidence is not yet strong enough.

## Broad100 Activation-Anomaly Threshold Policy

Evidence: `experiments/176_activation_anomaly_threshold_policy.py`,
`results/controlled/broad100_activation_anomaly_threshold_policy/ACTIVATION_ANOMALY_POLICY_MEMO.md`,
`results/controlled/broad100_activation_anomaly_threshold_policy/table_activation_anomaly_features.csv`,
`results/controlled/broad100_activation_anomaly_threshold_policy/table_activation_anomaly_policy_selected.csv`,
`results/controlled/broad100_activation_anomaly_threshold_policy/table_activation_anomaly_policy_all.csv`,
and `results/controlled/broad100_activation_anomaly_threshold_policy/table_activation_anomaly_policy_query_choices.csv`.

The activation-anomaly policy is a cached-only, threshold-only observability
probe. It reuses the Qwen3-4B prefill activation cache and computes train-fitted
global anomaly, benchmark-centroid anomaly, and activation nearest-neighbor
risk summaries. It makes no new model calls.

Scope is limited to the activation-covered stress subset:

- GPQA, MATH500, MMLUPro;
- `180` train queries, `60` validation queries, `60` held-out test queries.

On this subset, the target table has a large oracle:

- held-out best-local utility `0.6333`;
- held-out always-large utility `0.8244`;
- held-out local-vs-large/full cost-aware oracle utility `0.9145`;
- held-out oracle quality `0.9667`.

The validation-selected local-vs-large activation threshold,
`activation_lvlarge_signal_act_last_knn_delta_large_k5_thr0.1974`, reaches
held-out utility `0.7770`, quality `0.8333`, oracle-utility ratio `0.8497`,
frontier-call rate `0.3000`, and strong-or-frontier rate `0.6500`. The
validation-selected deployed-action activation threshold,
`activation_deployed_signal_act_last_knn_delta_large_k5_thr0.1833`, is worse:
held-out utility `0.5816`, quality `0.7333`, and oracle-utility ratio `0.6359`.
The best held-out diagnostic activation row reaches utility `0.8248` and
quality `0.9000`, roughly matching always-large, but it is test-picked.

Paper interpretation: cheap prefill activation anomaly/nearest-neighbor
summaries do not solve the observability problem. They can produce weak
diagnostic separation on a hard subset, but validation selection overfits and
the deployed concrete-action policy collapses. This supports the current
action-identity bottleneck story: the missing signal is not just "is this
query internally anomalous?", but "which candidate action/answer is actually
correct enough to justify its cost?"

## Broad100 Candidate Correctness Ranker Policy

Evidence: `experiments/177_candidate_correctness_ranker_policy.py`,
`results/controlled/broad100_candidate_correctness_ranker_policy/CANDIDATE_CORRECTNESS_RANKER_MEMO.md`,
`results/controlled/broad100_candidate_correctness_ranker_policy/table_candidate_ranker_cv.csv`,
`results/controlled/broad100_candidate_correctness_ranker_policy/table_candidate_ranker_policy_selected.csv`,
`results/controlled/broad100_candidate_correctness_ranker_policy/table_candidate_ranker_policy_all.csv`,
and `results/controlled/broad100_candidate_correctness_ranker_policy/table_candidate_ranker_query_choices.csv`.

This cached experiment trains small candidate-level correctness regressors on
train rows only, with train group-CV selecting the main configuration. It is
designed to attack the concrete action identity gap after local-vs-large
diagnostics became strong. It makes no GPT, Gemini, Claude, vLLM, or local
model calls.

The train-CV selected policy is `hgb_l2_all_rank_pen0.25`. On held-out test it
reaches selected-solver utility `0.7523`, quality `0.8198`, oracle-utility
ratio `0.8890`, frontier-call rate `0.5000`, and strong-or-frontier rate
`0.6977`. However, all-rank mode ranks over frontier candidate answers too.
After charging the cached candidate-generation norm cost (`1.7646`), utility
with candidate-generation cost is only `0.1347`, so this row is not a
deployable cost claim.

The best held-out practical/gated diagnostic row is
`hgb_l1_gate_rank_localplus_pen0.25`: utility `0.7529`, quality `0.8140`,
oracle-utility ratio `0.8896`, frontier-call rate `0.4012`, and
strong-or-frontier rate `0.7151`. It improves over the earlier deployed-action
bridge and public code-test verifier, but remains below the broad100 target
thresholds (`0.8209` utility and `0.8421` quality).

Paper interpretation: supervised candidate correctness is a partial positive
for the action-identity bottleneck, but cheap cached features are not enough.
The all-rank version is cost-infeasible if candidate generation is counted, and
the practical gated version still needs stronger verifier/checker evidence for
GPQA, MMLUPro, AIME, and exact math.

## Broad100 Answer-Group Verifier Policy

Evidence: `experiments/178_answer_group_verifier_policy.py`,
`results/controlled/broad100_answer_group_verifier_policy/ANSWER_GROUP_VERIFIER_POLICY_MEMO.md`,
`results/controlled/broad100_answer_group_verifier_policy/table_answer_group_policy_selected.csv`,
`results/controlled/broad100_answer_group_verifier_policy/table_answer_group_policy_all.csv`,
`results/controlled/broad100_answer_group_verifier_policy/table_answer_group_query_choices.csv`,
and `results/controlled/broad100_answer_group_verifier_policy/table_answer_group_reliability.csv`.

This cached experiment calibrates train-only local answer-group reliability by
benchmark, support count, and strong-local signature, then selects thresholds
on validation. It makes no GPT, Gemini, Claude, vLLM, or local model calls.

The validation-selected frontier-<=0.40 policy,
`answer_group_bench_support_strong_w0_fboverall_thr0.6_localnone_frontiernone`,
reaches held-out utility `0.7123`, quality `0.7907`, oracle-utility ratio
`0.8417`, frontier-call rate `0.3837`, and strong-or-frontier rate `0.5872`.
The validation-selected frontier-<=0.45 policy,
`answer_group_bench_support_strong_w0_fboverall_thr0.65_localnone_frontiernone`,
reaches held-out utility `0.7168`, quality `0.7965`, oracle-utility ratio
`0.8471`, frontier-call rate `0.4012`, and strong-or-frontier rate `0.5930`.
The best held-out diagnostic row reaches utility `0.7198` and quality
`0.8081`, but it is test-picked.

Paper interpretation: answer-group reliability is weaker than the candidate
correctness ranker and remains far below the broad100 targets (`0.8209`
utility and `0.8421` quality). This closes shallow local answer agreement as a
standalone verifier. The next branch should use stronger task-specific checking
for GPQA, MMLUPro, AIME, and exact math.

## Broad100 Cached Adjudicator Blend Policy

Evidence: `experiments/179_cached_adjudicator_blend_policy.py`,
`results/controlled/broad100_cached_adjudicator_blend_policy/CACHED_ADJUDICATOR_BLEND_POLICY_MEMO.md`,
`results/controlled/broad100_cached_adjudicator_blend_policy/table_cached_adjudicator_blend_selected.csv`,
`results/controlled/broad100_cached_adjudicator_blend_policy/table_cached_adjudicator_blend_all.csv`,
and `results/controlled/broad100_cached_adjudicator_blend_policy/table_cached_adjudicator_blend_query_choices.csv`.

This cached experiment overlays four existing answer-adjudicator tables on the
practical candidate-ranker policies. It makes no GPT, Gemini, Claude, vLLM, or
local model calls. The result table reports selected-solver utility and
route-cost-charged utility, where route cost is the cached adjudicator cost
normalized by the mean GPT solver cost.

The validation-selected frontier-<=0.40 solver-utility row is
`hgb_l1_gate_rank_localplus_pen0.35_override_gpt_local_thr0.75_benchgpqa-mmlupro-math500-gsm8k`.
It reaches validation utility `0.8219`, but on held-out test reaches only
utility `0.7190`, quality `0.7558`, oracle-utility ratio `0.8496`,
frontier-call rate `0.2326`, and route-cost-charged utility `0.6597`.

Selecting by route-cost-charged validation utility instead chooses
`hgb_l1_gate_rank_localplus_pen0.35_override_gemini_frontier_thr0.5_benchgpqa`.
On held-out test it reaches utility `0.7414`, route-cost-charged utility
`0.7386`, quality `0.8023`, oracle-utility ratio `0.8761`, frontier-call rate
`0.3953`, and strong-or-frontier rate `0.7151`.

The best held-out diagnostic row is
`hgb_l1_gate_rank_localplus_pen0.25_override_gemini_frontier_thr0_benchmmlupro`:
utility `0.7598`, route-cost-charged utility `0.7522`, quality `0.8140`,
oracle-utility ratio `0.8978`, frontier-call rate `0.3430`, and
strong-or-frontier rate `0.7035`. This row is test-picked and is only a
diagnostic.

Paper interpretation: cached generic adjudication gives a tiny diagnostic gain
over the best practical candidate-ranker row (`0.7529` to `0.7598` selected
solver utility), but it still misses the broad100 target thresholds (`0.8209`
utility and `0.8421` quality), and route-cost accounting erases most of the
gain. This supports the stronger thesis that RouteCode needs observable,
task-specific checker evidence for action identity rather than another generic
answer-agreement or adjudication threshold.

## Broad100 Benchmark Policy Portfolio

Evidence: `experiments/180_benchmark_policy_portfolio.py`,
`results/controlled/broad100_benchmark_policy_portfolio/BENCHMARK_POLICY_PORTFOLIO_MEMO.md`,
`results/controlled/broad100_benchmark_policy_portfolio/table_benchmark_policy_portfolio_selected.csv`,
`results/controlled/broad100_benchmark_policy_portfolio/table_benchmark_policy_portfolio_all.csv`,
`results/controlled/broad100_benchmark_policy_portfolio/table_benchmark_policy_library_eval.csv`,
and `results/controlled/broad100_benchmark_policy_portfolio/table_benchmark_policy_portfolio_maps.csv`.

This cached experiment recomputes validation/test query-level choices for the
strongest existing policy families and selects one policy per benchmark on
validation. It makes no GPT, Gemini, Claude, vLLM, or local model calls. The
library contains `821` candidate policy methods and `282424` query-choice rows.
Validation-selected portfolios exclude diagnostic policy families; test-picked
portfolios are reported only as full-library ceiling diagnostics.

The best validation-selected route-cost objective is
`benchmark_portfolio_val_mean_utility_with_route_cost_frontiercap1`. On
held-out test it reaches utility `0.7320`, route-cost-charged utility `0.7292`,
quality `0.7849`, oracle-utility ratio `0.8650`, frontier-call rate `0.3663`,
and strong-or-frontier rate `0.7093`.

The validation-selected raw-utility portfolio,
`benchmark_portfolio_val_mean_utility_frontiercap0.45`, overfits validation:
held-out utility `0.7094`, route-cost-charged utility `0.6247`, quality
`0.7326`, and oracle-utility ratio `0.8383`.

The test-picked full-library diagnostic ceiling reaches utility `0.7823`,
route-cost-charged utility `0.7676`, quality `0.8314`, oracle-utility ratio
`0.9244`, and frontier-call rate `0.2616`. This is still below the broad100
targets (`0.8209` utility and `0.8421` quality).

Paper interpretation: benchmark-specific composition of existing policies is
not enough. Even after allowing a test-picked benchmark portfolio over the
current library, the system remains below the target. The remaining gap
requires new observable evidence, likely task-specific checkers/verifiers for
GPQA, MMLUPro, AIME, and exact math, rather than more recombination of current
thresholds, rankers, and generic adjudicator overrides.

## Broad100 Task-Specific Verifier Action

Evidence: `experiments/181_task_specific_verifier_action.py`,
`results/controlled/broad100_task_specific_verifier_action/TASK_SPECIFIC_VERIFIER_ACTION_MEMO.md`,
`results/controlled/broad100_task_specific_verifier_action_gpt_mcq_512/TASK_SPECIFIC_VERIFIER_ACTION_MEMO.md`,
`results/controlled/broad100_task_specific_verifier_action_gpt_mcq_512/table_task_specific_verifier_outputs.csv`,
and `results/controlled/broad100_task_specific_verifier_action_gpt_mcq_512/table_task_specific_verifier_policy_selected.csv`.

This branch adds a real provider probe/action rather than recombining cached
route policies. The verifier sees the task plus candidate answers, returns its
own exact answer and confidence, and is scored as a possible route action.
Probe-cost utility charges verifier calls when the verifier is used as a gate
but not selected as the final answer.

The Gemini Flash run attempted `184` hard-slice verifier rows across
GPQA/MMLUPro/AIME/LiveMathBench/MATH500 validation/test, but every request
returned HTTP `429`. It produced no valid Gemini verifier rows and no recorded
valid Gemini spend.

The GPT-5.5 MCQ run used `512` max output tokens on GPQA and MMLUPro
validation/test. It produced `80` valid verifier rows with final-run cost
`$1.0940`. Including the GPT verifier smoke runs, recorded valid GPT verifier
spend for this branch is `$1.7712`, below the `$15` cap.

Verifier quality is benchmark-dependent:

- GPQA validation `0.45`, GPQA test `0.20`.
- MMLUPro validation `0.75`, MMLUPro test `0.80`.

Routing result:

- Validation-selected raw utility policy `task_verifier_conf_ge_0.95`:
  held-out quality `0.8198`, utility `0.7323`, probe-cost-charged utility
  `0.6088`, oracle-utility ratio `0.8637`, frontier-call rate `0.4302`.
- Validation-selected probe-cost policy `task_verifier_conf_ge_0.85`:
  held-out quality `0.8256`, utility `0.7201`, probe-cost-charged utility
  `0.6183`, oracle-utility ratio `0.8494`, frontier-call rate `0.4360`.

Paper interpretation: task-specific verifier evidence is directionally right,
especially for MMLUPro, but this GPT verifier is too expensive and too weak on
GPQA. It does not close the broad100 targets. The next verifier should either
be much more selective and abstention-aware, or should target MMLUPro-like
slices separately while using a different signal for GPQA.

## Broad100 Cached Verifier Supported-Action Policy

Evidence: `experiments/182_cached_verifier_support_policy.py`,
`results/controlled/broad100_cached_verifier_support_policy/CACHED_VERIFIER_SUPPORT_POLICY_MEMO.md`,
`results/controlled/broad100_cached_verifier_support_policy/table_cached_verifier_support_policy_selected.csv`,
`results/controlled/broad100_cached_verifier_support_policy/table_cached_verifier_support_policy_all.csv`,
and `results/controlled/broad100_cached_verifier_support_policy/table_cached_verifier_support_diagnostics.csv`.

This cached follow-up reuses the GPT-5.5 task-verifier rows from Experiment
181 and makes no new provider, vLLM, or local model calls. Instead of treating
the verifier answer as a final routed action, it uses the verifier's
`supported_model` field as evidence for selecting an existing cached candidate
action. Probe-cost utility charges the GPT verifier call whenever the policy
would need to call it.

The supported-action signal is benchmark-asymmetric:

- Held-out GPQA: valid support rate `0.20`, verifier quality `0.20`.
- Held-out MMLUPro: valid support rate `0.75`, verifier quality `0.80`.

Validation selects
`verifier_supported_support_local_only_thr0_benchmmlupro`. On held-out test it
reaches quality `0.8140`, selected-action utility `0.7569`, probe-cost-charged
utility `0.6960`, oracle-utility ratio `0.8944`, and frontier-call rate
`0.3663`. The candidate-ranker reference is quality `0.8140`, utility
`0.7529`, and no probe cost.

The best held-out non-oracle raw rows over GPQA+MMLUPro reach utility `0.7635`
and quality `0.8198`, but they are test-picked and fall to probe-cost-charged
utility `0.6076`. Diagnostic oracle rows between the base action and supported
action reach the same `0.7635` raw utility, so the support field itself has
only limited headroom over the current candidate-ranker base.

Paper interpretation: the GPT verifier contains real action-identity evidence,
especially for MMLUPro, but the gain is too small to pay for a GPT verifier
probe. This supports a sharper RouteCode diagnosis: the missing signal is not
generic answer agreement or generic LLM adjudication; it must be a much cheaper
or more targeted checker that exposes when an existing candidate action should
replace the base route.

## Broad100 Local-Safe Gain Gate

Evidence: `experiments/183_local_safe_gain_gate.py`,
`results/controlled/broad100_local_safe_gain_gate/LOCAL_SAFE_GAIN_GATE_MEMO.md`,
`results/controlled/broad100_local_safe_gain_gate/table_local_safe_gain_policy_selected.csv`,
`results/controlled/broad100_local_safe_gain_gate/table_local_safe_gain_policy_all.csv`,
and `results/controlled/broad100_local_safe_gain_gate/table_local_safe_gain_features.csv`.

This cached experiment attacks the over-escalation error mode. It trains a
train-only gain predictor for replacing the practical candidate-ranker action
with a local consensus action when cached local answers suggest the expensive
route is unnecessary. It makes no GPT, Gemini, Claude, vLLM, or local model
calls during the experiment; it reuses cached local outputs as probe evidence.

Validation selects `pred_rf_thr-0.0288`. On held-out test:

- quality `0.8140`;
- utility `0.7625`;
- oracle-utility ratio `0.9011`;
- frontier-call rate `0.3198`;
- strong-or-frontier rate `0.6221`;
- override rate `0.1279`.

The candidate-ranker reference is quality `0.8140`, utility `0.7529`,
oracle-utility ratio `0.8896`, and frontier-call rate `0.4012`. Thus the
local-safe gain gate improves utility by `0.0097` while preserving quality and
reducing frontier calls.

The best held-out non-oracle diagnostic row reaches utility `0.7724`, quality
`0.8198`, oracle-utility ratio `0.9127`, and frontier-call rate `0.2674`.
The diagnostic oracle between the base action and local consensus reaches
utility `0.7936`, quality `0.8314`, oracle-utility ratio `0.9378`, and
frontier-call rate `0.1744`.

Paper interpretation: cheap local consensus is useful for suppressing
over-escalation, and this is the best validation-selected cached probe policy
in this branch so far. However, even the base-vs-local-consensus oracle remains
below the broad100 targets (`0.8209` utility and `0.8421` quality). Local
agreement alone cannot solve the observability problem; the next method needs
stronger task-specific correctness evidence for GPQA, MMLUPro, and exact math.

## Broad100 Strict MCQ Verifier Policy

Evidence: `experiments/184_strict_mcq_verifier_policy.py`,
`results/controlled/broad100_strict_mcq_verifier_policy/STRICT_MCQ_VERIFIER_POLICY_MEMO.md`,
`results/controlled/broad100_strict_mcq_verifier_policy/table_strict_mcq_verifier_policy_selected.csv`,
`results/controlled/broad100_strict_mcq_verifier_policy/table_strict_mcq_verifier_policy_all.csv`,
and `results/controlled/broad100_strict_mcq_verifier_policy/table_strict_mcq_verifier_outputs.csv`.

This branch reruns a narrower GPT-5.5 multiple-choice verifier on GPQA and
MMLUPro with answer-only JSON, `reasoning.effort=none` where supported, and
`128` max output tokens. It makes no Claude, Gemini, vLLM, or local model
calls. The run evaluates `80` verifier rows with recorded GPT spend `$0.2351`.

The stricter prompt fixes the mechanics of the earlier GPT verifier: incomplete
rate is `0.0000` on GPQA and MMLUPro validation and test. Verifier answer
quality is:

- GPQA validation `0.70`, GPQA test `0.50`.
- MMLUPro validation `0.50`, MMLUPro test `0.35`.

Validation selects `strict_verifier_support_thr0.85_benchmmlupro`. On held-out
test it reaches quality `0.8140`, selected-action utility `0.7553`,
probe-cost-charged utility `0.7395`, oracle-utility ratio `0.8868`, and
frontier-call rate `0.3663`. The candidate-ranker reference in the same action
table reaches quality `0.8140`, utility `0.7529`, and frontier-call rate
`0.4012`.

The best held-out diagnostic strict-support rows reach utility `0.7607`,
quality `0.8140`, probe-cost utility `0.7449`, oracle-utility ratio `0.8932`,
and frontier-call rate `0.3430`, but these rows are test-picked and should not
be treated as selected method performance.

Paper interpretation: strict answer-only verifier prompting fixes the
no-visible-output failure and gives a better GPQA checker than Experiment 181,
but it still does not beat the local-safe gain gate from Experiment 183 after
validation selection, and it remains far below the broad100 target. This
falsifies the simple version of "ask GPT for the MCQ answer as a cheap probe":
the useful next signal needs either much cheaper local/task-specific checking
or a checker that directly identifies candidate-action correctness rather than
producing another noisy answer.

## Broad100 Probe Fusion Policy

Evidence: `experiments/185_probe_fusion_policy.py`,
`results/controlled/broad100_probe_fusion_policy/PROBE_FUSION_POLICY_MEMO.md`,
`results/controlled/broad100_probe_fusion_policy/table_probe_fusion_policy_selected.csv`,
`results/controlled/broad100_probe_fusion_policy/table_probe_fusion_policy_all.csv`,
and `results/controlled/broad100_probe_fusion_policy/table_probe_fusion_query_choices.csv`.

This branch is a no-new-call complementarity test. It fuses the local-safe gain
gate from Experiment 183 with cached strict MCQ verifier support from
Experiment 184. The oracle/action set remains the original broad100 action
matrix; strict verifier answers are used only as probe evidence. No GPT,
Gemini, Claude, vLLM, or local model calls are made by this experiment.

Validation selects
`fusion_strict_repair_base_pred_rf_thr-0.0288_strict0.95_benchmmlupro`.
On held-out test:

- quality `0.8140`;
- selected-action utility `0.7627`;
- probe-cost-charged utility `0.7469`;
- oracle-utility ratio `0.9013`;
- frontier-call rate `0.3140`;
- strict probe-call rate `0.1163`.

The no-probe local-safe reference at the same local threshold reaches held-out
utility `0.7625`, so the validation-selected fusion adds only `0.0002` raw
utility and loses utility after charging strict verifier probe cost. The best
test-picked fusion rows reach utility `0.7687` and quality `0.8198`, but they
are not validation-selected; their probe-cost utility is at most `0.7529` for
the MMLUPro-only probe setting.

Paper interpretation: the two partial positives are not sufficiently
complementary. Strict GPT support can repair a few held-out choices, but the
validation-selected fusion overfits and does not beat the no-probe local-safe
branch under the cost-aware objective. Do not spend more on broad GPT support
fusion unless the verifier signal changes substantially.

## Broad100 Qwen32 Verifier-Risk Veto

Evidence: `experiments/186_qwen32_verifier_risk_veto.py`,
`results/controlled/broad100_qwen32_verifier_risk_veto/QWEN32_VERIFIER_RISK_VETO_MEMO.md`,
`results/controlled/broad100_qwen32_verifier_risk_veto/table_qwen32_verifier_risk_policy_selected.csv`,
`results/controlled/broad100_qwen32_verifier_risk_veto/table_qwen32_verifier_risk_policy_all.csv`,
and `results/controlled/broad100_qwen32_verifier_risk_veto/table_qwen32_verifier_risk_probe.csv`.

This branch reuses cached Qwen3-32B-AWQ vLLM answer-verifier outputs from
Experiment 142. It makes no GPT, Gemini, Claude, vLLM, or local model calls.
Because the verifier judged an older base answer, this experiment treats its
verdict as a query-level/local-risk signal rather than as a direct judgment of
the current selected answer.

The cached verifier has a real relation to the old base answer:

- held-out `accept` rows: old-base quality `0.8351`;
- held-out `escalate` rows: old-base quality `0.5405`.

However, the signal does not transfer to the current local-safe/candidate-ranker
policy. Validation selects `qwen32_veto_escalate_pred_ridge_thr-0.2129`.
On held-out test:

- quality `0.7965`;
- utility `0.7466`;
- oracle-utility ratio `0.8823`;
- frontier-call rate `0.3023`;
- verifier-call rate `0.9942`;
- local-veto rate `0.1279`.

This is worse than the candidate-ranker reference utility `0.7529`, and worse
than the no-probe local-safe reference `local_pred_rf_thr-0.0419`, which reaches
utility `0.7653`, quality `0.8140`, and frontier-call rate `0.2965`. The best
held-out Qwen32-verifier policy reaches only utility `0.7593`, so even
test-picked rows do not beat the no-probe local-safe reference.

Paper interpretation: local answer verification can identify bad answers when
it judges the actual answer it was prompted with, but stale verifier judgments
do not transfer as generic risk labels for a different routing policy. The next
verifier branch must judge the current candidate actions/answers directly, or
the result is dominated by policy/action mismatch.

## Broad100 Current-Action vLLM Verifier

Evidence: `experiments/187_current_action_verifier_vllm.py`,
`results/controlled/broad100_current_action_verifier_qwen14b/CURRENT_ACTION_VERIFIER_MEMO.md`,
`results/controlled/broad100_current_action_verifier_qwen14b/table_current_action_oracle_targets.csv`,
`results/controlled/broad100_current_action_verifier_qwen14b/table_current_action_verifier_probe.csv`,
`results/controlled/broad100_current_action_verifier_qwen14b/table_current_action_verifier_policy_selected.csv`,
and `results/controlled/broad100_current_action_verifier_qwen14b/table_current_action_verifier_policy_all.csv`.

This branch directly addresses the Experiment 186 failure mode by judging the
current selected action instead of reusing a stale verifier. It serves
`Qwen/Qwen3-14B-AWQ` through local vLLM and makes no GPT, Gemini, or Claude
calls. The run collected `344/344` successful validation/test judgments over
the broad100 local-safe action set. It also writes the compact SLM/LLM oracle
target table with `best_local_action`, `best_large_action`, `local_utility`,
`large_utility`, `delta_large`, and `need_large`.

Validation selects `current_verifier_switch_conf0.85_pred_rf_thr-0.0288`.
On held-out test:

- quality `0.8198`;
- utility `0.7678`;
- bootstrap utility CI `[0.7052, 0.8230]`;
- oracle-utility ratio `0.9073`;
- frontier-call rate `0.3256`;
- strong-or-frontier rate `0.6279`;
- verifier-call rate `1.0000`;
- switch rate `0.0058`;
- override rate `0.1337`.

The local-safe reference reaches utility `0.7625` and quality `0.8140`, so the
current-action verifier gives a small raw held-out improvement. It remains far
below the broad100 oracle (`0.8463` utility, `0.8721` quality), and does not
meet the Phase 3 target of at least 95% oracle utility or within three quality
points of oracle.

Paper interpretation: current-action verification is slightly more useful than
stale answer verification, but it is still not the missing RouteCode signal.
The prompt sees cached candidate answer text, including frontier candidates, so
this result is best framed as post-candidate action verification, not cheap
pre-routing. The next promising branch is a narrower checker for GPQA,
MMLUPro, AIME, and exact math that can verify current candidates without
requiring all frontier answers to be generated first.

## Broad100 Current-Verifier Benchmark Composition

Evidence: `experiments/188_current_verifier_benchmark_policy.py`,
`results/controlled/broad100_current_verifier_benchmark_policy/CURRENT_VERIFIER_BENCHMARK_POLICY_MEMO.md`,
`results/controlled/broad100_current_verifier_benchmark_policy/table_current_verifier_benchmark_policy_selected.csv`,
`results/controlled/broad100_current_verifier_benchmark_policy/table_current_verifier_benchmark_policy_all.csv`,
and `results/controlled/broad100_current_verifier_benchmark_policy/table_current_verifier_benchmark_policy_map.csv`.

This no-new-call follow-up asks whether Experiment 187 failed mainly because a
single global verifier threshold was too blunt. It reconstructs cached
Experiment 187 validation/test choices and selects one verifier policy per
benchmark using validation only.

Validation selects `benchmark_best_eps0_fallback_global`. On held-out test:

- quality `0.8023`;
- utility `0.7495`;
- oracle-utility ratio `0.8856`;
- frontier-call rate `0.3488`;
- strong-or-frontier rate `0.6453`.

This is worse than the global Experiment 187 current-action verifier
(`0.7678` utility, `0.8198` quality) and worse than the local-safe reference
(`0.7625` utility, `0.8140` quality). The best held-out diagnostic row in the
188 table remains the global Experiment 187 verifier, not a benchmark-composed
policy.

Paper interpretation: benchmark-level recomposition over the same
current-verifier outputs overfits validation and does not close the broad100
target gap. This strengthens the conclusion that the missing ingredient is not
another threshold map over existing probe signals; it is a new checker signal
for the dominant residual slices, especially GPQA, MMLUPro, and exact math.

## Broad100 Targeted Residual Repair

Evidence: `experiments/189_targeted_residual_repair_policy.py`,
`results/controlled/broad100_targeted_residual_repair_policy/TARGETED_RESIDUAL_REPAIR_POLICY_MEMO.md`,
`results/controlled/broad100_targeted_residual_repair_policy/table_targeted_residual_repair_residuals.csv`,
`results/controlled/broad100_targeted_residual_repair_policy/table_targeted_residual_repair_policy_selected.csv`,
and `results/controlled/broad100_targeted_residual_repair_policy/table_targeted_residual_repair_policy_all.csv`.

This no-new-call branch applies simple residual action-repair rules on top of
Experiment 187's current-action verifier policy. Rules are generated from
validation residuals only. Validation residual mass is largest on GPQA
(`4.8987` total regret), BBH (`3.8255`), GSM8K (`3.0999`), and MMLUPro
(`2.6998`).

The pure validation-best rule reaches validation utility `0.8039` and quality
`0.8488`, but it has no held-out effect and stays at test utility `0.7678`.
The validation residual-coverage selector chooses
`scopegpqa+bbh+gsm8k+mmlupro_selected_qwen32_qwen3-14b-awq-local_none`.
On held-out test:

- quality `0.8256`;
- utility `0.7736`;
- bootstrap utility CI `[0.7139, 0.8292]`;
- oracle-utility ratio `0.9141`;
- frontier-call rate `0.3256`;
- changed-rate `0.0640`.

The best held-out diagnostic rule reaches utility `0.7794`, quality `0.8314`,
and oracle-utility ratio `0.9210`, but it is test-picked.

Paper interpretation: validation residuals support a small action-identity
repair, mainly replacing some Qwen32 choices with Qwen14 on high-residual
slices. This is a partial positive and the current best no-new-call selected
branch in the broad100 probe search, but it remains below the target
(`0.8463` oracle utility and `0.8721` oracle quality). The remaining gap
requires a stronger checker for current-candidate correctness rather than more
deterministic repair rules.

## Broad100 Variable-Option MCQ Verifier

Evidence: `experiments/190_variable_option_mcq_verifier_policy.py`,
`results/controlled/broad100_variable_option_mcq_verifier_policy/VARIABLE_OPTION_MCQ_VERIFIER_POLICY_MEMO.md`,
`results/controlled/broad100_variable_option_mcq_verifier_policy/table_variable_option_mcq_verifier_outputs.csv`,
`results/controlled/broad100_variable_option_mcq_verifier_policy/table_variable_option_mcq_verifier_policy_selected.csv`,
and `results/controlled/broad100_variable_option_mcq_verifier_policy/table_variable_option_mcq_verifier_policy_all.csv`.

This branch fixes a mechanical limitation in the strict MCQ verifier from
Experiment 184: MMLU-Pro rows have options `A` through `J`, but the old verifier
prompt/parser constrained answers to `A|B|C|D`. The new verifier accepts
`A-J` and reruns GPT-5.5 on GPQA and MMLU-Pro validation/test. It makes no
Claude or Gemini calls and records `$0.2516` GPT verifier spend for `80` rows.

Verifier quality:

- GPQA validation `0.8000`, GPQA test `0.4500`.
- MMLU-Pro validation `0.9000`, MMLU-Pro test `0.9000`.

Validation selects `strict_verifier_support_thr0.5_benchgpqa-mmlupro`.
On held-out test:

- quality `0.7907`;
- utility `0.7476`;
- probe-cost utility `0.7121`;
- oracle-utility ratio `0.8779`;
- frontier-call rate `0.2965`;
- probe-call rate `0.2326`.

The best held-out diagnostic support row reaches utility `0.7730` and quality
`0.8256`, still below Experiment 189's selected utility `0.7736`. A no-call
fusion check suggests MMLU-Pro-only support can reach about `0.7833` raw
held-out utility when test-picked, but validation selection and GPT probe cost
prevent a target-level claim.

Paper interpretation: variable-option parsing is a real verifier engineering
fix and makes GPT-5.5 highly accurate on this MMLU-Pro slice. It still does not
solve broad100 routing because GPQA remains unstable and route-time GPT verifier
cost erases much of the useful support signal. The next checker should be
cheaper or more targeted than broad GPT MCQ support.

## Broad100 Variable-Verifier Residual Fusion

Evidence: `experiments/191_variable_verifier_residual_fusion.py`,
`results/controlled/broad100_variable_verifier_residual_fusion/VARIABLE_VERIFIER_RESIDUAL_FUSION_MEMO.md`,
`results/controlled/broad100_variable_verifier_residual_fusion/table_variable_verifier_fusion_policy_selected.csv`,
`results/controlled/broad100_variable_verifier_residual_fusion/table_variable_verifier_fusion_policy_all.csv`,
and `results/controlled/broad100_variable_verifier_residual_fusion/table_variable_verifier_fusion_verifier_diagnostics.csv`.

This no-new-call branch fuses Experiment 190 variable-option verifier support
with the Experiment 189 residual-repair policy. It reports both raw
selected-action utility and utility after charging the GPT verifier as a
route-time probe.

Validation-selected raw utility overfits when GPQA is included:

- selected raw policy: `scopegpqa+mmlupro_thr0_always`;
- validation utility `0.8184`;
- held-out utility `0.7570`;
- held-out probe-cost utility `0.7214`.

A reliability selector that only activates benchmarks with validation verifier
quality at least `0.85` selects MMLU-Pro support:

- held-out quality `0.8256`;
- held-out raw utility `0.7765`;
- held-out probe-cost utility `0.7592`;
- oracle-utility ratio `0.9176`;
- frontier-call rate `0.2965`;
- probe-call rate `0.1163`.

The best held-out diagnostic fusion row reaches raw utility `0.7833` and
quality `0.8314`, but only `0.7660` after probe cost.

Paper interpretation: GPT-5.5 variable-option support contains useful
MMLU-Pro action-identity evidence, but treating it as an extra paid probe is
not cost-effective. The strict cost-aware selector correctly keeps Experiment
189's no-probe residual repair. This points to a cheaper checker or probe reuse
as the next requirement.

## Broad100 Gemini Variable-Option Residual Fusion

Evidence: `experiments/192_gemini_variable_option_residual_fusion.py`,
`results/controlled/broad100_gemini_variable_option_residual_fusion/GEMINI_VARIABLE_OPTION_RESIDUAL_FUSION_MEMO.md`,
`results/controlled/broad100_gemini_variable_option_residual_fusion/table_gemini_variable_option_verifier_outputs.csv`,
`results/controlled/broad100_gemini_variable_option_residual_fusion/table_gemini_variable_fusion_policy_selected.csv`,
and `results/controlled/broad100_gemini_variable_option_residual_fusion/table_gemini_variable_fusion_policy_all.csv`.

This branch attempts to replace the GPT-5.5 variable-option MMLU-Pro support
probe with cheaper Gemini 3.5 Flash verifier calls, then fuses that signal
with the Experiment 189 residual-repair policy. It uses no Claude calls.

The run attempted `40` MMLU-Pro validation/test rows:

- estimated uncached Gemini spend `$0.0565`;
- success rows `0/40`;
- all attempted calls returned HTTP 429;
- recorded Gemini spend `$0.0000`;
- verifier quality `nan` on both validation and test.

Because no Gemini verifier rows succeeded, the validation-selected policy falls
back to Experiment 189's base residual repair. On held-out test:

- quality `0.8256`;
- utility `0.7736`;
- probe-cost utility `0.7736`;
- oracle-utility ratio `0.9141`;
- frontier-call rate `0.3256`;
- probe-call rate `0.0000`;
- override rate `0.0000`.

Paper interpretation: this is not a negative result about Gemini verifier
quality. It is a provider-availability/rate-limit blocker in the current
environment. The cheap-verifier direction remains open, but the next
implementation should prefer local vLLM verification or a provider/model with
confirmed quota before treating the signal as evaluated.

## Broad100 Local vLLM Solve-Support Residual Fusion

Evidence: `experiments/193_local_vllm_solve_support_residual_fusion.py`,
`results/controlled/broad100_local_vllm_solve_support_residual_fusion/LOCAL_VLLM_SOLVE_SUPPORT_RESIDUAL_FUSION_MEMO.md`,
`results/controlled/broad100_local_vllm_solve_support_residual_fusion/table_local_vllm_solve_support_policy_selected.csv`,
`results/controlled/broad100_local_vllm_solve_support_residual_fusion_qwen32/LOCAL_VLLM_SOLVE_SUPPORT_RESIDUAL_FUSION_MEMO.md`,
and `results/controlled/broad100_local_vllm_solve_support_residual_fusion_qwen32/table_local_vllm_solve_support_policy_selected.csv`.

This branch tests the same support-verifier idea without provider APIs. A
local vLLM verifier solves independently, then supports a candidate model only
when that candidate's cached answer matches the verifier's answer. The support
signal is fused with Experiment 189 residual repair. No GPT, Gemini, or Claude
calls are made.

Qwen3-14B-AWQ local verifier:

- verifier rows `264/264` successful;
- mean local latency `0.8930s`;
- validation-best policy `scopebbh_thr0_always`;
- validation utility `0.8078`;
- held-out quality `0.7907`;
- held-out utility `0.7378`;
- held-out oracle-utility ratio `0.8718`;
- frontier-call rate `0.3140`;
- probe-call rate `0.1163`.

The Qwen14 validation-best row overfits badly. A reliability-gated selector
that requires validation verifier quality falls back to Experiment 189.

Qwen3-32B-AWQ local verifier:

- verifier rows `263/264` successful;
- mean local latency `1.4168s`;
- validation-best policy is the base Experiment 189 residual repair;
- held-out quality `0.8256`;
- held-out utility `0.7736`;
- held-out oracle-utility ratio `0.9141`;
- frontier-call rate `0.3256`;
- probe-call rate `0.0000`.

Verifier diagnostic quality is still weak on the dominant residual slices:
Qwen14 test quality is GPQA `0.4500` and MMLU-Pro `0.5000`; Qwen32 test
quality is GPQA `0.4211` and MMLU-Pro `0.6000`. Both models are highly
confident despite these error rates.

Paper interpretation: local solve-support verification is feasible and cheap
in remote-dollar terms, but it is not the missing action-identity signal. The
negative result is useful because it shows the gap is not just provider cost;
the checker itself must be more reliable or must use stronger evidence than
plain answer support.

## Broad100 Conservative Support-Abstention Policy

Evidence: `experiments/194_conservative_support_abstention_policy.py` and
`results/controlled/broad100_conservative_support_abstention_policy/CONSERVATIVE_SUPPORT_ABSTENTION_POLICY_MEMO.md`.

This branch is the smallest no-training probe loop over cached artifacts. It
builds the local-vs-large oracle target table with `query_id`, `query_text`,
best local action, best large action, local/large utilities, `delta_large`, and
`need_large`; it also writes cached probe signals for local answer entropy,
SLM-vs-medium disagreement, and Qwen14/Qwen32 solve-support outputs. It makes
no GPT, Gemini, Claude, local generation, or vLLM serving calls.

The validation-selected threshold policy is
`qwen14_bbh_support2_conf0_nonfrontier`. On BBH only, it switches to the
Qwen14-supported non-frontier candidate only when that candidate answer is
shared by at least two cached actions.

Held-out test result on 172 broad100 queries:

- base Experiment 189 utility `0.7736`, quality `0.8256`, oracle-utility ratio
  `0.9141`, frontier-call rate `0.3256`;
- conservative support-abstention utility `0.7799`, quality `0.8314`,
  oracle-utility ratio `0.9216`, frontier-call rate `0.3140`;
- probe-call rate `0.1163`;
- override rate `0.0116`, exactly `2/172` held-out switches;
- held-out cost-aware oracle utility remains `0.8463`.

Paper interpretation: conservative support-abstention is a real small positive
for cheap probe information, but it is not close enough to the oracle to
support the strong Phase 3 claim. The useful observation is that abstention and
answer-support thresholds can remove a few bad API calls and improve utility,
while broad plain support remains too noisy. The next paper-relevant direction
is stronger checker evidence, not another shallow support threshold.

## Broad100 Local Consensus Cost-Suppression Audit

Evidence: `experiments/195_local_consensus_cost_suppression_audit.py`,
`results/controlled/broad100_local_consensus_cost_suppression_audit/LOCAL_CONSENSUS_COST_SUPPRESSION_MEMO.md`,
`results/controlled/broad100_local_consensus_cost_suppression_audit/table_local_consensus_cost_suppression_selected.csv`,
and `results/controlled/broad100_local_consensus_cost_suppression_audit/table_local_consensus_cost_suppression_all.csv`.

This no-call audit asks whether local answer agreement can suppress
unnecessary frontier/API calls after Experiment 194. It reports deployable
local-majority rules separately from a diagnostic post-hoc same-answer upper
bound. The diagnostic branch is not a deployable router because it uses the
selected frontier answer as the anchor.

Validation selects the deployable rule
`local_majority_scopegsm8k_votes2_if_base_frontier_cheapest`. On held-out test:

- Experiment 194 base utility `0.7799`, quality `0.8314`, oracle-utility ratio
  `0.9216`, frontier-call rate `0.3140`;
- deployable local-majority utility `0.7806`, quality `0.8314`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`, probe-call rate
  `0.0349`, changed rate `0.0291`;
- validation-selected diagnostic same-answer utility `0.7898`, quality
  `0.8314`, oracle-utility ratio `0.9333`, frontier-call rate `0.2267`,
  changed rate `0.0872`.

Paper interpretation: local majority agreement alone is too weak to solve the
remaining gap. The diagnostic result is still useful because it isolates a
specific missing signal: a cheap pre-call checker for whether a local answer
would match the remote/frontier answer. This should be treated as an
observability target, not as achieved deployable performance.

## Broad100 Local Exact-Answer Verifier Cost-Suppression

Evidence: `experiments/196_local_exact_answer_verifier_cost_suppression.py`,
`results/controlled/broad100_local_exact_answer_verifier_cost_suppression/LOCAL_EXACT_ANSWER_VERIFIER_COST_SUPPRESSION_MEMO.md`,
`results/controlled/broad100_local_exact_answer_verifier_cost_suppression/table_exact_answer_verifier_policy_selected.csv`,
`results/controlled/broad100_local_exact_answer_verifier_cost_suppression_qwen32/LOCAL_EXACT_ANSWER_VERIFIER_COST_SUPPRESSION_MEMO.md`,
and `results/controlled/broad100_local_exact_answer_verifier_cost_suppression_qwen32/table_exact_answer_verifier_policy_selected.csv`.

This branch tests the deployable pre-call version of the Experiment 195
same-answer diagnostic. For exact-answer frontier-bound rows, it proposes local
substitute candidates from cached local actions, then asks a local vLLM verifier
whether the candidate answer is reliable enough to use before paying for the
frontier/API action. The verifier sees only the query and local candidate
answer, not the frontier answer. No GPT, Gemini, or Claude calls are made.

Qwen3-14B-AWQ exact-answer verifier:

- verifier rows `308/308` successful;
- validation-selected policy
  `exact_verifier_scopemath500_majority3_cheapest_thr0`;
- held-out quality `0.8314`;
- held-out utility `0.7799`;
- held-out oracle-utility ratio `0.9216`;
- frontier-call rate `0.3140`;
- probe-call rate `0.0116`;
- changed rate `0.0000`.

The selected Qwen14 policy makes no held-out switches, so it does not improve
on the Experiment 194 base. The best held-out diagnostic Qwen14 row,
`exact_verifier_scopemath500_qwen14_thr0.3`, reaches only `0.7812` utility,
`0.8314` quality, `0.9231` oracle-utility ratio, and `0.3081` frontier-call
rate.

Qwen3-32B-AWQ exact-answer verifier:

- verifier rows `308/308` successful;
- validation-selected policy `exact_verifier_scopemath500_qwen32_thr0`;
- validation utility `0.8095`;
- held-out quality `0.8023`;
- held-out utility `0.7561`;
- held-out oracle-utility ratio `0.8935`;
- frontier-call rate `0.2674`;
- probe-call rate `0.0814`;
- changed rate `0.0465`.

The selected Qwen32 policy overfits validation and hurts held-out performance.
The best held-out diagnostic Qwen32 majority rows reach only `0.7810` utility,
`0.8314` quality, `0.9229` oracle-utility ratio, and `0.2965` frontier-call
rate.

Paper interpretation: the pre-call equivalence target is real, but a plain
local LLM answer-reliability prompt is not reliable enough. Qwen32 in
particular accepts too many wrong exact-answer candidates despite being the
larger local verifier. The next paper-relevant checker should use stronger
evidence: calculator or symbolic checks for exact math, execution/unit tests
for code, answer-normalization plus proof/evidence constraints, or calibrated
abstention that is validated by benchmark slice before it can override a
frontier/API action.

## Broad100 Target-Gate Concrete Bridge Sweep

Evidence: `experiments/197_target_gate_concrete_bridge_sweep.py`,
`results/controlled/broad100_target_gate_concrete_bridge_sweep/TARGET_GATE_CONCRETE_BRIDGE_SWEEP_MEMO.md`,
`results/controlled/broad100_target_gate_concrete_bridge_sweep/table_target_gate_concrete_bridge_selected.csv`,
and `results/controlled/broad100_target_gate_concrete_bridge_sweep/table_target_gate_concrete_bridge_all.csv`.

This no-call bridge test keeps the strongest existing tool-aware
local-vs-large gate fixed:
`tool_aware_benchmark_composed_eps0.01_recall_then_quality`. It then sweeps
simple concrete local-action and large-action mappings to test whether the
local-vs-large diagnostic result survives when the router must choose an
actual model/action. It makes no GPT, Gemini, Claude, local generation, or
vLLM calls.

Validation selects
`target_gate_local_prior_utility_large_large_consensus_cheapest`. On held-out
test:

- quality `0.6802`;
- utility `0.6464`;
- oracle-utility ratio `0.7639`;
- frontier-call rate `0.2151`;
- large-gate rate `0.6395`;
- target utility threshold `0.8039`, not met;
- target quality threshold `0.8421`, not met.

The best held-out diagnostic rows are the same family and also fail the target
thresholds. This formally separates two claims that were easy to conflate:
the older tool-aware gate can pass the local-vs-large abstraction target, but
simple concrete action selection destroys the gain.

Paper interpretation: the remaining bottleneck is concrete action identity
after deciding that a larger action is useful. The next RouteCode/ProbeRoute
iteration should focus on reliable action choice or evidence-backed answer
adjudication, not another broad escalation gate.

## Broad100 Residual Action-Identity Audit

Evidence: `experiments/198_residual_action_identity_audit.py`,
`results/controlled/broad100_residual_action_identity_audit/RESIDUAL_ACTION_IDENTITY_AUDIT_MEMO.md`,
`results/controlled/broad100_residual_action_identity_audit/table_residual_action_identity_queries.csv`,
`results/controlled/broad100_residual_action_identity_audit/table_residual_by_benchmark.csv`,
`results/controlled/broad100_residual_action_identity_audit/table_action_confusion.csv`,
and `results/controlled/broad100_residual_action_identity_audit/table_evidence_ceilings.csv`.

This no-call audit uses the current validation-selected concrete policy
`local_majority_scopegsm8k_votes2_if_base_frontier_cheapest`, the cached
broad100 action matrix, and the cached early-signal target table. It does not
train a router and does not make provider or vLLM calls. The goal is to explain
which residual errors remain after the first SLM/LLM early-signal and local
consensus pilots.

Held-out test numbers:

- current policy quality `0.8314`;
- current policy utility `0.7806`;
- current policy oracle-utility ratio `0.9225`;
- current policy frontier-call rate `0.2849`;
- query/action oracle quality `0.8721`;
- query/action oracle utility `0.8463`;
- query/action oracle frontier-call rate `0.1395`;
- utility gap to oracle `0.0656`;
- quality gap to oracle `0.0407`.

Residual concentration on held-out test:

- GPQA contributes `5.0348` total residual utility;
- MMLUPro contributes `2.4272`;
- MATH500 contributes `1.6404`;
- BBH contributes `1.0915`;
- LiveMathBench contributes `1.0089`.

Diagnostic ceilings:

- perfect local-answer/oracle-answer equivalence would reach `0.8225` utility,
  `0.8547` quality, oracle-utility ratio `0.9719`, and frontier-call rate
  `0.1686`;
- post-hoc selected-frontier-answer local equivalence would reach `0.7982`
  utility, `0.8314` quality, oracle-utility ratio `0.9433`, and frontier-call
  rate `0.1744`;
- the true query/action oracle remains `0.8463`, so equivalence alone does not
  meet the full target.

Paper interpretation: the early-signal pilot and local-consensus policy expose
useful structure but do not solve broad100. The key residual is concrete
action identity: choosing the correct local/strong/frontier action and trusting
the correct answer. The strongest next route is an evidence-backed checker,
not another shallow confidence threshold: calculator/symbolic evidence for
exact math, execution-style checks for code where applicable, and
evidence-demanding MCQ adjudication with abstention for GPQA and MMLUPro.

## Broad100 Validation Policy-Library Portfolio

Evidence: `experiments/199_validation_policy_library_portfolio.py`,
`results/controlled/broad100_validation_policy_library_portfolio/POLICY_LIBRARY_PORTFOLIO_MEMO.md`,
`results/controlled/broad100_validation_policy_library_portfolio/table_policy_library_portfolio_selected.csv`,
`results/controlled/broad100_validation_policy_library_portfolio/table_policy_library_candidate_summary.csv`,
and `results/controlled/broad100_validation_policy_library_portfolio/table_policy_library_portfolio_query_choices.csv`.

This no-call audit composes existing cached query-choice policies by
benchmark. It excludes oracle, target-best, post-hoc same-answer, diagnostic,
and self-ingested portfolio rows, then chooses policies on validation only.

Held-out results:

- validation-utility portfolio: quality `0.7442`, utility `0.7199`,
  oracle-utility ratio `0.8507`, frontier-call rate `0.2093`;
- train-prior-stabilized portfolio, cap `0.4`: quality `0.7791`, utility
  `0.7431`, oracle-utility ratio `0.8781`, frontier-call rate `0.2442`;
- cleaned test-picked diagnostic portfolio: quality `0.8430`, utility
  `0.7988`, oracle-utility ratio `0.9439`, frontier-call rate `0.2035`;
- current selected policy from the residual audit remains better than the
  validation-selected portfolios: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`.

Paper interpretation: benchmark-specific policy pieces are not sufficient by
themselves. Test-picked composition gets close to the target quality band but
still misses the 95% utility target and is not deployable. Validation
composition overfits the small per-benchmark slices. The next method should
not be another recombination of existing shallow policies; it needs stronger
query/action evidence, especially for GPQA, MMLUPro, AIME, Math500, and
LiveMathBench.

## Broad100 Current-Policy Variable-Verifier Fusion

Evidence: `experiments/200_current_policy_variable_verifier_fusion.py`,
`results/controlled/broad100_current_policy_variable_verifier_fusion/CURRENT_POLICY_VARIABLE_VERIFIER_FUSION_MEMO.md`,
`results/controlled/broad100_current_policy_variable_verifier_fusion/table_current_policy_variable_verifier_selected.csv`,
`results/controlled/broad100_current_policy_variable_verifier_fusion/table_current_policy_variable_verifier_diagnostics.csv`,
and `results/controlled/broad100_current_policy_variable_verifier_fusion/table_current_policy_variable_verifier_query_choices.csv`.

This no-new-call replay adds cached GPT-5.5 variable-option MCQ verifier
support to the current best concrete policy
`local_majority_scopegsm8k_votes2_if_base_frontier_cheapest`. It does not
train a router and does not make provider, local generation, or vLLM calls.

Held-out results:

- current base policy: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`;
- validation-best GPQA+MMLUPro support: quality `0.8081`, utility `0.7753`,
  probe-cost utility `0.7398`, oracle-utility ratio `0.9161`;
- reliability-constrained support, selected from validation verifier quality:
  quality `0.8314`, raw utility `0.7836`, probe-cost utility `0.7663`,
  oracle-utility ratio `0.9259`, frontier-call rate `0.2558`;
- best held-out MMLUPro-only diagnostic: quality `0.8372`, raw utility
  `0.7904`, probe-cost utility `0.7731`, oracle-utility ratio `0.9340`.

Verifier diagnostics:

- GPQA verifier quality: validation `0.8000`, test `0.4500`;
- MMLUPro verifier quality: validation `0.9000`, test `0.9000`.

Paper interpretation: MMLUPro answer support is a real but small concrete
action-identity signal. GPQA support is unstable, and GPT verification is too
expensive as a route-time probe under the current utility objective. This
supports the direction of evidence-backed action verification, but not broad
paid verifier routing.

## Broad100 Benchmark-Agnostic Probe-State RouteCode

Evidence: `experiments/201_benchmark_agnostic_probe_state_routecode.py`,
`results/controlled/broad100_probe_state_routecode/PROBE_STATE_ROUTECODE_MEMO.md`,
`results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv`,
`results/controlled/broad100_probe_state_routecode/table_probe_state_policy_selected.csv`,
`results/controlled/broad100_probe_state_routecode/table_probe_state_benchmark_heldout.csv`,
and `results/controlled/broad100_probe_state_routecode/probe_state_code_cards.md`.

This no-new-call cached experiment implements the Phase 3 benchmark-agnostic
ProbeCode direction:

```text
query + cheap local model behavior -> probe_state -> cost-aware action
```

The probe-state table has `860` queries and `84` numeric probe features from
local answer agreement, small-vs-medium disagreement, self-consistency
entropy/margins, validity/malformed proxies, local output length/latency, and
cached logprob margins when present. The main probe-state method excludes
benchmark ID; benchmark-ID rows are diagnostic only.

Standard held-out results:

- benchmark lookup: quality `0.7791`, utility `0.6652`,
  oracle-utility ratio `0.7860`;
- text-only utility router: quality `0.7093`, utility `0.6189`,
  oracle-utility ratio `0.7314`;
- probe-state KMeans `K=16`: quality `0.7674`, utility `0.6876`,
  oracle-utility ratio `0.8125`, frontier-call rate `0.5174`;
- direct probe utility regressor: quality `0.7500`, utility `0.6733`,
  oracle-utility ratio `0.7956`;
- oracle fixed local-vs-large gate upper bound: utility `0.7207`;
- oracle RouteCode-label upper bound: quality `0.8140`, utility `0.7887`,
  oracle-utility ratio `0.9320`.

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

Paper interpretation: broad probe-state features transfer better than
benchmark lookup and text-only routing, so ProbeCode is a better main Phase 3
direction than benchmark-specific checkers. But it does not close the oracle
gap. The oracle RouteCode label and fixed local-vs-large gate upper bounds are
still substantially higher, so the bottleneck remains observability of compact
utility/action states from cheap local behavior. Do not claim that RouteCode
works broadly from this result; claim only that benchmark-agnostic probe states
are a promising but incomplete observability layer.

## Broad100 Benchmark-Agnostic Family Probe Gate

Evidence: `experiments/202_benchmark_agnostic_family_probe_gate.py`,
`results/controlled/broad100_benchmark_agnostic_family_probe_gate/BENCHMARK_AGNOSTIC_FAMILY_PROBE_GATE_MEMO.md`,
`results/controlled/broad100_benchmark_agnostic_family_probe_gate/table_family_probe_gate_selected.csv`,
`results/controlled/broad100_benchmark_agnostic_family_probe_gate/table_family_probe_gate_all.csv`,
`results/controlled/broad100_benchmark_agnostic_family_probe_gate/table_family_probe_gate_query_choices.csv`,
and `results/controlled/broad100_benchmark_agnostic_family_probe_gate/fig_family_probe_gate_utility.pdf`.

This no-new-call cached experiment tests a benchmark-agnostic ProbeCode
family gate:

```text
query + cheap local behavior -> local-vs-large family -> action
```

It excludes benchmark ID and task-specific checkers. The family-gate rows are
diagnostic because, after predicting local versus large, they use the cached
best concrete action inside that family.

Held-out results:

- full cost-aware oracle: quality `0.8721`, utility `0.8463`,
  frontier-call rate `0.1395`;
- current concrete base: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`;
- validation-selected ridge family gate: quality `0.8488`, utility `0.8140`,
  oracle-utility ratio `0.9618`, frontier-call rate `0.2267`;
- validation-selected probe-state family gate: quality `0.8488`, utility
  `0.8151`, oracle-utility ratio `0.9632`, frontier-call rate `0.2384`;
- validation-selected concrete bridge: quality `0.8023`, utility `0.6902`,
  oracle-utility ratio `0.8156`, frontier-call rate `0.6105`.

Paper interpretation: cheap local behavior is sufficient to observe the coarse
local-vs-large state well enough to pass the Broad100 numeric target in a
family-oracle abstraction. This is the strongest positive evidence for the
ProbeCode direction so far. It is not a solved deployed router: the concrete
action bridge is still worse than the current base, so the remaining bottleneck
is exact model/action identity inside the predicted family.

## Broad100 Current-Base Cached Adjudicator Bridge

Evidence: `experiments/203_current_base_cached_adjudicator_bridge.py`,
`results/controlled/broad100_current_base_cached_adjudicator_bridge/CACHED_ADJUDICATOR_BRIDGE_MEMO.md`,
`results/controlled/broad100_current_base_cached_adjudicator_bridge/table_cached_adjudicator_bridge_selected.csv`,
`results/controlled/broad100_current_base_cached_adjudicator_bridge/table_cached_adjudicator_bridge_all.csv`,
and `results/controlled/broad100_current_base_cached_adjudicator_bridge/table_cached_adjudicator_bridge_query_choices.csv`.

This no-new-call replay tests whether cached broad GPT/Gemini adjudicator
outputs can override the current best concrete Broad100 policy. It is a
diagnostic bridge rather than a main ProbeCode method because some cached
adjudicator prompts included benchmark metadata and adjudication is an
expensive route-time probe.

Held-out results:

- current base: quality `0.8314`, utility `0.7806`, oracle-utility ratio
  `0.9225`, frontier-call rate `0.2849`;
- validation-best raw override
  `gpt_with_frontier_thr0.75_if_adjudicator_local`: validation utility
  `0.8320`, but held-out utility `0.7587` and probe-cost utility `0.5383`;
- probe-cost-aware validation selection: no-adjudicator base;
- best raw held-out adjudicator diagnostic: utility `0.7834`, but probe-cost
  utility `0.4201`.

Paper interpretation: generic paid adjudication has at most a tiny raw residual
signal on the current base and is not cost-effective. Do not spend fresh GPT or
Gemini calls on broad route-time adjudication as the next main bridge. The next
method should focus on cheap local/probe-state evidence for concrete action
identity.

## Broad100 Benchmark-Agnostic Local-Candidate Selector

Evidence: `experiments/204_benchmark_agnostic_local_candidate_selector.py`,
`results/controlled/broad100_benchmark_agnostic_local_candidate_selector/LOCAL_CANDIDATE_SELECTOR_MEMO.md`,
`results/controlled/broad100_benchmark_agnostic_local_candidate_selector/table_local_candidate_selector_selected.csv`,
`results/controlled/broad100_benchmark_agnostic_local_candidate_selector/table_local_candidate_selector_benchmark_heldout.csv`,
and `results/controlled/broad100_benchmark_agnostic_local_candidate_selector/table_local_candidate_selector_query_choices.csv`.

This no-new-call cached experiment tests whether benchmark-agnostic local
behavior can identify the exact cheap/local action and safely override the
current concrete base policy. It uses answer agreement/support, validity,
output length/latency, local model identity, train-only model priors, and the
cached broad probe-state features. It does not use benchmark-specific checkers
or benchmark ID in the main rows.

Held-out results:

- full cost-aware oracle: quality `0.8721`, utility `0.8463`,
  frontier-call rate `0.1395`;
- current base: quality `0.8314`, utility `0.7806`, oracle-utility ratio
  `0.9225`, frontier-call rate `0.2849`;
- diagnostic current base plus all cached local candidates oracle: quality
  `0.8605`, utility `0.8321`, oracle-utility ratio `0.9833`,
  frontier-call rate `0.1279`;
- validation-selected local override
  `candidate_extra_trees_leaf4_override_if_base_frontier_score_thr0.798397`:
  validation utility `0.8093`, but held-out utility `0.7711`, quality
  `0.8198`, oracle-utility ratio `0.9112`, frontier-call rate `0.2616`;
- best validation-selected always-local learned ranker: utility `0.6628`;
- probe-state local KMeans: utility `0.6453`;
- text-only local utility router: utility `0.5465`.

Benchmark-heldout mean test utilities:

- current-base-plus-all-locals oracle `0.8125`;
- current base `0.7632`;
- current base plus learned local selector `0.7536`;
- local action oracle `0.7148`;
- local candidate ranker `0.5759`;
- probe-only local utility router `0.5648`;
- text-only local router `0.4852`;
- probe-state local KMeans `0.4648`;
- benchmark lookup/global-best local `0.4611`.

Paper interpretation: the candidate action set itself is strong; if we could
choose between the current base and cached local candidates post hoc, held-out
utility would be `0.8321`, within `0.0141` of the full oracle. The learned
benchmark-agnostic selector does not yet observe exact local action identity,
and validation-selected overrides hurt held-out test utility. This is strong
evidence that Phase 3 should focus on better cheap reliability/action-identity
signals, while keeping benchmark-specific verifiers as diagnostics or later
plugins rather than the main method.

## Broad100 Current-Base Local Support Verifier

Evidence: `experiments/205_current_base_local_support_verifier.py`,
`results/controlled/broad100_current_base_local_support_verifier/LOCAL_SUPPORT_VERIFIER_MEMO.md`,
`results/controlled/broad100_current_base_local_support_verifier/table_local_support_verifier_probe.csv`,
`results/controlled/broad100_current_base_local_support_verifier/table_local_support_verifier_policy_selected.csv`,
and `results/controlled/broad100_current_base_local_support_verifier/table_local_support_verifier_query_choices.csv`.

This local-vLLM experiment tests a benchmark-agnostic ProbeCode bridge:

```text
query + cheap local candidate answers -> local support state -> cost-aware action
```

The verifier used Qwen3-14B-AWQ through vLLM, omitted benchmark ID, and used no
task-specific checker. It made no GPT, Gemini, or Claude calls. It collected
`344` validation/test verifier rows.

Held-out results:

- current base: quality `0.8314`, utility `0.7806`, oracle-utility ratio
  `0.9225`, frontier-call rate `0.2849`;
- validation-selected local-support downshift
  `downshift_frontier_supported_conf0`: validation utility `0.8085`, but
  held-out quality `0.8023`, utility `0.7571`, oracle-utility ratio `0.8947`,
  frontier-call rate `0.2558`;
- best diagnostic held-out row `downshift_frontier_supported_conf0.95`:
  quality `0.8256`, utility `0.7754`, oracle-utility ratio `0.9163`,
  frontier-call rate `0.2791`.

Paper interpretation: a generic local support verifier can slightly reduce
frontier calls, but the local support state is not calibrated well enough and
over-downshifts from frontier to local. This is a useful negative result for
ProbeCode: local candidate-answer support alone does not solve concrete action
identity, so the next method needs richer uncertainty/calibration evidence or
a decision-aware probe-state learner.

## Broad100 Probe-State Composed YES/NO Policy

Evidence: `experiments/206_probe_state_composed_yesno_policy.py`,
`results/controlled/broad100_probe_state_composed_yesno_policy/PROBE_STATE_COMPOSED_POLICY_MEMO.md`,
`results/controlled/broad100_probe_state_composed_yesno_policy/table_probe_state_composed_policy_selected.csv`,
`results/controlled/broad100_probe_state_composed_yesno_policy/table_probe_state_composed_policy_all.csv`,
and `results/controlled/broad100_probe_state_composed_yesno_policy/probe_state_composed_code_cards.md`.

This cached experiment tests whether broad probe states can replace
per-benchmark lookup for the coarse local-vs-large decision:

```text
query + cheap broad local signals -> probe_state -> cost-aware family action
```

The main rows exclude benchmark ID, benchmark train priors, deterministic tool
availability, and task-specific checkers. No GPT, Gemini, Claude, vLLM, or
local generation calls were made. This is still an abstraction experiment: it
selects between cached best-local and best-large actions, not exact concrete
model identity.

Held-out local-vs-large oracle: quality `0.8721`, utility `0.8463`,
frontier-call rate `0.1395`.

Held-out validation-selected results:

- main no-benchmark/no-tool probe state `main_no_benchmark_no_tool_k2`:
  quality `0.8198`, utility `0.7747`, oracle-utility ratio `0.9154`,
  frontier-call rate `0.2907`;
- diagnostic train-prior row `main_plus_train_benchmark_prior_k16`: quality
  `0.8140`, utility `0.7763`, oracle-utility ratio `0.9174`,
  frontier-call rate `0.2267`;
- diagnostic tool-aware row `main_plus_tool_available_k8`: quality `0.8605`,
  utility `0.8248`, oracle-utility ratio `0.9747`, frontier-call rate
  `0.2500`.

Paper interpretation: the benchmark-agnostic no-tool probe-state method still
misses the Phase 3 target, but the tool-aware diagnostic clears it. This is the
most useful positive control so far: the bottleneck is not the local-vs-large
abstraction itself, but observing a reliable benchmark-agnostic
verifiability/answer-validity signal. Do not present the tool-aware row as the
main method; use it to motivate ProbeCode states that capture generic
verifiability without benchmark-specific checkers.

## Broad100 Learned Verifiability ProbeCode

Evidence: `experiments/207_learned_verifiability_probe_state.py`,
`results/controlled/broad100_learned_verifiability_probe_state/LEARNED_VERIFIABILITY_PROBE_STATE_MEMO.md`,
`results/controlled/broad100_learned_verifiability_probe_state/table_learned_verifiability_policy_selected.csv`,
`experiments/208_learned_verifiability_benchmark_heldout.py`,
and `results/controlled/broad100_learned_verifiability_benchmark_heldout/LEARNED_VERIFIABILITY_BENCHMARK_HELDOUT_MEMO.md`.

This cached experiment turns the tool-aware positive control into a learned
benchmark-agnostic verifiability signal:

```text
query + cheap local behavior -> learned verifiability state -> cost-aware family action
```

The main rows use train-only verifiability labels and block benchmark ID,
domain, metric, train benchmark priors, outcome utility/quality/cost columns,
direct tool flags, and direct tool outputs at validation/test time. No model
calls were made.

Standard held-out local-vs-large results:

- local-vs-large oracle: quality `0.8721`, utility `0.8463`;
- learned global verifiability, validation-selected:
  quality `0.8547`, utility `0.8232`, oracle-utility ratio `0.9727`,
  frontier-call rate `0.2209`;
- learned discrete verifiability state, validation-selected:
  quality `0.8488`, utility `0.8136`, oracle-utility ratio `0.9614`,
  frontier-call rate `0.2384`.

Benchmark-heldout mean results over nine benchmarks:

- learned global verifiability: quality `0.8315`, utility `0.7981`,
  oracle-utility ratio `0.9598`, frontier-call rate `0.2333`;
- learned discrete verifiability state: quality `0.8185`, utility `0.7808`,
  oracle-utility ratio `0.9488`, frontier-call rate `0.2593`;
- direct tool-flag positive control: quality `0.8426`, utility `0.8104`,
  oracle-utility ratio `0.9725`, frontier-call rate `0.2278`;
- reference: quality `0.7593`, utility `0.7298`, oracle-utility ratio
  `0.8308`.

Paper interpretation: learned verifiability is the strongest Phase 3 result so
far. It supports the benchmark-agnostic ProbeCode story better than prior
generic probes because it transfers across held-out benchmarks and nearly
matches the direct tool-flag positive control. The remaining gap is that this
is still a local-vs-large abstraction using cached best-local/best-large
actions; concrete action identity and a more stable discrete state policy are
not solved yet.

## Broad100 Decision-Aware Probe-State RouteCode

Evidence: `experiments/209_decision_aware_probe_state_routecode.py`,
`results/controlled/broad100_decision_aware_probe_state_routecode/DECISION_AWARE_PROBE_STATE_MEMO.md`,
`results/controlled/broad100_decision_aware_probe_state_routecode/table_decision_aware_probe_state_selected.csv`,
and
`results/controlled/broad100_decision_aware_probe_state_routecode/table_decision_aware_probe_state_benchmark_heldout_selected.csv`.

This cached experiment tests whether broad benchmark-agnostic probe states
improve when the discrete states are trained with utility/action supervision
instead of unsupervised KMeans. It uses existing cached Broad100 outputs and
probe features only; no model calls, provider calls, vLLM calls, or
benchmark-specific checkers are used.

Standard held-out results:

- previous plain KMeans probe state from experiment 201: quality `0.7674`,
  utility `0.6876`, oracle-utility ratio `0.8125`;
- decision-aware action-probability state, validation-selected
  `et_actionprob_state_depthnone_leaf8_k32`: quality `0.7733`, utility
  `0.7064`, oracle-utility ratio `0.8348`, frontier-call rate `0.5174`;
- decision-aware utility-direct row, validation-selected
  `et_utility_direct_est100_depthnone_leaf8`: quality `0.7965`, utility
  `0.7049`, oracle-utility ratio `0.8330`, frontier-call rate `0.5698`;
- oracle RouteCode-label upper bound from experiment 201: quality `0.8140`,
  utility `0.7887`, oracle-utility ratio `0.9320`.

Benchmark-heldout mean results:

- decision-aware utility-direct: utility `0.6403`, oracle-utility ratio
  `0.7388`;
- decision-aware utility-state: utility `0.6189`, oracle-utility ratio
  `0.7211`;
- decision-aware action-probability state: utility `0.5913` under validation
  frontier-cap selection and `0.5717` under raw validation-utility selection;
- earlier experiment-201 KMeans probe-state heldout mean: `0.6217`;
- earlier experiment-201 direct probe utility regressor heldout mean: `0.6507`;
- oracle RouteCode-label upper bound heldout mean: `0.7792`.

Paper interpretation: decision-aware state learning gives a modest
standard-split improvement but does not transfer better. This is a negative
result for treating the remaining gap as merely an unsupervised-clustering
problem. The stronger story remains learned verifiability: keep the
benchmark-agnostic verifiability signal, then solve concrete action identity
and transfer-stable state/action mapping.

## Broad100 Concrete Probe-Verifiability Policy

Evidence: `experiments/210_concrete_probe_verifiability_policy.py`,
`results/controlled/broad100_concrete_probe_verifiability_policy/CONCRETE_PROBE_VERIFIABILITY_POLICY_MEMO.md`,
`results/controlled/broad100_concrete_probe_verifiability_policy/table_concrete_probe_verifiability_selected.csv`,
and
`results/controlled/broad100_concrete_probe_verifiability_policy/table_concrete_probe_verifiability_benchmark_heldout_selected.csv`.

This cached experiment tries to bridge the learned-verifiability result into
concrete action selection:

```text
query + cheap local behavior -> learned verifiability/local-candidate scores -> concrete action
```

It uses no provider calls, no vLLM calls, no local generation calls, and no
benchmark-specific verifier calls. Benchmark-heldout rows refit both the
verifiability classifier and local-candidate ranker without the held-out
benchmark.

Standard held-out results:

- current concrete base: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`;
- learned verifiability-to-tool gate selected by validation: unchanged from
  base, utility `0.7806`;
- local-ranker override selected by validation: quality `0.8198`, utility
  `0.7711`, oracle-utility ratio `0.9112`, frontier-call rate `0.2616`;
- combined verifiability plus local-ranker override selected by validation:
  quality `0.8198`, utility `0.7711`, oracle-utility ratio `0.9112`;
- diagnostic current-base-plus-all-locals oracle: quality `0.8605`, utility
  `0.8321`, oracle-utility ratio `0.9833`, frontier-call rate `0.1279`;
- full query oracle: quality `0.8721`, utility `0.8463`.

Benchmark-heldout mean selected test utilities:

- current base `0.7632`;
- learned verifiability-to-tool `0.7632`;
- local-ranker override `0.7591`;
- combined verifiability plus local-ranker override `0.7591`;
- diagnostic current-base-plus-all-locals oracle `0.8125`.

Paper interpretation: the route-action headroom is real, but the deployable
observable state is still insufficient. Query-level verifiability and scalar
local-candidate utility scores do not identify the safe concrete local
substitution. The next credible method needs candidate-answer reliability
states, not more benchmark-specific verifiers.

## Broad100 Conformal Answer-Set Probe Policy

Evidence: `experiments/211_conformal_answer_set_probe_policy.py`,
`results/controlled/broad100_conformal_answer_set_probe_policy/CONFORMAL_ANSWER_SET_PROBE_POLICY_MEMO.md`,
`results/controlled/broad100_conformal_answer_set_probe_policy/table_conformal_answer_set_policy_selected.csv`,
`results/controlled/broad100_conformal_answer_set_probe_policy/table_conformal_answer_set_benchmark_heldout_summary.csv`,
and
`results/controlled/broad100_conformal_answer_set_probe_policy/table_answer_set_probe_states.csv`.

This cached experiment tests broad CP-Router/STEER-style uncertainty probes:

```text
non-tool local answers + self-consistency samples
  -> conformal answer set / confidence-weighted answer support
  -> trust local answer group or fall back to current base
```

It uses no provider calls, no vLLM calls, no local generation calls, and no
benchmark-specific verifier calls. The main answer-set probe excludes
`deterministic_math_tool` as an answer source, calibrates local model
reliability and conformal nonconformity on train only, and selects thresholds on
validation.

Standard held-out results:

- current concrete base: quality `0.8314`, utility `0.7806`,
  oracle-utility ratio `0.9225`, frontier-call rate `0.2849`;
- conformal answer set, validation-selected
  `conformal_answer_set_alpha0.1_set1_fallbackcurrent_base`: quality `0.8314`,
  utility `0.7806`, oracle-utility ratio `0.9225`, frontier-call rate
  `0.2849`;
- confidence-informed self-consistency conformal row, validation-selected
  `cisc_conformal_answer_set_alpha0.1_set1_fallbackcurrent_base`: also
  quality `0.8314`, utility `0.7806`;
- CISC confidence-threshold row, validation-selected
  `cisc_confidence_threshold_conf0.85_set1_fallbackcurrent_base`: also
  quality `0.8314`, utility `0.7806`;
- plain self-consistency-majority threshold, validation-selected
  `self_consistency_majority_threshold_conf0.75_set1_fallbackcurrent_base`:
  quality `0.7384`, utility `0.7020`, oracle-utility ratio `0.8296`.

The selected conformal answer-set row uses the answer set on `23.3%` of test
queries, and the selected CISC confidence row uses it on `19.2%`; the
substitutions do not improve aggregate utility over the current base.

Benchmark-heldout transfer mean results:

- conformal/CISC answer-set rows: mean quality `0.8130`, mean utility
  `0.7632`, oracle-utility ratio `0.9276`, frontier-call rate `0.2796`;
- current base: same mean utility `0.7632`;
- self-consistency-majority threshold: mean utility `0.6855`;
- full query oracle: mean utility `0.8260`.

Paper interpretation: conformal answer-set size and confidence-weighted
self-consistency are useful benchmark-agnostic probe-state features, but they
do not solve the Broad100 concrete action gap. The result is evidence against
plain answer agreement/support as the missing signal. The next credible path is
richer candidate-answer reliability evidence or a learned state/action mapper
that keeps the positive learned-verifiability signal without turning into
benchmark-specific checkers.

## Broad100 Target-Level Method Status

Evidence:
`experiments/212_broad100_target_level_method_status.py`,
`results/controlled/broad100_target_level_method_status/BROAD100_TARGET_LEVEL_METHOD_STATUS.md`,
and
`results/controlled/broad100_target_level_method_status/table_broad100_target_gate_comparison.csv`.

This cached status check answers whether a modification of the current
ProbeCode/RouteCode method can reach the Phase 3 Broad100 target gates.

Main result:

- learned-verifiability state `gb_depth2_thr0.9844_state_k8`, selected on
  validation, reaches held-out quality `0.8488`, utility `0.8136`,
  oracle-utility ratio `0.9614`, and frontier-call rate `0.2384`;
- learned-verifiability global `extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75`,
  selected on validation, reaches held-out quality `0.8547`, utility `0.8232`,
  oracle-utility ratio `0.9727`, and frontier-call rate `0.2209`;
- current base remains quality `0.8314`, utility `0.7806`, oracle-utility
  ratio `0.9225`;
- no-tool/no-benchmark probe state `main_no_benchmark_no_tool_k2` remains below
  target at quality `0.8198`, utility `0.7747`, oracle-utility ratio `0.9154`.

Paper interpretation: the target-level modification is possible, but the
passing Broad100 rows still rely on learned verifiability states whose
state/action maps include `tool_` behavior. This supports the claim that
verifiability is an observable route state, but it is not yet a clean
benchmark-agnostic ProbeCode headline result. Reaching exact oracle level would
require leakage, benchmark-specific verifiers/tools, or a qualitatively stronger
general probe. The next method should generalize this verifiability win into a
candidate-answer reliability state.

## Broad100 Target Method Package

Evidence:
`experiments/213_broad100_target_method_package.py`,
`results/controlled/broad100_target_method_package/BROAD100_TARGET_METHOD_PACKAGE.md`,
`results/controlled/broad100_target_method_package/table_broad100_target_method_main_eval.csv`,
`results/controlled/broad100_target_method_package/table_broad100_target_method_ablation.csv`,
and
`results/controlled/broad100_target_method_package/table_broad100_target_method_action_mix.csv`.

This cached package reconstructs the validation-selected learned-verifiability
policies query-by-query and compares the full action pool against a no-tool
local-pool ablation. It makes no provider calls, no vLLM calls, and no local
generation calls.

Main held-out Broad100 result:

- full cost-aware oracle: quality `0.8721`, utility `0.8463`,
  frontier-call rate `0.1395`;
- selected global learned-verifiability policy
  `extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75`: quality `0.8547`,
  utility `0.8232`, oracle-utility ratio `0.9727`, frontier-call rate
  `0.2209`, primary numeric target passed;
- selected RouteCode state policy `gb_depth2_thr0.9844_state_k8`: quality
  `0.8488`, utility `0.8136`, oracle-utility ratio `0.9614`,
  frontier-call rate `0.2384`, primary numeric target passed;
- same global policy with deterministic-tool local action removed: quality
  `0.7674`, utility `0.7360`, oracle-utility ratio `0.8697`, target failed;
- same state policy with deterministic-tool local action removed: quality
  `0.7616`, utility `0.7264`, oracle-utility ratio `0.8583`, target failed.

Paper interpretation: the method can be made target-level on cached Broad100,
but the positive result depends heavily on the verifiable local action pool.
The honest story is not "generic answer agreement reaches oracle"; it is:
learned broad verifiability states can expose when reliable local/tool actions
are safe and when fallback to strong local/frontier actions is worth the cost.
The no-tool ablation is now direct evidence that a stronger benchmark-agnostic
candidate-answer reliability probe is required before claiming a fully clean
ProbeCode method.

## Broad100 No-Tool Verifiability Repair

Evidence:
`experiments/214_broad100_no_tool_verifiability_repair.py`,
`results/controlled/broad100_no_tool_verifiability_repair/NO_TOOL_VERIFIABILITY_REPAIR_MEMO.md`,
`results/controlled/broad100_no_tool_verifiability_repair/table_no_tool_verifiability_repair_selected.csv`,
and
`results/controlled/broad100_no_tool_verifiability_repair/table_no_tool_verifiability_repair_all.csv`.

This cached experiment tests whether the learned-verifiability signal can still
reach the Broad100 target when deterministic-tool actions are removed by routing
predicted-verifiable states upward to strong/large actions.

Main result:

- best validation-selected repair under frontier cap:
  `logreg_c0.3_thr0.0915_pred_tool_large_else_cap_e0.75`;
- held-out quality `0.8140`, utility `0.7692`, oracle-utility ratio `0.9089`,
  frontier-call rate `0.2791`;
- best test-only diagnostic reaches oracle-utility ratio `0.9110`;
- no repair row meets the 3-point quality plus 95% utility target.

Paper interpretation: the clean no-tool gap is not closed by more aggressive
escalation. Deterministic tools are acting as high-precision local reliability
evidence. To make the method broadly benchmark-agnostic, the next signal needs
to estimate candidate-answer reliability directly rather than only query
verifiability or answer agreement.

## Broad100 Residual Oracle-Gap Repair

Evidence:
`experiments/215_broad100_residual_oracle_gap_repair.py`,
`results/controlled/broad100_residual_oracle_gap_repair/RESIDUAL_ORACLE_GAP_REPAIR_MEMO.md`,
`results/controlled/broad100_residual_oracle_gap_repair/table_residual_oracle_gap_repair_selected.csv`,
and
`results/controlled/broad100_residual_oracle_gap_repair/table_residual_oracle_gap_repair_all.csv`.

This cached experiment tests whether a residual reliability layer can make the
current learned-verifiability target-level method closer to the oracle. It uses
train-only fitted residual models and validation-selected thresholds; no API,
vLLM, or generation calls are made.

Main result:

- base learned-verifiability global policy: held-out quality `0.8547`,
  utility `0.8232`, oracle-utility ratio `0.9727`, frontier-call rate
  `0.2209`;
- conservative base-tethered residual `et_flip_leaf4_thr0.8502_capNone`:
  held-out quality `0.8547`, utility `0.8238`, oracle-utility ratio `0.9735`,
  frontier-call rate `0.1919`;
- aggressive validation-best residual `et_delta_leaf8_thr0.1194`: held-out
  quality `0.8488`, utility `0.8143`, oracle-utility ratio `0.9622`,
  frontier-call rate `0.2035`;
- large-recall-guard residual `ridge_delta_thr0.1015`: held-out quality
  `0.8430`, utility `0.8086`, oracle-utility ratio `0.9555`;
- best test-only diagnostic residual `et_flip_leaf4_thr0.7895`: held-out
  quality `0.8547`, utility `0.8259`, oracle-utility ratio `0.9759`,
  frontier-call rate `0.1802`.

Paper interpretation: a base-tethered residual layer exposes a small
validation-selected improvement, mainly by reducing frontier calls while
preserving quality. The improvement is too small to change the headline: the
positive Broad100 method should remain the learned-verifiability/action-pool
bridge. Residual reliability is still an open mechanism, not a solved
oracle-level deployment method.

## Broad100 Current Best Package

Evidence:
`experiments/216_broad100_current_best_method_package.py`,
`results/controlled/broad100_current_best_method_package/BROAD100_CURRENT_BEST_METHOD_PACKAGE.md`,
`results/controlled/broad100_current_best_method_package/table_broad100_current_best_main_eval.csv`,
`results/controlled/broad100_current_best_method_package/table_broad100_current_best_summary.csv`,
and
`results/controlled/broad100_current_best_method_package/fig_broad100_current_best_utility.pdf`.

This package names the conservative residual repair as the current valid
Broad100 best method. It is a cache-only packaging step, not a new provider or
vLLM run. On 172 held-out Broad100 test queries, the current best reaches
quality `0.8547` versus oracle `0.8721`, quality gap `0.0174`, utility
`0.8238` versus oracle `0.8463`, oracle-utility ratio `0.9735`, and
frontier-call rate `0.1919`. The previous base utility is `0.8232`, so the
residual layer adds only `0.0006` utility while reducing frontier rate by
`0.0291`.

Paper interpretation: the current Broad100 package satisfies the numeric
oracle-level target as defined for Phase 3, because it is within 3 quality
points and above 97% oracle utility with fewer than 40% frontier calls. It
should still be described as a verifiability/action-pool bridge, not as a clean
no-tool benchmark-agnostic ProbeCode solution.

## Broad100 No-Tool Feasibility Bound

Evidence:
`experiments/217_broad100_no_tool_feasibility_bound.py`,
`results/controlled/broad100_no_tool_feasibility_bound/NO_TOOL_FEASIBILITY_BOUND_MEMO.md`,
`results/controlled/broad100_no_tool_feasibility_bound/table_no_tool_feasibility_bound.csv`,
`results/controlled/broad100_no_tool_feasibility_bound/table_no_tool_repair_oracle_normalized.csv`,
and
`results/controlled/broad100_no_tool_feasibility_bound/fig_no_tool_feasibility_bound.pdf`.

This cache-only diagnostic asks whether a clean no-tool Broad100 method could
meet the full-action-pool oracle target at all. It compares the full oracle to
an oracle over the same model pool after removing deterministic-tool local
actions. On 172 held-out Broad100 test queries:

- full action-pool oracle: quality `0.8721`, utility `0.8463`, frontier-call
  rate `0.1395`;
- no-tool action-pool oracle against the full oracle: quality `0.8256`,
  utility `0.7903`, quality gap `0.0465`, full-oracle utility ratio `0.9338`,
  frontier-call rate `0.1802`;
- validation-selected no-tool repair against full oracle: quality gap
  `0.0581`, utility ratio `0.9089`;
- the same no-tool repair against the no-tool oracle: quality gap `0.0116`,
  utility ratio `0.9733`.

Paper interpretation: clean no-tool routing cannot meet the current Broad100
full-oracle target with the cached action pool, because even the no-tool oracle
misses the target. The selected no-tool repair is close to its own no-tool
oracle, so the remaining miss is substantially an action-pool limitation, not
only a router-observability limitation. The supported Broad100 method should
therefore be framed as ProbeCode with observable verifiability plus useful
local/tool actions.

## Phase 3 Final Claim Package

Evidence:
`experiments/218_phase3_final_claim_package.py`,
`results/controlled/phase3_final_claim_package/PHASE3_FINAL_CLAIM_PACKAGE.md`,
`results/controlled/phase3_final_claim_package/table_phase3_final_claims.csv`,
`results/controlled/phase3_final_claim_package/table_phase3_final_method_evidence.csv`,
and
`results/controlled/phase3_final_claim_package/fig_phase3_final_claim_status.pdf`.

This cache-only package consolidates the current Phase 3 claim posture:

- cached Broad100 current best is supported at the oracle-level numeric target:
  quality gap `0.0174`, utility ratio `0.9735`, frontier-call rate `0.1919`;
- controlled mixed exact-math target gates are supported: quality gap `0.0152`,
  utility ratio `0.9739`, normalized remote cost `0.0463`, p95 latency ratio
  `0.4799`, frontier-call rate `0.1061`;
- active state calibration is supported on the cached exact-math setting:
  4 target-model evaluations reach quality `0.8485` versus best direct-router
  retraining quality `0.7273`;
- clean no-tool Broad100 is not supported against the full-action-pool oracle:
  the no-tool oracle itself has quality gap `0.0465` and utility ratio
  `0.9338`;
- the clean no-tool benchmark-agnostic variant remains unsupported, but the
  controlled verifiability/action-pool Phase 3 claim is complete under the
  current evidence package.

Paper interpretation: the defensible Phase 3 story is not that a no-tool text
router now reaches the oracle. The completed controlled claim is that a
RouteCode / ProbeCode policy with learned verifiability states and verifiable
local/tool actions reaches the configured oracle-level target on the current
cached Broad100 and controlled exact-math evidence. The next scientific step is
paper drafting plus broader replication of the same verifiability/action-pool
method.

## Broad100 GPT-Strong Math Action Repair

Evidence:
`experiments/219_broad100_gpt_strong_math_action.py`,
`results/controlled/broad100_gpt_strong_math_action/GPT_STRONG_MATH_ACTION_MEMO.md`,
`results/controlled/broad100_gpt_strong_math_action/table_gpt_strong_math_action_bounds.csv`,
`results/controlled/broad100_gpt_strong_math_action/table_gpt_strong_math_action_policy_selected.csv`,
and
`results/controlled/broad100_gpt_strong_math_action/model_outputs_with_gpt_strong_math_action.parquet`.

This experiment tested whether adding a non-tool `gpt-5.5-strong-solve` action
on exact-answer Broad100 math rows can substitute for deterministic-tool oracle
wins. It ran on the held-out validation/test rows only, used cached test rows
after an interrupted broader run, made 72 uncached validation calls, and recorded
`$1.2514` actual GPT cost for 144 rows total.

Held-out test result:

- full original oracle: quality `0.8721`, utility `0.8463`;
- no-tool original oracle versus full: quality `0.8256`, utility `0.7903`,
  quality gap `0.0465`, utility ratio `0.9338`;
- no-tool + GPT-strong oracle versus full: quality `0.8256`, utility `0.7903`,
  quality gap `0.0465`, utility ratio `0.9338`;
- validation-selected threshold policy: quality `0.8140`, utility `0.7689`,
  quality gap `0.0581`, utility ratio `0.9085`;
- all 144 val/test GPT-strong rows succeeded; no errors were present in the
  scored output table.

Diagnostic: the full oracle uses `deterministic_math_tool` on 16 held-out test
rows. GPT-strong is correct on only 8 of those rows and is wrong on the 8 rows
where the no-tool oracle loses quality relative to the full oracle. Therefore a
stronger prompt to GPT alone does not repair the action-pool limitation.

Paper interpretation: reaching the full oracle without tool/verifier actions
requires either a substantially better non-tool action on the deterministic-tool
residual rows or a different oracle/action definition. Threshold tuning or this
single GPT-strong action is not enough.

## Broad100 Residual Action Repairs

Evidence:
`experiments/219_broad100_gpt_strong_math_action.py`,
`experiments/220_broad100_gemini_residual_thinking_action.py`,
`results/controlled/broad100_gpt_strong_residual2048/GPT_STRONG_MATH_ACTION_MEMO.md`,
`results/controlled/broad100_gemini_residual_loss_thinking/GEMINI_RESIDUAL_THINKING_MEMO.md`,
`results/controlled/broad100_gpt_strong_residual2048/table_gpt_strong_math_action_bounds.csv`,
and
`results/controlled/broad100_gemini_residual_loss_thinking/table_gemini_residual_thinking_outputs.csv`.

The 512-token GPT-strong run failed partly because hard rows exhausted the
visible answer budget: the raw responses were `status=incomplete` with all
output tokens spent on hidden reasoning. A residual-only 2048-token retry on
the 40 validation/test deterministic-tool rows fixed that output-format issue.
It solved 7/8 held-out quality-loss rows, but the action is expensive under the
cost-aware objective:

- no-tool original oracle versus full: quality `0.8256`, utility `0.7903`,
  quality gap `0.0465`, utility ratio `0.9338`;
- no-tool + residual GPT-strong oracle versus full: quality `0.8372`,
  utility `0.7924`, quality gap `0.0349`, utility ratio `0.9364`;
- selected threshold policy with the same action: quality `0.8256`, utility
  `0.7710`, utility ratio `0.9111`.

A narrower Gemini high-thinking residual retry was attempted on the 8 held-out
quality-loss rows, but all Gemini calls returned HTTP 429 after retry/backoff.
That run is a provider-rate failure and should not be read as a negative Gemini
model-quality result.

Paper interpretation: a stronger non-tool action can recover most of the
residual **quality** gap, but not the cost-aware oracle target at current
pricing and token use. The next useful action-pool repair would need to be
either a cheaper residual solver, a better low-token answer extraction prompt,
or a validated verifiable/tool action framed as part of ProbeCode rather than
as a clean no-tool method.

## Phase 3 Oracle-Level Modification Summary

Evidence:
`experiments/221_phase3_oracle_level_modification_summary.py`,
`results/controlled/phase3_oracle_level_modification/table_oracle_level_modification.csv`,
and
`results/controlled/phase3_oracle_level_modification/ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md`.

This cache-only summary compares the tested modifications against the
configured Phase 3 oracle-level gate: within 3 quality points, at least 95%
oracle utility, and at most 40% frontier calls.

Passing rows:

- Broad100 learned-verifiability / verifiable-action-pool current best:
  quality gap `0.0174`, utility ratio `0.9735`, frontier-call rate `0.1919`;
- Broad100 compact RouteCode state policy: quality gap `0.0233`, utility ratio
  `0.9614`, frontier-call rate `0.2384`;
- mixed exact-math tool-augmented min-cost policy: quality gap `0.0152`,
  utility ratio `0.9739`, frontier-call rate `0.1061`.

Failing diagnostic rows:

- clean no-tool Broad100 oracle upper bound: quality gap `0.0465`, utility
  ratio `0.9338`;
- forcing GPT-5.5 strong-solve on the eight Broad100 residual quality-loss rows:
  quality gap `0.0000`, but utility ratio only `0.9234`;
- 512-token GPT-strong residual cap: `0/8` residual rows parsed/correct.

Paper interpretation: the supported route to the target is not threshold
tuning over the same no-tool action pool. The method needs cheap verifiable
local actions, and RouteCode/ProbeCode should be framed as learning or
observing when those actions are safe enough to avoid frontier calls.
