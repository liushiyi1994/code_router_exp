# ProbeRoute++ Controlled Experiment Pack

This pack contains the controlled Phase 2 experiment design for ProbeRoute++.

Files:

- `CONTROLLED_EXPERIMENT_PLAN.md`: full experiment plan.
- `CODEX_GOAL_CONTROLLED_EXPERIMENTS.md`: Codex `/goal` file.
- `STARTING_PROMPT_CONTROLLED_EXPERIMENTS.md`: short prompt to launch Codex.
- `COST_LATENCY_SCHEMA.md`: exact logging and metric schema.
- `MODEL_AND_BENCHMARK_NOTES.md`: model and benchmark setup notes.
- `CONFIG_TEMPLATE_CONTROLLED.yaml`: starter config.

Use this plan instead of the older broad Phase 2 plan when running actual experiments. The actual run should use 4–5 modern local models, 2 frontier models, 8–9 exact-scored benchmarks, and full cost/latency accounting.

- `EXPECTED_RESULTS_AND_SUCCESS_CRITERIA.md` — numerical success targets, expected results for each experiment, and claim gates including near-oracle, cost, latency, and calibration thresholds.
