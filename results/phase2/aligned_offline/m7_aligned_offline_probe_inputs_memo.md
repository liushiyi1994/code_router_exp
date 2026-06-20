# Aligned Offline Probe Inputs

Command:

```bash
python experiments/56_aligned_offline_probe_inputs.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2/aligned_offline
```

These files are aligned benchmark-derived scaffolding for M4/M5. They use train-only kNN uncertainty as an offline probe feature; they are not true local model probe evidence.

Summary:

- Probe rows: `2318`.
- State-target rows: `2318`.
- Policy test queries: `580`.

Files:

| artifact | path |
| --- | --- |
| probe_features | results/phase2/aligned_offline/aligned_probe_features.parquet |
| state_targets | results/phase2/aligned_offline/aligned_state_targets.csv |
| query_features | results/phase2/aligned_offline/aligned_query_features.csv |
| before_beliefs | results/phase2/aligned_offline/aligned_before_beliefs.csv |
| after_beliefs | results/phase2/aligned_offline/aligned_after_beliefs.csv |
| state_model_utility | results/phase2/aligned_offline/aligned_state_model_utility.csv |
| query_model_utility | results/phase2/aligned_offline/aligned_query_model_utility.csv |
| probe_cost | results/phase2/aligned_offline/aligned_probe_cost.csv |
| predicted_gain | results/phase2/aligned_offline/aligned_predicted_gain.csv |
