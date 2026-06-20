# RouteCode Exact-Math Selection Gate

Command:

```bash
PYTHONPATH=src python experiments/72_routecode_exact_math_selection.py --config configs/llmrouterbench_pilot.yaml --query-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv --output-dir results/phase2/routecode_exact_math_selection --k-values 4,8,16,32,64,128 --alpha-values 0,0.05,0.1,0.3,1,3,10 --validation-datasets aime,math500 --threshold 0.03 --target-k 32
```

Training datasets: `all`.
Validation datasets: `aime,math500`.
Policy-slice threshold: `0.0300` relative gap to oracle.
Target K selector: `32`.

Validation-selected candidate:

- `routecode_embedding_predicted:k4:alpha0`: validation gap `0.0417`, test gap `0.0354`, policy-slice gap `0.0526`, within threshold `False`.

Target-rate validation candidate:

- `routecode_embedding_predicted:k32:alpha0`: validation gap `0.0625`, test gap `0.0354`, policy-slice gap `0.0263`, within threshold `True`.

Candidates within the policy-slice threshold:

| candidate | val_selection_rank | val_relative_gap_to_oracle | test_relative_gap_to_oracle | policy_slice_relative_gap_to_oracle | policy_slice_regret_count |
| --- | --- | --- | --- | --- | --- |
| routecode_embedding_predicted:k32:alpha0 | 5 | 0.0625 | 0.0354 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha0 | 6 | 0.0625 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha0.05 | 7 | 0.0625 | 0.0354 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha0 | 8 | 0.0625 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha1 | 18 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha3 | 26 | 0.0833 | 0.0708 | 0.0263 | 1 |

Interpretation: a candidate that is within 3% on the held-out policy slice is not enough by itself. It must also be selected by a pre-declared validation protocol before it can replace the current core policy.
