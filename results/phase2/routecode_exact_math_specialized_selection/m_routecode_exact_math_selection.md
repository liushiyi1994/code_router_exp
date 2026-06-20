# RouteCode Exact-Math Selection Gate

Command:

```bash
PYTHONPATH=src python experiments/72_routecode_exact_math_selection.py --config configs/llmrouterbench_pilot.yaml --query-model-utility results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv --output-dir results/phase2/routecode_exact_math_specialized_selection --k-values 1,2,4,8,16,32,64,128,256 --alpha-values 0,0.01,0.05,0.1,0.3,1,3,10 --validation-datasets aime,math500 --threshold 0.03 --target-k 32 --training-datasets aime,math500
```

Training datasets: `aime,math500`.
Validation datasets: `aime,math500`.
Policy-slice threshold: `0.0300` relative gap to oracle.
Target K selector: `32`.

Validation-selected candidate:

- `routecode_embedding_predicted:k4:alpha3`: validation gap `0.0312`, test gap `0.0531`, policy-slice gap `0.0789`, within threshold `False`.

Target-rate validation candidate:

- `routecode_embedding_predicted:k32:alpha0.1`: validation gap `0.0625`, test gap `0.0442`, policy-slice gap `0.0526`, within threshold `False`.

Candidates within the policy-slice threshold:

| candidate | val_selection_rank | val_relative_gap_to_oracle | test_relative_gap_to_oracle | policy_slice_relative_gap_to_oracle | policy_slice_regret_count |
| --- | --- | --- | --- | --- | --- |
| routecode_embedding_predicted:k4:alpha0.3 | 6 | 0.0521 | 0.0265 | 0.0000 | 0 |
| routecode_embedding_predicted:k16:alpha3 | 7 | 0.0521 | 0.0354 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha1 | 8 | 0.0521 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha3 | 9 | 0.0521 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha10 | 10 | 0.0521 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k2:alpha0 | 12 | 0.0625 | 0.0177 | 0.0000 | 0 |
| routecode_embedding_predicted:k2:alpha0.01 | 13 | 0.0625 | 0.0177 | 0.0000 | 0 |
| routecode_embedding_predicted:k2:alpha0.05 | 14 | 0.0625 | 0.0177 | 0.0000 | 0 |
| routecode_embedding_predicted:k2:alpha0.1 | 15 | 0.0625 | 0.0177 | 0.0000 | 0 |
| routecode_embedding_predicted:k2:alpha0.3 | 16 | 0.0625 | 0.0177 | 0.0000 | 0 |
| routecode_embedding_predicted:k8:alpha0.05 | 17 | 0.0625 | 0.0177 | 0.0000 | 0 |
| routecode_embedding_predicted:k8:alpha0 | 18 | 0.0625 | 0.0265 | 0.0263 | 1 |
| routecode_embedding_predicted:k8:alpha0.01 | 19 | 0.0625 | 0.0177 | 0.0263 | 1 |
| routecode_embedding_predicted:k8:alpha0.1 | 20 | 0.0625 | 0.0265 | 0.0263 | 1 |
| routecode_embedding_predicted:k16:alpha0.05 | 21 | 0.0625 | 0.0265 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha0.1 | 22 | 0.0625 | 0.0354 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha0.3 | 23 | 0.0625 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha0.05 | 24 | 0.0625 | 0.0708 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha0.1 | 25 | 0.0625 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha0.3 | 26 | 0.0625 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha3 | 27 | 0.0625 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha1 | 45 | 0.0729 | 0.0265 | 0.0000 | 0 |
| routecode_embedding_predicted:k16:alpha1 | 46 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k16:alpha10 | 47 | 0.0729 | 0.0442 | 0.0263 | 1 |
| routecode_embedding_predicted:k32:alpha0 | 48 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha0 | 49 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha0 | 50 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k128:alpha0.01 | 51 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha0 | 52 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha0.01 | 53 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha0.05 | 54 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha0.1 | 55 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k256:alpha0.3 | 56 | 0.0729 | 0.0531 | 0.0263 | 1 |
| routecode_embedding_predicted:k64:alpha1 | 64 | 0.0833 | 0.0442 | 0.0000 | 0 |
| routecode_embedding_predicted:k32:alpha0.3 | 65 | 0.0833 | 0.0354 | 0.0263 | 1 |

Interpretation: a candidate that is within 3% on the held-out policy slice is not enough by itself. It must also be selected by a pre-declared validation protocol before it can replace the current core policy.
