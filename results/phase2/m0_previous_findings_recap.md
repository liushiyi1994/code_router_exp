# Phase 2 M0 Previous Findings Recap

Command:

```bash
python experiments/50_observability_gap_strong_encoders.py --result-dir results/llmrouterbench_pilot --result-dir results/llmrouterbench_broad10 --result-dir results/llmrouterbench_broad20 --result-dir results/llmrouterbench_scale20 --result-dir results/llmrouterbench_32model --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2
```

Purpose: restate the Phase 1 oracle-vs-deployable result as a partial-observability problem before adding probes or running local models.

Strong encoder rows were produced for M1 using cached/local encoder paths only. Status counts: `executed=40`. These rows preserve the invariant `query -> state -> model`.

Inputs:

- `results/llmrouterbench_pilot`
- `results/llmrouterbench_broad10`
- `results/llmrouterbench_broad20`
- `results/llmrouterbench_scale20`
- `results/llmrouterbench_32model`

Strong encoder configs:

- `configs/llmrouterbench_pilot.yaml`

## Main Observability Rows

| result_id | comparison | K | alpha | oracle_state_mean_utility | deployable_state_mean_utility | state_observability_gap | state_observability_gap_ci_low | state_observability_gap_ci_high | query_oracle_gap | query_oracle_gap_ci_low | query_oracle_gap_ci_high | full_gap_closed_vs_query_oracle | strong_encoder_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llmrouterbench_32model | d2_embedding_centroid_alpha_0 | 16.0000 | 0.0000 | 0.8686 | 0.6591 | 0.2094 | 0.1426 | 0.2887 | 0.2793 | 0.2123 | 0.3555 | -2.1628 | not_run_in_m0 |
| llmrouterbench_32model | d2_logistic_label_predictor_alpha_0 | 16.0000 | 0.0000 | 0.8686 | 0.6283 | 0.2402 | 0.1755 | 0.3132 | 0.3101 | 0.2452 | 0.3801 | -2.5116 | not_run_in_m0 |
| llmrouterbench_32model | d2_embedding_centroid_alpha_0p1 | 16.0000 | 0.1000 | 0.8604 | 0.7598 | 0.1006 | 0.0358 | 0.1572 | 0.1786 | 0.1168 | 0.2333 | -1.0233 | not_run_in_m0 |
| llmrouterbench_32model | d2_logistic_label_predictor_alpha_0p1 | 16.0000 | 0.1000 | 0.8604 | 0.7515 | 0.1088 | 0.0388 | 0.1685 | 0.1869 | 0.1199 | 0.2446 | -1.1163 | not_run_in_m0 |
| llmrouterbench_32model | d2_embedding_centroid_alpha_0p3 | 16.0000 | 0.3000 | 0.8419 | 0.7885 | 0.0534 | 0.0020 | 0.1129 | 0.1499 | 0.0963 | 0.2024 | -0.6977 | not_run_in_m0 |
| llmrouterbench_32model | d2_logistic_label_predictor_alpha_0p3 | 16.0000 | 0.3000 | 0.8419 | 0.7762 | 0.0657 | 0.0061 | 0.1367 | 0.1622 | 0.1004 | 0.2262 | -0.8372 | not_run_in_m0 |
| llmrouterbench_32model | d2_embedding_centroid_alpha_1 | 16.0000 | 1.0000 | 0.8337 | 0.7556 | 0.0780 | 0.0152 | 0.1459 | 0.1828 | 0.1240 | 0.2446 | -1.0698 | not_run_in_m0 |
| llmrouterbench_32model | d2_logistic_label_predictor_alpha_1 | 16.0000 | 1.0000 | 0.8337 | 0.7577 | 0.0760 | 0.0040 | 0.1377 | 0.1807 | 0.1127 | 0.2363 | -1.0465 | not_run_in_m0 |
| llmrouterbench_32model | d2_embedding_centroid_alpha_3 | 16.0000 | 3.0000 | 0.8378 | 0.8316 | 0.0062 | -0.0565 | 0.0709 | 0.1068 | 0.0470 | 0.1696 | -0.2093 | not_run_in_m0 |
| llmrouterbench_32model | d2_logistic_label_predictor_alpha_3 | 16.0000 | 3.0000 | 0.8378 | 0.8398 | -0.0021 | -0.0627 | 0.0514 | 0.0986 | 0.0409 | 0.1501 | -0.1163 | not_run_in_m0 |
| llmrouterbench_32model | d2_embedding_centroid_alpha_10 | 16.0000 | 10.0000 | 0.8398 | 0.8398 | 0.0000 | -0.0669 | 0.0525 | 0.0986 | 0.0450 | 0.1471 | -0.1163 | not_run_in_m0 |
| llmrouterbench_32model | d2_logistic_label_predictor_alpha_10 | 16.0000 | 10.0000 | 0.8398 | 0.8460 | -0.0062 | -0.0740 | 0.0473 | 0.0924 | 0.0378 | 0.1419 | -0.0465 | not_run_in_m0 |
| llmrouterbench_32model | flat_routecode_logistic_label_predictor | 16.0000 |  | 0.8645 | 0.7844 | 0.0801 | 0.0091 | 0.1366 | 0.1540 | 0.0860 | 0.2055 | -0.7442 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_embedding_centroid_alpha_0 | 16.0000 | 0.0000 | 0.8244 | 0.5944 | 0.2301 | 0.2042 | 0.2630 | 0.2877 | 0.2604 | 0.3189 | -0.3289 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_logistic_label_predictor_alpha_0 | 16.0000 | 0.0000 | 0.8244 | 0.5816 | 0.2429 | 0.2176 | 0.2736 | 0.3006 | 0.2738 | 0.3295 | -0.3882 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_embedding_centroid_alpha_0p1 | 16.0000 | 0.1000 | 0.7621 | 0.6182 | 0.1439 | 0.1114 | 0.1760 | 0.2639 | 0.2303 | 0.2956 | -0.2187 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_logistic_label_predictor_alpha_0p1 | 16.0000 | 0.1000 | 0.7621 | 0.6207 | 0.1414 | 0.1147 | 0.1710 | 0.2614 | 0.2335 | 0.2907 | -0.2072 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_embedding_centroid_alpha_1 | 16.0000 | 1.0000 | 0.7062 | 0.7048 | 0.0014 | -0.0300 | 0.0366 | 0.1774 | 0.1505 | 0.2096 | 0.1809 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_logistic_label_predictor_alpha_1 | 16.0000 | 1.0000 | 0.7062 | 0.7044 | 0.0018 | -0.0334 | 0.0338 | 0.1777 | 0.1472 | 0.2069 | 0.1793 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_embedding_centroid_alpha_3 | 16.0000 | 3.0000 | 0.7009 | 0.7009 | 0.0000 | -0.0322 | 0.0264 | 0.1813 | 0.1534 | 0.2064 | 0.1628 | not_run_in_m0 |
| llmrouterbench_broad10 | d2_logistic_label_predictor_alpha_3 | 16.0000 | 3.0000 | 0.7009 | 0.7005 | 0.0004 | -0.0338 | 0.0299 | 0.1816 | 0.1519 | 0.2100 | 0.1612 | not_run_in_m0 |
| llmrouterbench_broad10 | flat_routecode_logistic_label_predictor | 16.0000 |  | 0.8159 | 0.5395 | 0.2764 | 0.2442 | 0.3071 | 0.3426 | 0.3143 | 0.3749 | -0.5822 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_0 | 16.0000 | 0.0000 | 0.7899 | 0.6585 | 0.1314 | 0.1040 | 0.1649 | 0.2575 | 0.2313 | 0.2855 | -0.2131 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_0 | 16.0000 | 0.0000 | 0.7899 | 0.6115 | 0.1784 | 0.1517 | 0.2151 | 0.3045 | 0.2790 | 0.3357 | -0.4346 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_0p05 | 16.0000 | 0.0500 | 0.7696 | 0.6592 | 0.1104 | 0.0788 | 0.1418 | 0.2568 | 0.2304 | 0.2853 | -0.2097 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_0p05 | 16.0000 | 0.0500 | 0.7696 | 0.6150 | 0.1546 | 0.1262 | 0.1850 | 0.3009 | 0.2778 | 0.3286 | -0.4178 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_0p1 | 16.0000 | 0.1000 | 0.7496 | 0.6802 | 0.0694 | 0.0380 | 0.0998 | 0.2358 | 0.2092 | 0.2657 | -0.1107 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_0p1 | 16.0000 | 0.1000 | 0.7496 | 0.6278 | 0.1218 | 0.0873 | 0.1518 | 0.2881 | 0.2585 | 0.3177 | -0.3574 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_0p3 | 16.0000 | 0.3000 | 0.7240 | 0.7197 | 0.0043 | -0.0284 | 0.0342 | 0.1962 | 0.1687 | 0.2224 | 0.0755 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_0p3 | 16.0000 | 0.3000 | 0.7240 | 0.7176 | 0.0064 | -0.0229 | 0.0369 | 0.1984 | 0.1743 | 0.2251 | 0.0654 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_1 | 16.0000 | 1.0000 | 0.7276 | 0.7229 | 0.0046 | -0.0332 | 0.0350 | 0.1930 | 0.1670 | 0.2218 | 0.0906 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_1 | 16.0000 | 1.0000 | 0.7276 | 0.7187 | 0.0089 | -0.0303 | 0.0378 | 0.1973 | 0.1698 | 0.2246 | 0.0705 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_3 | 16.0000 | 3.0000 | 0.7183 | 0.7172 | 0.0011 | -0.0278 | 0.0312 | 0.1987 | 0.1739 | 0.2264 | 0.0638 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_3 | 16.0000 | 3.0000 | 0.7183 | 0.7176 | 0.0007 | -0.0276 | 0.0339 | 0.1984 | 0.1741 | 0.2291 | 0.0654 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_embedding_centroid_alpha_10 | 16.0000 | 10.0000 | 0.7187 | 0.7190 | -0.0004 | -0.0349 | 0.0343 | 0.1969 | 0.1684 | 0.2269 | 0.0721 | not_run_in_m0 |
| llmrouterbench_broad20 | d2_logistic_label_predictor_alpha_10 | 16.0000 | 10.0000 | 0.7187 | 0.7194 | -0.0007 | -0.0284 | 0.0308 | 0.1966 | 0.1750 | 0.2235 | 0.0738 | not_run_in_m0 |
| llmrouterbench_broad20 | flat_routecode_logistic_label_predictor | 16.0000 |  | 0.7813 | 0.6022 | 0.1791 | 0.1425 | 0.2105 | 0.3137 | 0.2854 | 0.3434 | -0.4782 | not_run_in_m0 |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_0 | 16.0000 | 0.0000 | 0.8759 | 0.6948 | 0.1810 | 0.1163 | 0.2457 | 0.2017 | 0.1405 | 0.2673 | 0.1203 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_0 | 16.0000 | 0.0000 | 0.8759 | 0.6914 | 0.1845 | 0.1266 | 0.2509 | 0.2052 | 0.1508 | 0.2725 | 0.1053 | not_run_in_m0 |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_0p05 | 16.0000 | 0.0500 | 0.8707 | 0.6086 | 0.2621 | 0.1983 | 0.3268 | 0.2879 | 0.2276 | 0.3518 | -0.2556 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_0p05 | 16.0000 | 0.0500 | 0.8707 | 0.6328 | 0.2379 | 0.1732 | 0.2975 | 0.2638 | 0.2025 | 0.3225 | -0.1504 | not_run_in_m0 |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_0p1 | 16.0000 | 0.1000 | 0.8466 | 0.6500 | 0.1966 | 0.1353 | 0.2655 | 0.2466 | 0.1905 | 0.3147 | -0.0752 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_0p1 | 16.0000 | 0.1000 | 0.8466 | 0.6345 | 0.2121 | 0.1456 | 0.2768 | 0.2621 | 0.2008 | 0.3259 | -0.1429 | not_run_in_m0 |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_0p3 | 16.0000 | 0.3000 | 0.7534 | 0.7379 | 0.0155 | -0.0578 | 0.0793 | 0.1586 | 0.1000 | 0.2164 | 0.3083 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_0p3 | 16.0000 | 0.3000 | 0.7534 | 0.7241 | 0.0293 | -0.0441 | 0.0966 | 0.1724 | 0.1137 | 0.2337 | 0.2481 | not_run_in_m0 |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_1 | 16.0000 | 1.0000 | 0.7466 | 0.7431 | 0.0034 | -0.0664 | 0.0759 | 0.1534 | 0.0974 | 0.2191 | 0.3308 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_1 | 16.0000 | 1.0000 | 0.7466 | 0.7328 | 0.0138 | -0.0552 | 0.0802 | 0.1638 | 0.1086 | 0.2233 | 0.2857 | not_run_in_m0 |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_3 | 16.0000 | 3.0000 | 0.7466 | 0.7466 | 0.0000 | -0.0708 | 0.0603 | 0.1500 | 0.0905 | 0.2061 | 0.3459 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_3 | 16.0000 | 3.0000 | 0.7466 | 0.7448 | 0.0017 | -0.0681 | 0.0699 | 0.1517 | 0.0931 | 0.2156 | 0.3383 | not_run_in_m0 |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_centroid | 16.0000 | 3.0000 | 0.7517 | 0.7517 | 0.0000 | -0.0690 | 0.0638 | 0.1448 | 0.0862 | 0.1957 | 0.3684 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_centroid | 16.0000 | 3.0000 | 0.7448 | 0.7448 | 0.0000 | -0.0734 | 0.0699 | 0.1517 | 0.0904 | 0.2095 | 0.3383 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_centroid | 16.0000 | 3.0000 | 0.7448 | 0.7448 | 0.0000 | -0.0699 | 0.0716 | 0.1517 | 0.0922 | 0.2087 | 0.3383 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_centroid | 16.0000 | 3.0000 | 0.7241 | 0.7241 | 0.0000 | -0.0647 | 0.0725 | 0.1724 | 0.1172 | 0.2311 | 0.2481 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_centroid | 16.0000 | 3.0000 | 0.7517 | 0.7500 | 0.0017 | -0.0630 | 0.0708 | 0.1466 | 0.0879 | 0.2018 | 0.3609 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_knn | 16.0000 | 3.0000 | 0.7517 | 0.7534 | -0.0017 | -0.0803 | 0.0672 | 0.1431 | 0.0819 | 0.1975 | 0.3759 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_knn | 16.0000 | 3.0000 | 0.7448 | 0.7466 | -0.0017 | -0.0707 | 0.0647 | 0.1500 | 0.0948 | 0.2053 | 0.3459 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_knn | 16.0000 | 3.0000 | 0.7448 | 0.7448 | 0.0000 | -0.0657 | 0.0664 | 0.1517 | 0.0964 | 0.2104 | 0.3383 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_knn | 16.0000 | 3.0000 | 0.7241 | 0.7293 | -0.0052 | -0.0759 | 0.0699 | 0.1672 | 0.1060 | 0.2250 | 0.2707 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_knn | 16.0000 | 3.0000 | 0.7517 | 0.7259 | 0.0259 | -0.0423 | 0.0957 | 0.1707 | 0.1138 | 0.2277 | 0.2556 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_logistic | 16.0000 | 3.0000 | 0.7517 | 0.7466 | 0.0052 | -0.0586 | 0.0742 | 0.1500 | 0.0948 | 0.2018 | 0.3459 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_logistic | 16.0000 | 3.0000 | 0.7448 | 0.7483 | -0.0034 | -0.0716 | 0.0699 | 0.1483 | 0.0914 | 0.2035 | 0.3534 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_logistic | 16.0000 | 3.0000 | 0.7448 | 0.7483 | -0.0034 | -0.0716 | 0.0639 | 0.1483 | 0.0879 | 0.2035 | 0.3534 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_logistic | 16.0000 | 3.0000 | 0.7241 | 0.7241 | 0.0000 | -0.0750 | 0.0734 | 0.1724 | 0.1077 | 0.2311 | 0.2481 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_logistic | 16.0000 | 3.0000 | 0.7517 | 0.7517 | 0.0000 | -0.0716 | 0.0664 | 0.1448 | 0.0853 | 0.1984 | 0.3684 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_mlp | 16.0000 | 3.0000 | 0.7517 | 0.7448 | 0.0069 | -0.0647 | 0.0734 | 0.1517 | 0.0939 | 0.2096 | 0.3383 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_mlp | 16.0000 | 3.0000 | 0.7448 | 0.7466 | -0.0017 | -0.0707 | 0.0630 | 0.1500 | 0.0897 | 0.2044 | 0.3459 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_mlp | 16.0000 | 3.0000 | 0.7448 | 0.7534 | -0.0086 | -0.0794 | 0.0603 | 0.1431 | 0.0836 | 0.1992 | 0.3759 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_mlp | 16.0000 | 3.0000 | 0.7241 | 0.7155 | 0.0086 | -0.0673 | 0.0845 | 0.1810 | 0.1163 | 0.2406 | 0.2105 | executed |
| llmrouterbench_pilot | d2_predictability_constrained_strong_encoder_mlp | 16.0000 | 3.0000 | 0.7517 | 0.7431 | 0.0086 | -0.0595 | 0.0837 | 0.1534 | 0.0948 | 0.2104 | 0.3308 | executed |
| llmrouterbench_pilot | d2_embedding_centroid_alpha_10 | 16.0000 | 10.0000 | 0.7431 | 0.7414 | 0.0017 | -0.0673 | 0.0699 | 0.1552 | 0.0991 | 0.2130 | 0.3233 | not_run_in_m0 |
| llmrouterbench_pilot | d2_logistic_label_predictor_alpha_10 | 16.0000 | 10.0000 | 0.7431 | 0.7448 | -0.0017 | -0.0699 | 0.0708 | 0.1517 | 0.0966 | 0.2139 | 0.3383 | not_run_in_m0 |
| llmrouterbench_pilot | flat_routecode_logistic_label_predictor | 16.0000 |  | 0.8897 | 0.6138 | 0.2759 | 0.2146 | 0.3423 | 0.2828 | 0.2250 | 0.3501 | -0.2331 | not_run_in_m0 |
| llmrouterbench_pilot | flat_routecode_strong_encoder_centroid | 16.0000 |  | 0.8897 | 0.6655 | 0.2241 | 0.1586 | 0.2837 | 0.2310 | 0.1690 | 0.2880 | -0.0075 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_centroid | 16.0000 |  | 0.8793 | 0.6914 | 0.1879 | 0.1310 | 0.2457 | 0.2052 | 0.1500 | 0.2630 | 0.1053 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_centroid | 16.0000 |  | 0.8810 | 0.6810 | 0.2000 | 0.1352 | 0.2655 | 0.2155 | 0.1534 | 0.2750 | 0.0602 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_centroid | 16.0000 |  | 0.8810 | 0.5931 | 0.2879 | 0.2293 | 0.3509 | 0.3034 | 0.2466 | 0.3639 | -0.3233 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_centroid | 16.0000 |  | 0.8897 | 0.6155 | 0.2741 | 0.2103 | 0.3381 | 0.2810 | 0.2181 | 0.3425 | -0.2256 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_knn | 16.0000 |  | 0.8897 | 0.6466 | 0.2431 | 0.1741 | 0.3026 | 0.2500 | 0.1827 | 0.3070 | -0.0902 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_knn | 16.0000 |  | 0.8793 | 0.6431 | 0.2362 | 0.1741 | 0.2957 | 0.2534 | 0.1922 | 0.3112 | -0.1053 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_knn | 16.0000 |  | 0.8810 | 0.6362 | 0.2448 | 0.1853 | 0.3095 | 0.2603 | 0.1974 | 0.3208 | -0.1353 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_knn | 16.0000 |  | 0.8810 | 0.6224 | 0.2586 | 0.1966 | 0.3190 | 0.2741 | 0.2138 | 0.3354 | -0.1955 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_knn | 16.0000 |  | 0.8897 | 0.6293 | 0.2603 | 0.1939 | 0.3251 | 0.2672 | 0.2043 | 0.3294 | -0.1654 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_logistic | 16.0000 |  | 0.8897 | 0.6207 | 0.2690 | 0.2052 | 0.3380 | 0.2759 | 0.2155 | 0.3424 | -0.2030 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_logistic | 16.0000 |  | 0.8793 | 0.6828 | 0.1966 | 0.1336 | 0.2596 | 0.2138 | 0.1534 | 0.2759 | 0.0677 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_logistic | 16.0000 |  | 0.8810 | 0.6224 | 0.2586 | 0.2017 | 0.3190 | 0.2741 | 0.2155 | 0.3319 | -0.1955 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_logistic | 16.0000 |  | 0.8810 | 0.6259 | 0.2552 | 0.1947 | 0.3182 | 0.2707 | 0.2094 | 0.3286 | -0.1805 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_logistic | 16.0000 |  | 0.8897 | 0.6190 | 0.2707 | 0.2042 | 0.3397 | 0.2776 | 0.2137 | 0.3406 | -0.2105 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_mlp | 16.0000 |  | 0.8897 | 0.6431 | 0.2466 | 0.1810 | 0.3121 | 0.2534 | 0.1897 | 0.3130 | -0.1053 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_mlp | 16.0000 |  | 0.8793 | 0.6603 | 0.2190 | 0.1586 | 0.2897 | 0.2362 | 0.1759 | 0.2993 | -0.0301 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_mlp | 16.0000 |  | 0.8810 | 0.6466 | 0.2345 | 0.1732 | 0.2957 | 0.2500 | 0.1922 | 0.3070 | -0.0902 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_mlp | 16.0000 |  | 0.8810 | 0.6259 | 0.2552 | 0.1861 | 0.3191 | 0.2707 | 0.2034 | 0.3319 | -0.1805 | executed |
| llmrouterbench_pilot | flat_routecode_strong_encoder_mlp | 16.0000 |  | 0.8897 | 0.6224 | 0.2672 | 0.2008 | 0.3346 | 0.2741 | 0.2112 | 0.3380 | -0.1955 | executed |
| llmrouterbench_scale20 | d2_embedding_centroid_alpha_0 | 16.0000 | 0.0000 | 0.8276 | 0.6552 | 0.1724 | 0.0982 | 0.2397 | 0.3086 | 0.2499 | 0.3578 | -0.2013 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_logistic_label_predictor_alpha_0 | 16.0000 | 0.0000 | 0.8276 | 0.6241 | 0.2034 | 0.1361 | 0.2665 | 0.3397 | 0.2878 | 0.3846 | -0.3221 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_embedding_centroid_alpha_0p1 | 16.0000 | 0.1000 | 0.7828 | 0.6948 | 0.0879 | 0.0266 | 0.1555 | 0.2690 | 0.2206 | 0.3175 | -0.0470 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_logistic_label_predictor_alpha_0p1 | 16.0000 | 0.1000 | 0.7828 | 0.6603 | 0.1224 | 0.0507 | 0.1846 | 0.3034 | 0.2447 | 0.3466 | -0.1812 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_embedding_centroid_alpha_0p3 | 16.0000 | 0.3000 | 0.7362 | 0.7034 | 0.0328 | -0.0441 | 0.1018 | 0.2603 | 0.2007 | 0.3070 | -0.0134 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_logistic_label_predictor_alpha_0p3 | 16.0000 | 0.3000 | 0.7362 | 0.6862 | 0.0500 | -0.0200 | 0.1155 | 0.2776 | 0.2249 | 0.3207 | -0.0805 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_embedding_centroid_alpha_1 | 16.0000 | 1.0000 | 0.7345 | 0.7328 | 0.0017 | -0.0725 | 0.0718 | 0.2310 | 0.1792 | 0.2760 | 0.1007 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_logistic_label_predictor_alpha_1 | 16.0000 | 1.0000 | 0.7345 | 0.7224 | 0.0121 | -0.0622 | 0.0854 | 0.2414 | 0.1896 | 0.2897 | 0.0604 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_embedding_centroid_alpha_3 | 16.0000 | 3.0000 | 0.7224 | 0.7172 | 0.0052 | -0.0595 | 0.0768 | 0.2466 | 0.1947 | 0.2966 | 0.0403 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_logistic_label_predictor_alpha_3 | 16.0000 | 3.0000 | 0.7224 | 0.7241 | -0.0017 | -0.0673 | 0.0657 | 0.2397 | 0.1869 | 0.2855 | 0.0671 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_embedding_centroid_alpha_10 | 16.0000 | 10.0000 | 0.7345 | 0.7379 | -0.0034 | -0.0709 | 0.0614 | 0.2259 | 0.1739 | 0.2681 | 0.1208 | not_run_in_m0 |
| llmrouterbench_scale20 | d2_logistic_label_predictor_alpha_10 | 16.0000 | 10.0000 | 0.7345 | 0.7310 | 0.0034 | -0.0595 | 0.0709 | 0.2328 | 0.1853 | 0.2776 | 0.0940 | not_run_in_m0 |
| llmrouterbench_scale20 | flat_routecode_logistic_label_predictor | 16.0000 |  | 0.8155 | 0.6569 | 0.1586 | 0.0844 | 0.2148 | 0.3069 | 0.2516 | 0.3415 | -0.1946 | not_run_in_m0 |

## Strong Encoder Rows

| result_id | model_id | state_family | state_predictor | status | label_accuracy | deployable_state_mean_utility | deployable_state_mean_utility_ci_low | deployable_state_mean_utility_ci_high | state_observability_gap | state_observability_gap_ci_low | state_observability_gap_ci_high | full_gap_closed_vs_query_oracle | routing_invariant | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | d2_predictability_constrained | centroid | executed | 0.9983 | 0.7517 | 0.7224 | 0.7862 | 0.0000 | -0.0690 | 0.0638 | 0.3684 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | d2_predictability_constrained | centroid | executed | 0.9966 | 0.7448 | 0.7086 | 0.7820 | 0.0000 | -0.0734 | 0.0699 | 0.3383 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | d2_predictability_constrained | centroid | executed | 1.0000 | 0.7448 | 0.7094 | 0.7802 | 0.0000 | -0.0699 | 0.0716 | 0.3383 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | d2_predictability_constrained | centroid | executed | 1.0000 | 0.7241 | 0.6870 | 0.7552 | 0.0000 | -0.0647 | 0.0725 | 0.2481 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | d2_predictability_constrained | centroid | executed | 0.9948 | 0.7500 | 0.7163 | 0.7845 | 0.0017 | -0.0630 | 0.0708 | 0.3609 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | d2_predictability_constrained | knn | executed | 0.8310 | 0.7534 | 0.7207 | 0.7906 | -0.0017 | -0.0803 | 0.0672 | 0.3759 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | d2_predictability_constrained | knn | executed | 0.8500 | 0.7466 | 0.7129 | 0.7776 | -0.0017 | -0.0707 | 0.0647 | 0.3459 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | d2_predictability_constrained | knn | executed | 0.8431 | 0.7448 | 0.7077 | 0.7760 | 0.0000 | -0.0657 | 0.0664 | 0.3383 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | d2_predictability_constrained | knn | executed | 0.8069 | 0.7293 | 0.6931 | 0.7664 | -0.0052 | -0.0759 | 0.0699 | 0.2707 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | d2_predictability_constrained | knn | executed | 0.5690 | 0.7259 | 0.6905 | 0.7586 | 0.0259 | -0.0423 | 0.0957 | 0.2556 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | d2_predictability_constrained | logistic | executed | 0.9259 | 0.7466 | 0.7163 | 0.7776 | 0.0052 | -0.0586 | 0.0742 | 0.3459 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | d2_predictability_constrained | logistic | executed | 0.9207 | 0.7483 | 0.7146 | 0.7810 | -0.0034 | -0.0716 | 0.0699 | 0.3534 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | d2_predictability_constrained | logistic | executed | 0.9259 | 0.7483 | 0.7146 | 0.7845 | -0.0034 | -0.0716 | 0.0639 | 0.3534 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | d2_predictability_constrained | logistic | executed | 0.8724 | 0.7241 | 0.6870 | 0.7647 | 0.0000 | -0.0750 | 0.0734 | 0.2481 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | d2_predictability_constrained | logistic | executed | 0.9052 | 0.7517 | 0.7198 | 0.7871 | 0.0000 | -0.0716 | 0.0664 | 0.3684 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | d2_predictability_constrained | mlp | executed | 0.8293 | 0.7448 | 0.7085 | 0.7785 | 0.0069 | -0.0647 | 0.0734 | 0.3383 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | d2_predictability_constrained | mlp | executed | 0.8690 | 0.7466 | 0.7138 | 0.7828 | -0.0017 | -0.0707 | 0.0630 | 0.3459 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | d2_predictability_constrained | mlp | executed | 0.8741 | 0.7534 | 0.7190 | 0.7888 | -0.0086 | -0.0794 | 0.0603 | 0.3759 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | d2_predictability_constrained | mlp | executed | 0.8207 | 0.7155 | 0.6776 | 0.7561 | 0.0086 | -0.0673 | 0.0845 | 0.2105 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | d2_predictability_constrained | mlp | executed | 0.8397 | 0.7431 | 0.7077 | 0.7776 | 0.0086 | -0.0595 | 0.0837 | 0.3308 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | flat_routecode | centroid | executed | 0.1448 | 0.6655 | 0.6301 | 0.7034 | 0.2241 | 0.1586 | 0.2837 | -0.0075 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | flat_routecode | centroid | executed | 0.1466 | 0.6914 | 0.6552 | 0.7224 | 0.1879 | 0.1310 | 0.2457 | 0.1053 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | flat_routecode | centroid | executed | 0.1724 | 0.6810 | 0.6431 | 0.7191 | 0.2000 | 0.1352 | 0.2655 | 0.0602 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | flat_routecode | centroid | executed | 0.1276 | 0.5931 | 0.5543 | 0.6259 | 0.2879 | 0.2293 | 0.3509 | -0.3233 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | flat_routecode | centroid | executed | 0.1690 | 0.6155 | 0.5757 | 0.6544 | 0.2741 | 0.2103 | 0.3381 | -0.2256 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | flat_routecode | knn | executed | 0.2483 | 0.6466 | 0.6112 | 0.6897 | 0.2431 | 0.1741 | 0.3026 | -0.0902 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | flat_routecode | knn | executed | 0.2414 | 0.6431 | 0.6069 | 0.6802 | 0.2362 | 0.1741 | 0.2957 | -0.1053 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | flat_routecode | knn | executed | 0.2328 | 0.6362 | 0.5974 | 0.6750 | 0.2448 | 0.1853 | 0.3095 | -0.1353 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | flat_routecode | knn | executed | 0.2000 | 0.6224 | 0.5828 | 0.6586 | 0.2586 | 0.1966 | 0.3190 | -0.1955 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | flat_routecode | knn | executed | 0.1793 | 0.6293 | 0.5887 | 0.6681 | 0.2603 | 0.1939 | 0.3251 | -0.1654 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | flat_routecode | logistic | executed | 0.1517 | 0.6207 | 0.5758 | 0.6569 | 0.2690 | 0.2052 | 0.3380 | -0.2030 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | flat_routecode | logistic | executed | 0.2069 | 0.6828 | 0.6422 | 0.7190 | 0.1966 | 0.1336 | 0.2596 | 0.0677 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | flat_routecode | logistic | executed | 0.1672 | 0.6224 | 0.5862 | 0.6569 | 0.2586 | 0.2017 | 0.3190 | -0.1955 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | flat_routecode | logistic | executed | 0.1862 | 0.6259 | 0.5896 | 0.6630 | 0.2552 | 0.1947 | 0.3182 | -0.1805 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | flat_routecode | logistic | executed | 0.1741 | 0.6190 | 0.5776 | 0.6587 | 0.2707 | 0.2042 | 0.3397 | -0.2105 | query_to_state_to_model |  |
| llmrouterbench_pilot | BAAI/bge-small-en-v1.5 | flat_routecode | mlp | executed | 0.1431 | 0.6431 | 0.6052 | 0.6828 | 0.2466 | 0.1810 | 0.3121 | -0.1053 | query_to_state_to_model |  |
| llmrouterbench_pilot | answerdotai/ModernBERT-base | flat_routecode | mlp | executed | 0.1897 | 0.6603 | 0.6189 | 0.6966 | 0.2190 | 0.1586 | 0.2897 | -0.0301 | query_to_state_to_model |  |
| llmrouterbench_pilot | intfloat/e5-small-v2 | flat_routecode | mlp | executed | 0.1552 | 0.6466 | 0.6112 | 0.6802 | 0.2345 | 0.1732 | 0.2957 | -0.0902 | query_to_state_to_model |  |
| llmrouterbench_pilot | microsoft/deberta-v3-base | flat_routecode | mlp | executed | 0.1759 | 0.6259 | 0.5862 | 0.6691 | 0.2552 | 0.1861 | 0.3191 | -0.1805 | query_to_state_to_model |  |
| llmrouterbench_pilot | sentence-transformers/all-MiniLM-L6-v2 | flat_routecode | mlp | executed | 0.1776 | 0.6224 | 0.5801 | 0.6613 | 0.2672 | 0.2008 | 0.3346 | -0.1955 | query_to_state_to_model |  |

## Observations

- Flat utility-oracle labels remain a diagnostic upper bound: `llmrouterbench_broad10` has state-observability gap 0.2764 and full query-oracle gap 0.3426.
- Best current deployable state assignment in this M0 table: `llmrouterbench_pilot` / `d2_predictability_constrained_strong_encoder_knn` with 0.3759 of the best-single-to-query-oracle gap recovered.
- Best executed strong-encoder state predictor: `llmrouterbench_pilot` / `BAAI/bge-small-en-v1.5` / `d2_predictability_constrained_strong_encoder_knn` with label accuracy `0.8310` and recovered query-oracle gap `0.3759`.
- The Phase 1 evidence supports the Phase 2 premise: useful route states can exist while the query-only assignment remains partially observed.
- This is still not evidence that probes close the gap; probe rows and cost-adjusted policies must be added by later Phase 2 experiments.

## Outputs

- `table_observability_strong_encoders.csv`: route-state oracle vs deployable query-to-state comparisons.
- `fig_observability_gap.pdf`: visual summary of state and query-oracle gaps.
- `m0_previous_findings_recap.md`: this recap.
