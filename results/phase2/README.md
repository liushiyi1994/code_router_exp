# RouteCode Phase 2 Results

## Phase 2 Observability Gap

Command:

```bash
python experiments/50_observability_gap_strong_encoders.py --result-dir results/llmrouterbench_pilot --result-dir results/llmrouterbench_broad10 --result-dir results/llmrouterbench_broad20 --result-dir results/llmrouterbench_scale20 --result-dir results/llmrouterbench_32model --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2
```

This is the M0 bridge from Phase 1 to Phase 2. It does not run probes, APIs, or local generative models. It quantifies the current observability gap using saved Phase 1 tables.

Strong encoder rows were produced for M1 using cached/local encoder paths only. Status counts: `executed=40`. These rows preserve the invariant `query -> state -> model`.

Outputs:

- `table_observability_strong_encoders.csv`
- `fig_observability_gap.pdf`
- `m0_previous_findings_recap.md`

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

## Phase 2 Local Model Outcomes

Command:

```bash
python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_smoke.yaml
```

Mode: `dry_run`. This validates local-eval logging, parsing, scoring, and parquet output; it is not true model-performance evidence.

Outputs:

- `local_model_outcomes.parquet`
- `local_model_raw_outputs.jsonl`
- `local_model_errors.jsonl`
- `local_model_run_metadata.json`
- `m2_local_model_generation_memo.md`

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k_smoke | dry_run_model | 10 | 1.0000 | 0.0000 | 3.0000 | 0 |
| mmlu_smoke | dry_run_model | 10 | 1.0000 | 0.0000 | 1.0000 | 0 |

## Phase 2 Probe Features

Command:

```bash
python experiments/52_probe_collection.py --outcomes results/phase2/local_model_outcomes.parquet --output-dir results/phase2
```

This writes `probe_features.parquet` from local cheap-probe outputs without external API calls. The current dry-run validates the schema and logging path only.

Outputs:

- `probe_features.parquet`
- `m3_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_agreement | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- |
| local_answer_probe | dry_run_model | 20 | 20 | 1.0000 | 0.0020 | 0 |

## Phase 2 Probe Signal Analysis

Command:

```bash
python experiments/53_probe_signal_analysis.py --probe-features results/phase2/aligned_offline/aligned_probe_features.parquet --output-dir results/phase2 --state-targets results/phase2/aligned_offline/aligned_state_targets.csv --query-features results/phase2/aligned_offline/aligned_query_features.csv
```

M4 executed on aligned probe features and route-state targets.

Outputs:

- `table_probe_signal_analysis.csv`
- `fig_probe_signal_gain.pdf`
- `m4_probe_signal_analysis_memo.md`

| method | status | n_queries | n_train | n_test | state_prediction_accuracy | state_prediction_accuracy_ci_low | state_prediction_accuracy_ci_high | routing_utility | observability_gap_closed | mean_probe_cost_proxy | regret_prediction_auc | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| query_only_state_predictor | executed | 2318 | 1738 | 580 | 0.8724 | 0.8431 | 0.8992 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| probe_only_state_predictor | executed | 2318 | 1738 | 580 | 0.3603 | 0.3224 | 0.4017 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| query_plus_probe_state_predictor | executed | 2318 | 1738 | 580 | 0.8793 | 0.8534 | 0.9061 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| query_plus_knn_uncertainty_state_predictor | executed | 2318 | 1738 | 580 | 0.8724 | 0.8448 | 0.9000 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |
| query_plus_confidence_state_predictor | executed | 2318 | 1738 | 580 | 0.8724 | 0.8448 | 0.9000 |  |  | 0.0001 |  | State prediction only; routing utility requires a state-to-model utility table. |

## Phase 2 ProbeRoute++ Policy

M5 executed ProbeRoute++ policies through latent state beliefs with probe-cost accounting.

Outputs:

- `table_proberoute_policy.csv`
- `fig_gap_closed_vs_probe_cost.pdf`
- `m5_proberoute_policy_memo.md`

| policy | status | n_queries | mean_utility | mean_utility_ci_low | mean_utility_ci_high | mean_net_utility | mean_net_utility_ci_low | mean_net_utility_ci_high | mean_quality | mean_model_cost | mean_probe_cost_proxy | fraction_probed | mean_oracle_regret | observability_gap_closed | observability_gap_closed_ci_low | observability_gap_closed_ci_high | mean_latency_sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| never_probe | executed | 580 | 0.7431 | 0.7043 | 0.7802 | 0.7431 | 0.7034 | 0.7750 |  |  | 0.0000 | 0.0000 | 0.1534 | 0.0000 | -0.2584 | 0.2081 | 0.0000 | Routed through state belief and state-model utility. |
| always_probe | executed | 580 | 0.7448 | 0.7103 | 0.7776 | 0.7447 | 0.7120 | 0.7792 |  |  | 0.0001 | 1.0000 | 0.1518 | 0.0106 | -0.2029 | 0.2353 | 0.0000 | Routed through state belief and state-model utility. |
| entropy_threshold | executed | 580 | 0.7448 | 0.7103 | 0.7793 | 0.7448 | 0.7069 | 0.7785 |  |  | 0.0000 | 0.2121 | 0.1517 | 0.0111 | -0.2361 | 0.2305 | 0.0000 | Routed through state belief and state-model utility. |
| margin_threshold | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7103 | 0.7810 |  |  | 0.0000 | 0.0845 | 0.1517 | 0.0112 | -0.2135 | 0.2471 | 0.0000 | Routed through state belief and state-model utility. |
| voi_probe | executed | 580 | 0.7448 | 0.7069 | 0.7776 | 0.7448 | 0.7060 | 0.7785 |  |  | 0.0000 | 0.3138 | 0.1518 | 0.0110 | -0.2421 | 0.2304 | 0.0000 | Routed through state belief and state-model utility. |
| oracle_probe | executed | 580 | 0.7448 | 0.7060 | 0.7793 | 0.7448 | 0.7103 | 0.7776 |  |  | 0.0000 | 0.0017 | 0.1517 | 0.0112 | -0.2140 | 0.2247 | 0.0000 | Routed through state belief and state-model utility. |

## Phase 2 Active New-Model Calibration

Command:

```bash
python experiments/55_active_new_model_calibration.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2 --max-holdout-models 1 --r-values 1,2,4,8
```

Outputs:

- `table_active_new_model_calibration.csv`
- `fig_new_model_calibration_curve.pdf`
- `m6_active_new_model_calibration_memo.md`

Best rows:

| method | new_model_id | examples_per_label | new_model_evaluations | mean_utility | utility_ci_low | utility_ci_high | recovered_gap_vs_oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| active_route_state_calibration | Qwen3-8B | 4 | 61 | 0.7397 | 0.7155 | 0.7725 | 0.3158 |
| uniform_route_state_calibration | Qwen3-8B | 4 | 61 | 0.7379 | 0.7121 | 0.7707 | 0.3083 |
| dataset_stratified_calibration | Qwen3-8B | 2 | 31 | 0.7362 | 0.7121 | 0.7708 | 0.3008 |
| random_route_state_calibration | Qwen3-8B | 8 | 121 | 0.7345 | 0.7085 | 0.7716 | 0.2932 |
| embedding_cluster_calibration | Qwen3-8B | 4 | 61 | 0.7310 | 0.7059 | 0.7673 | 0.2782 |
| routecode_no_new_model | Qwen3-8B | 0 | 0 | 0.7190 | 0.6861 | 0.7475 | 0.2256 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 1 | 16 | 0.6069 | 0.5741 | 0.6544 | -0.2632 |

## Phase 2 Aligned Offline Inputs

Command:

```bash
python experiments/56_aligned_offline_probe_inputs.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2/aligned_offline
```

These artifacts make M4/M5 executable on aligned benchmark-derived route states. They are offline scaffolding, not true local probe evidence.

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

## Phase 2 Aligned Local Probes

Command:

```bash
python experiments/57_aligned_local_probe_collection.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2/aligned_local_probes --state-targets results/phase2/aligned_offline/aligned_state_targets.csv
```

This run uses a deterministic dry-run probe client over benchmark-aligned test queries. It validates aligned query selection, raw prompt/output logging, schema compatibility, and downstream M4 plumbing; it is not true local-model probe evidence.

Outputs:

- `results/phase2/aligned_local_probes/aligned_local_probe_features.parquet`
- `results/phase2/aligned_local_probes/aligned_local_probe_raw_outputs.jsonl`
- `results/phase2/aligned_local_probes/aligned_local_probe_errors.jsonl`
- `results/phase2/aligned_local_probes/aligned_local_probe_run_metadata.json`
- `results/phase2/aligned_local_probes/m8_aligned_local_probe_collection_memo.md`

Summary:

| probe_type | probe_model_id | rows | unique_queries | mean_self_confidence | mean_entropy_proxy | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_local_confidence_probe | dry_probe | 50 | 50 | 0.5358 | 0.4642 | 0.0040 | 0 |

## Phase 2 Aligned Local Probe Signal Check

Command:

```bash
python experiments/53_probe_signal_analysis.py --probe-features results/phase2/aligned_local_probes/aligned_local_probe_features.parquet --state-targets results/phase2/aligned_offline/aligned_state_targets.csv --query-features results/phase2/aligned_offline/aligned_query_features.csv --output-dir results/phase2/aligned_local_probes_eval
```

This is a separate M4 check over dry-run aligned local probe rows. The dry-run probe does not improve state prediction and should not be interpreted as a cheap-probe success result.

Outputs:

- `results/phase2/aligned_local_probes_eval/table_probe_signal_analysis.csv`
- `results/phase2/aligned_local_probes_eval/fig_probe_signal_gain.pdf`
- `results/phase2/aligned_local_probes_eval/m4_probe_signal_analysis_memo.md`

| method | state_prediction_accuracy | state_prediction_accuracy_ci_low | state_prediction_accuracy_ci_high | n_train | n_test |
| --- | --- | --- | --- | --- | --- |
| query_only_state_predictor | 0.3529 | 0.1176 | 0.5882 | 33 | 17 |
| probe_only_state_predictor | 0.2941 | 0.0588 | 0.5015 | 33 | 17 |
| query_plus_probe_state_predictor | 0.2353 | 0.0588 | 0.4118 | 33 | 17 |
| query_plus_knn_uncertainty_state_predictor | 0.3529 | 0.1176 | 0.5882 | 33 | 17 |
| query_plus_confidence_state_predictor | 0.2941 | 0.1176 | 0.5294 | 33 | 17 |

## Phase 2 Local Server Readiness

Command:

```bash
python experiments/58_local_server_readiness.py --config configs/phase2_local_server_readiness.yaml --output-dir results/phase2
```

At least one configured local model is blocked. This means true local Phase 2 runs should not be started until the local OpenAI-compatible endpoint is available.

Outputs:

- `table_local_server_readiness.csv`
- `m9_local_server_readiness_memo.md`

| check_id | status | base_url | model_id | models_endpoint_status | model_listed | completion_status | latency_sec | tokens_input | tokens_output | blocking_reasons | error_type | error_message | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_openai_server:Qwen3-8B | blocked | http://localhost:8000/v1 | Qwen3-8B | error | False | error | 2.0025 | 6 | 0 | completion_failed | URLError | <urlopen error timed out>; <urlopen error timed out> | 2026-06-17T02:54:30.628123+00:00 |
| local_openai_server:Qwen2.5-Coder-7B-Instruct | blocked | http://localhost:8000/v1 | Qwen2.5-Coder-7B-Instruct | error | False | error | 2.0024 | 6 | 0 | completion_failed | URLError | <urlopen error timed out>; <urlopen error timed out> | 2026-06-17T02:54:30.628123+00:00 |

## Phase 2 Exact Task Manifest

Command:

```bash
python experiments/59_exact_task_manifest.py --config configs/phase2_exact_task_manifest.yaml --output-dir results/phase2
```

This creates `local_exact_task_manifest.csv` for true local exact-scored math runs. It is a task substrate, not model-performance evidence.

Selection:

- Datasets requested: `aime, math500`.
- RouteCode split: `test`.
- Max queries: `200`.

Outputs:

- `local_exact_task_manifest.csv`
- `m10_exact_task_manifest_memo.md`

| dataset | task_type | routecode_split | rows | unique_queries |
| --- | --- | --- | --- | --- |
| aime | math | test | 14 | 14 |
| math500 | math | test | 104 | 104 |

## Phase 2 Manifest-Backed Local Dry Run

Command:

```bash
python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_exact_manifest_dryrun.yaml
```

This validates that the local generation runner can consume `results/phase2/local_exact_task_manifest.csv`. It uses `dry_run_model`, so it is logging and task-substrate evidence, not true model-performance evidence.

Outputs:

- `results/phase2/local_exact_manifest_dryrun/local_model_outcomes.parquet`
- `results/phase2/local_exact_manifest_dryrun/local_model_raw_outputs.jsonl`
- `results/phase2/local_exact_manifest_dryrun/local_model_errors.jsonl`
- `results/phase2/local_exact_manifest_dryrun/local_model_run_metadata.json`
- `results/phase2/local_exact_manifest_dryrun/m2_local_model_generation_memo.md`

| dataset | model_id | rows | mean_quality | errors |
| --- | --- | --- | --- | --- |
| aime | dry_run_model | 14 | 1.0000 | 0 |
| math500 | dry_run_model | 104 | 1.0000 | 0 |

## Phase 2 Exact Manifest Probe Collection

Command:

```bash
python experiments/60_exact_manifest_probe_collection.py --config configs/phase2_exact_manifest_probe_dryrun.yaml --output-dir results/phase2/exact_manifest_probes
```

This run uses a deterministic dry-run probe client over `results/phase2/local_exact_task_manifest.csv`. It validates exact-manifest probe logging and M4 plumbing; it is not true local-model probe evidence.

Outputs:

- `results/phase2/exact_manifest_probes/exact_manifest_probe_features.parquet`
- `results/phase2/exact_manifest_probes/exact_manifest_probe_raw_outputs.jsonl`
- `results/phase2/exact_manifest_probes/exact_manifest_probe_errors.jsonl`
- `results/phase2/exact_manifest_probes/exact_manifest_probe_run_metadata.json`
- `results/phase2/exact_manifest_probes/m11_exact_manifest_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_self_confidence | mean_entropy_proxy | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_local_confidence_probe | dry_probe | 118 | 118 | 0.4864 | 0.5136 | 0.0040 | 0 |

## Phase 2 Exact Manifest Probe Signal Check

Command:

```bash
python experiments/53_probe_signal_analysis.py --probe-features results/phase2/exact_manifest_probes/exact_manifest_probe_features.parquet --state-targets results/phase2/aligned_offline/aligned_state_targets.csv --query-features results/phase2/aligned_offline/aligned_query_features.csv --output-dir results/phase2/exact_manifest_probes_eval
```

This is a separate M4 check over the dry-run exact-manifest probe rows. The exact manifest contains 118 RouteCode test-split math queries, so this diagnostic uses an internal train/test split over those rows. The dry-run probe does not improve state prediction and should not be interpreted as a cheap-probe success result.

Outputs:

- `results/phase2/exact_manifest_probes_eval/table_probe_signal_analysis.csv`
- `results/phase2/exact_manifest_probes_eval/fig_probe_signal_gain.pdf`
- `results/phase2/exact_manifest_probes_eval/m4_probe_signal_analysis_memo.md`

| method | state_prediction_accuracy | state_prediction_accuracy_ci_low | state_prediction_accuracy_ci_high | n_train | n_test |
| --- | --- | --- | --- | --- | --- |
| query_only_state_predictor | 0.5641 | 0.4103 | 0.7179 | 79 | 39 |
| probe_only_state_predictor | 0.4103 | 0.2564 | 0.5641 | 79 | 39 |
| query_plus_probe_state_predictor | 0.5641 | 0.4103 | 0.7179 | 79 | 39 |
| query_plus_knn_uncertainty_state_predictor | 0.5641 | 0.4103 | 0.7179 | 79 | 39 |
| query_plus_confidence_state_predictor | 0.5641 | 0.4103 | 0.7179 | 79 | 39 |

## Phase 2 Active Calibration Replicates

Command:

```bash
python experiments/61_active_calibration_replicates.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2 --max-holdout-models 6 --seeds 0,1,2 --r-values 1,2,4,8
```

Outputs:

- `table_active_calibration_replicates.csv`
- `table_active_calibration_replicate_summary.csv`
- `table_active_calibration_active_vs_uniform_deltas.csv`
- `table_active_calibration_active_vs_random_deltas.csv`
- `table_active_calibration_active_vs_dataset_deltas.csv`
- `table_active_calibration_active_vs_embedding_deltas.csv`
- `m6_active_calibration_replicates_memo.md`

Replicate summary:

| method | new_model_id | examples_per_label | replicates | mean_utility_mean | mean_utility_std | mean_utility_min | mean_utility_max | recovered_gap_vs_oracle_mean | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6839 | 0.0342 | 0.6466 | 0.7138 | 0.0727 | 16.0000 | 16 | 16 |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.6793 | 0.0194 | 0.6672 | 0.7017 | 0.0526 | 32.0000 | 32 | 32 |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.7270 | 0.0155 | 0.7172 | 0.7448 | 0.2607 | 63.3333 | 62 | 64 |
| active_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.6730 | 0.0697 | 0.5983 | 0.7362 | 0.0251 | 126.0000 | 122 | 128 |
| active_route_state_calibration | Intern-S1-mini | 1 | 3 | 0.6575 | 0.0512 | 0.6034 | 0.7052 | -0.0426 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Intern-S1-mini | 2 | 3 | 0.7195 | 0.0070 | 0.7155 | 0.7276 | 0.2281 | 32.0000 | 32 | 32 |
| active_route_state_calibration | Intern-S1-mini | 4 | 3 | 0.7098 | 0.0318 | 0.6828 | 0.7448 | 0.1855 | 64.0000 | 64 | 64 |
| active_route_state_calibration | Intern-S1-mini | 8 | 3 | 0.7333 | 0.0078 | 0.7259 | 0.7414 | 0.2882 | 127.0000 | 126 | 128 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.6161 | 0.0202 | 0.5931 | 0.6310 | -0.2231 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7011 | 0.0297 | 0.6672 | 0.7224 | 0.1479 | 31.3333 | 30 | 32 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.6874 | 0.0552 | 0.6241 | 0.7259 | 0.0877 | 61.0000 | 58 | 64 |
| active_route_state_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7109 | 0.0101 | 0.7034 | 0.7224 | 0.1905 | 118.3333 | 113 | 125 |
| active_route_state_calibration | MiniCPM4.1-8B | 1 | 3 | 0.7305 | 0.0088 | 0.7207 | 0.7379 | 0.2757 | 16.0000 | 16 | 16 |
| active_route_state_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7305 | 0.0115 | 0.7172 | 0.7379 | 0.2757 | 31.3333 | 31 | 32 |
| active_route_state_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7253 | 0.0095 | 0.7155 | 0.7345 | 0.2531 | 61.3333 | 59 | 64 |
| active_route_state_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7379 | 0.0096 | 0.7293 | 0.7483 | 0.3083 | 119.3333 | 115 | 125 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6420 | 0.0248 | 0.6138 | 0.6603 | -0.1103 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.7057 | 0.0214 | 0.6810 | 0.7190 | 0.1679 | 31.6667 | 31 | 32 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7282 | 0.0173 | 0.7086 | 0.7414 | 0.2657 | 62.6667 | 61 | 64 |
| active_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7190 | 0.0135 | 0.7103 | 0.7345 | 0.2256 | 123.3333 | 121 | 127 |
| active_route_state_calibration | Qwen3-8B | 1 | 3 | 0.7046 | 0.0174 | 0.6845 | 0.7155 | 0.1629 | 16.0000 | 16 | 16 |
| active_route_state_calibration | Qwen3-8B | 2 | 3 | 0.7144 | 0.0263 | 0.6914 | 0.7431 | 0.2055 | 32.0000 | 32 | 32 |
| active_route_state_calibration | Qwen3-8B | 4 | 3 | 0.7282 | 0.0202 | 0.7052 | 0.7431 | 0.2657 | 63.3333 | 62 | 64 |
| active_route_state_calibration | Qwen3-8B | 8 | 3 | 0.7149 | 0.0275 | 0.6845 | 0.7379 | 0.2080 | 126.0000 | 122 | 128 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.7259 | 0.0194 | 0.7138 | 0.7483 | 0.2556 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.7253 | 0.0127 | 0.7155 | 0.7397 | 0.2531 | 32.0000 | 32 | 32 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.7132 | 0.0268 | 0.6914 | 0.7431 | 0.2005 | 63.3333 | 62 | 64 |
| dataset_stratified_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.7264 | 0.0127 | 0.7121 | 0.7362 | 0.2581 | 126.0000 | 122 | 128 |
| dataset_stratified_calibration | Intern-S1-mini | 1 | 3 | 0.6960 | 0.0354 | 0.6569 | 0.7259 | 0.1253 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Intern-S1-mini | 2 | 3 | 0.7195 | 0.0286 | 0.6897 | 0.7466 | 0.2281 | 32.0000 | 32 | 32 |
| dataset_stratified_calibration | Intern-S1-mini | 4 | 3 | 0.7374 | 0.0088 | 0.7276 | 0.7448 | 0.3058 | 64.0000 | 64 | 64 |
| dataset_stratified_calibration | Intern-S1-mini | 8 | 3 | 0.7339 | 0.0078 | 0.7259 | 0.7414 | 0.2907 | 127.0000 | 126 | 128 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.7057 | 0.0177 | 0.6862 | 0.7207 | 0.1679 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7086 | 0.0086 | 0.7000 | 0.7172 | 0.1805 | 31.3333 | 30 | 32 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7138 | 0.0113 | 0.7034 | 0.7259 | 0.2030 | 61.0000 | 58 | 64 |
| dataset_stratified_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7144 | 0.0277 | 0.6828 | 0.7345 | 0.2055 | 118.3333 | 113 | 125 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 1 | 3 | 0.6983 | 0.0702 | 0.6172 | 0.7397 | 0.1353 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7052 | 0.0352 | 0.6655 | 0.7328 | 0.1654 | 31.3333 | 31 | 32 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7385 | 0.0055 | 0.7345 | 0.7448 | 0.3108 | 61.3333 | 59 | 64 |
| dataset_stratified_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7420 | 0.0081 | 0.7328 | 0.7483 | 0.3258 | 119.3333 | 115 | 125 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6908 | 0.0145 | 0.6741 | 0.7000 | 0.1028 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.7109 | 0.0115 | 0.6983 | 0.7207 | 0.1905 | 31.6667 | 31 | 32 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7190 | 0.0086 | 0.7103 | 0.7276 | 0.2256 | 62.6667 | 61 | 64 |
| dataset_stratified_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7316 | 0.0115 | 0.7241 | 0.7448 | 0.2807 | 123.3333 | 121 | 127 |
| dataset_stratified_calibration | Qwen3-8B | 1 | 3 | 0.6960 | 0.0132 | 0.6845 | 0.7103 | 0.1253 | 16.0000 | 16 | 16 |
| dataset_stratified_calibration | Qwen3-8B | 2 | 3 | 0.7282 | 0.0261 | 0.6983 | 0.7466 | 0.2657 | 32.0000 | 32 | 32 |
| dataset_stratified_calibration | Qwen3-8B | 4 | 3 | 0.7034 | 0.0170 | 0.6897 | 0.7224 | 0.1579 | 63.3333 | 62 | 64 |
| dataset_stratified_calibration | Qwen3-8B | 8 | 3 | 0.7282 | 0.0125 | 0.7138 | 0.7362 | 0.2657 | 126.0000 | 122 | 128 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 32.0000 | 32 | 32 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 63.3333 | 62 | 64 |
| direct_retraining_budgeted_logistic_active_budget | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 126.0000 | 122 | 128 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 1 | 3 | 0.6810 | 0.0000 | 0.6810 | 0.6810 | 0.0602 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 2 | 3 | 0.6810 | 0.0000 | 0.6810 | 0.6810 | 0.0602 | 32.0000 | 32 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 4 | 3 | 0.6810 | 0.0000 | 0.6810 | 0.6810 | 0.0602 | 64.0000 | 64 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Intern-S1-mini | 8 | 3 | 0.6805 | 0.0010 | 0.6793 | 0.6810 | 0.0576 | 127.0000 | 126 | 128 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 1 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 2 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 31.3333 | 30 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 4 | 3 | 0.6759 | 0.0000 | 0.6759 | 0.6759 | 0.0376 | 61.0000 | 58 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Llama-3.1-8B-Instruct | 8 | 3 | 0.6770 | 0.0010 | 0.6759 | 0.6776 | 0.0426 | 118.3333 | 113 | 125 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 1 | 3 | 0.6690 | 0.0000 | 0.6690 | 0.6690 | 0.0075 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 2 | 3 | 0.6678 | 0.0020 | 0.6655 | 0.6690 | 0.0025 | 31.3333 | 31 | 32 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 4 | 3 | 0.6678 | 0.0020 | 0.6655 | 0.6690 | 0.0025 | 61.3333 | 59 | 64 |
| direct_retraining_budgeted_logistic_active_budget | MiniCPM4.1-8B | 8 | 3 | 0.6667 | 0.0020 | 0.6655 | 0.6690 | -0.0025 | 119.3333 | 115 | 125 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6667 | 0.0010 | 0.6655 | 0.6672 | -0.0025 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.6667 | 0.0010 | 0.6655 | 0.6672 | -0.0025 | 31.6667 | 31 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.6661 | 0.0010 | 0.6655 | 0.6672 | -0.0050 | 62.6667 | 61 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.6655 | 0.0034 | 0.6621 | 0.6690 | -0.0075 | 123.3333 | 121 | 127 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 1 | 3 | 0.6069 | 0.0000 | 0.6069 | 0.6069 | -0.2632 | 16.0000 | 16 | 16 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 2 | 3 | 0.6069 | 0.0000 | 0.6069 | 0.6069 | -0.2632 | 32.0000 | 32 | 32 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 4 | 3 | 0.6052 | 0.0030 | 0.6017 | 0.6069 | -0.2707 | 63.3333 | 62 | 64 |
| direct_retraining_budgeted_logistic_active_budget | Qwen3-8B | 8 | 3 | 0.6052 | 0.0030 | 0.6017 | 0.6069 | -0.2707 | 126.0000 | 122 | 128 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6856 | 0.0407 | 0.6431 | 0.7241 | 0.0802 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.6816 | 0.0259 | 0.6569 | 0.7086 | 0.0627 | 32.0000 | 32 | 32 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.6920 | 0.0313 | 0.6586 | 0.7207 | 0.1078 | 63.3333 | 62 | 64 |
| embedding_cluster_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.6730 | 0.0369 | 0.6500 | 0.7155 | 0.0251 | 126.0000 | 122 | 128 |
| embedding_cluster_calibration | Intern-S1-mini | 1 | 3 | 0.6971 | 0.0339 | 0.6759 | 0.7362 | 0.1303 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Intern-S1-mini | 2 | 3 | 0.7155 | 0.0299 | 0.6828 | 0.7414 | 0.2105 | 32.0000 | 32 | 32 |
| embedding_cluster_calibration | Intern-S1-mini | 4 | 3 | 0.7368 | 0.0098 | 0.7259 | 0.7448 | 0.3033 | 64.0000 | 64 | 64 |
| embedding_cluster_calibration | Intern-S1-mini | 8 | 3 | 0.7224 | 0.0149 | 0.7052 | 0.7310 | 0.2406 | 127.0000 | 126 | 128 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.6954 | 0.0145 | 0.6862 | 0.7121 | 0.1228 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7201 | 0.0140 | 0.7103 | 0.7362 | 0.2306 | 31.3333 | 30 | 32 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7011 | 0.0252 | 0.6741 | 0.7241 | 0.1479 | 61.0000 | 58 | 64 |
| embedding_cluster_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7144 | 0.0078 | 0.7069 | 0.7224 | 0.2055 | 118.3333 | 113 | 125 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 1 | 3 | 0.7230 | 0.0183 | 0.7034 | 0.7397 | 0.2431 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7213 | 0.0207 | 0.7000 | 0.7414 | 0.2356 | 31.3333 | 31 | 32 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7402 | 0.0112 | 0.7293 | 0.7517 | 0.3183 | 61.3333 | 59 | 64 |
| embedding_cluster_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7379 | 0.0034 | 0.7345 | 0.7414 | 0.3083 | 119.3333 | 115 | 125 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.7086 | 0.0130 | 0.6966 | 0.7224 | 0.1805 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.7069 | 0.0137 | 0.6966 | 0.7224 | 0.1729 | 31.6667 | 31 | 32 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7172 | 0.0124 | 0.7034 | 0.7276 | 0.2180 | 62.6667 | 61 | 64 |
| embedding_cluster_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7213 | 0.0277 | 0.6897 | 0.7414 | 0.2356 | 123.3333 | 121 | 127 |
| embedding_cluster_calibration | Qwen3-8B | 1 | 3 | 0.6822 | 0.0181 | 0.6638 | 0.7000 | 0.0652 | 16.0000 | 16 | 16 |
| embedding_cluster_calibration | Qwen3-8B | 2 | 3 | 0.7339 | 0.0105 | 0.7224 | 0.7431 | 0.2907 | 32.0000 | 32 | 32 |
| embedding_cluster_calibration | Qwen3-8B | 4 | 3 | 0.7230 | 0.0177 | 0.7034 | 0.7379 | 0.2431 | 63.3333 | 62 | 64 |
| embedding_cluster_calibration | Qwen3-8B | 8 | 3 | 0.7218 | 0.0131 | 0.7069 | 0.7310 | 0.2381 | 126.0000 | 122 | 128 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.7155 | 0.0290 | 0.6931 | 0.7483 | 0.2105 | 16.0000 | 16 | 16 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.7029 | 0.0413 | 0.6552 | 0.7276 | 0.1554 | 32.0000 | 32 | 32 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.7023 | 0.0160 | 0.6845 | 0.7155 | 0.1529 | 63.3333 | 62 | 64 |
| random_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.7276 | 0.0199 | 0.7052 | 0.7431 | 0.2632 | 126.0000 | 122 | 128 |
| random_route_state_calibration | Intern-S1-mini | 1 | 3 | 0.7345 | 0.0119 | 0.7276 | 0.7483 | 0.2932 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Intern-S1-mini | 2 | 3 | 0.7368 | 0.0156 | 0.7190 | 0.7483 | 0.3033 | 32.0000 | 32 | 32 |
| random_route_state_calibration | Intern-S1-mini | 4 | 3 | 0.7362 | 0.0062 | 0.7293 | 0.7414 | 0.3008 | 64.0000 | 64 | 64 |
| random_route_state_calibration | Intern-S1-mini | 8 | 3 | 0.7414 | 0.0034 | 0.7379 | 0.7448 | 0.3233 | 127.0000 | 126 | 128 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.6822 | 0.0101 | 0.6707 | 0.6897 | 0.0652 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.7132 | 0.0265 | 0.6828 | 0.7310 | 0.2005 | 31.3333 | 30 | 32 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7069 | 0.0062 | 0.7017 | 0.7138 | 0.1729 | 61.0000 | 58 | 64 |
| random_route_state_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7069 | 0.0141 | 0.6948 | 0.7224 | 0.1729 | 118.3333 | 113 | 125 |
| random_route_state_calibration | MiniCPM4.1-8B | 1 | 3 | 0.7293 | 0.0105 | 0.7172 | 0.7362 | 0.2707 | 16.0000 | 16 | 16 |
| random_route_state_calibration | MiniCPM4.1-8B | 2 | 3 | 0.7305 | 0.0174 | 0.7103 | 0.7414 | 0.2757 | 31.3333 | 31 | 32 |
| random_route_state_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7287 | 0.0199 | 0.7172 | 0.7517 | 0.2682 | 61.3333 | 59 | 64 |
| random_route_state_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7420 | 0.0098 | 0.7310 | 0.7500 | 0.3258 | 119.3333 | 115 | 125 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.7000 | 0.0069 | 0.6931 | 0.7069 | 0.1429 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.6994 | 0.0263 | 0.6707 | 0.7224 | 0.1404 | 31.6667 | 31 | 32 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.7213 | 0.0070 | 0.7138 | 0.7276 | 0.2356 | 62.6667 | 61 | 64 |
| random_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7299 | 0.0095 | 0.7207 | 0.7397 | 0.2732 | 123.3333 | 121 | 127 |
| random_route_state_calibration | Qwen3-8B | 1 | 3 | 0.7259 | 0.0225 | 0.7000 | 0.7414 | 0.2556 | 16.0000 | 16 | 16 |
| random_route_state_calibration | Qwen3-8B | 2 | 3 | 0.7195 | 0.0144 | 0.7034 | 0.7310 | 0.2281 | 32.0000 | 32 | 32 |
| random_route_state_calibration | Qwen3-8B | 4 | 3 | 0.7310 | 0.0179 | 0.7103 | 0.7414 | 0.2782 | 63.3333 | 62 | 64 |
| random_route_state_calibration | Qwen3-8B | 8 | 3 | 0.7287 | 0.0156 | 0.7121 | 0.7431 | 0.2682 | 126.0000 | 122 | 128 |
| routecode_no_new_model | DeepSeek-R1-Distill-Qwen-7B | 0 | 3 | 0.7374 | 0.0122 | 0.7241 | 0.7483 | 0.3058 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Intern-S1-mini | 0 | 3 | 0.7414 | 0.0052 | 0.7362 | 0.7466 | 0.3233 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Llama-3.1-8B-Instruct | 0 | 3 | 0.7299 | 0.0070 | 0.7259 | 0.7379 | 0.2732 | 0.0000 | 0 | 0 |
| routecode_no_new_model | MiniCPM4.1-8B | 0 | 3 | 0.7425 | 0.0072 | 0.7345 | 0.7483 | 0.3283 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Qwen2.5-Coder-7B-Instruct | 0 | 3 | 0.6828 | 0.0030 | 0.6793 | 0.6845 | 0.0677 | 0.0000 | 0 | 0 |
| routecode_no_new_model | Qwen3-8B | 0 | 3 | 0.7305 | 0.0072 | 0.7224 | 0.7362 | 0.2757 | 0.0000 | 0 | 0 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.6241 | 0.0268 | 0.5966 | 0.6500 | -0.1880 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | 0.7040 | 0.0127 | 0.6897 | 0.7138 | 0.1604 | 32.0000 | 32 | 32 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.6948 | 0.0210 | 0.6707 | 0.7086 | 0.1203 | 63.3333 | 62 | 64 |
| uniform_route_state_calibration | DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.7213 | 0.0010 | 0.7207 | 0.7224 | 0.2356 | 126.0000 | 122 | 128 |
| uniform_route_state_calibration | Intern-S1-mini | 1 | 3 | 0.6770 | 0.0742 | 0.5914 | 0.7207 | 0.0426 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Intern-S1-mini | 2 | 3 | 0.7000 | 0.0192 | 0.6793 | 0.7172 | 0.1429 | 32.0000 | 32 | 32 |
| uniform_route_state_calibration | Intern-S1-mini | 4 | 3 | 0.6943 | 0.0140 | 0.6845 | 0.7103 | 0.1178 | 64.0000 | 64 | 64 |
| uniform_route_state_calibration | Intern-S1-mini | 8 | 3 | 0.7264 | 0.0147 | 0.7121 | 0.7414 | 0.2581 | 127.0000 | 126 | 128 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 1 | 3 | 0.5862 | 0.0340 | 0.5483 | 0.6138 | -0.3534 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 2 | 3 | 0.6925 | 0.0548 | 0.6293 | 0.7259 | 0.1103 | 31.3333 | 30 | 32 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 4 | 3 | 0.7098 | 0.0020 | 0.7086 | 0.7121 | 0.1855 | 61.0000 | 58 | 64 |
| uniform_route_state_calibration | Llama-3.1-8B-Instruct | 8 | 3 | 0.7259 | 0.0121 | 0.7138 | 0.7379 | 0.2556 | 118.3333 | 113 | 125 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 1 | 3 | 0.6937 | 0.0426 | 0.6466 | 0.7293 | 0.1153 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 2 | 3 | 0.6672 | 0.0553 | 0.6345 | 0.7310 | 0.0000 | 31.3333 | 31 | 32 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 4 | 3 | 0.7276 | 0.0243 | 0.7017 | 0.7500 | 0.2632 | 61.3333 | 59 | 64 |
| uniform_route_state_calibration | MiniCPM4.1-8B | 8 | 3 | 0.7328 | 0.0017 | 0.7310 | 0.7345 | 0.2857 | 119.3333 | 115 | 125 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.6420 | 0.0316 | 0.6172 | 0.6776 | -0.1103 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.6672 | 0.0466 | 0.6224 | 0.7155 | 0.0000 | 31.6667 | 31 | 32 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.6822 | 0.0251 | 0.6586 | 0.7086 | 0.0652 | 62.6667 | 61 | 64 |
| uniform_route_state_calibration | Qwen2.5-Coder-7B-Instruct | 8 | 3 | 0.7241 | 0.0216 | 0.7034 | 0.7466 | 0.2481 | 123.3333 | 121 | 127 |
| uniform_route_state_calibration | Qwen3-8B | 1 | 3 | 0.7138 | 0.0216 | 0.6914 | 0.7345 | 0.2030 | 16.0000 | 16 | 16 |
| uniform_route_state_calibration | Qwen3-8B | 2 | 3 | 0.7339 | 0.0208 | 0.7121 | 0.7534 | 0.2907 | 32.0000 | 32 | 32 |
| uniform_route_state_calibration | Qwen3-8B | 4 | 3 | 0.7282 | 0.0259 | 0.6983 | 0.7431 | 0.2657 | 63.3333 | 62 | 64 |
| uniform_route_state_calibration | Qwen3-8B | 8 | 3 | 0.7149 | 0.0131 | 0.7000 | 0.7241 | 0.2080 | 126.0000 | 122 | 128 |

Active vs uniform deltas:

| new_model_id | examples_per_label | replicates | active_minus_uniform_mean_utility_mean | active_minus_uniform_mean_utility_std | active_minus_uniform_mean_utility_min | active_minus_uniform_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | 0.0598 | 0.0605 | -0.0034 | 0.1172 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0247 | 0.0320 | -0.0466 | 0.0121 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0322 | 0.0364 | 0.0086 | 0.0741 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | -0.0483 | 0.0689 | -0.1224 | 0.0138 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0195 | 0.0940 | -0.1155 | 0.0724 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | 0.0195 | 0.0258 | -0.0017 | 0.0483 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | 0.0155 | 0.0389 | -0.0086 | 0.0603 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | 0.0069 | 0.0119 | 0.0000 | 0.0207 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | 0.0299 | 0.0484 | -0.0207 | 0.0759 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | 0.0086 | 0.0261 | -0.0121 | 0.0379 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0224 | 0.0544 | -0.0845 | 0.0172 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | -0.0149 | 0.0043 | -0.0190 | -0.0103 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0368 | 0.0506 | -0.0086 | 0.0914 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | 0.0632 | 0.0667 | -0.0138 | 0.1017 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0023 | 0.0229 | -0.0155 | 0.0241 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | 0.0052 | 0.0079 | -0.0017 | 0.0138 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | 0.0000 | 0.0192 | -0.0172 | 0.0207 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.0385 | 0.0493 | 0.0034 | 0.0948 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0460 | 0.0319 | 0.0259 | 0.0828 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0052 | 0.0255 | -0.0345 | 0.0121 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | -0.0092 | 0.0088 | -0.0190 | -0.0017 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0195 | 0.0087 | -0.0276 | -0.0103 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | 0.0000 | 0.0069 | -0.0069 | 0.0069 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0000 | 0.0344 | -0.0397 | 0.0224 | 126.0000 | 122 | 128 |

Active vs random deltas:

| new_model_id | examples_per_label | replicates | active_minus_random_mean_utility_mean | active_minus_random_mean_utility_std | active_minus_random_mean_utility_min | active_minus_random_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | -0.0316 | 0.0286 | -0.0586 | -0.0017 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0236 | 0.0607 | -0.0603 | 0.0466 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0247 | 0.0111 | 0.0121 | 0.0328 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | -0.0546 | 0.0789 | -0.1448 | 0.0017 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0770 | 0.0513 | -0.1241 | -0.0224 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | -0.0172 | 0.0225 | -0.0328 | 0.0086 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | -0.0264 | 0.0264 | -0.0466 | 0.0034 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | -0.0080 | 0.0043 | -0.0121 | -0.0034 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | -0.0661 | 0.0267 | -0.0966 | -0.0466 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | -0.0121 | 0.0449 | -0.0586 | 0.0310 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0195 | 0.0533 | -0.0810 | 0.0121 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | 0.0040 | 0.0177 | -0.0155 | 0.0190 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0011 | 0.0147 | -0.0138 | 0.0155 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | -0.0000 | 0.0250 | -0.0241 | 0.0259 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0034 | 0.0287 | -0.0362 | 0.0172 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | -0.0040 | 0.0088 | -0.0138 | 0.0034 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | -0.0580 | 0.0184 | -0.0793 | -0.0466 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | 0.0063 | 0.0451 | -0.0414 | 0.0483 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0069 | 0.0105 | -0.0052 | 0.0138 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0109 | 0.0164 | -0.0276 | 0.0052 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | -0.0213 | 0.0061 | -0.0276 | -0.0155 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0052 | 0.0306 | -0.0397 | 0.0190 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | -0.0029 | 0.0345 | -0.0362 | 0.0328 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0138 | 0.0182 | -0.0276 | 0.0069 | 126.0000 | 122 | 128 |

Active vs dataset deltas:

| new_model_id | examples_per_label | replicates | active_minus_dataset_mean_utility_mean | active_minus_dataset_mean_utility_std | active_minus_dataset_mean_utility_min | active_minus_dataset_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | -0.0420 | 0.0242 | -0.0690 | -0.0224 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0460 | 0.0292 | -0.0707 | -0.0138 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0138 | 0.0121 | 0.0017 | 0.0259 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | -0.0534 | 0.0701 | -0.1328 | 0.0000 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0385 | 0.0166 | -0.0534 | -0.0207 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | 0.0000 | 0.0288 | -0.0310 | 0.0259 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | -0.0276 | 0.0284 | -0.0448 | 0.0052 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | -0.0006 | 0.0078 | -0.0086 | 0.0069 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | -0.0897 | 0.0316 | -0.1172 | -0.0552 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | -0.0075 | 0.0219 | -0.0328 | 0.0052 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0264 | 0.0562 | -0.0879 | 0.0224 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | -0.0034 | 0.0244 | -0.0224 | 0.0241 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0322 | 0.0618 | -0.0069 | 0.1034 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | 0.0253 | 0.0394 | 0.0000 | 0.0707 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0132 | 0.0065 | -0.0207 | -0.0086 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | -0.0040 | 0.0078 | -0.0121 | 0.0034 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | -0.0489 | 0.0320 | -0.0845 | -0.0224 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | -0.0052 | 0.0108 | -0.0172 | 0.0034 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0092 | 0.0208 | -0.0103 | 0.0310 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0126 | 0.0207 | -0.0328 | 0.0086 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | 0.0086 | 0.0121 | 0.0000 | 0.0224 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0138 | 0.0150 | -0.0310 | -0.0034 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | 0.0247 | 0.0117 | 0.0155 | 0.0379 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0132 | 0.0334 | -0.0517 | 0.0086 | 126.0000 | 122 | 128 |

Active vs embedding deltas:

| new_model_id | examples_per_label | replicates | active_minus_embedding_mean_utility_mean | active_minus_embedding_mean_utility_std | active_minus_embedding_mean_utility_min | active_minus_embedding_mean_utility_max | new_model_evaluations_mean | new_model_evaluations_min | new_model_evaluations_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-7B | 1 | 3 | -0.0017 | 0.0629 | -0.0431 | 0.0707 | 16.0000 | 16 | 16 |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 3 | -0.0023 | 0.0431 | -0.0397 | 0.0448 | 32.0000 | 32 | 32 |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 3 | 0.0351 | 0.0204 | 0.0224 | 0.0586 | 63.3333 | 62 | 64 |
| DeepSeek-R1-Distill-Qwen-7B | 8 | 3 | 0.0000 | 0.0756 | -0.0552 | 0.0862 | 126.0000 | 122 | 128 |
| Intern-S1-mini | 1 | 3 | -0.0397 | 0.0294 | -0.0724 | -0.0155 | 16.0000 | 16 | 16 |
| Intern-S1-mini | 2 | 3 | 0.0040 | 0.0366 | -0.0259 | 0.0448 | 32.0000 | 32 | 32 |
| Intern-S1-mini | 4 | 3 | -0.0270 | 0.0286 | -0.0569 | 0.0000 | 64.0000 | 64 | 64 |
| Intern-S1-mini | 8 | 3 | 0.0109 | 0.0222 | -0.0052 | 0.0362 | 127.0000 | 126 | 128 |
| Llama-3.1-8B-Instruct | 1 | 3 | -0.0793 | 0.0147 | -0.0931 | -0.0638 | 16.0000 | 16 | 16 |
| Llama-3.1-8B-Instruct | 2 | 3 | -0.0190 | 0.0254 | -0.0466 | 0.0034 | 31.3333 | 30 | 32 |
| Llama-3.1-8B-Instruct | 4 | 3 | -0.0138 | 0.0610 | -0.0810 | 0.0379 | 61.0000 | 58 | 64 |
| Llama-3.1-8B-Instruct | 8 | 3 | -0.0034 | 0.0121 | -0.0155 | 0.0086 | 118.3333 | 113 | 125 |
| MiniCPM4.1-8B | 1 | 3 | 0.0075 | 0.0190 | -0.0052 | 0.0293 | 16.0000 | 16 | 16 |
| MiniCPM4.1-8B | 2 | 3 | 0.0092 | 0.0249 | -0.0052 | 0.0379 | 31.3333 | 31 | 32 |
| MiniCPM4.1-8B | 4 | 3 | -0.0149 | 0.0105 | -0.0241 | -0.0034 | 61.3333 | 59 | 64 |
| MiniCPM4.1-8B | 8 | 3 | 0.0000 | 0.0079 | -0.0086 | 0.0069 | 119.3333 | 115 | 125 |
| Qwen2.5-Coder-7B-Instruct | 1 | 3 | -0.0667 | 0.0287 | -0.0931 | -0.0362 | 16.0000 | 16 | 16 |
| Qwen2.5-Coder-7B-Instruct | 2 | 3 | -0.0011 | 0.0218 | -0.0207 | 0.0224 | 31.6667 | 31 | 32 |
| Qwen2.5-Coder-7B-Instruct | 4 | 3 | 0.0109 | 0.0252 | -0.0121 | 0.0379 | 62.6667 | 61 | 64 |
| Qwen2.5-Coder-7B-Instruct | 8 | 3 | -0.0023 | 0.0252 | -0.0293 | 0.0207 | 123.3333 | 121 | 127 |
| Qwen3-8B | 1 | 3 | 0.0224 | 0.0096 | 0.0138 | 0.0328 | 16.0000 | 16 | 16 |
| Qwen3-8B | 2 | 3 | -0.0195 | 0.0230 | -0.0345 | 0.0069 | 32.0000 | 32 | 32 |
| Qwen3-8B | 4 | 3 | 0.0052 | 0.0340 | -0.0328 | 0.0328 | 63.3333 | 62 | 64 |
| Qwen3-8B | 8 | 3 | -0.0069 | 0.0147 | -0.0224 | 0.0069 | 126.0000 | 122 | 128 |

Interpretation:

Across paired active-vs-uniform rows, active calibration has mean utility delta `0.0082` over `72` pairs (`35` positive, `32` negative, `5` tied).

Across paired active-vs-random rows, active calibration has mean utility delta `-0.0172` over `72` pairs (`27` positive, `45` negative, `0` tied).

Across paired active-vs-dataset rows, active calibration has mean utility delta `-0.0138` over `72` pairs (`26` positive, `41` negative, `5` tied).

Across paired active-vs-embedding rows, active calibration has mean utility delta `-0.0080` over `72` pairs (`29` positive, `42` negative, `1` tied).

## Phase 2 Active Calibration Sensitivity

Command:

```bash
python experiments/62_active_calibration_sensitivity.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2 --max-holdout-models 3 --k-values 8,16,32 --alpha-values 3.0 --r-values 1,4,8 --seeds 0,1
```

Outputs:

- `table_active_calibration_sensitivity.csv`
- `table_active_calibration_sensitivity_summary.csv`
- `table_active_calibration_sensitivity_deltas.csv`
- `m7_active_calibration_sensitivity_memo.md`

Sensitivity delta summary:

| sensitivity_name | sensitivity_k | sensitivity_alpha | baseline | paired_rows | active_minus_baseline_mean | active_minus_baseline_std | active_minus_baseline_min | active_minus_baseline_max | positive | negative | tied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k_16_alpha_3p0 | 16 | 3.0000 | uniform_route_state_calibration | 18 | -0.0232 | 0.0506 | -0.1448 | 0.0638 | 5 | 11 | 2 |
| k_32_alpha_3p0 | 32 | 3.0000 | uniform_route_state_calibration | 18 | -0.0128 | 0.0402 | -0.0897 | 0.0517 | 7 | 11 | 0 |
| k_8_alpha_3p0 | 8 | 3.0000 | uniform_route_state_calibration | 18 | -0.0206 | 0.0543 | -0.0897 | 0.1190 | 3 | 14 | 1 |
| k_16_alpha_3p0 | 16 | 3.0000 | random_route_state_calibration | 18 | -0.0098 | 0.0597 | -0.1379 | 0.0966 | 9 | 9 | 0 |
| k_32_alpha_3p0 | 32 | 3.0000 | random_route_state_calibration | 18 | -0.0130 | 0.0352 | -0.1207 | 0.0241 | 7 | 11 | 0 |
| k_8_alpha_3p0 | 8 | 3.0000 | random_route_state_calibration | 18 | -0.0026 | 0.0711 | -0.1224 | 0.1793 | 7 | 11 | 0 |
| k_16_alpha_3p0 | 16 | 3.0000 | dataset_stratified_calibration | 18 | -0.0259 | 0.0441 | -0.1431 | 0.0414 | 6 | 12 | 0 |
| k_32_alpha_3p0 | 32 | 3.0000 | dataset_stratified_calibration | 18 | -0.0313 | 0.0346 | -0.0810 | 0.0190 | 5 | 13 | 0 |
| k_8_alpha_3p0 | 8 | 3.0000 | dataset_stratified_calibration | 18 | -0.0310 | 0.0734 | -0.2069 | 0.0655 | 6 | 10 | 2 |
| k_16_alpha_3p0 | 16 | 3.0000 | embedding_cluster_calibration | 18 | -0.0286 | 0.0373 | -0.1086 | 0.0259 | 4 | 14 | 0 |
| k_32_alpha_3p0 | 32 | 3.0000 | embedding_cluster_calibration | 18 | -0.0299 | 0.0527 | -0.1172 | 0.1069 | 4 | 13 | 1 |
| k_8_alpha_3p0 | 8 | 3.0000 | embedding_cluster_calibration | 18 | -0.0354 | 0.0700 | -0.1879 | 0.0897 | 5 | 11 | 2 |

Best active rows:

| sensitivity_name | sensitivity_k | sensitivity_alpha | new_model_id | examples_per_label | replicates | mean_utility_mean | mean_utility_std | new_model_evaluations_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k_8_alpha_3p0 | 8 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 4 | 2 | 0.7629 | 0.0037 | 32.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | Qwen2.5-Coder-7B-Instruct | 4 | 2 | 0.7603 | 0.0024 | 63.0000 |
| k_8_alpha_3p0 | 8 | 3.0000 | Qwen3-8B | 4 | 2 | 0.7534 | 0.0195 | 32.0000 |
| k_8_alpha_3p0 | 8 | 3.0000 | Qwen3-8B | 8 | 2 | 0.7491 | 0.0110 | 64.0000 |
| k_32_alpha_3p0 | 32 | 3.0000 | Qwen3-8B | 8 | 2 | 0.7474 | 0.0061 | 243.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 4 | 2 | 0.7431 | 0.0098 | 64.0000 |
| k_32_alpha_3p0 | 32 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 8 | 2 | 0.7431 | 0.0122 | 241.5000 |
| k_32_alpha_3p0 | 32 | 3.0000 | Qwen2.5-Coder-7B-Instruct | 8 | 2 | 0.7422 | 0.0158 | 235.0000 |
| k_8_alpha_3p0 | 8 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 8 | 2 | 0.7388 | 0.0305 | 64.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | Qwen3-8B | 8 | 2 | 0.7336 | 0.0085 | 125.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | Qwen2.5-Coder-7B-Instruct | 8 | 2 | 0.7310 | 0.0390 | 125.0000 |
| k_16_alpha_3p0 | 16 | 3.0000 | DeepSeek-R1-Distill-Qwen-7B | 8 | 2 | 0.7276 | 0.0536 | 125.0000 |

Interpretation:

Across active-vs-random sensitivity cells, active calibration has mean cell delta `-0.0085` over `3` cells (`0` positive, `3` negative, `0` tied).

## Phase 2 Probe Cost Sensitivity

Command:

```bash
python experiments/63_probe_cost_sensitivity.py --output-dir results/phase2 --before-beliefs results/phase2/aligned_offline/aligned_before_beliefs.csv --after-beliefs results/phase2/aligned_offline/aligned_after_beliefs.csv --state-model-utility results/phase2/aligned_offline/aligned_state_model_utility.csv --query-model-utility results/phase2/aligned_offline/aligned_query_model_utility.csv --probe-cost results/phase2/aligned_offline/aligned_probe_cost.csv --predicted-gain results/phase2/aligned_offline/aligned_predicted_gain.csv --probe-cost-multipliers 0.0,0.5,1.0,2.0,5.0,10.0,50.0,100.0
```

Outputs:

- `table_probe_cost_sensitivity.csv`
- `table_probe_cost_sensitivity_summary.csv`
- `fig_probe_cost_sensitivity.pdf`
- `m7_probe_cost_sensitivity_memo.md`

Summary:

| probe_cost_multiplier | n_queries | best_policy_by_mean_net_utility | best_mean_net_utility | never_probe_mean_net_utility | never_probe_fraction_probed | never_probe_gap_closed | always_probe_mean_net_utility | always_probe_fraction_probed | always_probe_gap_closed | entropy_threshold_mean_net_utility | entropy_threshold_fraction_probed | entropy_threshold_gap_closed | margin_threshold_mean_net_utility | margin_threshold_fraction_probed | margin_threshold_gap_closed | voi_probe_mean_net_utility | voi_probe_fraction_probed | voi_probe_gap_closed | oracle_probe_mean_net_utility | oracle_probe_fraction_probed | oracle_probe_gap_closed | voi_minus_never_mean_net_utility | voi_minus_best_threshold_mean_net_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 580 | always_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7448 | 1.0000 | 0.0112 | 0.7448 | 0.2121 | 0.0112 | 0.7448 | 0.0845 | 0.0112 | 0.7448 | 0.5086 | 0.0112 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | 0.0000 |
| 0.5000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7448 | 1.0000 | 0.0109 | 0.7448 | 0.2121 | 0.0112 | 0.7448 | 0.0845 | 0.0112 | 0.7448 | 0.3466 | 0.0111 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | -0.0000 |
| 1.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7447 | 1.0000 | 0.0106 | 0.7448 | 0.2121 | 0.0111 | 0.7448 | 0.0845 | 0.0112 | 0.7448 | 0.3138 | 0.0110 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | -0.0000 |
| 2.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7446 | 1.0000 | 0.0099 | 0.7448 | 0.2121 | 0.0110 | 0.7448 | 0.0845 | 0.0111 | 0.7448 | 0.2776 | 0.0109 | 0.7448 | 0.0017 | 0.0112 | 0.0017 | -0.0000 |
| 5.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7443 | 1.0000 | 0.0080 | 0.7447 | 0.2121 | 0.0105 | 0.7448 | 0.0845 | 0.0110 | 0.7447 | 0.2241 | 0.0105 | 0.7448 | 0.0017 | 0.0112 | 0.0016 | -0.0001 |
| 10.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7438 | 1.0000 | 0.0047 | 0.7446 | 0.2121 | 0.0099 | 0.7447 | 0.0845 | 0.0107 | 0.7446 | 0.2052 | 0.0099 | 0.7448 | 0.0017 | 0.0112 | 0.0015 | -0.0001 |
| 50.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7398 | 1.0000 | -0.0213 | 0.7438 | 0.2121 | 0.0043 | 0.7444 | 0.0845 | 0.0085 | 0.7443 | 0.1000 | 0.0080 | 0.7448 | 0.0017 | 0.0112 | 0.0012 | -0.0001 |
| 100.0000 | 580 | oracle_probe | 0.7448 | 0.7431 | 0.0000 | 0.0000 | 0.7348 | 1.0000 | -0.0539 | 0.7427 | 0.2121 | -0.0026 | 0.7440 | 0.0845 | 0.0057 | 0.7442 | 0.0655 | 0.0070 | 0.7448 | 0.0017 | 0.0111 | 0.0011 | 0.0002 |

Interpretation:

Across probe-cost multipliers, VOI minus the best threshold policy has mean net-utility delta `-0.0000` over `8` settings (`1` positive, `6` negative, `1` tied).
