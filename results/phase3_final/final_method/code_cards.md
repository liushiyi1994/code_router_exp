# ProbeCode-StateCal Code Cards

## State 0

- `chosen_policy`: `always_large`
- `n_train`: `108`
- `train_need_large_rate`: `0.25`
- `train_true_tool_available_rate`: `0.0`
- `train_pred_tool_available_rate`: `0.0`
- `train_mean_local_utility`: `0.5370370370370371`
- `train_mean_large_utility`: `0.7318441822343286`
- `benchmark_mix_json`: `{"aime": 29, "bbh": 2, "gsm8k": 47, "humaneval": 1, "livemathbench": 47, "math500": 44, "mbpp": 5}`
- `top_feature_diffs_json`: `{"answer_chars_mean": 12.4955, "answer_chars_std": 10.3694, "sc_input_tokens": 58.2712, "sc_output_tokens": 16.9427, "text_digit_count": 20.519, "text_numeric_token_count": 9.505, "text_query_chars": 411.0112, "text_query_words": 65.908}`

## State 1

- `chosen_policy`: `tool_cap_c0.25`
- `n_train`: `75`
- `train_need_large_rate`: `0.0133333333333333`
- `train_true_tool_available_rate`: `0.0266666666666666`
- `train_pred_tool_available_rate`: `0.0266666666666666`
- `train_mean_local_utility`: `0.96`
- `train_mean_large_utility`: `0.9730673671039562`
- `benchmark_mix_json`: `{"bbh": 35, "gpqa": 1, "gsm8k": 17, "humaneval": 11, "livemathbench": 6, "math500": 17, "mbpp": 2, "mmlupro": 8}`
- `top_feature_diffs_json`: `{"answer_chars_std": 11.5218, "output_tokens_mean": 49.8529, "output_tokens_std": 44.5578, "sc_input_tokens": 43.7161, "sc_output_tokens": 67.9203, "text_digit_count": 10.7589, "text_query_chars": 400.9822, "text_query_words": 63.2723}`

## State 2

- `chosen_policy`: `and_q0.287_e0.179`
- `n_train`: `104`
- `train_need_large_rate`: `0.0769230769230769`
- `train_true_tool_available_rate`: `0.0`
- `train_pred_tool_available_rate`: `0.0`
- `train_mean_local_utility`: `0.8365384615384616`
- `train_mean_large_utility`: `0.8991773804535925`
- `benchmark_mix_json`: `{"bbh": 16, "humaneval": 88, "mbpp": 91}`
- `top_feature_diffs_json`: `{"output_tokens_mean": 18.9858, "sc_input_tokens": 41.6335, "sc_output_tokens": 34.1185, "text_digit_count": 22.1177, "text_math_symbol_count": 9.2816, "text_numeric_token_count": 14.9029, "text_query_chars": 205.1268, "text_query_words": 45.8779}`

## State 3

- `chosen_policy`: `always_large`
- `n_train`: `113`
- `train_need_large_rate`: `0.4070796460176991`
- `train_true_tool_available_rate`: `0.0`
- `train_pred_tool_available_rate`: `0.0`
- `train_mean_local_utility`: `0.5132743362831859`
- `train_mean_large_utility`: `0.7876865111627164`
- `benchmark_mix_json`: `{"gpqa": 99, "mmlupro": 92}`
- `top_feature_diffs_json`: `{"answer_chars_mean": 14.6588, "answer_chars_std": 14.059, "output_tokens_mean": 8.8328, "output_tokens_std": 21.4857, "q4_lp_logprob_margin_mean": 6.6021, "sc_input_tokens": 46.0248, "text_query_chars": 33.0732, "text_query_words": 12.9567}`

## State 4

- `chosen_policy`: `tool_always_large`
- `n_train`: `1`
- `train_need_large_rate`: `0.0`
- `train_true_tool_available_rate`: `0.0`
- `train_pred_tool_available_rate`: `0.0`
- `train_mean_local_utility`: `0.0`
- `train_mean_large_utility`: `0.0`
- `benchmark_mix_json`: `{"mbpp": 1}`
- `top_feature_diffs_json`: `{"output_tokens_std": 23.0421, "sc_input_tokens": 247.5512, "sc_output_tokens": 142.343, "text_digit_count": 2667.1895, "text_math_symbol_count": 68.4919, "text_numeric_token_count": 308.6721, "text_query_chars": 3497.8116, "text_query_words": 206.6349}`

## State 5

- `chosen_policy`: `always_local`
- `n_train`: `40`
- `train_need_large_rate`: `0.0`
- `train_true_tool_available_rate`: `1.0`
- `train_pred_tool_available_rate`: `0.975`
- `train_mean_local_utility`: `1.0`
- `train_mean_large_utility`: `0.4947895342350014`
- `benchmark_mix_json`: `{"aime": 19, "livemathbench": 34, "math500": 26}`
- `top_feature_diffs_json`: `{"answer_chars_mean": 34.5965, "answer_chars_std": 39.5089, "output_tokens_mean": 34.3081, "output_tokens_std": 24.2676, "sc_input_tokens": 33.9436, "sc_output_tokens": 70.619, "text_query_chars": 336.2643, "text_query_words": 55.0487}`

## State 6

- `chosen_policy`: `and_q0.287_e0.179`
- `n_train`: `22`
- `train_need_large_rate`: `0.1818181818181818`
- `train_true_tool_available_rate`: `0.0`
- `train_pred_tool_available_rate`: `0.0`
- `train_mean_local_utility`: `0.6818181818181818`
- `train_mean_large_utility`: `0.8459788180164712`
- `benchmark_mix_json`: `{"bbh": 40}`
- `top_feature_diffs_json`: `{"output_tokens_mean": 57.4203, "output_tokens_std": 37.5232, "sc_input_tokens": 142.3738, "sc_output_tokens": 130.343, "text_digit_count": 20.4645, "text_newline_count": 17.736, "text_query_chars": 5446.1116, "text_query_words": 982.2349}`

## State 7

- `chosen_policy`: `tool_always_large`
- `n_train`: `53`
- `train_need_large_rate`: `0.1698113207547169`
- `train_true_tool_available_rate`: `0.0943396226415094`
- `train_pred_tool_available_rate`: `0.0943396226415094`
- `train_mean_local_utility`: `0.6792452830188679`
- `train_mean_large_utility`: `0.7909831841749219`
- `benchmark_mix_json`: `{"aime": 12, "bbh": 7, "gsm8k": 36, "livemathbench": 13, "math500": 13, "mbpp": 1}`
- `top_feature_diffs_json`: `{"output_tokens_mean": 24.7748, "output_tokens_std": 16.9863, "sc_input_tokens": 63.8682, "sc_output_tokens": 49.8552, "text_digit_count": 17.2617, "text_numeric_token_count": 9.2182, "text_query_chars": 459.0054, "text_query_words": 73.8407}`
