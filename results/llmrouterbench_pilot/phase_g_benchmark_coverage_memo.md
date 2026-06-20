# Phase G Benchmark Coverage Memo

Command: `python experiments/20_benchmark_coverage.py --config configs/llmrouterbench.yaml`

This audit scans raw LLMRouterBench result JSON files before canonical schema validation. It does not run routers and makes no external API calls.

## Coverage Summary

- Result files after latest-file filtering: `567`.
- Datasets with local results: `24`.
- Models with local results: `40`.
- Datasets covered by configured taxonomy: `24`.

## Candidate Complete Rectangles

| model_count | dataset_count | complete_query_count | complete_row_count | models | datasets | dataset_splits |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 18 | 14041 | 84246 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini | aime;arcc;bbh;emorynlp;finqa;gpqa;humaneval;kandk;korbench;livecodebench;livemathbench;math500;mathbench;mbpp;medqa;meld;mmlupro;winogrande | aime:hybrid:60;arcc:test:1172;bbh:test:1080;emorynlp:test:697;finqa:test:1147;gpqa:test:198;humaneval:test:164;kandk:test:700;korbench:test:1250;livecodebench:test:1055;livemathbench:test:121;math500:test:500;mathbench:test:150;mbpp:test:974;medqa:test:1273;meld:test:1232;mmlupro:test_1000:1001;winogrande:valid:1267 |
| 10 | 18 | 14041 | 140410 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini;Llama-3.1-8B-Instruct;Llama-3.1-8B-UltraMedical;Llama-3.1-Nemotron-Nano-8B-v1;MiMo-7B-RL-0530 | aime;arcc;bbh;emorynlp;finqa;gpqa;humaneval;kandk;korbench;livecodebench;livemathbench;math500;mathbench;mbpp;medqa;meld;mmlupro;winogrande | aime:hybrid:60;arcc:test:1172;bbh:test:1080;emorynlp:test:697;finqa:test:1147;gpqa:test:198;humaneval:test:164;kandk:test:700;korbench:test:1250;livecodebench:test:1055;livemathbench:test:121;math500:test:500;mathbench:test:150;mbpp:test:974;medqa:test:1273;meld:test:1232;mmlupro:test_1000:1001;winogrande:valid:1267 |
| 20 | 18 | 14041 | 280820 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini;Llama-3.1-8B-Instruct;Llama-3.1-8B-UltraMedical;Llama-3.1-Nemotron-Nano-8B-v1;MiMo-7B-RL-0530;MiniCPM4.1-8B;NVIDIA-Nemotron-Nano-9B-v2;OpenThinker3-7B;Qwen2.5-Coder-7B-Instruct;Qwen3-8B;cogito-v1-preview-llama-8B;gemma-2-9b-it;glm-4-9b-chat;granite-3.3-8b-instruct;internlm3-8b-instruct | aime;arcc;bbh;emorynlp;finqa;gpqa;humaneval;kandk;korbench;livecodebench;livemathbench;math500;mathbench;mbpp;medqa;meld;mmlupro;winogrande | aime:hybrid:60;arcc:test:1172;bbh:test:1080;emorynlp:test:697;finqa:test:1147;gpqa:test:198;humaneval:test:164;kandk:test:700;korbench:test:1250;livecodebench:test:1055;livemathbench:test:121;math500:test:500;mathbench:test:150;mbpp:test:974;medqa:test:1273;meld:test:1232;mmlupro:test_1000:1001;winogrande:valid:1267 |
| 32 | 5 | 2435 | 77920 | DeepHermes-3-Llama-3-8B-Preview;DeepSeek-R1-0528-Qwen3-8B;DeepSeek-R1-Distill-Qwen-7B;Fin-R1;GLM-Z1-9B-0414;Intern-S1-mini;Llama-3.1-8B-Instruct;Llama-3.1-8B-UltraMedical;Llama-3.1-Nemotron-Nano-8B-v1;MiMo-7B-RL-0530;MiniCPM4.1-8B;NVIDIA-Nemotron-Nano-9B-v2;OpenThinker3-7B;Qwen2.5-Coder-7B-Instruct;Qwen3-8B;cogito-v1-preview-llama-8B;gemma-2-9b-it;glm-4-9b-chat;granite-3.3-8b-instruct;internlm3-8b-instruct;claude-sonnet-4;deepseek-r1-0528;deepseek-v3-0324;gemini-2.5-flash;gemini-2.5-pro;glm-4.6;gpt-5;intern-s1;kimi-k2-0905;qwen3-235b-a22b-2507;qwen3-235b-a22b-thinking-2507;gpt-5-chat | aime;gpqa;livecodebench;livemathbench;mmlupro | aime:hybrid:60;gpqa:test:198;livecodebench:test:1055;livemathbench:test:121;mmlupro:test_1000:1001 |

## Readout

- Largest candidate by complete query count uses `20` models over `18` datasets with `14041` complete queries.
- The 18-dataset/20-model candidate is now evaluated in `results/llmrouterbench_broad20`.
- The 32-model/5-dataset candidate is now evaluated as a model-pool scale and disjoint 16-source/16-target transfer stress test in `results/llmrouterbench_32model`.
- Use these candidates to choose larger real-data configs; do not infer routing performance from coverage alone.
