# Phase E External Blocker Resolution Memo

This memo aggregates blocked exact-command readiness rows across RouteCode runs. It performs no downloads, installs, or external API calls.

Inputs:

- `/home/liush/projects/code_router_exp/results/llmrouterbench_pilot/table_external_command_readiness.csv`
- `/home/liush/projects/code_router_exp/results/llmrouterbench_broad20/table_external_command_readiness.csv`

Blocked rows: `4`.
Checkpoint-gated blocked rows: `4`.
Module-only blocked rows: `0`.

## Blockers

| check_id | blocked_runs | blocked_run_count | blocking_reasons | missing_modules | missing_checkpoints | missing_assets | service_requirements | other_blockers | can_progress_without_download | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_route_train_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_best_route_local_model_checkpoint;missing_python_modules:llm_blender | llm_blender | missing_best_route_local_model_checkpoint |  |  |  | False | Provision local checkpoints: missing_best_route_local_model_checkpoint. Install Python module: llm_blender. |
| modelsat_train_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_modelsat_base_model_checkpoint;missing_modelsat_embedding_model_checkpoint;missing_python_modules:nltk,deepspeed | deepspeed,nltk | missing_modelsat_base_model_checkpoint,missing_modelsat_embedding_model_checkpoint |  |  |  | False | Provision local checkpoints: missing_modelsat_base_model_checkpoint,missing_modelsat_embedding_model_checkpoint. Install Python modules: deepspeed,nltk. |
| routellm_bert_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_bert_checkpoint |  | missing_bert_checkpoint |  |  |  | False | Provision local checkpoints: missing_bert_checkpoint. |
| routerdc_train_cli | llmrouterbench_broad20,llmrouterbench_pilot | 2 | missing_python_modules:deepspeed;missing_routerdc_local_model_checkpoint | deepspeed | missing_routerdc_local_model_checkpoint |  |  |  | False | Provision local checkpoints: missing_routerdc_local_model_checkpoint. Install Python module: deepspeed. |

## Interpretation

- Rows with `missing_checkpoints` require local checkpoint/model assets before they can become runnable; installing Python packages alone is insufficient.
- Rows with only `missing_modules` can be advanced locally by installing the listed modules, subject to compatibility with the current Python environment.
- Rows with service requirements should use cached/local embeddings to preserve the no-external-API constraint.
- Current unresolved rows: `best_route_train_cli`, `modelsat_train_cli`, `routellm_bert_cli`, `routerdc_train_cli`.
