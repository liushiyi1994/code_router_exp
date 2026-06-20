# Cost And Price Sensitivity

This no-call experiment freezes the query-to-state assignments and recomputes only the state-to-action table.

## Inputs

- Outcome matrix: `results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet`
- State method: `gb_depth2_thr0.9844_state_k8`

## Frozen State Policy Rows

- lambda `0.00`, price x`0.5`: utility `0.7791`, quality `0.7791`, frontier rate `0.9186`
- lambda `0.00`, price x`1.0`: utility `0.7791`, quality `0.7791`, frontier rate `0.9186`
- lambda `0.00`, price x`2.0`: utility `0.7791`, quality `0.7791`, frontier rate `0.9186`
- lambda `0.00`, price x`5.0`: utility `0.7791`, quality `0.7791`, frontier rate `0.9186`
- lambda `0.10`, price x`0.5`: utility `0.7724`, quality `0.7907`, frontier rate `0.6163`
- lambda `0.10`, price x`1.0`: utility `0.7569`, quality `0.7907`, frontier rate `0.5523`
- lambda `0.10`, price x`2.0`: utility `0.7231`, quality `0.7907`, frontier rate `0.5523`
- lambda `0.10`, price x`5.0`: utility `0.6670`, quality `0.7500`, frontier rate `0.5523`
- lambda `0.35`, price x`0.5`: utility `0.7315`, quality `0.7907`, frontier rate `0.5523`
- lambda `0.35`, price x`1.0`: utility `0.6723`, quality `0.7907`, frontier rate `0.5523`
- lambda `0.35`, price x`2.0`: utility `0.6818`, quality `0.6919`, frontier rate `0.3198`
- lambda `0.35`, price x`5.0`: utility `0.6667`, quality `0.6919`, frontier rate `0.3198`
- lambda `0.70`, price x`0.5`: utility `0.6723`, quality `0.7907`, frontier rate `0.5523`
- lambda `0.70`, price x`1.0`: utility `0.6818`, quality `0.6919`, frontier rate `0.3198`
- lambda `0.70`, price x`2.0`: utility `0.6717`, quality `0.6919`, frontier rate `0.3198`
- lambda `0.70`, price x`5.0`: utility `0.6415`, quality `0.6919`, frontier rate `0.3198`
- lambda `1.00`, price x`0.5`: utility `0.6670`, quality `0.7500`, frontier rate `0.5523`
- lambda `1.00`, price x`1.0`: utility `0.6775`, quality `0.6919`, frontier rate `0.3198`
- lambda `1.00`, price x`2.0`: utility `0.6631`, quality `0.6919`, frontier rate `0.3198`
- lambda `1.00`, price x`5.0`: utility `0.6194`, quality `0.6453`, frontier rate `0.0930`

## Action Table Change Summary

- lambda `0.00`, price x`0.5`: `5/8` states select frontier actions
- lambda `0.00`, price x`1.0`: `5/8` states select frontier actions
- lambda `0.00`, price x`2.0`: `5/8` states select frontier actions
- lambda `0.00`, price x`5.0`: `5/8` states select frontier actions
- lambda `0.10`, price x`0.5`: `4/8` states select frontier actions
- lambda `0.10`, price x`1.0`: `3/8` states select frontier actions
- lambda `0.10`, price x`2.0`: `3/8` states select frontier actions
- lambda `0.10`, price x`5.0`: `3/8` states select frontier actions
- lambda `0.35`, price x`0.5`: `3/8` states select frontier actions
- lambda `0.35`, price x`1.0`: `3/8` states select frontier actions
- lambda `0.35`, price x`2.0`: `2/8` states select frontier actions
- lambda `0.35`, price x`5.0`: `2/8` states select frontier actions
- lambda `0.70`, price x`0.5`: `3/8` states select frontier actions
- lambda `0.70`, price x`1.0`: `2/8` states select frontier actions
- lambda `0.70`, price x`2.0`: `2/8` states select frontier actions
- lambda `0.70`, price x`5.0`: `2/8` states select frontier actions
- lambda `1.00`, price x`0.5`: `3/8` states select frontier actions
- lambda `1.00`, price x`1.0`: `2/8` states select frontier actions
- lambda `1.00`, price x`2.0`: `2/8` states select frontier actions
- lambda `1.00`, price x`5.0`: `1/8` states select frontier actions

## Caveat

This tests cost-table adaptability under cached outcomes. It does not update provider pricing documents or make live calls.
