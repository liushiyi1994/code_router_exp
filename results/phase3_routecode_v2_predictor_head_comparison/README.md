# RouteCode V2 Predictor Head Comparison

Date: 2026-06-20

This folder summarizes the first stronger query-to-state predictor pass after the
RouteCode v2 state learner split:

```text
utility matrix -> latent state
query text / cheap features -> p(state | query)
```

The goal was to test whether harder query-to-state predictors can make the
state assignment accurate enough for deployment. The target requested in this
pass was at least 90% held-out state accuracy.

## Commands Run

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_predictor_heads \
  --state-methods relative_kmeans \
  --k-values 16 \
  --predictors torch_mlp text_cnn \
  --active-label-budgets 64 128 256 \
  --torch-epochs 220 \
  --text-cnn-epochs 220
```

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_transformer_head_modernbert \
  --state-methods relative_kmeans \
  --k-values 16 \
  --predictors transformer \
  --active-label-budgets 64 \
  --transformer-model answerdotai/ModernBERT-base \
  --transformer-epochs 120
```

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_transformer_head_deberta \
  --state-methods relative_kmeans \
  --k-values 16 \
  --predictors transformer \
  --active-label-budgets 64 \
  --transformer-model microsoft/deberta-v3-base \
  --transformer-epochs 120
```

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_lowrate_predictor_sweep \
  --state-methods relative_kmeans \
  --k-values 2 4 8 16 \
  --predictors text_cnn \
  --active-label-budgets 64 \
  --text-cnn-epochs 180
```

```bash
PYTHONPATH=src python experiments/246_phase3_routecode_v2_state_pipeline.py \
  --output-dir results/phase3_routecode_v2_bge_predictor_heads \
  --state-methods relative_kmeans \
  --k-values 2 4 8 16 \
  --predictors knn mlp torch_mlp \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --active-label-budgets 64 \
  --torch-epochs 180
```

Verification:

```bash
pytest -q
```

Result:

```text
333 passed, 25 warnings
```

## Predictor Implementations Added

- `torch_mlp`: deeper PyTorch MLP over query embeddings.
- `text_cnn`: trainable token CNN over query text.
- `transformer`: frozen local Hugging Face encoder plus MLP head.
- Routing-aware auxiliary reward support for trainable heads.
- Robust small-batch MLP handling for active-label acquisition.

The transformer path defaults to local files only. It does not download models
unless `--allow-transformer-download` is explicitly passed.

## Main Result

The 90% held-out state-accuracy target was not reached.

Best held-out state accuracy in this pass:

| Run | State K | Predictor | Test state accuracy | Covered accuracy | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| BGE embedding sweep | 2 | KNN | 0.7235 | 0.8244 | Best test accuracy found |
| MiniLM low-rate sweep | 2 | text_cnn | 0.7176 | 0.7744 | Best text-CNN test accuracy |
| MiniLM K16 hard heads | 16 | text_cnn | 0.3118 | 0.3306 | Better than torch MLP |
| MiniLM K16 hard heads | 16 | torch_mlp | 0.2000 | 0.2126 | Overconfident and weak |
| ModernBERT frozen encoder | 16 | transformer | 0.2588 | 0.2636 | Local cached encoder |
| DeBERTa-v3 frozen encoder | 16 | transformer | 0.2059 | 0.2177 | Local cached encoder |
| BGE embedding sweep | 16 | KNN | 0.3471 | 0.3869 | Best K16 BGE row |
| BGE embedding sweep | 16 | MLP | 0.3176 | 0.3259 | Similar to earlier MLP |

Diagnostic true-state routing remains strong:

| State K | Policy | Test utility | Oracle utility ratio |
| ---: | --- | ---: | ---: |
| 16 | true state | 0.7295 | 0.9970 |
| 4 | true state | 0.6716 | 0.9180 |
| 8 | true state | 0.6537 | 0.8935 |
| 2 | true state | 0.5997 | 0.8196 |

This means the state-to-model table is strong when the state is known. The
current bottleneck is still query-to-state observability.

## Interpretation

More classifier capacity alone did not solve state prediction. The stronger
heads often became overconfident without improving held-out state accuracy.

The likely issue is that the current utility-derived states are not sufficiently
predictable from query text alone. Some states encode model-pool utility
differences that are weakly visible in text semantics.

## Next Steps

1. Add predictability-aware state learning directly into Model 1 instead of only
   training harder Model 2 heads.
2. Add probe-conditioned prediction: query text plus cheap local behavior,
   including local answer agreement, confidence, entropy, malformed/refusal
   flags, and local-vs-medium disagreement.
3. Treat the classifier confidence threshold as a trigger for probe acquisition,
   not as proof that the text-only classifier is good enough.
4. If the 90% state-accuracy target is mandatory, use lower-rate states first
   and evaluate K=2/K=4 separately from K=16.
