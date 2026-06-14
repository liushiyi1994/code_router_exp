from __future__ import annotations

import json
import os

from routecode.eval.transformer_backbones import inspect_transformer_backbone_cache


def test_inspect_transformer_backbone_cache_marks_missing_requested_and_cached_causal_lm(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen3-4B" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 2560,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"0" * 1024)

    table = inspect_transformer_backbone_cache(
        cache,
        requested_model_ids=["answerdotai/ModernBERT-base", "Qwen/Qwen3-4B"],
        max_runnable_gb=0.01,
    )

    by_model = table.set_index("model_id")
    assert by_model.loc["answerdotai/ModernBERT-base", "cache_status"] == "missing_local_cache"
    assert not bool(by_model.loc["answerdotai/ModernBERT-base", "runnable_as_encoder_baseline"])
    assert by_model.loc["Qwen/Qwen3-4B", "cache_status"] == "cached"
    assert by_model.loc["Qwen/Qwen3-4B", "architecture"] == "Qwen3ForCausalLM"
    assert by_model.loc["Qwen/Qwen3-4B", "reason"] == "causal_lm_not_lightweight_encoder"
    assert not bool(by_model.loc["Qwen/Qwen3-4B", "runnable_as_encoder_baseline"])


def test_inspect_transformer_backbone_cache_marks_cached_encoder_runnable(tmp_path):
    cache = tmp_path / "hub"
    snapshot = cache / "models--answerdotai--ModernBERT-base" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernbert",
                "architectures": ["ModernBertModel"],
                "hidden_size": 768,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"0" * 1024)

    table = inspect_transformer_backbone_cache(
        cache,
        requested_model_ids=["answerdotai/ModernBERT-base"],
        max_runnable_gb=0.01,
    )

    row = table.set_index("model_id").loc["answerdotai/ModernBERT-base"]
    assert row["cache_status"] == "cached"
    assert bool(row["runnable_as_encoder_baseline"])
    assert row["reason"] == "cached_encoder_candidate"


def test_inspect_transformer_backbone_cache_does_not_double_count_snapshot_symlinks(tmp_path):
    cache = tmp_path / "hub"
    model_dir = cache / "models--answerdotai--ModernBERT-base"
    blob_dir = model_dir / "blobs"
    snapshot = model_dir / "snapshots" / "abc123"
    blob_dir.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blob_dir / "weights"
    blob.write_bytes(b"0" * 1024)
    os.symlink(blob, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernbert",
                "architectures": ["ModernBertModel"],
                "hidden_size": 768,
            }
        ),
        encoding="utf-8",
    )

    table = inspect_transformer_backbone_cache(
        cache,
        requested_model_ids=["answerdotai/ModernBERT-base"],
        max_runnable_gb=0.01,
    )

    row = table.set_index("model_id").loc["answerdotai/ModernBERT-base"]
    assert row["size_gb"] < 0.0000015
