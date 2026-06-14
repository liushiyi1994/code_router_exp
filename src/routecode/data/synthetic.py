from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticData:
    outcomes: pd.DataFrame
    embeddings: pd.DataFrame


DOMAIN_NAMES = [
    "general_knowledge",
    "routine_code",
    "symbolic_math",
    "data_transformation",
    "instruction_following",
    "systems_debugging",
    "multi_step_reasoning",
    "safety_policy",
]


DATASET_NAMES = [
    "synthetic_easy",
    "synthetic_code",
    "synthetic_math",
    "synthetic_data",
    "synthetic_instructions",
    "synthetic_debug",
    "synthetic_reasoning",
    "synthetic_safety",
]


def _model_skills(n_models: int) -> np.ndarray:
    base = np.array([0.46, 0.56, 0.60, 0.60, 0.70, 0.82], dtype=float)
    if n_models <= len(base):
        return base[:n_models]
    extra = np.linspace(0.50, 0.78, n_models - len(base))
    return np.concatenate([base, extra])


def generate_synthetic_outcomes(config: dict[str, Any]) -> SyntheticData:
    run_config = config.get("run", {})
    synthetic = config.get("synthetic", {})
    seed = int(run_config.get("random_seed", synthetic.get("seed", 0)))
    rng = np.random.default_rng(seed)

    n_queries = int(synthetic.get("n_queries", 2400))
    n_models = int(synthetic.get("n_models", 6))
    n_domains = int(synthetic.get("n_domains", 6))
    n_route_labels = int(synthetic.get("n_route_labels", 12))
    embedding_dim = int(synthetic.get("embedding_dim", 16))
    label_predictability = float(synthetic.get("label_predictability", 1.25))
    domain_strength = float(synthetic.get("domain_affinity_strength", 0.24))
    route_strength = float(synthetic.get("route_affinity_strength", 0.34))
    residual_strength = float(synthetic.get("residual_interaction_strength", 0.12))
    noise_std = float(synthetic.get("noise_std", 0.045))

    model_ids = synthetic.get("model_ids") or [f"model_{idx}" for idx in range(n_models)]
    model_ids = list(model_ids)[:n_models]
    if len(model_ids) != n_models:
        raise ValueError("synthetic.model_ids length must match n_models")

    configured_costs = synthetic.get("model_costs", {})
    default_costs = np.linspace(0.05, 0.7, n_models)
    model_costs = np.array(
        [float(configured_costs.get(model_id, default_costs[idx])) for idx, model_id in enumerate(model_ids)]
    )
    model_skills = _model_skills(n_models)

    domains = DOMAIN_NAMES[:n_domains]
    datasets = DATASET_NAMES[:n_domains]

    domain_preferred_model = np.arange(n_domains) % n_models
    domain_affinity = rng.normal(0.0, 0.04, size=(n_domains, n_models))
    for domain_idx, model_idx in enumerate(domain_preferred_model):
        domain_affinity[domain_idx, model_idx] += domain_strength
        domain_affinity[domain_idx, -1] += domain_strength * 0.35

    route_preferred_model = np.arange(n_route_labels) % n_models
    route_affinity = rng.normal(0.0, 0.05, size=(n_route_labels, n_models))
    for label_idx, model_idx in enumerate(route_preferred_model):
        route_affinity[label_idx, model_idx] += route_strength
        route_affinity[label_idx, (model_idx + 1) % n_models] += route_strength * 0.35

    domain_centroids = rng.normal(0.0, 1.0, size=(n_domains, embedding_dim))
    label_centroids = rng.normal(0.0, 1.0, size=(n_route_labels, embedding_dim))
    label_domain = np.arange(n_route_labels) % n_domains
    domain_probs = rng.dirichlet(np.full(n_domains, 1.2))

    rows: list[dict[str, Any]] = []
    embedding_rows: list[np.ndarray] = []
    query_ids: list[str] = []

    for query_idx in range(n_queries):
        query_id = f"q{query_idx:05d}"
        domain_idx = int(rng.choice(n_domains, p=domain_probs))
        eligible_labels = np.flatnonzero(label_domain == domain_idx)
        if len(eligible_labels) == 0:
            eligible_labels = np.arange(n_route_labels)
        if rng.random() < 0.82:
            route_label = int(rng.choice(eligible_labels))
        else:
            route_label = int(rng.integers(0, n_route_labels))

        domain = domains[domain_idx]
        dataset = datasets[domain_idx]
        difficulty = float(np.clip(rng.normal(0.28 + 0.035 * domain_idx, 0.08), 0.05, 0.62))
        query_length = int(np.clip(rng.normal(90 + 18 * domain_idx, 22), 20, 220))
        label_name = f"route_{route_label:02d}"
        query_text = (
            f"Synthetic {domain.replace('_', ' ')} request {query_idx}: "
            f"solve a {label_name} case with difficulty {difficulty:.2f}."
        )
        predicted_topic = domain if rng.random() > 0.12 else domains[int(rng.integers(0, n_domains))]

        embedding = (
            0.55 * domain_centroids[domain_idx]
            + label_predictability * label_centroids[route_label]
            + rng.normal(0.0, 0.65, size=embedding_dim)
        )
        embedding_rows.append(embedding)
        query_ids.append(query_id)

        token_factor = query_length / 100.0
        residual = rng.normal(0.0, residual_strength, size=n_models)
        for model_idx, model_id in enumerate(model_ids):
            linear_quality = (
                model_skills[model_idx]
                - difficulty
                + domain_affinity[domain_idx, model_idx]
                + route_affinity[route_label, model_idx]
                + residual[model_idx]
                + rng.normal(0.0, noise_std)
            )
            quality = float(np.clip(linear_quality, 0.0, 1.0))
            cost_total = float(model_costs[model_idx] * (0.75 + 0.5 * token_factor))
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "dataset": dataset,
                    "domain": domain,
                    "predicted_topic": predicted_topic,
                    "latent_route_label": route_label,
                    "model_id": model_id,
                    "quality": quality,
                    "cost_input": cost_total * 0.45,
                    "cost_output": cost_total * 0.55,
                    "cost_total": cost_total,
                    "latency": cost_total * 9.0 + 0.01 * query_length,
                    "tokens_input": query_length,
                    "tokens_output": int(np.clip(rng.normal(140, 30), 30, 280)),
                    "judge": "synthetic_closed_form",
                    "metadata_json": json.dumps(
                        {
                            "latent_route_label": route_label,
                            "difficulty": round(difficulty, 4),
                            "domain_idx": domain_idx,
                            "preferred_model": model_ids[route_preferred_model[route_label]],
                        },
                        sort_keys=True,
                    ),
                }
            )

    embeddings = pd.DataFrame(
        embedding_rows,
        index=pd.Index(query_ids, name="query_id"),
        columns=[f"emb_{idx}" for idx in range(embedding_dim)],
    )
    return SyntheticData(outcomes=pd.DataFrame(rows), embeddings=embeddings)
