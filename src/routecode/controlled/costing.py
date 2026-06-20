from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float = 0.0


def estimate_token_cost(
    input_tokens: int,
    output_tokens: int,
    price: TokenPrice | None,
) -> tuple[float, float, float]:
    if price is None:
        return 0.0, 0.0, 0.0
    input_cost = (float(input_tokens) / 1_000_000.0) * float(price.input_per_mtok)
    output_cost = (float(output_tokens) / 1_000_000.0) * float(price.output_per_mtok)
    return input_cost, output_cost, input_cost + output_cost


def enforce_frontier_budget(
    model_costs: dict[str, float],
    *,
    max_total_frontier_spend_usd: float,
    max_spend_per_frontier_model_usd: float,
) -> None:
    total = sum(float(value) for value in model_costs.values())
    if total > float(max_total_frontier_spend_usd) + 1e-12:
        raise ValueError(
            f"Estimated frontier spend ${total:.4f} exceeds total cap "
            f"${float(max_total_frontier_spend_usd):.4f}"
        )
    for model_id, cost in model_costs.items():
        if float(cost) > float(max_spend_per_frontier_model_usd) + 1e-12:
            raise ValueError(
                f"Estimated spend for {model_id} is ${float(cost):.4f}, above per-model cap "
                f"${float(max_spend_per_frontier_model_usd):.4f}"
            )
