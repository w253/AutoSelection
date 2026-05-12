"""Cost-aware utility computation for recipe search.

U(r) = DevScore(r) - λ·C_search(r) - μ·C_train(r)

This module is deliberately simple and stateless — it takes numbers in,
returns a number out.  The search loop is responsible for bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UtilityConfig:
    """Weights for the cost-aware utility function."""

    lambda_search: float = 0.1  # penalty per GPU-hour of search cost
    mu_train: float = 0.05  # penalty per GPU-hour of train cost


def compute_utility(
    dev_score: float,
    search_cost: float,
    train_cost: float,
    config: UtilityConfig,
) -> float:
    """U(r) = DevScore - λ·C_search - μ·C_train"""
    return dev_score - config.lambda_search * search_cost - config.mu_train * train_cost
