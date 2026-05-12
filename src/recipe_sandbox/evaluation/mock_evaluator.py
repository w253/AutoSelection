"""Mock evaluator for smoke-testing the search loop.

Computes a deterministic pseudo-score from the DataStateVector of
a recipe's output, so the search loop can be tested end-to-end
without any GPU work.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from recipe_sandbox.evaluation.evaluator_base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class MockEvaluator(BaseEvaluator):
    """Deterministic evaluator based on dataset statistics.

    Scoring formula (design rationale):
      - Higher retain_ratio → good (data wasn't over-pruned)
      - Higher coverage → good (semantic diversity preserved)
      - Lower distribution_drift → good (data stayed on-distribution)
      - Lower redundancy → good (dedup worked)
      - Some stochastic noise from recipe name hash to simulate variance

    This is intentionally simplistic — the point is to make the search
    loop exercisable without any GPU.
    """

    def __init__(self, base_score: float = 50.0, noise_amplitude: float = 5.0):
        self.base_score = base_score
        self.noise_amplitude = noise_amplitude

    def evaluate(
        self,
        dataset_path: str,
        recipe_name: str,
        *,
        task_names: Optional[List[str]] = None,
        state_vector: Optional[Dict[str, float]] = None,
    ) -> EvalResult:
        """Compute a mock score from state vector statistics."""
        sv = state_vector or {}

        retain = sv.get("retain_ratio", 1.0)
        coverage = sv.get("score_mean", 0.5)
        drift = sv.get("distribution_drift", 0.0)
        varentropy = sv.get("mean_varentropy", 0.5)
        score_mean = sv.get("score_mean", 0.0)

        # Deterministic noise based on recipe name
        name_hash = int(hashlib.md5(recipe_name.encode()).hexdigest()[:8], 16)
        noise = ((name_hash % 1000) / 1000.0 - 0.5) * self.noise_amplitude

        dev_score = (
            self.base_score
            + retain * 20.0    # reward keeping data
            + coverage * 10.0  # reward score_mean
            - drift * 15.0     # penalise drift
            - abs(varentropy - 0.5) * 10.0  # penalise extreme varentropy
            + score_mean * 5.0  # reward high scoring data
            + noise
        )

        # Simulate training cost proportional to dataset size
        train_cost = max(0.05, retain * 0.5)  # GPU-hours

        task_names = task_names or ["mock_task"]
        task_scores = {t: dev_score + (i * 0.1) for i, t in enumerate(task_names)}

        logger.info(
            "[MockEval] %s → score=%.2f, train_cost=%.3fh (retain=%.2f, score_mean=%.2f, drift=%.2f)",
            recipe_name,
            dev_score,
            train_cost,
            retain,
            score_mean,
            drift,
        )

        return EvalResult(
            dev_score=dev_score,
            train_cost_gpu_hours=train_cost,
            eval_cost_gpu_hours=0.001,
            task_scores=task_scores,
            extra={"evaluator": "mock", "state_vector": sv},
        )
